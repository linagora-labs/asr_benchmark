
import copy
import itertools
import logging
import re
import statistics
import time
from pathlib import Path
import torch
import json
import tempfile
import ssak.utils.vad
import nemo.collections.asr as nemo_asr
logging.getLogger('nemo_logging').setLevel(logging.ERROR)
from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model
from asr_benchmark.benchmark import contextual_biasing

DEFAULT_NUM_THREADS = torch.get_num_threads()
# Language tag (e.g. " <fr-FR>") that prompt-conditioned models append after the
# terminal punctuation, at least in auto-detect mode.
LANG_TAG_PATTERN = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")

class NemoModel(Model):
    # Contextual biasing through NeMo's GPU phrase boosting (GPU-PB, "TurboBias",
    # nemo_toolkit >= 2.5): a boosting tree built from the terms, fused at decoding
    # time (greedy_batch strategy for CTC and RNN-T/TDT, beam for Canary). Config:
    # contextual_biasing_alpha (boosting_tree_alpha, required: NeMo says to tune it),
    # contextual_biasing_context_score (default 1.0), contextual_biasing_depth_scaling
    # (default 2.0, 1.0 for Canary, as recommended in NeMo's word boosting docs).
    SUPPORTS_CONTEXTUAL_BIASING = True

    def __init__(self, config) -> None:
        model_type = nemo_asr.models.EncDecCTCModelBPE
        if "nemotron-3.5-asr" in config['model'] or "nemotron-labs-unified-asr" in config['model']:
            # Language-ID prompt-conditioned RNNT: cache-aware streaming (nvidia/nemotron-3.5-asr-streaming-0.6b)
            # or unified offline/streaming (nvidia/nemotron-labs-unified-asr-0.6b, full context by default).
            model_type = getattr(nemo_asr.models, "EncDecRNNTBPEModelWithPrompt", None)
            if model_type is None:
                raise ImportError(f"{config['model']} needs nemo_toolkit>=3.0.0 (EncDecRNNTBPEModelWithPrompt)")
        elif "hybrid" in config['model'] or "linto_stt" in config['model']:
             model_type = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        elif "rnnt" in config['model'] or "tdt" in config['model']:
            model_type = nemo_asr.models.EncDecRNNTBPEModel
        elif "canary" in config['model']:
            model_type = nemo_asr.models.EncDecMultiTaskModel
        self.model_type = model_type
        self.is_prompt_model = model_type is getattr(nemo_asr.models, "EncDecRNNTBPEModelWithPrompt", None)
        super().__init__(config)


    def load(self) -> None:
        self.decoder = None
        self.streaming_context = {}
        logging.getLogger("nemo_logger").setLevel(logging.ERROR)
        if self.config['model'].endswith(".nemo"):
            self.model = nemo_asr.models.ASRModel.restore_from(self.config['model'], map_location=self.config['device'])
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.config['model'], map_location=self.config['device'])
        if self.model_type != type(self.model):
            raise ValueError(f"Model type mismatch {self.model_type} != {type(self.model)} for model {self.config['model']}")
        if self.config.get("ngram_model", None):
            import pyctcdecode
            self.model.change_decoding_strategy(decoder_type="ctc")
            vocab = self.model.tokenizer.vocab
            decoder = pyctcdecode.build_ctcdecoder(
                labels=vocab,
                kenlm_model_path=self.config["ngram_model"],
                alpha=0.5,
                beta=1,
            )
            self.decoder = decoder
        elif self.model_type == nemo_asr.models.EncDecHybridRNNTCTCBPEModel:
            self.model.change_decoding_strategy(decoder_type=self.config['decoder'])
        elif self.model_type == nemo_asr.models.EncDecMultiTaskModel:
            decode_cfg = self.model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            self.model.change_decoding_strategy(decode_cfg)
        self.set_chunk_size()
        self._boosting_key = None
        self._base_decoding_cfg = None
        self._boosting_times = []  # seconds to set each boosting tree (see _set_boosting)
        self._many_trees = False  # per-utterance lists: a new boosting tree for each group of rows
        if contextual_biasing.config_variant(self.config) is not None:
            if self.decoder is not None:
                raise ValueError("contextual_biasing is not supported with ngram_model (pyctcdecode)")
            self._base_decoding_cfg = copy.deepcopy(self._decoding_cfg())
            # Boosting needs this strategy (beam for Canary, greedy_batch otherwise)
            self._boosting_strategy = "beam" if self.model_type == nemo_asr.models.EncDecMultiTaskModel else "greedy_batch"
            strategy = self._base_decoding_cfg.get("strategy")
            if strategy != self._boosting_strategy:
                # The run without list (baseline) decodes with the model's default
                # strategy, so a part of the difference may come from it.
                logging.warning(f"{self.config['model']}: decoding strategy {strategy} replaced by {self._boosting_strategy} for phrase boosting")

    def _decoding_cfg(self):
        if self.model_type == nemo_asr.models.EncDecHybridRNNTCTCBPEModel and self.config['decoder'] == 'ctc':
            return self.model.cfg.aux_ctc.decoding
        return self.model.cfg.decoding

    def _set_boosting(self, terms):
        """Decoding with a boosting tree of `terms` (or without boosting when None)."""
        if self._base_decoding_cfg is None:
            return
        key = tuple(terms) if terms else None
        if key == self._boosting_key:
            return
        from omegaconf import open_dict

        cfg = copy.deepcopy(self._base_decoding_cfg)
        with open_dict(cfg):
            cfg.strategy = self._boosting_strategy
            section = cfg.beam if self._boosting_strategy == "beam" else cfg.greedy
            if terms:
                section.boosting_tree = {
                    "key_phrases_list": list(terms),
                    "context_score": float(self.config["contextual_biasing_context_score"]),
                    "depth_scaling": float(self.config["contextual_biasing_depth_scaling"]),
                    "source_lang": self.config.get("language") or "en",
                }
                section.boosting_tree_alpha = float(self.config["contextual_biasing_alpha"])
            else:
                section.boosting_tree = None
            if self._many_trees and self._boosting_strategy == "greedy_batch" and self.model_type != nemo_asr.models.EncDecCTCModelBPE \
                    and not (self.model_type == nemo_asr.models.EncDecHybridRNNTCTCBPEModel and self.config['decoder'] == 'ctc'):
                # RNN-T / TDT greedy decoding: CUDA graphs are captured again for each new
                # tree, and after a few thousand captures (LibriSpeech per-utterance lists)
                # NeMo 3.0 fails with "Capture must end on the same stream it began on".
                # Same decoding without CUDA graphs (same output, slower).
                section.use_cuda_graph_decoder = False
                section.boosting_tree_alpha = 0.0
        # Time to build the boosting tree and set it in the decoding (what a server
        # pays each time the hotwords change), reported in the metadata
        start = time.perf_counter()
        if self.model_type == nemo_asr.models.EncDecHybridRNNTCTCBPEModel:
            self.model.change_decoding_strategy(cfg, decoder_type=self.config['decoder'], verbose=False)
        elif self.model_type == nemo_asr.models.EncDecMultiTaskModel:
            self.model.change_decoding_strategy(cfg)  # Canary: no verbose argument
        else:
            self.model.change_decoding_strategy(cfg, verbose=False)
        if terms:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._boosting_times.append(time.perf_counter() - start)
        self._boosting_key = key

    def set_chunk_size(self):
        """Latency of cache-aware streaming models (att_context_style chunked_limited). Even when
        transcribing whole files, the encoder masks attention as in streaming: each chunk of
        right+1 frames only sees the `left` previous frames, so the chunk size changes the WER.
        `chunk_ms` picks the [left, right] of the model's att_context_size list with that chunk;
        without it the model's default (first of the list) is used. The effective value goes to
        metadata.json (not to the config: the folder name must not change after loading)."""
        encoder = self.model.encoder
        contexts = getattr(encoder, "att_context_size_all", None)
        if getattr(encoder, "att_context_style", None) != "chunked_limited" or not contexts:
            if self.config.get("chunk_ms"):
                raise ValueError(f"chunk_ms is only for cache-aware streaming models, not {self.config['model']}")
            return
        frame_ms = round(self.model.cfg.preprocessor.window_stride * encoder.subsampling_factor * 1000)
        chunk_ms = lambda context: (context[1] + 1) * frame_ms
        if self.config.get("chunk_ms"):
            matching = [c for c in contexts if chunk_ms(c) == int(self.config["chunk_ms"])]
            if not matching:
                supported = sorted(chunk_ms(c) for c in contexts)
                raise ValueError(f"chunk_ms {self.config['chunk_ms']} not supported by {self.config['model']}, choose among {supported}")
            encoder.set_default_att_context_size(list(matching[0]))
        context = list(encoder.att_context_size)
        self.streaming_context = {"chunk_ms": chunk_ms(context), "att_context_size": context}
    
    def load_audio(self, audio: str, start=0.0, duration=None):
        return load_audio(audio, start=start, duration=duration)
    
    def transcribe(self, audio: str) -> str:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        output = dict()
        self._set_boosting(self.biasing_terms)
        if isinstance(self.model, nemo_asr.models.EncDecMultiTaskModel):
            result = self.model.transcribe(
                audio,
                duration=None,
                task="asr",
                source_lang=self.config["language"],
                target_lang=self.config["language"],
                pnc="yes",
                answer="na",
                verbose=False
            )
        elif self.is_prompt_model:
            # In-memory audio has no per-cut language, so the prompt comes from target_lang.
            result = self.model.transcribe(audio, verbose=False, target_lang=self.config["language"])
        else:
            result = self.model.transcribe(audio, verbose=False)
        output['text'] = self.clean_text(result[0].text)
        return output

    def clean_text(self, text):
        if self.is_prompt_model:
            text = LANG_TAG_PATTERN.sub("", text).strip()
        return text

    def transcribe_batch(self, data: str) -> str:
        if self._base_decoding_cfg is None:
            return self._transcribe_batch(data)
        # Contextual biasing: one boosting tree per list, so the rows are transcribed
        # by groups of consecutive rows sharing the same list (a global list: one group;
        # per-utterance lists: one row at a time).
        outputs = []
        groups = [(terms, list(group)) for terms, group in itertools.groupby(data, key=lambda row: tuple(self.terms_for_row(row) or ()))]
        self._many_trees = len(groups) > 1
        for terms, group in groups:
            self._set_boosting(list(terms) or None)
            outputs.extend(self._transcribe_batch(group))
        return outputs

    def _transcribe_batch(self, data):
        # NeMo transcribes from a manifest file; a temp file per call so parallel runs
        # never share it, removed even if transcription fails.
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as f:
            manifest = f.name
            for i in data:
                if self.is_prompt_model:
                    # With a manifest, the lhotse prompt dataset ignores target_lang: it takes the
                    # language from each row's "lang" and, unless "prompt_mode" is "langID", randomly
                    # swaps it for the auto prompt half of the time ("unified" default mode).
                    language = self.config["language"]
                    i = dict(i, lang=language, prompt_mode="auto" if language == "auto" else "langID")
                i = {k: v for k, v in i.items() if not k.startswith("contextual_biasing")}
                f.write(json.dumps(i, ensure_ascii=False)+"\n")
        try:
            result = self._transcribe_manifest(manifest)
        finally:
            Path(manifest).unlink()
        outputs = list()
        for i in result:
            if self.decoder:
                logits = i.alignments
                text = self.decoder.decode(logits.numpy())
                outputs.append({'text': text})
            else:
                output = {'text': self.clean_text(i.text)}
                outputs.append(output)
        return outputs

    def _transcribe_manifest(self, manifest):
        import nemo.collections.asr as nemo_asr
        batch_size = int(self.config.get('batch_size', 16))
        if isinstance(self.model, nemo_asr.models.EncDecMultiTaskModel):
            result = self.model.transcribe(
                manifest,
                duration=None,
                task="asr",
                source_lang=self.config["language"],
                target_lang=self.config["language"],
                pnc="yes",
                answer="na",
                batch_size=batch_size,  # batch size to run the inference with
                num_workers=4
            )
        elif self.is_prompt_model:
            result = self.model.transcribe(manifest, batch_size=batch_size, num_workers=4, target_lang=self.config["language"])
        else:
            result = self.model.transcribe(manifest, batch_size=batch_size, num_workers=4, return_hypotheses=True if self.decoder else False)
        return result

    def can_output_word_timestamps(self):
        return True
    
    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config['vad'] = config.get('vad', 'false')
        config['device'] = config.get('device', 'cuda')
        config['num_threads'] = config.get('num_threads', DEFAULT_NUM_THREADS) if config['device'] == 'cpu' else None
        if self.is_prompt_model:
            # Key of the model's prompt_dictionary: "fr" and "fr-FR" map to the same prompt, "auto" detects.
            config['language'] = config.get('language') or 'fr'
            config['decoder'] = 'rnnt'
        elif self.model_type!=nemo_asr.models.EncDecMultiTaskModel:
            config['decoder'] = config.get('decoder', 'ctc')
        if contextual_biasing.config_variant(config) is not None:
            if config.get('contextual_biasing_alpha') is None:
                raise ValueError("NeMo contextual biasing needs contextual_biasing_alpha (boosting_tree_alpha, to tune on the data)")
            config['contextual_biasing_context_score'] = config.get('contextual_biasing_context_score', 1.0)
            default_depth = 1.0 if self.model_type == nemo_asr.models.EncDecMultiTaskModel else 2.0
            config['contextual_biasing_depth_scaling'] = config.get('contextual_biasing_depth_scaling', default_depth)
        return super().add_defaults_to_config(config)

    def biasing_folder_params(self):
        return [f"alpha{self.config['contextual_biasing_alpha']}"]

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.config['model'].endswith(".nemo"):
            metadata['model'] = Path(self.config['model']).stem
        if "batch_size" in metadata:
            del metadata['batch_size']
        if "ngram_model" in metadata:
            metadata["decoder"] = Path(self.config['ngram_model']).name
            del metadata['ngram_model']
        metadata.update(getattr(self, "streaming_context", {}))
        times = getattr(self, "_boosting_times", None)
        if times:
            # Setting the boosting tree: once for a global list, once per utterance
            # for per-utterance lists (the first one may include one-time setup)
            metadata["contextual_biasing_tree_time_ms"] = {
                "count": len(times),
                "first": round(1000 * times[0], 1),
                "median": round(1000 * statistics.median(times), 1),
                "max": round(1000 * max(times), 1),
                "total": round(1000 * sum(times), 1),
            }
        return metadata
    
    def get_folder_name(self):
        tot_config = self.config.copy()
        if tot_config['model'].endswith(".nemo"):
            model = tot_config['model'].split("/")[-1].replace(".nemo", "").replace("_","-")
        else:
            model = tot_config['model'].replace(".nemo", "").replace("_","-").replace("/","-")
        name = f"nemo_{model}{self.device_tag()}"
        if self.model_type == nemo_asr.models.EncDecHybridRNNTCTCBPEModel:
            if tot_config.get('ngram_model', False):
                name += f"_decoder-{Path(tot_config['ngram_model']).name}"
            else:
                name += f"_decoder-{tot_config['decoder']}"
        elif self.model_type == nemo_asr.models.EncDecRNNTModel:
            name += self.detail("_decoder-rnnt")
        elif self.model_type == nemo_asr.models.EncDecCTCModelBPE:
            name += self.detail("_decoder-ctc")
        if self.is_prompt_model:
            name += self.detail(f"_lang-{tot_config['language']}")
        if tot_config.get('chunk_ms'):
            name += self.detail(f"_chunk{tot_config['chunk_ms']}ms")
        if tot_config['compute_rtf']:
            name += self.vad_tag()
        name += self.detail(f"_threads{tot_config['num_threads']}") if tot_config['device'] == 'cpu' else ""
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name