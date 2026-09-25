
import logging
import re
from pathlib import Path
import torch
import json
import tempfile
import ssak.utils.vad
import nemo.collections.asr as nemo_asr
logging.getLogger('nemo_logging').setLevel(logging.ERROR)
from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model

DEFAULT_NUM_THREADS = torch.get_num_threads()
# Language tag (e.g. " <fr-FR>") that prompt-conditioned models append after the
# terminal punctuation, at least in auto-detect mode.
LANG_TAG_PATTERN = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")

class NemoModel(Model):

    def __init__(self, config) -> None:
        model_type = nemo_asr.models.EncDecCTCModelBPE
        if "nemotron-3.5-asr" in config['model']:
            # Language-ID prompt-conditioned cache-aware RNNT (nvidia/nemotron-3.5-asr-streaming-0.6b).
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
    
    def load_audio(self, audio: str, start=0.0, duration=None):
        return load_audio(audio, start=start, duration=duration)
    
    def transcribe(self, audio: str) -> str:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        output = dict()
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
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.config['model'].endswith(".nemo"):
            metadata['model'] = self.config['model'].split("/")[-1].replace(".nemo", "").replace("_","-")
        else:
            metadata['model'] = self.config['model'].replace("_","-")
        if "batch_size" in metadata:
            del metadata['batch_size']
        if "ngram_model" in metadata:
            metadata["decoder"] = Path(self.config['ngram_model']).name
            del metadata['ngram_model']
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
        if tot_config['compute_rtf']:
            name += self.vad_tag()
        name += self.detail(f"_threads{tot_config['num_threads']}") if tot_config['device'] == 'cpu' else ""
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name