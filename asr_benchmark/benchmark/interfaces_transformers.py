import logging
import os
import torch
import ssak.utils.vad
from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model


logger = logging.getLogger(__name__)

        
class TransformersModel(Model):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        # No language: the one of the model's generation config (e.g. monolingual fine-tunes
        # with their own tokenizer, whose language/task prompt is baked in).
        if self.config["language"]:
            self.transcribe_kwargs['language'] = self.config["language"]
            self.transcribe_kwargs['task'] = "transcribe"
        self.transcribe_kwargs['do_sample'] = self.config['do_sample']
        if self.config['do_sample']:
            self.transcribe_kwargs['temperature'] = self.config['temperature']
            self.transcribe_kwargs['top_k'] = self.config['top_k']
        else:
            self.transcribe_kwargs['num_beams'] = self.config['num_beams']
        # Other generate() arguments, e.g. the decoding recommended by a model card
        # (no_repeat_ngram_size, repetition_penalty...).
        self.transcribe_kwargs.update(self.config['generate_kwargs'] or {})
        if self.config['device']=="cpu":
            torch.set_num_threads(self.config['num_threads'])

    
    def load(self) -> None:
        from transformers import pipeline
        from transformers.utils import is_flash_attn_2_available
        model_kwargs = {}
        if self.config['attn'] == "flash2":
            if not is_flash_attn_2_available():
                raise ValueError("Flash attention 2 is not available.")
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif self.config['attn'] == "eager":
            model_kwargs["attn_implementation"] = "eager"
        elif self.config['attn'] == "sdpa":
            model_kwargs["attn_implementation"] = "sdpa"
        else:
            raise ValueError(f"Unknown attention implementation: {self.config['attn']}, can be flash2, eager or sdpa.")
        tokenizer = None
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.config['model'], # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
            torch_dtype=self.config['precision'],
            device_map=self.config['device'],
            model_kwargs=model_kwargs,
            token=True,
            tokenizer=tokenizer,
        )
        self.model = pipe

    def transcribe(self, audio: str) -> str:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        result = self.model(audio, chunk_length_s=self.config['chunk_length_s'], batch_size=int(self.config['batch_size']), \
                            stride_length_s=self.config['stride_length_s'], return_timestamps=False, generate_kwargs=self.transcribe_kwargs)
        return {'text': result['text']}

    def can_output_word_timestamps(self):
        if self.config['attn'] == "eager":
            return True
        return False
    
    def cleanup(self):
        torch.cuda.empty_cache()
    
    def add_defaults_to_config(self, config):
        model_name = config['model']
        if model_name in ['large-v3', 'tiny', 'base', 'medium', 'large-v2', 'large-v1', 'small']:
            model_name = f"openai/whisper-{model_name}"
        config['model'] = model_name
        config['language'] = config.get('language')
        config['generate_kwargs'] = config.get('generate_kwargs')
        config['vad'] = config.get('vad', 'false')
        config['device'] = config.get('device', 'cuda')
        config['attn'] = config.get('attn', 'sdpa')
        config['precision'] = config.get('precision', 'float16')
        config['batch_size'] = config.get('batch_size', 24)
        config['chunk_length_s'] = config.get('chunk_length_s', 30)
        if config['chunk_length_s'] is not None:
            config['stride_length_s'] = float(config.get('stride_length_s', float(config['chunk_length_s']) / 6))
            config['chunk_length_s'] = float(config['chunk_length_s'])
        else:
            config['stride_length_s'] = config.get('stride_length_s', None)
        config['do_sample'] = config.get('do_sample', False)
        if config['do_sample']:
            config['temperature'] = config.get('temperature', 0.0)
            config['top_k'] = config.get('top_k', 1)
        else:
            config['num_beams'] = config.get('num_beams', 1)
        if config['device']=="cpu":
            config['num_threads'] = int(config.get('num_threads', 4))
        return super().add_defaults_to_config(config)
    
    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"transformers_{tot_config['model']}{self.vad_tag()}{self.device_tag()}"
        name += self.detail(f"_attn-{tot_config['attn']}_precision-{tot_config['precision']}")
        name += self.detail(f"_batch-{tot_config['batch_size']}_chunk-{tot_config['chunk_length_s']}_stride-{tot_config['stride_length_s']}")
        if tot_config['do_sample']:
            name += self.detail(f"_temperature-{tot_config['temperature']}_topk-{tot_config['top_k']}")
        else:
            name += self.detail(f"_beams-{tot_config['num_beams']}")
        for k, v in sorted((tot_config['generate_kwargs'] or {}).items()):
            name += self.detail(f"_{k}-{v}")
        if tot_config['device'] == "cpu":
            name += self.detail(f"_numthreads-{tot_config['num_threads']}")
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name
    
class TransformersWhisperModel(TransformersModel):
    """Whisper models through model.generate(), without the ASR pipeline, which decodes an
    empty text with the per-language tokenizers of some fine-tunes (BuzzASR). Segments of
    at most 30 s (one Whisper window), transcribed by batches of `batch_size`."""
    MAX_SECONDS = 30

    def load(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        attn = {"flash2": "flash_attention_2", "eager": "eager", "sdpa": "sdpa"}[self.config['attn']]
        self.dtype = getattr(torch, self.config['precision'])
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config['model'], dtype=self.dtype, attn_implementation=attn,
        ).to(self.config['device']).eval()
        self.processor = WhisperProcessor.from_pretrained(self.config['model'])

    def transcribe_many(self, audios: list) -> list[dict]:
        for audio in audios:
            if len(audio) > self.MAX_SECONDS * 16000:
                raise ValueError(f"{len(audio) / 16000:.1f} s of audio: {self.config['backend']} only transcribes segments of at most {self.MAX_SECONDS} s")
        features = self.processor(audios, sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = self.model.generate(features.to(self.config['device'], self.dtype), **self.transcribe_kwargs)
        return [{'text': text} for text in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def transcribe(self, audio) -> dict:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        return self.transcribe_many([audio])[0]

    def transcribe_batch(self, data: list) -> list[dict]:
        return self.transcribe_by_batches(data, int(self.config['batch_size']))

    def can_output_word_timestamps(self):
        return False

    def get_folder_name(self):
        return "transformers-whisper_" + super().get_folder_name().split("_", 1)[1]


class IntelTransformersModel(TransformersModel):
    def load(self) -> None:
        from intel_extension_for_transformers.transformers.pipeline import pipeline as intel_pipeline      
        model_kwargs = {}
        if self.config['attn'] == "flash2":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        elif self.config['attn'] == "eager":
            model_kwargs["attn_implementation"] = "eager"
        elif self.config['attn'] == "sdpa":
            model_kwargs["attn_implementation"] = "sdpa"
        else:
            raise ValueError(f"Unknown attention implementation: {self.config['attn']}")
        model_kwargs['num_threads'] = 4
        pipe = intel_pipeline(
            "automatic-speech-recognition",
            model=self.config['model'], # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
            torch_dtype=self.config['precision'],
            device=torch.device(self.config['device']),
            model_kwargs=model_kwargs,
        )
        self.model = pipe

    
    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"intel-transformers_{tot_config['model']}{self.vad_tag()}{self.device_tag()}"
        name += self.detail(f"_attn-{tot_config['attn']}")
        name += self.detail(f"_batch-{tot_config['batch_size']}_chunk-{tot_config['chunk_length_s']}_stride-{tot_config['stride_length_s']}")
        if tot_config['do_sample']:
            name += self.detail(f"_temperature-{tot_config['temperature']}_topk-{tot_config['top_k']}")
        else:
            name += self.detail(f"_beams-{tot_config['num_beams']}")
        if tot_config['device'] == "cpu":
            name += self.detail(f"_numthreads-{tot_config['num_threads']}")
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name

class TransformersFacebookModel(TransformersModel):
    
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def load(self) -> None:
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        device = torch.device(self.config['device'])
        processor = AutoProcessor.from_pretrained(self.config['model'])
        processor.tokenizer.set_target_lang("fra")
        self.processor = processor
        model = Wav2Vec2ForCTC.from_pretrained(self.config['model']).to(device)
        model.load_adapter("fra")
        self.model = model

    def load_audio(self, audio, start=0.0, duration=None) -> None:
        return self.processor(load_audio(audio, start=start, duration=duration), sampling_rate=16_000, return_tensors="pt")

    def transcribe(self, audio: str) -> str:
        device = torch.device(self.config['device'])
        audio.to(device)
        with torch.no_grad():
            outputs = self.model(**audio).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.processor.decode(ids)
        return transcription

    def can_output_word_timestamps(self):
        return True
    
    def cleanup(self):
        torch.cuda.empty_cache()
    
    def add_defaults_to_config(self, config):
        return super().add_defaults_to_config(config)
    
    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"transformers_{tot_config['model']}{self.vad_tag()}{self.device_tag()}"
        if tot_config['device'] == "cpu":
            name += self.detail(f"_numthreads-{tot_config['num_threads']}")
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name
    
class TransformersVoxtralRealtimeModel(Model):

    def __init__(self, config) -> None:
        super().__init__(config)

    def add_defaults_to_config(self, config):
        config['device'] = config.get('device', 'cuda')
        config['max_tokens'] = config.get('max_tokens', 512)
        config['temperature'] = config.get('temperature', 0.0)
        config['dtype'] = config.get('dtype', 'float16')
        return super().add_defaults_to_config(config)

    def load(self) -> None:
        from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor

        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        torch_dtype = dtype_map.get(self.config['dtype'], torch.float16)

        self.processor = AutoProcessor.from_pretrained(self.config['model'])
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            self.config['model'],
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    def load_audio(self, audio, start=0.0, duration=None):
        audio_array = load_audio(audio, start=start, duration=duration)
        target_sr = self.processor.feature_extractor.sampling_rate
        if target_sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=16000, target_sr=target_sr)
        return audio_array

    def transcribe(self, audio) -> dict:
        inputs = self.processor(audio, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt")
        inputs = inputs.to(device=self.model.device, dtype=self.model.dtype)
        generate_kwargs = dict(max_new_tokens=self.config['max_tokens'])
        if self.config['temperature'] > 0:
            generate_kwargs['do_sample'] = True
            generate_kwargs['temperature'] = self.config['temperature']
        outputs = self.model.generate(**inputs, **generate_kwargs)
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return {"text": text}

    def cleanup(self):
        torch.cuda.empty_cache()

    def get_folder_name(self):
        c = self.config
        name = f"transformers-voxtral-realtime_{c['model']}{self.device_tag()}"
        name += self.detail(f"_dtype-{c['dtype']}_maxtokens-{c['max_tokens']}")
        name = name.replace("/", "-")
        name += "_rtf" if c.get('compute_rtf') else ""
        return name


class TransformersBofenghuangModel(TransformersModel):
    
    def __init__(self, config) -> None:
        super().__init__(config)
    
    def load(self) -> None:
        from transformers import AutoModelForCTC, Wav2Vec2ProcessorWithLM
        self.device = torch.device(self.config['device'])
        model = AutoModelForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(self.device)
        processor_with_lm = Wav2Vec2ProcessorWithLM.from_pretrained("bhuang/asr-wav2vec2-french")
        self.processor = processor_with_lm
        self.model = model

    def load_audio(self, audio: str, start=0.0, duration=None) -> None:
        return self.processor(load_audio(audio, start=start, duration=duration), sampling_rate=16_000, return_tensors="pt")

    def transcribe(self, audio: str) -> str:
        audio.to(self.device)
        with torch.inference_mode():
            logits = self.model(audio.input_values.to(self.device)).logits

        predicted_sentence = self.processor.batch_decode(logits.cpu().numpy()).text[0]
        return predicted_sentence

    def can_output_word_timestamps(self):
        return True
    
    def cleanup(self):
        torch.cuda.empty_cache()
    
    def add_defaults_to_config(self, config):
        return super().add_defaults_to_config(config)
    
    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"transformers_{tot_config['model']}{self.vad_tag()}{self.device_tag()}"
        if tot_config['device'] == "cpu":
            name += self.detail(f"_numthreads-{tot_config['num_threads']}")
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name