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
        self.transcribe_kwargs['language'] = self.language
        self.transcribe_kwargs['task'] = "transcribe"
        self.transcribe_kwargs['do_sample'] = self.config['do_sample']
        if self.config['do_sample']:
            self.transcribe_kwargs['temperature'] = self.config['temperature']
            self.transcribe_kwargs['top_k'] = self.config['top_k']
        else:
            self.transcribe_kwargs['num_beams'] = self.config['num_beams']
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
        text = result['text']
        return text

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
        name = f"transformers_{tot_config['model']}_vad-{tot_config['vad']}_device-{tot_config['device']}_attn-{tot_config['attn']}_precision-{tot_config['precision']}"
        name += f"_batch-{tot_config['batch_size']}_chunk-{tot_config['chunk_length_s']}_stride-{tot_config['stride_length_s']}"
        if tot_config['do_sample']:
            name += f"_temperature-{tot_config['temperature']}_topk-{tot_config['top_k']}"
        else:
            name += f"_beams-{tot_config['num_beams']}"
        if tot_config['device'] == "cpu":
            name += f"_numthreads-{tot_config['num_threads']}"
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name
    
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
        name = f"intel-transformers_{tot_config['model']}_vad-{tot_config['vad']}_device-{tot_config['device']}_attn-{tot_config['attn']}"
        name += f"_batch-{tot_config['batch_size']}_chunk-{tot_config['chunk_length_s']}_stride-{tot_config['stride_length_s']}"
        if tot_config['do_sample']:
            name += f"_temperature-{tot_config['temperature']}_topk-{tot_config['top_k']}"
        else:
            name += f"_beams-{tot_config['num_beams']}"
        if tot_config['device'] == "cpu":
            name += f"_numthreads-{tot_config['num_threads']}"
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
        name = f"transformers_{tot_config['model']}_vad-{tot_config['vad']}_device-{tot_config['device']}"
        if tot_config['device'] == "cpu":
            name += f"_numthreads-{tot_config['num_threads']}"
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
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
        name = f"transformers_{tot_config['model']}_vad-{tot_config['vad']}_device-{tot_config['device']}"
        if tot_config['device'] == "cpu":
            name += f"_numthreads-{tot_config['num_threads']}"
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name