import logging
import torch
import ssak.utils.vad
from asr_benchmark.benchmark.interfaces import Model
from asr_benchmark.benchmark import contextual_biasing


logger = logging.getLogger(__name__)

class OpenAIModel(Model):
    # Contextual biasing: the terms go into `initial_prompt` (previous text of the
    # first 30 s window).
    SUPPORTS_CONTEXTUAL_BIASING = True

    def load(self) -> None:
        import whisper
        self.model = whisper.load_model(self.config['model'], self.config['device'], download_root=self.config.get('cache_dir', None))

    def transcribe(self, audio: str) -> str:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        output = dict()
        result = self.model.transcribe(audio, word_timestamps=False, no_speech_threshold=None, initial_prompt=self.format_biasing(self.biasing_terms))
        output['text'] = result['text']
        return output

    def can_output_word_timestamps(self):
        return True

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config['vad'] = config.get('vad', 'false')
        config['device'] = config.get('device', 'cuda')
        return super().add_defaults_to_config(config)

    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"openai_{tot_config['model']}{self.vad_tag()}{self.device_tag()}"
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name

class FasterWhisperModel(Model):
    # Contextual biasing through faster-whisper's `hotwords` argument (prompt tokens put
    # before the previous text, on every 30 s window, at most 223 tokens), or through
    # `initial_prompt` with contextual_biasing_mode: initial_prompt.
    SUPPORTS_CONTEXTUAL_BIASING = True

    def __init__(self, config) -> None:
        super().__init__(config)
        if self.config['accurate']:
            self.transcribe_kwargs['beam_size'] = 5
            self.transcribe_kwargs['best_of'] = 5
            self.transcribe_kwargs['temperature'] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        else:
            self.transcribe_kwargs['beam_size'] = 1
            self.transcribe_kwargs['best_of'] = 1
            self.transcribe_kwargs['temperature'] = 0.0

    def load(self) -> None:
        model_kwargs = {'device': self.config['device'], 'precision': self.config['precision']}
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        if self.config['model'] is None:
            raise ValueError("path must be set")
        if model_kwargs['device']=="cpu" and model_kwargs['precision'] =="float16":
            raise ValueError("Float16 is not supported on CPU")
        model_kwargs['compute_type'] = model_kwargs.pop('precision', '')
        if model_kwargs['device'] == "cpu":
            model_kwargs['cpu_threads'] = self.config.get('num_threads') or 0  # 0: ctranslate2 default (4)
        self.model = WhisperModel(self.config['model'], download_root=self.config.get('cache_dir', None), **model_kwargs)
        if self.config['batch_size']>1:
            self.model = BatchedInferencePipeline(model=self.model)

    def transcribe(self, audio: str) -> str:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        output = dict()
        language = self.config.get("language", "fr")
        condition_on_previous_text = self.config.get("previous_text", False)
        kwargs = dict(self.transcribe_kwargs)
        biasing = self.format_biasing(self.biasing_terms)
        if biasing:
            kwargs[self.config["contextual_biasing_mode"]] = biasing
        if self.config['batch_size']>1:
            segments, info = self.model.transcribe(audio, language=language, **kwargs, batch_size=self.config['batch_size'], condition_on_previous_text=condition_on_previous_text)
        else:
            segments, info = self.model.transcribe(audio, language=language, **kwargs, condition_on_previous_text=condition_on_previous_text)
        output['text'] = " ".join([seg.text for seg in segments])
        return output

    def can_output_word_timestamps(self):
        return True

    def add_defaults_to_config(self, config):
        config['vad'] = config.get('vad', 'false')
        config['precision'] = config.get('precision', 'float16')
        config['device'] = config.get('device', 'cuda')
        config['accurate'] = config.get('accurate', False)
        config['previous_text'] = config.get('previous_text', False)
        config['batch_size'] = config.get('batch_size', 1)
        if contextual_biasing.config_variant(config) is not None:
            config['contextual_biasing_mode'] = config.get('contextual_biasing_mode', 'hotwords')
            if config['contextual_biasing_mode'] not in ('hotwords', 'initial_prompt'):
                raise ValueError(f"contextual_biasing_mode must be hotwords or initial_prompt, not {config['contextual_biasing_mode']}")
        return super().add_defaults_to_config(config)

    def biasing_folder_params(self):
        return ["initprompt"] if self.config["contextual_biasing_mode"] == "initial_prompt" else []

    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"faster-whisper_{tot_config['model']}{self.vad_tag()}{self.device_tag()}"
        name += self.detail(f"_precision-{tot_config['precision']}")
        name += self.detail(f"_accurate-{tot_config['accurate']}_previous-{tot_config['previous_text']}")
        if tot_config['batch_size']>1:
            name += self.detail(f"_batchsize-{tot_config['batch_size']}")
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name
