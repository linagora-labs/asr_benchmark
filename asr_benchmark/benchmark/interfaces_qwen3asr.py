import logging
from pathlib import Path

import torch

from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}

# Qwen3-ASR's apply_transcription_request accepts either a full language name
# ("French") or a 2-letter code ("fr"); the full names are the documented form.
_LANGUAGE_NAMES = {
    "fr": "French", "en": "English", "zh": "Chinese", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "ru": "Russian", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
}


class Qwen3ASRModel(Model):
    """Backend for Qwen/Qwen3-ASR-1.7B-hf, a dedicated speech-recognition model
    (checkpoint model type 'qwen3_asr' -- NOT the Qwen3-Omni MoE, which is a
    different architecture and needs the separate qwen3-omni backend).

    Requires transformers >= 5.13.0 (older versions do not recognise 'qwen3_asr').
    Transcription uses the processor's apply_transcription_request() API, with
    optional forced language.
    """

    def load(self) -> None:
        from transformers import AutoProcessor
        # Qwen3ASRForConditionalGeneration is the concrete class; fall back to the
        # Auto multimodal LM if this transformers exposes only the Auto mapping.
        try:
            from transformers import Qwen3ASRForConditionalGeneration as _ModelCls
        except ImportError:
            from transformers import AutoModelForMultimodalLM as _ModelCls

        torch_dtype = _DTYPES.get(self.config["dtype"], torch.bfloat16)
        device = self.config["device"]
        device_map = "auto" if device == "cuda" else device
        self.processor = AutoProcessor.from_pretrained(self.config["model"])
        self.model = _ModelCls.from_pretrained(
            self.config["model"], torch_dtype=torch_dtype, device_map=device_map,
        ).eval()

    def load_audio(self, audio: str, start=0.0, duration=None):
        # Return a 16 kHz mono wav path; apply_transcription_request loads it itself.
        return load_audio(audio, return_format="file", start=start, duration=duration)

    def transcribe(self, audio: str) -> dict:
        request_kwargs = {}
        language = self.config.get("language")
        if language:
            request_kwargs["language"] = _LANGUAGE_NAMES.get(language, language)
        try:
            inputs = self.processor.apply_transcription_request(audio=audio, **request_kwargs)
            inputs = inputs.to(self.model.device, self.model.dtype)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=self.config["max_new_tokens"],
                )
            # decode() forwards to tokenizer.decode, which takes ONE sequence -- hence
            # generated[0], not the (1, L) batch. With return_format="transcription_only"
            # it returns a str for a single sequence, so indexing [0] here would slice
            # the first character off the transcript instead of picking a batch item.
            generated = output_ids[0, input_len:]
            text = self.processor.decode(generated, return_format="transcription_only")
        finally:
            Path(audio).unlink(missing_ok=True)
        return {"text": text.strip()}

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        config["dtype"] = config.get("dtype", "bfloat16")
        config["max_new_tokens"] = int(config.get("max_new_tokens", 256))
        # Qwen3-ASR supports forced language natively; default to French for this
        # (mostly French) benchmark. Set to null for the model's auto-detection.
        config["language"] = config.get("language", "fr")
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["model"] = self.config["model"].replace("_", "-")
        return metadata

    def get_folder_name(self):
        c = self.config
        name = f"qwen3-asr_{c['model'].replace('/', '-')}_device-{c['device']}_dtype-{c['dtype']}"
        name += f"_lang-{c['language']}" if c.get("language") else "_lang-auto"
        name = name.replace("/", "-")
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
