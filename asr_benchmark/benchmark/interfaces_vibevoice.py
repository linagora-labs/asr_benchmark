import json
import logging
import re

import torch
import ssak.utils.audio

from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}

# Fallback when the JSON-like output cannot be parsed (e.g. cut by max_new_tokens):
# pull the "Content" values out of whatever segments were emitted.
_CONTENT_PATTERN = re.compile(r'"Content"\s*:\s*"((?:[^"\\]|\\.)*)')
# Non-speech annotations the model writes into "Content" (e.g. "[Silence]").
_ANNOTATION_PATTERN = re.compile(r"\s*\[[^\]]*\]")


def _unescape(content: str) -> str:
    try:
        return json.loads(f'"{content}"')
    except json.JSONDecodeError:  # truncated in the middle of an escape sequence
        return content


class VibeVoiceASRModel(Model):
    """Backend for microsoft/VibeVoice-ASR-HF, the transformers port of
    microsoft/VibeVoice-ASR (the original checkpoint needs the VibeVoice repo code).

    Requires transformers >= 5.3.0. The model has no language setting (it detects
    the language, code-switching included) and outputs speaker-timestamped
    segments as a JSON-like string; only the concatenated "Content" is kept.
    An optional `prompt` is passed as extra context (hotwords, topic...).
    """

    def load(self) -> None:
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        torch_dtype = _DTYPES.get(self.config["dtype"], torch.bfloat16)
        device = self.config["device"]
        device_map = "auto" if device == "cuda" else device
        self.processor = AutoProcessor.from_pretrained(self.config["model"])
        self.model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.config["model"], torch_dtype=torch_dtype, device_map=device_map,
        ).eval()

    def load_audio(self, audio: str, start=0.0, duration=None):
        # The acoustic tokenizer works at 24 kHz: load at that rate directly rather
        # than going through the 16 kHz benchmark loader and upsampling.
        end = start + duration if duration else None
        return ssak.utils.audio.load_audio(
            audio, start=start, end=end, sample_rate=self.config["sampling_rate"], mono=True, return_format="array",
        )

    def transcribe(self, audio) -> dict:
        return self.transcribe_many([audio])[0]

    def transcribe_batch(self, data: list) -> list[dict]:
        if self.config["batch_size"] <= 1:
            return super().transcribe_batch(data)
        return self.transcribe_by_batches(data, self.config["batch_size"])

    def transcribe_many(self, audios: list) -> list[dict]:
        # The processor left-pads the prompts, so generated tokens start at the same index.
        inputs = self.processor.apply_transcription_request(audio=audios, prompt=self.config.get("prompt"))
        inputs = inputs.to(self.model.device, self.model.dtype)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            # Warns "Both max_new_tokens and max_length" on each call (max_length comes from the
            # checkpoint's generation_config): harmless, max_new_tokens takes precedence.
            output_ids = self.model.generate(**inputs, max_new_tokens=self.config["max_new_tokens"])
        return [{"text": self._parse(generated)} for generated in output_ids[:, input_len:]]

    def _parse(self, generated) -> str:
        raw = self.processor.decode(generated, skip_special_tokens=True)
        try:
            # Returns the raw output when it is not a JSON array.
            text = self.processor.extract_transcription(raw)
        except ValueError:  # malformed JSON, e.g. cut by max_new_tokens
            text = raw.strip().removeprefix("assistant").strip()
        # Parsing failed (extract_transcription returned the raw output). Do not test for a
        # leading "[": a parsed transcription can start with an annotation like "[Silence]".
        if '"Content"' in text:
            contents = _CONTENT_PATTERN.findall(text)
            logger.warning(f"Could not parse VibeVoice output, recovered {len(contents)} segment(s): {text[:200]!r}")
            text = " ".join(_unescape(c) for c in contents)
        text = _ANNOTATION_PATTERN.sub("", text)
        return text.strip()

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        config["dtype"] = config.get("dtype", "bfloat16")
        # Each speaker segment costs JSON overhead (Start/End/Speaker keys) on top of the text.
        config["max_new_tokens"] = int(config.get("max_new_tokens", 1024))
        config["sampling_rate"] = int(config.get("sampling_rate", 24000))
        config["prompt"] = config.get("prompt")
        # Not in the folder name (like NeMo): it only changes speed, up to bf16 padding noise.
        config["batch_size"] = int(config.get("batch_size", 1))
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.pop("batch_size", None)
        return metadata

    def get_folder_name(self):
        c = self.config
        name = f"vibevoice-asr_{c['model'].replace('/', '-')}{self.device_tag()}"
        name += self.detail(f"_dtype-{c['dtype']}")
        name += self.detail("_prompt") if c.get("prompt") else ""
        name = name.replace("/", "-")
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
