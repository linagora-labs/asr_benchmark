import logging

import ssak.utils.audio
import torch

from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}


class KyutaiSTTModel(Model):
    """Backend for the Kyutai STT streaming models, in their transformers port (the
    `-trfs` repos, e.g. kyutai/stt-1b-en_fr-trfs), transformers >= 4.53.

    Delayed-streams model on Mimi codec tokens: 24 kHz audio, the text stream lags
    the audio by a fixed delay (0.5 s for stt-1b-en_fr). The feature extractor pads
    the end with that delay + 1 s of silence, so that the last words are emitted.
    Transcribed offline here (whole segments). No language setting.
    """

    def load(self) -> None:
        from transformers import KyutaiSpeechToTextForConditionalGeneration, KyutaiSpeechToTextProcessor

        torch_dtype = _DTYPES.get(self.config["dtype"], torch.bfloat16)
        device = self.config["device"]
        self.processor = KyutaiSpeechToTextProcessor.from_pretrained(self.config["model"])
        self.model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            self.config["model"], torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else device,
        ).eval()

    def load_audio(self, audio: str, start=0.0, duration=None):
        # Mimi works at 24 kHz: load at that rate directly.
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
        # The feature extractor directly: the processor's __call__ does not take the audio
        # as a positional argument (input_values would be missing).
        inputs = self.processor.feature_extractor(
            audios, sampling_rate=self.config["sampling_rate"], return_tensors="pt", padding=True,
        ).to(self.model.device)
        with torch.no_grad():
            output_tokens = self.model.generate(**inputs)
        texts = self.processor.batch_decode(output_tokens, skip_special_tokens=True)
        return [{"text": text.strip()} for text in texts]

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        config["dtype"] = config.get("dtype", "bfloat16")
        config["sampling_rate"] = int(config.get("sampling_rate", 24000))
        # Not in the folder name: it only changes speed, up to bf16 padding noise.
        config["batch_size"] = int(config.get("batch_size", 1))
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.pop("batch_size", None)
        return metadata

    def get_folder_name(self):
        c = self.config
        name = f"kyutai-stt_{c['model'].replace('/', '-')}{self.device_tag()}"
        name += self.detail(f"_dtype-{c['dtype']}")
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
