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
}


class Gemma3nModel(Model):
    """Backend for Google's Gemma 3n multimodal LLMs (e.g. google/gemma-3n-e4b-it),
    used here for speech transcription.

    Gemma 3n is a general audio+vision+text model, not a French-tuned ASR system, so
    treat it as a baseline point of comparison rather than a strong contender.
    """

    def load(self) -> None:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        device = self.config["device"]
        torch_dtype = _DTYPES[self.config["dtype"]]
        if device == "cpu" and self.config["dtype"] != "float32":
            logger.warning("Forcing float32 on CPU for Gemma 3n.")
            torch_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(self.config["model"])
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.config["model"], torch_dtype=torch_dtype, device_map=device,
        ).eval()

    def load_audio(self, audio: str, start=0.0, duration=None):
        # Return a file path: Gemma's processor loads/resamples the audio itself when
        # the chat template is applied (our tmp wav is already the sliced 16 kHz segment).
        return load_audio(audio, return_format="file", start=start, duration=duration)

    def transcribe(self, audio: str) -> dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": self.config["prompt"]},
                ],
            }
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            generate_kwargs = dict(max_new_tokens=self.config["max_new_tokens"], do_sample=False)
            # Repetition suppression: greedy decoding on this general (non-ASR) model
            # collapses into degenerate loops ("oh, oh, oh, ...") on noisy/short clips,
            # producing huge insertion counts that dominate the WER.
            if self.config["no_repeat_ngram_size"]:
                generate_kwargs["no_repeat_ngram_size"] = self.config["no_repeat_ngram_size"]
            if self.config["repetition_penalty"] and self.config["repetition_penalty"] != 1.0:
                generate_kwargs["repetition_penalty"] = self.config["repetition_penalty"]
            with torch.inference_mode():
                gen = self.model.generate(**inputs, **generate_kwargs)
            text = self.processor.decode(gen[0][input_len:], skip_special_tokens=True)
        finally:
            Path(audio).unlink(missing_ok=True)
        return {"text": text.strip()}

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        config["dtype"] = config.get("dtype", "bfloat16")
        config["max_new_tokens"] = int(config.get("max_new_tokens", 512))
        # A directive prompt curbs the instruct model's tendency to refuse or add
        # commentary ("Je ne peux pas transcrire ce son...") instead of transcribing.
        config["prompt"] = config.get(
            "prompt",
            "Transcribe the speech in this audio verbatim. "
            "Output only the transcription, with no comments or explanations.",
        )
        # Repetition suppression, ON by default here: unlike an ASR model, greedy
        # decoding on this general LLM loops on noisy/short clips and the resulting
        # insertions dominate the WER. Override in the config to tune or disable
        # (no_repeat_ngram_size: 0, repetition_penalty: 1.0).
        config["no_repeat_ngram_size"] = int(config.get("no_repeat_ngram_size", 3))
        config["repetition_penalty"] = float(config.get("repetition_penalty", 1.2))
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["model"] = self.config["model"].replace("_", "-")
        return metadata

    def get_folder_name(self):
        c = self.config
        name = f"gemma3n_{c['model'].replace('/', '-')}_device-{c['device']}_dtype-{c['dtype']}"
        name = name.replace("/", "-")
        if c["no_repeat_ngram_size"] or (c["repetition_penalty"] and c["repetition_penalty"] != 1.0):
            name += f"_norep{c['no_repeat_ngram_size']}-rep{c['repetition_penalty']}"
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
