import logging

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


class Qwen3OmniModel(Model):
    """Backend for Qwen3-Omni (e.g. Qwen/Qwen3-Omni-30B-A3B-Instruct), used for
    transcription via the *thinker* head (text output only -- no speech synthesis).

    This is a large general-purpose multimodal MoE, not a French-tuned ASR model, and
    the 30B variant needs a lot of VRAM. For pure ASR, Qwen/Qwen3-ASR-1.7B-hf is a much
    lighter, more appropriate alternative.
    """

    def load(self) -> None:
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        torch_dtype = _DTYPES.get(self.config["dtype"], torch.bfloat16)
        device = self.config["device"]
        # A 30B MoE rarely fits one GPU; device_map="auto" shards it. On CPU/explicit
        # devices we honour the request.
        device_map = "auto" if device == "cuda" else device
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.config["model"])
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            self.config["model"], torch_dtype=torch_dtype, device_map=device_map,
        ).eval()
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def load_audio(self, audio: str, start=0.0, duration=None):
        arr = load_audio(audio, return_format="librosa", start=start, duration=duration)
        target = getattr(self, "sampling_rate", 16000)
        if target != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=16000, target_sr=target)
        return arr

    def transcribe(self, audio) -> dict:
        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": self.config["prompt"]},
                ],
            }
        ]
        # add_generation_prompt=True is required: without the assistant prompt the
        # model never emits EOS and rambles to max_new_tokens on every file. In
        # transformers 5.x, processor-call kwargs (padding, ...) must be nested in
        # processor_kwargs, otherwise they are dropped with a warning.
        inputs = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", processor_kwargs={"padding": True},
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            text_ids = self.model.generate(
                **inputs, max_new_tokens=self.config["max_new_tokens"], do_sample=False,
            )
        text = self.processor.batch_decode(
            text_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return {"text": text.strip()}

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        config["dtype"] = config.get("dtype", "bfloat16")
        config["max_new_tokens"] = int(config.get("max_new_tokens", 512))
        config["prompt"] = config.get("prompt", "Transcribe this audio.")
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["model"] = self.config["model"].replace("_", "-")
        return metadata

    def get_folder_name(self):
        c = self.config
        name = f"qwen3-omni_{c['model'].replace('/', '-')}_device-{c['device']}_dtype-{c['dtype']}"
        name = name.replace("/", "-")
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
