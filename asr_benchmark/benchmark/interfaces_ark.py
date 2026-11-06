import logging

import torch

from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ArkASRModel(Model):
    """Backend for ARK-ASR (Edge0/ARK-ASR-3B): whisper-style encoder + MLP adapter +
    Qwen-style LLM decoder, ASR only, with remote code (trust_remote_code).

    Written for transformers 4.57 (the version of its remote code). No language
    setting: the model detects it. Follows the inference of the model card: chat
    prompt "Please transcribe this audio.", greedy decoding, special/control tokens
    banned from the generation.
    """

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        model = self.config["model"]
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # Batched generation: left padding, so that the generated tokens start at the same index.
        self.tokenizer.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        self.dtype = _DTYPES.get(self.config["dtype"], torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, torch_dtype=self.dtype, attn_implementation="sdpa",
        ).to(self.config["device"]).eval()
        self.bad_words_ids = self._bad_words_ids()

    def _bad_words_ids(self):
        """Every special / added control token but the end of sequence (model card)."""
        eos = self.tokenizer.eos_token_id
        keep = {eos} if isinstance(eos, int) else set(eos or [])
        bad = set(self.tokenizer.all_special_ids) - keep
        bad.update(
            token_id for token, token_id in self.tokenizer.get_added_vocab().items()
            if token.startswith("<") and token.endswith(">") and token_id not in keep
        )
        return [[token_id] for token_id in sorted(bad)]

    def transcribe(self, audio) -> dict:
        return self.transcribe_many([audio])[0]

    def transcribe_batch(self, data: list) -> list[dict]:
        if self.config["batch_size"] <= 1:
            return super().transcribe_batch(data)
        return self.transcribe_by_batches(data, self.config["batch_size"])

    def transcribe_many(self, audios: list) -> list[dict]:
        conversations = [[{"role": "user", "content": [
            {"type": "audio", "array": audio, "sampling_rate": 16000},
            {"type": "text", "text": self.config["prompt"]},
        ]}] for audio in audios]
        inputs = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, return_tensors="pt", sampling_rate=16000,
            audio_padding="longest", text_kwargs={"padding": "longest"}, audio_max_length=30 * 16000,
        ).to(self.model.device)
        if "audios" in inputs:
            inputs["audios"] = inputs["audios"].to(dtype=self.dtype)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.config["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=self.bad_words_ids,
            )
        texts = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [{"text": text.strip()} for text in texts]

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        config["dtype"] = config.get("dtype", "bfloat16")
        config["max_new_tokens"] = int(config.get("max_new_tokens", 256))
        config["prompt"] = config.get("prompt", "Please transcribe this audio.")
        # Not in the folder name: it only changes speed, up to bf16 padding noise.
        config["batch_size"] = int(config.get("batch_size", 1))
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.pop("batch_size", None)
        return metadata

    def get_folder_name(self):
        c = self.config
        name = f"ark-asr_{c['model'].replace('/', '-')}{self.device_tag()}"
        name += self.detail(f"_dtype-{c['dtype']}")
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
