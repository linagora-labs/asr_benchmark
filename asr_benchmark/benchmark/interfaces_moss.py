import logging
import re

import torch

from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)

# Canonical default prompt from the model card / examples/prompts.md (Chinese),
# optimized for timestamped transcription and speaker diarization. This exact string
# (auto-detect, no language specified) is the ONLY documented, model-intended prompt,
# and it is the best-performing setting on French in practice -- so it is the default.
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）"
    "开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

# --- Language pinning below is UNOFFICIAL / experimental -----------------------------
# MOSS has no documented way to force an output language; the model card only says it
# transcribes "in whatever language it detects". The template below inserts a Chinese
# language name before 文本 ("text") as an experiment. It is NOT from the model card.
# (An *English* language instruction like "Transcribe the audio in French" is worse: it
# makes the model translate French speech to English -- a large regression on VoxPopuli/
# YouTube -- which is why any pinning, if used at all, stays in the native Chinese.)
_PROMPT_ZH_LANG = (
    "请将音频转写为{lang}文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）"
    "开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
_LANGUAGE_NAMES_ZH = {
    "fr": "法语", "en": "英语", "de": "德语", "es": "西班牙语", "it": "意大利语",
    "pt": "葡萄牙语", "nl": "荷兰语", "zh": "中文", "ar": "阿拉伯语", "ru": "俄语",
    "ja": "日语", "ko": "韩语",
}


def build_prompt(language):
    """Transcription prompt.

    ``language`` is None by default -> the documented auto-detect prompt (recommended).
    Passing a language code applies an UNOFFICIAL Chinese language-pinned prompt (see the
    note above); unknown codes fall back to auto-detect rather than an English prompt.
    """
    if not language:
        return DEFAULT_PROMPT
    name = _LANGUAGE_NAMES_ZH.get(language)
    if name is None:
        logger.warning(f"No Chinese name for language {language!r}; using the auto-detect prompt.")
        return DEFAULT_PROMPT
    return _PROMPT_ZH_LANG.format(lang=name)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# Matches bracketed timestamps ([0.48]) and speaker labels ([S01]) in the output.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class MossTranscribeDiarizeModel(Model):
    """Backend for OpenMOSS-Team/MOSS-Transcribe-Diarize.

    An end-to-end audio LLM that jointly performs transcription, timestamping and
    speaker diarization in a single pass. For WER benchmarking, the timestamps and
    speaker labels are stripped by default to recover plain transcript text.
    """

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self.config["device"]
        torch_dtype = _DTYPES[self.config["dtype"]]
        if device == "cpu" and self.config["dtype"] != "float32":
            logger.warning("Forcing float32 on CPU for MOSS-Transcribe-Diarize.")
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.config["model"],
            trust_remote_code=True,
            dtype="auto",
        )
        self.model = model.to(dtype=torch_dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.config["model"],
            trust_remote_code=True,
        )
        self.torch_dtype = torch_dtype

    def load_audio(self, audio: str, start=0.0, duration=None):
        return load_audio(audio, return_format="librosa", start=start, duration=duration)

    def transcribe(self, audio) -> dict:
        return self.transcribe_many([audio])[0]

    def transcribe_batch(self, data: list) -> list[dict]:
        if self.config["batch_size"] <= 1:
            return super().transcribe_batch(data)
        return self.transcribe_by_batches(data, self.config["batch_size"])

    @staticmethod
    def _left_pad(inputs, pad_token_id):
        """The remote-code processor pads prompts on the right, which breaks batched
        generation (new tokens would follow the padding). Move the padding to the left:
        audio features are injected at the audio placeholder tokens in row order, which
        this does not change."""
        mask = inputs["attention_mask"]
        lengths = mask.sum(dim=1)
        input_ids = torch.full_like(inputs["input_ids"], pad_token_id)
        attention_mask = torch.zeros_like(mask)
        width = mask.shape[1]
        for row, length in enumerate(lengths.tolist()):
            input_ids[row, width - length:] = inputs["input_ids"][row, :length]
            attention_mask[row, width - length:] = 1
        inputs["input_ids"], inputs["attention_mask"] = input_ids, attention_mask
        return inputs

    def transcribe_many(self, audios: list) -> list[dict]:
        device = self.config["device"]
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": self.config["prompt"]},
                ]}],
                tokenize=False, add_generation_prompt=True,
            )
            for audio in audios
        ]
        inputs = self.processor(text=texts, audio=list(audios), return_tensors="pt")
        if len(audios) > 1:
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id or 0
            inputs = self._left_pad(inputs, pad_token_id)
        inputs = inputs.to(device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.torch_dtype)

        generate_kwargs = dict(max_new_tokens=self.config["max_new_tokens"])
        if self.config["do_sample"]:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.config["temperature"]
        else:
            generate_kwargs["do_sample"] = False
        # Repetition suppression: greedy decoding on short/ambiguous clips otherwise
        # collapses into degenerate loops ("a, a, a, ...") that explode the WER.
        if self.config["no_repeat_ngram_size"]:
            generate_kwargs["no_repeat_ngram_size"] = self.config["no_repeat_ngram_size"]
        if self.config["repetition_penalty"] and self.config["repetition_penalty"] != 1.0:
            generate_kwargs["repetition_penalty"] = self.config["repetition_penalty"]

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        # Keep only the newly generated tokens (drop the prompt).
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        predictions = []
        for row in generated:
            raw = self.processor.tokenizer.decode(row, skip_special_tokens=True)
            if self.config["raw_output"]:
                predictions.append({"text": raw.strip()})
            else:
                predictions.append({"text": self._strip_annotations(raw)})
        return predictions

    @staticmethod
    def _strip_annotations(text: str) -> str:
        text = _THINK_RE.sub(" ", text)
        text = _BRACKET_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        config["dtype"] = config.get("dtype", "bfloat16")
        config["max_new_tokens"] = int(config.get("max_new_tokens", 2048))
        config["do_sample"] = config.get("do_sample", False)
        if config["do_sample"]:
            config["temperature"] = float(config.get("temperature", 0.2))
        # Language: None by default -> the documented auto-detect prompt (recommended,
        # and best-performing on French). Set a code (e.g. "fr") to try the UNOFFICIAL
        # Chinese language-pinned prompt; an English pin makes MOSS translate to English.
        config["language"] = config.get("language", None)
        # An explicit prompt (if given) wins; otherwise build one from the language.
        config["prompt"] = config.get("prompt") or build_prompt(config["language"])
        # Repetition suppression, OFF by default to match the model's validated greedy
        # decoding (its generation_config sets neither). Opt in to combat the rare loop
        # cases, e.g. repetition_penalty ~1.1-1.15; see transcribe.
        config["no_repeat_ngram_size"] = int(config.get("no_repeat_ngram_size", 0))
        config["repetition_penalty"] = float(config.get("repetition_penalty", 1.0))
        # Strip timestamps/speaker labels for WER computation (default), or keep the
        # raw diarized transcript when raw_output is true.
        config["raw_output"] = config.get("raw_output", False)
        # Not in the folder name (like NeMo): it only changes speed, up to bf16 padding noise.
        config["batch_size"] = int(config.get("batch_size", 1))
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["model"] = self.config["model"].replace("_", "-")
        metadata.pop("batch_size", None)
        return metadata

    def get_folder_name(self):
        c = self.config
        model = c["model"].replace("/", "-")
        name = f"moss_{model}{self.device_tag()}"
        name += self.detail(f"_dtype-{c['dtype']}")
        name += self.detail(f"_lang-{c['language']}" if c["language"] else "_lang-auto")
        if c["do_sample"]:
            name += self.detail(f"_temperature-{c['temperature']}")
        else:
            name += self.detail("_greedy")
        if c["no_repeat_ngram_size"] or (c["repetition_penalty"] and c["repetition_penalty"] != 1.0):
            name += self.detail(f"_norep{c['no_repeat_ngram_size']}-rep{c['repetition_penalty']}")
        if c["raw_output"]:
            name += self.detail("_raw")
        name = name.replace("/", "-")
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
