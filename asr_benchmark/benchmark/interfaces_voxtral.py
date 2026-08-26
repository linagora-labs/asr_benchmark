import logging

import torch

from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class TransformersVoxtralModel(Model):
    """Backend for Mistral's Voxtral audio LLMs (e.g. mistralai/Voxtral-Small-24B-2507).

    Voxtral exposes a dedicated transcription mode: the processor builds a
    "transcription request" (audio + optional language hint) and the model
    generates the transcript. This is distinct from the realtime variant, which
    is handled by TransformersVoxtralRealtimeModel in interfaces_transformers.
    """

    def load(self) -> None:
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        device = self.config["device"]
        torch_dtype = _DTYPES[self.config["dtype"]]
        if device == "cpu" and self.config["dtype"] != "float32":
            logger.warning("Forcing float32 on CPU for Voxtral.")
            torch_dtype = torch.float32

        # mistral_format=True selects the MistralCommonBackend tokenizer, which
        # apply_transcription_request() requires (the default TokenizersBackend
        # has no .tokenizer.encode_transcription). It also uses Mistral's native
        # tokenizer, avoiding the incorrect-regex warning of the converted one.
        self.processor = AutoProcessor.from_pretrained(
            self.config["model"],
            mistral_format=True,
        )
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.config["model"],
            dtype=torch_dtype,
            device_map=device,
        ).eval()

    def load_audio(self, audio: str, start=0.0, duration=None):
        return load_audio(audio, return_format="librosa", start=start, duration=duration)

    def transcribe(self, audio) -> dict:
        # Audio is passed as an array (not a path) so that the manifest's
        # offset/duration are honoured; the processor re-encodes it through
        # soundfile, which requires an explicit container format.
        inputs = self.processor.apply_transcription_request(
            audio=audio,
            model_id=self.config["model"],
            language=self.config["language"],
            sampling_rate=16000,
            format="WAV",
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        generate_kwargs = dict(max_new_tokens=self.config["max_new_tokens"])
        if self.config["do_sample"]:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.config["temperature"]
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        # Keep only the newly generated tokens (drop the transcription request).
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return {"text": text.strip()}

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        config["dtype"] = config.get("dtype", "bfloat16")
        config["language"] = config.get("language", "fr")
        config["max_new_tokens"] = int(config.get("max_new_tokens", 2048))
        config["do_sample"] = config.get("do_sample", False)
        if config["do_sample"]:
            config["temperature"] = float(config.get("temperature", 0.2))
        return super().add_defaults_to_config(config)

    def get_folder_name(self):
        c = self.config
        model = c["model"].replace("/", "-")
        name = f"voxtral_{model}_device-{c['device']}_dtype-{c['dtype']}"
        if c["do_sample"]:
            name += f"_temperature-{c['temperature']}"
        else:
            name += "_greedy"
        name = name.replace("/", "-")
        name += "_rtf" if c["compute_rtf"] else ""
        return name

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["model"] = self.config["model"].replace("_", "-")
        return metadata
