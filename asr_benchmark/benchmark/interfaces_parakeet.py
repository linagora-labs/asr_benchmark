import json
import logging

import torch

from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)


def unpack_ternary(qweight, scales, in_features, group_size):
    """Dense weight of a ternary layer of moondream/parakeet-redux (format "thrush-ternary-v2"
    of its ternary.json): each byte packs 5 base-3 codes of a row, least significant digit
    first, code in {0,1,2} for -1/0/+1; w = scales[row, col // group_size] * (code - 1)."""
    digits = qweight.long().unsqueeze(-1) // (3 ** torch.arange(5, device=qweight.device)) % 3
    codes = digits.reshape(qweight.shape[0], -1)[:, :in_features]
    scales = scales.float().repeat_interleave(group_size, dim=1)[:, :in_features]
    return scales * (codes - 1)


class TransformersParakeetModel(Model):
    """Parakeet TDT checkpoints in the transformers format (ParakeetForTDT, transformers >= 5.6),
    e.g. the moondream post-trainings of nvidia/parakeet-tdt-0.6b-v3: parakeet-ultra (dense) and
    parakeet-redux (ternary encoder, unpacked to dense weights at load). Their repos have neither
    processor nor generation config: both are taken from `processor` (the original model, same
    features and tokenizer). Their VAD head (used by moondream's Photon runtime to segment long
    audio) is not loaded: the benchmark segments are short. Greedy TDT decoding by batches."""

    def load(self) -> None:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT, ParakeetTDTConfig

        repo, device = self.config['model'], self.config['device']
        self.processor = AutoProcessor.from_pretrained(self.config['processor'])
        model = ParakeetForTDT(ParakeetTDTConfig.from_pretrained(repo))
        model.generation_config = GenerationConfig.from_pretrained(self.config['processor'])
        state = load_file(hf_hub_download(repo, "model.safetensors"))
        params = dict(model.named_parameters())
        for name in [k[:-len(".qweight")] for k in state if k.endswith(".qweight")]:
            shape = params[f"{name}.weight"].shape
            weight = unpack_ternary(state.pop(f"{name}.qweight"), state.pop(f"{name}.scales"),
                                    shape[1], model.config.ternary_group_size)
            state[f"{name}.weight"] = weight.reshape(shape)
        missing, unexpected = model.load_state_dict(state, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("vad_head.")]
        if missing or unexpected:
            raise RuntimeError(f"{repo}: missing weights {missing}, unexpected weights {unexpected}")
        self.dtype = getattr(torch, self.config['dtype'])
        self.model = model.to(device, self.dtype).eval()

    def transcribe_many(self, audios: list) -> list[dict]:
        inputs = self.processor(audios, sampling_rate=16000, return_tensors="pt").to(self.config['device'])
        inputs["input_features"] = inputs["input_features"].to(self.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs)
        texts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        return [{'text': text.strip()} for text in texts]

    def transcribe(self, audio) -> dict:
        return self.transcribe_many([audio])[0]

    def transcribe_batch(self, data: list) -> list[dict]:
        return self.transcribe_by_batches(data, int(self.config['batch_size']))

    def can_output_word_timestamps(self):
        return False

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        # The checkpoints are float16/float32 and published as full precision.
        config["dtype"] = config.get("dtype", "float32")
        config["batch_size"] = int(config.get("batch_size", 16))
        config["processor"] = config.get("processor", "nvidia/parakeet-tdt-0.6b-v3")
        return super().add_defaults_to_config(config)

    def get_folder_name(self):
        c = self.config
        name = f"transformers-parakeet_{c['model'].replace('/', '-')}{self.device_tag()}"
        name += self.detail(f"_dtype-{c['dtype']}")
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
