import logging

import numpy as np
import torch

from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)


class HojoASRModel(Model):
    """Backend for Hojo-ASR (HojoAI/Hojo-ASR-Multi-V1): Qwen3-Omni audio encoder +
    conformer adapter + Qwen3-4B-Instruct decoder, run with the `hojo-asr` package
    (pip, needs transformers 4.57 and openai-whisper; installed --no-deps over the
    image's torch).

    No language setting (the model detects it). Decoding settings come from the
    config.yaml of the checkpoint (beam search 4, repetition penalty 2). The package
    builds the audio encoder with flash_attention_2: without flash-attn, sdpa.
    """

    def load(self) -> None:
        import importlib.util

        import hojo_asr.hojo_asr_model as hojo_asr_model

        if importlib.util.find_spec("flash_attn") is None:
            encoder_cls = hojo_asr_model.ModifyQwen3OmniMoeAudioEncoder

            def sdpa_encoder(config, *args, **kwargs):
                config._attn_implementation = "sdpa"
                return encoder_cls(config, *args, **kwargs)

            hojo_asr_model.ModifyQwen3OmniMoeAudioEncoder = sdpa_encoder
        if importlib.util.find_spec("torchcodec") is None:
            self._read_wav_with_soundfile()
        self.model = hojo_asr_model.HOJO_ASR.load_model(self.config["model"], device=self.config["device"])

    @staticmethod
    def _read_wav_with_soundfile():
        """The package reads the wavs with torchaudio.load, which recent torchaudio
        delegates to torchcodec (not in the image): read them with soundfile in its
        dataset module, torchaudio's resampling kept."""
        import types

        import hojo_asr.dataset as dataset
        import soundfile
        import torchaudio

        def load(source):
            data, sample_rate = soundfile.read(source, dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sample_rate

        dataset.torchaudio = types.SimpleNamespace(load=load, transforms=torchaudio.transforms)

    def load_audio(self, audio: str, start=0.0, duration=None):
        # 16 kHz arrays: run_infer takes them (in memory). Not return_format="file": the
        # benchmark rewrites one temporary wav per process, a batch would get N times
        # the same file.
        return load_audio(audio, return_format="librosa", start=start, duration=duration)

    def transcribe(self, audio) -> dict:
        return self.transcribe_many([audio])[0]

    def transcribe_batch(self, data: list) -> list[dict]:
        if self.config["batch_size"] <= 1:
            return super().transcribe_batch(data)
        return self.transcribe_by_batches(data, self.config["batch_size"])

    def transcribe_many(self, audios: list) -> list[dict]:
        results = self.model.run_infer([np.asarray(audio, dtype=np.float32) for audio in audios], batch_size=len(audios))
        # run_infer sorts each batch by length and loses the way back (prepare_sample
        # drops original_index): the texts come in the sorted order. Their key,
        # "sample_<index in the input list>", is right: map them back by key.
        texts = {result["key"]: result["text"].strip() for result in results}
        keys = [f"sample_{i}" for i in range(len(audios))]
        missing = [key for key in keys if key not in texts]
        if missing:
            logger.warning(f"Hojo-ASR returned no transcription for {len(missing)} file(s) (filtered out by its loader)")
        return [{"text": texts.get(key, "")} for key in keys]

    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config["device"] = config.get("device", "cuda")
        # Not in the folder name: it only changes speed, up to padding noise.
        config["batch_size"] = int(config.get("batch_size", 1))
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.pop("batch_size", None)
        return metadata

    def get_folder_name(self):
        c = self.config
        name = f"hojo-asr_{c['model'].replace('/', '-')}{self.device_tag()}"
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
