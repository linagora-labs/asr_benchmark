import logging
import requests
import subprocess
from tqdm import tqdm
from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark import contextual_biasing

logger = logging.getLogger(__name__)

class Model():
    # Contextual biasing (list of terms given at inference, see contextual_biasing.py).
    # Backends that can use such a list set this to True and read the terms of the
    # current utterance with self.biasing_terms (in transcribe) or
    # self.terms_for_row(row) (in batched paths). Default template of the text built
    # from the terms (format_biasing), overridable with the config key
    # contextual_biasing_template.
    SUPPORTS_CONTEXTUAL_BIASING = False
    DEFAULT_BIASING_TEMPLATE = "{terms}"

    def __init__(self, config) -> None:
        config = self.add_defaults_to_config(config)
        self.config = config
        self.transcribe_kwargs = {}
        self.biasing_terms = None
        if contextual_biasing.config_variant(config) is not None and not self.SUPPORTS_CONTEXTUAL_BIASING:
            raise ValueError(
                f"Backend {config.get('backend')} ({config.get('model')}) cannot take a contextual biasing list: "
                f"remove contextual_biasing from its config (its baseline is the run without it)"
            )

    def load(self) -> None:
        raise NotImplementedError("Not supposed to be called")

    def load_audio(self, audio: str, start=0.0, duration=None):
        return load_audio(audio, return_format="librosa", start=start, duration=duration)

    def transcribe(self, audio: str) -> str:
        raise NotImplementedError("Not supposed to be called")

    def set_biasing_terms(self, terms):
        """Terms (list of str, or None) used by the next transcribe() calls."""
        self.biasing_terms = list(terms) if terms else None

    def terms_for_row(self, row):
        """Terms to give to the model for a manifest row (None if the config gives none)."""
        return contextual_biasing.terms_for_row(row, self.config)

    def format_biasing(self, terms, **kwargs):
        """Text built from the terms with the template of the config
        (contextual_biasing_template) or of the backend: {terms} is replaced by the terms
        joined with ", ", other placeholders (e.g. {prompt}) by kwargs. None without terms."""
        if not terms:
            return None
        template = self.config.get("contextual_biasing_template") or self.DEFAULT_BIASING_TEMPLATE
        return template.format(terms=", ".join(terms), **kwargs)

    def transcribe_batch(self, data: list) -> list[str]:
        predictions = []
        for row in tqdm(data, desc=f"Transcribing {data[0]['name']} with {self.config['model']}"):
            audio = self.load_audio(row['audio_filepath'], start=row['offset'], duration=row['duration'])
            self.set_biasing_terms(self.terms_for_row(row))
            predictions.append(self.transcribe(audio))
        return predictions

    def transcribe_by_batches(self, data: list, batch_size: int) -> list[dict]:
        """Batched version of transcribe_batch for models implementing
        transcribe_many(audios) -> list[dict]. Rows are grouped by similar duration
        (longest first, so an OOM shows up immediately) to limit padding; predictions
        are returned in the order of `data`."""
        order = sorted(range(len(data)), key=lambda i: data[i].get('duration') or 0, reverse=True)
        predictions = [None] * len(data)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        for batch in tqdm(batches, desc=f"Transcribing {data[0]['name']} with {self.config['model']} (batches of {batch_size})"):
            audios = [self.load_audio(data[i]['audio_filepath'], start=data[i]['offset'], duration=data[i]['duration']) for i in batch]
            if contextual_biasing.config_variant(self.config) is None:
                outputs = self.transcribe_many(audios)
            else:
                outputs = self.transcribe_many(audios, terms=[self.terms_for_row(data[i]) for i in batch])
            for i, prediction in zip(batch, outputs):
                predictions[i] = prediction
        return predictions

    def transcribe_many(self, audios: list, terms: list = None) -> list[dict]:
        """Transcriptions of several audios. `terms`: one biasing list (or None) per audio,
        only passed when the config gives lists (backends supporting contextual biasing)."""
        raise NotImplementedError("Not supposed to be called")

    def cleanup(self):
        pass
    
    def can_output_word_timestamps(self):
        logger.warning("The function can_output_timestamps is not implemented for this model.")
        return False
    
    def get_metadata(self):
        metadata = self.config.copy()
        metadata['word_timestamps'] = self.can_output_word_timestamps()
        if contextual_biasing.config_variant(metadata) is None:
            for key in [k for k in metadata if k.startswith("contextual_biasing")]:
                del metadata[key]
        return metadata

    def biasing_folder_params(self):
        """Backend settings of the contextual biasing that name its results (e.g. a
        boosting weight), appended to the tag of get_full_folder_name."""
        return []

    def get_full_folder_name(self):
        """Folder of the results: get_folder_name() plus the contextual biasing tag
        ("_cb-<list>[-<backend setting>...]" when a list is given to the model, nothing
        otherwise)."""
        tag = contextual_biasing.folder_tag(self.config)
        if tag:
            tag += "".join(f"-{param}" for param in self.biasing_folder_params())
        return self.get_folder_name() + tag
    
    def count_parameters(self):
        """Number of parameters of the loaded model (self.model, or the model of a
        transformers pipeline), None when it is not a torch module of this process
        (servers, CTranslate2...)."""
        model = getattr(self, "model", None)
        for module in (model, getattr(model, "model", None)):
            if callable(getattr(module, "parameters", None)):
                try:
                    return sum(p.numel() for p in module.parameters())
                except Exception:
                    return None
        return None

    def get_folder_name(self):
        raise NotImplementedError("Not supposed to be called")

    # Folder names: by default only the model identity (backend, model, decoder...) so
    # that model comparisons stay readable. With `full_name: true` in the config, the
    # settings (dtype, device, vad, decoding params...) are added, to compare several
    # settings of the same model.
    def detail(self, tag):
        """`tag` if the config asks for full folder names, else ""."""
        return tag if self.config.get("full_name", False) else ""

    # vad/device are left out even in full names for the usual values (no VAD, cuda).
    def vad_tag(self):
        vad = self.config.get("vad")
        return "" if str(vad).lower() in ("false", "none", "") else self.detail(f"_vad-{vad}")

    def device_tag(self):
        device = self.config.get("device")
        return "" if device in ("cuda", None) else self.detail(f"_device-{device}")
    
    def add_defaults_to_config(self, config):
        # config["language"] = config.get("language", "fr")
        return config
    
class HttpAPIModel(Model):
    def load(self) -> None:
        server = self.config.get('server', None)
        if self.config.get('port', None):
            server = f"{server}:{self.config['port']}"
        if not server.startswith("http"):
            server = f"http://{server}"
        if requests.head(f"{server}/healthcheck").status_code == 200:
            logger.info("API is up and running")
        else:
            logger.error("API is not running")
            raise ValueError("API is not running")
        self.model = server
    
    def load_audio(self, audio, start=0.0, duration=None):
        return load_audio(audio, return_format="file", start=start, duration=duration)
    
    def transcribe(self, audio: str) -> str:        
        import json
        cmd = f'curl -X POST "{self.model}/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@{audio};type=audio/wav"'
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        text = res.stdout.decode('utf-8')
        text = json.loads(text)['text']
        output = {"text": text}
        return output
        
    def add_defaults_to_config(self, config):
        config['server'] = config.get('server', 'http://localhost:8080')
        return super().add_defaults_to_config(config)
    
    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"http-api-{tot_config['model']}"
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name
