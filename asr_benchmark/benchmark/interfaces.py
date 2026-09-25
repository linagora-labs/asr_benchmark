import logging
import requests
import subprocess
from tqdm import tqdm
from asr_benchmark.utils.benchmark import load_audio

logger = logging.getLogger(__name__)

class Model():
    def __init__(self, config) -> None:
        config = self.add_defaults_to_config(config)
        self.config = config
        self.transcribe_kwargs = {}
    
    def load(self) -> None:
        raise NotImplementedError("Not supposed to be called")
    
    def load_audio(self, audio: str, start=0.0, duration=None):
        return load_audio(audio, return_format="librosa", start=start, duration=duration)
        
    def transcribe(self, audio: str) -> str:
        raise NotImplementedError("Not supposed to be called")
    
    def transcribe_batch(self, data: list) -> list[str]:
        predictions = []
        for row in tqdm(data, desc=f"Transcribing {data[0]['name']} with {self.config['model']}"):
            audio = self.load_audio(row['audio_filepath'], start=row['offset'], duration=row['duration'])
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
            for i, prediction in zip(batch, self.transcribe_many(audios)):
                predictions[i] = prediction
        return predictions

    def transcribe_many(self, audios: list) -> list[dict]:
        raise NotImplementedError("Not supposed to be called")

    def cleanup(self):
        pass
    
    def can_output_word_timestamps(self):
        logger.warning("The function can_output_timestamps is not implemented for this model.")
        return False
    
    def get_metadata(self):
        metadata = self.config.copy()
        metadata['word_timestamps'] = self.can_output_word_timestamps()
        return metadata
    
    def get_folder_name(self):
        raise NotImplementedError("Not supposed to be called")
    
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
