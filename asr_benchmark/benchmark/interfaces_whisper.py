import logging
import torch
import os
import time
import requests
import subprocess
import ssak.utils.vad
from asr_benchmark.utils.benchmark import load_audio, linstt_streaming
from pathlib import Path
from asr_benchmark.benchmark.interfaces import Model


logger = logging.getLogger(__name__)

class LintoSttModel(Model): 
    
    def load(self) -> None:
        p = subprocess.Popen(["docker", "ps", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if b"bench_container" in out:        
            subprocess.run(["docker", "stop", "bench_container"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
        out = open("docker.log", "w")
        cache_folder = self.config.get('cache_folder',  Path.home() / ".cache")
        build_args = f"-v {str(cache_folder)}:/root/.cache --env SERVICE_MODE={'http' if not self.config['streaming'] else 'websocket'} --env VAD={self.config['vad']} --env DEVICE={self.config['device']} --env USE_ACCURATE={self.config['accurate']} \
--env LANGUAGE={self.language} --env MODEL={self.config['model']} --env NUM_THREADS=4 --env CONCURRENCY=0"
        if self.config['streaming']:
            build_args += f" --env STREAMING_MIN_CHUNK_SIZE={self.config['streaming_min_chunk_size']} --env STREAMING_BUFFER_TRIMMING_SEC={self.config['streaming_buffer_trimming_sec']}"
        if self.config['device']!="cpu":
            build_args += f" --gpus all"
        cmd=f"docker run --rm -p {self.config.get('port', 8080)}:80 --name bench_container {build_args} {self.config['docker_image']}"
        out.write(cmd+"\n")
        p = subprocess.Popen(cmd.split(), stdout=out, stderr=out)
        total_wait_time = 600
        retry_interval = 2
        elapsed_time = 0
        time.sleep(0.5)
        self.model = f"{self.config['server']}:{self.config['port']}"
        while elapsed_time < total_wait_time:
            try:
                response = requests.head(f"http://{self.model}/healthcheck")
                if response.status_code == 200 or response.status_code == 400:
                    return 
            except requests.ConnectionError:
                pass
            if p.poll() is not None:
                raise RuntimeError(f"The server container has stopped for an unexpected reason.")
            time.sleep(retry_interval)
            elapsed_time += retry_interval
        raise RuntimeError(f"Server did not start in {total_wait_time} seconds")
           
    def load_audio(self, audio, start=0.0, duration=None):
        return load_audio(audio, return_format="file", start=start, duration=duration)        
     
    def transcribe(self, audio: str) -> str:
        output = dict()
        if self.config['streaming']:
            text, latencies = linstt_streaming(audio, ws_api=f"ws://{self.model}/streaming", stream_wait=self.config['streaming_wait'], \
                stream_duration=self.config['streaming_chunk'], compute_latency=self.config['compute_latency'])
            output['text'] = text.replace("\n", " ")
            output['latency'] = latencies
        else:
            import json
            cmd = f'curl -X POST "http://{self.model}/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@"{audio}";type=audio/wav"'
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            text = res.stdout.decode('utf-8')
            output['text'] = json.loads(text)['text']
        os.remove(audio)
        return output
    
    def cleanup(self):
        p = subprocess.Popen(["docker", "ps", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if b"bench_container" in out:        
            subprocess.run(["docker", "stop", "bench_container"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 
    
    def can_output_word_timestamps(self):
        return True
      
    def add_defaults_to_config(self, config):
        config['server'] = "localhost"
        config['port'] = 8080
        config['vad'] = config.get('vad', 'false')
        config['device'] = config.get('device', 'cuda')
        config['accurate'] = config.get('accurate', 'true')
        if config['accurate'] == 'true':
            config['beam_size'] = 5
            config['best_of'] = 5
            config['temperature'] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        else:
            config['beam_size'] = 1
            config['best_of'] = 1
            config['temperature'] = 0.0
        config['streaming'] = config.get('streaming', False)
        config['docker_image'] = config.get('docker_image', 'whisper/Dockerfile.ctranslate2')
        if config['streaming']:
            config['streaming_min_chunk_size'] = config.get('streaming_min_chunk_size', 0.5)
            config['streaming_buffer_trimming_sec'] = config.get('streaming_buffer_trimming_sec', 10)
            config['streaming_wait'] = config.get('streaming_wait', 0.5)
            config['streaming_chunk'] = config.get('streaming_chunk', 0.5)
        return super().add_defaults_to_config(config)
    
    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"linto-stt_{tot_config['model']}_accurate-{tot_config['accurate']}"
        name += f"_vad-{tot_config['vad']}_device-{tot_config['device']}_{self.config['docker_image'].replace(':', '-')}"
        if tot_config['streaming']:
            name += f"_streaming-{tot_config['streaming_min_chunk_size']}-{tot_config['streaming_buffer_trimming_sec']}-{tot_config['streaming_wait']}-{tot_config['streaming_chunk']}"
            name += "_latency" if tot_config['compute_latency'] else ""
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name
    
class OpenAIModel(Model):
    
    def load(self) -> None:
        import whisper
        self.model = whisper.load_model(self.config['model'], self.config['device'], download_root=self.config.get('cache_dir', None))
    
    def transcribe(self, audio: str) -> str:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        output = dict()
        result = self.model.transcribe(audio, word_timestamps=False, no_speech_threshold=None)
        output['text'] = result['text']
        return output

    def can_output_word_timestamps(self):
        return True
    
    def cleanup(self):
        torch.cuda.empty_cache()

    def add_defaults_to_config(self, config):
        config['vad'] = config.get('vad', 'false')
        config['device'] = config.get('device', 'cuda')
        return super().add_defaults_to_config(config)

    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"openai_{tot_config['model']}_vad-{tot_config['vad']}_device-{tot_config['device']}"
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name

class FasterWhisperModel(Model):
    def __init__(self, config) -> None:
        super().__init__(config)
        if self.config['accurate']:
            self.transcribe_kwargs['beam_size'] = 5
            self.transcribe_kwargs['best_of'] = 5
            self.transcribe_kwargs['temperature'] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        else:
            self.transcribe_kwargs['beam_size'] = 1
            self.transcribe_kwargs['best_of'] = 1
            self.transcribe_kwargs['temperature'] = 0.0
    
    def load(self) -> None:
        model_kwargs = {'device': self.config['device'], 'precision': self.config['precision']}
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        if self.config['model'] is None:
            raise ValueError("path must be set")
        if model_kwargs['device']=="cpu" and model_kwargs['precision'] =="float16":
            raise ValueError("Float16 is not supported on CPU")
        model_kwargs['compute_type'] = model_kwargs.pop('precision', '')
        self.model = WhisperModel(self.config['model'], download_root=self.config.get('cache_dir', None), **model_kwargs)
        if self.config['batch_size']>1:
            self.model = BatchedInferencePipeline(model=self.model)

    def transcribe(self, audio: str) -> str:
        if self.config['vad'] and self.config['vad'] in ['auditok','silero', 'pyannote']:
            audio, _ = ssak.utils.vad.remove_non_speech(audio, method=self.config['vad'])
        output = dict()
        if self.config['batch_size']>1:
            segments, info = self.model.transcribe(audio, language=self.language, **self.transcribe_kwargs, batch_size=self.config['batch_size'], condition_on_previous_text=self.config['previous_text'])
        else:
            segments, info = self.model.transcribe(audio, language=self.language, **self.transcribe_kwargs, condition_on_previous_text=self.config['previous_text'])
        output['text'] = " ".join([seg.text for seg in segments])
        return output

    def can_output_word_timestamps(self):
        return True
    
    def add_defaults_to_config(self, config):
        config['vad'] = config.get('vad', 'false')
        config['precision'] = config.get('precision', 'float16')
        config['device'] = config.get('device', 'cuda')
        config['accurate'] = config.get('accurate', False)
        config['previous_text'] = config.get('previous_text', False)
        config['batch_size'] = config.get('batch_size', 1)
        return super().add_defaults_to_config(config)

    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"faster-whisper_{tot_config['model']}_vad-{tot_config['vad']}_device-{tot_config['device']}_precision-{tot_config['precision']}"
        name += f"_accurate-{tot_config['accurate']}_previous-{tot_config['previous_text']}"
        if tot_config['batch_size']>1:
            name += f"_batchsize-{tot_config['batch_size']}"
        name = name.replace("/", "-")
        name += "_rtf" if tot_config['compute_rtf'] else ""
        return name