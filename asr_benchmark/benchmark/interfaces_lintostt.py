import logging
import time
import requests
import websockets.sync.client, websockets.exceptions
import subprocess
from pathlib import Path
from asr_benchmark.utils.benchmark import load_audio, linstt_streaming
from asr_benchmark.benchmark.interfaces import Model


logger = logging.getLogger(__name__)


def healthcheck(model_url, streaming, check_transcribe=False):
    if streaming:
        try:
            with websockets.sync.client.connect("ws://localhost:8080") as ws: 
                ws.close()
            return True
        except (websockets.exceptions.WebSocketException, OSError):
            pass
    else:
        try:
            response = requests.get(f"http://{model_url}/healthcheck")
            if response.status_code == 200 or response.status_code == 400:
                if check_transcribe:
                    transcribe_check = requests.post(f"http://{model_url}/transcribe")
                    return transcribe_check.status_code != 405
                return True
        except requests.ConnectionError:
            pass
    return False

class LintoSttWhisperModel(Model):

    def load(self) -> None:
        p = subprocess.Popen(
            ["docker", "ps", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = p.communicate()
        if b"bench_container" in out:
            subprocess.run(
                ["docker", "stop", "bench_container"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
        out = open("docker.log", "w")
        cache_folder = self.config.get("cache_folder", Path.home() / ".cache")
        build_args = f"--env SERVICE_MODE={'http' if not self.config['streaming'] else 'websocket'} --env VAD={self.config['vad']} --env DEVICE={self.config['device']} --env USE_ACCURATE={self.config['accurate']} \
--env LANGUAGE={self.config['language']} --env MODEL={self.config['model']} --env NUM_THREADS=4 --env CONCURRENCY=0"
        build_args += f" -v {str(cache_folder)}:/home/appuser/.cache --env USER_ID={subprocess.check_output(['id', '-u']).decode().strip()} --env GROUP_ID={subprocess.check_output(['id', '-g']).decode().strip()}"
        if self.config["streaming"]:
            build_args += f" --env STREAMING_MIN_CHUNK_SIZE={self.config['streaming_min_chunk_size']} --env STREAMING_BUFFER_TRIMMING_SEC={self.config['streaming_buffer_trimming_sec']}"
        if self.config["device"] != "cpu":
            build_args += f" --gpus all"
        cmd = f"docker run --rm -p {self.config.get('port', 8080)}:80 --name bench_container {build_args} {self.config['docker_image']}"
        out.write(cmd + "\n")
        out.flush()
        p = subprocess.Popen(cmd.split(), stdout=out, stderr=out)
        total_wait_time = 800
        retry_interval = 2
        elapsed_time = 0
        time.sleep(0.5)
        self.model = f"{self.config['server']}:{self.config['port']}"
        while elapsed_time < total_wait_time:
            if healthcheck(self.model, self.config["streaming"], check_transcribe=True):
                return
            if p.poll() is not None:
                raise RuntimeError(
                    f"The server container has stopped for an unexpected reason."
                )
            time.sleep(retry_interval)
            print(f"Waiting for server to start... ({elapsed_time}/{total_wait_time} seconds elapsed)\r", end="")
            elapsed_time += retry_interval
        print()
        raise RuntimeError(f"Server did not start in {total_wait_time} seconds")

    def load_audio(self, audio, start=0.0, duration=None):
        return load_audio(audio, return_format="file", start=start, duration=duration)

    def transcribe(self, audio: str) -> str:
        output = dict()
        if self.config["streaming"]:
            text, latencies = linstt_streaming(
                audio,
                ws_api=f"ws://{self.model}/streaming",
                stream_wait=self.config["streaming_wait"],
                stream_duration=self.config["streaming_chunk"],
                compute_latency=self.config["compute_latency"],
            )
            output["text"] = text.replace("\n", " ")
            output["latency"] = latencies
        else:
            url = f"http://{self.model}/transcribe"
            with open(audio, "rb") as f:
                res = requests.post(
                    url,
                    files={"file": (Path(audio).name, f, "audio/wav")},
                    headers={"accept": "application/json"},
                )
            if res.status_code != 200:
                raise RuntimeError(
                    f"Transcription request failed (HTTP {res.status_code}): {res.text[:500]}"
                )
            parsed = res.json()
            if isinstance(parsed, str):
                import json

                parsed = json.loads(parsed)
            if "text" not in parsed:
                raise RuntimeError(
                    f"Transcription response missing 'text' key. Response: {parsed}"
                )
            output["text"] = parsed["text"]
        Path(audio).unlink()
        return output

    def cleanup(self):
        p = subprocess.Popen(
            ["docker", "ps", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = p.communicate()
        if b"bench_container" in out:
            subprocess.run(
                ["docker", "stop", "bench_container"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return

    def can_output_word_timestamps(self):
        return True

    def add_defaults_to_config(self, config):
        config["server"] = "localhost"
        config["port"] = 8080
        config["vad"] = config.get("vad", "false")
        config["device"] = config.get("device", "cuda")
        config["accurate"] = config.get("accurate", "true")
        if config["accurate"] == "true":
            config["beam_size"] = 5
            config["best_of"] = 5
            config["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        else:
            config["beam_size"] = 1
            config["best_of"] = 1
            config["temperature"] = 0.0
        config["streaming"] = config.get("streaming", False)
        config["docker_image"] = config.get(
            "docker_image", "whisper/Dockerfile.ctranslate2"
        )
        if config["streaming"]:
            config["streaming_min_chunk_size"] = config.get(
                "streaming_min_chunk_size", 0.5
            )
            config["streaming_buffer_trimming_sec"] = config.get(
                "streaming_buffer_trimming_sec", 10
            )
            config["streaming_wait"] = config.get("streaming_wait", 0.5)
            config["streaming_chunk"] = config.get("streaming_chunk", 0.5)
        return super().add_defaults_to_config(config)

    def get_folder_name(self):
        tot_config = self.config.copy()
        name = (
            f"linto-stt-whisper_{tot_config['model']}_accurate-{tot_config['accurate']}"
        )
        name += f"_vad-{tot_config['vad']}_device-{tot_config['device']}_{self.config['docker_image'].replace(':', '-')}"
        if tot_config["streaming"]:
            name += f"_streaming-{tot_config['streaming_min_chunk_size']}-{tot_config['streaming_buffer_trimming_sec']}-{tot_config['streaming_wait']}-{tot_config['streaming_chunk']}"
            name += "_latency" if tot_config["compute_latency"] else ""
        name = name.replace("/", "-")
        name += "_rtf" if tot_config["compute_rtf"] else ""
        return name


class LintoSttNemoModel(LintoSttWhisperModel):

    def load(self) -> None:
        p = subprocess.Popen(
            ["docker", "ps", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = p.communicate()
        if b"bench_container" in out:
            subprocess.run(
                ["docker", "stop", "bench_container"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
        out = open("docker.log", "w")
        cache_folder = self.config.get("cache_folder", Path.home() / ".cache")
        build_args = f"--env SERVICE_MODE={'http' if not self.config['streaming'] else 'websocket'} --env VAD={self.config['vad']} --env DEVICE={self.config['device']} --env ARCHITECTURE={self.config['architecture']} \
--env MODEL={self.config['model']} --env CONCURRENCY=0"
        build_args += f" -v {str(cache_folder)}:/home/appuser/.cache --env USER_ID={subprocess.check_output(['id', '-u']).decode().strip()} --env GROUP_ID={subprocess.check_output(['id', '-g']).decode().strip()}"
        if self.config["streaming"]:
            build_args += f" --env STREAMING_MIN_CHUNK_SIZE={self.config['streaming_min_chunk_size']} --env STREAMING_BUFFER_TRIMMING_SEC={self.config['streaming_buffer_trimming_sec']}"
        if self.config["device"] != "cpu":
            build_args += f" --gpus all"
        cmd = f"docker run --rm -p {self.config.get('port', 8080)}:80 --name bench_container {build_args} {self.config['docker_image']}"
        out.write(cmd + "\n")
        out.flush()
        p = subprocess.Popen(cmd.split(), stdout=out, stderr=out)
        total_wait_time = 800
        retry_interval = 2
        elapsed_time = 0
        time.sleep(0.5)
        self.model = f"{self.config['server']}:{self.config['port']}"
        while elapsed_time < total_wait_time:
            if healthcheck(self.model, self.config["streaming"]):
                return
            if p.poll() is not None:
                raise RuntimeError(
                    f"The server container has stopped for an unexpected reason. {p}"
                )
            time.sleep(retry_interval)
            print(f"Waiting for server to start... ({elapsed_time}/{total_wait_time} seconds elapsed)\r", end="")
            elapsed_time += retry_interval
        print()
        raise RuntimeError(f"Server did not start in {total_wait_time} seconds")

    def add_defaults_to_config(self, config):
        config["server"] = "localhost"
        config["port"] = 8080
        config["vad"] = config.get("vad", "false")
        config["device"] = config.get("device", "cuda")
        config["decoder"] = config.get("decoder", "ctc")
        decoder_to_architecture = {
            "ctc": "ctc_bpe",
            "rnnt": "rnnt_bpe",
            "hybrid": "hybrid_bpe",
        }
        config["architecture"] = config.get(
            "architecture",
            decoder_to_architecture.get(config["decoder"], config["decoder"]),
        )
        config["streaming"] = config.get("streaming", False)
        config["docker_image"] = config.get("docker_image", "lintoai/linto-stt-nemo")
        if config["streaming"]:
            config["streaming_min_chunk_size"] = config.get(
                "streaming_min_chunk_size", 0.5
            )
            config["streaming_buffer_trimming_sec"] = config.get(
                "streaming_buffer_trimming_sec", 10
            )
            config["streaming_wait"] = config.get("streaming_wait", 0.5)
            config["streaming_chunk"] = config.get("streaming_chunk", 0.5)
        return Model.add_defaults_to_config(self, config)

    def get_folder_name(self):
        tot_config = self.config.copy()
        name = f"linto-stt-nemo_{tot_config['model']}_decoder-{tot_config['decoder']}"
        name += f"_vad-{tot_config['vad']}_device-{tot_config['device']}_{self.config['docker_image'].replace(':', '-')}"
        if tot_config["streaming"]:
            name += f"_streaming-{tot_config['streaming_min_chunk_size']}-{tot_config['streaming_buffer_trimming_sec']}-{tot_config['streaming_wait']}-{tot_config['streaming_chunk']}"
            name += "_latency" if tot_config["compute_latency"] else ""
        name = name.replace("/", "-")
        name += "_rtf" if tot_config["compute_rtf"] else ""
        return name
