import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from asr_benchmark.utils.benchmark import load_audio
from asr_benchmark.benchmark.interfaces import Model

logger = logging.getLogger(__name__)

# Strip bracketed timestamps ([0.48]) / speaker labels ([S01]) and <think> blocks
# from diarization-style output (e.g. MOSS). Harmless for plain-text models.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class VllmTranscriptionModel(Model):
    """Backend for any model served by vLLM's OpenAI-compatible
    /v1/audio/transcriptions endpoint (Voxtral, MOSS, Whisper, ...).

    The vLLM engine runs as a separate server process, so this backend itself only
    needs `requests`. By default load() launches `vllm serve <model>` and cleanup()
    stops it; set launch=false to connect to a server you started yourself.
    """

    def load(self) -> None:
        self.server_url = self._base_url()
        self._proc = None
        self._log = None
        if self.config["launch"]:
            self._launch_server()
        elif not self._healthcheck():
            raise ValueError(
                f"No vLLM server reachable at {self.server_url} (launch=false). "
                f"Start one with: vllm serve {self.config['model']} --port {self.config['port']}"
            )

    def _base_url(self) -> str:
        server = self.config["server"]
        if not server.startswith("http"):
            server = f"http://{server}"
        return f"{server}:{self.config['port']}"

    def _healthcheck(self) -> bool:
        try:
            return requests.get(f"{self.server_url}/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def _launch_server(self) -> None:
        cmd = list(self.config["vllm_command"]) + [self.config["model"], "--port", str(self.config["port"])]
        if self.config["trust_remote_code"]:
            cmd.append("--trust-remote-code")
        if self.config["served_model_name"]:
            cmd += ["--served-model-name", self.config["served_model_name"]]
        cmd += list(self.config["extra_args"])

        self._log = open(self.config["log_file"], "w")
        self._log.write(" ".join(str(c) for c in cmd) + "\n")
        self._log.flush()
        logger.info(f"Launching vLLM server (logs -> {self.config['log_file']})")
        # start_new_session -> own process group, so cleanup() can kill the whole
        # tree (vLLM spawns worker processes that would otherwise hold the GPU/port).
        self._proc = subprocess.Popen(cmd, stdout=self._log, stderr=self._log, start_new_session=True)
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        total, interval, elapsed = self.config["startup_timeout"], 3, 0
        time.sleep(1)
        while elapsed < total:
            if self._healthcheck():
                logger.info(f"vLLM server ready at {self.server_url}")
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited early (code {self._proc.returncode}); see {self.config['log_file']}"
                )
            time.sleep(interval)
            elapsed += interval
            print(f"Waiting for vLLM server... ({elapsed}/{total}s)\r", end="")
        print()
        raise RuntimeError(f"vLLM server not ready after {total}s; see {self.config['log_file']}")

    def load_audio(self, audio: str, start=0.0, duration=None):
        return load_audio(audio, return_format="file", start=start, duration=duration)

    def _post(self, audio_path: str) -> str:
        """POST one audio file to /v1/audio/transcriptions and return the transcript."""
        data = {
            "model": self.config["served_model_name"] or self.config["model"],
            "response_format": "json",
        }
        if self.config["language"]:
            data["language"] = self.config["language"]
        if self.config["temperature"] is not None:
            data["temperature"] = self.config["temperature"]
        with open(audio_path, "rb") as f:
            res = requests.post(
                f"{self.server_url}/v1/audio/transcriptions",
                data=data,
                files={"file": (Path(audio_path).name, f, "audio/wav")},
                timeout=self.config["request_timeout"],
            )
        if res.status_code != 200:
            raise RuntimeError(f"vLLM transcription failed (HTTP {res.status_code}): {res.text[:500]}")
        parsed = res.json()
        text = parsed["text"] if isinstance(parsed, dict) and "text" in parsed else str(parsed)
        if self.config["strip_diarization"]:
            text = self._strip_annotations(text)
        return text.strip()

    def transcribe(self, audio: str) -> dict:
        # Serial per-file path (used when compute_rtf is True): latency-based RTF, with
        # hardware monitoring active in the harness.
        try:
            text = self._post(audio)
        finally:
            Path(audio).unlink(missing_ok=True)
        return {"text": text}

    def transcribe_batch(self, data: list) -> list:
        """Concurrent throughput path (used when compute_rtf is False).

        Sends up to `concurrency` requests in flight at once -- the fair way to measure
        a batching server like vLLM. Each record keeps its real individual latency, and
        every record also carries `throughput_rtfx` = total_audio / wall_time, the
        aggregate real-time factor that reflects vLLM's batched throughput.
        """
        import soundfile as sf

        concurrency = self.config["concurrency"]
        tmpdir = tempfile.mkdtemp(prefix="vllm_bench_")
        try:
            # Materialize each (offset/duration-sliced) segment to its own temp wav so
            # concurrent requests never collide on the shared tmp.wav that load_audio uses.
            items = []
            for i, row in enumerate(data):
                arr = load_audio(
                    row["audio_filepath"], return_format="librosa",
                    start=row.get("offset", 0.0), duration=row.get("duration"),
                )
                path = os.path.join(tmpdir, f"{i}.wav")
                sf.write(path, arr, 16000)
                duration = row.get("duration") or (len(arr) / 16000.0)
                items.append((i, path, duration))

            results = [None] * len(items)

            def work(item):
                i, path, duration = item
                t0 = time.time()
                text = self._post(path)
                latency = time.time() - t0
                return i, {
                    "text": text,
                    "prediction_duration": round(latency, 5),
                    "rtf": round(latency / duration, 5) if duration else None,
                }

            wall_start = time.time()
            with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
                for i, out in pool.map(work, items):
                    results[i] = out
            wall = time.time() - wall_start

            total_audio = sum(d for _, _, d in items)
            throughput = round(total_audio / wall, 3) if wall > 0 else None
            logger.info(
                f"vLLM concurrent throughput: {throughput}x realtime "
                f"(concurrency={concurrency}, {len(items)} files, {total_audio:.0f}s audio in {wall:.1f}s)"
            )
            for out in results:
                out["throughput_rtfx"] = throughput
            return results
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _strip_annotations(text: str) -> str:
        text = _THINK_RE.sub(" ", text)
        text = _BRACKET_RE.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    def cleanup(self):
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if getattr(self, "_log", None) is not None:
            self._log.close()

    def add_defaults_to_config(self, config):
        config["server"] = config.get("server", "http://localhost")
        config["port"] = int(config.get("port", 8000))
        config["launch"] = config.get("launch", True)
        vllm_command = config.get("vllm_command", ["vllm", "serve"])
        config["vllm_command"] = vllm_command.split() if isinstance(vllm_command, str) else list(vllm_command)
        config["trust_remote_code"] = config.get("trust_remote_code", False)
        config["served_model_name"] = config.get("served_model_name", None)
        extra_args = config.get("extra_args", [])
        config["extra_args"] = extra_args.split() if isinstance(extra_args, str) else list(extra_args)
        config["language"] = config.get("language", None)
        config["temperature"] = config.get("temperature", 0.0)
        # Concurrent in-flight requests for the throughput path (transcribe_batch, used
        # when compute_rtf is False). 1 = serial. Ignored on the compute_rtf latency path.
        config["concurrency"] = int(config.get("concurrency", 1))
        # Strip [timestamp]/[speaker] tags for WER (needed for MOSS; a no-op otherwise).
        config["strip_diarization"] = config.get("strip_diarization", True)
        config["startup_timeout"] = int(config.get("startup_timeout", 1200))
        config["request_timeout"] = int(config.get("request_timeout", 600))
        config["log_file"] = config.get("log_file", "vllm.log")
        return super().add_defaults_to_config(config)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["model"] = self.config["model"].replace("_", "-")
        return metadata

    def get_folder_name(self):
        c = self.config
        name = f"vllm_{c['model'].replace('/', '-')}"
        if c["language"]:
            name += f"_lang-{c['language']}"
        if not c.get("compute_rtf") and c["concurrency"] > 1:
            name += f"_conc{c['concurrency']}"
        name = name.replace("/", "-")
        name += "_rtf" if c.get("compute_rtf") else ""
        return name
