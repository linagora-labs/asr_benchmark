import logging
from datetime import datetime
from pathlib import Path

# Every log written during a benchmark run (python logs, docker/vLLM server output)
# goes here, relative to the directory the benchmark is launched from.
LOG_DIR = Path("logs")

logger = logging.getLogger(
    __name__,
)


def log_path(name):
    """Path of a log file named `name` in LOG_DIR (created if needed)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / name.replace("/", "-")


def setup_logging(log_file=None, name="benchmark"):
    """Send the python logs to `log_file` (appended), or by default to a new
    LOG_DIR/<name>_<date>.log. Returns the log file path."""
    if log_file is None:
        log_file = log_path(f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filemode="a",
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        # Some dependencies (ssak, ...) configure the root logger on import.
        force=True,
    )
    return log_file
