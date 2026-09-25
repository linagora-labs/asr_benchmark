"""Build the static leaderboard page from benchmark results.

Reads every <results>/<experiment>/{metadata.json, performances/*.json} and writes
<output>/index.html (the page, data embedded) and <output>/leaderboard.json.
Standard library only, so the GitHub Action needs no install.

    python tools/leaderboard/build.py                       # benchmarks/sota/results -> site/
    python tools/leaderboard/build.py --results X --output Y
"""
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
REPO_URL = "https://github.com/linagora-labs/asr_benchmark"

# Display name, domain and language of the known test sets (file stem of
# performances/*.json). Unknown datasets fall back to their stem, in DEFAULT_LANGUAGE.
DATASETS = {
    "CommonVoice_max30": ("Common Voice", "Read speech, crowd-sourced", "fr"),
    "MLS_Facebook_french_max30": ("MLS", "Read audiobooks", "fr"),
    "SUMM-RE_max30": ("SUMM-RE", "Spontaneous meetings", "fr"),
    "TEDX_fr_max30": ("TEDx", "Prepared talks", "fr"),
    "Voxpopuli_max30": ("VoxPopuli", "Parliament speeches", "fr"),
    "YouTubeFr_max30_split6": ("YouTube", "Web videos", "fr"),
}
DEFAULT_LANGUAGE = "fr"
METRICS = ["wer_nocasepunc", "wer", "cer_nocasepunc", "cer"]


def dataset_hours(experiments, stem):
    """Duration of a test set, from the first predictions file that has it."""
    for exp in experiments:
        pred = exp / "predictions" / f"{stem}.json"
        if pred.exists():
            rows = json.loads(pred.read_text(encoding="utf-8")).values()
            return round(sum(r.get("audio_duration") or 0 for r in rows) / 3600, 2)
    return None


def config_model_ids(bench_dir):
    """Model ids as written in the benchmark configs, keyed by their "_" -> "-" form:
    some backends (NeMo) store the model id that way in metadata.json, which breaks
    Hugging Face links (nvidia/stt_fr_fastconformer_hybrid_large_pc)."""
    ids = {}
    for config in bench_dir.glob("config*.yaml"):
        # No yaml parser in the standard library: model ids are the only "org/name" values.
        for model_id in re.findall(r"[\w.-]+/[\w.-]+", config.read_text(encoding="utf-8")):
            ids.setdefault(model_id.replace("_", "-"), model_id)
    return ids


def model_entry(exp, model_ids):
    meta = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
    backend, model = meta["backend"], meta["model"]
    display = f"whisper-{model}" if backend == "faster-whisper" and "whisper" not in model else model
    display = model_ids.get(display, display)
    # What the folder name adds after <backend>_<model> (e.g. "decoder-ctc") tells
    # apart several runs of the same model.
    prefix = f"{backend}_{model.replace('/', '-')}"
    variant = exp.name[len(prefix):].strip("_") if exp.name.startswith(prefix) else ""
    if "/" in display and not display.startswith("/"):
        url = f"https://huggingface.co/{display}"
    elif backend == "faster-whisper":
        url = f"https://huggingface.co/openai/{display}"
    else:
        url = None
    scores = {}
    for perf_file in sorted((exp / "performances").glob("*.json")):
        perf = json.loads(perf_file.read_text(encoding="utf-8"))
        scores[perf_file.stem] = {
            metric: {k: round(perf[metric][k], 3) for k in ("wer", "sub", "del", "ins") if k in perf[metric]}
            | ({"ci95": perf[metric]["ci95"]} if "ci95" in perf[metric] else {})
            for metric in METRICS
            if metric in perf
        }
        scores[perf_file.stem]["n"] = perf.get("num_data")
    return {
        "id": exp.name,
        "model": display,
        "variant": variant,
        "backend": backend,
        "url": url,
        "scores": scores,
    }


def last_commit_date(path):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build(results, output):
    experiments = sorted(p for p in results.iterdir() if (p / "metadata.json").exists())
    model_ids = config_model_ids(results.parent)
    models = [model_entry(exp, model_ids) for exp in experiments]
    stems = sorted({stem for m in models for stem in m["scores"]}, key=lambda s: DATASETS.get(s, (s,))[0].lower())
    datasets = [
        {
            "id": stem,
            "name": DATASETS.get(stem, (stem,))[0],
            "domain": DATASETS.get(stem, (stem, ""))[1],
            "language": DATASETS.get(stem, (stem, "", DEFAULT_LANGUAGE))[2],
            "hours": dataset_hours(experiments, stem),
            "utterances": max((m["scores"][stem]["n"] or 0) for m in models if stem in m["scores"]),
        }
        for stem in stems
    ]
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_updated": last_commit_date(results),
        "repo": REPO_URL,
        "datasets": datasets,
        "models": models,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "leaderboard.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    template = (HERE / "template.html").read_text(encoding="utf-8")
    # "</" is escaped so the data can never close the <script> tag.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (output / "index.html").write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print(f"{len(models)} models x {len(datasets)} datasets -> {output}/index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/sota/results"))
    parser.add_argument("--output", type=Path, default=Path("site"))
    args = parser.parse_args()
    build(args.results, args.output)
