"""Build the static leaderboard page from benchmark results.

Reads every <results>/<experiment>/{metadata.json (with its model_info: license, size,
languages, see tools/fill_model_info.py), performances/*.json}, the test
manifest (utterance durations) and the speed runs (<rtf_results>/<experiment>_rtf, for
the RTFx, and their manifest, described in the test sets tab), and writes <output>/index.html (the page, data
embedded) and <output>/leaderboard.json.
Standard library only, so the GitHub Action needs no install.

    python tools/leaderboard/build.py                       # benchmarks/sota/results -> site/
    python tools/leaderboard/build.py --results X --manifest M --output Y
"""
import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
REPO_URL = "https://github.com/linagora-labs/asr_benchmark"

# Display name, domain and language of the known test sets (file stem of
# performances/*.json). Unknown datasets fall back to their stem, in DEFAULT_LANGUAGE.
DATASETS = {
    "CommonVoice_max30": ("Common Voice", "Read speech", "fr"),
    "MLS_Facebook_french_max30": ("MLS", "Read audiobooks", "fr"),
    "SUMM-RE_max30": ("SUMM-RE", "Spontaneous meetings", "fr"),
    "TEDX_fr_max30": ("TEDx", "Prepared talks", "fr"),
    "Voxpopuli_max30": ("VoxPopuli", "Parliament speeches", "fr"),
    "YouTubeFr_max30_split6": ("YouTube", "Web videos", "fr"),
}
DEFAULT_LANGUAGE = "fr"
METRICS = ["wer_nocasepunc", "wer", "cer_nocasepunc", "cer"]

def speed(exp, rtf_results, model=None):
    """RTFx (seconds of audio transcribed per second of computation), hardware and backend of
    the speed run of `exp`: <rtf_results>/<exp>_rtf, from benchmarks/sota/config_rtf.yaml.
    Without it, the speed run of the same `model` with another backend: config_rtf.yaml times
    the audio LLMs through vLLM (faster) while their WER may come from transformers."""
    if not rtf_results or not rtf_results.exists():
        return None, None, None
    exp_rtf = rtf_results / f"{exp.name}_rtf"
    if not (exp_rtf / "metadata.json").exists():
        same_model = [
            p.parent for p in sorted(rtf_results.glob("*/metadata.json"))
            if model and json.loads(p.read_text(encoding="utf-8")).get("model") == model
        ]
        if len(same_model) != 1:  # none, or ambiguous (several variants of the model)
            return None, None, None
        exp_rtf = same_model[0]
    audio = compute = 0
    for pred_file in (exp_rtf / "predictions").glob("*.json"):
        for row in json.loads(pred_file.read_text(encoding="utf-8")).values():
            if row.get("prediction_duration"):
                audio += row["audio_duration"]
                compute += row["prediction_duration"]
    meta = json.loads((exp_rtf / "metadata.json").read_text(encoding="utf-8"))
    return (round(audio / compute, 1) if compute else None), meta.get("device_name") or meta.get("device"), meta.get("backend")
HISTOGRAM_BIN = 2  # seconds, durations histogram of the test sets


def model_entry(exp, rtf_results=None):
    meta = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
    backend, model = meta["backend"], meta["model"]
    # faster-whisper takes OpenAI size names ("large-v3"): show the original model.
    display = f"openai/whisper-{model}" if backend == "faster-whisper" and "/" not in model else model
    # What the folder name adds after <backend>_<model> (e.g. "decoder-ctc") tells
    # apart several runs of the same model. Some backends (NeMo) write "_" as "-" there.
    variant = ""
    for prefix in {f"{backend}_{model.replace('/', '-')}", f"{backend}_{re.sub('[/_]', '-', model)}"}:
        if exp.name.startswith(prefix):
            variant = exp.name[len(prefix):].strip("_")
    # Streaming models run with their default chunk size have no chunk in the folder name.
    if meta.get("chunk_ms") and f"chunk{meta['chunk_ms']}ms" not in variant:
        variant = "_".join(filter(None, [variant, f"chunk{meta['chunk_ms']}ms"]))
    # License, size and languages: filled by tools/fill_model_info.py.
    info = meta.get("model_info") or {}
    hf = info.get("hf") or display
    url = f"https://huggingface.co/{hf}" if "/" in hf and not hf.startswith(("/", ".")) else None
    rtfx, hardware, rtf_backend = speed(exp, rtf_results, model)
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
        scores[perf_file.stem]["seconds"] = perf.get("duration")
    return {
        "id": exp.name,
        "model": display,
        "variant": variant,
        "backend": backend,
        "url": url,
        "license": info.get("license"),
        "params": info.get("params"),
        "languages": info.get("languages"),
        "rtfx": rtfx,
        "hardware": hardware,
        "rtf_backend": rtf_backend if rtf_backend != backend else None,  # set when timed with another backend
        "scores": scores,
    }


def manifest_durations(manifest):
    """Durations (s) of the utterances of each test set of the manifest, {} if there is none."""
    durations = {}
    if manifest and manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                durations.setdefault(row.get("name") or row.get("dataset"), []).append(row["duration"])
    return durations


def speed_set(manifest, rtf_results):
    """Segments the RTFx is measured on (manifest of config_rtf.yaml), per source test set,
    and the hardware of the speed runs. None without the manifest."""
    durations = manifest_durations(manifest)
    if not durations:
        return None
    rows = [
        {"name": DATASETS.get(stem, (stem,))[0], "domain": DATASETS.get(stem, (stem, ""))[1], "durations": d}
        for stem, d in sorted(durations.items(), key=lambda kv: -len(kv[1]))
    ]
    everything = [x for d in durations.values() for x in d]
    hardware = sorted({
        meta.get("device_name") or meta.get("device")
        for meta in (json.loads(p.read_text(encoding="utf-8")) for p in rtf_results.glob("*/metadata.json"))
    } - {None}) if rtf_results and rtf_results.exists() else []
    stats = lambda d: {
        "utterances": len(d),
        "seconds": round(sum(d), 1),
        "min_seconds": round(min(d), 1),
        "max_seconds": round(max(d), 1),
        "mean_seconds": round(statistics.mean(d), 1),
        "median_seconds": round(statistics.median(d), 1),
        "histogram": histogram(d),
    }
    return {
        "sources": [{"name": r["name"], "domain": r["domain"]} | stats(r["durations"]) for r in rows],
        "total": stats(everything),
        "hardware": hardware,
    }


def histogram(durations):
    counts = [0] * (int(max(durations) // HISTOGRAM_BIN) + 1)
    for d in durations:
        counts[int(d // HISTOGRAM_BIN)] += 1
    return counts


def last_commit_date(path):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def build(results, manifest, output, rtf_results=None, rtf_manifest=None):
    experiments = sorted(p for p in results.iterdir() if (p / "metadata.json").exists())
    models = [model_entry(exp, rtf_results) for exp in experiments]
    durations = manifest_durations(manifest)
    stems = sorted({stem for m in models for stem in m["scores"]}, key=lambda s: DATASETS.get(s, (s,))[0].lower())
    datasets = [
        {
            "id": stem,
            "name": DATASETS.get(stem, (stem,))[0],
            "domain": DATASETS.get(stem, (stem, ""))[1],
            "language": DATASETS.get(stem, (stem, "", DEFAULT_LANGUAGE))[2],
            "hours": round(max((m["scores"][stem]["seconds"] or 0) for m in models if stem in m["scores"]) / 3600, 2) or None,
            "utterances": max((m["scores"][stem]["n"] or 0) for m in models if stem in m["scores"]),
        }
        # The manifest, when it has the test set, gives the exact figures and the histogram.
        | ({
            "hours": round(sum(durations[stem]) / 3600, 2),
            "utterances": len(durations[stem]),
            "mean_seconds": round(statistics.mean(durations[stem]), 1),
            "median_seconds": round(statistics.median(durations[stem]), 1),
            "histogram": histogram(durations[stem]),
        } if durations.get(stem) else {})
        for stem in stems
    ]
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_updated": last_commit_date(results),
        "repo": REPO_URL,
        "histogram_bin": HISTOGRAM_BIN,
        "datasets": datasets,
        "speed_set": speed_set(rtf_manifest, rtf_results),
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
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/sota/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("site"))
    parser.add_argument("--rtf_results", type=Path, default=Path("benchmarks/sota/results_rtf"),
                        help="Results of the speed runs (config_rtf.yaml), for the RTFx column")
    parser.add_argument("--rtf_manifest", type=Path, default=Path("benchmarks/sota/manifest_rtf.jsonl"),
                        help="Manifest of the speed runs, described in the test sets tab")
    args = parser.parse_args()
    build(args.results, args.manifest, args.output, args.rtf_results, args.rtf_manifest)
