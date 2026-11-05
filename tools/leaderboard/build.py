"""Build the static leaderboard page from benchmark results.

Writes <output>/index.html and <output>/leaderboard.json. Standard library only (GitHub Action).

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

# Display name, domain, language, source; column order of the page
DATASETS = {
    "FLEURS": ("FLEURS", "Read Wikipedia sentences", "fr", "https://huggingface.co/datasets/google/fleurs"),
    "CommonVoice": ("Common Voice 22.0", "Read speech", "fr", "https://commonvoice.mozilla.org/en/datasets"),
    "MLS_Facebook_french": ("MLS", "Read audiobooks", "fr", "https://openslr.org/94/"),
    "TEDX_fr": ("TEDx", "Prepared talks", "fr", "https://www.openslr.org/100/"),
    "Voxpopuli": ("VoxPopuli", "Parliament speeches", "fr", "https://huggingface.co/datasets/facebook/voxpopuli"),
    "SUMM-RE": ("SUMM-RE", "Spontaneous meetings", "fr", "https://huggingface.co/datasets/linagora/SUMM-RE"),
    "YouTubeFr_split6": ("YouTube", "Web videos", "fr", None),  # internal corpus
}
# Share kept (`datasets` of config.yaml), details on hover
SELECTION = {
    "CommonVoice": ("20%", None),
    "MLS_Facebook_french": ("50%", None),
    "TEDX_fr": ("Whole", None),
    "Voxpopuli": ("50%", None),
    "SUMM-RE": ("50%", "Segments of 4 s or more only"),
    "YouTubeFr_split6": ("Whole", None),
    "FLEURS": ("Whole", None),
}
DEFAULT_LANGUAGE = "fr"
METRICS = ["wer_nocasepunc", "wer", "cer_nocasepunc", "cer"]
EMPTY_WARNING = 0.01  # share of empty outputs from which the model gets a caveat
SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")


def dataset_info(stem):
    # Speed runs keep the old names
    name, domain, language, url = DATASETS.get(stem.replace("_max30", ""), (stem, "", DEFAULT_LANGUAGE, None))
    return {"name": name, "domain": domain, "language": language, "url": url,
            "selection": SELECTION.get(stem, (None, None))[0],
            "selection_note": SELECTION.get(stem, (None, None))[1]}


def speed(exp, rtf_results, model=None):
    """RTFx (seconds of audio transcribed per second of computation), hardware, backend and
    caveat (`rtf_note` of its model_info, filled by hand) of the speed run of `exp`: <rtf_results>/<exp>_rtf, from benchmarks/sota/config_rtf.yaml.
    Without it, the speed run of the same `model` with another backend: config_rtf.yaml times
    the audio LLMs through vLLM (faster) while their WER may come from transformers."""
    if not rtf_results or not rtf_results.exists():
        return None, None, None, None
    exp_rtf = rtf_results / f"{exp.name}_rtf"
    if not (exp_rtf / "metadata.json").exists():
        same_model = [
            p.parent for p in sorted(rtf_results.glob("*/metadata.json"))
            if model and json.loads(p.read_text(encoding="utf-8")).get("model") == model
        ]
        if len(same_model) != 1:  # none, or ambiguous (several variants of the model)
            return None, None, None, None
        exp_rtf = same_model[0]
    audio = compute = 0
    for pred_file in (exp_rtf / "predictions").glob("*.json"):
        for row in json.loads(pred_file.read_text(encoding="utf-8")).values():
            if row.get("prediction_duration"):
                audio += row["audio_duration"]
                compute += row["prediction_duration"]
    meta = json.loads((exp_rtf / "metadata.json").read_text(encoding="utf-8"))
    return ((round(audio / compute, 1) if compute else None), meta.get("device_name") or meta.get("device"),
            meta.get("backend"), (meta.get("model_info") or {}).get("rtf_note"))
HISTOGRAM_BIN = 2  # seconds, durations histogram of the test sets


def count_empty(pred_file):
    """Outputs with no letter or digit left once the special tokens (<|...|>) are removed."""
    rows = json.loads(pred_file.read_text(encoding="utf-8")).values()
    return sum(not any(c.isalnum() for c in SPECIAL_TOKEN.sub("", row.get("prediction") or "")) for row in rows)


def empty_note(scores):
    """Caveat on the WER of a model with more than EMPTY_WARNING of empty outputs."""
    total = sum(s.get("n") or 0 for s in scores.values() if "empty" in s)
    empty = sum(s.get("empty", 0) for s in scores.values())
    if not total or empty / total <= EMPTY_WARNING:
        return None
    return f"Empty outputs: {empty} / {total} ({100 * empty / total:.1f}%)."


def model_entry(exp, rtf_results=None):
    meta = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
    backend, model = meta["backend"], meta["model"]
    # faster-whisper takes OpenAI size names ("large-v3"): show the original model.
    display = f"openai/whisper-{model}" if backend == "faster-whisper" and "/" not in model else model
    # vLLM serves local checkpoints (model is a path) under served_model_name: show it,
    # or the Hugging Face repo of the checkpoint (model_info.hf) when there is one.
    info = meta.get("model_info") or {}
    name = meta.get("served_model_name") and (info.get("hf") or meta["served_model_name"]) or display
    # What the folder name adds after <backend>_<model> (e.g. "decoder-ctc") tells
    # apart several runs of the same model. Some backends (NeMo) write "_" as "-" there.
    variant = ""
    folder_model = meta.get("served_model_name") or model
    for prefix in {f"{backend}_{folder_model.replace('/', '-')}", f"{backend}_{re.sub('[/_]', '-', folder_model)}"}:
        if exp.name.startswith(prefix):
            variant = exp.name[len(prefix):].strip("_")
    # Streaming models run with their default chunk size have no chunk in the folder name.
    if meta.get("chunk_ms") and f"chunk{meta['chunk_ms']}ms" not in variant:
        variant = "_".join(filter(None, [variant, f"chunk{meta['chunk_ms']}ms"]))
    # License, size and languages: filled by tools/fill_model_info.py.
    hf = info.get("hf") or display
    url = f"https://huggingface.co/{hf}" if "/" in hf and not hf.startswith(("/", ".")) else None
    rtfx, hardware, rtf_backend, rtf_note = speed(exp, rtf_results, model)
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
        pred_file = exp / "predictions" / perf_file.name
        if pred_file.exists():
            scores[perf_file.stem]["empty"] = count_empty(pred_file)
    return {
        "id": exp.name,
        "model": name,
        "variant": variant,
        "backend": backend,
        "url": url,
        "license": info.get("license"),
        "params": info.get("params"),
        "languages": info.get("languages"),
        "architecture": info.get("architecture"),  # Encoder | Encoder-decoder | Audio LLM
        "streaming": info.get("streaming"),
        "task": info.get("task"),  # ASR | ASR/AST | Multi
        "rtfx": rtfx,
        "hardware": hardware,
        "rtf_backend": rtf_backend if rtf_backend != backend else None,  # set when timed with another backend
        "rtf_note": rtf_note,  # caveat on the speed measure, shown with a warning sign
        "wer_note": "\n".join(n.rstrip(".") + "." for n in (info.get("wer_note"), empty_note(scores)) if n) or None,
        "prompt": (meta.get("prompt") if backend != "vllm" or meta.get("endpoint") == "chat" else None) or info.get("prompt"),
        "prompt_source": info.get("prompt_source"),
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
        {k: v for k, v in dataset_info(stem).items() if k not in ("language", "selection", "selection_note")} | {"durations": d}
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
        "sources": [{"name": r["name"], "domain": r["domain"], "url": r["url"]} | stats(r["durations"]) for r in rows],
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
    order = list(DATASETS)
    stems = sorted({stem for m in models for stem in m["scores"]},
                   key=lambda s: (order.index(s), "") if s in DATASETS else (len(order), s.lower()))
    stats = lambda d: {
        "hours": round(sum(d) / 3600, 2),
        "utterances": len(d),
        "mean_seconds": round(statistics.mean(d), 1),
        "median_seconds": round(statistics.median(d), 1),
        "histogram": histogram(d),
    }
    datasets = [
        {
            "id": stem,
            **dataset_info(stem),
            "hours": round(max((m["scores"][stem]["seconds"] or 0) for m in models if stem in m["scores"]) / 3600, 2) or None,
            "utterances": max((m["scores"][stem]["n"] or 0) for m in models if stem in m["scores"]),
        }
        | (stats(durations[stem]) if durations.get(stem) else {})
        for stem in stems
    ]
    totals = {}
    for language in {d["language"] for d in datasets}:
        rows = [d for d in datasets if d["language"] == language]
        totals[language] = (
            stats([x for d in rows for x in durations[d["id"]]]) if all(durations.get(d["id"]) for d in rows)
            else {"hours": round(sum(d["hours"] or 0 for d in rows), 2), "utterances": sum(d["utterances"] for d in rows)}
        )
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_updated": last_commit_date(results),
        "repo": REPO_URL,
        "histogram_bin": HISTOGRAM_BIN,
        "datasets": datasets,
        "dataset_totals": totals,
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
