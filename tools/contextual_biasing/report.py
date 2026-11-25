"""Table of the contextual biasing benchmark: for each dataset, list variant and
model, the measures without the list (baseline run) and with it, and the gain.

The run of a model with a list lands in "<folder of the run without list>_cb-<list>"
(plus backend settings, see Model.get_full_folder_name): both are paired here. Models
without a biasing mechanism only have the baseline row (their scores on the lists).

    python tools/contextual_biasing/report.py benchmarks/contextual_biasing/results
    python tools/contextual_biasing/report.py RESULTS --mode cased --csv table.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from asr_benchmark.utils import biasing as contextual_biasing  # noqa: E402

MEASURES = ["f1", "precision", "recall", "b_wer", "u_wer", "false_alarms_per_hour"]


_WORD = re.compile(r"\w+")


def runaway_rate(predictions):
    """Percentage of the utterances whose transcription runs away: more than twice as
    many words as the reference, plus 10 (loops, invented sentences, recited prompt...).
    `predictions`: {id: {"text": reference, "prediction": transcription}}."""
    rows = [p for p in predictions.values() if p.get("text")]
    if not rows:
        return None
    runaway = sum(len(_WORD.findall(p.get("prediction") or "")) > 2 * len(_WORD.findall(p["text"])) + 10 for p in rows)
    return 100 * runaway / len(rows)


def rtfx(name, metadata, dataset, predictions, log_timings):
    """(RTFx, approximate) of a run on a dataset: audio duration / processing time, from
    (1) the transcription time recorded by the benchmark (metadata.json), (2) the
    throughput measured by the vllm backend (predictions), or (3) the benchmark logs
    (timings_from_logs.json, see timings_from_logs.py: time of the whole run, model
    loading and scoring included, hence approximate). (None, None) if unknown."""
    timing = (metadata.get("transcription_time") or {}).get(dataset)
    if timing and timing.get("seconds"):
        return timing["audio_seconds"] / timing["seconds"], False
    first = next(iter(predictions.get(dataset, {}).values()), {})
    if first.get("throughput_rtfx"):
        return float(first["throughput_rtfx"]), False
    entry = log_timings.get(name)
    if entry and dataset in entry["datasets"] and all(d in predictions for d in entry["datasets"]):
        audio = sum(float(p["audio_duration"]) for d in entry["datasets"] for p in predictions[d].values())
        return audio / entry["seconds"], True
    return None, None


STATS_FILE = "run_stats.json"


def load_runs(results):
    """{folder name: (metadata, {dataset: performances})}, with the runaway rate of each
    dataset ("runaway", from its predictions) and its throughput ("rtfx", "rtfx_approx")"""
    runs = {}
    log_file = Path(results) / "timings_from_logs.json"
    log_timings = json.loads(log_file.read_text(encoding="utf-8")) if log_file.exists() else {}
    for exp in sorted(p for p in Path(results).iterdir() if (p / "metadata.json").exists()):
        metadata = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
        predictions = {f.stem: json.loads(f.read_text(encoding="utf-8"))
                       for f in sorted((exp / "predictions").glob("*.json"))}
        saved = json.loads((exp / STATS_FILE).read_text(encoding="utf-8")) if (exp / STATS_FILE).exists() else {}
        perfs = {}
        for f in sorted((exp / "performances").glob("*.json")):
            perfs[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            if f.stem in predictions:
                perfs[f.stem]["runaway"] = runaway_rate(predictions[f.stem])
            perfs[f.stem]["rtfx"], perfs[f.stem]["rtfx_approx"] = rtfx(exp.name, metadata, f.stem, predictions, log_timings)
            for key, value in saved.get(f.stem, {}).items():
                if perfs[f.stem].get(key) is None:
                    perfs[f.stem][key] = value
        runs[exp.name] = (metadata, perfs)
    return runs


def save_stats(results):
    """What load_runs computes from the predictions of each run (runaway rate, throughput)
    in <run>/run_stats.json, read when the predictions are missing (those made with lists
    that cannot be redistributed are not in git)."""
    for name, (_, perfs) in load_runs(results).items():
        stats = {dataset: {key: perf.get(key) for key in ("runaway", "rtfx", "rtfx_approx")} for dataset, perf in perfs.items()}
        (Path(results) / name / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")


# Key of the WER in each mode: always without punctuation
WER_KEYS = {"nocasepunc": "wer_nocasepunc", "cased": "wer_nopunc"}


def measures(perf, scores, mode):
    """WER (and the upper bound of its 95 % confidence interval), runaway rate and
    throughput of the run, and biasing measures of one list."""
    # No WER with case when the references have no case
    wer = perf.get(WER_KEYS[mode]) or perf.get("wer_nocasepunc", {})
    return {"wer": wer.get("wer"), "wer_ci95_high": (wer.get("ci95") or [None, None])[1],
            "runaway": perf.get("runaway"), "rtfx": perf.get("rtfx"), "rtfx_approx": perf.get("rtfx_approx"),
            **{m: scores[m] for m in MEASURES}}


def collect(runs, mode):
    """Rows: one per (dataset, list, model), with the measures of the run without
    list and, when there is one, of the run with the list ("<key>_cb")."""
    rows = {}  # (baseline folder, dataset, list) -> row
    biased = []
    for name, (metadata, perfs) in runs.items():
        given = contextual_biasing.config_variant(metadata) is not None
        for dataset, perf in perfs.items():
            for list_name, scores in (perf.get(contextual_biasing.MANIFEST_KEY) or {}).items():
                if given:
                    biased.append((name, dataset, list_name, measures(perf, scores[mode], mode)))
                else:
                    rows[(name, dataset, list_name)] = {
                        "dataset": dataset, "list": list_name, "model": name,
                        "list_size": scores[mode]["list_size_mean"], **measures(perf, scores[mode], mode),
                    }
    table = list(rows.values())
    # A run with a list is in "<baseline folder>_cb-<list>[-<backend settings>]"
    for name, dataset, list_name, values in biased:
        base = name.rpartition("_cb-")[0]
        row = rows.get((base, dataset, list_name))
        if row is None or "run_cb" in row:
            # No baseline, or a second run with the same list (e.g. other settings): own
            # row, with the baseline measures repeated
            row = {k: v for k, v in (row or {"dataset": dataset, "list": list_name, "model": base}).items()
                   if k != "run_cb" and not k.endswith("_cb")}
            table.append(row)
        row["run_cb"] = name
        row.update({f"{key}_cb": value for key, value in values.items()})
    return sorted(table, key=lambda r: (r["dataset"], str(r["list"]), r["model"], r.get("run_cb", "")))


def fmt(value, digits=1):
    return "" if value is None else f"{value:.{digits}f}"


def delta(row, key):
    a, b = row.get(key), row.get(f"{key}_cb")
    return None if a is None or b is None else b - a


# Columns of the markdown table: (label, key), each shown without and with the list
COLUMNS = [("F1", "f1"), ("P", "precision"), ("R", "recall"), ("B-WER", "b_wer"),
           ("U-WER", "u_wer"), ("FA/h", "false_alarms_per_hour"), ("WER", "wer"), ("Runaway", "runaway"),
           ("RTFx", "rtfx")]


def print_markdown(table, mode):
    print(f"Contextual biasing measures ({mode}). Without list -> with list (delta).\n")
    groups = {}
    for row in table:
        groups.setdefault((row["dataset"], row["list"]), []).append(row)
    for (dataset, list_name), rows in groups.items():
        print(f"### {dataset}, list {list_name} (mean size {rows[0].get('list_size')})\n")
        header = ["model"] + [h for label, _ in COLUMNS for h in (label, f"{label} cb")]
        header.insert(3, "dF1")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for r in rows:
            cells = [fmt(r.get(k)) for _, key in COLUMNS for k in (key, f"{key}_cb")]
            cells.insert(2, fmt(delta(r, "f1")))
            model = r["model"] + (f" ({r['run_cb'].removeprefix(r['model'])})" if r.get("run_cb") else "")
            print("| " + " | ".join([model] + cells) + " |")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="Results folder of the benchmark (output_folder)")
    parser.add_argument("--mode", default="nocasepunc", choices=["nocasepunc", "cased"],
                        help="Text normalization of the measures")
    parser.add_argument("--csv", default=None, help="Also write the table as CSV")
    parser.add_argument("--save_stats", action="store_true",
                        help="Only write the runaway rate and the throughput of each run, computed from its "
                             "predictions, in <run>/run_stats.json (read when the predictions are missing)")
    args = parser.parse_args()

    if args.save_stats:
        save_stats(args.results)
        return

    table = collect(load_runs(args.results), args.mode)
    if not table:
        print(f"No contextual biasing measures found in {args.results}")
        return
    print_markdown(table, args.mode)
    if args.csv:
        keys = ["dataset", "list", "model", "run_cb", "list_size", "wer", "wer_cb", "runaway", "runaway_cb",
                "rtfx", "rtfx_cb", "rtfx_approx", "rtfx_approx_cb"]
        keys += [k for m in MEASURES for k in (m, f"{m}_cb")]
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in table:
                writer.writerow(row)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
