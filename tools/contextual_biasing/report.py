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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from asr_benchmark.benchmark import contextual_biasing  # noqa: E402

MEASURES = ["f1", "precision", "recall", "b_wer", "u_wer", "false_alarms_per_hour"]


def load_runs(results):
    """{folder name: (metadata, {dataset: performances})}"""
    runs = {}
    for exp in sorted(p for p in Path(results).iterdir() if (p / "metadata.json").exists()):
        metadata = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
        perfs = {
            f.stem: json.loads(f.read_text(encoding="utf-8"))
            for f in sorted((exp / "performances").glob("*.json"))
        }
        runs[exp.name] = (metadata, perfs)
    return runs


def measures(perf, scores):
    """WER of the run and biasing measures of one list."""
    return {"wer": perf.get("wer_nocasepunc", {}).get("wer"), **{m: scores[m] for m in MEASURES}}


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
                    biased.append((name, dataset, list_name, measures(perf, scores[mode])))
                else:
                    rows[(name, dataset, list_name)] = {
                        "dataset": dataset, "list": list_name, "model": name,
                        "list_size": scores[mode]["list_size_mean"], **measures(perf, scores[mode]),
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
           ("U-WER", "u_wer"), ("FA/h", "false_alarms_per_hour"), ("WER", "wer")]


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
    args = parser.parse_args()

    table = collect(load_runs(args.results), args.mode)
    if not table:
        print(f"No contextual biasing measures found in {args.results}")
        return
    print_markdown(table, args.mode)
    if args.csv:
        keys = ["dataset", "list", "model", "run_cb", "list_size", "wer", "wer_cb"]
        keys += [k for m in MEASURES for k in (m, f"{m}_cb")]
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in table:
                writer.writerow(row)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
