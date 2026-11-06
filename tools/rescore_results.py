"""Recompute performances/*.json of benchmark results from their predictions/*.json,
with the current scoring code (WER/CER in every mode + 95% confidence intervals).

Use it when the scoring changes, so that all the models of a benchmark are scored
the same way. Prints every score that changed.

    python tools/rescore_results.py benchmarks/sota/results [-n]
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

from asr_benchmark.benchmark.benchmark_manager import score_dataset


def main(results, dry_run=False):
    for exp in sorted(p for p in results.iterdir() if (p / "metadata.json").exists()):
        language = json.loads((exp / "metadata.json").read_text(encoding="utf-8")).get("language") or "fr"
        for pred_file in sorted((exp / "predictions").glob("*.json")):
            perf_file = exp / "performances" / pred_file.name
            data = json.loads(pred_file.read_text(encoding="utf-8"))
            with contextlib.redirect_stderr(io.StringIO()):  # ssak's progress bars
                new = score_dataset(data, language)
            old = json.loads(perf_file.read_text(encoding="utf-8")) if perf_file.exists() else {}
            for key, score in new.items():
                if isinstance(score, dict):
                    before = old.get(key, {}).get("wer")
                    if before is None or abs(before - score["wer"]) > 0.005:
                        print(f"{exp.name}  {pred_file.stem}  {key}: {before if before is None else round(before, 2)} -> {score['wer']:.2f}")
            if not dry_run:
                perf_file.parent.mkdir(exist_ok=True)
                perf_file.write_text(json.dumps(new, indent=4), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Folder of experiments (e.g. benchmarks/sota/results)")
    parser.add_argument("-n", "--dry_run", action="store_true", help="Only print the changes")
    args = parser.parse_args()
    main(args.results, args.dry_run)
