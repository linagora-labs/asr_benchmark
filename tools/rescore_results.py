"""Recompute performances/*.json of benchmark results from their predictions/*.json,
with the current scoring code (WER/CER in every mode + 95% confidence intervals).

Use it when the scoring changes, so that all the models of a benchmark are scored
the same way. Prints every score that changed.

    python tools/rescore_results.py benchmarks/sota/results [-n]

Contextual biasing measures (when the manifest has biasing lists) are kept as they
are, or recomputed with --manifest (the manifest of the benchmark, with the lists).
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

from asr_benchmark.benchmark import contextual_biasing
from asr_benchmark.benchmark.benchmark_manager import score_dataset, text_language


def main(results, dry_run=False, manifest=None):
    rows = contextual_biasing.manifest_rows(str(manifest)) if manifest else None
    for exp in sorted(p for p in results.iterdir() if (p / "metadata.json").exists()):
        metadata = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
        for pred_file in sorted((exp / "predictions").glob("*.json")):
            perf_file = exp / "performances" / pred_file.name
            data = json.loads(pred_file.read_text(encoding="utf-8"))
            language = text_language(metadata, data)
            with contextlib.redirect_stderr(io.StringIO()):  # ssak's progress bars
                new = score_dataset(data, next(iter(data.values())).get("language") or language)
            old = json.loads(perf_file.read_text(encoding="utf-8")) if perf_file.exists() else {}
            if rows is not None:
                biasing = contextual_biasing.score_dataset(data, rows.get(pred_file.stem, {}), metadata, language)
                if biasing:
                    new[contextual_biasing.MANIFEST_KEY] = biasing
            elif contextual_biasing.MANIFEST_KEY in old:
                new[contextual_biasing.MANIFEST_KEY] = old[contextual_biasing.MANIFEST_KEY]
            for key, score in new.items():
                if key == contextual_biasing.MANIFEST_KEY:
                    continue
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
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest with the contextual biasing lists, to recompute their measures")
    args = parser.parse_args()
    main(args.results, args.dry_run, args.manifest)
