"""Recompute performances/*.json of benchmark results from their predictions/*.json,
with the current scoring code (WER/CER in every mode + 95% confidence intervals).

Use it when the scoring changes, so that all the models of a benchmark are scored
the same way. Prints every score that changed.

    python tools/rescore_results.py benchmarks/sota/results [-n] [-j 16]

Contextual biasing measures (when the manifest has biasing lists) are kept as they
are, or recomputed with --config (the config of the benchmark, with its datasets) or
--manifest (a manifest with the lists and the names of the datasets). The measures of
the lists whose files are missing (not redistributable) are kept as they are:

    python tools/rescore_results.py benchmarks/contextual_biasing/results_fr -j 16 \\
        --config benchmarks/contextual_biasing/config_fr.yaml

The scores without predictions (not in git: made with these lists) are kept too.
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

import yaml

from asr_benchmark.benchmark import contextual_biasing
from asr_benchmark.benchmark.benchmark_manager import score_dataset, text_language
from asr_benchmark.utils.benchmark import select_datasets

KEY = contextual_biasing.MANIFEST_KEY


def config_rows(config_file):
    """{dataset name: {id: row}} of the datasets of a config."""
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    rows = {}
    for row in select_datasets(config["datasets"], **{k: config[k] for k in ("min_duration", "max_duration") if k in config}):
        rows.setdefault(row["name"], {})[row["id"]] = row
    return rows


def remove_missing_lists(rows):
    """Removes the lists whose files are missing from the rows ({dataset: {id: row}}).
    Returns {dataset: their names}."""
    missing = {}
    for dataset, dataset_rows in rows.items():
        variants = contextual_biasing.row_variants(next(iter(dataset_rows.values())))
        for name, value in variants.items():
            try:
                contextual_biasing.load_terms(value)
            except FileNotFoundError as err:
                print(f"{dataset}, list {name}: {err}. Its measures are kept as they are")
                missing.setdefault(dataset, []).append(name)
        for row in dataset_rows.values():
            if dataset in missing:
                row[KEY] = {k: v for k, v in row[KEY].items() if k not in missing[dataset]}
    return missing


def rescore(pred_file, rows, missing, dry_run):
    """Lines to print: the scores that changed."""
    exp, dataset = pred_file.parent.parent, pred_file.stem
    metadata = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
    perf_file = exp / "performances" / pred_file.name
    data = json.loads(pred_file.read_text(encoding="utf-8"))
    language = text_language(metadata, data)
    old = json.loads(perf_file.read_text(encoding="utf-8")) if perf_file.exists() else {}
    with contextlib.redirect_stderr(io.StringIO()):  # ssak's progress bars
        new = score_dataset(data, language)
        biasing = old.get(KEY)
        if rows is not None:
            dataset_rows, kept = rows.get(dataset, {}), missing.get(dataset, [])
            if contextual_biasing.config_variant(metadata) not in kept:
                biasing = contextual_biasing.score_dataset(data, dataset_rows, metadata, language) if dataset_rows else biasing
                if biasing is not None and contextual_biasing.config_variant(metadata) is None:
                    biasing = {k: v for k, v in (old.get(KEY) or {}).items() if k in kept} | biasing
    if biasing:
        new[KEY] = biasing
    if not dry_run:
        perf_file.parent.mkdir(exist_ok=True)
        perf_file.write_text(json.dumps(new, indent=4), encoding="utf-8")

    lines = []
    for key, score in new.items():
        if key != KEY and isinstance(score, dict):
            before = old.get(key, {}).get("wer")
            if before is None or abs(before - score["wer"]) > 0.005:
                lines.append(f"{exp.name}  {dataset}  {key}: {before if before is None else round(before, 2)} -> {score['wer']:.2f}")
    for name, modes in (biasing or {}).items():
        for mode, score in modes.items():
            before = ((old.get(KEY) or {}).get(name) or {}).get(mode, {})
            for key in ("f1", "b_wer"):
                a, b = before.get(key), score.get(key)
                if a != b and (a is None or b is None or abs(a - b) > 0.005):
                    lines.append(f"{exp.name}  {dataset}  {name} {mode} {key}: {a} -> {b}")
    return lines


def main(results, dry_run=False, manifest=None, config=None, jobs=1):
    rows, missing = None, {}
    if config:
        rows = config_rows(config)
    elif manifest:
        rows = contextual_biasing.manifest_rows(str(manifest))
    if rows is not None:
        missing = remove_missing_lists(rows)
    experiments = sorted(p for p in results.iterdir() if (p / "metadata.json").exists())
    for exp in experiments:
        for perf_file in sorted((exp / "performances").glob("*.json")):
            if not (exp / "predictions" / perf_file.name).exists():
                print(f"{exp.name}  {perf_file.stem}: no predictions, the scores are kept")
    files = [f for exp in experiments for f in sorted((exp / "predictions").glob("*.json"))]
    if rows is not None:
        for f in [f for f in files if f.stem not in rows]:
            print(f"{f.parent.parent.name}  {f.stem}: not in the benchmark, the contextual biasing measures are kept")
    function = partial(rescore, rows=rows, missing=missing, dry_run=dry_run)
    with mp.Pool(jobs) if jobs > 1 else contextlib.nullcontext() as pool:
        for i, lines in enumerate(pool.imap(function, files) if pool else map(function, files), 1):
            for line in lines:
                print(line)
            print(f"[{i}/{len(files)}]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Folder of experiments (e.g. benchmarks/sota/results)")
    parser.add_argument("-n", "--dry_run", action="store_true", help="Only print the changes")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of processes")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", type=Path, default=None,
                       help="Config of the benchmark, to recompute the contextual biasing measures with the lists of its datasets")
    group.add_argument("--manifest", type=Path, default=None,
                       help="Manifest with the contextual biasing lists, to recompute their measures")
    args = parser.parse_args()
    main(args.results, args.dry_run, args.manifest, args.config, args.jobs)
