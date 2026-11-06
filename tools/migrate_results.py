"""Align existing results on a new manifest: predictions of matching segments (audio file name,
offset, duration) are reused, the others removed, then the experiments are rescored.

    python tools/migrate_results.py benchmarks/sota/manifest_new.jsonl benchmarks/sota/results [-n]
"""
import argparse
import collections
import contextlib
import io
import json
import re
from pathlib import Path

from asr_benchmark.benchmark.benchmark_manager import make_perf_file, score_dataset


def segment_key(audio_filepath, offset):
    return Path(audio_filepath).stem, round(float(offset or 0.0), 1)


def row_id(row):
    id = f"{row['name']}_{Path(row['audio_filepath']).stem}"
    if float(row.get("offset", 0.0)) > 0.0:
        id += f"_{row['offset']:.1f}"
    return id


def same_dataset(old, name):
    """Old names end in "_max30"."""
    return old.replace("_max30", "") == name


def migrate(exp, manifest, language, dry_run):
    predictions_dir = exp / "predictions"
    old = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in predictions_dir.glob("*.json")}
    root = next((m.group(1) for preds in old.values() for p in preds.values()
                 if (m := re.match(r"(.*?)/(raw|converted_audios)/", p["audio_filepath"]))), None)
    new = {}
    for name, rows in manifest.items():
        index = {segment_key(p["audio_filepath"], p.get("audio_offset")): p
                 for stem, preds in old.items() if same_dataset(stem, name) for p in preds.values()}
        new[name] = {}
        for row in rows:
            pred = index.get(segment_key(row["audio_filepath"], row.get("offset")))
            if pred is None or abs(pred["audio_duration"] - row["duration"]) > 0.05:
                continue
            perf = make_perf_file(row | {"id": row_id(row)})
            new[name][row_id(row)] = pred | {
                "id": row_id(row), "dataset": perf["dataset"], "full_dataset": perf["full_dataset"], "text": row["text"],
                "audio_filepath": row["audio_filepath"].replace("<AUDIO_FOLDER>", root) if root else row["audio_filepath"]}
        changed = sum(p["text"] != index[segment_key(p["audio_filepath"], p.get("audio_offset"))]["text"] for p in new[name].values())
        print(f"{exp.name}  {name}: {len(new[name])}/{len(rows)} reused ({changed} references changed)")
    removed = [stem for stem in old if stem not in manifest]
    dropped = sum(len(old[stem]) for stem in old) - sum(len(preds) for preds in new.values())
    print(f"{exp.name}  {dropped} predictions dropped (not in the manifest), files removed: {removed}")
    if dry_run:
        return
    for stem in removed:
        (predictions_dir / f"{stem}.json").unlink()
        (exp / "performances" / f"{stem}.json").unlink(missing_ok=True)
        for alignment in (exp / "alignments").glob(f"*/{stem}.txt"):
            alignment.unlink()
    (exp / "performances").mkdir(exist_ok=True)
    for name, preds in new.items():
        if not preds:
            continue
        (predictions_dir / f"{name}.json").write_text(json.dumps(preds, indent=4, ensure_ascii=False), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):  # ssak's progress bars
            scores = score_dataset(preds, language)
        (exp / "performances" / f"{name}.json").write_text(json.dumps(scores, indent=4), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="New manifest")
    parser.add_argument("results", type=Path, help="Folder of experiments (e.g. benchmarks/sota/results)")
    parser.add_argument("-n", "--dry_run", action="store_true", help="Only print what would be reused")
    args = parser.parse_args()

    manifest = collections.defaultdict(list)
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            manifest[row["name"]].append(row)
    for exp in sorted(p for p in args.results.iterdir() if (p / "predictions").is_dir()):
        metadata = exp / "metadata.json"
        language = (json.loads(metadata.read_text(encoding="utf-8")).get("language") if metadata.exists() else None) or "fr"
        migrate(exp, manifest, language, args.dry_run)
