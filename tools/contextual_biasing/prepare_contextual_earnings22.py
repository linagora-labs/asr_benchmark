"""Build the Contextual Earnings-22 manifests of the contextual biasing benchmark.

Contextual Earnings-22 (Argmax, https://huggingface.co/datasets/argmaxinc/contextual-earnings22,
same licence as Earnings-22; paper: Durmus et al. 2026, arXiv:2604.07354): 15-second
clips of English earnings calls (Earnings-22), with keyword lists of company,
person and product names. Two lists per clip, as observed in the data:
- `dictionary`: the keywords spoken in the clip ("local" list, the targets);
- `keywords`: the keyword inventory of the whole call ("global" list: the local
  keywords plus names of the call that are not spoken in the clip).

The audio is stored in the parquet files (16 kHz WAV bytes): this script writes it
to <dataset_dir>/wav/<split>/<file_id>.wav, then one manifest per split, with the
lists "local" and "global" and the local keywords as targets.

Usage:
    huggingface-cli download argmaxinc/contextual-earnings22 --repo-type dataset \\
        --local-dir /home/data/audio/raw/transcript/en/ContextualEarnings22
    python tools/contextual_biasing/prepare_contextual_earnings22.py \\
        --dataset_dir /home/data/audio/raw/transcript/en/ContextualEarnings22
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True, help="Local copy of the HF dataset (data/<split>-*.parquet)")
    parser.add_argument("--output_dir", default="benchmarks/contextual_biasing", help="Output folder (manifests/)")
    parser.add_argument("--audio_prefix", default="<AUDIO_FOLDER>/raw/transcript/en/ContextualEarnings22/wav",
                        help="Audio folder written in the manifests")
    parser.add_argument("--splits", nargs="+", default=["test", "dev"])
    parser.add_argument("--keep_flagged", action="store_true", help="Keep the clips with is_flagged = true")
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    for split in args.splits:
        rows, n_flagged = [], 0
        wav_dir = dataset / "wav" / split
        wav_dir.mkdir(parents=True, exist_ok=True)
        for parquet in sorted((dataset / "data").glob(f"{split}-*.parquet")):
            for item in pq.read_table(parquet).to_pylist():
                if item["is_flagged"] and not args.keep_flagged:
                    n_flagged += 1
                    continue
                audio = item["audio"]["bytes"]
                if audio[:4] != b"RIFF":
                    raise ValueError(f"{item['file_id']}: audio is not a WAV file")
                wav = wav_dir / f"{item['file_id']}.wav"
                if not wav.exists():
                    wav.write_bytes(audio)
                rows.append({
                    "audio_filepath": f"{args.audio_prefix}/{split}/{wav.name}",
                    "duration": round(float(item["duration"]), 3),
                    "offset": 0.0,
                    "text": item["text"],
                    "id": item["file_id"],
                    "split": split,
                    "contextual_biasing": {"local": item["dictionary"], "global": item["keywords"]},
                    "contextual_biasing_targets": item["dictionary"],
                })
        if not rows:
            raise FileNotFoundError(f"No data for split {split} in {dataset / 'data'}")
        manifest = Path(args.output_dir) / "manifests" / f"ContextualEarnings22_{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in sorted(rows, key=lambda r: r["id"]))
        n_local = sum(len(r["contextual_biasing"]["local"]) for r in rows)
        n_global = sum(len(r["contextual_biasing"]["global"]) for r in rows)
        hours = sum(r["duration"] for r in rows) / 3600
        print(f"{split}: {len(rows)} clips ({hours:.2f} h), {n_flagged} flagged clips removed, "
              f"{n_local / len(rows):.1f} local / {n_global / len(rows):.1f} global keywords per clip -> {manifest}")


if __name__ == "__main__":
    main()
