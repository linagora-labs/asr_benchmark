"""Write the segments selected by the `datasets` of a config in a manifest.

    python tools/select_manifest.py benchmarks/sota/config.yaml benchmarks/sota/manifest.jsonl
"""
import argparse
import collections
import json

import yaml

from asr_benchmark.utils.benchmark import select_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Config with a `datasets` block")
    parser.add_argument("output_manifest")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    durations = {k: config[k] for k in ("min_duration", "max_duration") if k in config}
    rows = select_datasets(config["datasets"], **durations)
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    for name, n in collections.Counter(row["name"] for row in rows).items():
        hours = sum(row["duration"] for row in rows if row["name"] == name) / 3600
        print(f"{name}: {n} segments ({hours:.2f}h)")
    print(f"{len(rows)} segments -> {args.output_manifest}")
