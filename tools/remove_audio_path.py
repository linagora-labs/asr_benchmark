import argparse
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path, min_duration=0.05):
    data = []
    with open(manifest_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes a sub sample of the data')
    parser.add_argument('manifest', help="Input manifest", type=str)
    parser.add_argument('--output_manifest', help="Output directory", type=str, default="manifest_shrinked.jsonl")
    args = parser.parse_args()

    data = []
    data = load_manifest(args.manifest)
    with open(args.output_manifest, "w") as f:
        for row in data:
            row['audio_filepath'] = os.path.basename(row['audio_filepath'])
            f.write(json.dumps(row))
            f.write("\n")
    