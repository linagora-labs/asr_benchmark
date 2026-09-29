"""Fill the "model_info" of benchmark results (metadata.json) from the Hugging Face API:
repo id, license, number of parameters and languages of the model card. The leaderboard
(tools/leaderboard/build.py) displays them.

Only the empty fields are filled: what is already there, e.g. filled by hand, is kept
(--force: replaced by what the API gives, where it gives something). The fields still
empty are listed at the end, to be filled by hand in metadata.json (the benchmark keeps
model_info across reruns): the languages of gated models... The benchmark itself counts
the parameters of the models it loads in its process, e.g. .nemo checkpoints that have
no safetensors for the API.

    python tools/fill_model_info.py benchmarks/sota/results [-n] [--force]
"""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/models/"


def fetch(repo):
    try:
        with urllib.request.urlopen(API + repo, timeout=30) as response:
            return json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def candidates(meta):
    """Hugging Face repo ids the model of an experiment may have."""
    backend, model = meta["backend"], meta["model"]
    if backend == "faster-whisper" and "/" not in model:  # OpenAI size names ("large-v3")
        model = f"openai/whisper-{model}"
    yield model
    if "/" in model:  # NeMo models are named with "-" where their repo has "_"
        owner, name = model.split("/", 1)
        yield f"{owner}/{name.replace('-', '_')}"


def model_info(meta):
    for repo in candidates(meta):
        data = fetch(repo)
        if data and "error" not in data:
            break
    else:
        return None
    card = data.get("cardData") or {}
    tags = data.get("tags", [])
    license = card.get("license") or next((t[len("license:"):] for t in tags if t.startswith("license:")), None)
    if license == "other":
        license = card.get("license_name")
    languages = card.get("language")
    return {
        "hf": data.get("id", repo),
        "license": license,
        "params": (data.get("safetensors") or {}).get("total"),
        "languages": [languages] if isinstance(languages, str) else languages,
    }


FIELDS = ["hf", "license", "params", "languages"]


def main(results, dry_run=False, force=False):
    todo = []
    for meta_file in sorted(results.glob("*/metadata.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        info = dict(meta.get("model_info") or {})
        if force or any(not info.get(k) for k in FIELDS):
            fetched = model_info(meta)
            if fetched is None:
                print(f"{meta_file.parent.name}: not found on Hugging Face")
            else:
                # Only the empty fields are filled (--force: every field the API gives), so
                # that what was filled by hand is never lost.
                new = {k: v for k, v in fetched.items() if v and (force or not info.get(k))}
                if new:
                    info.update(new)
                    print(f"{meta_file.parent.name}: {json.dumps(new, ensure_ascii=False)[:150]}")
                    if not dry_run:
                        meta["model_info"] = {k: info.get(k) for k in FIELDS}
                        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        missing = [k for k in FIELDS if not info.get(k)]
        if missing:
            todo.append(f"{meta_file.parent.name}: {', '.join(missing)}")
    if todo:
        print("\nStill empty, to fill by hand in metadata.json (model_info):\n  " + "\n  ".join(todo))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Folder of experiments (e.g. benchmarks/sota/results)")
    parser.add_argument("-n", "--dry_run", action="store_true", help="Only print what would be written")
    parser.add_argument("--force", action="store_true", help="Replace the existing fields by what the API gives")
    args = parser.parse_args()
    main(args.results, args.dry_run, args.force)
