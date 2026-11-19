"""Build the LibriSpeech manifests of the contextual biasing benchmark, with the
biasing lists of Le et al., "Contextualized Streaming End-to-End Speech
Recognition with Trie-Based Deep Biasing and Shallow Fusion", Interspeech 2021
(arXiv:2104.02194), released in https://github.com/facebookresearch/fbai-speech
(folder is21_deep_bias, MIT licence).

Protocol of these lists (per utterance): the rare words of the reference (words
outside the 5,000 most frequent words of the LibriSpeech training data), plus N
distractors drawn from the other rare words, N in {100, 500, 1000, 2000}. The
reference scorer of the paper (score.py) reports WER, U-WER and B-WER.

Input files: is21_deep_bias/ref/<split>.biasing_<N>.tsv, one line per utterance:
utterance id, reference text, rare words of the reference (JSON list), biasing
list (JSON list).

Outputs:
- lists/librispeech/<split>.biasing_<N>.json and <split>.rare_words.json:
  {utterance id: list}, referenced from the manifests as "file.json#<utterance id>";
- manifests/LibriSpeech_<split>.jsonl: one row per utterance, with one list
  variant per N (named after N) and the rare words as targets.

Usage:
    git clone --depth 1 https://github.com/facebookresearch/fbai-speech
    python tools/contextual_biasing/prepare_librispeech.py \\
        --librispeech_dir /home/data/audio/raw/transcript/en/LIBRISPEECH/LibriSpeech \\
        --biasing_dir fbai-speech/is21_deep_bias \\
        --output_dir benchmarks/contextual_biasing
"""

import argparse
import json
from pathlib import Path

import soundfile


def read_biasing_tsv(path):
    """{utterance id: (text, rare words, biasing list)}"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            utt, text, rare, biasing = line.rstrip("\n").split("\t")
            out[utt] = (text, json.loads(rare), json.loads(biasing))
    return out


def read_transcripts(split_dir):
    """{utterance id: (flac path, text)} of a LibriSpeech split folder."""
    out = {}
    for trans in sorted(split_dir.glob("*/*/*.trans.txt")):
        with open(trans, "r", encoding="utf-8") as f:
            for line in f:
                utt, text = line.rstrip("\n").split(" ", 1)
                out[utt] = (trans.parent / f"{utt}.flac", text)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--librispeech_dir", required=True, help="LibriSpeech folder (test-clean/, test-other/)")
    parser.add_argument("--biasing_dir", required=True, help="is21_deep_bias folder of fbai-speech")
    parser.add_argument("--output_dir", default="benchmarks/contextual_biasing", help="Output folder")
    parser.add_argument("--audio_prefix", default="<AUDIO_FOLDER>/raw/transcript/en/LIBRISPEECH/LibriSpeech",
                        help="LibriSpeech folder written in the manifests")
    parser.add_argument("--splits", nargs="+", default=["test-clean", "test-other"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 1000, 2000], help="Numbers of distractors N")
    parser.add_argument("--lists_in_manifest_prefix", default=None,
                        help="Path of the lists folder written in the manifests (default: <output_dir>/lists/librispeech)")
    args = parser.parse_args()

    librispeech = Path(args.librispeech_dir)
    output = Path(args.output_dir)
    lists_dir = output / "lists" / "librispeech"
    lists_dir.mkdir(parents=True, exist_ok=True)
    lists_ref = Path(args.lists_in_manifest_prefix) if args.lists_in_manifest_prefix else lists_dir

    for split in args.splits:
        transcripts = read_transcripts(librispeech / split)
        rare_words = None
        list_files = {}
        for n in args.sizes:
            biasing = read_biasing_tsv(Path(args.biasing_dir) / "ref" / f"{split}.biasing_{n}.tsv")
            if set(biasing) != set(transcripts):
                raise ValueError(f"{split}, N={n}: the utterances of the lists differ from the LibriSpeech folder")
            rare = {utt: rare for utt, (_, rare, _) in biasing.items()}
            if rare_words is None:
                rare_words = rare
            elif rare != rare_words:
                raise ValueError(f"{split}, N={n}: rare words differ between list sizes")
            list_file = lists_dir / f"{split}.biasing_{n}.json"
            with open(list_file, "w", encoding="utf-8") as f:
                json.dump({utt: terms for utt, (_, _, terms) in biasing.items()}, f, ensure_ascii=False)
            list_files[str(n)] = list_file.name
            sizes = [len(terms) for _, _, terms in biasing.values()]
            print(f"{split} N={n}: lists of {min(sizes)} to {max(sizes)} terms -> {list_file}")
        rare_file = lists_dir / f"{split}.rare_words.json"
        with open(rare_file, "w", encoding="utf-8") as f:
            json.dump(rare_words, f, ensure_ascii=False)

        rows = []
        for utt in sorted(transcripts):
            flac, text = transcripts[utt]
            info = soundfile.info(str(flac))
            rows.append({
                "audio_filepath": f"{args.audio_prefix}/{flac.relative_to(librispeech)}",
                "duration": round(info.frames / info.samplerate, 3),
                "offset": 0.0,
                "text": text.lower(),
                "id": utt,
                "split": split,
                "contextual_biasing": {n: f"{lists_ref / name}#{utt}" for n, name in list_files.items()},
                "contextual_biasing_targets": f"{lists_ref / rare_file.name}#{utt}",
            })
        manifest = output / "manifests" / f"LibriSpeech_{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        n_rare = sum(len(r) for r in rare_words.values())
        n_words = sum(len(row["text"].split()) for row in rows)
        hours = sum(row["duration"] for row in rows) / 3600
        print(f"{split}: {len(rows)} utterances ({hours:.2f} h), {n_rare} rare words in the lists "
              f"({100 * n_rare / n_words:.1f}% of {n_words} words) -> {manifest}")


if __name__ == "__main__":
    main()
