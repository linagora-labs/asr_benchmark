"""Build the manifests of the whole test sets of the sota benchmark, references normalized.

    python tools/build_manifests.py [--output_folder benchmarks/sota/manifests]
    python tools/build_manifests.py --benchmark sota_multilingual  # FLEURS and mTEDx in other languages
"""
import argparse
import json
import logging
import os
import re
import unicodedata
from pathlib import Path

from ssak.utils.text_latin import format_text_latin, numbers_and_symbols_to_letters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA = "/data-server/datasets/audio"
KALDI = f"{DATA}/kaldi/fr"
NEMO = f"{DATA}/nemo/asr/fr/nocontext"


def normalize_numbers(text):
    """VoxPopuli: numbers in letters"""
    out = re.sub(r"\s+", " ", numbers_and_symbols_to_letters(text, lang="fr")).strip()
    for p in ".,?!;:":
        if " " + p not in text:
            out = out.replace(" " + p, p)
    return out


def normalize_youtube(text):
    """YouTube: removes dialog dashes, numbers in letters..., keeps the spaces before ? and !.
    Removes the commas added after a punctuation when the subtitles were merged ("Merde!, Clyde!,",
    "des...,... jouer", "vos sacs,, Ouais"): format_text_latin also turns ". -J'ai" into "., J'ai"."""
    out = format_text_latin(text, lang="fr", lower_case=False, keep_punc=True, wer_format=False)
    out = re.sub(r",{2,}", ",", re.sub(r"([.!?…]),(?:\s*(?:\.\.\.|…)(?=\s))?", r"\1", out))
    for p in "?!":
        if " " + p in text:
            out = re.sub(r"(?<=\S)" + re.escape(p), " " + p, out)
    return out


def clean(text):
    """All datasets:
    - NFC: accents written as a letter + a combining accent (some Common Voice sentences)
      would be other words than the usual "é" in the punctuated wer/cer
    - parentheses and brackets removed, their content kept: the FLEURS readers mostly read
      it (about 2 times out of 3), and the nocasepunc normalization would drop it"""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", re.sub(r"[()\[\]]", "", text)).strip()


DATASETS = {
    "CommonVoice": dict(nemo=f"{NEMO}/CommonVoice/test.jsonl"),  # Common Voice 22.0
    "MLS_Facebook_french": dict(nemo=f"{NEMO}/Multilingual_LibriSpeech/test.jsonl"),
    "Voxpopuli": dict(nemo=f"{NEMO}/VoxPopuli/test.jsonl", normalize=normalize_numbers),
    "SUMM-RE": dict(nemo=f"{NEMO}/SUMM-RE/test.jsonl"),
    "YouTubeFr_split6": dict(nemo=f"{NEMO}/YouTubeFr/test_split6.jsonl", normalize=normalize_youtube),
    # Only the wav conversions are on the DGX
    "TEDX_fr": dict(kaldi=f"{KALDI}/Multilingual_TEDx/casepunc/test", audio=f"{DATA}/converted_audios/transcript/fr/Multilingual_TEDx"),
    "FLEURS": dict(nemo=f"{NEMO}/FLEURS/test.jsonl"),
}

# sota_multilingual: FLEURS and Multilingual TEDx of the other languages (those of AudioBench),
# whole test sets, one config per language (benchmarks/sota_multilingual/config_<lang>.yaml).
# mTEDx: the manifests point to .flac, only the wav conversions are on the DGX (as TEDX_fr).
MULTILINGUAL_LANGUAGES = {"en": ["FLEURS"], "nl": ["FLEURS"]} | {lang: ["FLEURS", "TEDX"] for lang in ("it", "de", "es", "pt", "ar")}
DATASETS_MULTILINGUAL = {
    f"{name}_{lang}": dict(
        nemo=f"{DATA}/nemo/asr/{lang}/nocontext/{'FLEURS' if name == 'FLEURS' else 'Multilingual_TEDx'}/test.jsonl",
        **({"audio": f"{DATA}/converted_audios/transcript/{lang}/Multilingual_TEDx"} if name == "TEDX" else {}),
    )
    for lang, names in MULTILINGUAL_LANGUAGES.items() for name in names
}
# 3 test sets in English, 2 in Dutch. VoxPopuli: only the wav conversions are on the DGX.
# CommonVoice_en (16k segments): a ratio in the config.
for _lang in ("en", "nl"):
    DATASETS_MULTILINGUAL[f"Voxpopuli_{_lang}"] = dict(
        nemo=f"{DATA}/nemo/asr/{_lang}/nocontext/VoxPopuli/test.jsonl", audio=f"{DATA}/converted_audios/transcript/{_lang}/VoxPopuli")
DATASETS_MULTILINGUAL["CommonVoice_en"] = dict(nemo=f"{DATA}/nemo/asr/en/nocontext/CommonVoice/test.jsonl")  # Common Voice 22.0
BENCHMARKS = {"sota": DATASETS, "sota_multilingual": DATASETS_MULTILINGUAL}

# mTEDx sound events, removed with their parentheses (clean() keeps the content of the other
# parentheses: spoken asides, names in the Arabic talks). The segments left without any word
# (applause...) are dropped, TEDX_fr (kaldi casepunc) has neither.
TEDX_EVENTS = {
    "it": ["Applausi", "Risate"],
    "de": ["Lachen", "Applaus", "Gelächter", "Musik", "Beifall", "Klarinettenmusik"],
    "es": ["Risas", "Silbido en el público", "Aplausos", "Suspiro", "Ovación", "Vídeo", "Música", "Ruido",
           "Sonido de platillos", "Sonido de tambor", "Sonido de relincho de caballo", "Vítores",
           "Fin de aplausos", "Exhalación", "El público pregunta"],
    "pt": ["Aplausos", "Risos", "Vivas", "Música", "Fim da música", "ininteligível", "Som dos metrónomos dessincronizados",
           "Som dos metrónomos sincronizados", "Inglês", "Gritos da plateia", "Gritos"],
    "ar": ["تصفيق", "ضحك", "صوت أطفال", "فيديو", "غناء", "تصفيق وهتاف"],
}


def remove_tedx_events(lang):
    events = re.compile(r"\((?:" + "|".join(map(re.escape, TEDX_EVENTS[lang])) + r")\)"
                        # one Arabic talk writes its applause without parentheses, at the start
                        + (r"|^تصفيق(?=\s|$)" if lang == "ar" else ""))
    return lambda text: re.sub(r"\s+", " ", events.sub(" ", text)).strip()


for _lang, _names in MULTILINGUAL_LANGUAGES.items():
    if "TEDX" in _names:
        DATASETS_MULTILINGUAL[f"TEDX_{_lang}"]["normalize"] = remove_tedx_events(_lang)


def to_placeholder(path):
    path = path.replace(f"{DATA}/", "<AUDIO_FOLDER>/")
    converted = re.sub(r"^<AUDIO_FOLDER>/raw/(.*)\.mp3$", r"<AUDIO_FOLDER>/converted_audios/\1.wav", path)
    if converted != path and os.path.exists(converted.replace("<AUDIO_FOLDER>", DATA)):
        return converted
    return path


def load_kaldi(folder):
    def read(name):
        with open(f"{folder}/{name}", encoding="utf-8") as f:
            return dict(line.rstrip("\n").split(" ", 1) for line in f if line.strip())
    text, wav, spk, dur = read("text"), read("wav.scp"), read("utt2spk"), read("utt2dur")
    segments = {}
    if os.path.exists(f"{folder}/segments"):
        with open(f"{folder}/segments") as f:
            for line in f:
                utt, rec, start, _ = line.split()
                segments[utt] = (rec, float(start))
    rows = []
    for utt, t in text.items():
        rec, offset = segments.get(utt, (utt, 0.0))
        rows.append(dict(duration=float(dur[utt]), speaker=spk[utt], audio_filepath=to_placeholder(wav[rec].split()[0]), offset=offset, text=t))
    return rows


def load_nemo(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            audio = next(c for c in d["conversations"] if c["type"] == "audio")
            text = next(c for c in d["conversations"] if c["from"] == "Assistant")["value"]
            row = dict(duration=audio["duration"], audio_filepath=to_placeholder(audio["value"]), offset=audio.get("offset", 0.0), text=text)
            if "speaker" in d:
                row["speaker"] = d["speaker"]
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="sota")
    parser.add_argument("--output_folder", type=Path, default=None, help="Default: benchmarks/<benchmark>/manifests")
    args = parser.parse_args()

    args.output_folder = args.output_folder or Path(f"benchmarks/{args.benchmark}/manifests")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    for name, conf in BENCHMARKS[args.benchmark].items():
        rows = load_nemo(conf["nemo"]) if "nemo" in conf else load_kaldi(conf["kaldi"])
        normalize = conf.get("normalize", lambda t: t)
        for row in rows:
            if "audio" in conf:
                row["audio_filepath"] = to_placeholder(f"{conf['audio']}/{Path(row['audio_filepath']).stem}.wav")
            row.update(text=clean(normalize(row["text"])), split="test", name=name)
        empty = [row for row in rows if not row["text"]]
        if empty:
            logger.info(f"{name}: {len(empty)} segments without reference removed")
            rows = [row for row in rows if row["text"]]
        with open(args.output_folder / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"{name}: {len(rows)} segments ({sum(r['duration'] for r in rows) / 3600:.2f}h)")
