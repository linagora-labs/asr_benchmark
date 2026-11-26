"""Build the static leaderboard page from benchmark results.

Writes <output>/index.html, <output>/leaderboard.json and the result cards of cards.json
(<output>/cards/*.svg, see cards.py). Standard library only (GitHub Action).

    python tools/leaderboard/build.py                       # benchmarks/sota/results -> site/
    python tools/leaderboard/build.py --results X --manifest M --output Y

The Multilingual tab: benchmarks/sota_multilingual/results (--multilingual_results), plus the
French FLEURS and TEDx of the same models from --results.

The Contextual Biasing tab: the page of tools/contextual_biasing (--contextual_results).
"""
import argparse
import base64
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from cards import build_cards

HERE = Path(__file__).parent
REPO_URL = "https://github.com/linagora-labs/asr_benchmark"
SITE_URL = "https://linagora-labs.github.io/asr_benchmark/"

# Display name, domain, language, source; column order of the page
DATASETS = {
    "FLEURS": ("FLEURS", "Read Wikipedia sentences", "fr", "https://huggingface.co/datasets/google/fleurs"),
    "CommonVoice": ("Common Voice 22.0", "Read speech", "fr", "https://commonvoice.mozilla.org/en/datasets"),
    "MLS_Facebook_french": ("MLS", "Read audiobooks", "fr", "https://openslr.org/94/"),
    "TEDX_fr": ("TEDx", "Prepared talks", "fr", "https://www.openslr.org/100/"),
    "Voxpopuli": ("VoxPopuli", "Parliament speeches", "fr", "https://huggingface.co/datasets/facebook/voxpopuli"),
    "SUMM-RE": ("SUMM-RE", "Spontaneous meetings", "fr", "https://huggingface.co/datasets/linagora/SUMM-RE"),
    "YouTubeFr_split6": ("YouTubeFr", "Web videos", "fr", None),
}
# Type of speech, under the name of the test set (both tabs, by prefix in the Multilingual one); the
# read and spontaneous ones make the "no read speech" and "spontaneous speech only" selections
SPEECH = {"FLEURS": "read", "CommonVoice": "read", "MLS_Facebook_french": "audiobooks", "TEDX": "talks",
          "TEDX_fr": "talks", "Voxpopuli": "parliament", "SUMM-RE": "meetings", "YouTubeFr_split6": "web videos"}
SPONTANEOUS = {"SUMM-RE", "YouTubeFr_split6"}
READ = {"FLEURS", "CommonVoice", "MLS_Facebook_french"}  # by prefix in the Multilingual tab
# Shown on hover of the name of the test sets without a link
DATASET_NOTES = {
    "YouTubeFr_split6": "Internal corpus of French YouTube videos",
}
# Share kept (`datasets` of config.yaml), details on hover
SELECTION = {
    "CommonVoice": ("20%", None),
    "MLS_Facebook_french": ("50%", None),
    "TEDX_fr": ("Whole", None),
    "Voxpopuli": ("50%", None),
    "SUMM-RE": ("8.5%", "Half of the 3,002 segments of 4 to 30 s (1,453 of the 17,113 segments of the split)"),
    "YouTubeFr_split6": ("Whole", None),
    "FLEURS": ("Whole", None),
}
DEFAULT_LANGUAGE = "fr"
# Multilingual tab: <prefix>_<language> test sets of benchmarks/sota_multilingual (whole, no
# duration filter), plus the French ones of --results (id in that tab: its stem there)
MULTILINGUAL_DATASETS = {
    "FLEURS": ("FLEURS", "Read Wikipedia sentences", "https://huggingface.co/datasets/google/fleurs"),
    "TEDX": ("mTEDx", "Prepared talks", "https://www.openslr.org/100/"),
    "Voxpopuli": ("VoxPopuli", "Parliament speeches", "https://huggingface.co/datasets/facebook/voxpopuli"),
    "CommonVoice": ("Common Voice 22.0", "Read speech", "https://commonvoice.mozilla.org/en/datasets"),
}
MULTILINGUAL_SELECTION = {"CommonVoice_en": "10%"}
MULTILINGUAL_FRENCH = {"FLEURS": "FLEURS_fr", "TEDX_fr": "TEDX_fr"}  # stem in --results -> id
MULTILINGUAL_LANGUAGES = ["en", "fr", "de", "it", "es", "pt", "nl", "ar"]  # column order
LANGUAGE_NAMES = {"en": "English", "fr": "French", "de": "German", "it": "Italian", "es": "Spanish", "pt": "Portuguese",
                  "nl": "Dutch", "ar": "Arabic"}
METRICS = [metric + suffix for metric in ("wer", "cer") for suffix in ("_nocasepunc", "_nopunc", "_nocase", "")]
EMPTY_WARNING = 0.01  # share of empty outputs from which the model gets a caveat
SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")
PUNCTUATION = re.compile(r"[.,?!;:]")


def dataset_info(stem):
    # Speed runs keep the old names
    name, domain, language, url = DATASETS.get(stem.replace("_max30", ""), (stem, "", DEFAULT_LANGUAGE, None))
    return {"name": name, "domain": domain, "language": language, "url": url,
            "note": DATASET_NOTES.get(stem.replace("_max30", "")),
            "speech": SPEECH.get(stem.replace("_max30", "")), "spontaneous": stem in SPONTANEOUS, "read": stem in READ,
            "selection": SELECTION.get(stem, (None, None))[0],
            "selection_note": SELECTION.get(stem, (None, None))[1]}


def speed(exp, rtf_results, model=None):
    """RTFx (seconds of audio transcribed per second of computation), hardware, backend and
    caveat (`rtf_note` of its model_info, filled by hand) of the speed run of `exp`: <rtf_results>/<exp>_rtf, from benchmarks/sota/config_rtf.yaml.
    Without it, the speed run of the same `model` with another backend: config_rtf.yaml times
    the audio LLMs through vLLM (faster) while their WER may come from transformers."""
    if not rtf_results or not rtf_results.exists():
        return None, None, None, None
    exp_rtf = rtf_results / f"{exp.name}_rtf"
    if not (exp_rtf / "metadata.json").exists():
        same_model = [
            p.parent for p in sorted(rtf_results.glob("*/metadata.json"))
            if model and json.loads(p.read_text(encoding="utf-8")).get("model") == model
        ]
        if len(same_model) != 1:  # none, or ambiguous (several variants of the model)
            return None, None, None, None
        exp_rtf = same_model[0]
    audio = compute = 0
    for pred_file in (exp_rtf / "predictions").glob("*.json"):
        for row in json.loads(pred_file.read_text(encoding="utf-8")).values():
            if row.get("prediction_duration"):
                audio += row["audio_duration"]
                compute += row["prediction_duration"]
    meta = json.loads((exp_rtf / "metadata.json").read_text(encoding="utf-8"))
    return ((round(audio / compute, 1) if compute else None), meta.get("device_name") or meta.get("device"),
            meta.get("backend"), (meta.get("model_info") or {}).get("rtf_note"))
HISTOGRAM_BIN = 2  # seconds, durations histogram of the test sets

# Language control of the multilingual models, from their model cards: (pattern on the model
# name, (automatic detection, forced language)), the automatic detection being True, "returned"
# (True, and the detected language comes with the transcript) or False, the forced language "yes"
# (a language parameter or a documented language field), "prompt" (only an instruction in the
# prompt) or None. The models of one language have none (shown as not applicable).
LANGUAGE_CONTROL = [
    (r"^openai/whisper-", ("returned", "yes")),
    (r"^nvidia/canary-", (False, "yes")),
    (r"parakeet-(tdt-0\.6b-v3|redux|ultra)$", (True, None)),
    (r"^nvidia/nemotron-labs-unified-asr-", (True, "yes")),
    (r"^nvidia/nemotron-3\.5-asr-streaming-", ("returned", "yes")),  # <xx-XX> tag after the transcript
    (r"^nvidia/NVIDIA-Nemotron-3\.5-Transcribe-", (True, "yes")),
    (r"^mistralai/Voxtral-(Mini-3B|Small-24B)-2507$", (True, "yes")),
    (r"^mistralai/Voxtral-Mini-4B-Realtime-", (True, None)),
    (r"^CohereLabs/cohere-transcribe-", (False, "yes")),
    (r"^Qwen/Qwen3-ASR-", ("returned", "yes")),
    (r"^Qwen/Qwen3-Omni-", (True, "prompt")),
    (r"^Reson8/Rumble-", ("returned", "yes")),  # "language <Name><asr_text>"
    (r"^ibm-granite/granite-speech-5\.", (True, "yes")),
    (r"^ibm-granite/granite-speech-[34]\.", (True, "prompt")),
    (r"^google/gemma-", (True, "prompt")),
    (r"^OpenLLM-France/Luciole-", (True, "prompt")),
    (r"^OpenMOSS-Team/MOSS-", (True, "prompt")),
    (r"^Edge0/ARK-ASR-", (True, "prompt")),
    (r"^HojoAI/Hojo-ASR-", (True, None)),
    (r"^kyutai/stt-1b-en_fr", (True, None)),
    (r"^microsoft/VibeVoice-ASR", (True, None)),
]


def language_control(name, languages):
    """{"auto": bool | "returned", "force": "yes" | "prompt" | None} of a multilingual model, None for a model
    of one language, "?" when the table above does not have the model."""
    if languages is not None and len(languages) <= 1 and not (languages and languages[0].endswith("+")):
        return None
    for pattern, (auto, force) in LANGUAGE_CONTROL:
        if re.search(pattern, name):
            return {"auto": auto, "force": force}
    print(f"Warning: no language control for {name} (LANGUAGE_CONTROL)", file=sys.stderr)
    return "?"


def output_counts(pred_file):
    """[empty outputs (no letter or digit left once the special tokens <|...|> are removed), other
    outputs, those with an uppercase letter, those with a punctuation mark]"""
    texts = [SPECIAL_TOKEN.sub("", row.get("prediction") or "") for row in json.loads(pred_file.read_text(encoding="utf-8")).values()]
    filled = [t for t in texts if any(c.isalnum() for c in t)]
    return [len(texts) - len(filled), len(filled), sum(any(c.isupper() for c in t) for t in filled),
            sum(bool(PUNCTUATION.search(t)) for t in filled)]


def empty_note(scores):
    """Caveat on the WER of a model with more than EMPTY_WARNING of empty outputs."""
    total = sum(s.get("n") or 0 for s in scores.values() if "empty" in s)
    empty = sum(s.get("empty", 0) for s in scores.values())
    if not total or empty / total <= EMPTY_WARNING:
        return None
    return f"Empty outputs: {empty} / {total} ({100 * empty / total:.1f}%)."


def model_entry(exp, rtf_results=None, info_from=None):
    """info_from: results folder whose run of the same model (same folder name) completes the
    model_info (filled by hand for the sota benchmark only; the benchmark only writes params)."""
    meta = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
    backend, model = meta["backend"], meta["model"]
    if info_from and (info_from / exp.name / "metadata.json").exists():
        other = json.loads((info_from / exp.name / "metadata.json").read_text(encoding="utf-8")).get("model_info") or {}
        meta["model_info"] = (meta.get("model_info") or {}) | other
    # faster-whisper takes OpenAI size names ("large-v3"): show the original model.
    display = f"openai/whisper-{model}" if backend == "faster-whisper" and "/" not in model else model
    # The spelling of the Hugging Face repo when the run used another case (e.g. gemma-3n-e4b-it)
    if (hf_id := (meta.get("model_info") or {}).get("hf")) and hf_id.lower() == display.lower():
        display = hf_id
    # vLLM serves local checkpoints (model is a path) under served_model_name: show it,
    # or the Hugging Face repo of the checkpoint (model_info.hf) when there is one.
    info = meta.get("model_info") or {}
    name = meta.get("served_model_name") and (info.get("hf") or meta["served_model_name"]) or display
    # What the folder name adds after <backend>_<model> (e.g. "decoder-ctc") tells
    # apart several runs of the same model. Some backends (NeMo) write "_" as "-" there.
    variant = ""
    folder_model = meta.get("served_model_name") or model
    for prefix in {f"{backend}_{folder_model.replace('/', '-')}", f"{backend}_{re.sub('[/_]', '-', folder_model)}"}:
        if exp.name.startswith(prefix):
            variant = exp.name[len(prefix):].strip("_")
    # Streaming models run with their default chunk size have no chunk in the folder name.
    if meta.get("chunk_ms") and f"chunk{meta['chunk_ms']}ms" not in variant:
        variant = "_".join(filter(None, [variant, f"chunk{meta['chunk_ms']}ms"]))
    # License, size and languages: filled by tools/fill_model_info.py.
    hf = info.get("hf") or display
    url = f"https://huggingface.co/{hf}" if "/" in hf and not hf.startswith(("/", ".", "~")) else None
    rtfx, hardware, rtf_backend, rtf_note = speed(exp, rtf_results, model)
    scores, outputs = {}, [0, 0, 0]  # outputs: non-empty, with an uppercase letter, with a punctuation mark
    for perf_file in sorted((exp / "performances").glob("*.json")):
        perf = json.loads(perf_file.read_text(encoding="utf-8"))
        scores[perf_file.stem] = {
            metric: {k: round(perf[metric][k], 3) for k in ("wer", "sub", "del", "ins") if k in perf[metric]}
            | ({"ci95": perf[metric]["ci95"]} if "ci95" in perf[metric] else {})
            for metric in METRICS
            if metric in perf
        }
        scores[perf_file.stem]["n"] = perf.get("num_data")
        scores[perf_file.stem]["seconds"] = perf.get("duration")
        pred_file = exp / "predictions" / perf_file.name
        if pred_file.exists():
            empty, *counts = output_counts(pred_file)
            scores[perf_file.stem]["empty"] = empty
            outputs = [a + b for a, b in zip(outputs, counts)]
    return {
        "id": exp.name,
        "model": name,
        "variant": variant,
        "backend": backend,
        "url": url,
        "license": info.get("license"),
        "params": info.get("params"),
        "languages": info.get("languages"),
        "architecture": info.get("architecture"),  # Encoder | Encoder-decoder | Audio LLM
        "streaming": info.get("streaming"),
        "task": info.get("task"),  # ASR | ASR/AST | Multi
        "language_control": language_control(name, info.get("languages")),
        "rtfx": rtfx,
        "hardware": hardware,
        "rtf_backend": rtf_backend if rtf_backend != backend else None,  # set when timed with another backend
        "rtf_note": rtf_note,  # caveat on the speed measure, shown with a warning sign
        "wer_note": "\n".join(n.rstrip(".") + "." for n in (info.get("wer_note"), empty_note(scores)) if n) or None,
        # model_info's (by hand) first: e.g. a chat template that ignores the sent prompt
        "prompt": info.get("prompt") or (meta.get("prompt") if backend != "vllm" or meta.get("endpoint") == "chat" else None),
        "prompt_source": info.get("prompt_source"),
        "merge_as": info.get("merge_as"),  # one row for several models (e.g. one model per language)
        "scores": scores,
        "outputs": outputs if outputs[0] else None,  # punctuation and case of the outputs (page: shares)
    }


def merge_rows(models):
    """Models with the same model_info.merge_as in one row named after it: the scores of each on
    its test sets (e.g. one fine-tune per language), languages joined, linked to the Hugging Face
    organization, with a note."""
    merged, groups = [], {}
    for m in models:
        name = m.pop("merge_as", None)
        if not name:
            merged.append(m)
        elif name not in groups:
            groups[name] = m | {"id": name, "model": name, "variant": "", "languages": list(m["languages"] or []),
                                "url": m["url"] and m["url"].rsplit("/", 1)[0]}
            merged.append(groups[name])
        else:
            group = groups[name]
            group["scores"] |= m["scores"]
            if m.get("outputs"):
                group["outputs"] = [a + b for a, b in zip(group.get("outputs") or [0, 0, 0], m["outputs"])]
            group["languages"] += [l for l in m["languages"] or [] if l not in group["languages"]]
    for name, group in groups.items():
        note = "One model per language, each scored on the test sets of its language"
        group["wer_note"] = "\n".join(filter(None, [note + ".", group["wer_note"]]))
    return merged


def is_public(exp):
    """Local checkpoints (model given as a path) without a Hugging Face repo (model_info.hf)
    are internal: not on the page, which only shows models whose weights are public."""
    meta = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
    return not meta["model"].startswith(("/", ".", "~")) or bool((meta.get("model_info") or {}).get("hf"))


def prediction_durations(folders, stem, expected):
    """Durations (s) of the utterances of a test set, from the predictions of a run that has all
    of them (the expected number), else of the run with the most."""
    best = []
    for path in sorted(p for folder in folders for p in folder.glob(f"*/predictions/{stem}.json")):
        rows = json.loads(path.read_text(encoding="utf-8")).values()
        durations = [row["audio_duration"] for row in rows if row.get("audio_duration")]
        if len(durations) > len(best):
            best = durations
        if len(best) >= expected:
            break
    return best


def multilingual(results, french_results, rtf_results=None, include_private=False, french_extra=()):
    """Models and test sets of the Multilingual tab, None without results. include_private: also
    the local checkpoints (local preview, never published); french_extra: other results folders
    where to find their French scores (e.g. benchmarks/luciole_dev/results)."""
    if not results or not results.exists():
        return None
    models = []
    for exp in sorted(p for p in results.iterdir() if (p / "metadata.json").exists()):
        entry = model_entry(exp, rtf_results, info_from=french_results)
        if not include_private and not is_public(exp):
            continue
        # One prompt per language (`prompts`), or the language in the request body (`extra_bodies`):
        # the metadata keeps the last language's only
        meta = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("prompts"):
            entry["prompt"] = "\n".join(f"{lang}: {prompt}" for lang, prompt in meta["prompts"].items())
            entry["prompt_source"] = "model card prompt, with the language of the test set"
        elif meta.get("extra_bodies"):
            entry["prompt"] = "\n".join(f"{lang}: {json.dumps(body, ensure_ascii=False)}" for lang, body in meta["extra_bodies"].items())
            entry["prompt_source"] = "model card (vLLM chat): no text prompt, the language of the test set forced in the request"
        elif not meta.get("prompt") and entry["prompt"]:
            # Prompt of the French run (model_info): here the run gives the language of each test set,
            # which the default prompt names in place of French (e.g. vLLM's Qwen3-Omni prompt)
            languages = sorted({stem.rsplit("_", 1)[1] for stem in entry["scores"]}, key=lambda l: (
                MULTILINGUAL_LANGUAGES.index(l) if l in MULTILINGUAL_LANGUAGES else len(MULTILINGUAL_LANGUAGES), l))
            if "French" in entry["prompt"]:
                entry["prompt"] = "\n".join(f"{lang}: {entry['prompt'].replace('French', LANGUAGE_NAMES.get(lang, lang))}"
                                            for lang in languages)
                entry["prompt_source"] = f"{entry['prompt_source'] or 'default prompt'}, with the language of the test set"
            else:
                entry["prompt"] = entry["prompt_source"] = None  # not known for the other languages
        french = next((f / exp.name for f in (french_results, *french_extra) if (f / exp.name / "metadata.json").exists()),
                      french_results / exp.name)
        if (french / "metadata.json").exists():
            for stem, score in model_entry(french)["scores"].items():
                if stem in MULTILINGUAL_FRENCH:
                    entry["scores"][MULTILINGUAL_FRENCH[stem]] = score
        models.append(entry)
    # Models of a merged row (model_info.merge_as) run on French only (e.g. BuzzASR/french): their
    # French scores, from --results
    groups = {m["merge_as"] for m in models if m.get("merge_as")}
    for exp in sorted(p for p in french_results.iterdir() if (p / "metadata.json").exists()):
        if not (results / exp.name).exists():
            entry = model_entry(exp)
            if entry["merge_as"] in groups:
                entry["scores"] = {MULTILINGUAL_FRENCH[k]: v for k, v in entry["scores"].items() if k in MULTILINGUAL_FRENCH}
                models.append(entry)
    models = merge_rows(models)
    ids = {stem for m in models for stem in m["scores"]}
    order = lambda i: (MULTILINGUAL_LANGUAGES.index(i.rsplit("_", 1)[1]) if i.rsplit("_", 1)[1] in MULTILINGUAL_LANGUAGES
                       else len(MULTILINGUAL_LANGUAGES), list(MULTILINGUAL_DATASETS).index(i.rsplit("_", 1)[0])
                       if i.rsplit("_", 1)[0] in MULTILINGUAL_DATASETS else len(MULTILINGUAL_DATASETS), i)
    datasets = []
    for i in sorted(ids, key=order):
        prefix, language = i.rsplit("_", 1)
        name, domain, url = MULTILINGUAL_DATASETS.get(prefix, (prefix, "", None))
        selection = MULTILINGUAL_SELECTION.get(i, "Whole")
        utterances = max((m["scores"][i]["n"] or 0) for m in models if i in m["scores"])
        stem = next((s for s, j in MULTILINGUAL_FRENCH.items() if j == i), i)
        french = i in MULTILINGUAL_FRENCH.values()  # from the French leaderboard
        durations = prediction_durations([french_results, *french_extra] if french else [results], stem, utterances)
        datasets.append({
            "id": i, "name": name, "domain": domain, "language": language, "url": url, "selection": selection,
            "speech": SPEECH.get(prefix), "read": prefix in READ,
            "hours": round(max((m["scores"][i]["seconds"] or 0) for m in models if i in m["scores"]) / 3600, 2) or None,
            "utterances": utterances,
        } | (duration_stats(durations) if len(durations) == utterances else {}))
    return {"datasets": datasets, "models": models}


def manifest_durations(manifest):
    """Durations (s) of the utterances of each test set of the manifest, {} if there is none."""
    durations = {}
    if manifest and manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                durations.setdefault(row.get("name") or row.get("dataset"), []).append(row["duration"])
    return durations


def speed_set(manifest, rtf_results):
    """Segments the RTFx is measured on (manifest of config_rtf.yaml), per source test set,
    and the hardware of the speed runs. None without the manifest."""
    durations = manifest_durations(manifest)
    if not durations:
        return None
    rows = [
        {k: v for k, v in dataset_info(stem).items() if k not in ("language", "selection", "selection_note")} | {"durations": d}
        for stem, d in sorted(durations.items(), key=lambda kv: -len(kv[1]))
    ]
    everything = [x for d in durations.values() for x in d]
    hardware = sorted({
        meta.get("device_name") or meta.get("device")
        for meta in (json.loads(p.read_text(encoding="utf-8")) for p in rtf_results.glob("*/metadata.json"))
    } - {None}) if rtf_results and rtf_results.exists() else []
    stats = lambda d: {
        "utterances": len(d),
        "seconds": round(sum(d), 1),
        "min_seconds": round(min(d), 1),
        "max_seconds": round(max(d), 1),
        "mean_seconds": round(statistics.mean(d), 1),
        "median_seconds": round(statistics.median(d), 1),
        "histogram": histogram(d),
    }
    return {
        "sources": [{"name": r["name"], "domain": r["domain"], "url": r["url"]} | stats(r["durations"]) for r in rows],
        "total": stats(everything),
        "hardware": hardware,
    }


def histogram(durations):
    """Number of utterances per bin of HISTOGRAM_BIN seconds, ]a, b] (a segment of 30 s is in the
    bin 28-30 s, so that the histograms of the segments of 30 s at most end at 30 s)."""
    index = lambda d: max(0, math.ceil(d / HISTOGRAM_BIN) - 1)
    counts = [0] * (index(max(durations)) + 1)
    for d in durations:
        counts[index(d)] += 1
    return counts


def duration_stats(durations):
    return {
        "hours": round(sum(durations) / 3600, 2),
        "utterances": len(durations),
        "mean_seconds": round(statistics.mean(durations), 1),
        "median_seconds": round(statistics.median(durations), 1),
        "histogram": histogram(durations),
    }


def last_commit_date(*paths):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", *map(str, paths)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def contextual_page(results, model_info):
    """(HTML, script, test sets) of the Contextual Biasing tab, from the page of tools/contextual_biasing: its
    styles and body for a shadow root, its script as a function of the document, the description of its
    test sets (with the durations of the manifests next to the results). (None, None, []) without results.
    results: {language: results folder}; model_info: results folder with the model_info of the models."""
    results = {language: folder for language, folder in results.items() if folder.exists()}
    if not results:
        return None, None, []
    sys.path.insert(0, str(HERE.parent / "contextual_biasing"))
    import build_page

    data = build_page.page_data(results, model_info=model_info)
    page = build_page.page_html(data)
    style, body, script = (re.search(p, page, re.DOTALL).group(1) for p in (
        r"<style>(.*?)</style>", r"<body>(.*?)<script>", r"<script>(.*)</script>"))
    # The shadow root has no html nor body: their rules go to its host
    style = re.sub(r"(?m)^  (:root|body) \{", "  :host {", style).replace("html { overflow-y: scroll; }", "")
    style += "  :host { display: block; background: none; }\n  main { max-width: none; padding: 0; }\n  h1 { display: none; }\n"
    sets = build_page.test_sets(data)
    for d in sets:
        manifest = next((m for folder in results.values()
                         if (m := folder.parent / "manifests" / f"{d['id']}.jsonl").exists()), None)
        lines = manifest.read_text(encoding="utf-8").splitlines() if manifest else []
        durations = [json.loads(line)["duration"] for line in lines if line.strip()]
        d |= duration_stats(durations) if durations else {}
    return f"<style>{style}</style>{body}", f"function (document) {{{script}}}", sets


def build(results, manifest, output, rtf_results=None, rtf_manifest=None, cards=None, multilingual_results=None,
          include_private=False, french_extra=(), contextual_results=None):
    experiments = sorted(p for p in results.iterdir() if (p / "metadata.json").exists())
    models = [model_entry(exp, rtf_results) for exp in experiments]
    for m in models:
        m.pop("merge_as", None)  # rows merged in the Multilingual tab only
    durations = manifest_durations(manifest)
    order = list(DATASETS)
    stems = sorted({stem for m in models for stem in m["scores"]},
                   key=lambda s: (order.index(s), "") if s in DATASETS else (len(order), s.lower()))
    datasets = [
        {
            "id": stem,
            **dataset_info(stem),
            "hours": round(max((m["scores"][stem]["seconds"] or 0) for m in models if stem in m["scores"]) / 3600, 2) or None,
            "utterances": max((m["scores"][stem]["n"] or 0) for m in models if stem in m["scores"]),
        }
        | (duration_stats(durations[stem]) if durations.get(stem) else {})
        for stem in stems
    ]
    totals = {}
    for language in {d["language"] for d in datasets}:
        rows = [d for d in datasets if d["language"] == language]
        totals[language] = (
            duration_stats([x for d in rows for x in durations[d["id"]]]) if all(durations.get(d["id"]) for d in rows)
            else {"hours": round(sum(d["hours"] or 0 for d in rows), 2), "utterances": sum(d["utterances"] for d in rows)}
        )
    html, script, contextual_sets = contextual_page(contextual_results or {}, results)
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_updated": last_commit_date(*filter(None, [results, multilingual_results, *(contextual_results or {}).values()])),
        "repo": REPO_URL,
        "histogram_bin": HISTOGRAM_BIN,
        "datasets": datasets,
        "dataset_totals": totals,
        "speed_set": speed_set(rtf_manifest, rtf_results),
        "models": models,
        "multilingual": multilingual(multilingual_results, results, rtf_results, include_private, french_extra),
        "contextual_test_sets": contextual_sets,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "leaderboard.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    template = (HERE / "template.html").read_text(encoding="utf-8")
    # "</" is escaped so the data can never close the <script> tag.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    for kind in ("css", "js"):  # before the data
        template = template.replace(f"/*__COMMON_{kind.upper()}__*/", (HERE / f"common.{kind}").read_text(encoding="utf-8"))
    template = template.replace("/*__DATA__*/null", payload)
    logo = base64.b64encode((HERE / "linagora_logo.svg").read_bytes()).decode()
    template = template.replace("/*__LOGO__*/", f"data:image/svg+xml;base64,{logo}")
    if html:  # after the data of this page (the other page comes with its own)
        template = template.replace("<!--__CONTEXTUAL_PAGE__-->", html).replace("/*__CONTEXTUAL_SCRIPT__*/null", script)
    (output / "index.html").write_text(template, encoding="utf-8")
    print(f"{len(models)} models x {len(datasets)} datasets -> {output}/index.html")
    build_cards(cards, data, output, SITE_URL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/sota/results"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/sota/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("site"))
    parser.add_argument("--rtf_results", type=Path, default=Path("benchmarks/sota/results_rtf"),
                        help="Results of the speed runs (config_rtf.yaml), for the RTFx column")
    parser.add_argument("--rtf_manifest", type=Path, default=Path("benchmarks/sota/manifest_rtf.jsonl"),
                        help="Manifest of the speed runs, described in the test sets tab")
    parser.add_argument("--cards", type=Path, default=HERE / "cards.json",
                        help="Result cards to draw (SVG for model cards), see cards.py")
    parser.add_argument("--multilingual_results", type=Path, default=Path("benchmarks/sota_multilingual/results"),
                        help="Results of the Multilingual tab (benchmarks/sota_multilingual)")
    parser.add_argument("--contextual_results", nargs="*", metavar="LANG=FOLDER",
                        default=[f"{lang}=benchmarks/contextual_biasing/results_{lang}" for lang in ("fr", "en")],
                        help="Results of the Contextual Biasing tab (benchmarks/contextual_biasing)")
    parser.add_argument("--include_private", action="store_true",
                        help="Multilingual tab: also the local checkpoints (local preview only, not for the published page)")
    parser.add_argument("--french_results_extra", type=Path, nargs="*", default=[],
                        help="Multilingual tab: other folders for the French scores of models not in --results")
    args = parser.parse_args()
    build(args.results, args.manifest, args.output, args.rtf_results, args.rtf_manifest, args.cards, args.multilingual_results,
          args.include_private, args.french_results_extra,
          {lang: Path(folder) for lang, folder in (item.split("=", 1) for item in args.contextual_results)})
