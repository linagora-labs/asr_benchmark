"""HTML page of the contextual biasing benchmark: for each language, test set and list,
the measures of each model without and with the list (charts and tables).

    python tools/contextual_biasing/build_page.py \
        --results fr=benchmarks/contextual_biasing/results_fr en=benchmarks/contextual_biasing/results_en \
        --output benchmarks/contextual_biasing/page

Standard library only. The page is a single file (OUTPUT/index.html) built from
template.html with the measures of report.py.
"""

import argparse
import json
import re
from pathlib import Path

import report

HERE = Path(__file__).resolve().parent
MODES = ["nocasepunc", "cased"]

# Descriptions shown on the page (and in the Test sets tab of the leaderboard)
TEST_SETS = {
    "PxSLU_test": {"name": "PxSLU", "language": "fr", "domain": "Dictated drug prescriptions", "terms": "Drug names",
                   "url": "https://doi.org/10.5281/zenodo.6482586"},
    "ContextualEarnings22_test": {"name": "Contextual Earnings-22", "language": "en",
                                  "domain": "Earnings calls (conversations, many accents)",
                                  "terms": "Names of persons, companies and products",
                                  "url": "https://huggingface.co/datasets/argmaxinc/contextual-earnings22"},
    "LibriSpeech_test-other": {"name": "LibriSpeech test-other", "language": "en", "domain": "Read audiobooks",
                               "terms": "Rare words", "url": "https://www.openslr.org/12/"},
}
# Label and content of the lists (the keys are in the hover texts)
LISTS = {
    "pxslu25": ("25 names", "For each utterance: the ones of the 389 names that are spoken in it, and others of them drawn at random"),
    "pxslu25_ext": ("25 names (ext.)", "The same, with the 427 names"),
    "pxslu25_upper": ("25 names (caps)", "The same 25 names, in capitals: does the model follow the case of the vocabulary?"),
    "pxslu": ("389 names", "Drug names of the corpus: mostly brands, and some molecules (active substances)"),
    "pxslu_ext": ("427 names", "The 389 names, and 38 others spoken in the test set: 33 molecules (e.g. paracétamol) and 5 brands"),
    "pxslu_upper": ("389 names (caps)", "The same 389 names, in capitals: does the model follow the case of the vocabulary?"),
    "vidal": ("Vidal", "Drug names of the Vidal database, and the 389 names (private list: its terms of use forbid to share it)"),
    "bdpm": ("BDPM", "Drug names of the French public drug database, and the 389 names"),
    "theriaque": ("Thériaque", "Drug names of the Thériaque database, and the 389 names (private list: its terms of use forbid to share it)"),
    "global": ("Call keywords", "For each clip: the keywords of the whole call, of which 15.5% on average are spoken in the clip (median: 6.5%)"),
    "100": ("100 distractors", "For each utterance: its rare words (1.8 on average, the terms to recognize), and 100 other rare words drawn at random"),
    "500": ("500 distractors", "For each utterance: its rare words (1.8 on average, the terms to recognize), and 500 other rare words drawn at random"),
    "1000": ("1,000 distractors", "For each utterance: its rare words (1.8 on average, the terms to recognize), and 1,000 other rare words drawn at random (about 2 fewer on average in the released lists, probably duplicates removed)"),
    "2000": ("2,000 distractors", "For each utterance: its rare words (1.8 on average, the terms to recognize), and 2,000 other rare words drawn at random (about 7 fewer on average in the released lists, probably duplicates removed)"),
}


def model_label(metadata):
    """Short name of a model: name of its checkpoint, plus its decoding when several are compared."""
    label = Path(str(metadata.get("served_model_name") or metadata["model"])).name
    if metadata.get("backend") == "faster-whisper":
        label = f"whisper-{label} ({'beam 5' if metadata.get('accurate') else 'greedy'})"
    return label


def model_name(metadata):
    """Full name of a model (as in the leaderboard): its Hugging Face repo, or the name of a local checkpoint."""
    name = str(metadata.get("served_model_name") or metadata["model"])
    if metadata.get("backend") == "faster-whisper" and "/" not in name:
        name = f"openai/whisper-{name}"
    return name


def model_variant(metadata, folder):
    """What tells apart several runs of a model: its decoding."""
    if metadata.get("backend") == "faster-whisper":
        return "beam 5" if metadata.get("accurate") else "greedy"
    decoder = re.search(r"_(decoder-[a-z]+)", folder)
    return decoder.group(1) if decoder else ""


# How the list is given to the model, by backend (the others: in their prompt)
# Techniques to give the list to the model: "Context", an input made for context, that the model is trained
# to use as background knowledge; "Prompt", the text prompt of the model (made for instructions, or for the
# previous text in Whisper); "Boosting", the scores of the terms raised at decoding time (no training)
TECHNIQUES = {
    "faster-whisper": ("Prompt", "Prompt: the terms put before the previous text in the decoder prompt (faster-whisper hotwords, "
                                 "at most 223 tokens), a slot that Whisper was not trained to use for this"),
    "nemo": ("Boosting", "Boosting: the scores of the terms raised at decoding time (NeMo GPU phrase boosting), without training"),
    "vibevoice-asr": ("Context", "Context: extra text of the request, next to the audio, that VibeVoice-ASR is trained to use"),
}
# Models of the vLLM backend with an input made for context
CONTEXT_MODELS = {
    "Qwen3-ASR": "Context: the context of the request (system turn, as the context argument of the Qwen3-ASR SDK), "
                 "that Qwen3-ASR is trained to use as background knowledge",
}


def technique(metadata):
    """(Technique, details) of the way the list is given to the model in a run with a list."""
    template = metadata.get("contextual_biasing_template")
    for model, details in CONTEXT_MODELS.items():
        if model in model_name(metadata):
            return "Context", details + (f"; text given: {template}" if template else "")
    return TECHNIQUES.get(metadata.get("backend"), ("Prompt", "Prompt: the terms added to the instruction prompt of the model"
                                                               + (f": {template}" if template else "")))


def model_infos(folder):
    """{full name: model_info (license, size, architecture...)} of the models of a results folder."""
    infos = {}
    for path in sorted(Path(folder).glob("*/metadata.json")) if folder else []:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("model_info"):
            infos[model_name(metadata)] = metadata["model_info"]
    return infos


def model_url(metadata):
    """Model card on Hugging Face, None for a local checkpoint."""
    name = model_name(metadata)
    return f"https://huggingface.co/{name}" if re.fullmatch(r"[\w.-]+/[\w.-]+", name) else None


# Weight of the NeMo boosting by number of terms, as in linto-stt (hotwords_boosting_alpha)
def nemo_alpha(list_size):
    return 2.0 if list_size < 100 else 1.0


def is_other_alpha(row):
    """Is it a run with another boosting weight than the one of nemo_alpha?"""
    alpha = re.fullmatch(r"alpha ([\d.]+)", setting_label(row))
    return bool(alpha) and float(alpha.group(1)) != nemo_alpha(row["list_size"])


def setting_label(row):
    """Backend settings of the run with the list (e.g. "alpha 1.0"), from its folder name."""
    if not row.get("run_cb"):
        return ""
    tag = report.contextual_biasing.folder_tag({report.contextual_biasing.MANIFEST_KEY: row["list"]})
    suffix = row["run_cb"].removeprefix(row["model"]).removeprefix(tag)
    return " ".join(re.sub(r"^([a-z]+)(\d)", r"\1 \2", part) for part in suffix.split("-") if part)


def language_data(results, infos, excluded=(), excluded_datasets=(), excluded_settings=(), excluded_runs=(), excluded_models=()):
    runs = {name: run for name, run in report.load_runs(results).items()
            if not any(re.search(pattern, name) for pattern in excluded_models)}
    rows, models = {}, {}
    for mode in MODES:
        rows[mode] = []
        for row in report.collect(runs, mode):
            if row["list"] in excluded or row["dataset"] in excluded_datasets:
                continue
            if row.get("run_cb") in excluded_runs:
                # Only the measures without list are kept
                row = {k: v for k, v in row.items() if k != "run_cb" and not k.endswith("_cb")}
            metadata = runs[row["model"]][0] if row["model"] in runs else runs[row["run_cb"]][0]
            setting = setting_label(row)
            if setting in excluded_settings:
                continue
            name = model_name(metadata)
            model = models.setdefault(model_label(metadata), {
                "name": name, "variant": model_variant(metadata, row["model"]),
                **{key: infos.get(name, {}).get(key) for key in ("params", "license", "architecture", "streaming")},
            })
            if row.get("run_cb"):
                model["technique"] = technique(runs[row["run_cb"]][0])
            if setting.startswith("alpha"):
                if is_other_alpha(row):
                    continue
                setting = ""
            rows[mode].append(dict(row, label=model_label(metadata), backend=metadata.get("backend"),
                                   setting=setting, url=model_url(metadata)))
        # The rows without list that are repeated by a run with the list (the two weights of NeMo)
        with_run = {(r["dataset"], r["list"], r["model"]) for r in rows[mode] if r.get("run_cb")}
        rows[mode] = [r for r in rows[mode] if r.get("run_cb") or (r["dataset"], r["list"], r["model"]) not in with_run]
    # Lists from the smallest to the largest (mean number of terms per utterance)
    sizes = {}
    for row in rows[MODES[0]]:
        sizes.setdefault(row["list"], []).append(row["list_size"])
    lists = sorted(sizes, key=lambda name: (sum(sizes[name]) / len(sizes[name]), name))
    # Test sets first (the first one is shown by default), then dev sets
    datasets = sorted({row["dataset"] for row in rows[MODES[0]]}, key=lambda name: ("dev" in name.lower(), name))
    return {"lists": lists, "datasets": datasets, "rows": rows, "models": models}


# Left out of the page (defaults of the options)
EXCLUDED = {
    # The oracle lists, only the terms spoken in the utterance, not a realistic setting (PxSLU oracle,
    # Contextual Earnings-22 local)
    "lists": ["oracle", "oracle_upper", "local"],
    # LibriSpeech test-clean, left out of the benchmark (few errors on the rare words to correct); PxSLU dev,
    # only for tuning (e.g. the NeMo boosting weight)
    "datasets": ["LibriSpeech_test-clean", "PxSLU_dev"],
    # The Whisper runs with the hotwords separated by spaces
    "settings": ["spaces"],
    # The Luciole models without hotwords, Luciole in French (PxSLU is in its training data), and Whisper
    # large-v3, not better than large-v3-turbo with beam search
    "models": ["Luciole-.*-Audio", "faster-whisper_large-v3(_cb|$)", "fr:Luciole"],
    "runs": [],
}


def page_data(results, excluded=EXCLUDED, model_info=None):
    """Data of the page. results: {language: results folder}; model_info: results folder whose metadata
    have the license, size and architecture of the models."""
    data, infos = {}, model_infos(model_info)
    for language, folder in results.items():
        models = [p.removeprefix(f"{language}:") for p in excluded["models"]
                  if not re.match(r"[a-z]{2}:", p) or p.startswith(f"{language}:")]
        data[language] = language_data(folder, infos, set(excluded["lists"]), set(excluded["datasets"]),
                                       set(excluded["settings"]), set(excluded["runs"]), models)
    return data


def test_sets(data):
    """Test sets of the page with their lists (descriptions, mean number of terms), in the order of the page."""
    sets = []
    for lang in data.values():
        for dataset in lang["datasets"]:
            rows = [row for row in lang["rows"][MODES[0]] if row["dataset"] == dataset]
            used = {row["list"] for row in rows if row.get("run_cb")}
            lists = [{"id": name, "label": LISTS.get(name, (name, ""))[0], "content": LISTS.get(name, (name, ""))[1],
                      "size": next(row["list_size"] for row in rows if row["list"] == name)}
                     for name in lang["lists"] if name in used]
            sets.append({"id": dataset, "name": dataset, "language": "", "domain": "", "terms": "", "url": None}
                        | TEST_SETS.get(dataset, {}) | {"lists": lists})
    return sets


def page_html(data):
    template = (HERE / "template.html").read_text(encoding="utf-8")
    for kind in ("css", "js"):  # before the data
        common = (HERE.parent / "leaderboard" / f"common.{kind}").read_text(encoding="utf-8")
        template = template.replace(f"/*__COMMON_{kind.upper()}__*/", common)
    # "</" is escaped so the data can never close the <script> tag
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    info = json.dumps({"test_sets": TEST_SETS, "lists": LISTS}, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("/*__DATA__*/null", payload).replace("/*__INFO__*/null", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, metavar="LANG=FOLDER",
                        help="Results folder (output_folder of the config) of each language")
    parser.add_argument("--output", required=True, help="Folder of the page")
    for name, what in (("lists", "Lists"), ("datasets", "Test sets"), ("runs", "Runs with a list"),
                       ("settings", "Backend settings (of the NeMo boosting weights, only the one of linto-stt for the "
                                    "length of the list is kept, see nemo_alpha)"),
                       ("models", "Models (regular expressions on the folder names, for one language when they start "
                                  "with \"<language>:\")")):
        parser.add_argument(f"--exclude_{name}", nargs="*", default=EXCLUDED[name],
                            help=f"{what} left out of the page (default: {' '.join(EXCLUDED[name])}, see EXCLUDED)")
    parser.add_argument("--model_info", default="benchmarks/sota/results",
                        help="Results folder whose metadata have the license, size and architecture of the models")
    args = parser.parse_args()

    results = dict(item.split("=", 1) for item in args.results)
    data = page_data(results, {name: getattr(args, f"exclude_{name}") for name in EXCLUDED}, args.model_info)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(page_html(data), encoding="utf-8")
    print(f"Page written to {output / 'index.html'}")


if __name__ == "__main__":
    main()
