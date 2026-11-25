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


def model_label(metadata):
    """Short name of a model: name of its checkpoint, plus its decoding when several are compared."""
    label = Path(str(metadata.get("served_model_name") or metadata["model"])).name
    if metadata.get("backend") == "faster-whisper":
        label = f"whisper-{label} ({'beam 5' if metadata.get('accurate') else 'greedy'})"
    return label


def model_url(metadata):
    """Model card on Hugging Face, None for a local checkpoint."""
    name = str(metadata.get("served_model_name") or metadata["model"])
    if metadata.get("backend") == "faster-whisper":
        name = f"openai/whisper-{name}"
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


def language_data(results, excluded=(), excluded_datasets=(), excluded_settings=(), excluded_runs=(), excluded_models=()):
    runs = {name: run for name, run in report.load_runs(results).items()
            if not any(re.search(pattern, name) for pattern in excluded_models)}
    rows = {}
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
    return {"lists": lists, "datasets": datasets, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", required=True, metavar="LANG=FOLDER",
                        help="Results folder (output_folder of the config) of each language")
    parser.add_argument("--output", required=True, help="Folder of the page")
    parser.add_argument("--exclude_lists", nargs="*", default=["oracle", "oracle_upper", "local"],
                        help="Lists left out of the page (default: the oracle lists, only the terms spoken in the "
                             "utterance, not a realistic setting: PxSLU oracle, Contextual Earnings-22 local)")
    parser.add_argument("--exclude_datasets", nargs="*", default=["LibriSpeech_test-clean", "PxSLU_dev"],
                        help="Test sets left out of the page (default: LibriSpeech test-clean, left out of "
                             "the benchmark: few errors on the rare words to correct; PxSLU dev, only for "
                             "tuning, e.g. the NeMo boosting weight)")
    parser.add_argument("--exclude_settings", nargs="*", default=["spaces"],
                        help="Backend settings left out of the page (default: the Whisper runs with the "
                             "hotwords separated by spaces). Of the NeMo boosting weights, only the one of "
                             "linto-stt for the length of the list is kept (see nemo_alpha)")
    parser.add_argument("--exclude_models", nargs="*",
                        default=["Luciole-.*-Audio", "faster-whisper_large-v3(_cb|$)", "fr:Luciole"],
                        help="Models left out of the page: regular expressions on the folder names, for one "
                             "language when they start with \"<language>:\" (default: the Luciole models without "
                             "hotwords, Luciole in French (PxSLU is in its training data), and Whisper large-v3, "
                             "not better than large-v3-turbo with beam search)")
    parser.add_argument("--exclude_runs", nargs="*", default=["nemo_nvidia-parakeet-tdt-0.6b-v3_cb-1000-alpha1.0"],
                        help="Runs with a list left out of the page (default: parakeet with the LibriSpeech "
                             "1000 lists, the only NeMo run with them, done before they were left out)")
    args = parser.parse_args()

    data = {}
    for item in args.results:
        language, _, folder = item.partition("=")
        data[language] = language_data(folder, set(args.exclude_lists), set(args.exclude_datasets),
                                       set(args.exclude_settings), set(args.exclude_runs),
                                       [p.removeprefix(f"{language}:") for p in args.exclude_models
                                        if not re.match(r"[a-z]{2}:", p) or p.startswith(f"{language}:")])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    template = (HERE / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (output / "index.html").write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print(f"Page written to {output / 'index.html'}")


if __name__ == "__main__":
    main()
