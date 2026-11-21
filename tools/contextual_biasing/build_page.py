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
    if metadata.get("backend") == "faster-whisper" and metadata.get("accurate"):
        label += " (beam 5)"
    return label


def setting_label(row):
    """Backend settings of the run with the list (e.g. "alpha 1.0"), from its folder name."""
    if not row.get("run_cb"):
        return ""
    tag = report.contextual_biasing.folder_tag({report.contextual_biasing.MANIFEST_KEY: row["list"]})
    suffix = row["run_cb"].removeprefix(row["model"]).removeprefix(tag)
    return " ".join(re.sub(r"^([a-z]+)(\d)", r"\1 \2", part) for part in suffix.split("-") if part)


def language_data(results, excluded=(), excluded_datasets=(), excluded_settings=()):
    runs = report.load_runs(results)
    rows = {}
    for mode in MODES:
        rows[mode] = []
        for row in report.collect(runs, mode):
            if row["list"] in excluded or row["dataset"] in excluded_datasets:
                continue
            metadata = runs[row["model"]][0] if row["model"] in runs else runs[row["run_cb"]][0]
            if setting_label(row) in excluded_settings:
                continue
            rows[mode].append(dict(row, label=model_label(metadata), backend=metadata.get("backend"),
                                   setting=setting_label(row)))
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
    parser.add_argument("--exclude_lists", nargs="*", default=["oracle", "oracle_upper"],
                        help="Lists left out of the page (default: the PxSLU per utterance oracle lists, "
                             "not a realistic setting)")
    parser.add_argument("--exclude_datasets", nargs="*", default=["LibriSpeech_test-clean"],
                        help="Test sets left out of the page (default: LibriSpeech test-clean, left out of "
                             "the benchmark: few errors on the rare words to correct)")
    parser.add_argument("--exclude_settings", nargs="*", default=["alpha 0.5", "alpha 4.0"],
                        help="Backend settings left out of the page (default: the NeMo boosting weights "
                             "of the tuning on PxSLU dev other than 1.0 and 2.0)")
    args = parser.parse_args()

    data = {}
    for item in args.results:
        language, _, folder = item.partition("=")
        data[language] = language_data(folder, set(args.exclude_lists), set(args.exclude_datasets),
                                       set(args.exclude_settings))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    template = (HERE / "template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (output / "index.html").write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print(f"Page written to {output / 'index.html'}")


if __name__ == "__main__":
    main()
