#!/usr/bin/env python3
"""Generate WER / RTF (and RAM-VRAM when available) plots from benchmark outputs.

Takes one or more benchmark output folders (each containing per-model subfolders
with metadata.json + performances/ + predictions/, as written by benchmarker.py)
and produces:
  - a micro-average WER heatmap table (model x dataset), as reported by the benchmark
  - with --plot-macro-wer, a second table of the macro-average WER (mean of per-file
    WER), recomputed from the predictions, with each file's WER capped by --cap-macro-wer
  - an RTFx bar chart (speed, higher = faster)
  - RAM / VRAM bar charts, only when monitoring.json is present

If --output is given the figures are saved there as PNGs; otherwise they are
shown interactively with plt.show().

Usage:
    python generate_plots.py FOLDER [FOLDER ...] [--output OUT] [--casepunc]
                             [--plot-macro-wer] [--cap-macro-wer 100]

Examples:
    # show interactively
    python generate_plots.py benchmarks/linto_stt_fr_fastconformer/local_bench
    # save to a folder, from several benchmark dirs at once
    python generate_plots.py local_bench local_bench_rtf --output my_plots
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from asr_benchmark.visualization.table import plot_wer_table, prepare_model_name

# Short dataset labels (same as generate_plots.ipynb).
DATASET_RENAME = {
    "YouTubeFr_split6": "YouTube",
    "TEDX_fr": "TEDx",
    "MLS_Facebook_french": "MLS",
    "Voxpopuli": "VoxPopuli",
}
# Preferred column order for the WER table; unknown datasets are appended after.
DATASET_ORDER = ["CommonVoice", "MLS", "SUMM-RE", "VoxPopuli", "TEDx", "YouTube"]


def model_label(meta):
    """Readable, unique-ish label for one experiment's metadata."""
    model = meta["model"]
    if meta.get("backend") == "faster-whisper" and "whisper" not in model:
        model = f"whisper-{model}"
    label = prepare_model_name(model)
    if meta.get("decoder"):
        label += f"\n{meta['decoder'].upper()}"
    # Distinguishing extras (precision/dtype/accurate) keep otherwise-identical
    # model names apart; only appended when present.
    extras = [str(meta[k]) for k in ("precision", "dtype") if meta.get(k)]
    if meta.get("accurate") in ("accurate", "greedy"):
        extras.append(meta["accurate"])
    if extras:
        label += f"\n({', '.join(extras)})"
    return label


def collect(folders, casepunc=False, with_texts=False):
    """Read every experiment folder into a flat list of per-(model,dataset) rows.

    with_texts=True also keeps each file's reference/prediction text (needed for the
    macro-average WER, which is recomputed per file).
    """
    wer_key = "wer" if casepunc else "wer_nocasepunc"
    rows = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            print(f"  ! skipping (not a folder): {folder}")
            continue
        for exp in sorted(folder.iterdir()):
            meta_file = exp / "metadata.json"
            perf_dir = exp / "performances"
            if not meta_file.exists() or not perf_dir.is_dir():
                continue
            meta = json.loads(meta_file.read_text())
            label = model_label(meta)
            device = {"cuda": "GPU", "cpu": "CPU"}.get(meta.get("device"), meta.get("device"))

            # RAM / VRAM (optional) from a single monitoring.json per experiment.
            ram = vram = None
            mon_file = exp / "monitoring.json"
            if mon_file.exists():
                mon = json.loads(mon_file.read_text())
                if mon.get("ram_usage"):
                    ram = round(max(mon["ram_usage"]), 2)
                if mon.get("vram_usage"):
                    vram = round(max(mon["vram_usage"]), 2)

            for perf_file in sorted(perf_dir.iterdir()):
                perf = json.loads(perf_file.read_text())
                if wer_key not in perf:
                    continue
                # RTF list + canonical dataset name (+ optional texts) from predictions.
                rtfs = []
                refs, texts_pred = [], []
                dataset = perf_file.stem
                n_files = perf.get("num_data", 0)
                pred_file = exp / "predictions" / perf_file.name
                if pred_file.exists():
                    preds = json.loads(pred_file.read_text())
                    n_files = len(preds)
                    for rec in preds.values():
                        rtf = rec.get("rtf")
                        if rtf is None and rec.get("audio_duration"):
                            rtf = rec.get("prediction_duration", 0) / rec["audio_duration"]
                        if rtf:
                            rtfs.append(rtf)
                        if with_texts:
                            refs.append(rec.get("text", ""))
                            texts_pred.append(rec.get("prediction", ""))
                    # 'dataset' here is the short, suffix-stripped name (e.g. "MLS_Facebook_french").
                    first = next(iter(preds.values()), None)
                    if first and first.get("dataset"):
                        dataset = first["dataset"]
                dataset = DATASET_RENAME.get(dataset, dataset)
                rows.append({
                    "model": label,
                    "folder": exp.name,
                    "dataset": dataset,
                    "device": device,
                    "wer": perf[wer_key]["wer"],
                    "count": perf[wer_key].get("count", 0),
                    "n_files": n_files,
                    "rtfs": rtfs,
                    "refs": refs,
                    "preds": texts_pred,
                    "ram": ram,
                    "vram": vram,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["model"] = disambiguate_labels(df)
    return df


def disambiguate_labels(df):
    """Give each experiment folder its own display label.

    Rows are keyed by folder (one benchmark run), so runs of the same model with
    different settings never collapse into one line. When several folders share the
    same model name, the tokens that differ between their folder names are appended
    in parentheses so they can be told apart in the plots.
    """
    from collections import defaultdict

    folder_base = df.groupby("folder")["model"].first().to_dict()
    by_base = defaultdict(list)
    for folder, base in folder_base.items():
        by_base[base].append(folder)

    folder_display = {}
    for base, folders in by_base.items():
        if len(folders) == 1:
            folder_display[folders[0]] = base
            continue
        # Tokens (split on "_") shared by every colliding folder are the common
        # context; what remains per folder is its distinguishing setting.
        token_sets = {f: f.split("_") for f in folders}
        common = set.intersection(*(set(t) for t in token_sets.values()))
        for f in folders:
            variable = [t for t in token_sets[f] if t not in common]
            paren = ", ".join(variable) if variable else "default"
            folder_display[f] = f"{base}\n({paren})"

    return df["folder"].map(folder_display)


def order_datasets(columns):
    ordered = [d for d in DATASET_ORDER if d in columns]
    return ordered + [c for c in columns if c not in ordered]


def save_or_show(fig, output, name):
    # When no output folder is given, leave the figure open so main() can show
    # every figure at once with a single plt.show() at the end.
    if output:
        path = Path(output) / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


def plot_wer(df, output):
    piv = df.pivot_table(index="model", columns="dataset", values="wer", aggfunc="mean")
    piv = piv[order_datasets(piv.columns)]
    # Micro-averaged WER across datasets (weighted by word count) as an extra column.
    wsum = df.pivot_table(index="model", columns="dataset", values="wer", aggfunc="mean")
    cnt = df.pivot_table(index="model", columns="dataset", values="count", aggfunc="mean")
    piv["Average"] = (wsum * cnt).sum(axis=1) / cnt.sum(axis=1)
    piv = piv.sort_values("Average")
    # show=False: the figure is left open and shown once at the end (see main()).
    plot_wer_table(
        piv,
        output_filename=str(Path(output) / "wer_table.png") if output else None,
        show=False,
        y_label="micro-avg WER (%)",
        color_lims=(0, 50),
    )
    if output:
        print(f"  saved {Path(output) / 'wer_table.png'}")


def compute_macro_wer(df, cap, casepunc):
    """Add a 'macro_wer' column: the mean of per-file WER (each file recomputed with the
    same normalization as the benchmark, then optionally capped at `cap` percent).

    cap <= 0 means no cap. A single tqdm bar spans every file across all models, since
    per-file re-scoring (text normalization + alignment) is what makes this slow.
    """
    import ssak.utils.wer as sw
    from ssak.utils.wer import compute_wer

    normalization = "" if casepunc else "fr+"
    replacements = {"euh": "", "hum": ""}
    total = int(df["refs"].map(len).sum())

    def per_file_wer(ref, pred):
        res = compute_wer(
            [ref], [pred], normalization=normalization, use_percents=True,
            replacements_ref=replacements, replacements_pred=replacements,
        )
        return res["wer"] if res.get("count", 0) > 0 else None

    macro = []
    saved_tqdm = sw.tqdm  # silence compute_wer's own inner progress bars
    sw.tqdm = type("_Q", (), {"tqdm": staticmethod(lambda x, *a, **k: x)})
    try:
        with tqdm(total=total, desc="Computing macro-avg WER", unit="file") as bar:
            for refs, preds in zip(df["refs"], df["preds"]):
                vals = []
                for ref, pred in zip(refs, preds):
                    w = per_file_wer(ref, pred)
                    if w is not None:
                        vals.append(min(w, cap) if cap and cap > 0 else w)
                    bar.update(1)
                macro.append(sum(vals) / len(vals) if vals else float("nan"))
    finally:
        sw.tqdm = saved_tqdm

    df = df.copy()
    df["macro_wer"] = macro
    return df


def plot_macro_wer(df, output, cap):
    piv = df.pivot_table(index="model", columns="dataset", values="macro_wer", aggfunc="mean")
    piv = piv[order_datasets(piv.columns)]
    # Overall macro = per-file WER averaged across all files (weighted by file count).
    macro = df.pivot_table(index="model", columns="dataset", values="macro_wer", aggfunc="mean")
    nf = df.pivot_table(index="model", columns="dataset", values="n_files", aggfunc="mean")
    piv["Average"] = (macro * nf).sum(axis=1) / nf.sum(axis=1)
    piv = piv.sort_values("Average")
    plot_wer_table(
        piv,
        output_filename=str(Path(output) / "macro_wer_table.png") if output else None,
        show=False,
        y_label="macro-avg WER (%)",
        color_lims=(0, 50),
    )
    cap_note = f"capped at {cap:g}%/file" if cap and cap > 0 else "uncapped"
    if output:
        print(f"  saved {Path(output) / 'macro_wer_table.png'} ({cap_note})")
    else:
        print(f"  macro-avg WER table ({cap_note})")


def plot_rtf(df, output):
    # Aggregate per (model, device): mean RTF over all files -> RTFx = 1 / mean.
    recs = []
    for (model, device), g in df.groupby(["model", "device"]):
        arr = np.array([r for lst in g["rtfs"] for r in lst], dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            continue
        recs.append({"model": model, "device": device, "rtfx": 1.0 / arr.mean()})
    if not recs:
        print("  ! no RTF data found, skipping RTF plot")
        return
    rtf = pd.DataFrame(recs).sort_values("rtfx")
    devices = sorted(rtf["device"].unique())
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(rtf) + 1)))
    if len(devices) > 1:
        import seaborn as sns
        sns.barplot(data=rtf, y="model", x="rtfx", hue="device", ax=ax)
    else:
        ax.barh(rtf["model"], rtf["rtfx"], color="tab:blue")
        for y, v in enumerate(rtf["rtfx"]):
            ax.text(v, y, f" {v:.1f}", va="center", fontsize=11)
    ax.set_xlabel("RTFx (higher = faster than real time)")
    ax.set_ylabel("")
    ax.set_title("Inverse real-time factor")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, output, "rtf.png")


def plot_resource(df, output, column, title, fname):
    sub = df[df[column].notna()][["model", column]].drop_duplicates("model")
    if sub.empty:
        print(f"  ! no {title} data (no monitoring.json), skipping")
        return
    sub = sub.sort_values(column)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(sub) + 1)))
    ax.barh(sub["model"], sub[column], color="tab:green")
    for y, v in enumerate(sub[column]):
        ax.text(v, y, f" {v:.0f}", va="center", fontsize=11)
    ax.set_xlabel(f"{title} (MB)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, output, fname)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folders", nargs="+", help="benchmark output folder(s)")
    parser.add_argument("-o", "--output", default=None,
                        help="folder to save PNGs into (default: show interactively)")
    parser.add_argument("--casepunc", action="store_true",
                        help="use case+punctuation WER instead of normalized WER")
    parser.add_argument("--plot-macro-wer", action="store_true",
                        help="also plot the macro-average WER (mean of per-file WER), "
                             "recomputed from the predictions")
    parser.add_argument("--cap-macro-wer", type=float, default=100.0,
                        help="cap each file's WER at this percent for the macro average "
                             "(default 100; 0 or negative = no cap)")
    args = parser.parse_args()

    df = collect(args.folders, casepunc=args.casepunc, with_texts=args.plot_macro_wer)
    if df.empty:
        raise SystemExit("No benchmark results found in the given folder(s).")
    print(f"Loaded {df['model'].nunique()} models across {df['dataset'].nunique()} datasets.")

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    plot_wer(df, args.output)
    if args.plot_macro_wer:
        df = compute_macro_wer(df, cap=args.cap_macro_wer, casepunc=args.casepunc)
        plot_macro_wer(df, args.output, cap=args.cap_macro_wer)
    plot_rtf(df, args.output)
    plot_resource(df, args.output, "vram", "VRAM usage", "vram.png")
    plot_resource(df, args.output, "ram", "RAM usage", "ram.png")

    # No output folder: show all figures at once.
    if not args.output:
        plt.show()


if __name__ == "__main__":
    main()
