#!/usr/bin/env python3
"""Generate WER / RTF (and RAM-VRAM when available) plots from benchmark outputs.

Takes one or more benchmark output folders (each containing per-model subfolders
with metadata.json + performances/ + predictions/, as written by benchmarker.py)
and produces:
  - a micro-average WER heatmap table (model x dataset), as reported by the benchmark
  - with --plot-macro-wer, a second table of the macro-average WER (mean of per-file
    WER), recomputed from the predictions, with each file's WER capped by --cap-macro-wer.
    Per-file re-scoring runs across --macro-workers processes and the (uncapped) per-file
    WER is cached on disk, so re-runs -- including with a different --cap-macro-wer -- are
    near-instant.
  - an RTFx bar chart (speed, higher = faster)
  - RAM / VRAM bar charts, only when monitoring.json is present

If --output is given the figures are saved there as PNGs; otherwise they are
shown interactively with plt.show().

Usage:
    python generate_plots.py FOLDER [FOLDER ...] [--output OUT] [--casepunc]
                             [--plot-macro-wer] [--cap-macro-wer 100]
                             [--macro-workers N] [--macro-cache PATH | --no-macro-cache]

Examples:
    # show interactively
    python generate_plots.py benchmarks/linto_stt_fr_fastconformer/local_bench
    # save to a folder, from several benchmark dirs at once
    python generate_plots.py local_bench local_bench_rtf --output my_plots
"""
import argparse
import json
import multiprocessing as mp
import os
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


def append_detail(label, detail):
    """Append a parenthetical detail to a label, keeping every detail in a SINGLE
    parenthesis at the end of the last line -- so a label never grows past two lines
    (model names namespaced with "/" already occupy two lines).

    If the last line already ends with a "(...)" group, the detail is merged into it
    (comma-separated); on a one-line label the parenthesis starts a second line;
    otherwise it is placed at the end of the existing last line.
    """
    lines = label.split("\n")
    last = lines[-1]
    if last.endswith(")"):
        lines[-1] = f"{last[:-1]}, {detail})"
    elif len(lines) == 1:
        lines.append(f"({detail})")
    else:
        lines[-1] = f"{last} ({detail})"
    return "\n".join(lines)


def model_label(meta):
    """Readable, unique-ish label for one experiment's metadata."""
    model = meta["model"]
    if meta.get("backend") == "faster-whisper" and "whisper" not in model:
        model = f"whisper-{model}"
    label = prepare_model_name(model)
    # Everything that distinguishes otherwise-identical model names -- the decoder
    # and any precision/dtype/accurate extras -- goes into one parenthesis at the end
    # of the last line, so the label stays (at most) two lines.
    details = []
    if meta.get("decoder"):
        details.append(meta["decoder"].upper())
    details += [str(meta[k]) for k in ("precision", "dtype") if meta.get(k)]
    if meta.get("accurate") in ("accurate", "greedy"):
        details.append(meta["accurate"])
    if details:
        label = append_detail(label, ", ".join(details))
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
            # Skip deprecated / hidden-by-convention experiment folders.
            if exp.name.startswith("_") or "deprecated" in exp.name.lower():
                continue
            meta_file = exp / "metadata.json"
            perf_dir = exp / "performances"
            if not meta_file.exists() or not perf_dir.is_dir():
                continue
            meta = json.loads(meta_file.read_text())
            label = model_label(meta)
            # Runs served through vLLM (folder name prefixed "vllm_") are marked so
            # they are distinguishable from the same model run via other backends;
            # the marker joins the parenthesis at the end of the second line.
            if exp.name.startswith("vllm_"):
                label = append_detail(label, "vLLM")
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
            # append_detail keeps the label to two lines: the "(...)" goes at the end
            # of the second line when the base already has one (e.g. a decoder).
            folder_display[f] = append_detail(base, paren)

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


# Per-worker state + task function for the macro-WER process pool (must be top-level so
# they can be pickled to worker processes). The worker returns the UNCAPPED per-file WER
# (or None for empty references) -- the cap is applied later, so the cache is cap-agnostic.
_MACRO = {}


def _macro_init(normalization):
    _MACRO["norm"] = normalization
    import ssak.utils.wer as sw  # silence compute_wer's own inner progress bars in each worker
    sw.tqdm = type("_Q", (), {"tqdm": staticmethod(lambda x, *a, **k: x)})


def _macro_score(pair):
    ref, pred = pair
    from ssak.utils.wer import compute_wer
    res = compute_wer(
        [ref], [pred], normalization=_MACRO["norm"], use_percents=True,
        replacements_ref={"euh": "", "hum": ""}, replacements_pred={"euh": "", "hum": ""},
    )
    return res["wer"] if res.get("count", 0) > 0 else None


def _macro_key(normalization, ref, pred):
    import hashlib
    h = hashlib.sha1()
    h.update(normalization.encode("utf-8"))
    for s in ("\x00", ref, "\x00", pred):
        h.update(s.encode("utf-8"))
    return h.hexdigest()


def compute_macro_wer(df, cap, casepunc, workers=None, cache_path=None):
    """Add a 'macro_wer' column: the mean of per-file WER (each file recomputed with the
    same normalization as the benchmark, then optionally capped at `cap` percent).

    cap <= 0 means no cap. Per-file re-scoring (text normalization + alignment) is the slow
    part, so: only files not already in the on-disk cache are (re)computed, across a process
    pool. The cache stores the *uncapped* per-file WER keyed by a hash of (normalization,
    reference, prediction) -- so it is safe across models/datasets and re-used when the cap
    changes; a changed prediction gets a new key and is recomputed.
    """
    df = df.reset_index(drop=True)
    normalization = "" if casepunc else "fr+"

    # Flatten every file into a (ref, pred) task + its content key, remembering its row.
    pairs, keys, row_of = [], [], []
    for ri, (refs, preds) in enumerate(zip(df["refs"], df["preds"])):
        for ref, pred in zip(refs, preds):
            pairs.append((ref, pred))
            keys.append(_macro_key(normalization, ref, pred))
            row_of.append(ri)

    # Load cache and figure out what actually needs computing (deduped by content key).
    cache = {}
    if cache_path and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text())
        except (ValueError, OSError):
            cache = {}
    todo = {}
    for key, pair in zip(keys, pairs):
        if key not in cache:
            todo[key] = pair
    n_cached = len(pairs) - sum(1 for k in keys if k in todo)
    print(f"  macro-WER: {len(pairs)} files, {n_cached} cached, {len(todo)} to compute")

    if todo:
        if workers is None:
            workers = min(os.cpu_count() or 4, 16)
        workers = max(1, min(workers, len(todo)))
        tkeys = list(todo)
        tpairs = [todo[k] for k in tkeys]
        desc = f"Computing macro-avg WER ({workers} worker{'s' if workers > 1 else ''})"
        if workers == 1:
            _macro_init(normalization)
            results = map(_macro_score, tpairs)
            pool = None
        else:
            pool = mp.Pool(workers, initializer=_macro_init, initargs=(normalization,))
            results = pool.imap(_macro_score, tpairs, chunksize=64)
        try:
            for key, val in zip(tkeys, tqdm(results, total=len(tpairs), desc=desc, unit="file")):
                cache[key] = val
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        if cache_path:
            try:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                Path(cache_path).write_text(json.dumps(cache))
                print(f"  macro-WER cache -> {cache_path} ({len(cache)} entries)")
            except OSError as e:
                print(f"  ! could not write macro-WER cache ({e})")

    # Aggregate per row, applying the cap here (kept out of the cache).
    sums = [0.0] * len(df)
    cnts = [0] * len(df)
    for key, ri in zip(keys, row_of):
        val = cache.get(key)
        if val is not None:
            sums[ri] += min(val, cap) if cap and cap > 0 else val
            cnts[ri] += 1

    df = df.copy()
    df["macro_wer"] = [sums[i] / cnts[i] if cnts[i] else float("nan") for i in range(len(df))]
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
    parser.add_argument("--macro-workers", type=int, default=None,
                        help="processes for the macro-WER computation (default: CPU count, "
                             "capped at 16)")
    parser.add_argument("--macro-cache", default=None,
                        help="path to the macro-WER cache file (default: "
                             "<first folder>/.macro_wer_cache.json)")
    parser.add_argument("--no-macro-cache", action="store_true",
                        help="disable reading/writing the macro-WER cache")
    args = parser.parse_args()

    df = collect(args.folders, casepunc=args.casepunc, with_texts=args.plot_macro_wer)
    if df.empty:
        raise SystemExit("No benchmark results found in the given folder(s).")
    print(f"Loaded {df['model'].nunique()} models across {df['dataset'].nunique()} datasets.")

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    plot_wer(df, args.output)
    if args.plot_macro_wer:
        if args.no_macro_cache:
            cache_path = None
        else:
            cache_path = args.macro_cache or str(Path(args.folders[0]) / ".macro_wer_cache.json")
        df = compute_macro_wer(df, cap=args.cap_macro_wer, casepunc=args.casepunc,
                               workers=args.macro_workers, cache_path=cache_path)
        plot_macro_wer(df, args.output, cap=args.cap_macro_wer)
    plot_rtf(df, args.output)
    plot_resource(df, args.output, "vram", "VRAM usage", "vram.png")
    plot_resource(df, args.output, "ram", "RAM usage", "ram.png")

    # No output folder: show all figures at once.
    if not args.output:
        plt.show()


if __name__ == "__main__":
    main()
