#!/usr/bin/env python3
"""Plot processing time, RAM usage, and VRAM usage from benchmark results."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_data(result_folder):
    """Load prediction files and monitoring data from a benchmark result folder."""
    predictions_dir = result_folder / "predictions"
    monitoring_file = result_folder / "monitoring.json"

    # Load per-audio predictions
    audio_durations = []
    prediction_durations = []
    file_ids = []
    for pred_file in sorted(predictions_dir.glob("*.json")):
        with open(pred_file) as f:
            data = json.load(f)
        for file_id, entry in data.items():
            file_ids.append(file_id)
            audio_durations.append(entry["audio_duration"])
            prediction_durations.append(entry["prediction_duration"])

    # Load monitoring data
    with open(monitoring_file) as f:
        monitoring = json.load(f)

    return audio_durations, prediction_durations, file_ids, monitoring


def get_max_usages(monitoring, usage_key):
    """Compute max RAM or VRAM usage per step."""
    time_points = monitoring["time_points"]
    usage = monitoring[usage_key]
    steps_end = monitoring["steps_end"]

    max_usages = []
    for i, end_time in enumerate(steps_end):
        start_time = steps_end[i - 1] if i > 0 else 0.0
        max_val = max(
            u for t, u in zip(time_points, usage)
            if start_time <= t <= end_time
        )
        max_usages.append(max_val)
    return max_usages


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark VRAM/RAM results")
    parser.add_argument("folders", type=Path, nargs="+", help="Result folder(s) containing predictions/ and monitoring.json")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output folder for saving figures")
    parser.add_argument("--legend", type=str, nargs="+", default=None, help="Custom legend labels for each folder (in order)")
    parser.add_argument("--complete", action="store_true", help="Whether to plot RAM usage and detailed VRAM over time (in addition to processing time and max VRAM)")
    parser.add_argument("--title", type=str, default=None, help="Overall figure title (also used as output filename prefix)")
    parser.add_argument("--num_cols", type=int, default=1, help="Number of columns; folders are split equally across columns")
    parser.add_argument("--same_scale", action="store_true", help="Use the same Y-axis scale across columns")
    args = parser.parse_args()

    show_legend = len(args.folders) > 1 or args.legend is not None
    if args.legend:
        assert len(args.legend) == len(args.folders), "Number of legend labels must match number of folders"

    # Load all datasets
    datasets = []
    for i, folder in enumerate(args.folders):
        label = args.legend[i].replace("_", " ") if args.legend and i < len(args.legend) else folder.name
        if not folder.exists():
            print(f"WARNING: folder does not exist, skipping: {folder}")
            datasets.append({"label": f"{label} (MISSING)", "missing": True})
            continue
        if not (folder / "predictions").exists() or not (folder / "monitoring.json").exists():
            print(f"WARNING: missing predictions/ or monitoring.json, skipping: {folder}")
            datasets.append({"label": f"{label} (MISSING)", "missing": True})
            continue
        audio_durations, prediction_durations, file_ids, monitoring = load_data(folder)
        # Sort by audio duration
        sorted_indices = sorted(range(len(audio_durations)), key=lambda i: audio_durations[i])
        datasets.append({
            "label": label,
            "missing": False,
            "audio_durations": [audio_durations[i] for i in sorted_indices],
            "prediction_durations": [prediction_durations[i] for i in sorted_indices],
            "monitoring": monitoring,
        })

    colors = plt.cm.tab10.colors
    num_cols = args.num_cols

    # Split datasets into columns (as evenly as possible)
    chunk_size = math.ceil(len(datasets) / num_cols)
    columns = [datasets[i:i + chunk_size] for i in range(0, len(datasets), chunk_size)]

    # Build combined figure with subplots
    nrows = 3 if args.complete else 2  # processing time + (RAM if complete) + VRAM
    fig, axes = plt.subplots(nrows=nrows, ncols=num_cols, figsize=(8 * num_cols, 4 * nrows),
                             sharex="all" if args.same_scale else "col",
                             sharey="row" if args.same_scale else False, squeeze=False)

    for col, col_datasets in enumerate(columns):
        row = 0

        # Processing time vs audio duration
        ax = axes[row, col]
        for i, ds in enumerate(col_datasets):
            if ds["missing"]:
                ax.plot([], [], " ", label=ds["label"])
            else:
                ax.plot(ds["audio_durations"], ds["prediction_durations"], "o-", color=colors[i % len(colors)], label=ds["label"])
        if col == 0:
            ax.set_ylabel("Processing time (s)")
        ax.set_title("Processing time vs. audio duration")
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend()
        row += 1

        if args.complete:
            # Max RAM usage vs audio duration
            ax = axes[row, col]
            for i, ds in enumerate(col_datasets):
                if ds["missing"]:
                    ax.plot([], [], " ", label=ds["label"])
                else:
                    max_ram = get_max_usages(ds["monitoring"], "ram_usage")
                    ax.plot(ds["audio_durations"], max_ram, "o-", color=colors[i % len(colors)], label=ds["label"])
            if col == 0:
                ax.set_ylabel("Max RAM usage (GB)")
            ax.set_title("Max RAM usage vs. audio duration")
            ax.grid(True, alpha=0.3)
            if show_legend:
                ax.legend()
            row += 1

        # Max VRAM usage vs audio duration
        ax = axes[row, col]
        for i, ds in enumerate(col_datasets):
            if ds["missing"]:
                ax.plot([], [], " ", label=ds["label"])
            else:
                max_vram = get_max_usages(ds["monitoring"], "vram_usage")
                color = colors[i % len(colors)]
                ax.plot(ds["audio_durations"], max_vram, "o-", color=color, label=ds["label"])
                peak_idx = max(range(len(max_vram)), key=lambda k: max_vram[k])
                peak_x = ds["audio_durations"][peak_idx]
                peak_y = max_vram[peak_idx]
                ax.annotate(
                    f"{peak_y:.2f}",
                    xy=(peak_x, peak_y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold",
                )
        ax.set_xlabel("Audio duration (s)")
        if col == 0:
            ax.set_ylabel("Max VRAM usage (GB)")
        ax.set_title("Max VRAM usage vs. audio duration")
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend()

    if args.title:
        fig.suptitle(args.title, fontsize=14, fontweight="bold")

    fig.tight_layout()

    if args.complete:
        # Separate figure: VRAM detail over time
        fig_detail, axes_detail = plt.subplots(nrows=1, ncols=num_cols, figsize=(8 * num_cols, 4),
                                               squeeze=False)
        for col, col_datasets in enumerate(columns):
            ax = axes_detail[0, col]
            for i, ds in enumerate(col_datasets):
                if ds["missing"]:
                    ax.plot([], [], " ", label=ds["label"])
                    continue
                mon = ds["monitoring"]
                color = colors[i % len(colors)]
                ax.plot(mon["time_points"], mon["vram_usage"], color=color, label=ds["label"])
                for end_time, step_name in zip(mon["steps_end"], mon["steps"]):
                    ax.axvline(x=end_time, color=color, linestyle="--", alpha=0.4)
                    ax.text(
                        end_time, max(mon["vram_usage"]), step_name,
                        rotation=90, va="top", ha="right", fontsize=6, alpha=0.7, color=color,
                    )
            ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("VRAM usage (GB)")
            ax.set_title("VRAM usage over time (per processed file)")
            ax.grid(True, alpha=0.3)
            if show_legend:
                ax.legend()
        fig_detail.tight_layout()

    prefix = f"{args.title}_" if args.title else ""

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        filepath = args.output / f"{prefix}monitoring.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved {filepath}")
        if args.complete:
            filepath = args.output / f"{prefix}vram_detail.png"
            fig_detail.savefig(filepath, dpi=150, bbox_inches="tight")
            print(f"Saved {filepath}")
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
