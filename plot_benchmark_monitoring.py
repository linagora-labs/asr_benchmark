#!/usr/bin/env python3
"""Plot processing time, RAM usage, and VRAM usage from benchmark results."""

import argparse
import json
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
    args = parser.parse_args()

    show_legend = len(args.folders) > 1

    # Load all datasets
    datasets = []
    for folder in args.folders:
        audio_durations, prediction_durations, file_ids, monitoring = load_data(folder)
        # Sort by audio duration
        sorted_indices = sorted(range(len(audio_durations)), key=lambda i: audio_durations[i])
        datasets.append({
            "label": folder.name,
            "audio_durations": [audio_durations[i] for i in sorted_indices],
            "prediction_durations": [prediction_durations[i] for i in sorted_indices],
            "monitoring": monitoring,
        })

    colors = plt.cm.tab10.colors

    # Figure 1: Processing time vs audio duration
    fig_proc, ax = plt.subplots()
    for i, ds in enumerate(datasets):
        ax.plot(ds["audio_durations"], ds["prediction_durations"], "o-", color=colors[i % len(colors)], label=ds["label"])
    ax.set_xlabel("Audio duration (s)")
    ax.set_ylabel("Processing time (s)")
    ax.set_title("Processing time vs. audio duration")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend()

    # Figure 2: Max RAM usage vs audio duration
    fig_ram, ax = plt.subplots()
    for i, ds in enumerate(datasets):
        max_ram = get_max_usages(ds["monitoring"], "ram_usage")
        ax.plot(ds["audio_durations"], max_ram, "o-", color=colors[i % len(colors)], label=ds["label"])
    ax.set_xlabel("Audio duration (s)")
    ax.set_ylabel("Max RAM usage (GB)")
    ax.set_title("Max RAM usage vs. audio duration")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend()

    # Figure 3: Max VRAM usage vs audio duration
    fig_vram, ax = plt.subplots()
    for i, ds in enumerate(datasets):
        max_vram = get_max_usages(ds["monitoring"], "vram_usage")
        ax.plot(ds["audio_durations"], max_vram, "o-", color=colors[i % len(colors)], label=ds["label"])
    ax.set_xlabel("Audio duration (s)")
    ax.set_ylabel("Max VRAM usage (GB)")
    ax.set_title("Max VRAM usage vs. audio duration")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend()

    # Figure 4: VRAM detail over time
    fig_detail, ax = plt.subplots()
    for i, ds in enumerate(datasets):
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
    ax.set_ylabel("VRAM usage (GB)")
    ax.set_title("VRAM usage over time (per processed file)")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend()

    figures = {
        "processing_time": fig_proc,
        "ram_usage": fig_ram,
        "vram_usage": fig_vram,
        "vram_detail": fig_detail,
    }

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(args.output / f"{name}.png", dpi=150, bbox_inches="tight")
            print(f"Saved {args.output / f'{name}.png'}")
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
