import argparse
import atexit
import json
import logging
import os
import tempfile
from collections import Counter
from pathlib import Path
import yaml
from asr_benchmark.benchmark import launch_benchmark
from asr_benchmark.utils import setup_logging
from asr_benchmark.utils.benchmark import select_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking tool for STT models")
    parser.add_argument(
        "config", type=str, default="config.yaml", help="Config file for the benchmark", nargs="+"
    )
    parser.add_argument(
        "--input_manifest", type=str, default=None, help="Input manifest to be processed"
    )
    parser.add_argument(
        "--output_folder", type=str, default=None, help="Output folder to be written"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode, transcribing only 2 files per benchmarks",
    )
    parser.add_argument(
        "--not_compute_rtf",
        action="store_true",
        default=None,
        help="Do not compute RTF",
    )
    parser.add_argument(
        "--not_save_alignments",
        action="store_true",
        default=None,
        help="",
    )
    parser.add_argument(
        "--not_save_predictions",
        action="store_true",
        default=None,
        help="",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Log file (appended). Default: logs/<config name>_<date>.log",
    )

    args = parser.parse_args()
    log_file = setup_logging(args.log_file, name=Path(args.config[0]).stem)
    print(f"Logs: {log_file}")
    for config_file in args.config:
        config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        input_manifest = (
            args.input_manifest
            if args.input_manifest is not None
            else config.pop("input_manifest", None)
        )
        datasets = config.pop("datasets", None)
        durations = {k: config.pop(k) for k in ("min_duration", "max_duration") if k in config}
        if input_manifest is None and datasets:
            rows = select_datasets(datasets, **durations)
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", prefix="manifest_", delete=False, encoding="utf-8") as f:
                f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            input_manifest = f.name
            atexit.register(os.remove, input_manifest)
            for name, n in Counter(row["name"] for row in rows).items():
                logging.info(f"{name}: {n} segments selected")
        output_folder = (
            args.output_folder
            if args.output_folder is not None
            else config.pop("output_folder")
        )
        
        compute_rtf = (
            not args.not_compute_rtf
            if args.not_compute_rtf is not None
            else config.pop("compute_rtf", True)
        )
        
        save_alignments = (
            not args.not_save_alignments
            if args.not_save_alignments is not None
            else config.pop("save_alignments", True)
        )
        
        save_predictions = (
            not args.not_save_predictions
            if args.not_save_predictions is not None
            else config.pop("save_predictions", True)
        )
        
        compute_latency = config.pop("compute_latency", False)
        
        input_audios_paths = config.pop("input_audios_paths", "")
        
        launch_benchmark(
            config,
            input_manifest,
            output_folder,
            compute_rtf,
            args.debug,
            skip_errors=config.pop("skip_errors", False),
            save_alignments=save_alignments,
            save_predictions=save_predictions,
            compute_latency=compute_latency,
            input_audios_paths=input_audios_paths
        )
