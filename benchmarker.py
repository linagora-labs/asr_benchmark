import argparse
import yaml
from asr_benchmark.benchmark import launch_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking tool for STT models")
    parser.add_argument(
        "config", type=str, default="config.yaml", help="Config file for the benchmark"
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


    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    input_manifest = (
        args.input_manifest
        if args.input_manifest is not None
        else config.pop("input_manifest", None)
    )
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
    )
