import os
import time
import json
from tqdm import tqdm
from itertools import product

import asr_benchmark.utils.benchmark as utils
from asr_benchmark.benchmark.backend_to_model import get_model
from asr_benchmark.utils import logger
from ssak.utils.wer import compute_wer
from ssak.utils.monitoring import Monitoring

REPLACEMENTS_WER = {"euh": "", "hum": ""}
PATH_TO_WARMUP_FILE = "examples/bonjour.wav"

def check_if_benched(output_folder, input_file, config, debug):
    data = []
    if os.path.exists(os.path.join(output_folder, "error.log")):
        with open(os.path.join(output_folder, "error.log"), "r") as f:
            txt = f.read()
            if txt.startswith("CUDA out of memory"):
                logger.error(f"Skipping, CUDA out of memory error")
                return data
    all_data = utils.get_data(input_file)
    benched = dict()
    compute_rtf = config.get("compute_rtf", True)
    for dataset in os.listdir(os.path.join(output_folder, "performances")):
        dataset = os.path.splitext(dataset)[0]
        with open(os.path.join(output_folder, "performances", f"{dataset}.json"), "r") as f:
            benched[dataset] = json.load(f)
    if debug:
        all_data = all_data[:1]
    for row in all_data:
        basename = os.path.splitext(os.path.basename(row["audio_filepath"]))[0]
        dataset = row['name']
        if debug or basename not in benched.get(dataset, {}):
            data.append(row)
        elif compute_rtf and "rtf" not in benched[dataset][basename]:
            data.append(row)
    return data, benched

def make_perf_file(row):
    perfs = {
        "audio_filepath": row["audio_filepath"],
        "id": row['id'],
        "audio_duration": round(row.get("duration") if row.get("duration") else utils.get_audio_duration(row["audio_filepath"]), 3),
        "audio_offset": round(row.get("offset", 0.0), 3),
    }
    if "name" in row:
        perfs["full_dataset"] = row["name"]
        perfs["dataset"] = row["name"].replace("_max30","").replace("_nocasepunc","").replace("_test","").replace("_devtest","").replace("_dev","")
    elif "dataset" in row:
        perfs["dataset"] = row["dataset"]
    return perfs

def make_perf_dataset(data):
    row = data[0]
    perfs = {
        "full_dataset": row["name"],
        "dataset": row["name"].replace("_max30","").replace("_nocasepunc","").replace("_test","").replace("_devtest","").replace("_dev",""),
        "number_of_files": len(data),
    }
    return perfs

def transcribe_with_rtf(model, data, output_folder, config):
    progress_bar = tqdm(data, desc="Loading...".ljust(45))
    progress_bar.set_description(f"Warmup...".ljust(45))
    monitor = Monitoring(
        output_folder, device=config.get("device", 0), plot_monitoring=config.get("plot_monitoring", True)
    )
    monitor.start(
        steps=[
            os.path.splitext(os.path.basename(row['id']))[0]
            for row in data
        ]
    )
    model.config['device_name'] = monitor.get_device_name()
    for row in progress_bar:
        perfs = make_perf_file(row)
        audio_file, dataset, audio_duration = perfs["audio_filepath"], perfs["dataset"], perfs["audio_duration"]
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        progress_bar.set_description(f"Transcribing {basename}".ljust(45))
        try:
            audio = model.load_audio(audio_file, start=row.get("offset", 0.0), duration=audio_duration)
            start = time.time()
            output = model.transcribe(audio)
        except Exception as e:
            logger.error(f"while transcribing {audio_file}")
            monitor.stop(error=True)
            raise e
        end = time.time()
        output['prediction_duration'] = round(end - start, 5)
        output['rtf'] = round(output['prediction_duration'] / audio_duration, 5)
        yield output, row
        monitor.next()
        progress_bar.set_description(f"Finished {model.get_folder_name()}".ljust(45))
    monitor.stop()

def transcribe_fast(model, data, output_folder, config):
    outputs = model.transcribe_batch(data)
    for i, row in enumerate(tqdm(data, desc="Computing WER...".ljust(45))):     # keep details for each file 
        yield outputs[i], row

def process_result(iterator, bench_results, output_folder, config):
    for output, row in iterator:
        prediction = output['text'].strip().encode('utf-8').decode('utf-8')
        output.pop('text')
        perfs = make_perf_file(row)
        perfs.update(output)
        audio_file, dataset = row["audio_filepath"], row["name"]
        basename = os.path.splitext(os.path.basename(audio_file))[0]
        if config.get("save_predictions", True):
            os.makedirs(os.path.join(output_folder, "predictions", dataset), exist_ok=True)
            with open(
                os.path.join(output_folder, "predictions", dataset, basename + ".txt"), "w", encoding="utf-8"
            ) as f:
                f.write(prediction)
        ref = row.get("text", None)
        if ref:
            alignment = None
            if config.get("save_alignments", True):
                os.makedirs(os.path.join(output_folder, "wer", dataset), exist_ok=True)
                alignment = os.path.join(output_folder, "wer", dataset, basename + ".txt")
            wer_score = compute_wer(
                [ref],
                [prediction],
                normalization="fr",
                use_percents=True,
                alignment=alignment,
                replacements_pred=REPLACEMENTS_WER,
                replacements_ref=REPLACEMENTS_WER,
            )
            if alignment:
                del wer_score['alignment']
                del wer_score['raw_alignement']
            perfs["wer"] = wer_score
        if dataset not in bench_results:
            bench_results[dataset] = dict()
        bench_results[dataset][basename] = perfs
        with open(
            os.path.join(output_folder, "performances", dataset+".json"), "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(bench_results[dataset], indent=2, ensure_ascii=False))

def bench_model(config, input_manifest, output_folder, debug=False):
    data, bench_results = check_if_benched(
        output_folder, input_manifest, config,  debug
    )
    if len(data) == 0:
        logger.info(f"Skipping, it has already been benched")
        return
    logger.info(
        f"Benching {len(data)} files, loaded {len(utils.get_data(input_manifest))-len(data)})"
    )
    model = get_model(config)
    model.load()
    audio = model.load_audio(PATH_TO_WARMUP_FILE)
    _ = model.transcribe(audio)
    if config.get('compute_rtf', True):
        iterator = transcribe_with_rtf(model, data, output_folder, config)
    else:
        iterator = transcribe_fast(model, data, output_folder, config)
    process_result(iterator, bench_results, output_folder, config)
    model.cleanup()
    with open(os.path.join(output_folder, "metadata.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(model.get_metadata(), indent=2, ensure_ascii=False))


def make_configs(configs):
    new_configs = []
    for config in configs["benchmarks"]:
        keys, values = zip(*config.items())
        all_combinations = [
            dict(zip(keys, prod))
            for prod in product(*[v if isinstance(v, list) else [v] for v in values])
        ]
        new_configs.extend(all_combinations)
    return new_configs


def launch_benchmark(
    configs,
    input_manifest,
    output_folder,
    compute_rtf=True,
    debug=False,
    skip_errors=False,
    save_predictions=False,
    save_alignments=False,
    compute_latency=False,
):
    plot_monitoring = configs.pop("plot_monitoring", True)
    configs = make_configs(configs)
    progress_bar = tqdm(configs, desc="Backends and models...".ljust(45))
    logger.info(f"Starting benchmarking with {len(configs)} configurations")
    for config in progress_bar:
        model = get_model(config)
        config['input_manifest'] = config.get('input_manifest', input_manifest)
        config['compute_rtf'] = config.get('compute_rtf', compute_rtf)
        config['save_predictions'] = config.get('save_predictions', save_predictions)
        config['save_alignments'] = config.get('save_alignments', save_alignments)
        config['compute_latency'] = config.get('compute_latency', compute_latency)
        config['plot_monitoring'] = config.get('plot_monitoring', plot_monitoring)
        bench_id = model.get_folder_name()
        logger.info(
            f"Benching {bench_id} (progress {progress_bar.n}/{progress_bar.total})"
        )
        progress_bar.set_description(f"Using {bench_id}".ljust(45))
        config_output = os.path.join(output_folder, bench_id)
        os.makedirs(config_output, exist_ok=True)
        os.makedirs(os.path.join(config_output, "performances"), exist_ok=True)
        try:
            start = time.time()
            bench_model(
                config, config["input_manifest"], config_output, debug
            )
            end = time.time()
            logger.info(f"Finished benching {bench_id} after {end-start:.0f}sec")
        except Exception as e:
            if skip_errors:
                end = time.time()
                logger.error(
                    f"while benching {bench_id} (failed after {start-end:.0f}sec):"
                )
                logger.error(f"{e}")
                if config.get("device", "cuda"):
                    import traceback
                    import torch

                    if debug or not isinstance(e, torch.cuda.OutOfMemoryError):
                        logger.info(traceback.format_exc())
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        with open(os.path.join(config_output, "error.log"), "w") as f:
                            f.write(f"{e}")
                    torch.cuda.empty_cache()
                if len(
                    os.listdir(os.path.join(config_output, "predictions"))
                ) == 0 and not os.path.exists(os.path.join(config_output, "error.log")):
                    os.rmdir(os.path.join(config_output, "predictions"))
                    os.rmdir(os.path.join(config_output, "performances"))
                    os.rmdir(os.path.join(config_output, "wer"))
                    os.rmdir(config_output)
                    logger.error(f"Benched folder is empty, removing it")
                else:
                    logger.error(f"Benched folder is not empty")
                logger.error(f"Skipping to next configuration")
            else:
                raise e