from pathlib import Path
import re
import time
import json
import matplotlib
matplotlib.use("Agg")
import torch
from tqdm import tqdm
from itertools import product

import asr_benchmark.utils.benchmark as utils
from asr_benchmark.benchmark.backend_to_model import get_model
from asr_benchmark.utils import logger
from ssak.utils.wer import compute_wer
from ssak.utils.monitoring import Monitoring

REPLACEMENTS_WER = {"euh": "", "hum": ""}
PATH_TO_WARMUP_FILE = "examples/bonjour.wav"

# Typographic punctuation that models emit but references usually write in ASCII
# (or the reverse). Mapped to the ASCII equivalent before tokenizing, otherwise
# e.g. "c\u2019est" and "c'est" yield different tokens and every French elision
# counts as an error in the punctuated wer/cer.
UNICODE_PUNCTUATION = {
    "\u2019": "'",   # right single quotation mark, the usual ASR apostrophe
    "\u2018": "'",   # left single quotation mark
    "\u02bc": "'",   # modifier letter apostrophe
    "\u2032": "'",   # prime
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u00ab": '"',   # left-pointing double angle quotation mark
    "\u00bb": '"',   # right-pointing double angle quotation mark
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2212": "-",   # minus sign
    "\u2026": "...", # horizontal ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u202f": " ",   # narrow no-break space (French thin space before ; : ! ?)
}

_UNICODE_PUNCTUATION_RE = re.compile("|".join(map(re.escape, UNICODE_PUNCTUATION)))


def normalize_punctuation(text):
    """
    Map typographic punctuation to its ASCII equivalent.

    Example: "c\u2019est \u2014 oui\u2026" -> "c'est - oui..."
    """
    return _UNICODE_PUNCTUATION_RE.sub(lambda m: UNICODE_PUNCTUATION[m.group()], text)


def separate_punctuation(text):
    """
    Example: "Hello,world!" -> "Hello , world !"
    """
    text = normalize_punctuation(text)
    text = re.sub(r'([.,!?;:()\"\'\-])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def check_if_benched(output_folder, input_file, config, debug):
    data = dict()
    output_path = Path(output_folder)
    error_log = output_path / "error.log"
    if error_log.exists():
        with open(error_log, "r") as f:
            txt = f.read()
            if txt.startswith("CUDA out of memory"):
                logger.error(f"Skipping, CUDA out of memory error")
                return data
    all_data = utils.get_data(input_file, config['input_audios_paths'])
    benched = dict()
    compute_rtf = config.get("compute_rtf", True)

    predictions_dir = output_path / "predictions"
    for dataset_file in predictions_dir.iterdir():
        dataset = dataset_file.stem
        with open(dataset_file, "r") as f:
            benched[dataset] = json.load(f)
    if debug:
        all_data = all_data[:1]
    for row in all_data:
        dataset, id = row['name'], row['id']
        if dataset not in data:
            data[dataset] = []
        if debug or id not in benched.get(dataset, {}):
            data[dataset].append(row)
        elif compute_rtf and "rtf" not in benched[dataset][id]:
            data[dataset].append(row)
    data = {k: v for k, v in data.items() if v}
    for dataset in data:
        data[dataset] = sorted(data[dataset], key=lambda x: x['id'])
    return data, benched

def make_perf_file(row):
    if "name" in row:
        full_dataset = row["name"]
        dataset = full_dataset.replace("_max30","").replace("_nocasepunc","").replace("_test","").replace("_devtest","").replace("_dev","")
    elif "dataset" in row:
        dataset = row["dataset"]
        full_dataset = dataset
    perfs = {
        "audio_filepath": row["audio_filepath"],
        "id": row['id'],
        "audio_duration": round(row.get("duration") if row.get("duration") else utils.get_audio_duration(row["audio_filepath"]), 3),
        "audio_offset": round(row.get("offset", 0.0), 3),
        "dataset": dataset,
        "full_dataset": full_dataset,
        "text": row["text"],
    }

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
            Path(row['id']).stem
            for row in data
        ]
    )
    model.config['device_name'] = monitor.get_device_name()
    if model.config['device_name'] == 'cpu':
        torch.set_num_threads(model.config.get("num_threads", 4))
    for i, row in enumerate(progress_bar):
        perfs = make_perf_file(row)
        audio_file, dataset, audio_duration = perfs["audio_filepath"], perfs["dataset"], perfs["audio_duration"]
        basename = Path(audio_file).stem
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
        yield i, output, row
        monitor.next()
        progress_bar.set_description(f"Finished {model.get_folder_name()}".ljust(45))
    monitor.stop()

def transcribe_with_rtf_concurrent(model, data, output_folder, config):
    """RTF path for backends that process a batch concurrently (e.g. vLLM with
    concurrency>1). Hardware is monitored over the whole batch (so VRAM/RAM are captured
    -- device-wide, which includes a separate server process), and each file is given an
    amortized, throughput-consistent time: rtf = wall_time / total_audio for every file,
    so 1/mean(rtf) = total_audio/wall_time = the real concurrent throughput, and
    prediction_duration = rtf * audio_duration (these sum to the wall time). The true
    per-request latency (if the backend measured it) is kept as 'latency'.
    """
    monitor = Monitoring(
        output_folder, device=config.get("device", 0), plot_monitoring=config.get("plot_monitoring", True)
    )
    monitor.start(steps=[Path(row['id']).stem for row in data])
    model.config['device_name'] = monitor.get_device_name()
    if model.config['device_name'] == 'cpu':
        torch.set_num_threads(model.config.get("num_threads", 4))
    durations = [make_perf_file(row)["audio_duration"] for row in data]
    try:
        start = time.time()
        outputs = model.transcribe_batch(data)
        wall = time.time() - start
    except Exception as e:
        logger.error("while transcribing (concurrent batch)")
        monitor.stop(error=True)
        raise e
    monitor.stop()

    total_audio = sum(durations) or 1e-9
    rtf_const = wall / total_audio
    throughput_rtfx = round(1.0 / rtf_const, 3) if rtf_const > 0 else None
    logger.info(
        f"Concurrent RTF: throughput={throughput_rtfx}x realtime "
        f"({len(data)} files, {total_audio:.0f}s audio in {wall:.1f}s, "
        f"concurrency={config.get('concurrency', 1)})"
    )
    for i, (output, row, dur) in enumerate(zip(outputs, data, durations)):
        if "prediction_duration" in output:  # the real per-request latency, kept for reference
            output["latency"] = output["prediction_duration"]
        output["prediction_duration"] = round(rtf_const * dur, 5)
        output["rtf"] = round(rtf_const, 5)
        output["throughput_rtfx"] = throughput_rtfx
        yield i, output, row

def transcribe_fast(model, data, output_folder, config):
    outputs = model.transcribe_batch(data)
    for i, row in enumerate(data):
        yield i, outputs[i], row

def process_result(iterator, bench_result, output_folder, config, save_interval=None):
    output_path = Path(output_folder)
    def write_results(results, dataset_name):
        with open(
            output_path / "predictions" / (dataset_name + ".json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    for i, output, row in iterator:
        output['prediction'] = output['text'].strip().encode('utf-8').decode('utf-8')
        output.pop('text')
        perfs = make_perf_file(row)
        perfs.update(output)
        id, dataset = row['id'], row["name"]
        if config.get("save_predictions", False):
            (output_path / "detailed_predictions" / dataset).mkdir(parents=True, exist_ok=True)
            with open(
                output_path / "detailed_predictions" / dataset / (id + ".txt"), "w", encoding="utf-8"
            ) as f:
                f.write(output['prediction'])
        bench_result[id] = perfs
        if save_interval and i%save_interval==0:
            write_results(bench_result, dataset)
    write_results(bench_result, dataset)

def process_wer(output_folder, config):
    output_path = Path(output_folder)
    predictions_dir = output_path / "predictions"
    for dataset_file in tqdm(list(predictions_dir.iterdir()), desc="Computing WER"):
        dataset = dataset_file.stem
        perf_file = output_path / "performances" / f"{dataset}.json"
        if perf_file.exists():
            continue
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not list(data.values())[0].get("text", False):
            logger.info(f"No reference for {dataset}, skipping WER computation")
            continue
        predictions = [data[id]["prediction"] for id in data]
        references = [data[id]["text"] for id in data]
        results = dict(num_data=len(predictions), duration=sum([data[id]["audio_duration"] for id in data]))

        modes = ["wer_nocasepunc", "cer_nocasepunc"]
        if any(re.search( r"[^\w\s'-]", ref) for ref in references):
            modes = ["wer", "cer", "wer_nocasepunc", "cer_nocasepunc"]

        for key in modes:
            alignment = None
            if config.get("save_alignments", True):
                alignment_dir = output_path / "alignments" / key
                alignment_dir.mkdir(parents=True, exist_ok=True)
                alignment = str(alignment_dir / (dataset + ".txt"))
            if "wer" in modes:
                references = [separate_punctuation(ref) for ref in references]
                predictions = [separate_punctuation(pred) for pred in predictions]
            wer_score = compute_wer(
                references,
                predictions,
                normalization=f"{config.get('language', 'fr')}+" if "nocasepunc" in key else "",
                character_level="cer" in key,
                use_percents=True,
                alignment=alignment,
                replacements_pred=REPLACEMENTS_WER,
                replacements_ref=REPLACEMENTS_WER,
            )
            if alignment:
                del wer_score['alignment']
                del wer_score['raw_alignement']
            results[key] = wer_score
        with open(perf_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

def bench_model(config, input_manifest, output_folder, debug=False):
    data, bench_results = check_if_benched(
        output_folder, input_manifest, config,  debug
    )
    if len(data)>0:
        model = get_model(config)
        model.load()
        audio = model.load_audio(PATH_TO_WARMUP_FILE)
        _ = model.transcribe(audio)
        for dataset in data:
            dataset_data = data[dataset]
            bench_dataset_result = bench_results.get(dataset, dict())
            if config.get('compute_rtf', True):
                if int(config.get('concurrency', 1) or 1) > 1:
                    iterator = transcribe_with_rtf_concurrent(model, dataset_data, output_folder, config)
                else:
                    iterator = transcribe_with_rtf(model, dataset_data, output_folder, config)
            else:
                iterator = transcribe_fast(model, dataset_data, output_folder, config)
            process_result(iterator, bench_dataset_result, output_folder, config, save_interval=1 if config.get('compute_rtf', True) else None)
        model.cleanup()
        with open(Path(output_folder) / "metadata.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(model.get_metadata(), indent=2, ensure_ascii=False))
    else:
        logger.info(f"Skipping transcriptions, it has already been transcribed")
    process_wer(output_folder, config)

def make_configs(configs):
    main_config = {k: v for k, v in configs.items() if k != "benchmarks"}
    new_configs = []
    for config in configs["benchmarks"]:
        keys, values = zip(*config.items())
        all_combinations = [
            main_config | dict(zip(keys, prod))
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
    input_audios_paths=""
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
        config['input_audios_paths'] = config.get('input_audios_paths', input_audios_paths)
        bench_id = model.get_folder_name()
        logger.info(
            f"Benching {bench_id} (progress {progress_bar.n}/{progress_bar.total})"
        )
        progress_bar.set_description(f"Using {bench_id}".ljust(45))
        config_output = Path(output_folder) / bench_id
        config_output.mkdir(parents=True, exist_ok=True)
        (config_output / "predictions").mkdir(parents=True, exist_ok=True)
        (config_output / "performances").mkdir(parents=True, exist_ok=True)
        try:
            start = time.time()
            bench_model(
                config, config["input_manifest"], str(config_output), debug
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
                        with open(config_output / "error.log", "w") as f:
                            f.write(f"{e}")
                    torch.cuda.empty_cache()
                predictions_dir = config_output / "predictions"
                if len(list(predictions_dir.iterdir())) == 0 and not (config_output / "error.log").exists():
                    predictions_dir.rmdir()
                    (config_output / "performances").rmdir()
                    (config_output / "wer").rmdir()
                    config_output.rmdir()
                    logger.error(f"Benched folder is empty, removing it")
                else:
                    logger.error(f"Benched folder is not empty")
                logger.error(f"Skipping to next configuration")
            else:
                raise e
