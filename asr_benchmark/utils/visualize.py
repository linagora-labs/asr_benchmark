import json
import os
import re
from tqdm import tqdm

def natural_key(string_):
    """See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', str(string_))]

def sort_result(list_to_sort, key):
    if key=="model":
        order = sorted(list_to_sort)
    elif key=="precision":
        order = ["int8", "float16", "float32"]
    elif key=="streaming":
        if len(list_to_sort)==2:
            order = ["low", "high"]
        else:
            order = ["offline", "low", "high"]
    else: 
        order = sorted(list_to_sort, key=natural_key)
    return order

def load_data(input_folder, selected_dataset=None):
    experiments = os.listdir(input_folder)
    data = list()
    for experiment in tqdm(experiments):
        if not os.path.exists(os.path.join(input_folder, experiment, 'metadata.json')):
            continue
        with open(os.path.join(input_folder, experiment, 'metadata.json'), 'r') as f:
            exp_data = json.load(f)
        if exp_data['backend'] == "faster-whisper" and "whisper" not in exp_data['model']:
            exp_data['model'] = f"whisper-{exp_data['model']}"
        if 'accurate' in exp_data:
            if exp_data['accurate']:
                exp_data['accurate'] = 'accurate'
            else:
                exp_data['accurate'] = 'greedy'
        datasets = os.listdir(os.path.join(input_folder, experiment, 'performances'))
        for dataset in datasets:
            names = list()
            wer_list = list()
            wer_details = list()
            audio_duration_list = list()
            process_duration_list = list()
            latencies = list()
            if selected_dataset and dataset.lower().replace(".json", "") != selected_dataset:
                continue
            row = exp_data.copy()
            with open(os.path.join(input_folder, experiment, 'performances', dataset), 'r') as f:
                json_data = json.load(f)
            for file in json_data:
                perf = json_data[file]
                names.append(file)
                audio_duration_list.append(perf['audio_duration'])
                if "prediction_duration" in perf:
                    process_duration_list.append(perf['prediction_duration'])
                if "wer" in perf:
                    wer_list.append(perf["wer"]["wer"])
                    wer_details.append({k: v for k, v in perf["wer"].items() if k != "alignment"})
                if "latency" in perf:
                    latencies.append(perf['latency'])
            row['dataset'] = json_data[file]['dataset']
            row['audio_duration'] = audio_duration_list
            if latencies:
                row['latency'] = latencies
            if process_duration_list:
                row['process_duration'] = process_duration_list
            row['wer'] = wer_list
            row['wer_details'] = wer_details
            row['audio_file'] = names
            if os.path.exists(os.path.join(input_folder, experiment, 'monitoring.json')):
                with open(os.path.join(input_folder, experiment, 'monitoring.json'), 'r') as f:
                    monitoring = json.load(f)
                row['RAM usage'] = round(max(monitoring['ram_usage']), 2)
                if 'vram_usage' in monitoring and monitoring['vram_usage']:
                    row['VRAM usage'] = round(max(monitoring['vram_usage']), 2)
                if "total_gpu_usage" in monitoring:
                    row['GPU usage'] = round(monitoring['total_gpu_usage'], 0)
            data.append(row)
    return data