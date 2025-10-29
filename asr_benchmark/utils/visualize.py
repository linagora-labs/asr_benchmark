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

def load_data(input_folder, selected_dataset=None, casepunc=False):
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
            if selected_dataset and dataset.lower().replace(".json", "") != selected_dataset:
                continue
            row = exp_data.copy()
            with open(os.path.join(input_folder, experiment, 'performances', dataset), 'r') as f:
                json_data = json.load(f)
            with open(os.path.join(input_folder, experiment, 'predictions', dataset), 'r') as f:
                json_pred_data = json.load(f)
            key = 'wer_nocasepunc' if not casepunc else 'wer'
            if key not in json_data:
                continue
            row['duration'] = json_data["duration"] if "duration" in json_data else "durations"
            row['num_data'] = json_data['num_data']
            row['dataset'] = json_pred_data[next(iter(json_pred_data))]['dataset']
            row['wer'] = json_data[key]['wer']
            row['wer_details'] = json_data[key]
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