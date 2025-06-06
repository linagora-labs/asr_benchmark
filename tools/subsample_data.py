import argparse
from sklearn.model_selection import train_test_split
from ssak.utils.text import format_text_latin
import json

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# subsample = {   # for cpu
#     "commonvoice": 100,
#     "mls": 200,
#     "summ-re": 100,
# }
# subsample = {
#     "commonvoice": 500,
#     "mls": 500,
#     "summ-re": 500,
#     "voxpopuli": 500
# }
# datasets_names = {
#     "mls_facebook_french": "MLS",
#     "youtubefr_split6": "YouTube",
# }


def name_to_dataset(row):
    name = row["name"]
    name = name.replace("_nocasepunc_max30", "")
    name = name.replace("_nocasepunc_eval_max30", "")
    name = name.replace("_nocasepunc", "")
    name = name.replace("_max30", "")
    return name.lower()

def load_manifest(manifest_path, config, min_duration=0.05, max_duration=30.0):
    dataset_names = {config[i].get('name', i): i for i in config}
    data = list()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            min_dur = min_duration
            if name_to_dataset(json_line) in dataset_names:
                min_dur = config[dataset_names[name_to_dataset(json_line)]].get('min_duration', min_duration)
            if float(json_line['duration']) < min_dur or float(json_line['duration']) > max_duration:
                logger.warning(f"Skipping {json_line['audio_filepath']} with duration {json_line['duration']}")
                continue
            data.append(json_line)
    logger.info(f"Total number of samples: {len(data)}")
    data_sorted = dict()
    for i in data:
        name = name_to_dataset(i)
        if name in dataset_names:
            name = dataset_names[name]
        if name not in data_sorted:
            data_sorted[name] = list()
        data_sorted[name].append(i)
    data = data_sorted
    logger.info(f"total number of datasets: {len(data)}")
    return data

def write_manifest(data, path):
    with open(path, 'w') as f:
        for d in data:
            for row in data[d]:
                row['text'] = format_text_latin(row['text'], lang='fr')
                f.write(json.dumps(row) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes a sub sample of the data')
    parser.add_argument('--manifest', help="Input manifest", type=str, default="../data/test_manifest.jsonl")
    parser.add_argument('--subsample_config', type=str, default="../benchmarks/linto_stt_fr_fastconformer/subsample.json")
    parser.add_argument('--output_manifest', help="Output directory", type=str, default="manifest_subsampled.jsonl")
    parser.add_argument('--remove_others', help="Remove other datasets", action="store_true", default=False)
    parser.add_argument('--min_duration', default=1.0, type=float)
    parser.add_argument('--max_duration', default=30.0, type=float)
    args = parser.parse_args()

    with open(args.subsample_config, 'r') as f:
        subsample_config = json.load(f)
    logger.info(subsample_config)
    data = load_manifest(args.manifest, subsample_config, args.min_duration, args.max_duration)

    print()
    logger.info("Durations")
    for dataset, dataset_data in data.items():
        duration = sum([i['duration'] for i in dataset_data])
        logger.info(f"{dataset} contains {duration/3600:.2f}h ({len(dataset_data)} rows)")
    
    print()
    logger.info("Subsampling")
    new_data = dict()
    for d, dataset_data in data.items():
        if d.lower() in subsample_config and "subsample" in subsample_config[d.lower()]:
            subsample_value = subsample_config[d.lower()]['subsample']
            if subsample_value>0 and subsample_value!=1 and (subsample_value<1 or len(dataset_data)>subsample_value):
                keep, _ = train_test_split(dataset_data, train_size=subsample_value, random_state=42)
                logger.info(f"Subsampling {d} to {subsample_value} ({len(keep)} rows)")
                new_data[d] = keep
            elif subsample_value!=0:
                logger.info(f"Keeping all {d} ({len(dataset_data)} rows)")
                new_data[d] = dataset_data
        elif not args.remove_others:
            logger.info(f"Keeping {d}")
            new_data[d] = dataset_data
        else:
            logger.info(f"Removing {d}")
    print()
    logger.info("New durations")
    for dataset, dataset_data in new_data.items():
        duration = sum([i['duration'] for i in dataset_data])
        logger.info(f"{dataset} contains {duration/3600:.2f}h")
    print()
    logger.info(f"Total new number of segments: {sum([len(i) for _,i in new_data.items()])}")
    write_manifest(new_data, args.output_manifest)