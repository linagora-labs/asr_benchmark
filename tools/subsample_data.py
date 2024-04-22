import argparse
from sklearn.model_selection import train_test_split
from linastt.utils.text import format_text_latin
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


def load_manifest(manifest_path, config, min_duration=0.05):
    dataset_names = {config[i].get('name'): i for i in config}
    data = list()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            if float(json_line['duration']) < min_duration:
                logger.warning(f"Skipping {json_line['audio_filepath']} with duration {json_line['duration']}")
                continue
            data.append(json_line)
    logger.info(f"Computing WER on {len(data)} samples")
    data_sorted = dict()
    for i in data:
        name = i["name"]
        name = name.replace("_nocasepunc_max30", "")
        name = name.replace("_nocasepunc_eval_max30", "")
        name = name.replace("_nocasepunc", "")
        name = name.replace("_max30", "")
        if name.lower() in dataset_names:
            name = dataset_names[name.lower()]
        if name not in data_sorted:
            data_sorted[name] = list()
        data_sorted[name].append(i)
    data = data_sorted
    logger.info(f"Computing WER on {len(data)} datasets")
    return data

def write_manifest(data, path):
    with open(path, 'w') as f:
        for d in data:
            for row in data[d]:
                row['text'] = format_text_latin(row['text'], lang='fr')
                f.write(json.dumps(row) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes a sub sample of the data')
    parser.add_argument('manifest', help="Input manifest", type=str)
    parser.add_argument('--subsample_config', type=str, default="subsample_gpu.json")
    parser.add_argument('--output_manifest', help="Output directory", type=str, default="manifest_subsampled.jsonl")
    parser.add_argument('--remove_others', help="Remove other datasets", action="store_true", default=False)
    args = parser.parse_args()

    import json
    with open(args.subsample_config, 'r') as f:
        subsample_config = json.load(f)
    print(subsample_config)
    data = load_manifest(args.manifest, subsample_config)

    new_data = dict()
    for d, dataset_data in data.items():
        if d.lower() in subsample_config and "subsample" in subsample_config[d.lower()]:
            subsample_value = subsample_config[d.lower()]['subsample']
            if subsample_value>0 and len(dataset_data)>subsample_value:
                logger.info(f"Subsampling {d} to {subsample_value}")
                keep, _ = train_test_split(dataset_data, train_size=subsample_value, random_state=42)
                new_data[d] = keep
            elif subsample_value!=0:
                logger.info(f"Keeping all {d}")
                new_data[d] = dataset_data
        elif not args.remove_others:
            logger.info(f"Keeping {d}")
            new_data[d] = dataset_data
        else:
            logger.info(f"Removing {d}")
            
    write_manifest(new_data, args.output_manifest)