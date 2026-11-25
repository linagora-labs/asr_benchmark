"""Processing times of the contextual biasing runs, read from the benchmark logs, for the
runs made before the benchmark recorded them (metadata.json, "transcription_time").

A log entry "Finished benching RUN after N sec" covers the loading of the model, the
transcriptions and the scoring: the RTFx computed from it (audio duration / N) is a
lower bound of the real throughput, approximate for short runs. For each run folder, the
last entry of the logs that transcribed something is kept (logs in modification order:
the debug runs and the runs done again come before the final one). The results folder
of an entry is the one whose datasets are those of its log ("N segments selected").

    python tools/contextual_biasing/timings_from_logs.py LOGS/*.log \\
        --results benchmarks/contextual_biasing/results_fr benchmarks/contextual_biasing/results_en

Writes RESULTS/timings_from_logs.json: {run folder: {"seconds", "datasets", "log"}},
"datasets": those of the log that the run folder has predictions for.
"""

import argparse
import json
import os
import re
from pathlib import Path

_SELECTED = re.compile(r" (\S+): \d+ segments selected")
_BENCHING = re.compile(r"Benching (\S+) \(progress")
_FINISHED = re.compile(r"Finished benching (\S+) after (\d+)sec")


def parse_log(path):
    """[(run name, seconds, datasets of the log)] of the runs that transcribed something"""
    entries, datasets, skipped, new_selection = [], set(), False, True
    for line in open(path, encoding="utf-8", errors="replace"):
        if m := _SELECTED.search(line):
            if new_selection:
                datasets, new_selection = set(), False
            datasets.add(m.group(1))
        elif _BENCHING.search(line):
            skipped, new_selection = False, True
        elif "Skipping transcriptions" in line:
            skipped = True
        elif (m := _FINISHED.search(line)) and not skipped and int(m.group(2)) > 0:
            entries.append((m.group(1), int(m.group(2)), set(datasets)))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="Log files of the benchmark (*.console.log are ignored)")
    parser.add_argument("--results", nargs="+", required=True, help="Results folders (output_folder)")
    args = parser.parse_args()

    results = [Path(r) for r in args.results]
    # Datasets of each results folder: names of the prediction files of its runs
    known = {r: {f.stem for f in r.glob("*/predictions/*.json")} for r in results}
    timings = {r: {} for r in results}
    logs = sorted((Path(p) for p in args.logs if not p.endswith(".console.log")), key=os.path.getmtime)
    for log in logs:
        for run, seconds, datasets in parse_log(log):
            for r in results:
                predictions = r / run / "predictions"
                if not (datasets & known[r]) or not predictions.is_dir():
                    continue
                done = sorted(datasets & {f.stem for f in predictions.glob("*.json")})
                if done:
                    timings[r][run] = {"seconds": seconds, "datasets": done, "log": log.name}
    for r, runs in timings.items():
        out = r / "timings_from_logs.json"
        out.write_text(json.dumps(dict(sorted(runs.items())), indent=1) + "\n", encoding="utf-8")
        print(f"{len(runs)} runs in {out}")


if __name__ == "__main__":
    main()
