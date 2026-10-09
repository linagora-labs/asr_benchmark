# ASR Benchmark

Toolkit to benchmark various speech recognition APIs (NeMo, Whisper...) and visualize the results. Supported models are mostly french. It can compute WER, RTF (or latencies when streaming) and measure hardware usage.

🏆 **[French ASR Leaderboard](https://linagora-labs.github.io/asr_benchmark/)**

## How to bench

```
uv run python benchmarker.py CONFIG_FILE [--output_folder DIR] [--input_manifest FILE] [--debug] [--not_compute_rtf] [--not_save_predictions] [--not_save_alignments] [--log_file FILE]
```

- `--debug`: only 2 files per benchmark.
- The config (YAML) has a `benchmarks` list, each entry with a `backend` and a `model`. Parameter lists are expanded (grid).
- Results go to `output_folder/<model folder>`. Already computed folders are skipped (resumption). `full_name: True` adds the settings to the folder name.
- Scores: `wer`, `cer`, `wer_nocasepunc`, `cer_nocasepunc`, with a 95% bootstrap confidence interval. Each file's errors are capped at 100%.
- Logs go to `logs/`.

### Data

The input data file (manifest) is a jsonl file (one json per line). Each line must have these fields:
- "audio_filepath", the path to the audio file
- "text", the text associated with the segment

They can also have:
- "offset", the start of the segment in the audio (if not specified, it is equal to 0)
- "duration", the duration of the segment (if not specified, the whole audio is used)
- "name" or "dataset", the name of the dataset

### Examples

Examples are provided in the `examples` folder. There is an audio file, a config file, and a notebook for generating plots.

## Requirements

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) then install the package:

```bash
# Install core dependencies
uv sync

# Install backend-specific dependencies (as needed)
uv sync --extra whisper
uv sync --extra nemo
uv sync --extra voxtral --extra qwen-asr   # transformers 5.x backends can share an env
```

Available extras:
- backends: `whisper` (faster-whisper), `openai-whisper`, `nemo`, `transformers`, `voxtral`, `qwen-asr`, `vibevoice`, `moss`, `gemma3n`, `linto`
- options: `vad` (auditok / silero VAD)
- analysis: `visu` (notebooks), `tools` (scripts of `tools/`)

`whisper` and `nemo` are incompatible with each other and with the transformers 5.x extras (`moss`, `voxtral`, `qwen-asr`, `vibevoice`, `gemma3n`): use one environment per group (`uv sync --all-extras` does not work). The `vllm` backend only talks HTTP to a `vllm serve` process, vLLM itself being expected in its own environment/container.

<details>
<summary>Alternative: pip install</summary>

```bash
pip install -e .
pip install -e ".[whisper]"   # for whisper backends
pip install -e ".[nemo]"      # for nemo backends
```
</details>



## Tools

Some tools are available in the `tools` folder:
- add_silence.py: adds white noise to audio files
- concat_audios.py: concatenates audio files
- remove_audio_path.py: keeps only file names in a manifest
- subsample_data.py: for selecting a subset of specified datasets
- generate_plots.py: WER / RTF (and RAM-VRAM) plots from benchmark outputs
- plot_benchmark_monitoring.py: processing time, RAM and VRAM plots from benchmark results
- leaderboard/build.py: builds the leaderboard page (`site/`) from a results folder (and the speed runs of `results_rtf` for the RTFx column)
- fill_model_info.py: fills the `model_info` (license, size, languages) of results `metadata.json` from the Hugging Face API
- rescore_results.py: recomputes the scores (and confidence intervals) of existing results from their predictions, after a scoring change

## Backends (interfaces)

The current available backends:
- HTTP-API ("http-api")
- LinTO-STT ("linto-stt", "linto-stt-whisper", "linto-stt-nemo"): for using whisper, kaldi or nemo models through a LinTO-STT container. Can be streaming (can compute latencies) or offline
- Whisper ("openai")
- Faster Whisper ("faster-whisper")
- Transformers ("transformers", "transformers-whisper")
- Transformers Intel ("intel-transformers"): for using intel extension
- Transformers Facebook ("transformers-facebook"): for MMS model
- Transformers Bofenghuang ("transformers-bofenghuang"): for the french finetuned wav2vec
- NeMo ("nemo"): CTC, RNNT, hybrid and Canary models
- vLLM ("vllm"): Voxtral and other audio models served by `vllm serve`
- Voxtral ("transformers-voxtral", "transformers-voxtral-realtime")
- Qwen ("qwen3-asr", "qwen3-omni")
- VibeVoice ("vibevoice-asr")
- MOSS ("moss")
- Gemma 3n ("gemma3n")
- Parakeet ("transformers-parakeet")
- ARK ASR ("ark-asr")
- Kyutai STT ("kyutai-stt")
- Hojo ASR ("hojo-asr")


To add a backend: inherit from `asr_benchmark.benchmark.interfaces.Model`, implement `load`, `transcribe`, `get_folder_name`, `get_metadata`, and register it in `asr_benchmark.benchmark.backend_to_model`.

## Leaderboard

Built from `benchmarks/sota/results` (and `results_rtf` for the RTFx column) by `tools/leaderboard/build.py`, and published by a GitHub Action (`.github/workflows/leaderboard.yml`) on each push to `main`. Local preview: `python tools/leaderboard/build.py`, then open `site/index.html`.

For any results folder:
```bash
python tools/leaderboard/build.py --results RESULTS_DIR --manifest MANIFEST --output OUT_DIR  # -> OUT_DIR/index.html
```
`--rtf_results` / `--rtf_manifest` add the RTFx column and the speed-runs description (optional).

For a new model, fill `model_info` in its `metadata.json` with `tools/fill_model_info.py`, then by hand (`architecture`, `task`, `streaming`, `wer_note`, `rtf_note`, `prompt_source`).

## LinTO STT FR Fastconformer benchmark

Benchmark of ASR models for the french language used to make the [LinTO STT FR Fastconformer Huggingface page](https://huggingface.co/linagora/linto_stt_fr_fastconformer). The datasets used in the benchmark are : Common Voice, Multilingual LibriSpeech, Voxpopuli, SUMM-RE, TEDx and YouTube.

See the results [here](benchmarks/linto_stt_fr_fastconformer/README.md)