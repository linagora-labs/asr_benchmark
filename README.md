# ASR Benchmark

Toolkit to benchmark various speech recognition APIs (NeMo, Whisper...) and visualize the results. Supported models are mostly french. It can compute WER, RTF (or latencies when streaming) and measure hardware usage.

## How to bench

Just run:

```
python benchmarker.py CONFIG_FILE
```

### Data

The input data file (manifest) is a jsonl file (one json per line). Each line must have these fields:
- "audio_filepath", the path to the audio file
- "text", the text associated with the segment

They can also have:
- "offset", the start of the segment in the audio (if not specified, it is equal to 0)
- "duration", the duration of the segment (if not specified, the whole audio is used)
- "name" or "dataset", the name of the dataset

### Examples

Examples are provided in the `examples` folder. There is a audio file to test with benchmark config file, and a notebook for generating plots.

## Requirements

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) then install the package:

```bash
# Install core dependencies
uv sync

# Install backend-specific dependencies (as needed)
uv sync --extra whisper
uv sync --extra nemo

# Or install everything
uv sync --all-extras
```

Then run benchmarks with:
```bash
uv run python benchmarker.py CONFIG_FILE
```

<details>
<summary>Alternative: pip install</summary>

```bash
pip install -e .
pip install -e ".[whisper]"   # for whisper backends
pip install -e ".[nemo]"      # for nemo backends
```
</details>



## Tools

Some tools are avaialble in the `tools` folder:
- add_silence.py: a script for adding white noise to audio files
- subsample_data.py: for selecting a subset of specified datasets

Don't hesitate to submit your tools (for converting datasets to the jsonl format for example). I used scripts from ssak to do it but datasets were in kaldi format.

## Backends (interfaces)

The current available backends:
- HTTP-API ("http-api")
- LinTo-STT ("linto-stt"): for using whisper, kaldi or nemo models. Can be streaming (can compute latencies) or offline
- Whisper ("openai")
- Faster Whisper ("faster-whisper")
- Transformers ("transformers"): work for Whisper
- Transformers Intel ("intel-transformers"): for using intel extension
- Transformers Facebook ("transformers-facebook"): for MMS model
- Transformers Bofenghuang ("transformers-bofenghuang"): for the french finetuned wav2vec
- NeMo ("nemo")


If the available interfaces don't allow to bench a model you want, you can easily add it by folliwing these steps:
- You create new class that inherits from `asr_benchmark.benchmark.interfaces.Model`
- You implement the various functions (load, transcribe, ...)
- You add your backend in `asr_benchmark.benchmark.backend_to_model`

## LinTO STT FR Fastconformer benchmark

Benchmark of ASR models for the french language used to make the [LinTO STT FR Fastconformer Huggingface page](https://huggingface.co/linagora/linto_stt_fr_fastconformer). The datasets used in the benchmark are : Common Voice, Multilingual LibriSpeech, Voxpopuli, SUMM-RE, TEDx and YouTube.

See the results [here](benchmarks/linto_stt_fr_fastconformer/README.md)