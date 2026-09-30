# ASR Benchmark

Toolkit to benchmark various speech recognition APIs (NeMo, Whisper...) and visualize the results. Supported models are mostly french. It can compute WER, RTF (or latencies when streaming) and measure hardware usage.

**[French ASR Leaderboard](https://linagora-labs.github.io/asr_benchmark/)**: results of `benchmarks/sota`, rebuilt by a GitHub Action (`.github/workflows/leaderboard.yml`) each time results are pushed to `main`. To preview it locally: `python tools/leaderboard/build.py` then open `site/index.html`.

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
uv sync --extra voxtral --extra qwen-asr   # transformers 5.x backends can share an env
```

Available extras:
- backends: `whisper` (faster-whisper), `openai-whisper`, `nemo`, `transformers`, `voxtral`, `qwen-asr`, `vibevoice`, `moss`, `gemma3n`, `linto`
- options: `vad` (auditok / silero VAD)
- analysis: `visu` (notebooks), `tools` (scripts of `tools/`)

`whisper` and `nemo` are incompatible with each other and with the transformers 5.x extras (`moss`, `voxtral`, `qwen-asr`, `vibevoice`, `gemma3n`): use one environment per group (`uv sync --all-extras` does not work). The `vllm` backend only talks HTTP to a `vllm serve` process, vLLM itself being expected in its own environment/container.

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

Some tools are available in the `tools` folder:
- add_silence.py: a script for adding white noise to audio files
- subsample_data.py: for selecting a subset of specified datasets
- generate_plots.py: WER / RTF (and RAM-VRAM) plots from benchmark outputs
- plot_benchmark_monitoring.py: processing time, RAM and VRAM plots from benchmark results
- leaderboard/build.py: builds the leaderboard page (`site/`) from a results folder (and the speed runs of `results_rtf` for the RTFx column)
- fill_model_info.py: fills the `model_info` (license, size, languages) of results `metadata.json` from the Hugging Face API
- rescore_results.py: recomputes the scores (and confidence intervals) of existing results from their predictions, after a scoring change

Don't hesitate to submit your tools (for converting datasets to the jsonl format for example). I used scripts from ssak to do it but datasets were in kaldi format.

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


If the available interfaces don't allow to bench a model you want, you can easily add it by following these steps:
- You create new class that inherits from `asr_benchmark.benchmark.interfaces.Model`
- You implement the various functions (load, transcribe, ...)
- You add your backend in `asr_benchmark.benchmark.backend_to_model`

## LinTO STT FR Fastconformer benchmark

Benchmark of ASR models for the french language used to make the [LinTO STT FR Fastconformer Huggingface page](https://huggingface.co/linagora/linto_stt_fr_fastconformer). The datasets used in the benchmark are : Common Voice, Multilingual LibriSpeech, Voxpopuli, SUMM-RE, TEDx and YouTube.

See the results [here](benchmarks/linto_stt_fr_fastconformer/README.md)