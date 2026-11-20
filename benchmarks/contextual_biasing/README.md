# Contextual Biasing ASR Benchmark

Speech recognition models fail on the rare words that matter most: proper names,
product names, drug names, jargon. Many models accept, at inference time, a list of
terms likely to be spoken ("contextual biasing", also called hotwords, keyword
boosting, phrase boosting, context terms or custom vocabulary). This benchmark
measures, for each model:

1. how well it transcribes the listed terms **without and with** the list
   (absolute scores and gain);
2. what the list costs: errors on the other words, listed terms written although
   not spoken (false alarms);
3. how it behaves when the list grows with terms that are not spoken
   (distractors), as with a domain dictionary.

Models without a mechanism are run as baselines (scored on the same lists).

## Design decisions

- Languages: French and English.
- Lists as defined by existing benchmarks, to stay comparable with published
  results: per-utterance lists (targets + distractors) where the benchmark defines
  them (LibriSpeech), global lists per dataset or per call otherwise (PxSLU,
  Contextual Earnings-22). Production-like per-document lists are a later step.
- Only public data with existing references (no hand annotation, no internal
  recordings).
- Every model with a mechanism is run without list and with each list; the run
  without list is scored on every list, so the gain is measured on the same data.
- Lists that do not fit in a model (prompt length, documented limits) are left out
  of its runs (see "List sizes" below).

## Datasets

| Dataset | Lang. | Lists | Source |
|---|---|---|---|
| PxSLU test / dev (1.38 h / 0.57 h) | fr | global: `pxslu` (the 389 drug names of the corpus: the targets), `bdpm`, `vidal`, `theriaque` (targets + drug dictionaries); per utterance: `pxslu25` (25 drug names of `pxslu`: the ones spoken in the utterance plus others drawn at random; the size of the per call `global` lists of Contextual Earnings-22, 25 terms on average, and short enough for Whisper's 223-token hotwords); `pxslu_upper`, `pxslu25_upper`: the same lists in capitals. The manifests also hold per utterance `oracle` lists (only the targets spoken in the utterance) and `oracle_upper`, not run: not a realistic setting (the list would have to know what is said) | Kocabiyikoglu et al., LREC 2022 (arXiv:2207.08292), CC BY 4.0: spoken drug prescriptions |
| LibriSpeech test-other (test-clean left out: its "clean" speakers leave few errors on the rare words to correct) | en | per utterance: rare words of the reference + N distractors, N in {100, 500, 1000, 2000} | Le et al., Interspeech 2021 (arXiv:2104.02194), `facebookresearch/fbai-speech` (`is21_deep_bias`, MIT) |
| Contextual Earnings-22 test / dev (test: 771 clips, 3.2 h) | en | per clip: `local` (keywords spoken in the clip: the targets), `global` (keywords of the whole call) | Durmus et al. 2026 (arXiv:2604.07354), `argmaxinc/contextual-earnings22` (licence of Earnings-22) |

### PxSLU

- 1,981 recordings of drug prescriptions (doctors, experts, non-experts). The
  official split files hold line numbers of the raw corpus (`raw/`); test =
  doctors only.
- The dataset's cleaning script (`scripts/clean_PxSLU.py`) replaces substrings in
  sequence, which creates wrong words ("alepsal" becomes "alalepsal", "amikacine"
  becomes "amikacinee"). `prepare_pxslu.py` applies the same spelling corrections
  as whole words in one pass, removes the same utterances (unidentifiable drug
  names) and the truncated words ("qu/").
- Drug lists: `pxslu` comes from the curated list of the 2024 LINAGORA internship
  on drug name recognition, made consistent with the corrected transcripts. `bdpm`
  is built from the French public drug database (Base de données publique des
  médicaments, ANSM): brand names and active substances, homeopathic and herbal
  registrations left out. `vidal` and `theriaque` are larger dictionaries of the
  internship.
- Case: the PxSLU transcripts are in lower case, while the models write case.
  Drug names are given their usual case (brands capitalized, each part of a
  hyphenated name except Un-alfa; active substances in lower case; acronyms in
  capitals) and one spelling per drug, from a reviewed table
  (`lists/pxslu/drugs_pxslu_case.tsv`); the transcripts are written in sentence
  case with these forms. Names of the larger lists that are not targets are cased
  automatically (lower case for active substances of the BDPM, else capitalized).
- Licences: `drugs_bdpm.txt` is derived from the BDPM (source: ANSM, files of
  2026-11-18), under the Licence Ouverte / Open Licence 2.0. The Vidal and
  Thériaque lists are not in git: the terms of use of these databases forbid
  extracting and redistributing their content.

### Other datasets considered

- Earnings-21 ships two global lists (`oracle_list.txt`: entities of the
  references; `distractor_list.txt`: the same plus Fortune 500 companies and CEOs);
  Contextual Earnings-22 covers the same setting with per-clip lists.
- Plain Earnings-22 and Common Voice have no released lists (the Common Voice NER
  lists of Thorbecke et al. 2024 were not released).

## Models and mechanisms

| Model (backend) | How the list is given | Limits |
|---|---|---|
| Whisper large-v3 / large-v3-turbo (`faster-whisper`) | `hotwords` argument (prompt tokens before the previous text, on every 30 s window) | 223 tokens |
| Qwen3-ASR 1.7B (`vllm`) | `prompt` field, put in the system turn; template `Vocabulary: {terms}.` (the format is not documented by Qwen; this one is linto-stt's) | context length of the server |
| Nemotron-3.5-Transcribe (`vllm`, chat endpoint) | `{prompt} Context terms: {terms}.` (model card) | |
| Granite Speech 5.0 2B (`vllm`, chat endpoint) | `{prompt} Keywords: {terms}` (model card) | "up to 2000" keywords |
| MOSS-Transcribe-Diarize (`vllm`) | `{prompt}热词提示：{terms}` (French runs, default Chinese prompt) or `{prompt} Hotwords: {terms}` (English), as documented in the model's GitHub examples | |
| VibeVoice-ASR (`vibevoice-asr`) | extra context of the request | |
| Parakeet TDT 0.6B v3, Canary 1B v2, linto-stt-fr-fastconformer (RNN-T decoder; French only) (`nemo`) | GPU phrase boosting of NeMo (boosting tree fused at decoding time, `greedy_batch`, beam search for Canary; Andrusenko et al. 2025, arXiv:2508.07014); weight `contextual_biasing_alpha` 1.0 and 2.0 (PxSLU dev with Parakeet, `config_fr_nemo_alpha.yaml`: 1.0 best F1 / false alarm trade-off, 2.0 +5 F1 with 5x false alarms) | |
| linto-stt, whisper engine (`linto-stt`) | HTTP `hotwords` parameter | |
| Cohere Transcribe (`vllm`) | none: baseline | |

Not included: Voxtral (open weights have no mechanism; `context_bias` exists only
in Mistral's hosted API). Through vLLM's `/v1/audio/transcriptions` endpoint, the
`prompt` field reaches Qwen3-ASR, Whisper and MOSS, but Nemotron and Granite ignore
it: they are run through the chat endpoint.

List sizes (tokens): PxSLU `pxslu25` at most ~130 (~160 in capitals), `pxslu` ~1,700 (Whisper tokenizer), `bdpm` ~17.8k,
`vidal` ~12k, `theriaque` ~38k; LibriSpeech N=100 ~370, N=2000 ~7.2k; Contextual
Earnings-22 at most 161. The configs leave out the lists a model cannot take
(Whisper: only `pxslu25` and the Contextual Earnings-22 lists;
Granite: no list above 16k tokens).

## Preparation

Where the audio is (e.g. on the DGX machines, under `/home/data/audio`):

```bash
# PxSLU (BDPM files: https://base-donnees-publique.medicaments.gouv.fr/telechargement)
mkdir bdpm && for f in CIS_bdpm.txt CIS_COMPO_bdpm.txt; do
    curl -L -o bdpm/$f https://base-donnees-publique.medicaments.gouv.fr/download/file/$f; done
uv run python tools/contextual_biasing/prepare_pxslu.py \
    --pxslu_dir /home/data/audio/raw/transcript/fr/PxSLU \
    --drug_lists_dir <Medical-ASR-adaptation-techniques>/phonetisation_transcription/dictionary/drug_list \
    --bdpm_dir bdpm
# LibriSpeech
git clone --depth 1 https://github.com/facebookresearch/fbai-speech /tmp/fbai-speech
uv run python tools/contextual_biasing/prepare_librispeech.py \
    --librispeech_dir /home/data/audio/raw/transcript/en/LIBRISPEECH/LibriSpeech \
    --biasing_dir /tmp/fbai-speech/is21_deep_bias
# Contextual Earnings-22 (writes the audio of the parquet files to <dataset_dir>/wav/)
hf download argmaxinc/contextual-earnings22 --repo-type dataset \
    --local-dir /home/data/audio/raw/transcript/en/ContextualEarnings22
uv run python tools/contextual_biasing/prepare_contextual_earnings22.py \
    --dataset_dir /home/data/audio/raw/transcript/en/ContextualEarnings22
```

They write `manifests/*.jsonl` and `lists/` in this folder. The LibriSpeech lists
(`lists/librispeech/*.json`, 230 MB) are not in git: run the script to get them.

## Running

```bash
AUDIO_FOLDER=/home/data/audio uv run python benchmarker.py benchmarks/contextual_biasing/config_fr.yaml
uv run python tools/contextual_biasing/report.py benchmarks/contextual_biasing/results_fr
uv run python tools/contextual_biasing/report.py benchmarks/contextual_biasing/results_fr --mode cased
uv run python tools/contextual_biasing/build_page.py --output benchmarks/contextual_biasing/page \
    --results fr=benchmarks/contextual_biasing/results_fr en=benchmarks/contextual_biasing/results_en
```

`report.py` prints, per dataset and list, the measures of each model without list,
with the list, and the difference (optionally as CSV). `build_page.py` writes the same
measures as an HTML page (`page/index.html`, a single file): overview table of a measure
for each model and list, charts of each list without and with it, and all the measures
of one list; both text normalizations, and a selection of the models shown (boosting weight tuning runs hidden by default).

Notes for the configs: Nemotron runs on a vLLM server started separately (with a
`--max-model-len` above the longest list); Granite 5.0 is served from a folder of
links to the Hugging Face snapshot without `out_llm.safetensors` (vLLM 0.28 fails
on it); NeMo conditions are separate config entries (one process per condition:
NeMo 3.0 fails to load a model twice in one process).

## How it works

- Manifest fields: `contextual_biasing` (a dict {list name: list}) and
  `contextual_biasing_targets` (terms to score, default: the list). A list is an
  inline list of strings, a text file (one term per line), or `file.json#key`.
- Config key `contextual_biasing`: the list given to the model (`null` for none).
  A list of values runs every condition. Runs with a list are written to
  `<model folder>_cb-<list>` (plus backend settings, e.g. `-alpha1.0` for NeMo),
  and only on the datasets that have this list. Backends without a biasing
  mechanism refuse the key.
- Other config keys: `contextual_biasing_template` (text given to the model:
  `{terms}`, the terms joined with `", "`, and, for some backends, `{prompt}`);
  per backend: `contextual_biasing_mode` (faster-whisper: `hotwords` or
  `initial_prompt`), `contextual_biasing_field` (vLLM transcriptions: `prompt` or
  `hotwords`), `contextual_biasing_alpha`, `contextual_biasing_context_score`,
  `contextual_biasing_depth_scaling` (NeMo phrase boosting).
- Backends with a mechanism: faster-whisper, openai-whisper, qwen3-asr, vllm
  (transcriptions and chat endpoints), vibevoice-asr, moss, nemo (GPU phrase
  boosting), linto-stt (whisper engine).
- Without the key, nothing changes for the standard ASR benchmark.

## Measures

In `performances/<dataset>.json`, key `contextual_biasing`, for each list and two
normalizations (`nocasepunc`: as for the WER; `cased`: case checked on the list
terms only):

- `precision`, `recall`, `f1` of the target terms: for each term and utterance,
  tp = min(#ref, #hyp), fp = max(0, #hyp - #ref), fn = max(0, #ref - #hyp), where #
  counts the occurrences of the term (one or more words). Same definition as
  ssak's `compute_wer(words_list=...)`, used in the PxSLU internship.
- `b_wer`, `u_wer` (Le et al. 2021): WER on the reference words that belong to a
  list term, and on the other words; an inserted word counts in B-WER if it
  belongs to a list term. With multi-word terms, a word is biased only when it is
  part of an occurrence of a term.
- `false_alarms` (and per hour): occurrences of list terms in the prediction
  beyond their count in the reference (e.g. distractors written although not
  spoken).

`report.py` and the page also give the runaway rate of each run (from its
predictions, not stored in `performances/`): the percentage of utterances whose
transcription has more than twice as many words as the reference, plus 10 (loops,
invented sentences, prompt recited). A few such outputs weigh a lot on B-WER
(inserted list terms count in it), FA/h and WER: e.g. on PxSLU test, Qwen3-ASR
recites the start of the `bdpm` list in 6 of 556 utterances (B-WER above 100 %),
and MOSS keeps generating after the transcription in 23 utterances with `pxslu`.

NeMo runs also record, in `metadata.json` (`contextual_biasing_tree_time_ms`), the
time to build the boosting tree and set it in the decoding (count, first, median,
max, total, in ms): once for a global list, once per utterance for per-utterance
lists. It is what a server pays each time the hotwords of a request change.

Case (`cased`): terms are found without case; an occurrence in the prediction is
correct only when written as in the list given to the model (else as in the
targets), with the first letter of the utterance as in the reference; the other
words are compared without case. So a model given "DOLIPRANE" has to write
"DOLIPRANE" (`*_upper` lists: does the model follow the case of the terms?). The
other datasets are used as published: compare them with the literature in
`nocasepunc`. In English, apostrophes stay inside words in `cased` ("eden's"), as
in the scorer of Le et al. (on their published hypotheses, the LibriSpeech
`cased` B-WER/U-WER equal those of their `score.py`).

## State of the art

Public benchmarks with released lists (checked in the primary sources in November
2026):

| Benchmark | Lang. | Lists | Measures |
|---|---|---|---|
| LibriSpeech biasing lists (Le et al. 2021, arXiv:2104.02194) | en | per utterance: rare words + N distractors (the de facto standard) | WER, B-WER, U-WER |
| Earnings-21 bias lists (Fox and Delworth 2022, arXiv:2209.01250) | en | global oracle and distractor lists | rare-word recall, WER |
| ConEC (Huang et al. 2024) | en | per call, real context (slides, press releases, names) | WER on rare words and entities |
| Contextual Earnings-22 (Durmus et al. 2026, arXiv:2604.07354) | en | per clip, local and global | WER, keyword P/R/F |
| SlideSpeech (Wang et al. 2024, arXiv:2309.05396) | en | per segment, keywords from slides | WER, B-WER, U-WER |
| ContextASR-Bench (Wang et al. 2025, arXiv:2507.05727) | en, zh | per utterance, synthetic speech | WER, NE-WER, NE-FNR |
| Test/Dev-Aishell1-NE (SeACo-Paraformer, arXiv:2308.03266) | zh | global named entities; "R1" subset of terms the baseline misses | CER, hotword R/P/F1 |
| IndicContextEval (Joshi et al. 2026, arXiv:2606.19157) | 8 Indic languages | context levels, in-domain and adversarial entities | WER, NEER |

Other Mandarin sets: ATPC hotword-161, WenetSpeech biased words, AISHELL-NER and
derived sets, ST-cmds-NE. Protocols without released lists: Common Voice NER lists
(Thorbecke et al. 2024, arXiv:2409.13499; en, ca, fr, de, es, it), NVIDIA's
evaluations of phrase boosting (arXiv:2406.07096, arXiv:2508.07014), Microsoft
LOGIC (internal data, 11 locales including fr-FR).

No public French contextual biasing benchmark was found: the PxSLU part of this
benchmark, with publishable lists (`pxslu`, `bdpm`), fills that gap. Worth adding
later: a "hard" subset (terms the baseline misses, as SeACo-Paraformer and NVIDIA
do), and a Common Voice FR protocol following Thorbecke et al.
