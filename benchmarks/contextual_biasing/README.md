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

Three test sets, chosen to cover different settings: a French domain dictionary
(PxSLU), the standard English protocol with lists of controlled size (LibriSpeech),
and real names in conversational English with realistic per-call lists (Contextual
Earnings-22). Figures of the test splits as run (after the cleaning described below).

| | PxSLU test | LibriSpeech test-other | Contextual Earnings-22 test |
|---|---|---|---|
| Language | French | English | English |
| Speech | dictated drug prescriptions, spontaneous | read audiobooks | earnings calls: conversational, many accents |
| Size | 540 utterances, 1.47 h (mean 10 s, max 30 s) | 2,939 utterances, 5.34 h (mean 6.5 s, max 35 s) | 771 clips of 15 s from 54 calls, 3.21 h |
| Terms to recognize (targets) | drug names: 329 occurrences in 324 utterances, 168 distinct | rare words: 5,350 occurrences in 2,141 utterances, 3,838 distinct | names of persons, companies, products: 1,324 occurrences in 729 clips, 651 distinct |
| Lists run | global: `pxslu` (389 terms), `pxslu_ext` (427), `vidal` (2,753), `bdpm` (4,050), `theriaque` (8,851); per utterance: `pxslu25`, `pxslu25_ext` (25) | per utterance: `100`, `500`, `1000`, `2000` (rare words of the utterance + N distractors) | per clip: `global` (keywords of the whole call, 25 on average, 1 to 46) |
| Distractors (list terms not spoken) | other drug names (real dictionaries) | rare words drawn at random | other names of the same call |
| Licence | CC BY 4.0 | audio and text CC BY 4.0; lists MIT | transcripts and lists CC BY-SA 4.0 (licence of Earnings-22); audio licence not stated |
| Reference | Kocabiyikoglu et al., LREC 2022 ([arXiv:2207.08292](https://arxiv.org/abs/2207.08292)) | Panayotov et al., ICASSP 2015; lists: Le et al., Interspeech 2021 ([arXiv:2104.02194](https://arxiv.org/abs/2104.02194)) | Durmus et al. 2026 ([arXiv:2604.07354](https://arxiv.org/abs/2604.07354)) |

The dev splits (PxSLU dev: 282 utterances, 0.53 h; Contextual Earnings-22 dev: 100
clips, 0.42 h) only serve for tuning (e.g. the NeMo boosting weight,
`config_fr_nemo_alpha.yaml`): they are not in the runs of `config_fr.yaml` and
`config_en.yaml`, and are left out of the page (`build_page.py --exclude_datasets`). LibriSpeech
test-clean is prepared but left out of the runs: its speakers leave few errors on the
rare words to correct.

In every dataset, all the targets are in the lists (the lists are "targets +
distractors"): the benchmark measures how well a model uses a list that contains the
right terms, not what happens when a spoken term is missing from the list.

### PxSLU (French, drug names)

**What it is.** A corpus of spoken drug prescriptions in French: 1,981 recordings
(4.4 h) of 55 participants (doctors, other medical experts such as pharmacists or
biologists, and non-experts) saying drug prescriptions: drug, dosage, frequency,
duration ("Bactrim forte hum 1 comprimé le matin et 2 comprimés le soir pendant euh
14 jours"), sometimes only a part of it ("Nicopatch", "3 mois"). Recorded "in wild
conditions" according to the corpus documentation (16 kHz mono WAV), with
hesitations. Manual transcripts (lower case, no punctuation) and
entity labels (drug, dose, frequency, ...). The official test split is the doctors
only (570 recordings); dev is a part of the experts and non-experts.

**Preparation** (`prepare_pxslu.py`): the drug name spellings are corrected with the
corrections shipped with the corpus, the utterances with unidentifiable drug names
are removed (as in the cleaned version of the corpus), and so are the truncated words
("qu/") and 17 test recordings where a drug name is misspelled, misheard
or unusable (`lists/pxslu/excluded_recordings.txt`): 540 test utterances are left. Three
other names are corrected in the references (`lists/pxslu/reference_spelling.tsv`:
Sintrom, Levemir, Glibénèse). The transcripts are written in sentence case,
with the drug names in their usual case and one spelling per drug (brands
capitalized, active substances in lower case, acronyms in capitals; reviewed table
`lists/pxslu/drugs_pxslu_case.tsv`).

**Hotwords.** Drug names (brands and active substances), in global lists of growing
size, all containing the 389 drug names of the corpus:
- `pxslu`: the 389 drug names of the corpus (all splits), curated and made
  consistent with the corrected transcripts. 168 of them are spoken in the test split. It plays the role of a small domain list given to every request.
- `bdpm` (4,050 terms): brand names and active substances of the French public drug
  database (Base de données publique des médicaments, ANSM; homeopathic and herbal
  products left out), plus the targets missing from it. Licence Ouverte 2.0.
- `vidal` (2,753) and `theriaque` (8,851): drug names of the Vidal and Thériaque
  drug databases, plus the targets. Not in git: the terms of use of these databases
  forbid redistributing their content, so runs with them can only be reproduced
  by someone who has these lists.
- `pxslu25` (per utterance): 25 names of `pxslu`, the ones spoken in the utterance
  plus others drawn at random. It has the size of the per-call lists of Contextual
  Earnings-22, and fits in Whisper's 223-token hotwords.
- `pxslu_ext` (427 terms), `pxslu25_ext`: the same lists completed with the 38 drug
  names spoken in the test split that are not among the annotated names of the corpus
  (`lists/pxslu/drugs_pxslu_added.txt`, reviewed by hand). The targets are then the 427
  names (391 occurrences in 382 utterances).
- `pxslu_upper`, `pxslu25_upper`: the same lists in capitals (does a model follow
  the case of the list?).
- The manifests also hold per-utterance `oracle` lists (only the targets spoken in
  the utterance), not run: not a realistic setting (the list would have to know
  what is said).

**Advantages.** The only public French benchmark of this kind that we know of.
Real domain vocabulary that ASR models get wrong (F1 of the drug names without list:
31 to 57 depending on the model). Real dictionaries as lists, from a few hundred to
almost 9,000 terms: the "domain dictionary" use case. Short utterances, so every
model can process them.

**Drawbacks.** Small: 329 target occurrences in the test split, so differences of a
few F1 points are within the noise (see the confidence intervals of the page).
One kind of term only (drug names) and one speaking style (dictation). The
transcripts were made in lower case: the case of the references is reconstructed
(drug names and first letter), so the measures with case are only indicative for the
rest of the sentence. Two of the four global lists are not redistributable.

### LibriSpeech test-other (English, rare words)

**What it is.** Read English speech from public domain audiobooks (LibriVox), 16 kHz
FLAC. "test-other" holds the speakers that were harder for the recognizer used to
build the corpus (higher WER than "test-clean"). Reference transcripts in upper
case without punctuation.

**Hotwords.** The biasing lists of Le et al. (2021), released by Meta
(`facebookresearch/fbai-speech`, folder `is21_deep_bias`), the de facto standard of
the literature: one list per utterance, made of the rare words of its reference
(words outside the 5,000 most frequent words of the LibriSpeech training data,
audio transcripts and text-only corpus) plus N distractors drawn at random from the
other rare words, N in {100, 500, 1000, 2000}. Rare words are single words, in lower
case: names ("darcy", "cain") but also uncommon words ("balsam", "squalid") and
inflections of common ones ("sings", "grinned", "ambitions").

**Advantages.** Results comparable with the published ones (B-WER, U-WER, with the
scorer of the paper). The largest test set here (5,350 target occurrences: narrow
confidence intervals). The list size is the only thing that changes between 100 and
2,000 terms, which isolates its effect. Clean read speech: the errors come from the
words, not from the acoustics.

**Drawbacks.** Read speech of old books, far from production audio. Many rare words
are easy for modern models (F1 86 to 94 without list), so there is little to gain.
The distractors are random rare words, unrelated to the topic, easier to reject than
real ones. The lists are built from the reference (each list contains exactly the
rare words spoken, which no real system knows) and change with every utterance:
this costs the systems that compile the list (NeMo rebuilds its boosting tree for
each utterance, see "List sizes"). Too long for Whisper's hotwords.

### Contextual Earnings-22 (English, names in conversational speech)

**What it is.** Built by Argmax from Earnings-22 (Rev, [arXiv:2203.15591](https://arxiv.org/abs/2203.15591)): 125 English earnings calls
(119 h) of companies from all over the world, released as a benchmark of accented
speech. Argmax selected 55 calls, found the names of persons, companies and products
in their transcripts (named entity extraction with an LLM, then manual review of
each candidate, of the transcript and of the alignment), and cut 15-second clips
centered on mentions of these names. The released audio is 16 kHz WAV (in parquet
files). Test: 772 clips (771 used: one flagged by the dataset is left out), dev:
100.

**Hotwords.** Two lists per clip:
- `global` (run): the keywords of the whole call (25 on average, 1 to 46), of which
  about 7 % are spoken in the clip. It is what a system could get from the context
  of a meeting (participants, companies, products), and its distractors are
  realistic: other names of the same call.
- `local`: only the keywords spoken in the clip (the targets). Not run: an oracle
  setting, as the PxSLU `oracle` lists.

**Advantages.** The closest to production use: spontaneous conversational speech,
many accents, real names (the main use of hotwords, F1 65 to 81 without list),
realistic distractors, per-call lists. Short clips and short lists: every model can take them, Whisper
included.

**Drawbacks.** Short lists only (at most 46 terms): it does not test long lists.
Almost every clip contains a keyword (729 of 771), so the false alarms are measured
on keyword-dense audio and the gain is larger than on a whole call. The clips are
fixed 15-second windows that start and end in the middle of sentences. The
references are verbatim (hesitations "uh", word fragments "th-"), which weighs on the
WER of the models that leave them out. The keyword
inventories come from the transcripts (not from documents available before the
call). The figures of the paper (760 clips, 630 test) and of the released dataset
(872 clips, 772 test) differ, and the dataset card says the annotation is still in
progress. The licence of the audio is not stated by Rev (only the transcripts are
under CC BY-SA 4.0).

### Earnings-21 (English, not run yet)

**What it is.** A Rev corpus ([arXiv:2104.11348](https://arxiv.org/abs/2104.11348)) of 44 full earnings calls of 2020 (39.3 h; 18 to 96
minutes each), 5 calls in each of nine sectors (4 for conglomerates), MP3 files at
11 to 44 kHz. The transcripts (`.nlp` files) have speakers, punctuation, case and
entity tags (24 classes: persons, organizations, products, money, dates, ...), but no
word timestamps; speaker turns are given with times (RTTM files). Rev also defines
Eval-10, a representative 10-hour subset. Transcripts under CC BY-SA 4.0; the
licence of the audio is not stated.

**Hotwords.** Two sources:
- Rev's bias lists (Fox and Delworth, Interspeech 2022, [arXiv:2209.01250](https://arxiv.org/abs/2209.01250)), global for
  the whole corpus, in capitals: `oracle_list.txt` (1,013 words and phrases of the
  reference transcripts: names of persons and companies, acronyms, products) and
  `distractor_list.txt` (1,782: the same plus Fortune 500 companies and famous CEOs
  not in the calls).
- ConEC (Huang et al., LREC-COLING 2024, `github.com/huangruizhe/ConEC`): for each
  call, the words of its presentation slides and press release and the names of the
  participants, i.e. context available before the call; spoken terms are not
  guaranteed to be in these lists. ConEC also corrects the transcripts and adds word
  timestamps from an automatic alignment. No licence stated.

**Advantages.** Whole calls, as in production (long-form transcription). With ConEC,
the most realistic lists: built from real documents, with terms that are not spoken
and spoken names that are missing, the only setting here where the list is not
"targets + distractors". Rev's lists give long global lists (1,000 to 1,800 terms)
of real names, between the PxSLU dictionaries and the random LibriSpeech
distractors.

**Drawbacks.** Long recordings: most of the models here (LLM-based, limited to a few
minutes of audio) need the calls cut into segments, which needs word timestamps
(those of ConEC or a new alignment) and segmented references. Costly: 39 h per run
(Eval-10 or a few calls would be needed), and the long lists slow down the
prompt-based models a lot (see the RTFx in the page). Rev's lists are global to the
corpus (not per call) and contain only terms found in the references. Same domain as
Contextual Earnings-22 (earnings calls).

### Other datasets considered

- Plain Earnings-22 and Common Voice have no released lists (the Common Voice named
  entity lists of Thorbecke et al. 2024 were not released).
- More public benchmarks are listed in "State of the art" below.

## Models and mechanisms

| Model (backend) | How the list is given | Limits |
|---|---|---|
| Whisper large-v3 / large-v3-turbo (`faster-whisper`) | `hotwords` argument (prompt tokens before the previous text, on every 30 s window) | 223 tokens |
| Qwen3-ASR 1.7B (`vllm`) | `prompt` field, put in the system turn; template `Vocabulary: {terms}.` (the format is not documented by Qwen; this one is linto-stt's) | context length of the server |
| Nemotron-3.5-Transcribe (`vllm`, chat endpoint) | `{prompt} Context terms: {terms}.` (model card) | |
| Granite Speech 5.0 2B (`vllm`, chat endpoint) | `{prompt} Keywords: {terms}` (model card) | "up to 2000" keywords |
| MOSS-Transcribe-Diarize (`vllm`) | `{prompt}热词提示：{terms}` (French runs, default Chinese prompt) or `{prompt} Hotwords: {terms}` (English), as documented in the model's GitHub examples | |
| VibeVoice-ASR (`vibevoice-asr`) | extra context of the request | slow with long lists: an 8.3B model run with Hugging Face transformers (not vLLM), so the list is processed again with every batch, every generated token attends to it, and the batches must be made smaller to fit in memory (4 with `theriaque`). RTFx on LibriSpeech test-other: 21 without list, 5.9 with 500 terms, 1.7 with 2,000; below real time with the PxSLU dictionaries |
| Parakeet TDT 0.6B v3, Canary 1B v2, linto-stt-fr-fastconformer (RNN-T decoder; French only) (`nemo`) | GPU phrase boosting of NeMo (boosting tree fused at decoding time, `greedy_batch`, beam search for Canary; Andrusenko et al. 2025, [arXiv:2508.07014](https://arxiv.org/abs/2508.07014)); weight `contextual_biasing_alpha` 1.0 and 2.0 (PxSLU dev with Parakeet, `config_fr_nemo_alpha.yaml`: 1.0 best F1 / false alarm trade-off, 2.0 +5 F1 with 5x false alarms) | each new list costs a rebuild of the decoding: the boosting tree is built (about 60 ms for 389 terms, 140 ms for 1,000, 0.8 s for 8,851, on a DGX Spark) and the decoding strategy is set again, and the RNN-T / TDT greedy decoding captures its CUDA graphs again at the next batch. Free with a fixed list; with a list per utterance, no batching across utterances and this cost for each one (LibriSpeech, Parakeet: RTFx about 150 without list, 11 with the 500 lists) |
| Cohere Transcribe (`vllm`) | none: baseline | |

Not included: Voxtral (open weights have no mechanism; `context_bias` exists only
in Mistral's hosted API). Through vLLM's `/v1/audio/transcriptions` endpoint, the
`prompt` field reaches Qwen3-ASR, Whisper and MOSS, but Nemotron and Granite ignore
it: they are run through the chat endpoint.

List sizes (tokens): PxSLU `pxslu25` at most ~130 (~160 in capitals), `pxslu` ~1,700 (Whisper tokenizer), `bdpm` ~17.8k,
`vidal` ~12k, `theriaque` ~38k; LibriSpeech N=100 ~370, N=2000 ~7.2k; Contextual
Earnings-22 at most 161. The configs leave out the lists a model cannot take
(Whisper: `pxslu25`, the Contextual Earnings-22 lists, and two lists given anyway and
cut to their first 223 tokens, unrelated to the targets as the lists are sorted
alphabetically: the LibriSpeech N=100 lists, ~60 % of each, and `pxslu`, its first 51
names of 389; Granite: no list above 16k tokens). Left out for time: NeMo with the
LibriSpeech N=1000 and 2000 lists (the boosting tree is rebuilt for each utterance:
2.8 h for Parakeet with N=1000 on test-clean and test-other), and N=500 at alpha 2.0.
VibeVoice-ASR with the longest lists (`theriaque`, LibriSpeech N >= 500) runs last:
it processes them at about real time or slower.

## Preparation

The manifests (`manifests/*.jsonl`) and the redistributable PxSLU lists
(`lists/pxslu/`) are in git: to run the benchmark, only the audio is needed (paths
relative to `AUDIO_FOLDER`), plus the LibriSpeech lists (230 MB, not in git: run
`prepare_librispeech.py`). The runs with the `vidal` and `theriaque` lists need these
private lists. To rebuild everything, where the audio is (e.g. on the DGX machines,
under `/home/data/audio`):

```bash
# PxSLU (BDPM files: https://base-donnees-publique.medicaments.gouv.fr/telechargement)
mkdir bdpm && for f in CIS_bdpm.txt CIS_COMPO_bdpm.txt; do
    curl -L -o bdpm/$f https://base-donnees-publique.medicaments.gouv.fr/download/file/$f; done
uv run python tools/contextual_biasing/prepare_pxslu.py \
    --pxslu_dir /home/data/audio/raw/transcript/fr/PxSLU \
    --drug_lists_dir <folder of drug_PxSLU.txt, drug_vidal.txt, drug_theriaque.txt> \
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

They write `manifests/*.jsonl` and `lists/` in this folder.

## Running

```bash
AUDIO_FOLDER=/home/data/audio uv run python benchmarker.py benchmarks/contextual_biasing/config_fr.yaml
uv run python tools/contextual_biasing/report.py benchmarks/contextual_biasing/results_fr
uv run python tools/contextual_biasing/report.py benchmarks/contextual_biasing/results_fr --mode cased
uv run python tools/contextual_biasing/timings_from_logs.py logs/*.log \
    --results benchmarks/contextual_biasing/results_fr benchmarks/contextual_biasing/results_en
uv run python tools/contextual_biasing/build_page.py --output benchmarks/contextual_biasing/page \
    --results fr=benchmarks/contextual_biasing/results_fr en=benchmarks/contextual_biasing/results_en
```

`report.py` prints, per dataset and list, the measures of each model without list,
with the list, and the difference (optionally as CSV). `build_page.py` writes the same
measures as an HTML page (`page/index.html`, a single file): leaderboard of a measure
for each model and list, plots of WER by B-WER and of WER, B-WER, F1 and RTFx by list
size, charts of each list without and with it, and all the measures of one list; both
text normalizations, and a selection of the models shown (NeMo runs at alpha 1.0 and
2.0; the other weights of the tuning on PxSLU dev, and the Whisper runs with the
hotwords separated by spaces instead of commas, are left out, `--exclude_settings`;
the Parakeet run with the LibriSpeech N=1000 lists, the only NeMo run with them, is
left out too, `--exclude_runs`). An empty cell of the leaderboard means no run with this
list; it is marked with a dagger when the run was left out only because the model is
too slow with the list (`TOO_SLOW` in `template.html`). A model trained on a test set is marked with
a double dagger on it, listed last and left out of its ranking and of the aggregated
test sets (`TRAINED_ON` in `template.html`): the Luciole models on PxSLU. The training
data published for Luciole-Audio (`OpenLLM-France/Luciole-Audio-Training-Dataset`,
`PxSLU_fr`) holds all the PxSLU test and dev recordings; this is presumed, not
checked, for the `v5_transcribe_v2` checkpoint.

After the test sets, two aggregated test sets, "English" (Contextual Earnings-22 and
LibriSpeech test-other) and "French and English" (the same plus PxSLU test), give the
mean over the test sets, each weighing the same (geometric mean for RTFx). As the test
sets have different lists, the columns are size classes of the lists: `short`
(`pxslu25`, `global`, `100`), `medium` (`pxslu`, `500`), `long` (`vidal`, `bdpm`,
`theriaque`, `1000`, `2000`). The lists of a class on one test set are averaged first;
a class is kept when at least two test sets have lists of it (English: `short` only).
A model has a value only if it has one on every test set of the class, and with the
list only with a run with each of its lists (`SIZE_CLASSES` and `AGGREGATES` in
`template.html`). The confidence interval of the mean WER, used by the ranking,
combines those of the test sets as independent errors.

Ranking of the leaderboard, on one list ("Rank on"; by default `pxslu`, Contextual
Earnings-22 `global`, LibriSpeech `100`, `short` for the aggregated test sets): each model with this list, or without list
when it has no run with it (no biasing mechanism, or list too long). First by WER, with
a tolerance: the models whose WER is within the 95 % confidence interval of the best
WER form a level, and the next level starts from the best of the others; within a
level, by B-WER (the measure of the literature, in the unit of the WER, which counts
both the missed terms and the inserted list words).

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
  `{terms}`, the terms joined with `contextual_biasing_separator`, `", "` by
  default, and, for some backends, `{prompt}`; another separator adds `-spaces`,
  or `-sep<...>`, to the result folder);
  per backend: `contextual_biasing_mode` (faster-whisper: `hotwords` or
  `initial_prompt`), `contextual_biasing_field` (vLLM transcriptions: `prompt` or
  `hotwords`), `contextual_biasing_alpha`, `contextual_biasing_context_score`,
  `contextual_biasing_depth_scaling` (NeMo phrase boosting).
- Backends with a mechanism: faster-whisper, openai-whisper, qwen3-asr, vllm
  (transcriptions and chat endpoints), vibevoice-asr, moss, nemo (GPU phrase
  boosting), linto-stt (whisper engine).
- Without the key, nothing changes for the standard ASR benchmark.

## Measures

All the measures are without punctuation marks, removed after the conversion of the
numbers, acronyms, units... ("1,5 mg" -> "un virgule cinq milligramme", "S.N.C.F." ->
"SNCF"), with the normalization of ssak (`normalize_text(..., for_wer=True)`).

The WER is given without case (`wer_nocasepunc`) and with case (`wer_nopunc`: the case
of all the words counts; only when the references have lower and upper case, else the
page shows the WER without case, e.g. for LibriSpeech).

In `performances/<dataset>.json`, key `contextual_biasing`, for each list and two
normalizations (`nocasepunc`: without case; `cased`: with case):

- `precision`, `recall`, `f1` of the target terms: for each term and utterance,
  tp = min(#ref, #hyp), fp = max(0, #hyp - #ref), fn = max(0, #ref - #hyp), where #
  counts the occurrences of the term (one or more words). A term has at most one
  false alarm (fp) per utterance, so that a transcription that loops on a term does not
  dominate the score.
- `b_wer`, `u_wer` (Le et al. 2021): WER on the reference words that belong to a
  list term, and on the other words; an inserted word counts in B-WER if it
  belongs to a list term. With multi-word terms, a word is biased only when it is
  part of an occurrence of a term. The errors are those of the WER, split in two
  (before its cap at 100 % per utterance).
- `false_alarms` (and per hour): occurrences of list terms in the prediction
  beyond their count in the reference (e.g. distractors written although not
  spoken), at most one per term and utterance.

`report.py` and the page also give the runaway rate of each run (from its
predictions, not stored in `performances/`): the percentage of utterances whose
transcription has more than twice as many words as the reference, plus 10 (loops,
invented sentences, prompt recited). A few such outputs weigh a lot on B-WER
(inserted list terms count in it), FA/h and WER: e.g. on PxSLU test, Qwen3-ASR
recites the start of the `bdpm` list in 2 of the 540 utterances (B-WER above 100 %),
and 18 transcriptions of MOSS run away with `pxslu`.

RTFx (seconds of audio per second of processing; `report.py` and the page), for each
run and dataset, from: the transcription time that the benchmark records in
`metadata.json` (`transcription_time`: seconds, audio seconds and files of each
dataset, model loading and scoring excluded); else, for the runs made before, the
throughput measured by the `vllm` backend (`throughput_rtfx` of the predictions); else
the benchmark logs (`timings_from_logs.py` writes `timings_from_logs.json` in the
results folder: time of the whole run, model loading and scoring included, so an
approximate value, shown with "~"). The speeds are those of the benchmark runs on one
DGX Spark (vLLM servers with 32 to 128 concurrent requests, NeMo and VibeVoice-ASR in
batches, faster-whisper one file at a time): they compare a model with and without a
list more than the models with each other.

NeMo runs also record, in `metadata.json` (`contextual_biasing_tree_time_ms`), the
time to build the boosting tree and set it in the decoding (count, first, median,
max, total, in ms): once for a global list, once per utterance for per-utterance
lists. It is what a server pays each time the hotwords of a request change.

Case (`cased`): terms are found without case. For `precision`, `recall` and `f1`, an
occurrence in the prediction is correct only when written as in the list given to the
model (else as in the targets), with the first letter of the utterance as in the
reference. So a model given "DOLIPRANE" has to write "DOLIPRANE" (`*_upper` lists:
does the model follow the case of the terms?). `b_wer`, `u_wer` and the WER compare the
case of all the words with the one of the reference (without case when the references
have no case): with the `*_upper` lists, look at F1. The
other datasets are used as published: compare them with the literature in
`nocasepunc`, and approximately: the words are those of the WER, after the
normalization of the numbers, spelling..., split at apostrophes ("eden's" is two words,
unlike in the scorer of Le et al.).

## State of the art

Public benchmarks with released lists (checked in the primary sources in November
2026):

| Benchmark | Lang. | Lists | Measures |
|---|---|---|---|
| LibriSpeech biasing lists (Le et al. 2021, [arXiv:2104.02194](https://arxiv.org/abs/2104.02194)) | en | per utterance: rare words + N distractors (the de facto standard) | WER, B-WER, U-WER |
| Earnings-21 bias lists (Fox and Delworth 2022, [arXiv:2209.01250](https://arxiv.org/abs/2209.01250)) | en | global oracle and distractor lists | rare-word recall, WER |
| ConEC (Huang et al. 2024) | en | per call, real context (slides, press releases, names) | WER on rare words and entities |
| Contextual Earnings-22 (Durmus et al. 2026, [arXiv:2604.07354](https://arxiv.org/abs/2604.07354)) | en | per clip, local and global | WER, keyword P/R/F |
| SlideSpeech (Wang et al. 2024, [arXiv:2309.05396](https://arxiv.org/abs/2309.05396)) | en | per segment, keywords from slides | WER, B-WER, U-WER |
| ContextASR-Bench (Wang et al. 2025, [arXiv:2507.05727](https://arxiv.org/abs/2507.05727)) | en, zh | per utterance, synthetic speech | WER, NE-WER, NE-FNR |
| Test/Dev-Aishell1-NE (SeACo-Paraformer, [arXiv:2308.03266](https://arxiv.org/abs/2308.03266)) | zh | global named entities; "R1" subset of terms the baseline misses | CER, hotword R/P/F1 |
| IndicContextEval (Joshi et al. 2026, [arXiv:2606.19157](https://arxiv.org/abs/2606.19157)) | 8 Indic languages | context levels, in-domain and adversarial entities | WER, NEER |

Other Mandarin sets: ATPC hotword-161, WenetSpeech biased words, AISHELL-NER and
derived sets, ST-cmds-NE. Protocols without released lists: Common Voice NER lists
(Thorbecke et al. 2024, [arXiv:2409.13499](https://arxiv.org/abs/2409.13499); en, ca, fr, de, es, it), NVIDIA's
evaluations of phrase boosting ([arXiv:2406.07096](https://arxiv.org/abs/2406.07096), [arXiv:2508.07014](https://arxiv.org/abs/2508.07014)), Microsoft
LOGIC (internal data, 11 locales including fr-FR).

No public French contextual biasing benchmark was found: the PxSLU part of this
benchmark, with publishable lists (`pxslu`, `bdpm`), fills that gap. Worth adding
later: a "hard" subset (terms the baseline misses, as SeACo-Paraformer and NVIDIA
do), and a Common Voice FR protocol following Thorbecke et al.
