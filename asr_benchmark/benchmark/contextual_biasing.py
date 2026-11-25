"""Contextual biasing: lists of terms (names, jargon, drug names...) given to the
model at inference time, and the measures of how well these terms are transcribed.

Manifest fields (all optional, a manifest without them is a regular ASR manifest):

- ``contextual_biasing``: the biasing lists of the utterance, as a dict
  {list name: list} (e.g. {"100": [...], "1000": [...]} for lists with 100 or 1000
  distractors, or {"local": [...], "global": [...]}). A list is given inline (list of strings), as the path of a text file with one
  term per line (a global list shared by a whole dataset), or as "file.json#key":
  the list stored under `key` in a JSON dict {key: list of terms} (per-utterance
  lists of a whole dataset in one file, to keep manifests small).
- ``contextual_biasing_targets``: the terms to score (same forms: inline list or
  file). Default: the biasing list itself. Useful when the list given to
  the model contains many words that are not targets (large dictionaries), or to
  score every condition on the same targets. A dict {list name: terms} gives targets
  per list, with those of the other lists under "default".

The config key ``contextual_biasing`` of a benchmark selects the list given to the
model: null (no list: the baseline) or the name of a list. In every condition, the transcriptions are scored on
the lists, so that the gain of giving a list can be measured (see score_dataset).

Measures, for each list and two text normalizations ("nocasepunc": lower
case without punctuation, as for WER; "cased": case kept, punctuation removed):

- term precision / recall / F1 on the targets: for each term and utterance,
  tp = min(#ref, #hyp), fp = max(0, #hyp - #ref), fn = max(0, #ref - #hyp), where
  # counts the occurrences of the term (a sequence of words) in the reference and
  in the prediction. Same definition as ssak's compute_wer(words_list=...).
In "cased" mode, the terms are found without case, and for precision / recall / F1
an occurrence in the prediction is correct only when written as in the list given to
the model (else as in the targets), except for the first letter of the utterance,
written as in the reference. So a model given "DOLIPRANE" has to write "DOLIPRANE",
and an active substance at the start of a sentence-case reference is expected
capitalized. The word error rates (WER, B-WER, U-WER) compare the case of all the
words with the one of the reference (without case when the references have no case).
The words are those of the WER (split at apostrophes: "eden's" is two words, unlike in
the scorer of Le et al., so the values are not exactly comparable with the published ones).

- B-WER / U-WER (Le et al., Interspeech 2021, arXiv:2104.02194): word error rate
  on the reference words that belong to the biasing list, and on the other words.
  An inserted word counts in B-WER if it belongs to the list, in U-WER otherwise. The
  errors are those of the WER, split in two.
  With multi-word terms, a word is "biased" when it is part of an occurrence of a
  term (so that "un" is not biased because of the term "un-alfa").
- false alarms: occurrences of list terms in the prediction beyond their count
  in the reference (for instance distractors written although not spoken).
"""

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import jiwer

logger = logging.getLogger(__name__)

MANIFEST_KEY = "contextual_biasing"
TARGETS_KEY = "contextual_biasing_targets"

_NOT_WORD = re.compile(r"[^\w]+")


@lru_cache(maxsize=64)
def _read_terms_file(path):
    """Terms of a text file: one per line, empty lines and lines starting with '#' skipped."""
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    return tuple(_dedup(terms))


def _dedup(terms):
    seen = set()
    out = []
    for term in terms:
        term = " ".join(str(term).split())
        if term and term not in seen:
            seen.add(term)
            out.append(term)
    return out


@lru_cache(maxsize=2)
def _read_json_lists(path):
    """{key: deduplicated terms} of a JSON file of lists. Few files kept in memory: a
    file of per-utterance lists can take hundreds of MB (LibriSpeech, 2000 distractors)."""
    with open(path, "r", encoding="utf-8") as f:
        return {key: tuple(_dedup(terms)) for key, terms in json.load(f).items()}


def load_terms(value):
    """List of terms from a manifest value: inline list, path of a text file, or
    "file.json#key"."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return _dedup(value)
    if isinstance(value, str):
        if "#" in value and value.split("#", 1)[0].endswith(".json"):
            path, key = value.split("#", 1)
            if not Path(path).is_file():
                raise FileNotFoundError(f"Contextual biasing lists not found: {path}")
            lists = _read_json_lists(str(Path(path).resolve()))
            if key not in lists:
                raise KeyError(f"No contextual biasing list '{key}' in {path}")
            return list(lists[key])
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Contextual biasing list not found: {value}")
        return list(_read_terms_file(str(path.resolve())))
    raise ValueError(f"Invalid contextual biasing list (expected a list or a file path): {value!r}")


def row_variants(row):
    """{list name: raw value} of the biasing lists of a manifest row ({} if none)."""
    value = row.get(MANIFEST_KEY) or {}
    if not isinstance(value, dict):
        raise ValueError(f"'{MANIFEST_KEY}' of row {row.get('id')} must be a dict {{list name: list}}")
    return {str(k): v for k, v in value.items()}


def config_variant(config):
    """Name of the list given to the model according to the config, None for no list."""
    value = config.get("contextual_biasing")
    if value is None or str(value).lower() in ("none", "null", "false", ""):
        return None
    return str(value)


def applies_to(row, config):
    """Whether a run with this config concerns a manifest row: always without list,
    else only when the row has the selected list (a benchmark can mix datasets with
    different lists, e.g. LibriSpeech "100"/"1000" and Contextual Earnings-22 "local")."""
    variant = config_variant(config)
    return variant is None or variant in row_variants(row)


def terms_for_row(row, config):
    """Terms to give to the model for this row (None when the config gives no list)."""
    variant = config_variant(config)
    if variant is None:
        return None
    variants = row_variants(row)
    if not variants:
        raise ValueError(
            f"contextual_biasing is set ({variant}) but the row {row.get('id')} of {row.get('name')} "
            f"has no '{MANIFEST_KEY}' field"
        )
    if variant not in variants:
        raise ValueError(
            f"Contextual biasing list '{variant}' not found for row {row.get('id')} "
            f"(available: {sorted(variants)})"
        )
    return load_terms(variants[variant])


def folder_tag(config):
    """Suffix of the result folder: "_cb-<list>" when a list is given, else ""."""
    variant = config_variant(config)
    return "" if variant is None else re.sub(r"[^\w.\-]", "-", f"_cb-{variant}")


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def _make_normalizers(language):
    """{mode: function text -> list of tokens}."""
    from ssak.utils.text import normalize_text

    # Same typographic punctuation folding as the WER (imported here: benchmark_manager
    # imports this module)
    from asr_benchmark.benchmark.benchmark_manager import normalize_punctuation

    def words(text, case_sensitive):
        # Same normalization as the WER
        return normalize_text(
            normalize_punctuation(text), language, case_sensitive=case_sensitive, remove_apostrophes_and_dashes=True, for_wer=True
        ).split()

    def nocasepunc(text):
        return words(text, case_sensitive=False)

    def cased(text):
        return words(text, case_sensitive=True)

    return {"nocasepunc": nocasepunc, "cased": cased}


_DIGIT = re.compile(r"\d")
_LIGATURES = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae"})


def _quick_words(text):
    """Set of the words of a text, lower case, without accents nor ligatures."""
    text = unicodedata.normalize("NFKD", text.translate(_LIGATURES).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return frozenset(_NOT_WORD.sub(" ", text).split())


def _count_ngrams(tokens, phrases_by_length, text=None, expected=None):
    """{phrase (tuple of tokens): number of occurrences in tokens}, for the given phrases.
    With text and expected (prediction tokens), also the counts of the occurrences
    written as expected."""
    counts, written = {}, {}
    for n, phrases in phrases_by_length.items():
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            if ngram in phrases:
                counts[ngram] = counts.get(ngram, 0) + 1
                if text is not None and text.written_as(i, expected[ngram]):
                    written[ngram] = written.get(ngram, 0) + 1
    return counts if text is None else (counts, written)


def _by_length(phrases):
    out = {}
    for phrase in phrases:
        out.setdefault(len(phrase), set()).add(phrase)
    return out


def _phrase_tokens(tokens, phrases_by_length, expected):
    """For each token: its expected form when it is part of an occurrence of one of
    the phrases (expected: {phrase: tokens as they should be written}), else None."""
    forms = [None] * len(tokens)
    for n, phrases in phrases_by_length.items():
        for i in range(len(tokens) - n + 1):
            phrase = tuple(tokens[i:i + n])
            if phrase in phrases:
                forms[i:i + n] = expected[phrase]
    return forms


def _with_first_letter(form, upper):
    return form[:1].upper() + form[1:] if upper else form


def _split_errors(text, phrases_by_length, expected):
    """(B errors, B count, U errors, U count) of one utterance. A reference word is
    biased when it is part of an occurrence of a list term in the reference, an
    inserted word when it is part of an occurrence of a list term in the prediction
    (for single-word terms: the word belongs to the list, as in Le et al. 2021;
    multi-word terms do not make their common words, like "un" in "un-alfa", biased).
    The errors are those of the WER: B errors + U errors = errors of the WER."""
    b_err = b_cnt = u_err = u_cnt = 0
    ref_forms = _phrase_tokens(text.ref, phrases_by_length, expected)
    hyp_biased = [f is not None for f in _phrase_tokens(text.hyp, phrases_by_length, expected)]
    if not text.ref:
        b_err = sum(hyp_biased)
        return b_err, 0, len(text.hyp) - b_err, 0
    for chunk in text.alignment:
        if chunk.type == "insert":
            for biased in hyp_biased[chunk.hyp_start_idx:chunk.hyp_end_idx]:
                if biased:
                    b_err += 1
                else:
                    u_err += 1
            continue
        for k in range(chunk.ref_start_idx, chunk.ref_end_idx):
            error = chunk.type != "equal"
            if ref_forms[k] is None:
                u_cnt += 1
                u_err += error
            else:
                b_cnt += 1
                b_err += error
    return b_err, b_cnt, u_err, u_cnt


def _pct(num, den):
    return round(100 * num / den, 3) if den else None


class _Text:
    """Tokens of a reference and a prediction: in lower case (to find the terms), and as
    written (prediction), with the case of the reference's first letter. The alignment is
    the one of the WER (with case, or not)."""

    def __init__(self, ref_tokens, hyp_tokens, case_sensitive):
        self.ref = [t.lower() for t in ref_tokens]
        self.hyp = [t.lower() for t in hyp_tokens]
        self.hyp_written = hyp_tokens
        self.upper_start = bool(ref_tokens) and ref_tokens[0][:1].isupper()
        ref, hyp = (ref_tokens, hyp_tokens) if case_sensitive else (self.ref, self.hyp)
        self.alignment = jiwer.process_words(" ".join(ref), " ".join(hyp)).alignments[0] if ref else None

    def written_as(self, start, form):
        """Is the occurrence of a phrase at this position of the prediction written as `form`?"""
        form = list(form)
        if start == 0:
            form[0] = _with_first_letter(form[0], self.upper_start)
        return list(self.hyp_written[start:start + len(form)]) == form


class _Normalized:
    """Normalized texts and terms of one normalization mode, shared by the list
    variants (the lists of a dataset overlap: e.g. 100 to 2000 distractors)."""

    def __init__(self, normalize, key_cache, case_sensitive=True):
        self.normalize = normalize
        self.case_sensitive = case_sensitive  # of the word error rates
        self.phrases = {}  # term -> normalized tokens, as written
        self.keys = key_cache  # term -> its words without case and accents (quick pre-filter)
        self.texts = {}  # id -> (_Text, words)

    def text(self, i, ref, hyp):
        if i not in self.texts:
            ref_tokens, hyp_tokens = self.normalize(ref), self.normalize(hyp)
            words = _quick_words(ref) | _quick_words(hyp) | _quick_words(" ".join(ref_tokens + hyp_tokens))
            self.texts[i] = (_Text(ref_tokens, hyp_tokens, self.case_sensitive), words)
        return self.texts[i]

    def candidates(self, terms, words):
        """{phrase in lower case: phrase as written in the list} of the terms that may
        occur in a text with these words (normalizing every term of large lists for
        every utterance would be slow). Terms with digits are always kept: the
        normalization may spell them out."""
        out = {}
        for term in terms:
            if term not in self.keys:
                self.keys[term] = None if _DIGIT.search(term) else _quick_words(term)
            key = self.keys[term]
            if key is None or key <= words:
                if term not in self.phrases:
                    self.phrases[term] = tuple(self.normalize(term))
                if self.phrases[term]:
                    out[tuple(t.lower() for t in self.phrases[term])] = self.phrases[term]
        return out


def score_variant(items, norm, duration):
    """Measures of one list variant and one normalization.

    items: list of (id, reference, prediction, biasing terms, target terms);
    norm: _Normalized of the normalization."""
    tp = fp = fn = 0
    fa = 0
    b_err = b_cnt = u_err = u_cnt = 0
    list_sizes = []
    for i, ref, hyp, terms, targets in items:
        text, words = norm.text(i, ref, hyp)
        list_phrases = norm.candidates(terms, words)
        target_phrases = list_phrases if targets is terms else norm.candidates(targets, words)
        expected = target_phrases | list_phrases  # written as in the list given, else as the target
        list_sizes.append(len(terms))

        all_phrases = _by_length(expected)
        ref_counts = _count_ngrams(text.ref, all_phrases)
        hyp_counts, hyp_written = _count_ngrams(text.hyp, all_phrases, text, expected)

        for p in target_phrases:
            r, h = ref_counts.get(p, 0), hyp_written.get(p, 0)
            tp += min(r, h)
            fp += max(0, h - r)
            fn += max(0, r - h)
        for p in list_phrases:
            fa += max(0, hyp_counts.get(p, 0) - ref_counts.get(p, 0))

        be, bc, ue, uc = _split_errors(text, all_phrases, expected)
        b_err, b_cnt, u_err, u_cnt = b_err + be, b_cnt + bc, u_err + ue, u_cnt + uc

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(100 * precision, 3),
        "recall": round(100 * recall, 3),
        "f1": round(100 * f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "b_wer": _pct(b_err, b_cnt),
        "u_wer": _pct(u_err, u_cnt),
        "b_count": b_cnt,
        "u_count": u_cnt,
        "false_alarms": fa,
        "false_alarms_per_hour": round(fa * 3600 / duration, 3) if duration else None,
        "list_size_mean": round(sum(list_sizes) / len(list_sizes), 1) if list_sizes else 0,
    }


def score_dataset(data, rows, config, language="fr"):
    """Contextual biasing measures of one dataset.

    data: predictions {id: {"text", "prediction", "audio_duration", ...}};
    rows: manifest rows {id: row}. Returns None when the rows have no biasing list.
    When the config gives a list to the model, only that variant is scored;
    otherwise (baseline) every variant of the manifest is."""
    ids = [i for i in data if i in rows and row_variants(rows[i])]
    if not ids:
        return None
    if len(ids) < len(data):
        logger.warning(f"{len(data) - len(ids)} predictions have no contextual biasing list: not scored")
    from asr_benchmark.benchmark.benchmark_manager import has_case

    given = config_variant(config)
    names = [given] if given is not None else list(row_variants(rows[ids[0]]))
    duration = sum(data[i].get("audio_duration") or 0 for i in ids)
    key_cache = {}
    # Word error rates without case when the references have no case, as for the WER
    case_sensitive = has_case([row["text"] for row in data.values()])
    norms = {mode: _Normalized(normalize, key_cache, case_sensitive) for mode, normalize in _make_normalizers(language).items()}
    # Targets (the same for every list variant unless given per list), None: the list itself
    def targets_of(row, name):
        value = row.get(TARGETS_KEY)
        return load_terms(value.get(name, value.get("default")) if isinstance(value, dict) else value)

    results = {}
    for name in names:
        items = []
        for i in ids:
            row_lists = row_variants(rows[i])
            if name not in row_lists:
                raise ValueError(f"Contextual biasing list '{name}' missing for row {i}")
            terms, targets = load_terms(row_lists[name]), targets_of(rows[i], name)
            items.append((i, data[i]["text"], data[i]["prediction"], terms, terms if targets is None else targets))
        results[name] = {mode: score_variant(items, norm, duration) for mode, norm in norms.items()}
    return results


@lru_cache(maxsize=1)
def manifest_rows(input_manifest, input_audios_paths=""):
    """{dataset name: {id: row}} of a manifest, for the scoring (read once for all the
    models of a benchmark)."""
    import asr_benchmark.utils.benchmark as utils

    rows = {}
    if input_manifest and Path(input_manifest).is_file():
        for row in utils.get_data(input_manifest, input_audios_paths):
            rows.setdefault(row["name"], {})[row["id"]] = row
    return rows
