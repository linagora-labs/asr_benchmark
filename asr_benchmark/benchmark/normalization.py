"""Normalization of the texts before scoring (WER, CER, contextual biasing measures)."""
import re
import unicodedata

from ssak.utils.text import normalize_text
from ssak.utils.text_numbers import base_language

# Typographic punctuation that models emit but references usually write in ASCII
# (or the reverse). Mapped to the ASCII equivalent before tokenizing, otherwise
# e.g. "c\u2019est" and "c'est" yield different tokens and every French elision
# counts as an error in the punctuated wer/cer.
UNICODE_PUNCTUATION = {
    "\u2019": "'",   # right single quotation mark, the usual ASR apostrophe
    "\u2018": "'",   # left single quotation mark
    "\u02bc": "'",   # modifier letter apostrophe
    "\u2032": "'",   # prime
    "\u201c": '"',   # left double quotation mark
    "\u201d": '"',   # right double quotation mark
    "\u00ab": '"',   # left-pointing double angle quotation mark
    "\u00bb": '"',   # right-pointing double angle quotation mark
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2212": "-",   # minus sign
    "\u2026": "...", # horizontal ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u202f": " ",   # narrow no-break space (French thin space before ; : ! ?)
}

_UNICODE_PUNCTUATION_RE = re.compile("|".join(map(re.escape, UNICODE_PUNCTUATION)))


def normalize_punctuation(text):
    """
    Map typographic punctuation to its ASCII equivalent.

    Example: "c\u2019est \u2014 oui\u2026" -> "c'est - oui..."
    """
    return _UNICODE_PUNCTUATION_RE.sub(lambda m: UNICODE_PUNCTUATION[m.group()], text)


# Languages normalized by ssak (normalize_text). The others (Arabic in the sota_multilingual
# benchmark) use num2words and normalize_generic(): ssak's Arabic normalization needs
# lang_trans and data files (assets/) that are not installed with the package.
SSAK_LANGUAGES = ("fr", "en", "es", "de", "it", "pt", "nl")
DECIMAL_WORDS = {"ar": "فاصلة"}
_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:[ .  ]\d{3})+\b")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Arabic: short vowels / shadda / sukun (written or not, FLEURS mixes both), tatweel,
# hamza forms of alef (أ إ آ -> ا) that references and models write inconsistently.
_ARABIC_DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ARABIC_ALEF_RE = re.compile(r"[آأإٱ]")


def spell_numbers(text, language):
    """"23,5" -> "ventitré virgola cinque": digits in letters with num2words, thousands
    groups ("10 000", "10.000") read as one number. Left unchanged when num2words fails."""
    from num2words import num2words

    def spell(m):
        parts = re.split(r"[.,]", m.group(), maxsplit=1)
        try:
            words = num2words(int(parts[0]), lang=language)
            if len(parts) == 2:
                decimals = parts[1]
                zeros = len(decimals) - len(decimals.lstrip("0"))
                tail = [num2words(0, lang=language)] * zeros + ([num2words(int(decimals), lang=language)] if decimals.lstrip("0") else [])
                words = " ".join([words, DECIMAL_WORDS.get(language, ",")] + tail)
        except (ValueError, NotImplementedError, OverflowError):
            return m.group()
        return f" {words} "

    text = _THOUSANDS_RE.sub(lambda m: re.sub(r"\D", "", m.group()), text)
    return re.sub(r"\s+", " ", _NUMBER_RE.sub(spell, text)).strip()


def normalize_generic(text, language, case_sensitive=False):
    """Normalization of the languages ssak does not handle: numbers in letters, lower case
    (unless case_sensitive), punctuation and symbols (apostrophes and hyphens included) replaced by
    spaces, accents kept. Arabic: diacritics and tatweel removed, alef forms merged.

    Example (ar): "دفع 15 دولاراً." -> "دفع خمسة عشر دولارا"
    """
    text = unicodedata.normalize("NFKC", normalize_punctuation(text))
    text = spell_numbers(text, language)
    if not case_sensitive:
        text = text.lower()
    if language == "ar":
        text = _ARABIC_ALEF_RE.sub("ا", _ARABIC_DIACRITICS_RE.sub("", text))
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def separate_punctuation(text):
    """
    Example: "Hello,world!" -> "Hello , world !"
    """
    text = normalize_punctuation(text)
    # ¿ ¡ (Spanish), ، ؟ ؛ (Arabic): sota_multilingual
    text = re.sub(r'([.,!?;:()\"\'\-¿¡،؟؛])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_for_wer(text, language, case_sensitive=False, keep_punctuation=False):
    """Normalization of the texts for all the scores (WER, CER, contextual biasing), in any
    language: numbers in letters, apostrophes and dashes removed, punctuation removed or
    separated from the words."""
    language = base_language(language)
    if language in SSAK_LANGUAGES:
        return normalize_text(
            normalize_punctuation(text), language, case_sensitive=case_sensitive, keep_punctuation=keep_punctuation,
            remove_apostrophes_and_dashes=True, for_wer=True,
        )
    if keep_punctuation:
        # Numbers first: "10.000" must not be split. The case is kept.
        return separate_punctuation(spell_numbers(text, language))
    return normalize_generic(text, language, case_sensitive)
