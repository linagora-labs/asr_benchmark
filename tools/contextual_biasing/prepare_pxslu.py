"""Build the PxSLU manifests and drug lists of the contextual biasing benchmark.

PxSLU (Kocabiyikoglu et al., LREC 2022, CC BY 4.0) is a French corpus of spoken
drug prescriptions: 1,981 recordings with transcripts and BIO entity labels.
This script:

1. reads the raw corpus (paths.txt, seq.in, seq.out) and its official split
   (data_splits/<split>/<split>.txt: 1-based line numbers of the raw corpus);
2. fixes drug name spellings with the corrections of the dataset
   (scripts/drug_name_correction_V4.txt), as whole words, in a single pass
   (the dataset's own clean_PxSLU.py replaces substrings one after another,
   which creates words like "alalepsal" or "amikacinee");
3. removes the utterances with unidentifiable drug names
   (scripts/drug_name_exclusion.txt, same rule as clean_PxSLU.py);
4. removes truncated words (tokens containing "/", e.g. "qu/", "/pone");
5. writes the drug lists (one term per line), in their usual case: brand names
   capitalized (each part of a hyphenated name), active substances in lower
   case, acronyms in capitals:
   - drugs_pxslu.txt: the drug names of PxSLU (curated list of the 2024
     internship, with the same spelling corrections, plus the corrected
     spellings), with the case and the single spelling per drug of the reviewed
     table lists/pxslu/drugs_pxslu_case.tsv; it is the list of targets of every
     condition;
   - drugs_pxslu_upper.txt: the same in capitals (does a model write the drug
     names in capitals when given them in capitals?);
   - drugs_bdpm.txt: drug names of the French public drug database (Base de
     donnees publique des medicaments, ANSM, Licence Ouverte / Open Licence 2.0),
     plus the PxSLU targets missing from it (list = targets + distractors);
     brand names (first word of the name) and active substances (leading
     substance words of the names of generics); homeopathic and herbal
     registrations are left out;
   - drugs_vidal.txt, drugs_theriaque.txt: larger drug dictionaries of the
     internship (not redistributable), built the same way;
   the case of the terms that are not targets is set automatically (lower case
   when all their words are substance words of the BDPM, else capitalized);
6. writes one manifest per split: the transcript (with one spelling per drug,
   drug names in their case, first letter in capitals), the global lists (file
   paths) and per-utterance lists:
   - "pxslu25": 25 PxSLU drug names, the ones spoken in the utterance (if any)
     plus others drawn at random (seeded by the utterance id), sorted; the size
     of the per call lists of Contextual Earnings-22 ("global": 25 terms on
     average), and short enough for every model (Whisper hotwords: at most 223
     tokens; this list: at most ~130 tokens, ~160 in capitals); "pxslu25_upper"
     in capitals;
   - "oracle", "oracle_upper": only the targets spoken in the utterance (not a
     realistic setting, not run in the configs).

Usage:
    # BDPM files (https://base-donnees-publique.medicaments.gouv.fr/telechargement)
    mkdir bdpm && for f in CIS_bdpm.txt CIS_COMPO_bdpm.txt; do
        curl -L -o bdpm/$f https://base-donnees-publique.medicaments.gouv.fr/download/file/$f; done
    python tools/contextual_biasing/prepare_pxslu.py \\
        --pxslu_dir /home/data/audio/raw/transcript/fr/PxSLU \\
        --drug_lists_dir /path/to/Medical-ASR-adaptation-techniques/phonetisation_transcription/dictionary/drug_list \\
        --bdpm_dir bdpm \\
        --output_dir benchmarks/contextual_biasing
(the case table is read from <output_dir>/lists/pxslu/drugs_pxslu_case.tsv)
"""

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import soundfile

LOCAL_LIST_SIZE = 25  # terms of the per utterance list "pxslu25"

LISTS = {
    # variant name: file of the drug lists directory
    "vidal": "drug_vidal.txt",
    "theriaque": "drug_theriaque.txt",
}


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_corrections(path):
    """{wrong spelling: correct spelling} from a tab-separated file."""
    corrections = {}
    for line in read_lines(path):
        parts = line.strip().split("\t")
        if len(parts) == 2 and parts[0]:
            corrections[parts[0].strip()] = parts[1].strip()
    return corrections


def make_corrector(corrections):
    """Function applying all corrections in one pass, on whole words, longest first."""
    if not corrections:
        return lambda text: text
    keys = sorted(corrections, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, keys)) + r")(?!\w)")
    return lambda text: pattern.sub(lambda m: corrections[m.group(1)], text)


def is_excluded(text, exclusions):
    # Same rule as the dataset's clean_PxSLU.py: the excluded word starts a word and is
    # followed by a space
    return any(re.search(r"\b" + re.escape(word) + " ", text) for word in exclusions)


def clean_text(text):
    """Collapse spaces and remove truncated words (tokens with a "/")."""
    return " ".join(token for token in text.split() if "/" not in token)


def read_terms(path):
    return [line.strip() for line in read_lines(path) if line.strip() and not line.startswith("#")]


def tokens(text):
    return re.sub(r"[^\w]+", " ", text.lower()).split()


def find_terms(text, terms_tokens):
    """Terms (given as {term: tuple of tokens}) occurring in text, in list order."""
    words = tokens(text)
    found = []
    for term, toks in terms_tokens.items():
        n = len(toks)
        if any(tuple(words[i:i + n]) == toks for i in range(len(words) - n + 1)):
            found.append(term)
    return found


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def read_bdpm(path):
    """Rows (lists of tab-separated fields) of a BDPM file (Latin-1 encoded)."""
    with open(path, "r", encoding="latin-1") as f:
        return [line.rstrip("\r\n").split("\t") for line in f if line.strip()]


# Acronyms of the drug dictionaries, written in capitals
ACRONYMS = {"aas"}


def read_case_table(path):
    """{term in lower case: its form (case, single spelling)} of the reviewed table."""
    table = {}
    for line in read_lines(path):
        fields = line.split("\t")
        if line.startswith("#") or len(fields) < 2 or fields[0] == "cased":
            continue
        table[fields[1].strip()] = fields[0].strip()
    return table


def bdpm_substance_words(bdpm_dir):
    """Words of the active substances of the BDPM, in capitals without accents."""
    return {
        strip_accents(word)
        for row in read_bdpm(Path(bdpm_dir) / "CIS_COMPO_bdpm.txt")
        for word in re.findall(r"[^\W\d_]+", row[3].upper())
    }


def usual_case(term, substance_words):
    """Case of a drug name that is not in the reviewed table: lower case for an
    active substance, capitals for an acronym, else capitalized (each part)."""
    words = re.findall(r"[^\W\d_]+", term)
    if words and all(strip_accents(w.upper()) in substance_words for w in words):
        return term
    if term in ACRONYMS:
        return term.upper()
    return re.sub(r"(^|[\s\-])(\w)", lambda m: m.group(1) + m.group(2).upper(), term)


def bdpm_drug_names(bdpm_dir):
    """Drug names of the BDPM (lower case): brand names and active substances.

    The names of CIS_bdpm.txt ("DOLIPRANE 1000 mg, comprimé", "AMOXICILLINE ARROW
    500 mg, gélule") are cut before the dosage and split on "/" (combinations).
    A name starting with a substance word of CIS_COMPO_bdpm.txt is a generic: its
    leading substance words are kept ("amoxicilline"); otherwise it is a brand:
    its first word is kept ("doliprane"). The names of CIS_bdpm.txt have no
    accents: they are restored from the substance names when unambiguous.
    """
    bdpm_dir = Path(bdpm_dir)
    spellings = {}
    for row in read_bdpm(bdpm_dir / "CIS_COMPO_bdpm.txt"):
        for word in re.findall(r"[^\W\d_]+", row[3].upper()):
            spellings.setdefault(strip_accents(word), set()).add(word)
    substance_words = set(spellings)
    accented = {k: next(iter(v)) for k, v in spellings.items() if len(v) == 1 and len(k) >= 4}

    names = set()
    for row in read_bdpm(bdpm_dir / "CIS_bdpm.txt"):
        if row[5].startswith("Enreg"):  # homeopathic and herbal registrations
            continue
        words = []
        for word in row[1].split(",")[0].split():
            if re.search(r"[\d%]", word):
                break
            words.append(word)
        for part in " ".join(words).split("/"):
            part = part.split()
            if not part:
                continue
            if strip_accents(part[0].upper()) in substance_words:
                kept = []
                for word in part:
                    if strip_accents(word.upper()) not in substance_words:
                        break
                    kept.append(word)
            else:
                kept = part[:1]
            while len(kept) > 1 and len(kept[-1]) == 1:
                kept.pop()
            name = " ".join(accented.get(strip_accents(w.upper()), w) for w in kept).lower()
            if len(name) > 2 and re.search(r"[a-z]", strip_accents(name)):
                names.add(name)
    return names


def sort_key(term):
    return strip_accents(term).lower(), term


def with_targets(terms, targets):
    """Targets plus the terms that are not a target (ignoring case and accents), sorted."""
    known = {strip_accents(t).lower() for t in targets}
    return sorted(set(targets) | {t for t in terms if strip_accents(t).lower() not in known}, key=sort_key)


def recase(text, forms):
    """Transcript with the drug names in their form ({tokens in lower case: form}),
    and its first letter in capitals."""
    for toks, form in sorted(forms.items(), key=lambda x: -len(x[0])):
        pattern = r"(?<!\w)" + r"[^\w]+".join(map(re.escape, toks)) + r"(?!\w)"
        text = re.sub(pattern, form, text)
    return text[:1].upper() + text[1:]


def local_list(spoken, targets, size, seed):
    """Per utterance list of `size` targets: the ones spoken in the utterance plus
    others drawn at random (reproducible with `seed`), sorted."""
    others = [t for t in targets if t not in spoken]
    return sorted(spoken + random.Random(seed).sample(others, size - len(spoken)), key=sort_key)


def write_list(path, terms):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(t + "\n" for t in terms)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pxslu_dir", required=True, help="PxSLU folder (raw/, data_splits/, scripts/, recordings/)")
    parser.add_argument("--drug_lists_dir", required=True, help="Folder with drug_PxSLU.txt, drug_vidal.txt, drug_theriaque.txt")
    parser.add_argument("--bdpm_dir", required=True, help="Folder with CIS_bdpm.txt and CIS_COMPO_bdpm.txt (list \"bdpm\")")
    parser.add_argument("--output_dir", default="benchmarks/contextual_biasing", help="Output folder (manifests/ and lists/pxslu/)")
    parser.add_argument("--audio_prefix", default="<AUDIO_FOLDER>/raw/transcript/fr/PxSLU/recordings",
                        help="Audio folder written in the manifests")
    parser.add_argument("--splits", nargs="+", default=["test", "dev"], help="Official splits to write")
    parser.add_argument("--lists_in_manifest_prefix", default=None,
                        help="Path of the lists folder written in the manifests (default: <output_dir>/lists/pxslu)")
    args = parser.parse_args()

    pxslu = Path(args.pxslu_dir)
    output = Path(args.output_dir)
    lists_dir = output / "lists" / "pxslu"
    lists_ref = Path(args.lists_in_manifest_prefix) if args.lists_in_manifest_prefix else lists_dir

    corrections = read_corrections(pxslu / "scripts" / "drug_name_correction_V4.txt")
    correct = make_corrector(corrections)
    exclusions = [w.strip() for w in read_lines(pxslu / "scripts" / "drug_name_exclusion.txt") if w.strip()]

    # PxSLU drug names, with the same corrections as the transcripts, then in the
    # form (case, single spelling) of the reviewed table
    names = sorted({correct(t.lower()) for t in read_terms(Path(args.drug_lists_dir) / "drug_PxSLU.txt")}
                   | {v.lower() for v in corrections.values() if not v.isdigit() and v != "milligrammes"})
    case_table = read_case_table(lists_dir / "drugs_pxslu_case.tsv")
    missing = [t for t in names if t not in case_table]
    if missing:
        raise ValueError(f"Drug names missing from {lists_dir / 'drugs_pxslu_case.tsv'}: {missing}")
    form = {t: case_table[t] for t in names}  # spelling in the corpus -> form
    # Other spellings of a drug are replaced by its form in the transcripts
    spelling = make_corrector({t: f.lower() for t, f in form.items() if f.lower() != t})
    targets = sorted(set(form.values()), key=sort_key)
    write_list(lists_dir / "drugs_pxslu.txt", targets)
    write_list(lists_dir / "drugs_pxslu_upper.txt", [t.upper() for t in targets])
    lists = {"pxslu": lists_dir / "drugs_pxslu.txt", "pxslu_upper": lists_dir / "drugs_pxslu_upper.txt"}
    substance_words = bdpm_substance_words(args.bdpm_dir)
    distractors = {"bdpm": bdpm_drug_names(args.bdpm_dir)} | {
        name: {t.lower() for t in read_terms(Path(args.drug_lists_dir) / filename)} for name, filename in LISTS.items()
    }
    for name, terms in distractors.items():
        terms = {usual_case(t, substance_words) for t in terms}
        write_list(lists_dir / f"drugs_{name}.txt", with_targets(terms, targets))
        lists[name] = lists_dir / f"drugs_{name}.txt"
    for name, path in lists.items():
        print(f"List {name}: {len(read_terms(path))} terms -> {path}")

    paths = read_lines(pxslu / "raw" / "paths.txt")
    texts = read_lines(pxslu / "raw" / "seq.in")
    assert len(paths) == len(texts), "paths.txt and seq.in are not aligned"
    targets_tokens = {t: tuple(tokens(t)) for t in targets}
    forms = {toks: t for t, toks in targets_tokens.items()}

    for split in args.splits:
        line_numbers = [int(x) for x in read_lines(pxslu / "data_splits" / split / f"{split}.txt") if x.strip()]
        rows, n_excluded, n_corrected = [], 0, 0
        for number in line_numbers:
            path, raw = paths[number - 1].strip(), texts[number - 1].strip()
            text = spelling(correct(raw))
            n_corrected += text != raw
            if is_excluded(text, exclusions):
                n_excluded += 1
                continue
            text = clean_text(text)
            if not text:
                continue
            oracle = find_terms(text, targets_tokens)
            text = recase(text, forms)
            info = soundfile.info(str(pxslu / "recordings" / path))
            utterance_id = f"PxSLU_{path.replace('/', '_').removesuffix('.wav')}"
            local = local_list(oracle, targets, LOCAL_LIST_SIZE, utterance_id)
            rows.append({
                "audio_filepath": f"{args.audio_prefix}/{path}",
                "duration": round(info.frames / info.samplerate, 3),
                "offset": 0.0,
                "text": text,
                "id": utterance_id,
                "split": split,
                "line": number,
                "contextual_biasing": {name: str(lists_ref / path_.name) for name, path_ in lists.items()}
                | {f"pxslu{LOCAL_LIST_SIZE}": local, f"pxslu{LOCAL_LIST_SIZE}_upper": [t.upper() for t in local],
                   "oracle": oracle, "oracle_upper": [t.upper() for t in oracle]},
                "contextual_biasing_targets": str(lists_ref / "drugs_pxslu.txt"),
            })
        manifest = output / "manifests" / f"PxSLU_{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        n_with_target = sum(bool(row["contextual_biasing"]["oracle"]) for row in rows)
        n_occurrences = sum(len(row["contextual_biasing"]["oracle"]) for row in rows)
        hours = sum(row["duration"] for row in rows) / 3600
        print(f"{split}: {len(rows)} utterances ({hours:.2f} h), {n_excluded} excluded, {n_corrected} corrected, "
              f"{n_with_target} with a target ({n_occurrences} distinct targets per utterance, summed) -> {manifest}")


if __name__ == "__main__":
    main()
