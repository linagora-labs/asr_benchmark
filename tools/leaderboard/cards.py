"""Result cards: small SVG tables of the leaderboard, to embed in model cards (Hugging Face
README...) as an image that follows the leaderboard instead of a frozen screenshot.

The cards to draw are listed in cards.json ({card name: spec}); each is written to
<output>/cards/<name>.svg and <name>.png (2x, with rsvg-convert, else ImageMagick; no PNG
without either) and <output>/cards/index.json lists them (urls, models shown).
Spec keys, all optional:

    title       text above the table (default: "French ASR benchmark")
    subtitle    text under the title (default: metric and test sets)
    language    language of the test sets (default "fr")
    metric      wer_nocasepunc (default) | wer | cer_nocasepunc | cer, as on the page
    datasets    test sets (ids or display names), default: all those of the language
                scored in this metric
    filters     {field: allowed values} on the model fields of leaderboard.json
                (architecture, task, license, backend, streaming...): a list of values, or
                one value. "architecture": ["Streaming"] keeps the streaming models, as on
                the page. min_params / max_params ("600M", "1.5B", or a number of millions)
                and min_rtfx bound those fields, match /
                exclude are regexes on the model name or folder id.
    sort        average (default) | rank (average rank) | rtfx | a test set
    models      models shown, in any order (rows follow `sort`): folder ids, model names
                (all their variants), or keywords resolved on the filtered models sorted
                by `sort`, so they follow the leaderboard: first, second, third,
                rank:N (Nth), top:N (N first), last, all. Default: top:10.
    highlight   models drawn in bold on a tinted row (same syntax), default: the models
                named explicitly in `models`
    columns     among average, rank, datasets, rtfx, params, license, architecture, task
                (default: average, datasets, rtfx, params)
    short_names drop the organization of the model names, unless two would clash (default false)

Ranks (# column, colors) are those of the filtered models, like the page with the same
filters. A model named explicitly but filtered out is still shown, without a rank.
"""
import json
import re
import shutil
import subprocess
from html import escape

FONT = "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
COLORS = {"first": "#5dade2", "second": "#82e0aa", "second-last": "#fdcb6e", "last": "#e74c3c"}
TEXT, MUTED, LINE, ACCENT, BG, HIGHLIGHT = "#1d1d1b", "#6b6a66", "#e6e4df", "#1f6f78", "#ffffff", "#eef5f6"
METRIC_LABELS = {"wer_nocasepunc": "WER (%), without case and punctuation", "wer": "WER (%), case and punctuation",
                 "cer_nocasepunc": "CER (%), without case and punctuation", "cer": "CER (%), case and punctuation"}
DEFAULT_COLUMNS = ["average", "datasets", "rtfx", "params"]
KEYWORDS = {"first": 1, "second": 2, "third": 3}


def text_width(text, size):
    """Rough width (px) of a sans-serif text: an SVG <img> cannot measure its text."""
    narrow, wide = set("iljtf.,:;'|!()[] "), set("mwMWÉ%")
    return size * sum(0.3 if c in narrow else 0.85 if c in wide else 0.62 if c.isupper() or c.isdigit() else 0.55
                      for c in text)


def shown(v):
    return round(v * 10) / 10


def rank_class(value, values):
    """Same as rankClass() of the page: first, second, second-last, last of a column."""
    if value is None or len(values) < 4:
        return ""
    distinct = sorted({shown(v) for v in values})
    i, n = distinct.index(shown(value)), len(distinct)
    return "first" if i == 0 else "second" if i == 1 else "last" if i == n - 1 else "second-last" if i == n - 2 else ""


def params_text(n):
    return f"{n / 1e9:.{1 if n < 1e10 else 0}f}B" if n >= 1e9 else f"{round(n / 1e6)}M"


def parse_params(n):
    """Number of parameters: "600M", "1.5B", or a plain number of millions."""
    if isinstance(n, str):
        match = re.fullmatch(r"\s*([\d.]+)\s*([MB])\s*", n, re.IGNORECASE)
        if not match:
            raise ValueError(f"number of parameters {n!r}: write it like 600M or 1.5B")
        return float(match[1]) * (1e9 if match[2].upper() == "B" else 1e6)
    return n * 1e6


def keep(m, filters):
    for field, allowed in filters.items():
        if field in ("min_params", "max_params", "min_rtfx"):
            value = m.get(field[4:])
            if field != "min_rtfx":
                allowed = parse_params(allowed)
            if value is None or (value < allowed if field.startswith("min") else value > allowed):
                return False
        elif field == "match":
            if not re.search(allowed, m["model"]) and not re.search(allowed, m["id"]):
                return False
        elif field == "exclude":
            if re.search(allowed, m["model"]) or re.search(allowed, m["id"]):
                return False
        else:
            allowed = allowed if isinstance(allowed, list) else [allowed]
            value = m.get(field)
            if field == "architecture" and m.get("streaming") and "Streaming" in allowed:
                continue
            if value not in allowed and not (value is None and "—" in allowed):
                return False
    return True


def resolve(names, ranked, models, card):
    """Models of `names` (ids, model names, keywords) in order, without duplicates."""
    found = []
    for name in names:
        if name in KEYWORDS or re.fullmatch(r"(rank|top):\d+|last|all", name):
            if name == "all":
                picked = ranked
            elif name == "last":
                picked = ranked[-1:]
            elif name.startswith("top:"):
                picked = ranked[:int(name[4:])]
            else:
                n = KEYWORDS.get(name) or int(name[5:])
                picked = ranked[n - 1:n]
            if not picked:
                print(f"card {card}: no model for {name!r} after filtering")
        else:
            picked = [m for m in models if m["id"] == name] or [m for m in models if m["model"] == name]
            if not picked:
                raise ValueError(f"card {card}: unknown model {name!r} (folder id or model name of leaderboard.json)")
        found += [m for m in picked if m not in found]
    return found


def card_rows(spec, data, card):
    language = spec.get("language", "fr")
    metric = spec.get("metric", "wer_nocasepunc")
    if metric not in METRIC_LABELS:
        raise ValueError(f"card {card}: unknown metric {metric!r}")
    lang_sets = [d for d in data["datasets"] if d["language"] == language]
    scored = [d for d in lang_sets if any(metric in m["scores"].get(d["id"], {}) for m in data["models"])]
    if "datasets" in spec:
        by_key = {k: d for d in lang_sets for k in (d["id"], d["name"])}
        unknown = [k for k in spec["datasets"] if k not in by_key]
        if unknown:
            raise ValueError(f"card {card}: unknown test sets {unknown}")
        datasets = [by_key[k] for k in spec["datasets"] if by_key[k] in scored]
    else:
        datasets = scored
    if not datasets:
        raise ValueError(f"card {card}: no test set scored in {metric}")

    def average(m):
        results = [m["scores"].get(d["id"], {}).get(metric) for d in datasets]
        return None if any(r is None for r in results) else sum(r["wer"] for r in results) / len(results)

    rows = {m["id"]: {"m": m, "avg": average(m)} for m in data["models"]}
    population = [r for r in rows.values() if r["avg"] is not None and keep(r["m"], spec.get("filters", {}))]
    # Average rank among the filtered models, displayed-equal scores sharing a rank (averageRanks()).
    for r in population:
        r["rank"] = 0
    for d in datasets:
        scores = [shown(r["m"]["scores"][d["id"]][metric]["wer"]) for r in population]
        for r, s in zip(population, scores):
            r["rank"] += 1 + sum(v < s for v in scores)
    for r in population:
        r["rank"] /= len(datasets)

    sort = spec.get("sort", "average")
    sort_set = next((d for d in datasets if sort in (d["id"], d["name"])), None)
    if sort not in ("average", "rank", "rtfx") and not sort_set:
        raise ValueError(f"card {card}: unknown sort {sort!r}")

    def key(r):
        if sort == "average":
            return r["avg"]
        if sort == "rank":
            return r.get("rank")
        if sort == "rtfx":
            return -r["m"]["rtfx"] if r["m"]["rtfx"] else None
        return (r["m"]["scores"].get(sort_set["id"], {}).get(metric) or {}).get("wer")

    order = lambda r: (key(r) is None, key(r) or 0, r["m"]["model"].lower())
    population.sort(key=order)
    for i, r in enumerate(population, 1):
        r["position"] = i
    ranked = [r["m"] for r in population]
    models = [r["m"] for r in rows.values()]
    names = spec.get("models", ["top:10"])
    selected = resolve(names, ranked, models, card)
    explicit = [n for n in names if not (n in KEYWORDS or re.fullmatch(r"(rank|top):\d+|last|all", n))]
    highlighted = resolve(spec.get("highlight", explicit), ranked, models, card)
    shown_rows = sorted((rows[m["id"]] for m in selected), key=lambda r: (r not in population, *order(r)))
    for r in shown_rows:
        r["highlight"] = r["m"] in highlighted
        if r not in population:
            print(f"card {card}: {r['m']['id']} is filtered out (or misses a test set): shown without rank")
    return datasets, metric, population, shown_rows, sort


def render_card(spec, data, card):
    datasets, metric, population, rows, sort = card_rows(spec, data, card)
    columns = spec.get("columns", DEFAULT_COLUMNS)
    unknown = set(columns) - {"average", "rank", "datasets", "rtfx", "params", "license", "architecture", "task"}
    if unknown:
        raise ValueError(f"card {card}: unknown columns {sorted(unknown)}")
    short = spec.get("short_names", False)
    score = lambda r, d: (r["m"]["scores"].get(d["id"], {}).get(metric) or {}).get("wer")
    values = {d["id"]: [score(r, d) for r in population] for d in datasets}
    avg_values = [r["avg"] for r in population]
    rank_values = [r["rank"] for r in population]

    # Columns: (header, sub header, align, cell(row) -> (text, color class, bold))
    cols = [("#", "", "end", lambda r: (str(r["position"]) if "position" in r else "–", "", False))]
    # Detail in brackets: the variant, and the backend when two shown rows have the same name.
    label = lambda m: (m["model"], m["variant"])
    twins = {label(r["m"]) for r in rows if sum(label(o["m"]) == label(r["m"]) for o in rows) > 1}
    detail = lambda m: ", ".join(filter(None, [m["variant"], label(m) in twins and m["backend"]]))
    # Short names keep the organization when two shown models would get the same one.
    bare = lambda m: m["model"].split("/", 1)[-1]
    clash = {bare(r["m"]) for r in rows if len({o["m"]["model"] for o in rows if bare(o["m"]) == bare(r["m"])}) > 1}
    name = lambda r: bare(r["m"]) if short and bare(r["m"]) not in clash else r["m"]["model"]
    cols.append(("Model", "", "start", lambda r: (name(r) + (f"  ({detail(r['m'])})" if detail(r["m"]) else ""), "", r["highlight"])))
    for c in columns:
        if c == "average":
            cols.append(("Average", f"{len(datasets)} test sets", "end", lambda r: (
                f"{r['avg']:.1f}" if r["avg"] is not None else "—",
                rank_class(r["avg"], avg_values) if "position" in r else "", True)))
        elif c == "rank":
            cols.append(("Avg rank", f"{len(datasets)} test sets", "end", lambda r: (
                f"{r['rank']:.1f}" if "rank" in r else "—", rank_class(r.get("rank"), rank_values), False)))
        elif c == "datasets":
            for d in datasets:
                cols.append((d["name"], d["domain"], "end", lambda r, d=d: (
                    f"{score(r, d):.1f}" if score(r, d) is not None else "—",
                    rank_class(score(r, d), values[d["id"]]) if "position" in r else "", False)))
        elif c == "rtfx":
            cols.append(("RTFx", "speed", "end", lambda r: (f"{round(r['m']['rtfx'])}×" if r["m"]["rtfx"] else "—", "", False)))
        elif c == "params":
            cols.append(("Size", "parameters", "end", lambda r: (params_text(r["m"]["params"]) if r["m"]["params"] else "—", "", False)))
        else:
            cols.append((c.capitalize(), "", "start", lambda r, c=c: (r["m"].get(c) or "—", "", False)))

    size, small, pad, row_h, head_h = 13, 10.5, 10, 26, 36
    cells = [[col[3](r) for col in cols] for r in rows]
    widths = [max([text_width(h, size), text_width(s, small)] + [text_width(c[i][0], size) for c in cells]) + 2 * pad
              for i, (h, s, _, _) in enumerate(cols)]
    title = spec.get("title", "French ASR benchmark")
    subtitle = spec.get("subtitle", f"{METRIC_LABELS[metric]}, lower is better · sorted by "
                        f"{ {'average': 'average', 'rank': 'average rank', 'rtfx': 'speed'}.get(sort, sort) }")
    footer = f"linagora-labs.github.io/asr_benchmark · updated {(data.get('results_updated') or data['generated'])[:10]}"
    margin, top = 16, 62
    width = max(sum(widths), text_width(subtitle, 12), text_width(footer, 11)) + 2 * margin
    height = top + head_h + row_h * len(rows) + 34

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
           f'font-family="{FONT}" font-size="{size}" fill="{TEXT}">',
           f'<title>{escape(title)}</title>',
           f'<rect width="100%" height="100%" rx="8" fill="{BG}" stroke="{LINE}"/>',
           f'<text x="{margin}" y="28" font-size="17" font-weight="700">{escape(title)}</text>',
           f'<text x="{margin}" y="47" font-size="12" fill="{MUTED}">{escape(subtitle)}</text>']
    x = margin
    for (head, sub, align, _), w in zip(cols, widths):
        tx = x + pad if align == "start" else x + w - pad
        out.append(f'<text x="{tx:.1f}" y="{top + 15}" text-anchor="{align}" font-weight="600">{escape(head)}</text>')
        if sub:
            out.append(f'<text x="{tx:.1f}" y="{top + 29}" text-anchor="{align}" font-size="{small}" fill="{MUTED}">{escape(sub)}</text>')
        x += w
    y = top + head_h
    out.append(f'<line x1="{margin}" x2="{width - margin:.0f}" y1="{y}" y2="{y}" stroke="{LINE}"/>')
    for r, row in zip(rows, cells):
        if r["highlight"]:
            out.append(f'<rect x="{margin}" y="{y}" width="{width - 2 * margin:.0f}" height="{row_h}" fill="{HIGHLIGHT}"/>')
        x = margin
        for (_, _, align, _), w, (text, cls, bold) in zip(cols, widths, row):
            if cls:
                out.append(f'<rect x="{x + 2:.1f}" y="{y + 2}" width="{w - 4:.1f}" height="{row_h - 4}" rx="3" fill="{COLORS[cls]}"/>')
            tx = x + pad if align == "start" else x + w - pad
            weight = ' font-weight="700"' if bold or r["highlight"] else ""
            fill = f' fill="{MUTED}"' if text in ("—", "–") else ""
            out.append(f'<text x="{tx:.1f}" y="{y + 17.5}" text-anchor="{align}"{weight}{fill}>{escape(text)}</text>')
            x += w
        y += row_h
        out.append(f'<line x1="{margin}" x2="{width - margin:.0f}" y1="{y}" y2="{y}" stroke="{LINE}"/>')
    out.append(f'<text x="{margin}" y="{y + 22}" font-size="11" fill="{ACCENT}">{escape(footer)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n", [r["m"]["id"] for r in rows]


def to_png(svg):
    """<svg>.png at twice the size, with rsvg-convert (GitHub Action) or ImageMagick. False without either."""
    png = svg.with_suffix(".png")
    if shutil.which("rsvg-convert"):
        command = ["rsvg-convert", "--zoom", "2", "--output", str(png), str(svg)]
    elif shutil.which("magick") or shutil.which("convert"):
        command = [shutil.which("magick") or shutil.which("convert"), "-density", "192", "-background", "none", str(svg), str(png)]
    else:
        return False
    subprocess.run(command, check=True)
    return True


def build_cards(config, data, output, site_url):
    """Draws the cards of `config` (cards.json) into <output>/cards/, with their index.json."""
    if not config or not config.exists():
        return
    specs = json.loads(config.read_text(encoding="utf-8"))
    folder = output / "cards"
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for card, spec in specs.items():
        if not re.fullmatch(r"[\w.-]+", card):
            raise ValueError(f"card name {card!r}: letters, digits, '_', '-' and '.' only (file name)")
        svg, shown_models = render_card(spec, data, card)
        (folder / f"{card}.svg").write_text(svg, encoding="utf-8")
        png = to_png(folder / f"{card}.svg")
        index[card] = {"url": f"{site_url}cards/{card}.svg", "png": f"{site_url}cards/{card}.png" if png else None,
                       "models": shown_models, "spec": spec}
        if not png:
            print(f"card {card}: no PNG, install rsvg-convert (librsvg2-bin) or ImageMagick")
    (folder / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(index)} cards -> {folder}/")
