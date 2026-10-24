"""S5 · Lock key words.

Combines regex patterns (numbers with units, money, dates, identifiers, placeholders,
inline code, quotes, format names), spaCy named-entity recognition, dependency-parsed rule
words (must, never, only, exactly, at most, not … and the verb they govern) and label-set
detection ("positive, negative or neutral", "spam/ham"). Locked spans are handed to every
later step as untouchable and re-checked in S9.
"""
from __future__ import annotations

import regex as re
from wordfreq import zipf_frequency

from .. import resources, textutils
from ..config import thresholds
from ..schemas import LabelSet, LockedSpan, S1Result, S2Result, S5Result

# Higher number wins when two locks cross.
PRIORITY = {"code": 9, "placeholder": 8, "data": 7, "quote": 6, "identifier": 5, "number": 4,
            "name": 3, "label": 3, "restriction": 2, "format": 1}

_UNITS = (r"%|percent|x\b|×|kb|mb|gb|tb|ms|s\b|secs?|seconds?|mins?|minutes?|hours?|hrs?|days?|weeks?|months?|years?"
          r"|km|m\b|cm|mm|kg|g\b|mg|lbs?|°\s?[cf]?|words?|tokens?|characters?|chars?|sentences?|paragraphs?|lines?"
          r"|items?|points?|bullets?|rows?|columns?|times|pages?|steps?|calls?|users?|requests?|stars?|reviews?")
_WORD_NUMBERS = (r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty"
                 r"|hundred|thousand|million|billion|lakh|crore|twice|thrice|half|double|single")

NUMBER_RES = [re.compile(p, re.I) for p in [
    r"(?:[$€£₹¥]|rs\.?|inr|usd)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|bn|million|billion|thousand|lakh|crore))?",
    r"(?<![\w.])\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?![\w.])",
    r"(?<![\w.])\d{1,2}:\d{2}(?::\d{2})?\s?(?:am|pm)?",
    rf"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:\s?(?:-|–|to)\s?\d[\d,]*(?:\.\d+)?)?(?:\s?(?:{_UNITS}))?(?![\w])",
    rf"\b(?:{_WORD_NUMBERS})[\s-](?:{_UNITS})\b",
]]

IDENTIFIER_RES = [(re.compile(p), why) for p, why in [
    (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "email"),
    (r"\bhttps?://[^\s<>\"')\]]+|\bwww\.[^\s<>\"')\]]+", "url"),
    (r"(?:(?<=\s)|^)(?:[A-Za-z]:\\|~?/|\./)[\w./\\-]+", "path"),
    (r"\bv?\d+\.\d+(?:\.\d+)+\b|\bv\d+(?:\.\d+)?\b", "version"),
    (r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", "snake_case"),
    (r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b", "camelCase"),
    (r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b", "PascalCase"),
    (r"\b[A-Z]{2,}[-_]?\d+[A-Z0-9-]*\b|\b[A-Z]{2,}(?:[-_][A-Z0-9]+)+\b", "code/ID"),
    (r"#\d{2,}\b", "number ID"),
    (r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b", "hash"),
    (r"\b[\w-]+\.(?:py|js|ts|tsx|jsx|json|csv|md|txt|yaml|yml|html|css|sql|java|go|rs|cpp|c|h|ipynb|pdf|xlsx|docx)\b", "file"),
    (r"\b\w+\(\)", "function"),
    (r"\b(?!I\b|OK\b|AM\b|PM\b)[A-Z]{2,6}s?\b", "acronym"),
]]

PLACEHOLDER_RE = re.compile(r"\{\{\s*[\w.-]+\s*\}\}|\$\{[\w.-]+\}|(?<!\{)\{[A-Za-z_][\w.-]*\}(?!\})|<[A-Z][A-Z0-9_]+>"
                            r"|\[[A-Z][A-Z0-9_ ]{1,30}\]|%\([\w]+\)[sd]|%[sd]\b|\$[A-Z_]{2,}\b")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
FORMAT_RE = re.compile(r"\b(?:markdown|json|jsonl|csv|tsv|yaml|xml|html|tables?|bullet(?:ed)? (?:points?|list)"
                       r"|numbered list|plain text|code block|one line|single word|one word)\b", re.I)

RULE_PHRASE_RE = re.compile(r"\b(?:at most|at least|no more than|no less than|no fewer than|up to|fewer than|less than"
                            r"|more than|maximum(?: of)?|minimum(?: of)?|exactly|only)\s+(?:\S+)(?:\s+(?:" + _UNITS + r"))?", re.I)
RULE_WORDS = {"must", "never", "always", "not", "n't", "no", "without", "avoid", "cannot", "only", "exactly", "nothing",
              "none", "neither", "nor", "required", "mandatory", "forbidden", "prohibited", "strictly", "solely",
              "exclusively", "unless", "except", "refrain", "ensure"}

ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "NORP", "FAC", "LANGUAGE"}
NUMERIC_ENTS = {"MONEY", "PERCENT", "QUANTITY", "CARDINAL", "DATE", "TIME", "ORDINAL"}
_MONTH_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\d", re.I)
_LABEL_CUE_RE = re.compile(r"\b(?:classify|classif\w+|label\w*|categor\w+|class(?:es)?|sentiment|one of|either|as|choose"
                           r"|select|tag\w*|options?|values?|types?|mark(?:ed)?|rate|bucket|is|are|be)\b", re.I)
_LABEL_ITEM = r"[\"'“]?[\w&'-]+(?:\s[\w&'-]+){0,2}[\"'”]?"
LABEL_LIST_RE = re.compile(rf"(?<![\w-])({_LABEL_ITEM})(?:\s*,\s*({_LABEL_ITEM}))+\s*,?\s+(?:or|and)\s+({_LABEL_ITEM})(?![\w-])")
SLASH_SET_RE = re.compile(r"(?<![\w/.-])([A-Za-z][\w&-]*)(?:\s?/\s?([A-Za-z][\w&-]*))+(?![\w/.-])")
_DROP_POS = {"AUX", "VERB", "ADP", "DET", "PRON", "SCONJ", "CCONJ", "PART", "ADV"}


class _Collector:
    def __init__(self, text: str):
        self.text = text
        self.items: list[dict] = []

    def add(self, start: int, end: int, category: str, source: str, detail: str | None = None):
        while start < end and self.text[start].isspace():
            start += 1
        while end > start and self.text[end - 1] in " \t\n.,;:":
            end -= 1
        if end > start:
            self.items.append({"start": start, "end": end, "category": category, "source": source, "detail": detail})


def _label_sets(text: str, doc, segments, col: _Collector) -> list[LabelSet]:
    sets: list[LabelSet] = []
    for seg in segments:
        if seg.label in ("data", "padding") or seg.remove or seg.kind in ("code", "json", "table"):
            continue
        s0, sentence = seg.content_start, seg.text
        for m in list(LABEL_LIST_RE.finditer(sentence)) + list(SLASH_SET_RE.finditer(sentence)):
            if m.re is SLASH_SET_RE:
                raw_items = list(zip(m.captures(1) + m.captures(2), m.starts(1) + m.starts(2)))
                if any(it.lower() in ("and", "or", "he", "she", "s") for it, _ in raw_items):
                    continue
            else:
                raw_items = list(zip(m.captures(1) + m.captures(2) + m.captures(3), m.starts(1) + m.starts(2) + m.starts(3)))
            if not _LABEL_CUE_RE.search(sentence[:m.end()]) and not any(it[:1] in "\"'“" for it, _ in raw_items):
                continue
            items = []
            for item, off in raw_items:
                span = doc.char_span(s0 + off, s0 + off + len(item), alignment_mode="expand")
                toks = list(span) if span is not None else []
                # keep only what follows the last verb/aux/preposition: "review is positive" -> "positive"
                cut = max((k for k, t in enumerate(toks[:-1]) if t.pos_ in _DROP_POS), default=-1)
                toks = toks[cut + 1:]
                if not toks or all(t.is_stop for t in toks) or len(toks) > 3:
                    items = []
                    break
                items.append((toks[0].idx, toks[-1].idx + len(toks[-1])))
            if len(items) < 2:
                continue
            for a, b in items:
                col.add(a, b, "label", "label-set", detail=f"one of {len(items)} allowed labels")
            sets.append(LabelSet(start=items[0][0], end=items[-1][1], items=[text[a:b].strip("\"'“”") for a, b in items]))
    return sets


def _restrictions(doc, segments, col: _Collector, max_scope: int):
    instr = [s for s in segments if s.label in ("task", "rules", "format", "question", "role") and not s.remove]
    for seg in instr:
        for m in RULE_PHRASE_RE.finditer(seg.text):
            col.add(seg.content_start + m.start(), seg.content_start + m.end(), "restriction", "rule phrase")
        span = doc.char_span(seg.content_start, seg.content_end, alignment_mode="expand")
        if span is None:
            continue
        for tok in span:
            if tok.lower_ not in RULE_WORDS and tok.lemma_.lower() not in RULE_WORDS:
                continue
            if tok.lower_ in ("no", "only") and tok.dep_ == "det":
                head = tok.head
            elif tok.dep_ in ("neg", "advmod", "aux", "det", "amod", "prep") and tok.head.i != tok.i:
                head = tok.head
            else:
                head = tok
            lo, hi = min(tok.i, head.i), max(tok.i, head.i)
            if hi - lo > max_scope:
                lo = hi = tok.i
            # include an auxiliary right before a negation ("do n't", "must not")
            if lo > span.start and doc[lo - 1].pos_ == "AUX" and doc[lo].dep_ == "neg":
                lo -= 1
            # include a verb particle ("make up")
            for child in doc[hi].children:
                if child.dep_ == "prt" and child.i == hi + 1:
                    hi += 1
            scope = " ".join(t.text for t in head.subtree)[:80]
            col.add(doc[lo].idx, doc[hi].idx + len(doc[hi]), "restriction", "dependency parse", detail=f"scope: {scope}")


def run(s1: S1Result, s2: S2Result) -> S5Result:
    text = s1.clean_text
    doc = resources.parse(text)
    col = _Collector(text)
    max_scope = thresholds()["s5"]["max_restriction_scope_tokens"]

    for b in s2.blocks:
        if b.kind in ("code", "json"):
            col.add(b.start, b.end, "code", f"{b.kind} block")
        elif b.kind == "table":
            col.add(b.start, b.end, "format" if not b.meta.get("rows") else "data", "table")
        else:
            col.add(b.start, b.end, "data", f"<{b.meta.get('tag')}> block")
    for seg in s2.segments:
        if seg.label in ("data", "examples") and seg.kind in ("sentence", "list_item"):
            col.add(seg.content_start, seg.content_end, "data", f"{seg.label} part {seg.id}")

    for m in INLINE_CODE_RE.finditer(text):
        col.add(m.start(), m.end(), "code", "inline code")
    for m in PLACEHOLDER_RE.finditer(text):
        col.add(m.start(), m.end(), "placeholder", "template / redaction placeholder")
    for m in textutils.QUOTE_RE.finditer(text):
        col.add(m.start(), m.end(), "quote", "quoted text")
    for pat in NUMBER_RES:
        for m in pat.finditer(text):
            col.add(m.start(), m.end(), "number", "number pattern")
    for pat, why in IDENTIFIER_RES:
        for m in pat.finditer(text):
            col.add(m.start(), m.end(), "identifier", why)
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            if len(ent) == 1 and (ent[0].is_sent_start or ent[0].i == 0) and zipf_frequency(ent.text.lower(), "en") >= 4.0:
                continue        # a capitalised common word at the start of a line, not a name
            col.add(ent.start_char, ent.end_char, "name", f"spaCy NER ({ent.label_})")
        elif ent.label_ in NUMERIC_ENTS and _MONTH_RE.search(ent.text) and any(c.isdigit() for c in ent.text):
            col.add(ent.start_char, ent.end_char, "number", f"spaCy NER ({ent.label_})")
    removed = [(s.content_start, s.content_end) for s in s2.segments if s.remove]
    for seg in s2.segments:
        if seg.label in ("format", "rules", "task") and not seg.remove:
            for m in FORMAT_RE.finditer(seg.text):
                col.add(seg.content_start + m.start(), seg.content_start + m.end(), "format", "answer format term")
    _restrictions(doc, s2.segments, col, max_scope)
    label_sets = _label_sets(text, doc, s2.segments, col)

    # Resolve: drop locks inside parts S2 marked for removal (padding, repeats), exact duplicates,
    # and the weaker of two crossing locks.
    items = [it for it in col.items if not any(a <= it["start"] and it["end"] <= b for a, b in removed)]
    items.sort(key=lambda it: (-PRIORITY[it["category"]], -(it["end"] - it["start"])))
    kept: list[dict] = []
    for it in items:
        clash = False
        for k in kept:
            same = k["start"] == it["start"] and k["end"] == it["end"]
            crossing = textutils.overlaps(it["start"], it["end"], k["start"], k["end"]) and not (
                (k["start"] <= it["start"] and it["end"] <= k["end"]) or (it["start"] <= k["start"] and k["end"] <= it["end"]))
            if same or crossing:
                clash = True
                break
        if not clash:
            kept.append(it)
    kept.sort(key=lambda it: (it["start"], -(it["end"] - it["start"])))

    spans: list[LockedSpan] = []
    for n, it in enumerate(kept, 1):
        parent = next((p.id for p in reversed(spans) if p.start <= it["start"] and it["end"] <= p.end), None)
        spans.append(LockedSpan(id=f"L{n}", start=it["start"], end=it["end"], text=text[it["start"]:it["end"]],
                                category=it["category"], source=it["source"], detail=it["detail"], parent=parent))
    by_category: dict[str, int] = {}
    for s in spans:
        by_category[s.category] = by_category.get(s.category, 0) + 1
    lock_set = list(dict.fromkeys(s.text for s in spans if s.parent is None))
    return S5Result(spans=spans, label_sets=label_sets, by_category=by_category, lock_set=lock_set)
