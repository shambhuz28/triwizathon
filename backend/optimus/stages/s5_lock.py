"""S5 · Lock key words and remove unnecessary ones.

Locks: regex patterns (numbers with units, money, dates, identifiers, placeholders, inline
code, quotes, format names), spaCy named-entity recognition, dependency-parsed rule words
(must, never, only, exactly, at most, not … and the verb they govern) and label-set detection
("positive, negative or neutral", "spam/ham"). Locked spans are handed to every later step as
untouchable and re-checked in S9.

Compression: LLMLingua-2 scores how worth keeping every word is; that score is mixed with S4's
informativeness and steered by the earlier stages —
  • S2  padding, filler and repeated parts are dropped whole; data, examples, code, tables and
        headers are kept verbatim; each label gets its own keep-rate band (thresholds.yaml)
  • S3  high-risk tasks keep every word of every part (only S2's drops apply, ladder L2)
  • S4  a part's mean informativeness picks its rate inside the band; S4 filler and
        low-information phrases are pushed to the front of the removal queue
  • S5  locked words are always kept, so every lock survives exactly
The text is rebuilt from the original words (LLMLingua's own detokeniser splits "don't" into
"don ' t"). The result, every part's decision and a handoff summary feed S6 and S7.
"""
from __future__ import annotations

import bisect
import math

import numpy as np
import regex as re
from wordfreq import zipf_frequency

from .. import resources, textutils
from ..config import thresholds
from ..schemas import (Assumption, Compression, Handoff, LabelSet, LockedSpan, PIRSegment, S1Result, S2Result,
                       S3Result, S4Result, S5Result)

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


# ── Compression: remove unnecessary words (LLMLingua-2) ─────────────────────
_KEEP_WHOLE_KINDS = {"header", "code", "json", "table", "tagged"}
_CLOSERS = ".?!:;"
_CHUNK_RE = re.compile(r"\S+")


def llmlingua_token_probs(text: str) -> list[tuple[int, int, float]] | None:
    """(start, end, keep probability) for every LLMLingua-2 token, or None when the model is unavailable.
    Long texts are scored in windows of the model's maximum length."""
    comp = resources.llmlingua()
    if comp is None or not text.strip():
        return None
    import torch

    tok, model = comp.tokenizer, comp.model
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    window = getattr(comp, "max_seq_len", 512) - 2
    out: list[tuple[int, int, float]] = []
    for i in range(0, len(ids), window):
        inp = torch.tensor([[tok.cls_token_id] + ids[i:i + window] + [tok.sep_token_id]], device=model.device)
        with torch.inference_mode():
            probs = model(input_ids=inp, attention_mask=torch.ones_like(inp)).logits.softmax(-1)[0, 1:-1, 1]
        out += [(a, b, p) for (a, b), p in zip(offsets[i:i + window], probs.float().cpu().tolist()) if b > a]
    return out


class _WordScorer:
    """Keep score for any stretch of the prompt, from LLMLingua-2 token probabilities and S4 word scores."""

    def __init__(self, s4: S4Result, token_probs, cfg: dict):
        self.heat = [w for w in s4.heatmap if any(c.isalnum() for c in w.text)]
        self.heat_starts = [w.start for w in self.heat]
        self.probs = token_probs or []
        self.prob_starts = [a for a, _, _ in self.probs]
        self.low_info = [(p.start, p.end) for p in s4.low_info]
        self.model_weight = cfg["model_weight"] if token_probs else 0.0
        self.penalty = cfg["low_info_penalty"]

    def _overlapping(self, starts, items, a, b, end_of):
        k = max(0, bisect.bisect_left(starts, a) - 1)
        out = []
        while k < len(items) and starts[k] < b:
            if end_of(items[k]) > a:
                out.append(items[k])
            k += 1
        return out

    def score(self, a: int, b: int) -> float:
        words = self._overlapping(self.heat_starts, self.heat, a, b, lambda w: w.end)
        info = float(np.mean([w.score for w in words])) if words else 0.5
        toks = self._overlapping(self.prob_starts, self.probs, a, b, lambda t: t[1])
        keep = float(np.mean([p for _, _, p in toks])) if toks else info
        return self.model_weight * keep + (1 - self.model_weight) * info

    def low_info_phrase(self, a: int, b: int) -> int | None:
        """Index of the S4 low-information phrase (filler phrases, S2 padding, least informative
        chunks) the stretch falls in, if any. S4's single-word filler list is not used here: it
        would also drop "much" in "how much" or "help" in "help me debug"."""
        return next((k for k, (x, y) in enumerate(self.low_info) if textutils.overlaps(a, b, x, y)), None)


def _compress_part(text: str, cs: int, ce: int, rate: float, drop_below: float, scorer: _WordScorer,
                   locks: list[tuple[int, int]]):
    """Remove the whitespace-separated words of text[cs:ce] that score below `drop_below`, but keep at
    least `rate` of the words outside S4's low-information phrases (those may always go) and every
    locked word; rebuild the text from the original words.
    Returns (text, removed spans, (start, end, score) per word)."""
    chunks = [(cs + m.start(), cs + m.end()) for m in _CHUNK_RE.finditer(text[cs:ce])]
    locked = [any(textutils.overlaps(a, b, x, y) for x, y in locks) for a, b in chunks]
    wordy = [k for k, (a, b) in enumerate(chunks) if any(c.isalnum() for c in text[a:b])]
    score = {k: scorer.score(*chunks[k]) for k in wordy}
    phrase = {k: scorer.low_info_phrase(*chunks[k]) for k in wordy}
    # Words of one low-information phrase share its mean score (penalised) and go or stay together,
    # so "a bunch of" leaves whole and "some help" never loses just its noun.
    for p in set(phrase.values()) - {None}:
        members = [k for k in wordy if phrase[k] == p]
        shared = scorer.penalty * float(np.mean([score[k] for k in members]))
        score.update(dict.fromkeys(members, shared))
    keep = {k for k in wordy if score[k] >= drop_below} | {k for k, lk in enumerate(locked) if lk}
    floor = math.ceil(rate * sum(1 for k in wordy if phrase[k] is None) - 1e-9)
    # Top up to the floor with the best weak words, those outside low-information phrases first.
    for k in sorted((k for k in wordy if k not in keep), key=lambda k: (phrase[k] is not None, -score[k], k)):
        if len(keep & set(wordy)) >= floor:
            break
        keep.add(k)
    # Removing words between two copies of a word leaves "years years": keep the first copy.
    kept = sorted(k for k in keep if k in score)
    for prev, k in zip(kept, kept[1:]):
        same = text[slice(*chunks[k])].strip(".,;:!?").lower() == text[slice(*chunks[prev])].strip(".,;:!?").lower()
        if same and k != prev + 1 and not locked[k]:
            keep.discard(k)
    # A lone symbol ("-", "/") survives only between two kept words.
    keep |= {k for k in range(1, len(chunks) - 1) if k not in wordy and k - 1 in keep and k + 1 in keep}

    out, prev = "", None
    for k in sorted(keep):
        a, b = chunks[k]
        if prev is not None:
            out += text[chunks[prev][1]:a] if k == prev + 1 else " "
        out += text[a:b]
        prev = k
    last = text[chunks[-1][0]:chunks[-1][1]] if chunks else ""
    if out and len(chunks) - 1 not in keep and last[-1:] in _CLOSERS and out[-1] not in _CLOSERS + "\"'”’)]":
        out = out.rstrip(",;:") + last[-1]          # keep the sentence's closing mark

    removed: list[tuple[int, int]] = []
    for k, (a, b) in enumerate(chunks):
        if k in keep:
            continue
        if removed and not text[removed[-1][1]:a].strip() and k - 1 not in keep:
            removed[-1] = (removed[-1][0], b)
        else:
            removed.append((a, b))
    word_scores = [(a, b, 1.0 if locked[k] else round(score[k], 3)) for k, (a, b) in enumerate(chunks) if k in score]
    return out, removed, word_scores


def compress(s1: S1Result, s2: S2Result, s3: S3Result, s4: S4Result,
             spans: list[LockedSpan], lock_set: list[str]) -> tuple[list[PIRSegment], Compression]:
    cfg = thresholds()["s5"]["compression"]
    text = s1.clean_text
    enc = resources.tiktoken_encoding("o200k_base")
    top = [s for s in spans if s.parent is None]
    token_probs = llmlingua_token_probs(text)
    scorer = _WordScorer(s4, token_probs, cfg)
    high_risk = s3.risk.level == "high"
    bands = cfg["keep_rate"]

    def eligible(seg):
        return not seg.remove and seg.kind in ("sentence", "list_item") and seg.label in bands
    infos = [s4.segment_info.get(s.id, 0.5) for s in s2.segments if eligible(s)]
    info_lo, info_hi = (min(infos), max(infos)) if infos else (0.0, 1.0)

    def keep_rate(seg) -> float:
        lo, hi = bands[seg.label]
        info = s4.segment_info.get(seg.id, 0.5)
        t = 0.5 if info_hi - info_lo < 1e-6 else (info - info_lo) / (info_hi - info_lo)
        rate = lo + (hi - lo) * t
        return round(min(1.0, max(rate, cfg["high_risk_min_rate"]) if high_risk else rate), 3)

    parts: list[PIRSegment] = []
    for seg in s2.segments:
        cs, ce = seg.content_start, seg.content_end
        seg_locks = [s for s in top if textutils.overlaps(cs, ce, s.start, s.end)]
        rate, compressed, removed, word_scores = 1.0, seg.text, [], []
        if seg.remove and not seg_locks:
            action, rate, compressed, removed = "drop", 0.0, "", [(cs, ce)]
            reason = (f"S2: repeats {seg.duplicate_of}" if seg.duplicate_of
                      else f"S2: {textutils.filler_kind(seg.text)}" if seg.kind == "filler" else "S2: padding")
        elif seg.remove:
            action, reason = "keep", "S2 marked it for removal, but it holds locked text"
        elif seg.kind in _KEEP_WHOLE_KINDS or seg.label in ("data", "examples") or seg.label not in bands:
            action, reason = "keep", f"{seg.label} {seg.kind} is kept verbatim"
        else:
            rate = keep_rate(seg)
            n_words = sum(1 for m in _CHUNK_RE.finditer(seg.text) if any(c.isalnum() for c in m.group()))
            if n_words < cfg["min_words"]:
                action, reason = "keep", f"short part ({n_words} words)"
            elif high_risk and rate >= 1.0:
                action, reason = "keep", "S3: high-risk task, only S2's padding and repeats are removed"
            else:
                compressed, removed, word_scores = _compress_part(text, cs, ce, rate, cfg["drop_below"], scorer,
                                                                  [(s.start, s.end) for s in seg_locks])
                action = "compress" if removed else "keep"
                n_left = sum(1 for m in _CHUNK_RE.finditer(compressed) if any(c.isalnum() for c in m.group()))
                reason = (f"removed {n_words - n_left} of {n_words} words; kept at least {rate:.0%} of those outside"
                          f" S4 low-info phrases (S2 label {seg.label}, S4 info {s4.segment_info.get(seg.id, 0):.2f})"
                          if removed else "nothing worth removing")
        parts.append(PIRSegment(
            id=seg.id, label=seg.label, kind=seg.kind, requirement_id=seg.requirement_id, action=action, reason=reason,
            start=cs, end=ce, text=seg.text, compressed=compressed, tokens=len(enc.encode(seg.text)),
            compressed_tokens=len(enc.encode(compressed)) if compressed else 0, keep_rate=rate,
            info=s4.segment_info.get(seg.id, 0.0), locks=[s.id for s in seg_locks], dropped_spans=removed,
            word_scores=word_scores))

    # Reassemble. Where parts were dropped: keep the strongest line break that stood between them,
    # restore a sentence's closing mark or capital letter that left with a dropped filler
    # ("Please | give it to me in a table | or something like that." -> "Give it to me in a table.").
    lock_text = {s.id: s.text for s in top}
    pieces: list[str] = []
    gaps: list[str] = []
    closer, lead, last_label = "", "", None

    def close_previous():
        if pieces and closer and pieces[-1][-1:].isalnum():
            # "dont make anything up | ok?" closes with "." — only a question keeps its "?"
            pieces[-1] += closer if last_label == "question" else "."

    for k, (seg, part) in enumerate(zip(s2.segments, parts)):
        gaps.append(text[s2.segments[k - 1].content_end:seg.content_start] if k else "")
        if part.action == "drop":
            lead = lead or part.text[:1]
            if not closer and part.text[-1:] in ".?!" and "\n" not in "".join(gaps):   # ends the kept sentence
                closer = part.text[-1]
            continue
        body = part.compressed
        if part.action == "compress" and part.dropped_spans[0][0] == part.start:
            lead = lead or part.text[:1]
        sentence_start = not pieces or "".join(pieces).rstrip()[-1:] in ".?!:" or "\n" in "".join(gaps)
        if (lead.isupper() and body[:1].islower() and sentence_start
                and not any(body.startswith(lock_text[i]) for i in part.locks)):
            body = body[0].upper() + body[1:]
        close_previous()
        sep = max(reversed(gaps), key=lambda g: g.count("\n"))
        if not sep and len(gaps) > 1:
            sep = " "
        pieces.append((sep if pieces else "") + body)
        gaps, closer, lead, last_label = [], "", "", part.label
    close_previous()
    prompt = re.sub(r"[ \t]+\n", "\n", "".join(pieces)).strip()

    original_tokens = len(enc.encode(text))
    compressed_tokens = len(enc.encode(prompt))
    missing = [t for t in lock_set if t not in prompt]
    method = ("LLMLingua-2 keep probability mixed with S4 informativeness" if token_probs is not None
              else "S4 informativeness only (LLMLingua-2 unavailable)")
    return parts, Compression(
        method=method, model=resources.LLMLINGUA_MODEL if token_probs is not None else None, tokenizer="o200k_base",
        original_tokens=original_tokens, compressed_tokens=compressed_tokens,
        saved_tokens=original_tokens - compressed_tokens,
        ratio=round(compressed_tokens / original_tokens, 3) if original_tokens else 1.0,
        prompt=prompt, locks_preserved=not missing, missing_locks=missing)


def build_handoff(s2: S2Result, s3: S3Result, parts: list[PIRSegment]) -> Handoff:
    sections: dict[str, list[str]] = {}
    for p in parts:
        if p.action != "drop":
            sections.setdefault(p.label, []).append(p.id)
    return Handoff(
        task_type=s3.task_type.type, difficulty=s3.difficulty.level, risk_level=s3.risk.level,
        max_ladder_level="L2" if s3.risk.level == "high" else "L5",
        answer_budget=s3.answer_length.budget, max_tokens=s3.answer_length.max_tokens,
        sections=sections, requirements=[r.id for r in s2.requirements],
        assumptions=[Assumption(kind=g.kind, message=g.message, placeholder=f"{{{{ASSUMPTION: {g.assumption}}}}}",
                                segment_ids=g.segment_ids) for g in s3.gaps])


def run(s1: S1Result, s2: S2Result, s3: S3Result, s4: S4Result) -> S5Result:
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
    parts, compression = compress(s1, s2, s3, s4, spans, lock_set)
    return S5Result(spans=spans, label_sets=label_sets, by_category=by_category, lock_set=lock_set,
                    segments=parts, compression=compression, handoff=build_handoff(s2, s3, parts))
