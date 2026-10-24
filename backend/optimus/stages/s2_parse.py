"""S2 · Break into parts.

Code, JSON, tables and tag-fenced blocks are pulled out first and kept whole. The rest is
split into sentences/list items (pysbd), and every part gets one of nine labels by
combining weighted rule hits (regex + spaCy imperative detection + header sections) with a
vote from sentence-embedding similarity to labelled prototypes. Parts that say the same thing
are linked (embedding cosine and lemma overlap via RapidFuzz), keeping the most specific.
Parts tile the input exactly, so every character belongs to one part.
"""
from __future__ import annotations

import numpy as np
import regex as re
from rapidfuzz import fuzz

from .. import resources, textutils
from ..config import thresholds
from ..schemas import LABELS, Block, DuplicateGroup, Requirement, S1Result, S2Result, Segment

# ── Prototypes for the embedding vote ────────────────────────────────────────
PROTOTYPES: dict[str, list[str]] = {
    "role": ["You are an expert data analyst.", "Act as a senior software engineer.",
             "Your role is a friendly customer support agent.", "You are a helpful assistant for lawyers."],
    "task": ["Classify each review by sentiment.", "Summarize the following article.",
             "Write a Python function that sorts a list.", "I want you to analyze these documents and tell me the main themes.",
             "Translate this text into French.", "Tell me what customers are complaining about."],
    "context": ["I run a small bakery and we get many customer emails.", "The company launched a new product last month.",
                "This data comes from our internal sales system.", "Our users are mostly students in India."],
    "data": ['"The food arrived late and cold."', "1. Delivery was quick but the packaging was torn.",
             "Order #4521 was refunded on 3 March.", "The quarterly revenue was 4.2 million dollars."],
    "examples": ["For example, 'great food' should be labelled positive.", "Input: hello Output: bonjour",
                 "Example: 'The app keeps crashing' -> negative"],
    "rules": ["Do not make up any information.", "You must only use the provided text.",
              "Never include personal opinions.", "Always cite the source sentence.", "Each item must get exactly one label."],
    "format": ["Return the answer as a JSON object.", "Put the results in a table.",
               "Respond in three bullet points.", "Keep the answer under 100 words.", "Output only the label, nothing else."],
    "question": ["What is the capital of France?", "Why did revenue drop last quarter?",
                 "How does this algorithm work?", "Which of these options is cheapest?"],
    "padding": ["Hi! I hope you are doing well.", "Thanks a lot in advance, I really appreciate it!",
                "Could you please help me out with something?", "Hello there, hope you're having a great day."],
}

# ── Rule hits: (pattern, weight, reason) per label ───────────────────────────
_R = lambda p: re.compile(p, re.I)  # noqa: E731
RULES: dict[str, list[tuple[re.Pattern, float, str]]] = {
    "role": [(_R(r"^\W*(?:you are|you're|act as|acting as|pretend (?:to be|you are)|imagine you are|your role is"
                 r"|you will act as|take on the role|as an? [\w -]{2,40}, you)\b"), 3.0, "assigns a role")],
    "format": [
        (_R(r"\b(?:tables?|json|csv|yaml|xml|markdown|html|bullet(?:ed)?(?: points?| list)?|numbered list|columns?"
            r"|headings?|code block|plain text|one line|single word|one word)\b"), 1.8, "names an output format"),
        (_R(r"\b(?:format(?:ted)?|respond (?:with|in|using)|reply (?:with|in)|return (?:a|an|the|only|just)"
            r"|output (?:only|just|a|an|the|as|in)|give (?:it|them|me) (?:to me )?(?:in|as)|structure (?:it|the answer))\b"),
         1.6, "asks for a response shape"),
        (_R(r"\b(?:in|under|within|at most|no more than|max(?:imum)?(?: of)?)\s+\d+\s+(?:words?|sentences?|paragraphs?"
            r"|lines?|bullets?|points?|characters?|tokens?)\b|\bword limit\b"), 1.8, "sets an answer length"),
    ],
    "rules": [
        (_R(r"\b(?:must|never|always|do not|don't|dont|should not|shouldn't|must not|mustn't|avoid|refrain"
            r"|make sure|ensure|be sure|under no circumstances|only use|only include|exactly one|at most|at least"
            r"|no more than|no less than|required|mandatory|forbidden|prohibited|strictly)\b"), 1.5, "obligation or prohibition"),
        (_R(r"\b(?:without|unless|except|not allowed|isn't allowed|cannot|can't)\b"), 0.7, "restriction word"),
    ],
    "examples": [
        (_R(r"\b(?:for example|for instance|e\.g\.|example\s*:|sample\s*:|here is an example|like this one)\b"), 2.0,
         "introduces an example"),
        (_R(r"^\W*(?:input|output|q|a)\s*:"), 1.2, "input/output pair"),
        (_R(r"(?:->|=>|→)"), 0.6, "mapping arrow"),
    ],
    "question": [
        (_R(r"^\W*(?:what|why|how|when|where|which|who|whom|whose|is|are|does|do|did|should|shall|will|was|were"
            r"|has|have)\b[^?]*\?\s*$"), 1.6, "direct question"),
    ],
    "task": [
        (_R(r"\b(?:i want you to|i need you to|i'd like you to|i would like you to|your (?:task|job|goal) is"
            r"|i want to know|i need to know|i'd like to know|let me know|tell me|show me|help me (?:to )?\w+"
            r"|if you could|(?:can|could|would|will) you\s+(?:please\s+)?(?:kindly\s+)?(?!help me(?: out)?(?: with something)?\W*$)\w+"
            r"|figure out|determine|go through)\b"), 1.2, "request phrasing"),
        (_R(r"\b(?:classify|categori[sz]e|summari[sz]e|write|list|extract|translate|analy[sz]e|generate|create|explain"
            r"|review|identify|find|compare|rewrite|draft|design|plan|fix|debug|answer|label|tag|rank|score|evaluate"
            r"|calculate|compute|solve|convert|detect|predict)\b"), 0.5, "task verb"),
    ],
    "context": [
        (_R(r"\bmy name is\b|\bi(?: a|')m an? \w+"), 1.2, "introduces the writer"),
        (_R(r"\b(?:i have|we have|i am|i'm|we are|we're|i run|we run|i work|our (?:\w+ )?(?:app|company|team|product"
            r"|users|customers|business|store|service)|background|currently|for context)\b"), 0.6, "background statement"),
    ],
}
_NEGATION_RE = re.compile(r"\b(?:not|never|no|don't|dont|do not|without|avoid)\b", re.I)
_ENSURE_RE = re.compile(r"^\W*(?:\w+\s+)?(?:please\s+)?(?:make sure|ensure|be sure|remember to)\b", re.I)
_FORMAT_CUE_RE = re.compile(r"\b(?:format|schema|template|structure|like this|as follows|following shape)\b", re.I)
_EXAMPLE_CUE_RE = re.compile(r"\b(?:example|e\.g\.|sample|for instance)\b", re.I)


def _segment_kind(text: str, content_start: int, content_end: int, is_list: bool, sections) -> str:
    if is_list:
        return "list_item"
    if any(s.start == content_start and not s.inline for s in sections):
        return "header"
    return "sentence"


def _rule_scores(sentence: str, doc_sent) -> tuple[dict[str, float], list[str]]:
    scores = dict.fromkeys(LABELS, 0.0)
    reasons: list[str] = []
    for label, pats in RULES.items():
        for pat, weight, why in pats:
            if pat.search(sentence):
                scores[label] += weight
                reasons.append(f"{label}: {why}")

    # Padding: greeting/thanks/pleasantry with no content words left.
    content = [t for t in doc_sent if t.is_alpha and not t.is_stop and t.lemma_.lower() not in textutils.FILLER_WORDS
               and t.lower_ not in textutils.FILLER_WORDS]
    polite = bool(textutils.GREETING_RE.search(sentence) or textutils.THANKS_RE.search(sentence)
                  or textutils.PLEASANTRY_RE.search(sentence))
    if not content:
        scores["padding"] += 3.0
        reasons.append("padding: no content words")
    elif len(content) <= 1 and polite:
        scores["padding"] += 2.0
        reasons.append("padding: pleasantry with one content word")
    elif polite:
        scores["padding"] += 0.8
        reasons.append("padding: greeting or thanks")

    # Imperative mood ("Classify ...", "Please list ...") is a task, or a rule if negated.
    toks = [t for t in doc_sent if not t.is_punct and not t.is_space and t.lower_ not in ("please", "kindly", "also", "and", "now", "then")]
    if toks and toks[0].tag_ == "VB" and toks[0].dep_ in ("ROOT", "conj"):
        target = "rules" if _NEGATION_RE.search(sentence) or _ENSURE_RE.search(sentence) else "task"
        scores[target] += 1.0
        reasons.append(f"{target}: imperative verb '{toks[0].text}'")
    if toks and toks[0].lower_ in ("do", "does") and len(toks) > 1 and toks[1].lower_ in ("n't", "not"):
        scores["rules"] += 1.0
        reasons.append("rules: negative imperative")
    return scores, reasons


def _softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = (x - x.max()) * t
    e = np.exp(z)
    return e / e.sum()


_proto_cache: dict[str, tuple[list[str], np.ndarray]] = {}


def _prototype_matrix():
    if "m" not in _proto_cache:
        labels, texts = [], []
        for label, examples in PROTOTYPES.items():
            labels += [label] * len(examples)
            texts += examples
        _proto_cache["m"] = (labels, resources.embed(texts))
    return _proto_cache["m"]


def _embedding_votes(emb: np.ndarray) -> np.ndarray:
    """(n_segments, n_labels) probability that each segment looks like each label."""
    labels, protos = _prototype_matrix()
    sims = emb @ protos.T
    per_label = np.full((emb.shape[0], len(LABELS)), -1.0, dtype="float32")
    for j, label in enumerate(labels):
        k = LABELS.index(label)
        per_label[:, k] = np.maximum(per_label[:, k], sims[:, j])
    return np.vstack([_softmax(row, 12.0) for row in per_label]) if len(per_label) else per_label


def _content_lemmas(doc_span) -> str:
    return " ".join(sorted({t.lemma_.lower() for t in doc_span if t.is_alpha and not t.is_stop
                            and t.lower_ not in textutils.FILLER_WORDS}))


def _block_label(block, prev_text: str, section_label: str | None) -> tuple[str, str]:
    if section_label:
        return section_label, f"inside '{section_label}' section"
    if _EXAMPLE_CUE_RE.search(prev_text):
        return "examples", "introduced as an example"
    if _FORMAT_CUE_RE.search(prev_text) or (block.kind == "table" and block.meta.get("rows", 1) == 0):
        return "format", "introduced as the answer shape"
    return "data", f"{block.kind} block kept whole"


def run(s1: S1Result) -> S2Result:
    cfg = thresholds()["s2"]
    text = s1.clean_text
    raw_blocks = textutils.find_blocks(text)
    sections = textutils.find_sections(text, raw_blocks)

    # 1. Atomic units: blocks whole, prose split into sentences / list items / headers.
    units: list[tuple[int, int, str, object]] = []   # (start, end, kind, block|None)
    pos = 0
    for b in raw_blocks:
        units += [(s, e, "list" if is_list else "prose", None) for s, e, is_list in textutils.sentence_spans(text, pos, b.start)]
        units.append((b.start, b.end, b.kind, b))
        pos = b.end
    units += [(s, e, "list" if is_list else "prose", None) for s, e, is_list in textutils.sentence_spans(text, pos)]
    units.sort(key=lambda u: u[0])

    def section_of(start: int):
        for s in sections:
            if s.start <= start < max(s.end, s.header_end) or start == s.start:
                return s
        return None

    # 2. Rule scores + embedding votes for prose units.
    doc = resources.parse(text)
    prose_idx = [i for i, u in enumerate(units) if u[3] is None]
    emb_all = resources.embed([text[units[i][0]:units[i][1]] for i in range(len(units))])
    votes = _embedding_votes(emb_all[prose_idx]) if prose_idx else np.zeros((0, len(LABELS)))
    vote_of = {i: votes[k] for k, i in enumerate(prose_idx)}

    segments: list[Segment] = []
    blocks: list[Block] = []
    for i, (cs, ce, ukind, raw) in enumerate(units):
        seg_start = 0 if i == 0 else cs
        seg_end = units[i + 1][0] if i + 1 < len(units) else len(text)
        content = text[cs:ce]
        sec = section_of(cs)
        sec_label = sec.label if sec else None
        scores = dict.fromkeys(LABELS, 0.0)

        if raw is not None:
            bid = f"B{len(blocks) + 1}"
            blocks.append(Block(id=bid, kind=raw.kind, start=raw.start, end=raw.end, meta=raw.meta))
            prev = segments[-1].text if segments else ""
            label, why = _block_label(raw, prev, sec_label)
            scores[label] = 1.0
            kind, reasons, conf = raw.kind, [why], 1.0
        else:
            span = doc.char_span(cs, ce, alignment_mode="expand")
            kind = _segment_kind(text, cs, ce, ukind == "list", sections)
            if sec_label and not (sec and sec.inline and kind == "sentence" and cs != sec.start):
                label, reasons, conf = sec_label, [f"inside '{sec.name}' section"], 0.95
                scores[label] = 1.0
            else:
                rule, reasons = _rule_scores(content, span)
                vote = vote_of[i]
                final = np.array([rule[l] for l in LABELS]) + cfg["embedding_weight"] * vote
                best = int(final.argmax())
                if textutils.QUOTE_LINE_RE.match(content) and rule["rules"] == 0 and rule["format"] == 0:
                    best = LABELS.index("data")
                    reasons.append("data: quoted line")
                label = LABELS[best]
                reasons.append(f"embedding vote: {LABELS[int(vote.argmax())]} ({vote.max():.2f})")
                probs = _softmax(final, 2.0)
                conf = float(probs[best])
                scores = {l: round(float(final[k]), 3) for k, l in enumerate(LABELS)}

        hit = next((f for f in s1.injections if textutils.overlaps(cs, ce, f.start, f.end)), None)
        if hit is not None and label != "data":
            label, conf = "data", 1.0
            reasons.insert(0, f"data: fenced — embedded instruction (score {hit.score:.2f})")

        segments.append(Segment(
            id=f"P{i + 1}", start=seg_start, end=seg_end, content_start=cs, content_end=ce, text=content,
            kind=kind, label=label, confidence=round(conf, 3), scores=scores, reasons=reasons,
            section=sec.name if sec else None, remove=label == "padding",
        ))

    # 3. Link parts that say the same thing; keep the most specific one.
    dup_ok = [k for k, s in enumerate(segments) if s.label in ("task", "rules", "format", "question", "context", "role")
              and s.kind in ("sentence", "list_item")]
    lemmas = {k: _content_lemmas(doc.char_span(segments[k].content_start, segments[k].content_end, alignment_mode="expand"))
              for k in dup_ok}
    parent = {k: k for k in dup_ok}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    link_info: dict[tuple[int, int], tuple[float, str]] = {}
    for a_i, a in enumerate(dup_ok):
        for b in dup_ok[a_i + 1:]:
            cos = float(emb_all[a] @ emb_all[b])
            lex = fuzz.token_set_ratio(lemmas[a], lemmas[b]) / 100 if lemmas[a] and lemmas[b] else 0.0
            if cos >= cfg["same_sentence_similarity"]:
                method = f"embedding cosine {cos:.2f}"
            elif lex >= cfg["lexical_overlap"] and cos >= cfg["lexical_min_similarity"]:
                method = f"lemma overlap {lex:.2f}, cosine {cos:.2f}"
            else:
                continue
            parent[find(b)] = find(a)
            link_info[(a, b)] = (cos, method)

    groups: dict[int, list[int]] = {}
    for k in dup_ok:
        groups.setdefault(find(k), []).append(k)
    duplicates: list[DuplicateGroup] = []
    for members in groups.values():
        if len(members) < 2:
            continue

        def specificity(k):
            seg = segments[k]
            return len(lemmas[k].split()) + (1.5 if seg.label == "rules" else 0) + (1.0 if seg.label == "format" else 0)

        keep = max(members, key=lambda k: (specificity(k), -k))
        drop = [k for k in members if k != keep]
        for k in drop:
            segments[k].duplicate_of = segments[keep].id
            segments[k].remove = True
        pairs = [v for (x, y), v in link_info.items() if x in members and y in members]
        duplicates.append(DuplicateGroup(keep=segments[keep].id, drop=[segments[k].id for k in drop],
                                         similarity=round(max(p[0] for p in pairs), 3), method=pairs[0][1]))

    # 4. Requirement IDs for everything that asks something of the model.
    requirements: list[Requirement] = []
    for seg in segments:
        if seg.label in ("task", "rules", "format", "question") and seg.kind != "header":
            if seg.duplicate_of is None:
                seg.requirement_id = f"R{len(requirements) + 1}"
                requirements.append(Requirement(id=seg.requirement_id, segment_id=seg.id, label=seg.label, text=seg.text))
    by_id = {s.id: s for s in segments}
    for seg in segments:
        if seg.duplicate_of:
            seg.requirement_id = by_id[seg.duplicate_of].requirement_id

    label_counts = {l: sum(1 for s in segments if s.label == l) for l in LABELS}
    data_items = sum(1 for s in segments if s.label == "data" and s.kind in ("list_item", "sentence"))
    data_items += sum(b.meta.get("rows", 0) for b in blocks if b.kind == "table")
    reconstructs = "".join(text[s.start:s.end] for s in segments) == text and all(
        segments[k].end == segments[k + 1].start for k in range(len(segments) - 1))

    return S2Result(segments=segments, blocks=blocks, duplicates=duplicates, requirements=requirements,
                    label_counts=label_counts, data_items=data_items, reconstructs=reconstructs)
