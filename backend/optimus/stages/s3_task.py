"""S3 · Understand the task.

Task type: lemma keyword evidence (spaCy) over the instruction parts, combined with
sentence-embedding similarity to task prototypes. Difficulty: weighted count of sub-tasks,
rules, reasoning words and length. Risk: legal / medical / financial vocabulary, code, or
three or more numeric rules. Gaps: missing audience, format, length, scope, or vague wording.
Answer length: a per-task table adjusted by explicit cues ("in 50 words", "list 5", item count),
with a per-task floor so the budget never backfires.
"""
from __future__ import annotations

import math

import numpy as np
import regex as re

from .. import resources, textutils
from ..config import thresholds
from ..schemas import AnswerLength, Difficulty, Gap, Risk, S1Result, S2Result, S3Result, TaskType

TASK_TYPES = ["code", "debug", "summarise", "extract", "classify", "qa_docs", "reasoning",
              "creative", "translate", "plan", "chat"]

TASK_LABELS = {
    "code": "Write code", "debug": "Debug", "summarise": "Summarise", "extract": "Extract",
    "classify": "Classify", "qa_docs": "Answer from documents", "reasoning": "Reasoning / maths",
    "creative": "Creative", "translate": "Translate", "plan": "Plan / analyse", "chat": "Chat",
}

# lemma / phrase evidence, weight
KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "code": [(r"\b(?:write|implement|build|create)\b.{0,30}\b(?:function|class|script|program|api|endpoint|query|regex|component)\b", 2.5),
             (r"\b(?:python|javascript|typescript|java|sql|rust|golang|c\+\+|react|bash|html|css)\b", 1.2),
             (r"\b(?:function|algorithm|code|refactor|unit tests?)\b", 1.0)],
    "debug": [(r"\b(?:debug|fix|bug|error|traceback|exception|stack trace|crash(?:es|ing)?|not working|fails?|failing)\b", 1.5)],
    "summarise": [(r"\b(?:summari[sz]e|summary|tl;?dr|condense|key points|main points|gist|overview|recap)\b", 2.5)],
    "extract": [(r"\b(?:extract|pull out|find all|list all|identify all|parse|fields?|entities|named entities)\b", 2.0),
                (r"\b(?:main complaint|names?|dates?|emails?|phone numbers?|amounts?)\b\s+(?:in|from)\b", 0.8)],
    "classify": [(r"\b(?:classify|classif\w+|categori[sz]e|label|tag|sentiment|which category|sort into|bucket)\b", 2.5),
                 (r"\b(?:positive|negative|neutral|spam|not spam|relevant|irrelevant)\b", 0.8)],
    "qa_docs": [(r"\b(?:according to|based on the (?:text|document|passage|context|article)|from the (?:document|passage|text)"
                 r"|using (?:only )?the (?:provided|given|above))\b", 2.0),
                (r"\?\s*$", 0.5)],
    "reasoning": [(r"\b(?:solve|calculate|compute|prove|derive|probability|equation|how many|how much|step by step"
                   r"|logic(?:al)?|puzzle|math(?:s|ematics)?)\b", 2.0)],
    "creative": [(r"\b(?:story|poem|haiku|lyrics|song|slogan|tagline|essay|blog post|tweet|caption|ad copy|fiction|creative"
                  r"|imagine|script for)\b", 2.0)],
    "translate": [(r"\b(?:translate|translation|in(?:to)? (?:english|hindi|marathi|french|spanish|german|tamil|telugu"
                   r"|bengali|japanese|chinese|arabic|kannada|gujarati))\b", 3.0)],
    "plan": [(r"\b(?:plan|roadmap|schedule|itinerary|strategy|steps to|outline|design a|proposal|timeline|milestones)\b", 2.0),
             (r"\b(?:analy[sz]e|evaluate|compare|pros and cons|trade-?offs?|recommend)\b", 1.0)],
    "chat": [],
}
_KEYWORD_RES = {t: [(re.compile(p, re.I | re.M), w) for p, w in pats] for t, pats in KEYWORDS.items()}

PROTOTYPES = {
    "code": "Write a Python function that parses a CSV file.",
    "debug": "Fix the bug in this code; it throws a KeyError.",
    "summarise": "Summarize this article in three bullet points.",
    "extract": "Extract all names, dates and amounts from this invoice.",
    "classify": "Classify each review as positive, negative or neutral.",
    "qa_docs": "Based on the passage above, answer the question.",
    "reasoning": "Solve this math problem step by step.",
    "creative": "Write a short poem about the monsoon.",
    "translate": "Translate this paragraph into Hindi.",
    "plan": "Create a two-week plan to launch our product.",
    "chat": "Hey, how is it going? Tell me something fun.",
}

REASONING_WORDS = re.compile(r"\b(?:why|prove|step|steps|reason|reasoning|explain|derive|justify|compare|analy[sz]e"
                             r"|evaluate|trade-?offs?|implications?|calculate|infer)\b", re.I)

RISK_DOMAINS = {
    "legal": re.compile(r"\b(?:legal|law|lawsuit|litigation|contract|clause|liabilit\w+|compliance|gdpr|hipaa|regulat\w+"
                        r"|statute|court|attorney|lawyer|plaintiff|defendant|terms of service|nda|copyright|trademark)\b", re.I),
    "medical": re.compile(r"\b(?:medical|diagnos\w+|symptoms?|dosage|dose|medication|prescri\w+|patients?|clinical"
                          r"|disease|treatment|therapy|drug|surgery|allerg\w+|mental health|doctor)\b", re.I),
    "financial": re.compile(r"\b(?:financ\w+|invest\w*|portfolio|tax(?:es)?|loans?|mortgage|interest rate|stocks?|trading"
                            r"|credit score|revenue|profit|accounting|audit|invoices?|banking|insurance|crypto\w*"
                            r"|securities|dividend|valuation)\b", re.I),
}

_AUDIENCE_RE = re.compile(r"\b(?:for (?:a |an |the |our )?(?:\w+ ){0,2}(?:audience|readers?|team|beginners?|experts?"
                          r"|students?|customers?|users?|managers?|executives?|kids|children|developers?|engineers?)"
                          r"|audience|reader|non-technical|technical|layman|eli5|explain like)\b", re.I)
_LENGTH_RE = re.compile(r"\b(?:\d+\s*(?:words?|sentences?|paragraphs?|lines?|bullets?|points?|characters?|tokens?|items?)"
                        r"|brief(?:ly)?|short|concise|one[- ](?:line|word|sentence|paragraph)|single (?:word|sentence)"
                        r"|in detail|detailed|comprehensive|long-form|word limit|at most|no more than|under \d+)\b", re.I)
_NUMBER_RE = re.compile(r"\d")
_WORD_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20}
_NUM = r"(\d+|" + "|".join(_WORD_NUMS) + r")"

# Typical answer size by task: (base tokens, tokens per input item)
LENGTH_TABLE = {
    "classify": (20, 25), "extract": (40, 30), "summarise": (150, 0), "qa_docs": (120, 0), "reasoning": (220, 0),
    "code": (300, 0), "debug": (250, 0), "creative": (400, 0), "translate": (0, 0), "plan": (380, 0), "chat": (80, 0),
}


def _num(s: str) -> int:
    return int(s) if s.isdigit() else _WORD_NUMS.get(s.lower(), 0)


def classify_task(instr_text: str, has_code_block: bool) -> TaskType:
    kw = np.zeros(len(TASK_TYPES))
    for k, t in enumerate(TASK_TYPES):
        for pat, w in _KEYWORD_RES[t]:
            kw[k] += w * min(3, len(pat.findall(instr_text)))
    if has_code_block:
        kw[TASK_TYPES.index("code")] += 1.0
        kw[TASK_TYPES.index("debug")] += 0.5
    protos = resources.embed([PROTOTYPES[t] for t in TASK_TYPES])
    q = resources.embed([instr_text or "chat"])[0]
    sims = protos @ q
    emb = np.exp((sims - sims.max()) * 15)
    emb /= emb.sum()
    total = kw / (kw.sum() + 1e-9) * 0.65 + emb * 0.35 if kw.sum() > 0 else emb
    if kw.sum() == 0 and sims.max() < 0.3:
        total = np.eye(len(TASK_TYPES))[TASK_TYPES.index("chat")]
    best = int(total.argmax())
    return TaskType(type=TASK_TYPES[best], confidence=round(float(total[best]), 3),
                    scores={t: round(float(v), 3) for t, v in zip(TASK_TYPES, total)})


def estimate_difficulty(s2: S2Result, instr_text: str, task: str, n_tokens: int) -> Difficulty:
    cfg = thresholds()["s3"]["difficulty"]
    live = [s for s in s2.segments if not s.remove]
    subtasks = sum(1 for s in live if s.label in ("task", "question"))
    subtasks += len(re.findall(r"\b(?:and also|also|additionally|then|as well as|plus)\b", instr_text, re.I))
    rules = sum(1 for s in live if s.label == "rules")
    reasoning = len(REASONING_WORDS.findall(instr_text))
    length = max(0.0, math.log2(max(n_tokens, 1) / 100))
    type_weight = {"reasoning": 2.0, "code": 2.0, "debug": 2.0, "plan": 1.5, "qa_docs": 1.0}.get(task, 0.0)
    factors = {"sub_tasks": float(subtasks), "rules": rules * 0.5, "reasoning_words": reasoning * 1.5,
               "length": round(length, 2), "task_type": type_weight}
    score = sum(factors.values())
    level = "high" if score >= cfg["high_at"] else "medium" if score >= cfg["medium_at"] else "low"
    return Difficulty(level=level, score=round(score, 2), factors=factors)


def assess_risk(s2: S2Result, instr_text: str, data_text: str, task: str) -> Risk:
    domains, reasons = [], []
    for name, pat in RISK_DOMAINS.items():
        hits_instr = {m.lower() for m in pat.findall(instr_text)}
        hits_data = {m.lower() for m in pat.findall(data_text)}
        if hits_instr or len(hits_data) >= 2:
            domains.append(name)
            reasons.append(f"{name} terms: {', '.join(sorted(hits_instr | hits_data)[:5])}")
    if task in ("code", "debug") or s2.label_counts.get("data", 0) and any(b.kind == "code" for b in s2.blocks):
        domains.append("code")
        reasons.append("code is involved")
    numeric_rules = [s for s in s2.segments if s.label in ("rules", "format") and _NUMBER_RE.search(s.text)]
    need = thresholds()["s3"]["numeric_rules_for_high_risk"]
    if len(numeric_rules) >= need:
        reasons.append(f"{len(numeric_rules)} rules contain numbers")
    level = "high" if domains or len(numeric_rules) >= need else "medium" if numeric_rules else "low"
    return Risk(level=level, domains=domains, reasons=reasons or ["no legal, medical, financial or code signals"])


def find_gaps(s2: S2Result, instr_text: str, task: str, has_data: bool, budget: int) -> list[Gap]:
    gaps: list[Gap] = []
    labels = {s.label for s in s2.segments if not s.remove}
    if "format" not in labels:
        default = {"classify": "a Markdown table with one row per item", "extract": "a JSON object",
                   "summarise": "3–5 bullet points", "code": "a single code block with a one-line explanation"}.get(task, "plain text, no preamble")
        gaps.append(Gap(kind="format", message="No answer format given.", assumption=f"Answer as {default}."))
    if not _LENGTH_RE.search(instr_text) and task not in ("code", "translate"):
        gaps.append(Gap(kind="length", message="No answer length given.",
                        assumption=f"Keep the answer under about {budget} tokens."))
    if task in ("summarise", "creative", "plan", "qa_docs", "reasoning", "chat") and not _AUDIENCE_RE.search(instr_text):
        gaps.append(Gap(kind="audience", message="No audience named.",
                        assumption="Write for a general professional audience."))
    if task in ("classify", "extract", "summarise", "translate", "qa_docs", "debug") and not has_data:
        gaps.append(Gap(kind="data", message="The task needs input data but none was found.",
                        assumption="The input will be supplied at the {{input}} placeholder."))
    for seg in s2.segments:
        if seg.remove or seg.label not in ("task", "rules", "format", "question"):
            continue
        m = textutils.VAGUE_RE.search(seg.text)
        if m:
            gaps.append(Gap(kind="ambiguity", message=f"Vague wording: '{m.group(0)}'.",
                            assumption="Interpret it as the most common reading and state that choice.",
                            evidence=seg.text))
    if task in ("classify",) and not re.search(r"\b(?:mixed|both|tie|unclear|ambiguous)\b", instr_text, re.I):
        gaps.append(Gap(kind="scope", message="No rule for items that fit more than one label.",
                        assumption='An item with both praise and a complaint is "neutral" unless one side clearly dominates.'))
    return gaps


def estimate_answer_length(task: str, instr_text: str, items: int, data_tokens: int, has_format: bool) -> AnswerLength:
    cfg = thresholds()["s3"]
    base, per_item = LENGTH_TABLE[task]
    basis = []
    if task == "translate":
        expected = int(data_tokens * 1.15) + 10
        basis.append("translation ≈ 1.15 × input data tokens")
    elif task == "summarise" and data_tokens:
        expected = min(400, int(0.25 * data_tokens) + 60)
        basis.append("summary ≈ 25% of the input data + 60")
    else:
        expected = base + per_item * items
        basis.append(f"{TASK_LABELS[task]}: {base} tokens" + (f" + {per_item} × {items} items" if per_item and items else ""))

    explicit = None
    for pat, mult, why in [
        (rf"\b{_NUM}\s+words?\b", 1.35, "words"),
        (rf"\b{_NUM}\s+sentences?\b", 22, "sentences"),
        (rf"\b{_NUM}\s+paragraphs?\b", 90, "paragraphs"),
        (rf"\b{_NUM}\s+(?:bullet(?:s| points?)?|points|items|lines|ideas|examples|tips|steps)\b", 20, "list items"),
        (rf"\b(?:top|list)\s+{_NUM}\b", 20, "list items"),
    ]:
        m = re.search(pat, instr_text, re.I)
        if m:
            n = _num(m.group(1))
            explicit = int(n * mult) + 10
            basis.append(f"explicit cue '{m.group(0)}' → {explicit} tokens")
            break
    if explicit is not None:
        expected = explicit
    if re.search(r"\bone[- ]word\b|\bsingle word\b|\bonly the label\b", instr_text, re.I):
        expected = min(expected, 5 * max(items, 1) + 10)
        basis.append("one-word answers")
    if re.search(r"\b(?:brief(?:ly)?|short|concise|succinct)\b", instr_text, re.I):
        expected = int(expected * 0.6)
        basis.append("'brief/short' × 0.6")
    if re.search(r"\b(?:detailed|in depth|in-depth|comprehensive|thorough)\b", instr_text, re.I):
        expected = int(expected * 1.6)
        basis.append("'detailed' × 1.6")
    if not has_format:
        expected = int(expected * 1.3)
        basis.append("no format given → chatty preamble × 1.3")

    floor_cfg = cfg["floors"].get(task, 40)
    floor = max(floor_cfg["base"], floor_cfg["per_item"] * items) if isinstance(floor_cfg, dict) else int(floor_cfg)
    budget = max(floor, math.ceil(cfg["answer_budget_multiplier"] * expected))
    max_tokens = math.ceil(cfg["max_tokens_multiplier"] * budget)
    basis.append(f"budget = max(floor {floor}, ⌈1.2 × {expected}⌉); max_tokens = ⌈1.5 × budget⌉")
    return AnswerLength(expected_tokens=expected, floor=floor, budget=budget, max_tokens=max_tokens, basis=basis)


def run(s1: S1Result, s2: S2Result) -> S3Result:
    live = [s for s in s2.segments if not s.remove]
    instr_text = " ".join(s.text for s in live if s.label in ("role", "task", "rules", "format", "question", "context"))
    data_text = " ".join(s.text for s in s2.segments if s.label in ("data", "examples"))
    has_code = any(b.kind == "code" for b in s2.blocks)
    enc = resources.tiktoken_encoding("o200k_base")
    n_tokens = len(enc.encode(s1.clean_text))
    data_tokens = len(enc.encode(data_text)) if data_text else 0

    task = classify_task(instr_text, has_code)
    items = s2.data_items
    has_format = any(s.label == "format" for s in live)
    answer = estimate_answer_length(task.type, instr_text, items, data_tokens, has_format)
    return S3Result(
        task_type=task,
        difficulty=estimate_difficulty(s2, instr_text, task.type, n_tokens),
        risk=assess_risk(s2, instr_text, data_text, task.type),
        gaps=find_gaps(s2, instr_text, task.type, bool(data_text), answer.budget),
        answer_length=answer,
        items=items,
    )
