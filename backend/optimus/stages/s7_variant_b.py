"""S7 · Build Variant B (short, cost-first), delivered as CSV.

Climbs the PRD's reduction ladder from the safest change to the riskiest. After every step the
whole prompt is checked (every S5 lock present, every requirement still has a part); on the
first failure the step is undone and the climb stops. High-risk tasks (S3) stop at L2.
  L0  drop S2's padding, filler and repeated parts
  L1  remove the words S5 found unnecessary (LLMLingua-2 + S4 informativeness)
  L2  rule-based shortening: lead-ins, politeness, wordy phrases ("each and every" → "every")
  L3  constrained rewrite: format sentences become "Output: …", wordy clauses get short forms
  L4  extractive long-context compressor — not in this build (PRD P1.1)
  L5  shape the answer: answer only, length budget from S3 (never below its floor)
It stays in full sentences, never a keyword list. The result is also written as `field,value` CSV
rows (task, rules, format, the data, the answer limit); Variant B is sent in whichever of the two
forms takes fewer tokens (`format`), and its cost is for that form.
"""
from __future__ import annotations

import csv
import io

import regex as re

from .. import resources
from ..schemas import (LadderStep, PIRSegment, S0Result, S2Result, S3Result, S4Result, S5Result, S7Result)
from . import s6_variant_a as va

L3_RULES = [(re.compile(p, re.I), r) for p, r in [
    (r"^(?:answer|respond|reply|return (?:it|the answer|them)|give (?:it|them|the answer)(?: to me)?)\s+(?:in|as)\s+"
     r"(?:a |an |the )?", "Output: "),
    (r"\bin (?:a|the) (\w+ )?format\b", r"as \1"),
    (r"\bif there is one\b", "(if any)"),
    (r"\bif there are any\b", "(if any)"),
    (r"\bgo through (?:them|these|those|it|the \w+) and\s+", ""),
    (r"\bthat (?:is|are) (not)\b", r"\1"),
    (r"\bat the end of the day\b|\bas (?:a|an) (?:matter of fact|aside)\b", ""),
    (r"\bin each (\w+)\b", r"per \1"),
]]
LEVEL_NAMES = {"L0": "Drop padding and repeats", "L1": "Remove unnecessary words", "L2": "Shorten wording (rules)",
               "L3": "Constrained rewrite", "L4": "Long-context compressor", "L5": "Shape the answer"}


def _assemble(pieces: list[tuple[PIRSegment, str]], text: str, extra: str = "") -> str:
    """Join kept parts in prompt order: running sentences with a space, and the strongest line
    break that stood between two kept parts in the original."""
    out, prev_end = "", None
    for p, body in pieces:
        if not body:
            continue
        if prev_end is None:
            out = body
        else:
            gap = text[prev_end:p.start]
            sep = "\n\n" if "\n\n" in gap else "\n" if "\n" in gap else " "
            out += sep + body
        prev_end = p.end
    out = re.sub(r"[ \t]+\n", "\n", out).strip()
    return f"{out}\n{extra}" if extra else out


CSV_ORDER = ["role", "context", "task", "rules", "format", "limit"]     # instructions first, then the data rows


def _field(p: PIRSegment, segs: dict) -> str:
    if p.label == "data":
        return "documents" if p.kind == "tagged" else va.tag_name(segs[p.id].section, "data")
    return {"question": "task", "examples": "example"}.get(p.label, p.label)


def _csv(pieces: list[tuple[PIRSegment, str]], segs: dict, limit: str) -> str:
    """One row per instruction field (its parts joined in prompt order), the answer limit, then one
    row per data item; a data header ("Reviews:") becomes the field name of its rows."""
    instr: dict[str, list[str]] = {}
    data: list[list[str]] = []
    for p, body in pieces:
        if not body or (p.label == "data" and p.kind == "header"):
            continue
        field = _field(p, segs)
        if p.label == "data" or field not in CSV_ORDER:
            data.append([field, body])
        else:
            instr.setdefault(field, []).append(body)
    if limit:
        instr["limit"] = [limit]
    rows = [[f, (" " if f != "format" else "\n").join(instr[f])] for f in CSV_ORDER if f in instr] + data
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["field", "value"])
    w.writerows(rows)
    return buf.getvalue().rstrip("\n")


def _csv_values(text: str) -> str:
    return "\n".join(v for row in csv.reader(io.StringIO(text)) for v in row[1:])


def run(s2: S2Result, s3: S3Result, s4: S4Result, s5: S5Result, s0: S0Result | None = None,
        clean_text: str = "") -> S7Result:
    enc = resources.tiktoken_encoding("o200k_base")
    segs = {s.id: s for s in s2.segments}
    h = s5.handoff
    top = "L2" if h.max_ladder_level == "L2" else "L5"
    rank = list(LEVEL_NAMES)

    def ok(prompt: str, pieces) -> str | None:
        """Why the prompt fails the checks, or None."""
        if missing := [t for t in s5.lock_set if t not in prompt]:
            return f"lost lock(s): {', '.join(m[:30] for m in missing[:3])}"
        live = {p.id for p, b in pieces if b}
        for r in s2.requirements:
            ids = [s.id for s in s2.segments if s.requirement_id == r.id]
            if not any(i in live or (segs[i].duplicate_of in live) for i in ids):
                return f"requirement {r.id} lost its part"
        return None

    steps: list[LadderStep] = []
    parts = va.variant_parts(s2, s5, s0 is not None and s0.used)
    pieces = [(p, p.text) for p in parts]                # the prompt as written
    limit = ""
    stopped = False

    def step(level: str, new_pieces, new_limit: str | None = None, why_skip: str | None = None, note: str = ""):
        nonlocal pieces, limit, stopped
        current = _assemble(pieces, clean_text, limit)
        if why_skip or stopped:
            steps.append(LadderStep(level=level, name=LEVEL_NAMES[level], status="skipped",
                                    reason=why_skip or "an earlier step failed its checks",
                                    tokens=len(enc.encode(current)), prompt=current))
            return
        cand_limit = limit if new_limit is None else new_limit
        cand = _assemble(new_pieces, clean_text, cand_limit)
        if cand == current:
            steps.append(LadderStep(level=level, name=LEVEL_NAMES[level], status="no change", reason=note or "nothing to change",
                                    tokens=len(enc.encode(current)), prompt=current))
            return
        if fail := ok(cand, new_pieces):
            stopped = True
            steps.append(LadderStep(level=level, name=LEVEL_NAMES[level], status="reverted",
                                    reason=f"{fail}; kept the previous step", tokens=len(enc.encode(current)),
                                    prompt=current))
            return
        pieces, limit = new_pieces, cand_limit
        steps.append(LadderStep(level=level, name=LEVEL_NAMES[level], status="applied", reason=note,
                                tokens=len(enc.encode(cand)), prompt=cand))

    def beyond(level: str) -> str | None:
        return (f"S3 rates the task {h.risk_level} risk ({', '.join(s3.risk.domains) or 'numeric rules'}): "
                f"Variant B stops at {top}") if rank.index(level) > rank.index(top) else None

    dropped = [p for p in s5.segments if p.action == "drop"]
    step("L0", [(p, "" if p.action == "drop" else b) for p, b in pieces],
         note=f"dropped {len(dropped)} part(s) S2 marked as padding, filler or repeats")
    n_comp = sum(p.action == "compress" for p in s5.segments)
    step("L1", [(p, p.compressed if b else "") for p, b in pieces],
         note=f"{n_comp} part(s) lose words S5 scored as unnecessary ({s5.compression.method})")

    def rewrite(rules, body: str, p: PIRSegment) -> str:
        if not body or p.kind not in ("sentence", "list_item") or p.label not in va.INSTRUCTION_LABELS:
            return body
        return va.sentence(va.safe_tidy(body, p, s5, rules))

    step("L2", [(p, rewrite(va.INNER, b, p)) for p, b in pieces], why_skip=beyond("L2"),
         note="lead-ins, politeness and wordy phrases cut where they touch no lock")
    step("L3", [(p, rewrite(L3_RULES, b, p)) for p, b in pieces], why_skip=beyond("L3"),
         note="format sentences become 'Output: …'; long clauses get short forms")
    step("L4", pieces, why_skip="not in this build (PRD P1.1, extractive compressor for long context)")
    answer, basis = va.answer_estimate(s3, shaped=True)
    budget = max(s3.answer_length.floor, round(1.2 * answer))
    step("L5", pieces, new_limit=f"Answer only, no preamble, under {budget} tokens.", why_skip=beyond("L5"),
         note=f"answer budget {budget} tokens (1.2 × {answer}, floor {s3.answer_length.floor}); "
              f"API max_tokens {round(1.5 * budget)}")

    prompt = _assemble(pieces, clean_text, limit)
    table = _csv(pieces, segs, limit)
    reached = next((s.level for s in reversed(steps) if s.status == "applied"), "none")
    shaped = bool(limit)
    if not shaped:
        answer, basis = va.answer_estimate(s3, shaped=False)
        budget = s3.answer_length.budget
    csv_tokens = len(enc.encode(table))

    place_of = {p.id: _field(p, segs) + " row" for p, b in pieces if b}
    reqs, where = va.trace(s2, s5, place_of)
    checks = va.check(_csv_values(table), s2, s5, where)
    # Send whichever form costs fewer tokens: CSV saves on repeated structure, but its field names
    # and doubled quotes cost tokens on prompts with quoted data or code.
    plain_tokens = len(enc.encode(prompt))
    fmt = "plain" if plain_tokens < csv_tokens else "csv"
    best = min(plain_tokens, csv_tokens)
    diff = abs(csv_tokens - plain_tokens)
    reason = (f"Plain text is {diff} token{'s' if diff != 1 else ''} shorter than CSV ({plain_tokens} vs {csv_tokens}): "
              "CSV's field names and quoting cost more than they save here" if fmt == "plain" else
              f"CSV is {diff} token{'s' if diff != 1 else ''} shorter than plain text ({csv_tokens} vs {plain_tokens})"
              if diff else f"Both forms take {best} tokens; CSV keeps the fields separable")
    return S7Result(prompt=prompt, csv=table, tokens=plain_tokens, csv_tokens=csv_tokens, format=fmt, format_reason=reason,
                    best_tokens=best, steps=steps,
                    reached=reached, max_ladder_level=h.max_ladder_level, answer_tokens_est=answer, answer_basis=basis,
                    answer_budget=budget, max_tokens=round(1.5 * budget),
                    baseline=va.baseline(s0, s4, s5, s3.answer_length.expected_tokens),
                    cost=va.price(s4, best, answer), checks=checks, trace=reqs)
