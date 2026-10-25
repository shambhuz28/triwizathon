"""Pipeline driver: runs S0 → S1 → S2 → S3 → S4 → S5 → S6 + S7.

Each stage lives in its own module under optimus/stages/ and exposes a single `run(...)`.
Every stage reads the outputs of the stages before it: S3 and S4 work on S2's parts, and S5
uses S2–S4 to decide which words LLMLingua-2 may remove. S5's result (locks, compressed
prompt, per-part decisions, handoff) is the input S6 and S7 build their variants from: S6 the
detailed Variant A, S7 the short Variant B (as CSV).
S0 runs only when files are attached: it pulls the passages the prompt is about from their
FAISS indexes and puts them in front of the prompt, so S1–S5 see them as one data block.
To change a step, edit its module; to add a stage, add a module and a line in `analyze`.

    python -m optimus.pipeline path/to/prompt.txt      # print a summary
    python -m optimus.pipeline path/to/prompt.txt --json
    python -m optimus.pipeline path/to/prompt.txt --doc=report.pdf --doc=data.csv   # upload, then ask about them
"""
from __future__ import annotations

import asyncio
import sys
import time

from .schemas import AnalyzeRequest, AnalyzeResponse, StageTiming
from .stages import s0_retrieve, s1_clean, s2_parse, s3_task, s4_profile, s5_lock, s6_variant_a, s7_variant_b

STAGES = {
    "S0": "Retrieve from documents",
    "S1": "Clean up",
    "S2": "Break into parts",
    "S3": "Understand the task",
    "S4": "Count and profile",
    "S5": "Lock and compress",
    "S6": "Variant A (detailed)",
    "S7": "Variant B (short, CSV)",
}


async def _timed(stage: str, timings: list[StageTiming], fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = await asyncio.to_thread(fn, *args, **kwargs)
    timings.append(StageTiming(stage=stage, name=STAGES[stage], ms=round((time.perf_counter() - t0) * 1000, 1)))
    return result


async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    timings: list[StageTiming] = []
    t0 = time.perf_counter()

    s0 = None
    if req.document_ids:
        s0 = await _timed("S0", timings, s0_retrieve.run, req.prompt, req.document_ids, mode=req.rag_mode)
    s1 = await _timed("S1", timings, s1_clean.run, s0_retrieve.augment(req.prompt, s0), redact_pii=req.redact_pii)
    s2 = await _timed("S2", timings, s2_parse.run, s1)
    s3 = await _timed("S3", timings, s3_task.run, s1, s2)
    s4 = await _timed("S4", timings, s4_profile.run, s1, s2, s3, target_model=req.target_model,
                      monthly_volume=req.monthly_volume)
    s5 = await _timed("S5", timings, s5_lock.run, s1, s2, s3, s4)
    s6 = await _timed("S6", timings, s6_variant_a.run, s1, s2, s3, s4, s5, s0)
    s7 = await _timed("S7", timings, s7_variant_b.run, s2, s3, s4, s5, s0, s1.clean_text)

    return AnalyzeResponse(s0=s0, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6, s7=s7, timings=timings,
                           total_ms=round((time.perf_counter() - t0) * 1000, 1))


def analyze_sync(prompt: str, **kwargs) -> AnalyzeResponse:
    return asyncio.run(analyze(AnalyzeRequest(prompt=prompt, **kwargs)))


def _summary(r: AnalyzeResponse) -> str:
    out = []
    if r.s0:
        out.append(f"S0  used={r.s0.used}  best={r.s0.best_score} (needs {r.s0.threshold})  {r.s0.reason}")
        out += [f"    {p.score:.2f} {p.filename} {p.location or ''} chunks={p.chunks} {p.text[:60]!r}" for p in r.s0.passages]
    out += [f"S1  language={r.s1.language.code} ({r.s1.language.confidence})  changes={[c.kind for c in r.s1.changes]}"
           f"  pii={len(r.s1.pii)}  injections={len(r.s1.injections)}",
           f"S2  {len(r.s2.segments)} parts, {len(r.s2.blocks)} blocks, reconstructs={r.s2.reconstructs}"]
    for s in r.s2.segments:
        flag = " ✗dup→" + s.duplicate_of if s.duplicate_of else (" ✗" if s.remove else "")
        out.append(f"    {s.id:>4} {s.label:<9} {s.confidence:.2f} {s.requirement_id or '':<4}{flag:<10} {s.text[:70]!r}")
    t = r.s3
    out.append(f"S3  task={t.task_type.type} ({t.task_type.confidence})  difficulty={t.difficulty.level}  risk={t.risk.level}"
               f"  items={t.items}  answer≈{t.answer_length.expected_tokens} budget={t.answer_length.budget}"
               f" max_tokens={t.answer_length.max_tokens}  task parts={t.task_segments}")
    out += [f"    gap[{g.kind}] {g.message}" for g in t.gaps]
    p = r.s4
    out.append(f"S4  tokens={p.primary.count} ({p.primary.tokenizer}{'' if p.primary.exact else ' ±20%'})  removable={p.removable_tokens}"
               f"  quality={p.quality.overall}  cost/call=${p.cost.per_call:.6f}  per 10k=${p.cost.per_10k:.2f}")
    out.append("    surprising: " + ", ".join(f"{w.text}({w.score:.2f})" for w in p.surprising))
    out.append("    low-info:   " + " | ".join(l.text for l in p.low_info[:15]))
    out += [f"    warn[{w.severity}] {w.message}" for w in p.warnings]
    out.append(f"S5  {len(r.s5.spans)} locks {r.s5.by_category}  label sets={[ls.items for ls in r.s5.label_sets]}")
    out += [f"    {s.id:>4} {s.category:<11} {s.text[:60]!r}" for s in r.s5.spans if s.parent is None]
    c = r.s5.compression
    out.append(f"    compressed {c.original_tokens} → {c.compressed_tokens} tokens ({1 - c.ratio:.0%} saved)"
               f"  locks preserved={c.locks_preserved}  method={c.method}")
    out += [f"    {p.id:>4} {p.action:<8} {p.keep_rate:.2f} {p.compressed[:70]!r}" for p in r.s5.segments if p.action != "keep"]
    out.append("    prompt: " + c.prompt.replace("\n", "\n            "))
    h = r.s5.handoff
    out.append(f"    handoff  max ladder={h.max_ladder_level}  budget={h.answer_budget}  sections={h.sections}")
    a, b = r.s6, r.s7
    out.append(f"S6  Variant A {a.tokens} tokens (limit {a.size_limit}) vs {a.baseline.label} {a.baseline.tokens}"
               f"  sections={[x.key for x in a.sections]}  locks kept={a.checks.locks_preserved}"
               f"  ${a.cost.per_call:.6f}/call vs ${a.baseline.cost.per_call:.6f}")
    out.append("    " + a.prompt.replace("\n", "\n    "))
    out.append(f"S7  Variant B {b.csv_tokens} tokens as CSV ({b.tokens} plain)  reached={b.reached} (max {b.max_ladder_level})"
               f"  locks kept={b.checks.locks_preserved}  ${b.cost.per_call:.6f}/call")
    out += [f"    {s.level} {s.status:<9} {s.tokens:>5}  {s.reason}" for s in b.steps]
    out.append("    " + b.csv.replace("\n", "\n    "))
    out.append("time " + "  ".join(f"{t.stage}={t.ms:.0f}ms" for t in r.timings) + f"  total={r.total_ms:.0f}ms")
    return "\n".join(out)


if __name__ == "__main__":
    from pathlib import Path

    from . import documents

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    text = open(args[0], encoding="utf-8").read() if args else sys.stdin.read()
    doc_ids = [documents.store().add(Path(p).name, Path(p).read_bytes()).id
               for a in sys.argv[1:] if a.startswith("--doc=") for p in [a[len("--doc="):]]]
    result = analyze_sync(text, redact_pii="--redact" in sys.argv, document_ids=doc_ids)
    print(result.model_dump_json(indent=2) if "--json" in sys.argv else _summary(result))
