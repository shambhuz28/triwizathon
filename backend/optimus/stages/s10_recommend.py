"""S10 · Compare and recommend: the original prompt, Variant A (S6) or Variant B (S7).

Severity comes from S3: risk counts double (legal, medical, money, code, numeric rules), plus
difficulty. A variant is only eligible when its checks passed (every lock, every requirement,
the size limit). Then, as in the PRD:
  high severity    Variant A, the most precise; the original if A failed its checks
  medium           Variant B with A as the fallback when B saves enough, else Variant A
  low              Variant B; the original when B would not save enough to be worth a rewrite
Every trade-off line is generated from the measured numbers, never written freely. Answer
quality is not measured here (no model is called), so precision is stated as expected, not scored.
"""
from __future__ import annotations

from ..schemas import PromptOption, Recommendation, S0Result, S3Result, S4Result, S6Result, S7Result

LEVEL = {"low": 0, "medium": 1, "high": 2}
MIN_SAVING = 0.10            # below this, a rewrite is not worth it for a low-severity task
CASCADE_SAVING = 0.20        # medium severity: B first only when it saves at least this much


def _pct(new: float, old: float) -> float:
    return (new - old) / old if old else 0.0


def run(s0: S0Result | None, s3: S3Result, s4: S4Result, s6: S6Result, s7: S7Result) -> Recommendation:
    volume = s4.cost.monthly_volume
    base = s6.baseline
    risk, diff = LEVEL[s3.risk.level], LEVEL[s3.difficulty.level]
    score = 2 * risk + diff
    severity = "high" if risk == 2 or score >= 4 else "medium" if score >= 2 else "low"
    factors = [f"risk {s3.risk.level}" + (f" ({', '.join(s3.risk.domains)})" if s3.risk.domains else ""),
               f"difficulty {s3.difficulty.level}", f"task {s3.task_type.type}"]

    def ok(c) -> bool:
        return c.locks_preserved and not c.requirements_missing and c.within_size_limit

    ok_a, ok_b = ok(s6.checks), ok(s7.checks)
    month = lambda per_call: round(per_call * volume, 2)                       # noqa: E731
    saving_b = -_pct(s7.cost.per_call, base.cost.per_call)
    docs = s0 is not None and s0.used

    # ── trade-offs, from the numbers ──
    o_pros = ["Exactly what was written: no rewrite risk"]
    o_cons = []
    if s4.removable_tokens:
        o_cons.append(f"{s4.removable_tokens} tokens of padding and repeats "
                      f"({s4.removable_tokens / max(s4.primary.count, 1):.0%} of the prompt) are paid on every call")
    if any(g.kind == "format" for g in s3.gaps):
        o_cons.append("No answer format: the model may add a preamble and run long")
    if docs:
        o_cons.append(f"Without retrieval all {s0.document_tokens:,} file tokens go with every call")
    if s3.gaps:
        o_cons.append(f"{len(s3.gaps)} gap(s) S3 found stay open (the model guesses)")

    da = _pct(s6.tokens, base.tokens)
    a_pros = ["Exact answer format, implied rules and a self-check: the most precise answer to expect",
              "Goal first, key rules repeated last; fixed instructions first so prompt caching can reuse them"]
    if any(s.key == "assumptions" for s in s6.sections):
        a_pros.append("S3's gaps are written down as assumptions you can edit")
    a_cons = []
    if da > 0:
        a_cons.append(f"{da:+.0%} input tokens vs the baseline ({s6.tokens:,} vs {base.tokens:,})")
    else:
        a_pros.append(f"Also {-da:.0%} fewer input tokens than the baseline")
    dm = month(s6.cost.per_call) - month(base.cost.per_call)
    (a_cons if dm > 0 else a_pros).append(
        f"${abs(dm):,.2f} {'more' if dm > 0 else 'less'} per month at {volume:,} calls")
    if not ok_a:
        a_cons.append("Failed a check: " + ", ".join(
            ([f"{len(s6.checks.missing_locks)} lock(s) lost"] if s6.checks.missing_locks else [])
            + ([f"missing {', '.join(s6.checks.requirements_missing)}"] if s6.checks.requirements_missing else [])
            + ([] if s6.checks.within_size_limit else ["over the size limit"])))

    fmt_name = "plain text" if s7.format == "plain" else "CSV"
    db = _pct(s7.best_tokens, base.tokens)
    b_pros, b_cons = [f"Sent as {fmt_name}, the shorter form: {s7.format_reason}"], []
    dropped = next((st for st in s7.steps if st.level == "L0" and st.status == "applied"), None)
    if dropped and ok_b:
        b_pros.append(f"Padding, filler and repeats gone ({dropped.reason.split(' part')[0].replace('dropped ', '')} "
                      f"part(s)) with every lock kept")
    if db < 0:
        b_pros.append(f"{-db:.0%} fewer input tokens ({s7.best_tokens:,} vs {base.tokens:,})")
    else:
        b_cons.append(f"{db:+.0%} input tokens vs the baseline ({s7.best_tokens:,} vs {base.tokens:,}): little padding to remove")
    mb = month(base.cost.per_call) - month(s7.cost.per_call)
    (b_pros if mb > 0 else b_cons).append(f"${abs(mb):,.2f} {'saved' if mb > 0 else 'more'} per month at {volume:,} calls")
    if any(st.level == "L5" and st.status == "applied" for st in s7.steps):
        b_pros.append(f"Answer capped at {s7.answer_budget} tokens (max_tokens {s7.max_tokens}): the answer side costs less too")
    b_cons.append("Least guidance: no self-check, assumptions left out, wording trimmed")
    if s7.reached != "L5":
        b_cons.append(f"Ladder stopped at {s7.reached} (max {s7.max_ladder_level}), so it saves less")
    if s7.format == "plain":
        b_cons.append("Plain text: the fields are not separable by a CSV reader (use the CSV form if a tool parses it)")
    if not ok_b:
        b_cons.append("Failed a check: locks or requirements were lost")

    options = [
        PromptOption(key="original", name=base.label, input_tokens=base.tokens, answer_tokens=base.cost.output_tokens_est,
                     per_call=base.cost.per_call, per_month=month(base.cost.per_call), checks_passed=True,
                     precision="as written", best_for="prompts that are already short and exact, or when a rewrite is not allowed",
                     pros=o_pros, cons=o_cons),
        PromptOption(key="A", name="Variant A · detailed", input_tokens=s6.tokens, answer_tokens=s6.answer_tokens_est,
                     per_call=s6.cost.per_call, per_month=month(s6.cost.per_call), checks_passed=ok_a,
                     precision="highest", best_for="high-stakes, complex or audited work where a wrong answer is expensive",
                     pros=a_pros, cons=a_cons),
        PromptOption(key="B", name=f"Variant B · short ({fmt_name})", input_tokens=s7.best_tokens, answer_tokens=s7.answer_tokens_est,
                     per_call=s7.cost.per_call, per_month=month(s7.cost.per_call), checks_passed=ok_b,
                     precision="lean", best_for="simple, repeated, high-volume calls where cost matters most",
                     pros=b_pros, cons=b_cons),
    ]

    # ── the choice ──
    fallback = None
    if severity == "high":
        choice = "A" if ok_a else "original"
        why = (f"High severity ({'; '.join(factors)}): precision matters more than cost, so Variant A"
               if ok_a else "High severity, but Variant A failed its checks: send the original")
    elif severity == "medium":
        if ok_b and saving_b >= CASCADE_SAVING:
            choice, fallback = "B", "A" if ok_a else "original"
            why = (f"Medium severity: Variant B saves {saving_b:.0%} per call; resend with "
                   f"{'Variant A' if fallback == 'A' else 'the original'} if its answer fails a quick check")
        else:
            choice = "A" if ok_a else "original"
            why = (f"Medium severity and Variant B saves only {saving_b:.0%}: the extra precision of Variant A is worth it"
                   if ok_a else "Medium severity, Variant A failed its checks: send the original")
    else:
        if ok_b and saving_b >= MIN_SAVING:
            choice, fallback = "B", "A" if ok_a else None
            why = f"Low severity ({'; '.join(factors)}): Variant B is {saving_b:.0%} cheaper per call with every lock kept"
        elif ok_b or ok_a:
            choice = "original"
            why = f"Low severity and Variant B saves only {saving_b:.0%}: the original is lean enough, no rewrite needed"
        else:
            choice = "original"
            why = "Both variants failed their checks: send the original"

    tips = []
    if s7.max_tokens:
        tips.append(f"Set the API's max_tokens to {s7.max_tokens if choice == 'B' else s6.max_tokens}.")
    if volume >= 10_000:
        tips.append("For non-interactive volume, the providers' batch APIs roughly halve the price.")
    if choice == "A":
        tips.append("Keep Variant A's fixed sections unchanged between calls so prompt caching applies.")
    return Recommendation(severity=severity, severity_score=score, severity_max=6, factors=factors, choice=choice,
                          fallback=fallback, reason=why, options=options, tips=tips,
                          note="Precision is the expected effect of each prompt's structure; answer quality is not measured.")
