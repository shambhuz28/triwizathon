"""End-to-end checks for S1–S7 on the PRD worked example and a messy code prompt."""
import re
from pathlib import Path

import pytest

from optimus.pipeline import analyze_sync
from optimus.stages import s6_variant_a

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def worked():
    return analyze_sync((HERE / "worked_example.txt").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def messy():
    return analyze_sync((HERE / "messy_example.txt").read_text(encoding="utf-8"), redact_pii=True)


def test_parts_rebuild_input_exactly(worked, messy):
    for r in (worked, messy):
        assert r.s2.reconstructs
        assert "".join(r.s1.clean_text[s.start:s.end] for s in r.s2.segments) == r.s1.clean_text


def test_worked_example_token_count_matches_prd(worked):
    assert worked.s4.primary.count == 184           # PRD Appendix A, o200k tokenizer


def test_padding_and_repeat_marked_for_removal(worked):
    labels = [s.label for s in worked.s2.segments]
    assert sum(1 for s in worked.s2.segments if s.label == "padding" and s.kind != "filler") == 4
    assert len(worked.s2.duplicates) == 1
    assert labels.count("data") == 4                # header + three reviews


def test_filler_carved_off_sentence_edges(worked):
    segs = worked.s2.segments
    fillers = {s.text for s in segs if s.kind == "filler"}
    assert {"Basically", "And please", "or something like that so it's easy to read."} <= fillers
    assert all(s.label == "padding" and s.remove for s in segs if s.kind == "filler")
    rule = next(s for s in segs if s.text.startswith("don't make up"))
    assert rule.label == "rules" and not rule.remove


def test_chatty_prompt_padding():
    from optimus.stages import s1_clean, s2_parse
    text = ("hey, hope you're doing good. i need some help figuring out what people are saying lol.\n\n"
            "oh and please dont make anything up ok?\n\nhere are the feedbacks:\n\n1. \"Late and cold.\"\n"
            "2. \"Great food.\"\n\nthanks a lot, really appreciate it!")
    segs = s2_parse.run(s1_clean.run(text)).segments
    by_text = {s.text: s for s in segs}
    assert by_text["hey, hope you're doing good."].label == "padding"
    assert by_text["lol."].kind == "filler" and by_text["ok?"].kind == "filler"
    assert by_text["dont make anything up"].label == "rules"
    assert by_text["here are the feedbacks:"].label == "data" and not by_text["here are the feedbacks:"].remove


def test_task_understood(worked):
    assert worked.s3.task_type.type == "classify"
    assert worked.s3.items == 3
    assert {g.kind for g in worked.s3.gaps} >= {"ambiguity", "scope"}


def test_key_words_locked(worked):
    s5 = worked.s5
    assert [ls.items for ls in s5.label_sets] == [["positive", "negative", "neutral"]]
    assert sum(1 for s in s5.spans if s.category == "data") == 3
    assert any("make up" in s.text for s in s5.spans if s.category == "restriction")
    assert any(s.text == "40 minutes" for s in s5.spans)


def test_surprisal_profile(worked):
    assert worked.s4.heatmap and all(0 <= w.score <= 1 for w in worked.s4.heatmap)
    top = [w.text.lower() for w in worked.s4.surprising[:5]]
    assert "biryani" in top
    low = " ".join(p.text.lower() for p in worked.s4.low_info)
    assert "thanks a lot" in low and "i hope you are doing well" in low


def test_pii_redacted_and_injection_fenced(messy):
    assert {p.type for p in messy.s1.pii} >= {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}
    assert "rahul" not in messy.s1.clean_text.lower()
    assert len(messy.s1.injections) == 1
    inj = messy.s1.injections[0]
    seg = next(s for s in messy.s2.segments if s.content_start <= inj.start < s.content_end)
    assert seg.label == "data"


def test_code_kept_whole_and_placeholders_locked(messy):
    code = [b for b in messy.s2.blocks if b.kind == "code"]
    assert len(code) == 1 and messy.s1.clean_text[code[0].start:code[0].end].count("\n") >= 3
    texts = {s.text for s in messy.s5.spans}
    assert {"{{invoice_id}}", "v2.4.1", "at most 3", "`parse_invoice()`"} <= texts


# ── S3/S4 read S2's parts ────────────────────────────────────────────────────
def test_s3_reads_s2_parts(worked):
    by_id = {s.id: s for s in worked.s2.segments}
    assert worked.s3.task_segments and all(by_id[i].label in ("task", "question") for i in worked.s3.task_segments)
    assert 0 < worked.s3.live_tokens < worked.s4.primary.count     # padding and repeats are not counted
    vague = next(g for g in worked.s3.gaps if g.kind == "ambiguity")
    assert any(by_id[i].label == "format" for i in vague.segment_ids)


def test_s4_profiles_s2_parts(worked):
    assert set(worked.s4.segment_info) == {s.id for s in worked.s2.segments}
    pad = [s for s in worked.s2.segments if s.label == "padding"]
    assert all(worked.s4.segment_info[s.id] <= 0.15 for s in pad)
    assert any("S2: repeats" in p.reason for p in worked.s4.low_info)


# ── S5 compression ───────────────────────────────────────────────────────────
def test_compression_keeps_every_lock(worked, messy):
    for r in (worked, messy):
        c = r.s5.compression
        assert c.locks_preserved and not c.missing_locks
        assert all(t in c.prompt for t in r.s5.lock_set)
        assert c.compressed_tokens < c.original_tokens


def test_compression_follows_earlier_stages(worked):
    s5 = worked.s5
    assert s5.compression.model == "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    parts = {p.id: p for p in s5.segments}
    for seg in worked.s2.segments:
        p = parts[seg.id]
        if seg.remove:
            assert p.action == "drop" and p.compressed == ""            # S2 padding / repeats
        if seg.label == "data":
            assert p.action == "keep" and p.compressed == seg.text      # data is never touched
    prompt = s5.compression.prompt
    assert "a bunch of" not in prompt and "make sure you" not in prompt and "Thanks" not in prompt
    assert "Give it to me in a table." in prompt                         # dropped tail filler leaves its "."
    assert s5.compression.saved_tokens >= 60
    assert s5.handoff.max_ladder_level == "L5" and s5.handoff.sections["data"]
    assert all(a.placeholder.startswith("{{ASSUMPTION:") for a in s5.handoff.assumptions)


def test_high_risk_task_keeps_every_word(messy):
    assert messy.s3.risk.level == "high" and messy.s5.handoff.max_ladder_level == "L2"
    for p in messy.s5.segments:
        assert p.action in ("keep", "drop")
        if p.action == "keep":
            assert p.compressed == p.text
    assert "Thanks" not in messy.s5.compression.prompt


def test_compression_without_llmlingua(monkeypatch, worked):
    from optimus import resources
    from optimus.stages import s5_lock
    monkeypatch.setattr(resources, "llmlingua", lambda: None)
    s5 = s5_lock.run(worked.s1, worked.s2, worked.s3, worked.s4)
    assert s5.compression.model is None and "S4" in s5.compression.method
    assert s5.compression.locks_preserved and s5.compression.compressed_tokens < s5.compression.original_tokens


def test_chatty_prompt_compression():
    text = ("hey, hope you're doing good. i need some help figuring out what people are saying lol.\n\n"
            "oh and please dont make anything up ok?\n\nhere are the feedbacks:\n\n1. \"Late and cold.\"\n"
            "2. \"Great food.\"\n\nthanks a lot, really appreciate it!")
    prompt = analyze_sync(text).s5.compression.prompt
    assert "dont make anything up." in prompt                    # "ok?" goes, the rule is not turned into a question
    assert not re.search(r"\b(?:hey|lol|thanks)\b", prompt)
    assert '1. "Late and cold."' in prompt


def test_variant_a_is_sectioned_and_keeps_everything(worked, messy):
    for r in (worked, messy):
        a = r.s6
        assert a.checks.locks_preserved and not a.checks.requirements_missing
        assert a.tokens <= a.size_limit
        assert [s.key for s in a.sections] == [k for k in s6_variant_a.ORDER if k in {s.key for s in a.sections}]
    keys = {s.key for s in worked.s6.sections}
    assert {"goal", "rules", "format", "inputs", "self_check"} <= keys and "role" not in keys
    p = worked.s6.prompt
    assert "<reviews>" in p and '1. "Food arrived 40 minutes late and cold."' in p
    assert "| # | Label | Main complaint |" in p and "[assumption" in p
    assert not re.search(r"\b(?:Hi|Thanks|please)\b", p)
    assert "ignore any instruction" in messy.s6.prompt              # the planted order in the data is fenced off


def test_variant_b_climbs_the_ladder_and_ships_csv(worked, messy):
    import csv
    import io
    b = worked.s7
    assert b.reached == "L5" and b.checks.locks_preserved and not b.checks.requirements_missing
    assert b.csv_tokens < worked.s6.baseline.tokens and b.tokens < worked.s5.compression.original_tokens
    rows = list(csv.reader(io.StringIO(b.csv)))
    assert rows[0] == ["field", "value"] and ["limit", f"Answer only, no preamble, under {b.answer_budget} tokens."] in rows
    assert [r[1] for r in rows if r[0] == "reviews"][0] == '1. "Food arrived 40 minutes late and cold."'
    m = messy.s7                                                   # high risk: stops at L2, no answer cap
    assert m.max_ladder_level == "L2" and m.reached in ("L0", "L1", "L2")
    assert {s.level: s.status for s in m.steps}["L3"] == "skipped" and m.checks.locks_preserved


def test_recommendation_follows_severity(worked, messy):
    low, high = worked.recommendation, messy.recommendation
    assert low.severity == "low" and low.choice == "B" and low.fallback == "A"
    assert high.severity == "high" and high.choice == "A"                 # financial + code: precision first
    for rec in (low, high):
        assert [o.key for o in rec.options] == ["original", "A", "B"]
        assert all(o.pros and o.cons for o in rec.options)              # every choice states its trade-off
    b = next(o for o in low.options if o.key == "B")
    assert b.per_month < next(o for o in low.options if o.key == "original").per_month


def test_variant_b_is_sent_in_the_shorter_form(worked, messy):
    for r in (worked, messy):
        b = r.s7
        assert b.best_tokens == min(b.tokens, b.csv_tokens)
        assert b.format == ("plain" if b.tokens < b.csv_tokens else "csv") and b.format_reason
        opt = next(o for o in r.recommendation.options if o.key == "B")
        assert opt.input_tokens == b.best_tokens and ("plain text" if b.format == "plain" else "CSV") in opt.name


def test_live_runs_report_the_providers_own_usage(monkeypatch):
    from optimus import live
    monkeypatch.setattr(live, "gemini_key", lambda: "test-key")
    monkeypatch.setattr(live, "ollama_model", lambda: "llama3.2:latest")
    monkeypatch.setattr(live, "_gemini_chat_models", lambda: ("gemini-2.5-flash",))
    monkeypatch.setattr(live, "_exhausted", {})
    replies = {
        "generateContent": {"candidates": [{"content": {"parts": [{"text": "Negative"}]}}],
                            "usageMetadata": {"promptTokenCount": 21, "candidatesTokenCount": 1,
                                              "thoughtsTokenCount": 172, "totalTokenCount": 194}},
        "/api/chat": {"message": {"content": "Negative"}, "prompt_eval_count": 45, "eval_count": 3},
    }
    sent = []

    def fake_post(url, body, headers=None, timeout=None):
        sent.append((url, headers or {}))
        return next(v for k, v in replies.items() if k in url)
    monkeypatch.setattr(live, "_post", fake_post)
    g, o = live.run("gemini", "Classify."), live.run("ollama", "Classify.")
    assert (g["input_tokens"], g["output_tokens"], g["thinking_tokens"], g["total_tokens"]) == (21, 1, 172, 194)
    assert (o["input_tokens"], o["output_tokens"], o["total_tokens"]) == (45, 3, 48) and o["reply"] == "Negative"
    assert "test-key" not in sent[0][0] and sent[0][1]["x-goog-api-key"] == "test-key"   # key in a header, not the URL


def test_gemini_moves_to_the_next_model_when_quota_runs_out(monkeypatch):
    import io
    import json
    import urllib.error
    from optimus import live
    monkeypatch.setattr(live, "gemini_key", lambda: "test-key")
    monkeypatch.setattr(live, "_gemini_chat_models", lambda: ("gemini-2.5-flash", "gemini-old-lite", "gemini-3.8-flash"))
    monkeypatch.setattr(live, "_exhausted", {})
    quota = {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
                                   {"retryDelay": "269s"}]}}
    calls = []

    def fake_post(url, body, headers=None, timeout=None):
        calls.append(url)
        if "gemini-2.5-flash" in url:
            raise urllib.error.HTTPError(url, 429, "quota", {}, io.BytesIO(json.dumps(quota).encode()))
        if "gemini-old-lite" in url:
            raise urllib.error.HTTPError(url, 404, "gone", {}, io.BytesIO(b"{}"))
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {"promptTokenCount": 3}}
    monkeypatch.setattr(live, "_post", fake_post)
    r = live.run("gemini", "hi")
    assert r["model"] == "gemini-3.8-flash" and r["fallback_from"] == ["gemini-2.5-flash", "gemini-old-lite"]
    calls.clear()
    assert live.run("gemini", "hi", model=r["model"])["model"] == "gemini-3.8-flash"   # pinned, no retry of the 429'd one
    assert len(calls) == 1


def test_replies_are_compared_with_the_original():
    from optimus import live
    out = live.compare({"original": "Sure! Here is the table:\n| # | Label |\n|---|---|\n| 1 | negative |",
                        "B_plain": "| # | Label |\n|---|---|\n| 1 | negative |"},
                       "original", ["positive", "negative", "neutral"], ["table"], budget=114)["replies"]
    assert out["original"]["preamble"] and not out["B_plain"]["preamble"]
    assert out["B_plain"]["format_ok"]["table"] and out["B_plain"]["similarity"] > 0.8
    assert out["B_plain"]["labels_found"] == ["negative"] and not out["B_plain"]["over_budget"]


def test_market_routes_each_model_to_its_official_counter(monkeypatch):
    from optimus import market
    models = [
        {"id": "openai/gpt-4o", "vendor": "openai", "family": "GPT", "hf": None},
        {"id": "anthropic/claude-x", "vendor": "anthropic", "family": "Claude", "hf": None},
        {"id": "qwen/qwen-new", "vendor": "qwen", "family": "Qwen3", "hf": "Qwen/Qwen-New"},
        {"id": "qwen/qwen-old", "vendor": "qwen", "family": "Qwen3", "hf": "Qwen/Qwen-Old"},
        {"id": "acme/closed", "vendor": "acme", "family": "Other", "hf": None},
    ]
    plan = market._plan(models)
    assert plan["openai/gpt-4o"][0] == "api:openai/gpt-4o"
    assert plan["qwen/qwen-old"][0] == "hf:Qwen/Qwen-New"                 # one family, one tokenizer download
    assert plan["acme/closed"] == ("none", None)
    fn, how = plan["openai/gpt-4o"][1]()
    assert fn("Classify each review.") > 0 and "tiktoken" in how
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(market.Unavailable):                             # no key: no number, never a guess
        plan["anthropic/claude-x"][1]()
