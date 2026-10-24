"""End-to-end checks for S1–S5 on the PRD worked example and a messy code prompt."""
import re
from pathlib import Path

import pytest

from optimus.pipeline import analyze_sync

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
