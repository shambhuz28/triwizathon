"""End-to-end checks for S1–S5 on the PRD worked example and a messy code prompt."""
from pathlib import Path

import pytest

from optimus.pipeline import analyze_sync

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def worked():
    return analyze_sync((HERE / "worked_example.txt").read_text())


@pytest.fixture(scope="module")
def messy():
    return analyze_sync((HERE / "messy_example.txt").read_text(), redact_pii=True)


def test_parts_rebuild_input_exactly(worked, messy):
    for r in (worked, messy):
        assert r.s2.reconstructs
        assert "".join(r.s1.clean_text[s.start:s.end] for s in r.s2.segments) == r.s1.clean_text


def test_worked_example_token_count_matches_prd(worked):
    assert worked.s4.primary.count == 184           # PRD Appendix A, o200k tokenizer


def test_padding_and_repeat_marked_for_removal(worked):
    labels = [s.label for s in worked.s2.segments]
    assert labels.count("padding") == 4
    assert len(worked.s2.duplicates) == 1
    assert labels.count("data") == 4                # header + three reviews


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
