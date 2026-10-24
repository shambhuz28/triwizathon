"""S4 · Count and profile.

Tokens: tiktoken (o200k_base, cl100k_base) and, when an Anthropic key is set, Claude's own
counter; otherwise the count is labelled approximate (±20%).

Information profile, word by word, from several complementary signals:
  • surprisal   −log2 p(word | left context) under GPT-2 small (Selective Context [P4])
  • entropy     the LM's predictive entropy just before the word (how open the slot was)
  • excess      surprisal − entropy: surprise beyond what the model itself expected, which
                discounts slots where any word would have been surprising
  • rarity      −log2 corpus frequency (wordfreq) — information independent of context
  • echo        rare word that the LM nevertheless predicts well: it is only predictable
                because it already appeared, a sign of repetition
  • keyphrase   salience from YAKE (statistical) and KeyBERT (embedding) extractors
They are z-scored within the prompt and mixed with weights from thresholds.yaml into one
0–1 informativeness score that drives the heat map, the "surprising words" list and the
low-information phrases (first in line for removal in S7).

Prompt quality (clarity, specificity, rules, structure, repetition) and plain warnings.
"""
from __future__ import annotations

import math
import os

import numpy as np
import regex as re
from wordfreq import zipf_frequency

from .. import resources, textutils
from ..config import model_config, prices, thresholds
from ..schemas import (CostPreview, Keyphrase, LabelTokens, LowInfoPhrase, PromptWarning, QualityScore, S1Result,
                       S2Result, S3Result, S4Result, SurprisingWord, TokenCount, WordInfo)

LN2 = math.log(2)
APPROX_MARGIN = 0.20


# ── Token counting ───────────────────────────────────────────────────────────
def _anthropic_count(text: str, model_id: str) -> int | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(timeout=5.0)
        resp = client.messages.count_tokens(model=model_id, messages=[{"role": "user", "content": text}])
        return int(resp.input_tokens)
    except Exception:
        return None


def count_tokens(text: str, model_id: str, model_cfg: dict) -> tuple[list[TokenCount], TokenCount]:
    counts = [TokenCount(tokenizer=name, count=len(resources.tiktoken_encoding(name).encode(text)),
                         exact=model_cfg.get("tokenizer") == name, margin=0.0 if model_cfg.get("tokenizer") == name else APPROX_MARGIN)
              for name in ("o200k_base", "cl100k_base")]
    if model_cfg.get("tokenizer") == "anthropic":
        exact = _anthropic_count(text, model_id)
        if exact is not None:
            primary = TokenCount(tokenizer=f"anthropic:{model_id}", count=exact, exact=True)
            counts.insert(0, primary)
            return counts, primary
    exact_match = next((c for c in counts if c.exact), None)
    return counts, exact_match or counts[0]


# ── LM surprisal / entropy per GPT-2 token ───────────────────────────────────
def lm_token_stats(text: str):
    """Returns (char_offsets, surprisal_bits, entropy_bits) for every GPT-2 token.
    Long texts use a sliding window so every token has at least half a window of context."""
    import torch

    tok, model = resources.lm()
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    n = len(ids)
    surprisal = np.zeros(n, dtype="float32")
    entropy = np.zeros(n, dtype="float32")
    filled = np.zeros(n, dtype=bool)
    window = model.config.n_positions - 1
    stride = window // 2
    start = 0
    while start < n:
        chunk = ids[start:start + window]
        inp = torch.tensor([[tok.bos_token_id] + chunk])
        with torch.inference_mode():
            logits = model(inp).logits[0, :-1].float()
        logp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(chunk)
        s = (-logp[torch.arange(len(chunk)), tgt] / LN2).numpy()
        h = (-(logp.exp() * logp).sum(-1) / LN2).numpy()
        for k in range(len(chunk)):
            if not filled[start + k]:
                surprisal[start + k], entropy[start + k], filled[start + k] = s[k], h[k], True
        if start + window >= n:
            break
        start += stride
    return offsets, surprisal, entropy


# ── Word-level profile ───────────────────────────────────────────────────────
def _z(x: np.ndarray) -> np.ndarray:
    if len(x) < 2 or x.std() < 1e-6:
        return np.zeros_like(x)
    return (x - x.mean()) / x.std()


def _keyphrases(text: str) -> list[Keyphrase]:
    import yake

    out: list[Keyphrase] = []
    try:
        yk = yake.KeywordExtractor(lan="en", n=3, top=20, dedupLim=0.8).extract_keywords(text)
        for rank, (kw, _) in enumerate(yk):
            out.append(Keyphrase(text=kw, score=round(1 - rank / max(len(yk), 1), 3), source="yake"))
    except Exception:
        pass
    try:
        kb = resources.keybert().extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words="english",
                                                  top_n=15, use_mmr=True, diversity=0.5)
        out += [Keyphrase(text=kw, score=round(float(s), 3), source="keybert") for kw, s in kb]
    except Exception:
        pass
    return out


def _filler_hits(text: str) -> list[tuple[int, int, str]]:
    hits = []
    for pat, why in textutils.FILLER_PHRASES:
        for m in pat.finditer(text):
            hits.append((m.start(), m.end(), why))
    return hits


def profile_words(text: str) -> tuple[list[WordInfo], list[Keyphrase], dict]:
    w = thresholds()["s4"]["weights"]
    doc = resources.parse(text)
    words = [t for t in doc if not t.is_space]
    if not words:
        return [], [], {}

    offsets, s_tok, h_tok = lm_token_stats(text)
    # Map each GPT-2 token to the spaCy word containing its last character.
    char_to_word = np.full(len(text) + 1, -1, dtype=int)
    for k, t in enumerate(words):
        char_to_word[t.idx:t.idx + len(t)] = k
    surprisal = np.zeros(len(words))
    entropy = np.full(len(words), np.nan)
    for (a, b), s, h in zip(offsets, s_tok, h_tok):
        if b <= a:
            continue
        k = char_to_word[b - 1]
        if k < 0:
            continue
        surprisal[k] += s
        if np.isnan(entropy[k]):
            entropy[k] = h          # uncertainty before the word started
    entropy = np.nan_to_num(entropy, nan=float(np.nanmean(entropy)) if np.any(~np.isnan(entropy)) else 0.0)
    excess = surprisal - entropy
    rarity = np.array([min(25.0, (9.0 - zipf_frequency(t.lower_, "en")) * math.log2(10)) if (t.is_alpha or t.like_num)
                       else 0.0 for t in words])

    # Echo: rare in general, yet predictable here, and seen earlier in the prompt.
    seen: set[str] = set()
    echo = np.zeros(len(words), dtype=bool)
    for k, t in enumerate(words):
        lem = t.lemma_.lower()
        if t.is_alpha and not t.is_stop and lem in seen and rarity[k] > 13 and surprisal[k] < 3.0:
            echo[k] = True
        seen.add(lem)

    keyphrases = _keyphrases(text)
    key = np.zeros(len(words))
    low = text.lower()
    for kp in keyphrases:
        for m in re.finditer(rf"(?<!\w){re.escape(kp.text.lower())}(?!\w)", low):
            for k in set(char_to_word[m.start():m.end()]) - {-1}:
                key[k] = max(key[k], kp.score)

    filler_hits = _filler_hits(text)
    filler = np.array([t.lower_ in textutils.FILLER_WORDS for t in words], dtype=bool)
    for a, b, _ in filler_hits:
        for k in set(char_to_word[a:b]) - {-1}:
            filler[k] = True

    contentful = np.array([not (t.is_punct or t.is_space) for t in words])
    mix = np.zeros(len(words))
    idx = np.where(contentful)[0]
    mix[idx] = (w["surprisal"] * _z(surprisal[idx]) + w["excess"] * _z(excess[idx]) + w["rarity"] * _z(rarity[idx])
                + w["keyphrase"] * key[idx] + w["echo"] * echo[idx])
    score = 1 / (1 + np.exp(-mix))
    for k, t in enumerate(words):
        if not contentful[k]:
            score[k] = 0.0
        elif filler[k]:
            score[k] = min(score[k], 0.15)
        elif t.is_stop:
            score[k] *= 0.6

    heat = [WordInfo(start=t.idx, end=t.idx + len(t), text=t.text, surprisal=round(float(surprisal[k]), 2),
                     entropy=round(float(entropy[k]), 2), excess=round(float(excess[k]), 2),
                     rarity=round(float(rarity[k]), 2), keyphrase=round(float(key[k]), 3), echo=bool(echo[k]),
                     filler=bool(filler[k]), score=round(float(score[k]), 3)) for k, t in enumerate(words)]
    extras = {"doc": doc, "words": words, "score": score, "surprisal": surprisal, "excess": excess,
              "rarity": rarity, "key": key, "filler_hits": filler_hits, "contentful": contentful}
    return heat, keyphrases, extras


def _is_list_marker(t) -> bool:
    nxt = t.nbor(1) if t.i + 1 < len(t.doc) else None
    prev_newline = t.i == 0 or "\n" in t.doc[t.i - 1].text_with_ws
    return t.like_num and prev_newline and nxt is not None and nxt.text in (".", ")")


def surprising_words(heat: list[WordInfo], extras: dict, k: int, label_at) -> list[SurprisingWord]:
    words, score = extras["words"], extras["score"]
    zs, zx, zr = _z(extras["surprisal"]), _z(extras["excess"]), _z(extras["rarity"])
    order = np.argsort(-score)
    out, seen = [], set()
    for i in order:
        t = words[i]
        if t.is_stop or not (t.is_alpha or t.like_num) or t.lower_ in seen or heat[i].filler or _is_list_marker(t):
            continue
        drivers = [name for name, v in (("surprising in context", zs[i]), ("beyond LM expectation", zx[i]),
                                        ("rare word", zr[i])) if v > 0.8]
        if extras["key"][i] > 0.5:
            drivers.append("keyphrase")
        drivers.append(f"in {label_at(t.idx)}")
        out.append(SurprisingWord(text=t.text, start=t.idx, end=t.idx + len(t), score=heat[i].score, drivers=drivers))
        seen.add(t.lower_)
        if len(out) >= k:
            break
    return out


def low_info_phrases(text: str, s2: S2Result, extras: dict, percentile: float) -> list[LowInfoPhrase]:
    """Lexical units (noun chunks, else single words) whose mean informativeness falls in the
    bottom percentile, plus filler-phrase hits. Data, examples and code are never candidates."""
    doc, words, score, surprisal = extras["doc"], extras["words"], extras["score"], extras["surprisal"]
    protected = [(s.content_start, s.content_end) for s in s2.segments
                 if s.label in ("data", "examples") or s.kind in ("code", "json", "table", "tagged")]

    def is_protected(a, b):
        return any(textutils.overlaps(a, b, x, y) for x, y in protected)

    word_index = {t.i: k for k, t in enumerate(words)}
    units: list[tuple[int, int, list[int]]] = []
    in_chunk: set[int] = set()
    for chunk in doc.noun_chunks:
        ks = [word_index[t.i] for t in chunk if t.i in word_index]
        if ks:
            units.append((chunk.start_char, chunk.end_char, ks))
            in_chunk.update(t.i for t in chunk)
    for t in words:
        if t.i not in in_chunk and not t.is_punct and not t.is_quote and t.text.strip():
            units.append((t.idx, t.idx + len(t), [word_index[t.i]]))
    units = [u for u in units if not is_protected(u[0], u[1])]
    if not units:
        return []
    unit_scores = np.array([score[u[2]].mean() for u in units])
    cut = np.percentile(unit_scores, percentile)

    out: list[LowInfoPhrase] = []
    for (a, b, ks), sc in zip(units, unit_scores):
        if sc <= cut and not (len(ks) == 1 and words[ks[0]].is_stop):   # a lone stopword says nothing useful
            out.append(LowInfoPhrase(start=a, end=b, text=text[a:b], score=round(float(sc), 3),
                                     info_bits=round(float(surprisal[ks].sum()), 2), reason="low informativeness"))
    for a, b, why in extras["filler_hits"]:
        if not is_protected(a, b):
            out.append(LowInfoPhrase(start=a, end=b, text=text[a:b], score=0.1, info_bits=0.0, reason=f"filler: {why}"))
    # merge overlapping / touching spans into readable phrases
    out.sort(key=lambda p: (p.start, -p.end))
    merged: list[LowInfoPhrase] = []
    for p in out:
        if merged and p.start <= merged[-1].end + 1 and not text[merged[-1].end:p.start].strip(" ") \
                and "\n" not in text[merged[-1].start:p.end]:
            m = merged[-1]
            reasons = m.reason if p.reason in m.reason else f"{m.reason}; {p.reason}"
            merged[-1] = LowInfoPhrase(start=m.start, end=max(m.end, p.end), text=text[m.start:max(m.end, p.end)],
                                       score=round(min(m.score, p.score), 3), info_bits=round(m.info_bits + p.info_bits, 2),
                                       reason=reasons)
        else:
            merged.append(p)
    return merged


# ── Quality score and warnings ───────────────────────────────────────────────
def assess_quality(s1: S1Result, s2: S2Result, s3: S3Result, total_tokens: int, removable: int,
                   by_label: dict[str, int]) -> tuple[QualityScore, list[PromptWarning]]:
    warns: list[PromptWarning] = []
    segs = s2.segments
    live = [s for s in segs if not s.remove]
    labels = {s.label for s in live}
    limits = thresholds()["limits"]

    def warn(code, sev, msg, hint, spans=()):
        warns.append(PromptWarning(code=code, severity=sev, message=msg, hint=hint, spans=list(spans)))

    if "task" not in labels and "question" not in labels:
        warn("no_task", "error", "No clear task or question found.", "Start with one sentence that says what to do.")
    if "format" not in labels:
        warn("no_format", "warn", "No answer format given.", "Say exactly what the answer should look like (e.g. a table with named columns).")
    if any(g.kind == "length" for g in s3.gaps):
        warn("no_length", "info", "No answer length given.", f"Cap the answer, e.g. about {s3.answer_length.budget} tokens.")
    vague = [s for s in live if s.label in ("task", "rules", "format") and textutils.VAGUE_RE.search(s.text)]
    if vague:
        warn("vague_wording", "warn", f"{len(vague)} instruction(s) use vague wording.",
             "Replace 'or something', 'etc.', 'kind of' with the exact requirement.",
             [(s.content_start, s.content_end) for s in vague])
    if s2.duplicates:
        dropped = [s for s in segs if s.duplicate_of]
        warn("duplicate_rules", "warn", f"{len(dropped)} instruction(s) repeat an earlier one.",
             "State each rule once; repetition costs tokens without adding meaning.",
             [(s.content_start, s.content_end) for s in dropped])
    pad = by_label.get("padding", 0)
    if total_tokens and pad / total_tokens > 0.10:
        warn("padding", "warn", f"Greetings and pleasantries use {pad} tokens ({pad / total_tokens:.0%}).",
             "Drop greetings and thank-yous; the model doesn't need them.",
             [(s.content_start, s.content_end) for s in segs if s.label == "padding"])
    data_segs = [s for s in segs if s.label == "data" and s.kind in ("sentence", "list_item")]
    fenced = any(b.kind in ("tagged", "code") for b in s2.blocks)
    if data_segs and not fenced:
        warn("unfenced_data", "info", "Data is not fenced off from the instructions.",
             "Wrap pasted data in tags like <reviews>…</reviews> so it can't be read as orders.")
    rule_segs = [s for s in live if s.label == "rules"]
    if rule_segs and total_tokens > 300:
        n = len(s1.clean_text)
        middle = [s for s in rule_segs if 0.3 * n < s.content_start < 0.7 * n]
        if len(middle) == len(rule_segs):
            warn("rules_buried", "info", "All rules sit in the middle of a long prompt.",
                 "Put the goal first and repeat the key rules at the end.", [(s.content_start, s.content_end) for s in middle])
    if s1.injections:
        warn("injection", "error", f"{len(s1.injections)} passage(s) try to give the model new orders.",
             "They will be fenced as data; check the source of the pasted text.",
             [(i.start, i.end) for i in s1.injections])
    if s1.pii and not s1.pii_redacted:
        warn("pii", "warn", f"{len(s1.pii)} piece(s) of personal information found.",
             "Turn on redaction to swap them for placeholders.", [(p.start, p.end) for p in s1.pii])
    if total_tokens < limits["min_prompt_tokens"]:
        warn("too_short", "info", "Prompt is very short; there is little to optimise.", "Add the missing details instead.")
    if total_tokens > limits["max_prompt_tokens"]:
        warn("too_long", "error", f"Prompt exceeds {limits['max_prompt_tokens']} tokens.", "Split it before optimising.")
    if not s1.language.is_english and s1.language.code != "und":
        warn("non_english", "info", f"Detected {s1.language.name}; name detection is English-only.",
             "Results for names and rules may be less reliable.")
    if s3.risk.level == "high":
        warn("high_risk", "info", f"High-risk task ({', '.join(s3.risk.domains) or 'numeric rules'}).",
             "Variant B will stop at the safe reduction steps (L2).")

    # Component scores, 0–100.
    vague_n = len(vague)
    clarity = 100 - (10 if s1.pii and not s1.pii_redacted else 0) - (35 if "task" not in labels and "question" not in labels else 0) - 12 * vague_n \
        - int(60 * (pad / total_tokens if total_tokens else 0))
    specificity = 25 + (30 if "format" in labels else 0) + (20 if not any(g.kind == "length" for g in s3.gaps) else 0) \
        + (10 if not any(g.kind == "audience" for g in s3.gaps) else 0) + (15 if "role" in labels else 0)
    rules = min(100, 40 + 20 * len(rule_segs)) if rule_segs else 35
    structure = 40 - 30 * bool(s1.injections) + (25 if fenced or not data_segs else 0) + (15 if any(s.kind == "header" for s in segs) else 0) \
        + (20 if live and live[0].label in ("role", "task", "question") else 0)
    repetition = 100 - 25 * len(s2.duplicates) - int(100 * removable / total_tokens * 0.5 if total_tokens else 0)
    comps = [max(0, min(100, int(c))) for c in (clarity, specificity, rules, structure, repetition)]
    overall = int(round(np.average(comps, weights=[0.25, 0.25, 0.15, 0.2, 0.15])))
    return QualityScore(overall=overall, clarity=comps[0], specificity=comps[1], rules=comps[2],
                        structure=comps[3], repetition=comps[4]), warns


# ── Cost preview (full pricing arrives in S8) ────────────────────────────────
def cost_preview(model_id: str, cfg: dict, in_tokens: int, out_tokens: int, volume: int) -> CostPreview:
    p_in, p_out = cfg["input"], cfg["output"]
    lc = cfg.get("long_context")
    if lc and in_tokens > lc["above_tokens"]:
        p_in, p_out = lc["input"], lc["output"]
    c_in, c_out = in_tokens * p_in / 1e6, out_tokens * p_out / 1e6
    per_call = c_in + c_out
    return CostPreview(model=model_id, model_label=cfg["label"], input_tokens=in_tokens, output_tokens_est=out_tokens,
                       price_input=p_in, price_output=p_out, per_call=round(per_call, 6), per_10k=round(per_call * 1e4, 4),
                       per_month=round(per_call * volume, 2), monthly_volume=volume,
                       answer_seconds=round(out_tokens / cfg.get("tokens_per_second", 42), 2),
                       output_share_of_cost=round(c_out / per_call, 3) if per_call else 0.0,
                       last_checked=str(prices()["last_checked"]))


# ── Stage entry point ────────────────────────────────────────────────────────
def run(s1: S1Result, s2: S2Result, s3: S3Result, target_model: str | None = None,
        monthly_volume: int = 1_000_000) -> S4Result:
    cfg_s4 = thresholds()["s4"]
    text = s1.clean_text
    model_id, mcfg = model_config(target_model)
    counts, primary = count_tokens(text, model_id, mcfg)

    enc = resources.tiktoken_encoding("o200k_base")
    seg_tokens = {s.id: len(enc.encode(s.text)) for s in s2.segments}
    by_label_raw: dict[str, int] = {}
    for s in s2.segments:
        by_label_raw[s.label] = by_label_raw.get(s.label, 0) + seg_tokens[s.id]
    total_seg = sum(seg_tokens.values()) or 1
    by_label = [LabelTokens(label=l, tokens=t, share=round(t / total_seg, 3))
                for l, t in sorted(by_label_raw.items(), key=lambda kv: -kv[1])]
    removable = sum(seg_tokens[s.id] for s in s2.segments if s.remove)

    def label_at(pos: int) -> str:
        for s in s2.segments:
            if s.start <= pos < s.end:
                return s.label
        return "text"

    heat, keyphrases, extras = profile_words(text)
    surprising = surprising_words(heat, extras, cfg_s4["surprising_top_k"], label_at) if heat else []
    low_info = low_info_phrases(text, s2, extras, cfg_s4["low_info_percentile"]) if heat else []
    quality, warns = assess_quality(s1, s2, s3, primary.count, removable, by_label_raw)
    cost = cost_preview(model_id, mcfg, primary.count, s3.answer_length.expected_tokens, monthly_volume)

    return S4Result(tokens=counts, primary=primary, by_label=by_label, segment_tokens=seg_tokens,
                    removable_tokens=removable, heatmap=heat, surprising=surprising, low_info=low_info,
                    keyphrases=keyphrases, quality=quality, warnings=warns, cost=cost,
                    lm=resources.LM_MODEL)
