"""S0 · Retrieve from documents.

Runs before S1 when files are attached. The prompt (whole, and each of its sentences) is
embedded with the MiniLM model the later stages use and searched against every attached file's
FAISS index; a chunk's score is its best cosine to any of those queries. In "auto" mode the files
are used only when the prompt is about them: a chunk reaches s0.min_score, or the prompt points
at a file ("the attached report", "according to the pdf", its name). "always" skips that check
and "off" skips S0.

Queries are the prompt's own sentences without greetings, thanks and polite wrappers. Ranking is
hybrid: (1 − w) × dense cosine + w × BM25 over the prompt's words (exact names, codes and rare
terms that embeddings blur), then a cross-encoder (ms-marco MiniLM) reads the prompt and each of
the best rerank_k chunks together and decides relevance (≥ rerank_min and ≥ rerank_relative of the
best). Each sub-question's own best chunk is kept when it reaches s0.min_query_score, and a chunk
that repeats one already picked (cosine ≥ duplicate_similarity, not its overlapping neighbour) is
skipped. Without the reranker, chunks within s0.relative_drop of the top cosine are kept, at most top_k.
Whether the files are used at all is still decided by the dense cosine alone.
A prompt that points at a file without any chunk reaching min_score ("summarise the slides")
asks about the file as a whole, so that file is taken from the start instead. Either way the
text stays within max_context_tokens, overlapping chunks are merged, and the passages are put
back in reading order and placed in front of the prompt as one <documents> block. S2 keeps that block whole as data, S5 locks it, S1 checks it for personal
info and planted instructions, and S3 counts it towards an answer-from-documents task.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import regex as re

from .. import documents, resources, textutils
from ..config import thresholds
from ..schemas import ChunkScore, RagMode, RetrievedPassage, S0Result

CONTEXT_TAG = "documents"

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")
# "..., and what are the risks?": a second question inside one sentence becomes its own query
_CLAUSE_RE = re.compile(r"\s*;\s*|,?\s+(?:and|but|also|plus)\s+(?=(?:what|how|which|who|when|where|why|whether|is|are"
                        r"|does|do|did|can|could|should|list|give|tell|show|explain|summari[sz]e|describe)\b)", re.I)
_DOC_NOUN = (r"(?:documents?|docs?|files?|pdfs?|reports?|papers?|articles?|chapters?|slides?|decks?|presentations?"
             r"|ppts?|spreadsheets?|sheets?|workbooks?|csvs?|excel|manuals?|contracts?|policy|policies|notes"
             r"|attachments?|uploads?|sources?|passages?)")
_DOC_REF_RE = re.compile(
    rf"\b(?:the|this|that|these|those|my|our|attached|uploaded|provided|given|shared)\s+(?:[\w-]+\s+){{0,2}}?{_DOC_NOUN}\b"
    r"|\b(?:according to|based on|refer(?:ring)? to|in|from)\s+(?:the\s+)?(?:attached|uploaded|provided|shared)\b"
    r"|\battach(?:ed|ment)\b|\buploaded\b", re.I)


def _strip_filler(sentence: str) -> str:
    """The sentence without greetings, thanks and polite wrappers, which only add noise to a search."""
    if textutils.GREETING_RE.search(sentence) and len(sentence.split()) <= 3:
        return ""
    for pat, _ in textutils.FILLER_PHRASES:
        sentence = pat.sub(" ", sentence)
    sentence = textutils.THANKS_RE.sub(" ", textutils.PLEASANTRY_RE.sub(" ", sentence))
    return " ".join(sentence.split()).strip(" ,;")


def _queries(prompt: str, limit: int) -> list[str]:
    parts = []
    for s in _SENTENCE_RE.findall(prompt):
        s = _strip_filler(s)
        clauses = _CLAUSE_RE.split(s)
        parts += [s, *clauses] if len(clauses) > 1 else [s]
    parts = [p.strip() for p in parts if len(p.split()) >= 3]
    whole = " ".join(p for p in (_strip_filler(s) for s in _SENTENCE_RE.findall(prompt)) if p) or " ".join(prompt.split())
    return list(dict.fromkeys([whole, *parts[:limit]]))


_TERM_RE = re.compile(r"[\p{L}\p{N}][\p{L}\p{N}_'-]*")


def _terms(text: str) -> list[str]:
    stop = resources.nlp().Defaults.stop_words
    return [t for t in (m.group().lower().strip("'-") for m in _TERM_RE.finditer(text)) if t and t not in stop]


def _bm25(docs: list[documents.Document], query: str, k1: float, b: float) -> dict[tuple[int, int], float]:
    """BM25 of every chunk of the attached files against the query's words, scaled so the best
    chunk scores 1. Catches exact names, codes and rare terms that embeddings blur."""
    q = set(_terms(query))
    if not q:
        return {}
    chunk_terms = {(d, c): _terms(doc.chunk_text(c)) for d, doc in enumerate(docs) for c in range(len(doc.chunks))}
    n = len(chunk_terms)
    avg = sum(map(len, chunk_terms.values())) / max(n, 1)
    df = {t: sum(1 for ts in chunk_terms.values() if t in ts) for t in q}
    raw = {}
    for key, ts in chunk_terms.items():
        score = 0.0
        for t in q:
            tf = ts.count(t)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(ts) / max(avg, 1)))
        if score:
            raw[key] = score
    top = max(raw.values(), default=0.0)
    return {k: v / top for k, v in raw.items()} if top else {}


def _rerank(queries: list[str], docs: list[documents.Document], keys: list[tuple[int, int]]) -> dict | None:
    """Cross-encoder relevance (0..1) of each chunk to the prompt or one of its sub-questions."""
    model = resources.reranker()
    if model is None or not keys:
        return None
    qs = queries[:5]
    pairs = [(q, " ".join(docs[d].chunk_text(c).split())) for d, c in keys for q in qs]
    logits = np.asarray(model.predict(pairs, batch_size=32), dtype="float64").reshape(len(keys), len(qs))
    probs = 1 / (1 + np.exp(-logits))
    return {key: float(p) for key, p in zip(keys, probs.max(axis=1))}


def _explicit_reference(prompt: str, docs: list[documents.Document]) -> str | None:
    if m := _DOC_REF_RE.search(prompt):
        return m.group(0)
    flat = re.sub(r"[\W_]+", " ", prompt.lower())
    for d in docs:
        stem = re.sub(r"[\W_]+", " ", Path(d.info.filename).stem.lower()).strip()
        if len(stem) >= 4 and re.search(rf"\b{re.escape(stem)}\b", flat):
            return d.info.filename
    return None


_NOUN_KINDS = {
    "pdf": {"pdf"}, "slide": {"pptx"}, "deck": {"pptx"}, "presentation": {"pptx"}, "ppt": {"pptx"},
    "spreadsheet": {"xlsx", "xlsm", "csv", "tsv"}, "sheet": {"xlsx", "xlsm", "csv", "tsv"}, "workbook": {"xlsx", "xlsm"},
    "excel": {"xlsx", "xlsm"}, "csv": {"csv", "tsv"}, "notes": {"md", "txt"},
}
_GENERIC_NOUNS = {"attached", "uploaded", "attachment", "upload", "file", "document", "doc", "source"}


def _referenced_docs(explicit: str, docs: list[documents.Document]) -> list[int]:
    """Files a generic reference points at: the one named; for a plural ("the slides") or a bare
    "attached", every file of that type, else all of them; otherwise the single file whose
    name/opening, then type, fit the noun (latest upload wins ties)."""
    named = [d for d, doc in enumerate(docs) if doc.info.filename == explicit]
    if named or len(docs) == 1:
        return named or [0]
    noun = explicit.lower().split()[-1]
    singular = (noun if not noun.endswith("s") or noun == "notes" else
                noun[:-3] + "y" if noun.endswith("ies") else noun[:-1])
    if singular != noun or noun in ("attached", "uploaded"):
        typed = [d for d, doc in enumerate(docs) if doc.info.kind in _NOUN_KINDS.get(singular, ())]
        return typed or list(range(len(docs)))

    def fit(d: int) -> tuple:
        doc = docs[d]
        head = f"{Path(doc.info.filename).stem.replace('_', ' ')} {doc.chunk_text(0)[:300]}".lower()
        named_here = singular not in _GENERIC_NOUNS and re.search(rf"\b{re.escape(singular)}", head) is not None
        return named_here, doc.info.kind in _NOUN_KINDS.get(singular, ()), doc.info.created

    return [max(range(len(docs)), key=fit)]


def _escape(text: str) -> str:
    return re.sub(rf"(?i)<(/?\s*){CONTEXT_TAG}", rf"<\1{CONTEXT_TAG[0]} {CONTEXT_TAG[1:]}", text)


def _merge(picked: list[tuple[int, int, float]], docs: list[documents.Document]) -> list[RetrievedPassage]:
    """Join chunks of the same unit whose ranges overlap or touch; reading order across files
    follows the order the files were attached in."""
    enc = resources.tiktoken_encoding("o200k_base")
    spans: dict[tuple[int, int], list[list]] = {}
    for d, c, score in picked:
        u, s, e = docs[d].chunks[c]
        spans.setdefault((d, u), []).append([s, e, score, [c]])
    out = []
    for (d, u), items in sorted(spans.items()):
        items.sort()
        merged = [items[0]]
        for s, e, score, cs in items[1:]:
            last = merged[-1]
            if s <= last[1] + 1:
                last[1], last[2], last[3] = max(last[1], e), max(last[2], score), last[3] + cs
            else:
                merged.append([s, e, score, cs])
        doc, unit = docs[d], docs[d].units[u]
        for s, e, score, cs in merged:
            text = unit.text[s:e].strip()
            out.append(RetrievedPassage(doc_id=doc.info.id, filename=doc.info.filename, location=unit.location,
                                        chunks=sorted(cs), score=round(score, 3), text=text,
                                        tokens=len(enc.encode(text))))
    return out


def build_context(passages: list[RetrievedPassage]) -> str:
    parts = [f"[{n}] {p.filename}{f' ({p.location})' if p.location else ''}\n{_escape(p.text)}"
             for n, p in enumerate(passages, 1)]
    return f"<{CONTEXT_TAG}>\n" + "\n\n".join(parts) + f"\n</{CONTEXT_TAG}>"


def augment(prompt: str, s0: S0Result | None) -> str:
    """The prompt S1 receives: the retrieved passages first, then the user's own text."""
    return f"{s0.context}\n\n{prompt}" if s0 and s0.used else prompt


def run(prompt: str, document_ids: list[str], mode: RagMode = "auto") -> S0Result:
    cfg = thresholds()["s0"]
    docs, missing = documents.store().get(document_ids)
    enc = resources.tiktoken_encoding("o200k_base")
    prompt_tokens = len(enc.encode(prompt))
    document_tokens = sum(d.info.tokens for d in docs)
    result = dict(mode=mode, used=False, explicit_reference=None, documents=[d.info for d in docs], missing=missing,
                  queries=[], searched_chunks=sum(d.index.ntotal for d in docs), best_score=0.0,
                  threshold=cfg["min_score"], candidates=[], passages=[], context="", context_tokens=0,
                  model=resources.EMBED_MODEL, index="FAISS IndexFlatIP (cosine, exact)",
                  prompt_tokens=prompt_tokens, document_tokens=document_tokens,
                  tokens_before=document_tokens + prompt_tokens, tokens_after=prompt_tokens)
    if mode == "off":
        return S0Result(reason="Retrieval is switched off.", **result)
    if not docs:
        return S0Result(reason="None of the attached files are in the store.", **result)

    queries = _queries(prompt, cfg["max_queries"])
    vectors = resources.embed(queries)
    best_of: dict[tuple[int, int], float] = {}
    query_top: dict[int, tuple[float, int, int]] = {}      # query -> its own best (score, doc, chunk)
    for d, doc in enumerate(docs):
        dist, ids = doc.nearest(vectors, cfg["search_k"])
        for q, (row_d, row_i) in enumerate(zip(dist.tolist(), ids.tolist())):
            for s, c in zip(row_d, row_i):
                if c >= 0 and s > best_of.get((d, c), -1.0):
                    best_of[d, c] = s
            if row_i[0] >= 0 and row_d[0] > query_top.get(q, (-1.0,))[0]:
                query_top[q] = (row_d[0], d, row_i[0])
    scored = sorted(((s, d, c) for (d, c), s in best_of.items()), reverse=True)
    best = scored[0][0] if scored else 0.0
    explicit = _explicit_reference(prompt, docs)
    threshold = cfg["min_score"]
    result.update(queries=queries, best_score=round(best, 3), explicit_reference=explicit)

    if mode == "auto" and best < threshold and not explicit:
        result["candidates"] = [_candidate(docs, d, c, s, False) for s, d, c in scored[:8]]
        return S0Result(reason=f"The prompt does not look like a question about the attached files (best match "
                               f"{best:.2f} < {threshold:.2f}); the prompt goes on unchanged.", **result)

    whole = explicit is not None and best < threshold
    if whole:
        # "Summarise the attached report": no passage stands out, so take the file(s) the prompt
        # points at from the start, in turn, as far as the token budget allows.
        targets = _referenced_docs(explicit, docs)
        scores = {d: docs[d].search(vectors, docs[d].index.ntotal) for d in targets}
        ordered = [(scores[d].get(c, 0.0), d, c) for c in range(max(len(docs[d].chunks) for d in targets))
                   for d in targets if c < len(docs[d].chunks)]
    # Hybrid ranking: dense cosine mixed with BM25 over the prompt's words, so exact names, codes
    # and rare terms count; chunks only BM25 found get their cosine from the stored vectors.
    kw = _bm25(docs, queries[0], cfg["bm25"]["k1"], cfg["bm25"]["b"])
    for d, c in set(kw) - set(best_of):
        best_of[d, c] = float((vectors @ docs[d].index.reconstruct(c)).max())
    w = cfg["keyword_weight"]
    hybrid = {k: (1 - w) * s + w * kw.get(k, 0.0) for k, s in best_of.items()}
    rr: dict | None = None
    notes: dict[tuple[int, int], str] = {}
    if whole:
        limit = len(ordered)
    else:
        # the top passages overall, plus each sub-question's own best passage ("..., and what
        # are the risks?") when it is a fair match, so every part of the prompt is covered
        floor = max(best - cfg["relative_drop"], threshold if mode == "auto" else -1.0)
        own = {(d, c) for s, d, c in query_top.values() if s >= cfg["min_query_score"]}
        pool = sorted(hybrid, key=lambda k: -hybrid[k])[:cfg["rerank_k"]]
        pool += [k for k in own if k not in pool]
        rr = _rerank(queries, docs, pool)
        if rr is not None:
            # the cross-encoder reads prompt and passage together: it decides what is relevant
            top_rr = max(rr.values(), default=0.0)
            if mode == "auto" and not explicit and top_rr < cfg["rerank_min"]:
                # a cosine near the threshold, but no passage actually answers the prompt
                result["candidates"] = [_candidate(docs, d, c, best_of[d, c], False, kw.get((d, c), 0.0), hybrid[d, c],
                                                   rr[d, c], "reranker: not relevant")
                                        for d, c in sorted(pool, key=lambda k: -rr[k])[:8]]
                result.update(ranking=f"hybrid ({1 - w:.0%} dense + {w:.0%} BM25), then cross-encoder rerank",
                              reranker=resources.RERANK_MODEL)
                return S0Result(reason=f"The files look close (best cosine {best:.2f}) but the reranker finds no passage "
                                       f"that answers the prompt (best {top_rr:.2f} < {cfg['rerank_min']:.2f}); "
                                       "the prompt goes on unchanged.", **result)
            keep = max(cfg["rerank_min"], cfg["rerank_relative"] * top_rr)
            for k in pool:
                # a sub-question's own best chunk may sit below the relative cut, never below the floor
                if rr[k] < (cfg["rerank_min"] if k in own else keep):
                    notes[k] = f"reranker {rr[k]:.2f} < {keep:.2f}"
            ordered = [(best_of[k], *k) for k in sorted(pool, key=lambda k: -rr[k]) if k not in notes]
        else:
            ordered = [(best_of[k], *k) for k in sorted(pool, key=lambda k: -hybrid[k])
                       if best_of[k] >= floor or k in own]
        limit = cfg["top_k"]
    picked: list[tuple[int, int, float]] = []
    passages: list[RetrievedPassage] = []
    picked_vecs: list[tuple[int, int, np.ndarray]] = []
    duplicates = 0
    for score, d, c in ordered:
        if len(picked) >= limit:
            break
        vec = docs[d].index.reconstruct(c)
        # a chunk saying what a picked one already says (not its overlapping neighbour) adds nothing
        if not whole and any(float(vec @ v) >= cfg["duplicate_similarity"] and not (pd == d and abs(pc - c) <= 1)
                             for pd, pc, v in picked_vecs):
            notes[d, c] = "repeats a chunk already picked"
            duplicates += 1
            continue
        trial = _merge([*picked, (d, c, score)], docs)
        if picked and sum(p.tokens for p in trial) > cfg["max_context_tokens"]:
            notes[d, c] = "over the token budget"
            continue
        picked.append((d, c, score))
        picked_vecs.append((d, c, vec))
        passages = trial
    chosen = {(d, c) for d, c, _ in picked}
    context = build_context(passages)
    context_tokens = len(enc.encode(context))
    if whole:
        names = ", ".join(dict.fromkeys(docs[d].info.filename for d, _, _ in picked))
        reason = (f"The prompt refers to the files (“{explicit}”) without a passage standing out, so "
                  f"{names} {'is' if ',' not in names else 'are'} passed on from the start")
    elif mode == "always":
        reason = "Retrieval is set to always"
    else:
        reason = f"The prompt matches the files (best {best:.2f} ≥ {threshold:.2f})"
    rank = (lambda k: -rr.get(k, -1.0)) if rr is not None else (lambda k: -hybrid.get(k, 0.0))
    shown = sorted(set(hybrid) | chosen, key=lambda k: (k not in chosen, rank(k)))[:max(10, len(picked))]
    result.update(
        used=True, passages=passages, context=context, context_tokens=context_tokens,
        tokens_after=len(enc.encode(f"{context}\n\n{prompt}")),      # what augment() hands S1
        candidates=[_candidate(docs, d, c, best_of.get((d, c), 0.0), (d, c) in chosen, kw.get((d, c), 0.0),
                               hybrid.get((d, c), 0.0), None if rr is None else rr.get((d, c)), notes.get((d, c)))
                    for d, c in shown],
        ranking=("whole file in reading order" if whole else
                 f"hybrid ({1 - w:.0%} dense + {w:.0%} BM25), then cross-encoder rerank" if rr is not None
                 else f"hybrid ({1 - w:.0%} dense + {w:.0%} BM25); reranker unavailable"),
        reranker=resources.RERANK_MODEL if rr is not None else None, duplicates_skipped=duplicates,
    )
    return S0Result(reason=f"{reason}: {len(picked)} chunk{'s' if len(picked) != 1 else ''} from "
                           f"{len({p.doc_id for p in passages})} file(s) go in front of the prompt.", **result)


def _candidate(docs: list[documents.Document], d: int, c: int, score: float, selected: bool, keyword: float = 0.0,
               hybrid: float = 0.0, rerank: float | None = None, note: str | None = None) -> ChunkScore:
    doc = docs[d]
    text = " ".join(doc.chunk_text(c).split())
    return ChunkScore(doc_id=doc.info.id, filename=doc.info.filename, chunk=c, location=doc.units[doc.chunks[c][0]].location,
                      score=round(score, 3), preview=text[:220] + ("…" if len(text) > 220 else ""), selected=selected,
                      keyword=round(keyword, 3), hybrid=round(hybrid, 3),
                      rerank=None if rerank is None else round(rerank, 3), note=note)
