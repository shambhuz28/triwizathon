"""S0 · Retrieve from documents.

Runs before S1 when files are attached. The prompt (whole, and each of its sentences) is
embedded with the MiniLM model the later stages use and searched against every attached file's
FAISS index; a chunk's score is its best cosine to any of those queries. In "auto" mode the files
are used only when the prompt is about them: a chunk reaches s0.min_score, or the prompt points
at a file ("the attached report", "according to the pdf", its name). "always" skips that check
and "off" skips S0.

Normally the best chunks are chosen (within s0.relative_drop of the top score, at most top_k),
plus the best chunk of each sentence or sub-question when it reaches s0.min_query_score.
A prompt that points at a file without any chunk reaching min_score ("summarise the slides")
asks about the file as a whole, so that file is taken from the start instead. Either way the
text stays within max_context_tokens, overlapping chunks are merged, and the passages are put
back in reading order and placed in front of the prompt as one <documents> block. S2 keeps that block whole as data, S5 locks it, S1 checks it for personal
info and planted instructions, and S3 counts it towards an answer-from-documents task.
"""
from __future__ import annotations

from pathlib import Path

import regex as re

from .. import documents, resources
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


def _queries(prompt: str, limit: int) -> list[str]:
    parts = []
    for s in _SENTENCE_RE.findall(prompt):
        clauses = _CLAUSE_RE.split(s)
        parts += [s, *clauses] if len(clauses) > 1 else [s]
    parts = [p.strip() for p in parts if len(p.split()) >= 3]
    return list(dict.fromkeys([" ".join(prompt.split()), *parts[:limit]]))


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
        limit, floor = len(ordered), -1.0
    else:
        # the top passages overall, plus each sub-question's own best passage ("..., and what
        # are the risks?") when it is a fair match, so every part of the prompt is covered
        floor = max(best - cfg["relative_drop"], threshold if mode == "auto" else -1.0)
        own = {(d, c) for s, d, c in query_top.values() if s >= cfg["min_query_score"]}
        ordered = [(s, d, c) for s, d, c in scored if s >= floor or (d, c) in own]
        limit = cfg["top_k"]
    picked: list[tuple[int, int, float]] = []
    passages: list[RetrievedPassage] = []
    for score, d, c in ordered:
        if len(picked) >= limit:
            break
        trial = _merge([*picked, (d, c, score)], docs)
        if picked and sum(p.tokens for p in trial) > cfg["max_context_tokens"]:
            continue
        picked.append((d, c, score))
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
    result.update(
        used=True, passages=passages, context=context, context_tokens=context_tokens,
        tokens_after=len(enc.encode(f"{context}\n\n{prompt}")),      # what augment() hands S1
        candidates=[_candidate(docs, d, c, s, (d, c) in chosen) for s, d, c in scored[:max(8, len(picked))]],
    )
    return S0Result(reason=f"{reason}: {len(picked)} chunk{'s' if len(picked) != 1 else ''} from "
                           f"{len({p.doc_id for p in passages})} file(s) go in front of the prompt.", **result)


def _candidate(docs: list[documents.Document], d: int, c: int, score: float, selected: bool) -> ChunkScore:
    doc = docs[d]
    text = " ".join(doc.chunk_text(c).split())
    return ChunkScore(doc_id=doc.info.id, filename=doc.info.filename, chunk=c, location=doc.units[doc.chunks[c][0]].location,
                      score=round(score, 3), preview=text[:220] + ("…" if len(text) > 220 else ""), selected=selected)
