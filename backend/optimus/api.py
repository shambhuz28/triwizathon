"""FastAPI service: /api/v1 endpoints, and the built web app at / when frontend/dist exists.

    uvicorn optimus.api:app --port 8000
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import documents, live, market, resources
from .config import prices, thresholds
from .pipeline import STAGES, analyze
from .schemas import (AnalyzeRequest, AnalyzeResponse, CompareRequest, DocumentInfo, LiveRequest, MarketRequest,
                      UploadError, UploadResponse)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(resources.warmup)
    yield


app = FastAPI(title="OptimusPrompt", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/v1/health")
def health():
    return {"ok": True, "stages": STAGES}


@app.get("/api/v1/models")
def models():
    p = prices()
    return {"last_checked": str(p["last_checked"]), "default": p["default_model"],
            "models": [{"id": k, **v} for k, v in p["models"].items()]}


@app.get("/api/v1/documents", response_model=list[DocumentInfo])
def list_documents():
    return documents.store().list()


@app.post("/api/v1/documents", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    """Extract, chunk, embed and index each file; a file that fails is reported, not fatal."""
    limit = int(thresholds()["s0"]["max_file_mb"] * 1024 * 1024)
    out, errors = [], []
    for f in files:
        name = f.filename or "upload.txt"
        data = await f.read(limit + 1)
        if len(data) > limit:
            errors.append(UploadError(filename=name, error=f"larger than {thresholds()['s0']['max_file_mb']} MB"))
            continue
        try:
            out.append(await asyncio.to_thread(documents.store().add, name, data))
        except documents.UnsupportedFile as exc:
            errors.append(UploadError(filename=name, error=str(exc)))
    return UploadResponse(documents=out, errors=errors)


@app.delete("/api/v1/documents/{doc_id}")
def delete_document(doc_id: str):
    if not documents.store().delete(doc_id):
        raise HTTPException(404, "no such document")
    return {"ok": True}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_prompt(req: AnalyzeRequest):
    return await analyze(req)


def _with_whole_files(prompt: str, document_ids: list[str]) -> str:
    """The prompt with every attached file in full in front: what asking about them costs without retrieval."""
    docs, _ = documents.store().get(document_ids) if document_ids else ([], [])
    whole = "\n\n".join(u.text for d in docs for u in d.units)
    return f"{whole}\n\n{prompt}" if whole else prompt


@app.post("/api/v1/market-tokens")
async def market_tokens(req: MarketRequest):
    """Tokens and cost of the original prompt and both variants on every model on sale, each
    counted by its vendor's own counter (see optimus/market.py)."""
    texts = {"original": _with_whole_files(req.prompt, req.document_ids)}
    if req.document_ids:
        if req.retrieved:
            texts["retrieved"] = req.retrieved
    texts |= {"variant_a": req.variant_a, "variant_b": req.variant_b}
    if req.variant_b_csv:
        texts["variant_b_csv"] = req.variant_b_csv
    try:
        return await asyncio.to_thread(market.count, texts, {k: req.answers.get(k, 0) for k in texts})
    except OSError as exc:
        raise HTTPException(502, f"model catalogue unreachable ({exc})") from exc


@app.get("/api/v1/live/status")
async def live_status():
    return await asyncio.to_thread(live.status)


@app.post("/api/v1/live/run")
async def live_run(req: LiveRequest):
    """Send one prompt to Gemini or the local Ollama model and return the tokens it really used."""
    try:
        return await asyncio.to_thread(live.run, req.provider, _with_whole_files(req.prompt, req.document_ids), req.model)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/v1/live/compare")
async def live_compare(req: CompareRequest):
    """How each reply compares with the reference reply and with what the prompt asked for."""
    return await asyncio.to_thread(live.compare, req.replies, req.reference, req.labels, req.formats, req.budget)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="web")
