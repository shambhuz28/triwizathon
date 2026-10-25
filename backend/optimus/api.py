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

from . import documents, resources
from .config import prices, thresholds
from .pipeline import STAGES, analyze
from .schemas import AnalyzeRequest, AnalyzeResponse, DocumentInfo, UploadError, UploadResponse

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


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="web")
