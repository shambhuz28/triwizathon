"""FastAPI service: /api/v1 endpoints, and the built web app at / when frontend/dist exists.

    uvicorn optimus.api:app --port 8000
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import resources
from .config import prices
from .pipeline import STAGES, analyze
from .schemas import AnalyzeRequest, AnalyzeResponse

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


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_prompt(req: AnalyzeRequest):
    return await analyze(req)


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="web")
