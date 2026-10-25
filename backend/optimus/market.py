"""Token counts and prices for the prompt on every model on the market.

Nothing here is a fixed table. The model list and per-token prices come live from OpenRouter's
public catalogue (https://openrouter.ai/api/v1/models, no key needed), refreshed hourly. Each
count comes from the vendor's own counter:
  OpenAI            tiktoken, OpenAI's tokenizer library (encoding picked by tiktoken itself)
  Anthropic         POST /v1/messages/count_tokens            needs ANTHROPIC_API_KEY
  Google Gemini     POST models/{model}:countTokens            GEMINI_API_KEY, or the live.gemini_key_file
  xAI Grok          POST /v1/tokenize-text                     needs XAI_API_KEY
  Cohere            POST /v1/tokenize                          needs CO_API_KEY or COHERE_API_KEY
  open-weight       the tokenizer published in the model's own Hugging Face repo (gated repos need HF_TOKEN)
A model whose counter cannot be reached gets no number and says why: nothing is estimated.
Open-weight models of one family and maker share one tokenizer download (the newest model's).
"""
from __future__ import annotations

import hashlib
import json
import re
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from . import resources

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
CATALOGUE_URL = "https://openrouter.ai/api/v1/models"
CATALOGUE_TTL = 3600
log = logging.getLogger(__name__)
_lock = threading.Lock()
_catalogue: tuple[float, list[dict]] | None = None
_results: dict[str, dict] = {}


def _http(url: str, body: dict | None = None, headers: dict | None = None, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "OptimusPrompt/0.1",
                                          **(headers or {})}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _key(*names: str) -> str | None:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def catalogue() -> list[dict]:
    """Text models on sale now, one entry per model (":batch", ":free" … variants folded in)."""
    global _catalogue
    with _lock:
        if _catalogue and time.time() - _catalogue[0] < CATALOGUE_TTL:
            return _catalogue[1]
    data = _http(CATALOGUE_URL)["data"]
    seen, out = set(), []
    for m in sorted(data, key=lambda m: -(m.get("created") or 0)):
        base = m["id"].split(":")[0]
        arch = m.get("architecture") or {}
        if base in seen or "text" not in (arch.get("output_modalities") or ["text"]) or arch.get("tokenizer") == "Router":
            continue
        seen.add(base)
        price = m.get("pricing") or {}
        out.append({"id": base, "name": m.get("name") or base, "vendor": base.split("/")[0],
                    "family": arch.get("tokenizer") or "Other", "hf": m.get("hugging_face_id") or None,
                    "created": m.get("created") or 0, "context": m.get("context_length"),
                    "input": float(price.get("prompt") or 0) * 1e6, "output": float(price.get("completion") or 0) * 1e6})
    with _lock:
        _catalogue = (time.time(), out)
    return out


# ── Official counters ───────────────────────────────────────────────────────
class Unavailable(Exception):
    pass


def _openai(model: str):
    import tiktoken

    name = model.split("/", 1)[1]
    try:
        enc, how = tiktoken.encoding_for_model(name), "tiktoken (model registry)"
    except KeyError:
        # tiktoken's registry lags new releases; current models use its newest (largest) base encoding
        newest = max((n for n in tiktoken.list_encoding_names() if n.endswith("_base")),
                     key=lambda n: tiktoken.get_encoding(n).n_vocab)
        enc, how = tiktoken.get_encoding(newest), f"tiktoken {newest} (newest encoding; model not in registry yet)"
    return lambda text: len(enc.encode(text, disallowed_special=())), how


@lru_cache(maxsize=1)
def _anthropic_models() -> dict[str, str]:
    key = _key("ANTHROPIC_API_KEY")
    data = _http("https://api.anthropic.com/v1/models?limit=1000",
                 headers={"x-api-key": key, "anthropic-version": "2023-06-01"})["data"]
    return {"".join(ch for ch in m["id"] if ch.isalnum()): m["id"] for m in data}


def _anthropic(model: str):
    key = _key("ANTHROPIC_API_KEY")
    if not key:
        raise Unavailable("Anthropic count_tokens API: set ANTHROPIC_API_KEY")
    norm = "".join(ch for ch in model.split("/", 1)[1] if ch.isalnum())
    api_id = next((v for k, v in _anthropic_models().items() if k.startswith(norm)), None)
    if not api_id:
        raise Unavailable("not in Anthropic's model list")

    def count(text: str) -> int:
        r = _http("https://api.anthropic.com/v1/messages/count_tokens",
                  {"model": api_id, "messages": [{"role": "user", "content": text}]},
                  {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return int(r["input_tokens"])
    return count, f"Anthropic count_tokens ({api_id})"


def _gemini(model: str):
    from .live import gemini_key

    key = gemini_key()
    if not key:
        raise Unavailable("Gemini countTokens API: set GEMINI_API_KEY")
    name = model.split("/", 1)[1]

    def count(text: str) -> int:
        r = _http(f"https://generativelanguage.googleapis.com/v1beta/models/{name}:countTokens",
                  {"contents": [{"parts": [{"text": text}]}]}, {"x-goog-api-key": key})
        return int(r["totalTokens"])
    return count, f"Gemini countTokens ({name})"


def _xai(model: str):
    key = _key("XAI_API_KEY")
    if not key:
        raise Unavailable("xAI tokenize-text API: set XAI_API_KEY")
    name = model.split("/", 1)[1]

    def count(text: str) -> int:
        r = _http("https://api.x.ai/v1/tokenize-text", {"text": text, "model": name}, {"Authorization": f"Bearer {key}"})
        return len(r["token_ids"])
    return count, f"xAI tokenize-text ({name})"


def _cohere(model: str):
    key = _key("CO_API_KEY", "COHERE_API_KEY")
    if not key:
        raise Unavailable("Cohere tokenize API: set CO_API_KEY")
    name = model.split("/", 1)[1]

    def count(text: str) -> int:
        r = _http("https://api.cohere.com/v1/tokenize", {"text": text, "model": name}, {"Authorization": f"Bearer {key}"})
        return len(r["tokens"])
    return count, f"Cohere tokenize ({name})"


@lru_cache(maxsize=None)
def _hf_tokenizer(repo: str):
    """The tokenizer a model's own Hugging Face repo publishes (tokenizer.json, else the files
    transformers reads; remote code is never run)."""
    token = _key("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(repo, token=token)
        return lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    except Exception as first:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(repo, token=token, trust_remote_code=False)
            return lambda text: len(tok.encode(text, add_special_tokens=False))
        except Exception as exc:
            msg = str(exc) or str(first)
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                raise Unavailable(f"{repo} is gated on Hugging Face: set HF_TOKEN") from exc
            raise Unavailable(f"{repo}: no tokenizer loadable without remote code") from exc


VENDOR_COUNTERS = {"openai": _openai, "anthropic": _anthropic, "google": _gemini, "x-ai": _xai, "cohere": _cohere}


def _plan(models: list[dict]) -> dict[str, tuple[str, object]]:
    """model id -> (counter key, how to build it). Open-weight models of one family and maker
    share the newest model's tokenizer."""
    family_repo: dict[tuple[str, str], str] = {}
    for m in models:                                          # newest first
        if m["hf"]:
            family_repo.setdefault((m["family"], m["hf"].split("/")[0].lower()), m["hf"])
    plan = {}
    generation_rep: dict[str, str] = {}
    for m in models:
        api = VENDOR_COUNTERS.get(m["vendor"])
        if m["hf"]:                                           # published weights: the repo's own tokenizer
            repo = family_repo.get((m["family"], m["hf"].split("/")[0].lower()), m["hf"])
            plan[m["id"]] = (f"hf:{repo}", lambda r=repo: (_hf_tokenizer(r), f"Hugging Face tokenizer ({r})"))
        elif api is not None:                                 # the vendor's own counter
            # rate-limited count APIs: one call set per model generation ("gemini-3", "claude-opus-5"),
            # made with its newest model, since a generation shares one tokenizer
            gen = re.match(r"[a-z]+(?:-[a-z]+)*-\d+", m["id"].split("/", 1)[1]) if api is not _openai else None
            rep = generation_rep.setdefault(f"{m['vendor']}:{gen.group()}", m["id"]) if gen else m["id"]
            plan[m["id"]] = (f"api:{rep}", lambda mid=rep, f=api: f(mid))
        else:
            plan[m["id"]] = ("none", None)
    return plan


def count(texts: dict[str, str], answers: dict[str, int]) -> dict:
    """Tokens and cost of each text on every catalogue model. Cached per set of texts."""
    sig = hashlib.sha256(json.dumps([texts, answers], sort_keys=True).encode()).hexdigest()
    if sig in _results:
        return _results[sig]
    models = catalogue()
    plan = _plan(models)
    counters: dict[str, tuple] = {}

    def build(ck: str, make):
        try:
            fn, how = make()
            counts = {k: fn(t) for k, t in texts.items()}
            counters[ck] = (counts, how, None)
        except Unavailable as exc:
            counters[ck] = (None, None, str(exc))
        except urllib.error.HTTPError as exc:
            counters[ck] = (None, None, f"official counter returned HTTP {exc.code}")
        except Exception as exc:                              # network, parsing
            counters[ck] = (None, None, f"counter failed: {type(exc).__name__}")

    jobs = {ck: make for ck, make in plan.values() if make is not None}
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda kv: build(*kv), jobs.items()))

    rows = []
    for m in models:
        ck, _ = plan[m["id"]]
        counts, how, why = counters.get(ck, (None, None, "no public token counter for this model"))
        cost = None if counts is None else {k: round((counts[k] * m["input"] + answers.get(k, 0) * m["output"]) / 1e6, 8)
                                           for k in texts}
        rows.append({**{k: m[k] for k in ("id", "name", "vendor", "family", "context", "input", "output")},
                     "tokens": counts, "cost": cost, "counter": how, "unavailable": why})
    o200k = resources.tiktoken_encoding("o200k_base")
    out = {"source": CATALOGUE_URL, "fetched_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(_catalogue[0])),
           "texts": list(texts), "answers": answers, "reference": {k: len(o200k.encode(t)) for k, t in texts.items()},
           "models": rows,
           "counted": sum(r["tokens"] is not None for r in rows), "total": len(rows)}
    _results[sig] = out
    return out
