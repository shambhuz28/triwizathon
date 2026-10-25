"""Live runs: send a prompt to Gemini and to a local Ollama model and report the tokens they
actually used, as the "chat models" script next to this project does.

  Gemini   POST models/{model}:generateContent   usageMetadata: promptTokenCount, candidatesTokenCount,
                                                  thoughtsTokenCount, totalTokenCount
  Ollama   POST /api/chat (stream off)           prompt_eval_count, eval_count

The Gemini key comes from GEMINI_API_KEY / GOOGLE_API_KEY, else from live.gemini_key_file
(the chat-models key.txt); it is sent in a header, never in the URL, and never returned.
Free-tier quotas are per model (gemini-2.5-flash: 20 requests a day), so when one model answers
429 the run moves to the next text model the key can use, from Google's model list.
Each call is one real request: it costs what the provider charges and sends the prompt to it.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from .config import thresholds

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    return thresholds()["live"]


def gemini_key() -> str | None:
    if key := os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return key.strip()
    path = Path(_cfg()["gemini_key_file"])
    path = path if path.is_absolute() else BACKEND_DIR / path
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _post(url: str, body: dict, headers: dict | None = None, timeout: float | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout or _cfg()["timeout_s"]) as r:
        return json.loads(r.read().decode())


def _ollama_models() -> list[str]:
    with urllib.request.urlopen(f"{_cfg()['ollama_url']}/api/tags", timeout=3) as r:
        return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]


def ollama_model() -> str | None:
    """The configured model if Ollama has it, else the first installed one; None when Ollama is down."""
    try:
        names = _ollama_models()
    except (OSError, ValueError):
        return None
    want = _cfg()["ollama_model"]
    return next((n for n in names if n == want or n.split(":")[0] == want), names[0] if names else None)


def status() -> dict:
    model = ollama_model()
    return {"gemini": {"available": gemini_key() is not None, "model": _cfg()["gemini_model"],
                       "why": None if gemini_key() else "no GEMINI_API_KEY and no key file"},
            "ollama": {"available": model is not None, "model": model,
                       "why": None if model else f"Ollama is not running at {_cfg()['ollama_url']}"}}


GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"
# Not chat models: audio, image, video, agents and tools share the generateContent method but not the job.
_NOT_CHAT = re.compile(r"tts|image|banana|lyria|transcribe|robotics|computer-use|antigravity|deep-research|customtools"
                       r"|embedding|aqa|veo|imagen|live|audio|omni", re.I)
_exhausted: dict[str, float] = {}       # model -> time its quota is expected back


@lru_cache(maxsize=1)
def _gemini_chat_models() -> tuple[str, ...]:
    """Text models this key can call, from Google's own model list: the configured one first, then
    flash models (newest first), then the rest. Each model has its own free-tier quota."""
    req = urllib.request.Request(f"{GEMINI_API}/models?pageSize=1000", headers={"x-goog-api-key": gemini_key() or ""})
    with urllib.request.urlopen(req, timeout=20) as r:
        models = json.loads(r.read().decode()).get("models", [])
    names = [m["name"].split("/", 1)[1] for m in models
             if "generateContent" in m.get("supportedGenerationMethods", []) and not _NOT_CHAT.search(m["name"])
             and m["name"].split("/", 1)[1].startswith("gemini")]

    def version(n: str) -> float:
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return float(v.group(1)) if v else 0.0
    preferred = _cfg()["gemini_model"]
    ranked = sorted((n for n in names if n != preferred),
                    key=lambda n: ("flash" not in n, "preview" in n or "latest" in n, -version(n), n))
    return tuple([preferred, *ranked])


def _quota_wait(exc: urllib.error.HTTPError) -> float:
    """Seconds until a 429'd model may work again: the retry delay Google gives, or a day for a daily quota."""
    try:
        details = json.loads(exc.read().decode()).get("error", {}).get("details", [])
    except ValueError:
        return 60.0
    daily = any("PerDay" in (v.get("quotaId") or "") for d in details for v in d.get("violations", []))
    delay = next((d["retryDelay"] for d in details if "retryDelay" in d), "60s")
    return 6 * 3600.0 if daily else float(re.sub(r"[^\d.]", "", delay) or 60)


def ask_gemini(prompt: str, model: str | None = None) -> dict:
    """Run on `model` (or the configured one); when its quota is used up, move to the next text model
    the key can use and say so in `fallback_from`."""
    key = gemini_key()
    if not key:
        raise RuntimeError("no Gemini API key")
    try:
        order = list(_gemini_chat_models())
    except (OSError, ValueError):
        order = [_cfg()["gemini_model"]]
    if model:
        order = [model, *[m for m in order if m != model]]
    skipped: list[str] = []
    now = time.time()
    for name in order:
        if _exhausted.get(name, 0) > now:
            skipped.append(name)
            continue
        t0 = time.time()
        try:
            res = _post(f"{GEMINI_API}/models/{name}:generateContent",
                        {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}, {"x-goog-api-key": key})
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                _exhausted[name] = time.time() + _quota_wait(exc)
                skipped.append(name)
                continue
            if exc.code == 404:                     # retired since the list was fetched
                skipped.append(name)
                continue
            raise
        seconds = time.time() - t0
        usage = res.get("usageMetadata", {})
        parts = (res.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return {"provider": "gemini", "model": name, "reply": "".join(p.get("text", "") for p in parts),
                "input_tokens": usage.get("promptTokenCount", 0), "output_tokens": usage.get("candidatesTokenCount", 0),
                "thinking_tokens": usage.get("thoughtsTokenCount", 0), "total_tokens": usage.get("totalTokenCount", 0),
                "seconds": round(seconds, 2), "fallback_from": skipped}
    raise RuntimeError(f"gemini: every model's quota is used up for now ({len(skipped)} tried); "
                       "free-tier quotas reset daily, or use a paid key")


def ask_ollama(prompt: str) -> dict:
    model = ollama_model()
    if not model:
        raise RuntimeError("Ollama is not running")
    t0 = time.time()
    res = _post(f"{_cfg()['ollama_url']}/api/chat",
                {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False})
    seconds = time.time() - t0
    inp, out = res.get("prompt_eval_count", 0), res.get("eval_count", 0)
    return {"provider": "ollama", "model": model, "reply": res.get("message", {}).get("content", ""),
            "input_tokens": inp, "output_tokens": out, "thinking_tokens": 0, "total_tokens": inp + out,
            "seconds": round(seconds, 2), "fallback_from": []}


_PREAMBLE_RE = re.compile(r"^\s*(?:sure|certainly|of course|absolutely|great|okay|ok|here(?:'s| is| are)|below is|let me"
                          r"|i(?:'d| would|'m| am) (?:be )?(?:happy|glad|delighted)|happy to|glad to|you(?:'re| are) (?:very )?welcome"
                          r"|thanks? (?:you )?for|no problem|alright)\b", re.I)
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$\n^\s*\|?\s*:?-{3,}", re.M)


def _format_ok(reply: str, fmt: str) -> bool:
    fmt = fmt.lower()
    if "table" in fmt:
        return bool(_TABLE_RE.search(reply))
    if "json" in fmt:
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
        try:
            json.loads(body)
            return True
        except ValueError:
            return False
    if "bullet" in fmt or "list" in fmt:
        return bool(re.search(r"^\s*(?:[-*•]|\d+[.)])\s", reply, re.M))
    if "csv" in fmt:
        return reply.count(",") >= 2 and "\n" in reply.strip()
    return True


def _content(reply: str) -> str:
    """The reply without a chat opener line and Markdown/table syntax, so similarity measures what
    it says rather than how it is dressed ("Sure! Here is the table:" adds no content)."""
    lines = reply.strip().splitlines()
    while lines and (_PREAMBLE_RE.search(lines[0]) and len(lines[0].split()) <= 12 or not lines[0].strip()):
        lines = lines[1:]
    text = "\n".join(lines)
    text = re.sub(r"^\s*\|?\s*:?-{3,}.*$", " ", text, flags=re.M)      # table separator rows
    return " ".join(re.sub(r"[|*#`_>]+", " ", text).split()) or reply


def compare(replies: dict[str, str], reference: str, labels: list[str], formats: list[str], budget: int | None) -> dict:
    """For each reply: length, meaning kept vs the reference reply (MiniLM cosine of the content), the
    labels it uses, whether it answers in the requested format, and whether it opens with chat preamble."""
    from . import resources

    enc = resources.tiktoken_encoding("o200k_base")
    keys = [k for k, v in replies.items() if v.strip()]
    vecs = dict(zip(keys, resources.embed([_content(replies[k]) for k in keys]))) if keys else {}
    ref = vecs.get(reference)
    out = {}
    for k in keys:
        text = replies[k]
        low = text.lower()
        found = [lb for lb in labels if re.search(rf"\b{re.escape(lb.lower())}\b", low)]
        out[k] = {
            "words": len(text.split()), "tokens": len(enc.encode(text)),
            "similarity": None if ref is None else round(float(vecs[k] @ ref), 3),
            "labels_found": found, "labels_missing": [lb for lb in labels if lb not in found],
            "format_ok": {f: _format_ok(text, f) for f in formats},
            "preamble": bool(_PREAMBLE_RE.search(text)),
            "over_budget": budget is not None and len(enc.encode(text)) > budget,
        }
    return {"reference": reference, "replies": out}


def run(provider: str, prompt: str, model: str | None = None) -> dict:
    try:
        return ask_gemini(prompt, model) if provider == "gemini" else ask_ollama(prompt)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(f"{provider}: quota or rate limit reached (HTTP 429); wait a minute, "
                               "or check the key's plan and billing") from exc
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"{provider} returned HTTP {exc.code}: {detail}") from exc
