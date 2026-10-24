"""Lazily loaded local models shared by the stages (all CPU, no API cost)."""
from __future__ import annotations

import logging
import threading
from functools import lru_cache, wraps

_lock = threading.RLock()

SPACY_MODEL = "en_core_web_sm"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LM_MODEL = "openai-community/gpt2"
LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def _load_once(fn):
    cached = lru_cache(maxsize=None)(fn)

    @wraps(fn)
    def wrapper():
        with _lock:
            return cached()

    return wrapper


@_load_once
def nlp():
    import spacy

    return spacy.load(SPACY_MODEL)


@lru_cache(maxsize=64)
def parse(text: str):
    """spaCy Doc for a text, cached so several stages can share one parse."""
    return nlp()(text)


@_load_once
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(texts: list[str]):
    """Unit-normalised sentence embeddings (rows), so a dot product is a cosine."""
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype="float32")
    return embedder().encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)


@_load_once
def lm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL)
    model.eval()
    torch.set_grad_enabled(False)
    return tok, model


@_load_once
def llmlingua():
    """LLMLingua-2 compressor (a token classifier scoring how worth keeping each token is), or None
    when the package or model is unavailable; S5 then ranks words by S4 informativeness alone."""
    try:
        from llmlingua import PromptCompressor

        return PromptCompressor(model_name=LLMLINGUA_MODEL, use_llmlingua2=True, device_map="cpu")
    except Exception as exc:
        logging.getLogger(__name__).warning("LLMLingua-2 unavailable (%s); S5 falls back to S4 scores", exc)
        return None


@_load_once
def keybert():
    from keybert import KeyBERT

    return KeyBERT(model=embedder())


@_load_once
def presidio():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
    })
    engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    # Phone numbers: Presidio's default recogniser only parses a few regions.
    from presidio_analyzer.predefined_recognizers import PhoneRecognizer

    engine.registry.add_recognizer(PhoneRecognizer(supported_regions=("IN", "US", "GB", "CA", "AU", "SG", "AE")))
    # Indian identifiers are not in Presidio's default registry.
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="IN_AADHAAR",
        patterns=[Pattern("aadhaar", r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b", 0.6)],
        context=["aadhaar", "aadhar", "uid", "uidai"],
    ))
    engine.registry.add_recognizer(PatternRecognizer(
        supported_entity="IN_PAN",
        patterns=[Pattern("pan", r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.7)],
        context=["pan", "permanent account number"],
    ))
    return engine


@lru_cache(maxsize=None)
def tiktoken_encoding(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def warmup() -> None:
    """Load every model once so the first request is not slow."""
    parse("Warm up the parser.")
    embed(["warm up"])
    lm()
    keybert()
    llmlingua()
    presidio().analyze(text="warm up", language="en")
    tiktoken_encoding("o200k_base")
    tiktoken_encoding("cl100k_base")
