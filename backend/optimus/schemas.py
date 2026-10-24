"""Typed outputs of every stage. All character offsets refer to S1Result.clean_text."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["role", "task", "context", "data", "examples", "rules", "format", "question", "padding"]
LABELS: tuple[str, ...] = ("role", "task", "context", "data", "examples", "rules", "format", "question", "padding")


# ── S1 · Clean up ────────────────────────────────────────────────────────────
class TextChange(BaseModel):
    kind: str
    count: int
    detail: str = ""


class Language(BaseModel):
    code: str
    name: str
    confidence: float
    is_english: bool
    script: str
    candidates: list[dict] = Field(default_factory=list)


class PIIEntity(BaseModel):
    type: str
    start: int
    end: int
    text: str            # the original value (kept server-side for swap-back)
    score: float
    placeholder: str


class InjectionFlag(BaseModel):
    start: int
    end: int
    text: str
    score: float
    in_data: bool
    reasons: list[str]


class S1Result(BaseModel):
    original_text: str
    clean_text: str
    changes: list[TextChange]
    language: Language
    pii: list[PIIEntity]
    pii_redacted: bool
    pii_map: dict[str, str]          # placeholder -> original, used to swap back later
    injections: list[InjectionFlag]


# ── S2 · Break into parts ────────────────────────────────────────────────────
class Block(BaseModel):
    id: str
    kind: Literal["code", "json", "table", "tagged"]
    start: int
    end: int
    meta: dict = Field(default_factory=dict)


class Segment(BaseModel):
    id: str
    start: int                 # contiguous coverage: segments tile the whole text
    end: int
    content_start: int         # the segment without surrounding whitespace
    content_end: int
    text: str
    kind: Literal["sentence", "list_item", "header", "filler", "code", "json", "table", "tagged"]
    label: Label
    confidence: float
    scores: dict[str, float]
    reasons: list[str]
    section: str | None = None
    requirement_id: str | None = None
    duplicate_of: str | None = None
    remove: bool = False


class DuplicateGroup(BaseModel):
    keep: str
    drop: list[str]
    similarity: float
    method: str


class Requirement(BaseModel):
    id: str
    segment_id: str
    label: Label
    text: str


class S2Result(BaseModel):
    segments: list[Segment]
    blocks: list[Block]
    duplicates: list[DuplicateGroup]
    requirements: list[Requirement]
    label_counts: dict[str, int]
    data_items: int
    reconstructs: bool


# ── S3 · Understand the task ─────────────────────────────────────────────────
class TaskType(BaseModel):
    type: str
    confidence: float
    scores: dict[str, float]


class Difficulty(BaseModel):
    level: Literal["low", "medium", "high"]
    score: float
    factors: dict[str, float]


class Risk(BaseModel):
    level: Literal["low", "medium", "high"]
    domains: list[str]
    reasons: list[str]


class Gap(BaseModel):
    kind: Literal["audience", "format", "length", "scope", "ambiguity", "data"]
    message: str
    assumption: str
    evidence: str | None = None


class AnswerLength(BaseModel):
    expected_tokens: int
    floor: int
    budget: int
    max_tokens: int
    basis: list[str]
    estimated: bool = True


class S3Result(BaseModel):
    task_type: TaskType
    difficulty: Difficulty
    risk: Risk
    gaps: list[Gap]
    answer_length: AnswerLength
    items: int


# ── S4 · Count and profile ───────────────────────────────────────────────────
class TokenCount(BaseModel):
    tokenizer: str
    count: int
    exact: bool
    margin: float = 0.0


class LabelTokens(BaseModel):
    label: str
    tokens: int
    share: float


class WordInfo(BaseModel):
    start: int
    end: int
    text: str
    surprisal: float      # bits, -log2 p(word | left context)
    entropy: float        # bits, LM uncertainty before the word
    excess: float         # surprisal - entropy
    rarity: float         # bits, -log2 corpus frequency
    keyphrase: float      # 0..1
    echo: bool            # rare in general, predictable here because it already appeared
    filler: bool
    score: float          # 0..1 combined informativeness


class SurprisingWord(BaseModel):
    text: str
    start: int
    end: int
    score: float
    drivers: list[str]


class LowInfoPhrase(BaseModel):
    start: int
    end: int
    text: str
    score: float
    info_bits: float
    reason: str


class Keyphrase(BaseModel):
    text: str
    score: float
    source: str


class CostPreview(BaseModel):
    model: str
    model_label: str
    input_tokens: int
    output_tokens_est: int
    price_input: float
    price_output: float
    per_call: float
    per_10k: float
    per_month: float
    monthly_volume: int
    answer_seconds: float
    output_share_of_cost: float
    last_checked: str


class QualityScore(BaseModel):
    overall: int
    clarity: int
    specificity: int
    rules: int
    structure: int
    repetition: int


class PromptWarning(BaseModel):
    code: str
    severity: Literal["info", "warn", "error"]
    message: str
    hint: str
    spans: list[tuple[int, int]] = Field(default_factory=list)


class S4Result(BaseModel):
    tokens: list[TokenCount]
    primary: TokenCount
    by_label: list[LabelTokens]
    segment_tokens: dict[str, int]
    removable_tokens: int
    heatmap: list[WordInfo]
    surprising: list[SurprisingWord]
    low_info: list[LowInfoPhrase]
    keyphrases: list[Keyphrase]
    quality: QualityScore
    warnings: list[PromptWarning]
    cost: CostPreview
    lm: str


# ── S5 · Lock key words ──────────────────────────────────────────────────────
LockCategory = Literal["number", "name", "identifier", "restriction", "label", "code",
                       "quote", "placeholder", "format", "data"]


class LockedSpan(BaseModel):
    id: str
    start: int
    end: int
    text: str
    category: LockCategory
    source: str
    detail: str | None = None
    parent: str | None = None     # id of an enclosing lock, if nested


class LabelSet(BaseModel):
    start: int
    end: int
    items: list[str]


class S5Result(BaseModel):
    spans: list[LockedSpan]
    label_sets: list[LabelSet]
    by_category: dict[str, int]
    lock_set: list[str]           # unique top-level texts every later version must contain


# ── Pipeline ─────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    target_model: str | None = None
    redact_pii: bool = False
    monthly_volume: int = Field(default=1_000_000, ge=0)


class StageTiming(BaseModel):
    stage: str
    name: str
    ms: float


class AnalyzeResponse(BaseModel):
    s1: S1Result
    s2: S2Result
    s3: S3Result
    s4: S4Result
    s5: S5Result
    timings: list[StageTiming]
    total_ms: float
