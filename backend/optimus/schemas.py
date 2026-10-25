"""Typed outputs of every stage. All character offsets refer to S1Result.clean_text."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["role", "task", "context", "data", "examples", "rules", "format", "question", "padding"]
LABELS: tuple[str, ...] = ("role", "task", "context", "data", "examples", "rules", "format", "question", "padding")
RagMode = Literal["auto", "always", "off"]


# ── S0 · Retrieve from documents ─────────────────────────────────────────────
class DocumentInfo(BaseModel):
    id: str                       # content hash, so the same file uploaded twice is indexed once
    filename: str
    kind: str                     # pdf, docx, pptx, xlsx, csv, txt, ...
    size: int                     # bytes
    pages: int | None = None      # pages, slides or sheets, when the format has them
    chunks: int
    tokens: int                   # o200k tokens of the extracted text
    model: str                    # embedding model the index was built with
    created: str


class UploadError(BaseModel):
    filename: str
    error: str


class UploadResponse(BaseModel):
    documents: list[DocumentInfo]
    errors: list[UploadError]


class ChunkScore(BaseModel):
    doc_id: str
    filename: str
    chunk: int
    location: str | None
    score: float                  # best cosine to the whole prompt or one of its sentences
    preview: str
    selected: bool
    keyword: float = 0.0          # BM25 against the prompt's words, 0..1 (1 = best keyword match)
    hybrid: float = 0.0           # (1 - w) × score + w × keyword
    rerank: float | None = None   # cross-encoder relevance 0..1, for the candidates it read
    note: str | None = None       # why a candidate was left out


class RetrievedPassage(BaseModel):
    doc_id: str
    filename: str
    location: str | None          # "p. 3", "slide 2", "Sheet1: rows 41–80", a Word heading
    chunks: list[int]             # overlapping neighbours are merged into one passage
    score: float
    text: str
    tokens: int


class S0Result(BaseModel):
    mode: RagMode
    used: bool
    reason: str
    explicit_reference: str | None    # words in the prompt that point at the files, if any
    documents: list[DocumentInfo]     # files searched
    missing: list[str]                # requested ids that are not in the store
    queries: list[str]
    searched_chunks: int
    best_score: float
    threshold: float
    candidates: list[ChunkScore]
    passages: list[RetrievedPassage]  # in reading order
    context: str                      # the <documents> block put in front of the prompt ("" when unused)
    context_tokens: int
    model: str
    index: str
    # Token totals for comparing the two ways of asking about the files (o200k):
    prompt_tokens: int = 0            # the user's own prompt
    document_tokens: int = 0          # every attached file in full
    tokens_before: int = 0            # without retrieval: whole files + prompt
    tokens_after: int = 0             # with retrieval: retrieved passages + prompt
    ranking: str = ""                 # how candidates were ordered (hybrid, reranker)
    reranker: str | None = None       # cross-encoder used, None when unavailable
    duplicates_skipped: int = 0


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
    segment_ids: list[str] = Field(default_factory=list)   # S2 parts the gap is about; empty = whole prompt


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
    task_segments: list[str] = Field(default_factory=list)   # S2 parts that state the task / question
    live_tokens: int = 0                                     # prompt tokens once S2's padding and repeats are gone


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
    segment_info: dict[str, float] = Field(default_factory=dict)   # mean word informativeness per S2 part, 0..1
    heatmap: list[WordInfo]
    surprising: list[SurprisingWord]
    low_info: list[LowInfoPhrase]
    keyphrases: list[Keyphrase]
    quality: QualityScore
    warnings: list[PromptWarning]
    cost: CostPreview
    lm: str


# ── S5 · Lock key words and remove unnecessary ones ─────────────────────────
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


class PIRSegment(BaseModel):
    """One S2 part as handed to S6/S7: what S5 did to it and why."""
    id: str
    label: Label
    kind: str
    requirement_id: str | None = None
    action: Literal["keep", "compress", "drop"]
    reason: str
    start: int                    # content offsets in clean_text
    end: int
    text: str                     # original wording
    compressed: str               # wording after unnecessary words are removed ("" when dropped)
    tokens: int
    compressed_tokens: int
    keep_rate: float              # share of words S5 aimed to keep
    info: float                   # S4 mean informativeness, 0..1
    locks: list[str]              # ids of the top-level locks inside this part
    dropped_spans: list[tuple[int, int]] = Field(default_factory=list)   # removed words, clean_text offsets
    # (start, end, keep score) per word of a compressed part (locked words = 1.0), so S7 can
    # re-threshold at other ladder levels without running LLMLingua-2 again
    word_scores: list[tuple[int, int, float]] = Field(default_factory=list)


class Compression(BaseModel):
    method: str
    model: str | None
    tokenizer: str
    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    ratio: float                  # compressed / original
    prompt: str                   # the prompt with unnecessary words removed and every lock intact
    locks_preserved: bool
    missing_locks: list[str]


class Assumption(BaseModel):
    kind: str
    message: str
    placeholder: str              # "{{ASSUMPTION: ...}}" text for Variant A
    segment_ids: list[str]


class Handoff(BaseModel):
    """Everything S6 (Variant A) and S7 (Variant B) need, so they can build from S5 alone."""
    task_type: str
    difficulty: str
    risk_level: str
    max_ladder_level: Literal["L2", "L5"]   # high-risk tasks stop Variant B at the safe steps
    answer_budget: int
    max_tokens: int
    sections: dict[str, list[str]]          # S2 label -> ids of the parts kept, in prompt order
    requirements: list[str]                 # requirement ids every variant must cover
    assumptions: list[Assumption]


class S5Result(BaseModel):
    spans: list[LockedSpan]
    label_sets: list[LabelSet]
    by_category: dict[str, int]
    lock_set: list[str]           # unique top-level texts every later version must contain
    segments: list[PIRSegment]
    compression: Compression
    handoff: Handoff


# ── S6 / S7 · Variant A (detailed) and Variant B (short) ─────────────────────
class VariantCheck(BaseModel):
    locks_preserved: bool
    missing_locks: list[str]
    requirements_covered: list[str]
    requirements_missing: list[str]
    within_size_limit: bool


class RequirementTrace(BaseModel):
    id: str
    text: str
    where: str                    # section / row the requirement ended up in, or why it is gone


class Baseline(BaseModel):
    """What the variants are compared with: the prompt as written, or, when files were attached,
    the prompt plus the whole files (what it would cost to send them without retrieval)."""
    label: str
    tokens: int
    prompt_tokens: int            # the cleaned prompt S1–S5 worked on (after retrieval, if any)
    cost: CostPreview


class PromptSection(BaseModel):
    key: Literal["role", "goal", "rules", "assumptions", "examples", "plan", "format", "self_check", "inputs",
                 "reminder"]
    title: str
    text: str
    source: Literal["copied", "built"]    # copied from S2/S5 parts, or built from S3/S5 findings
    segment_ids: list[str]
    tokens: int


class SkippedSection(BaseModel):
    key: str
    title: str
    reason: str


class S6Result(BaseModel):
    prompt: str
    sections: list[PromptSection]
    skipped: list[SkippedSection]
    tokens: int
    size_limit: int               # PRD: 2.5 × the original, or original + 250 for very short prompts
    answer_tokens_est: int
    answer_basis: str
    max_tokens: int
    baseline: Baseline
    cost: CostPreview
    checks: VariantCheck
    trace: list[RequirementTrace]


class LadderStep(BaseModel):
    level: Literal["L0", "L1", "L2", "L3", "L4", "L5"]
    name: str
    status: Literal["applied", "no change", "reverted", "skipped"]
    reason: str
    tokens: int                   # plain-text prompt tokens after the step
    prompt: str


class S7Result(BaseModel):
    prompt: str                   # plain text
    csv: str                      # the same prompt as field,value rows
    tokens: int
    csv_tokens: int
    format: Literal["plain", "csv"] = "csv"    # the form to send: whichever takes fewer tokens
    format_reason: str = ""
    best_tokens: int = 0                       # tokens of that form; `cost` is for it
    steps: list[LadderStep]
    reached: str                  # highest step kept
    max_ladder_level: Literal["L2", "L5"]
    answer_tokens_est: int
    answer_basis: str
    answer_budget: int
    max_tokens: int
    baseline: Baseline
    cost: CostPreview             # for the CSV prompt
    checks: VariantCheck
    trace: list[RequirementTrace]


# ── S10 · Compare and recommend ─────────────────────────────────────────────
class PromptOption(BaseModel):
    key: Literal["original", "A", "B"]
    name: str
    input_tokens: int
    answer_tokens: int
    per_call: float
    per_month: float
    checks_passed: bool
    precision: str                # expected, from the prompt's structure: "highest", "as written", "lean"
    best_for: str
    pros: list[str]               # trade-offs, generated from the numbers
    cons: list[str]


class Recommendation(BaseModel):
    severity: Literal["low", "medium", "high"]
    severity_score: int           # 2 × risk + difficulty (0..6)
    severity_max: int
    factors: list[str]
    choice: Literal["original", "A", "B"]
    fallback: Literal["original", "A", "B"] | None
    reason: str
    options: list[PromptOption]
    tips: list[str]
    note: str


# ── Pipeline ─────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    target_model: str | None = None
    redact_pii: bool = False
    monthly_volume: int = Field(default=1_000_000, ge=0)
    document_ids: list[str] = Field(default_factory=list, max_length=50)   # uploaded files S0 may draw on
    rag_mode: RagMode = "auto"        # auto: only when the prompt is about the files


class MarketRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)     # the prompt as the user wrote it
    document_ids: list[str] = Field(default_factory=list, max_length=50)
    retrieved: str | None = None                              # the prompt after S0 (passages + prompt)
    variant_a: str
    variant_b: str                                            # plain text
    variant_b_csv: str | None = None                          # the same prompt as CSV
    answers: dict[str, int] = Field(default_factory=dict)     # expected answer tokens per text key


class LiveRequest(BaseModel):
    provider: Literal["gemini", "ollama"]
    prompt: str = Field(min_length=1, max_length=200_000)
    document_ids: list[str] = Field(default_factory=list, max_length=50)   # put these files in full in front
    model: str | None = None          # Gemini: keep one run on the model its first prompt landed on


class CompareRequest(BaseModel):
    replies: dict[str, str]           # prompt key -> the model's reply
    reference: str = "original"       # the reply the others are compared with
    labels: list[str] = Field(default_factory=list)      # allowed labels the answer should use (S5 label sets)
    formats: list[str] = Field(default_factory=list)     # answer formats the prompt asked for (S5 format locks)
    budget: int | None = None         # answer budget in tokens (S3)


class StageTiming(BaseModel):
    stage: str
    name: str
    ms: float


class AnalyzeResponse(BaseModel):
    s0: S0Result | None = None        # present when files were attached
    s1: S1Result
    s2: S2Result
    s3: S3Result
    s4: S4Result
    s5: S5Result
    s6: S6Result
    s7: S7Result
    recommendation: Recommendation
    timings: list[StageTiming]
    total_ms: float
