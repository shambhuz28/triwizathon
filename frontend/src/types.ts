// Mirrors backend/optimus/schemas.py. All offsets index into s1.clean_text.

export type RagMode = 'auto' | 'always' | 'off'
export interface DocumentInfo {
  id: string; filename: string; kind: string; size: number; pages: number | null
  chunks: number; tokens: number; model: string; created: string
}
export interface UploadResponse { documents: DocumentInfo[]; errors: { filename: string; error: string }[] }
export interface ChunkScore {
  doc_id: string; filename: string; chunk: number; location: string | null; score: number; preview: string; selected: boolean
  keyword: number; hybrid: number; rerank: number | null; note: string | null
}
export interface RetrievedPassage { doc_id: string; filename: string; location: string | null; chunks: number[]; score: number; text: string; tokens: number }
export interface S0Result {
  mode: RagMode; used: boolean; reason: string; explicit_reference: string | null
  documents: DocumentInfo[]; missing: string[]; queries: string[]; searched_chunks: number
  best_score: number; threshold: number; candidates: ChunkScore[]; passages: RetrievedPassage[]
  context: string; context_tokens: number; model: string; index: string
  prompt_tokens: number; document_tokens: number; tokens_before: number; tokens_after: number
  ranking: string; reranker: string | null; duplicates_skipped: number
}

export type Label = 'role' | 'task' | 'context' | 'data' | 'examples' | 'rules' | 'format' | 'question' | 'padding'

export interface TextChange { kind: string; count: number; detail: string }
export interface Language { code: string; name: string; confidence: number; is_english: boolean; script: string; candidates: { code: string; prob: number }[] }
export interface PIIEntity { type: string; start: number; end: number; text: string; score: number; placeholder: string }
export interface InjectionFlag { start: number; end: number; text: string; score: number; in_data: boolean; reasons: string[] }
export interface S1Result {
  original_text: string; clean_text: string; changes: TextChange[]; language: Language
  pii: PIIEntity[]; pii_redacted: boolean; pii_map: Record<string, string>; injections: InjectionFlag[]
}

export interface Block { id: string; kind: 'code' | 'json' | 'table' | 'tagged'; start: number; end: number; meta: Record<string, unknown> }
export interface Segment {
  id: string; start: number; end: number; content_start: number; content_end: number; text: string
  kind: string; label: Label; confidence: number; scores: Record<string, number>; reasons: string[]
  section: string | null; requirement_id: string | null; duplicate_of: string | null; remove: boolean
}
export interface DuplicateGroup { keep: string; drop: string[]; similarity: number; method: string }
export interface Requirement { id: string; segment_id: string; label: Label; text: string }
export interface S2Result {
  segments: Segment[]; blocks: Block[]; duplicates: DuplicateGroup[]; requirements: Requirement[]
  label_counts: Record<string, number>; data_items: number; reconstructs: boolean
}

export interface Gap { kind: string; message: string; assumption: string; evidence: string | null; segment_ids: string[] }
export interface S3Result {
  task_type: { type: string; confidence: number; scores: Record<string, number> }
  difficulty: { level: string; score: number; factors: Record<string, number> }
  risk: { level: string; domains: string[]; reasons: string[] }
  gaps: Gap[]
  answer_length: { expected_tokens: number; floor: number; budget: number; max_tokens: number; basis: string[]; estimated: boolean }
  items: number
  task_segments: string[]; live_tokens: number
}

export interface TokenCount { tokenizer: string; count: number; exact: boolean; margin: number }
export interface WordInfo {
  start: number; end: number; text: string; surprisal: number; entropy: number; excess: number
  rarity: number; keyphrase: number; echo: boolean; filler: boolean; score: number
}
export interface S4Result {
  tokens: TokenCount[]; primary: TokenCount
  by_label: { label: string; tokens: number; share: number }[]
  segment_tokens: Record<string, number>; removable_tokens: number; segment_info: Record<string, number>
  heatmap: WordInfo[]
  surprising: { text: string; start: number; end: number; score: number; drivers: string[] }[]
  low_info: { start: number; end: number; text: string; score: number; info_bits: number; reason: string }[]
  keyphrases: { text: string; score: number; source: string }[]
  quality: { overall: number; clarity: number; specificity: number; rules: number; structure: number; repetition: number }
  warnings: { code: string; severity: 'info' | 'warn' | 'error'; message: string; hint: string; spans: [number, number][] }[]
  cost: CostPreview
  lm: string
}
export interface CostPreview {
  model: string; model_label: string; input_tokens: number; output_tokens_est: number; price_input: number; price_output: number
  per_call: number; per_10k: number; per_month: number; monthly_volume: number; answer_seconds: number
  output_share_of_cost: number; last_checked: string
}

export type LockCategory = 'number' | 'name' | 'identifier' | 'restriction' | 'label' | 'code' | 'quote' | 'placeholder' | 'format' | 'data'
export interface LockedSpan { id: string; start: number; end: number; text: string; category: LockCategory; source: string; detail: string | null; parent: string | null }
export interface PIRSegment {
  id: string; label: Label; kind: string; requirement_id: string | null
  action: 'keep' | 'compress' | 'drop'; reason: string; start: number; end: number
  text: string; compressed: string; tokens: number; compressed_tokens: number; keep_rate: number; info: number
  locks: string[]; dropped_spans: [number, number][]; word_scores: [number, number, number][]
}
export interface Compression {
  method: string; model: string | null; tokenizer: string; original_tokens: number; compressed_tokens: number
  saved_tokens: number; ratio: number; prompt: string; locks_preserved: boolean; missing_locks: string[]
}
export interface Handoff {
  task_type: string; difficulty: string; risk_level: string; max_ladder_level: 'L2' | 'L5'
  answer_budget: number; max_tokens: number; sections: Record<string, string[]>; requirements: string[]
  assumptions: { kind: string; message: string; placeholder: string; segment_ids: string[] }[]
}
export interface S5Result {
  spans: LockedSpan[]; label_sets: { start: number; end: number; items: string[] }[]
  by_category: Record<string, number>; lock_set: string[]
  segments: PIRSegment[]; compression: Compression; handoff: Handoff
}

export interface VariantCheck {
  locks_preserved: boolean; missing_locks: string[]; requirements_covered: string[]; requirements_missing: string[]
  within_size_limit: boolean
}
export interface RequirementTrace { id: string; text: string; where: string }
export interface Baseline { label: string; tokens: number; prompt_tokens: number; cost: CostPreview }
export type SectionKey = 'role' | 'goal' | 'rules' | 'assumptions' | 'examples' | 'plan' | 'format' | 'self_check' | 'inputs' | 'reminder'
export interface PromptSection { key: SectionKey; title: string; text: string; source: 'copied' | 'built'; segment_ids: string[]; tokens: number }
export interface S6Result {
  prompt: string; sections: PromptSection[]; skipped: { key: string; title: string; reason: string }[]
  tokens: number; size_limit: number; answer_tokens_est: number; answer_basis: string; max_tokens: number
  baseline: Baseline; cost: CostPreview; checks: VariantCheck; trace: RequirementTrace[]
}
export interface LadderStep {
  level: 'L0' | 'L1' | 'L2' | 'L3' | 'L4' | 'L5'; name: string; status: 'applied' | 'no change' | 'reverted' | 'skipped'
  reason: string; tokens: number; prompt: string
}
export interface S7Result {
  prompt: string; csv: string; tokens: number; csv_tokens: number
  format: 'plain' | 'csv'; format_reason: string; best_tokens: number
  steps: LadderStep[]; reached: string
  max_ladder_level: 'L2' | 'L5'; answer_tokens_est: number; answer_basis: string; answer_budget: number; max_tokens: number
  baseline: Baseline; cost: CostPreview; checks: VariantCheck; trace: RequirementTrace[]
}

export interface AnalyzeResponse {
  s0: S0Result | null
  s1: S1Result; s2: S2Result; s3: S3Result; s4: S4Result; s5: S5Result; s6: S6Result; s7: S7Result
  recommendation: Recommendation
  timings: { stage: string; name: string; ms: number }[]; total_ms: number
}

export interface ModelInfo { id: string; label: string; input: number; output: number; note?: string }

export type PromptKey = 'original' | 'A' | 'B'
export interface PromptOption {
  key: PromptKey; name: string; input_tokens: number; answer_tokens: number; per_call: number; per_month: number
  checks_passed: boolean; precision: string; best_for: string; pros: string[]; cons: string[]
}
export interface Recommendation {
  severity: 'low' | 'medium' | 'high'; severity_score: number; severity_max: number; factors: string[]
  choice: PromptKey; fallback: PromptKey | null; reason: string; options: PromptOption[]; tips: string[]; note: string
}

export type MarketText = 'original' | 'retrieved' | 'variant_a' | 'variant_b' | 'variant_b_csv'

export type LiveProvider = 'gemini' | 'ollama'
export interface LiveStatus { gemini: { available: boolean; model: string | null; why: string | null }; ollama: { available: boolean; model: string | null; why: string | null } }
export interface LiveResult {
  provider: LiveProvider; model: string; reply: string; input_tokens: number; output_tokens: number
  thinking_tokens: number; total_tokens: number; seconds: number
  fallback_from: string[]            // Gemini models skipped because their quota was used up
}
export interface ReplyComparison {
  words: number; tokens: number; similarity: number | null
  labels_found: string[]; labels_missing: string[]; format_ok: Record<string, boolean>
  preamble: boolean; over_budget: boolean
}
export interface CompareResponse { reference: string; replies: Record<string, ReplyComparison> }
export interface MarketModel {
  id: string; name: string; vendor: string; family: string; context: number | null
  input: number; output: number                      // USD per million tokens, live from the catalogue
  tokens: Partial<Record<MarketText, number>> | null; cost: Partial<Record<MarketText, number>> | null
  counter: string | null; unavailable: string | null
}
export interface MarketResponse {
  source: string; fetched_at: string; texts: MarketText[]; answers: Record<string, number>
  reference: Record<string, number>; models: MarketModel[]; counted: number; total: number
}
