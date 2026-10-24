// Mirrors backend/optimus/schemas.py. All offsets index into s1.clean_text.

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
  cost: {
    model: string; model_label: string; input_tokens: number; output_tokens_est: number; price_input: number; price_output: number
    per_call: number; per_10k: number; per_month: number; monthly_volume: number; answer_seconds: number
    output_share_of_cost: number; last_checked: string
  }
  lm: string
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

export interface AnalyzeResponse {
  s1: S1Result; s2: S2Result; s3: S3Result; s4: S4Result; s5: S5Result
  timings: { stage: string; name: string; ms: number }[]; total_ms: number
}

export interface ModelInfo { id: string; label: string; input: number; output: number; note?: string }
