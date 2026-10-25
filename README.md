# OptimusPrompt — triwizathon

Analysis pipeline stages **S0–S7** from the PRD (`worst case .pdf`) plus a web UI.

| Stage | File | What it does | Libraries |
|---|---|---|---|
| S0 Retrieve from documents | `backend/optimus/stages/s0_retrieve.py`, `backend/optimus/documents.py` | when files are attached and the prompt is about them, pull the relevant passages and put them in front of the prompt | pypdf, python-docx, python-pptx, openpyxl, MiniLM embeddings, FAISS |
| S1 Clean up | `backend/optimus/stages/s1_clean.py` | normalise text, detect language, find/redact personal info, flag embedded instructions | ftfy, regex, langdetect, Presidio + spaCy, MiniLM embeddings |
| S2 Break into parts | `backend/optimus/stages/s2_parse.py` | pull out code/JSON/tables, split sentences, label 9 kinds of content, link repeats | pysbd, spaCy, MiniLM, RapidFuzz |
| S3 Understand the task | `backend/optimus/stages/s3_task.py` | task type (from S2's task/question parts), difficulty, risk, gaps linked to S2 parts, answer-length budget | spaCy, MiniLM, tiktoken |
| S4 Count and profile | `backend/optimus/stages/s4_profile.py` | token counts, cost preview, surprisal/entropy heat map, surprising words, low-info phrases (incl. S2 padding/repeats), informativeness per S2 part, quality score, warnings | tiktoken, GPT-2 small, wordfreq, YAKE, KeyBERT |
| S5 Lock and compress | `backend/optimus/stages/s5_lock.py` | locks numbers, names, identifiers, restrictions, label sets, code, quotes, placeholders; removes unnecessary words; hands off to S6/S7 | regex, spaCy NER + dependency parse, LLMLingua-2 |
| S6 Variant A (detailed) | `backend/optimus/stages/s6_variant_a.py` | up to ten sections (role, goal, rules, assumptions, examples, plan, format, self-check, fenced inputs, reminder); implied constraints added, gaps marked as assumptions, every lock kept, ≤ 2.5× the original | rule-based, tiktoken |
| S7 Variant B (short, CSV) | `backend/optimus/stages/s7_variant_b.py` | reduction ladder L0–L5 with a lock/requirement check after each step (high-risk stops at L2), answer cap, delivered as `field,value` CSV | rule-based, tiktoken |

`backend/optimus/pipeline.py` is the driver: S0 → S1 → S2 → S3 → S4 → S5 → S6 → S7 (S0 only when files are attached).
For a question about attached files, S0 reports `tokens_before` (whole files + prompt) and `tokens_after` (retrieved passages + prompt); S6/S7 compare against `tokens_before`. Each stage reads the outputs of the ones
before it and exposes one `run(...)`, so a step can be changed on its own. Prices and thresholds live in
`backend/optimus/config/*.yaml`.

## Asking about your documents (S0)
Attach files under the prompt box (or `POST /api/v1/documents`, multipart field `files`): PDF, Word `.docx`,
PowerPoint `.pptx`, Excel `.xlsx`, CSV/TSV and text (`.txt`, `.md`, `.json`, `.html`, ...). Old binary
`.doc/.ppt/.xls` and scanned PDFs without a text layer are rejected with a message.
- **Indexing** (once per file): text is extracted with its location (page, slide, Word heading, sheet rows; table
  rows become `column: value` lines), cut into ~140-word sentence-aligned chunks with 30 words of overlap, embedded
  with the same small `all-MiniLM-L6-v2` model the other stages use (384 dims) and stored in a FAISS `IndexFlatIP`
  (unit vectors, so exact cosine search). Files are kept under `backend/data/documents/<content hash>/`
  (`OPTIMUS_DOC_DIR` overrides), so the same file is indexed once and survives a restart.
- **Retrieval** (~20 ms): the whole prompt, each sentence and each sub-question ("…, and what are the risks?") are
  searched. In `auto` mode the files are used only when a chunk scores ≥ 0.35 or the prompt points at a file
  ("the attached report", "the slides", the file name). The best chunks (and each sub-question's own best one) are
  merged where they overlap, capped at 1,500 tokens, and put in front of the prompt as one `<documents>` block.
  "Summarise the slides" has no single best passage, so that file is passed on from the start instead.
- **Next stages**: S1 checks the block for personal info and planted instructions, S2 keeps it whole as data, S3
  leans towards *answer from documents*, S4 counts its cost, and S5 locks it, so it reaches the compressed prompt intact.
`rag_mode` (`auto` / `always` / `off`) and `document_ids` go in the analyze request; settings are under `s0:` in
`thresholds.yaml`. Endpoints: `GET /api/v1/documents`, `POST /api/v1/documents`, `DELETE /api/v1/documents/{id}`.

## Removing unnecessary words (S5)
LLMLingua-2 (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`) gives every word a keep
probability, mixed 70/30 with S4's informativeness. The earlier stages decide what may go:
- **S2**: padding, carved-off filler and repeated instructions are dropped whole. Data, examples, code, tables
  and headers are never touched. Each label has a keep-rate floor (instructions 80–100%, context 50–75%).
- **S3**: high-risk tasks (legal, medical, financial, code, numeric rules) keep every word. Only S2's drops apply (ladder L2).
- **S4**: a part's informativeness sets its floor inside the band. Words in S4 low-information phrases are
  removed first, and each phrase is kept or dropped as a whole.
- **S5 locks**: always kept, so `compression.locks_preserved` must be true.

The text is rebuilt from the original words; LLMLingua's own detokeniser turns "isn't" into "t". S5 returns
`compression.prompt`, a `segments` list (one entry per S2 part with action, original and compressed text, locks, removed
spans and word scores) and a `handoff` (task, risk, max ladder level, answer budget, label → part IDs,
requirement IDs, `{{ASSUMPTION: …}}` placeholders). That is the input for S6 (Variant A) and S7 (Variant B).
Without the model (offline), S5 falls back to S4 scores and says so in `compression.method`.

## Surprising words (S4)
Each word gets several signals, z-scored within the prompt and mixed (weights in `thresholds.yaml`):
surprisal −log₂p(word | context) from GPT-2; predictive entropy before the word; **excess surprise**
(surprisal − entropy, i.e. surprise beyond what the model expected); corpus **rarity** (wordfreq);
**echo** (a rare word the LM predicts easily because it already appeared — a repetition signal); and
YAKE + KeyBERT keyphrase salience. Filler phrases are capped low.

## Run
```bash
python -m venv --system-site-packages .venv && .venv/bin/pip install -r backend/requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
cd frontend && npm install && npm run build && cd ..
cd backend && ../.venv/bin/python -m uvicorn optimus.api:app --port 8000   # http://localhost:8000
```
Dev UI with hot reload: `cd frontend && npm run dev` (proxies `/api` to :8000).

CLI: `cd backend && ../.venv/bin/python -m optimus.pipeline tests/worked_example.txt [--json] [--redact] [--doc=report.pdf ...]`
Tests: `cd backend && ../.venv/bin/python -m pytest tests`
