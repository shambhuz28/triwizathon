# OptimusPrompt — triwizathon

Analysis pipeline stages **S1–S5** from the PRD (`worst case .pdf`) plus a web UI.

| Stage | File | What it does | Libraries |
|---|---|---|---|
| S1 Clean up | `backend/optimus/stages/s1_clean.py` | normalise text, detect language, find/redact personal info, flag embedded instructions | ftfy, regex, langdetect, Presidio + spaCy, MiniLM embeddings |
| S2 Break into parts | `backend/optimus/stages/s2_parse.py` | pull out code/JSON/tables, split sentences, label 9 kinds of content, link repeats | pysbd, spaCy, MiniLM, RapidFuzz |
| S3 Understand the task | `backend/optimus/stages/s3_task.py` | task type, difficulty, risk, gaps, answer-length budget | spaCy, MiniLM, tiktoken |
| S4 Count and profile | `backend/optimus/stages/s4_profile.py` | token counts, cost preview, surprisal/entropy heat map, surprising words, low-info phrases, quality score, warnings | tiktoken, GPT-2 small, wordfreq, YAKE, KeyBERT |
| S5 Lock key words | `backend/optimus/stages/s5_lock.py` | numbers, names, identifiers, restrictions, label sets, code, quotes, placeholders | regex, spaCy NER + dependency parse |

`backend/optimus/pipeline.py` is the driver: S1 → S2 → S3 → (S4 ∥ S5). Each stage exposes one `run(...)`,
so a step can be changed on its own. Prices and thresholds live in `backend/optimus/config/*.yaml`.

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

CLI: `cd backend && ../.venv/bin/python -m optimus.pipeline tests/worked_example.txt [--json] [--redact]`
Tests: `cd backend && ../.venv/bin/python -m pytest tests`
