# DermaScan AI — Chatbot Module

This is the conversational AI layer of DermaScan AI: a role-aware (Doctor / Patient)
dermatology chat agent built with LangGraph, grounded in a real clinical knowledge base via
RAG, and served through a single FastAPI endpoint.

---

## 📁 Files in this module

| File | Responsibility |
|---|---|
| `agent.py` | Builds the LangGraph ReAct agent, defines its tools, the medical-report builder, the booking-priority formula, and the booking helper |
| `rag.py` | Loads the knowledge base, chunks it, embeds it, and builds/caches the FAISS vectorstore that powers `search_knowledge_base` |
| `main.py` (`/chat`, `/set-role`) | Exposes the chatbot over HTTP, manages per-session role and retry/fallback logic |
| `data/kb/knowledge_base.md` | The clinical knowledge base — diagnoses, risk factors, lesion characteristics, genetics, patient management guidelines |
| `data/updated_file_2.xlsx` / `five_sample_patients_with_features.xlsx` | The structured patient/family dataset the agent queries directly (not via RAG) |

---

## 🧠 How the chatbot works

```
User message ──▶ /chat (main.py)
                    │
                    ▼
          role_prefix + message
        ("[DOCTOR MODE] ..." / "[PATIENT MODE] ...")
                    │
                    ▼
        LangGraph ReAct Agent (agent.py)
          │                        │
          ▼                        ▼
  search_knowledge_base      lookup_patient
  (RAG over data/kb/*.md)   (doctor-only, Excel dataset)
          │                        │
          └───────────┬────────────┘
                       ▼
              Final reply (role-aware,
              same language as the user)
```

1. **Role prefix** — every message is tagged `[DOCTOR MODE]` or `[PATIENT MODE]` before
   reaching the model, so the system prompt's role-based rules always apply deterministically.
2. **Tool selection** — the model decides whether to call `search_knowledge_base` (general
   clinical questions), `lookup_patient` (specific patient/family records, doctor-only), both,
   or neither.
3. **History trimming** — before each call, `_trim_history()` keeps only the last 12 messages
   (plus the system prompt), so token usage stays bounded on long conversations instead of
   growing with the entire chat history.
4. **Shared memory** — the primary and fallback agents both write to the same
   `MemorySaver()` checkpoint, so switching models mid-conversation (on a rate limit) doesn't
   lose context.
5. **Automatic fallback** — if the primary model hits a rate limit, `/chat` automatically
   retries on a lighter fallback model with its own separate quota.
6. **Reply sanitation** — `_sanitize_reply()` strips any stray non-Arabic/non-English tokens
   (e.g. random Cyrillic/CJK characters occasionally produced by smaller models).

---

## 🛠️ Tools exposed to the LLM (exactly two, by design)

| Tool | Access | Purpose |
|---|---|---|
| `search_knowledge_base(query)` | Both roles | Semantic search over `data/kb/knowledge_base.md` via FAISS; cites the source file |
| `lookup_patient(query)` | **Doctor only** | Returns a fully formatted report for a patient ID (`P0042`), a family ID (`F001`), or dataset-wide statistics |

`book_appointment` is defined with `@tool` in `agent.py` but is **deliberately excluded** from
the agent's toolset — it's only ever invoked directly by the "Book appointment" button in the
UI, never by the LLM. This keeps booking a deterministic action, not an AI decision.

---

## 📋 Key non-LLM functions (deterministic, for accuracy)

- **`build_medical_report(patient_id, current_analysis=None, source="chat"|"image")`** — merges
  a patient's static record (diagnosis, genetics, family history) with a *live* image analysis
  (lesion shape, classification confidence, affected area) into one report. Used both by
  `/analyze-image` (doctor mode, `source="image"`) and by `lookup_patient` in chat
  (`source="chat"`).
- **`compute_booking_priority(label, confidence_pct, infection_pct)`** — turns a classification
  result into a 3-tier urgency score (routine / urgent / emergency).
- **`_interpret_hereditary_risk(score)`** — maps a 0–1 hereditary risk score to a plain-language
  recommendation.
- **`format_family_text(family_id)`** — returns family-level statistics and member list.

---

## ⚙️ Setup

```bash
pip install langchain-core langchain-groq langgraph langchain-community \
            langchain-huggingface langchain-text-splitters faiss-cpu \
            sentence-transformers pandas openpyxl python-dotenv
```

`.env`:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Rebuilding the knowledge base index: delete the cached `.faiss/` folder — `rag.py` rebuilds it
automatically from `data/kb/*.md` on the next server start.

## ▶️ Running

The chatbot is served as part of the main API:
```bash
uvicorn main:app --reload --port 8000
```
Chat endpoint: `POST /chat` — body: `{"thread_id": "...", "message": "..."}`
Role must be set first via `POST /set-role` — body: `{"thread_id": "...", "role": "doctor"|"patient"}`

## 🧠 Model notes

- Primary: `openai/gpt-oss-120b` · Fallback: `openai/gpt-oss-20b` (both via Groq)
- Groq's `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` are deprecated (shutdown
  **Aug 16, 2026**) — this branch already runs on their replacements.
