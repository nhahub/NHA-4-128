# DermaScan AI

An AI-powered dermatology assistant that analyzes skin lesion images and provides role-aware
clinical support — one experience for **doctors**, another for **patients** — backed by a real
clinical dataset (1,000 patients, 8 diagnosis categories, 100 families) and a RAG-based chat agent.

---

## ✨ Features

**Doctor mode**
- Upload a lesion image → automatic match against the registered patient database
- Combined report: current image analysis (shape/label/confidence) merged with the patient's
  hereditary and family risk record
- Register unmatched images against a patient ID for future recognition
- Direct chat lookups: `"Show me patient P0042"`, `"family F001"`, dataset-wide statistics
- Full access to genetics, mutation distribution, and high-risk family rankings

**Patient mode**
- Simple, jargon-free result after image upload (Benign/Malignant + affected area %)
- Automatic booking priority score (routine / urgent / emergency)
- One-click appointment booking — date/time, doctor, room, and nearest clinic
- General Q&A on skin care, prevention, and monitoring from a trusted knowledge base
- No access to other patients' data (enforced server-side, not just by prompt)

**Under the hood**
- Dual image analysis: a CNN classifier (Benign/Malignant) + a U-Net segmentation model
  (affected area mask)
- RAG-based LLM agent (LangGraph ReAct agent) grounded in a markdown knowledge base via FAISS
- Conversation history is trimmed before each LLM call to control token usage and avoid
  drifting/repeated replies
- Primary + fallback model pair sharing one conversation memory, so a rate-limit switch
  doesn't lose context

---

## 🏗️ Architecture

```
                ┌──────────────┐
   Image ──────▶│  CNN (.keras) │──▶ label + confidence
                └──────────────┘
                ┌──────────────┐
   Image ──────▶│ U-Net (.keras)│──▶ segmentation mask + affected area %
                └──────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Doctor: match vs.            Patient: booking
   patient DB + genetic          priority score
   record → merged report        → booking button
          │
          ▼
   Chat (FastAPI /chat) ──▶ LangGraph ReAct Agent ──▶ tools:
                              • search_knowledge_base (FAISS + embeddings)
                              • lookup_patient (doctor-only, Excel dataset)
```

**Stack:** FastAPI · TensorFlow/Keras (CNN + U-Net) · LangGraph · LangChain · FAISS ·
HuggingFace sentence-transformers · Groq (LLM inference) · Pandas/OpenPyXL · OpenCV/Pillow

---

## 📁 Project Structure

```
dermascan/
├── main.py                    # FastAPI app: endpoints, session state, image pipeline
├── agent.py                   # LangGraph agent, tools, patient/family report builder
├── rag.py                     # Knowledge base loader + FAISS vectorstore builder
├── image_registry.py          # MobileNetV2-based image similarity matching for doctor mode
├── build_kb.py                # (legacy/manual) standalone KB builder — NOT used by the running app
│
├── skin_cancer_detection_final.keras   # Classification model (not committed — see below)
├── UNet_model.keras                    # Segmentation model (not committed — see below)
│
├── data/
│   ├── kb/
│   │   └── knowledge_base.md           # Consolidated clinical knowledge base (RAG source)
│   ├── updated_file_2.xlsx             # Core dataset: Patients / Summary / Family_Relationships
│   ├── five_sample_patients_with_features.xlsx  # Feature vectors for image-matching demo
│   └── image_registry.json             # (generated) registered image ↔ patient ID links
│
├── static/
│   └── index.html             # Frontend chat UI (served at the API's root)
│
├── .faiss/                    # (generated) cached FAISS index — safe to delete, rebuilds on boot
├── .env                       # API keys (not committed)
└── requirements.txt / pyproject.toml
```

### Key files explained

| File | Responsibility |
|---|---|
| `main.py` | FastAPI routes (`/set-role`, `/analyze-image`, `/register-image`, `/book-appointment`, `/chat`, `/health`), runs the CNN + U-Net inference pipeline, manages per-thread session state (role, last analysis) |
| `agent.py` | Builds the LangGraph ReAct agent, defines its two tools (`search_knowledge_base`, `lookup_patient`), the booking helper (`book_appointment`), the booking-priority formula (`compute_booking_priority`), and `build_medical_report()` — the function that merges live image analysis with a patient's genetic/family record into one report |
| `rag.py` | Loads all `.md` files under `data/kb/`, chunks them, embeds them (`sentence-transformers/all-MiniLM-L6-v2`), and builds/caches a FAISS vectorstore |
| `image_registry.py` | Extracts global image features with MobileNetV2 and cosine-matches an uploaded photo against previously registered patient images |
| `build_kb.py` | A standalone, manual knowledge-base builder kept for reference — **the running server does not call this file**; `rag.py` builds/loads the vectorstore automatically on startup |
| `static/index.html` | Single-page chat UI: role selection, image upload, analysis cards, booking flow |

---

## ⚙️ Setup

### Requirements
- Python 3.10+
- A Groq API key ([console.groq.com/keys](https://console.groq.com/keys))
- The two `.keras` model files and `data/updated_file_2.xlsx` placed locally (see **Data & Models** below — these are intentionally not committed to the repo)

### Option A — pip + requirements.txt
```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` should include, at minimum:
```
fastapi
uvicorn[standard]
python-multipart
pydantic
tensorflow
opencv-python
pillow
numpy
pandas
openpyxl
scikit-learn
langchain-core
langchain-groq
langgraph
langchain-community
langchain-huggingface
langchain-text-splitters
faiss-cpu
sentence-transformers
python-dotenv
```

### Option B — uv + pyproject.toml
```bash
uv init --no-readme
uv add fastapi "uvicorn[standard]" python-multipart pydantic tensorflow opencv-python \
       pillow numpy pandas openpyxl scikit-learn langchain-core langchain-groq langgraph \
       langchain-community langchain-huggingface langchain-text-splitters faiss-cpu \
       sentence-transformers python-dotenv

uv sync
```
Then run everything through `uv run` (see below), or activate `uv`'s virtualenv the same way as Option A.

### Environment variables (`.env`)
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Data & models (not committed — see `.gitignore`)
Place these locally before first run:
- `skin_cancer_detection_final.keras`, `UNet_model.keras` — in the project root
- `data/updated_file_2.xlsx` — the 1,000-patient clinical dataset
- `data/five_sample_patients_with_features.xlsx` — sample feature vectors for image matching
- `data/kb/knowledge_base.md` — already tracked in the repo; delete `.faiss/` if you edit it, so it rebuilds

---

## ▶️ Running

```bash
uvicorn main:app --reload --port 8000
```
or, with uv:
```bash
uv run uvicorn main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000** — the API serves the chat UI directly from `static/`.

First boot will build the FAISS index from `data/kb/*.md` (cached afterward in `.faiss/`) and
load both the primary and fallback LLM agents — this can take a little while the first time.

Check readiness anytime at:
```
GET http://127.0.0.1:8000/health
```

---

## 🔌 API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/set-role` | POST | Lock a session to `"doctor"` or `"patient"` |
| `/analyze-image` | POST | Run classification + segmentation; doctor gets a matched/merged report, patient gets a booking priority score |
| `/register-image` | POST | Link an uploaded image to a patient ID (doctor only) |
| `/book-appointment` | POST | Book a real appointment from the last analysis's severity score |
| `/chat` | POST | Send a message to the LLM agent (role-aware, tool-using) |
| `/health` | GET | Model/agent readiness check |

---

## 🧠 Notes on the Chat Agent

- Only two tools are exposed to the LLM by design: `search_knowledge_base` and `lookup_patient`.
  Booking and image analysis are **not** LLM tools — they're deterministic backend logic
  triggered by UI buttons, so the model can never "decide" to book or diagnose on its own.
- Conversation history sent to the model is trimmed to the last ~12 messages per call
  (`_trim_history` in `agent.py`) to keep token usage bounded on long conversations.
- The primary and fallback models share one `MemorySaver` checkpoint, so switching models
  mid-conversation (on a rate limit) doesn't lose context.
- Groq's `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` are deprecated (shutdown
  **Aug 16, 2026**); this project runs on `openai/gpt-oss-120b` / `openai/gpt-oss-20b` instead.

---

## 🗺️ Roadmap / Known Limitations

- Clinic and appointment data (`DEFAULT_CLINIC`, `CLINICS_BY_CITY`) are demo placeholders —
  not yet wired to a real hospital booking system or maps service
- Image-to-patient matching relies on a small sample feature set
  (`five_sample_patients_with_features.xlsx`), separate from the main clinical dataset
- CORS is wide open (`allow_origins=["*"]`) for local development — tighten before production

---

## 👤 Author

Built by **Eng. Youssef Bastawisy**
