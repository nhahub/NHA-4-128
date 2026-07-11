# DermaScan AI — Project Setup & Run Guide

## Project Structure

```text
dermascan_app/
├── main.py              # FastAPI backend
├── agent.py             # LangGraph agent with role-based permissions
├── rag.py               # Knowledge base builder
├── image_registry.py    # Previously registered image matching
├── requirements.txt
└── static/
    └── index.html       # Frontend chat interface
```

## Required Files

Before running the application, place the following files in the project directory:

### Model Files

* `skin_cancer_detection_final.keras`
* `UNet_model.keras`

### Data Directory

Create a `data/` folder containing:

```text
data/
├── updated_file_2.xlsx
└── kb/
    ├── 01_diagnoses_overview.md
    ├── 02_risk_factors.md
    ├── 03_genetics_and_hereditary.md
    ├── 04_lesion_characteristics.md
    ├── 05_patient_management.md
    └── 06_image_and_dataset_info.md
```

## Installation & Execution (Windows PowerShell)

### 1. Navigate to the Project Folder

```powershell
cd dermascan_app
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 5. Configure the Groq API Key

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
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

Built by **DERMASCAN TEAM**
