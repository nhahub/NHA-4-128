<<<<<<< HEAD
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