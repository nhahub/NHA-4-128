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
```

### 3. Activate the Virtual Environment

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Configure the Groq API Key

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### 6. Start the Application

```powershell
uvicorn main:app --reload --port 8000
```

When the terminal displays:

```text
Application startup complete
```

open your browser and navigate to:

```text
http://localhost:8000
```

## Features

* Role-based AI assistant (Patient / Doctor)
* Interactive chat interface
* Skin lesion image upload
* Skin cancer classification
* Lesion segmentation using U-Net
* AI-generated medical guidance
* Patient priority assessment
* Doctor report generation for matched cases
* Retrieval-Augmented Generation (RAG) knowledge base
* LangGraph-powered workflow orchestration

## Tech Stack

* Python
* FastAPI
* LangGraph
* LangChain
* TensorFlow / Keras
* U-Net
* FAISS
* HTML, CSS, JavaScript
* Groq API

- أول تشغيل بياخد وقت أطول (تحميل الموديلات + بناء الـ vectorstore) — طبيعي.
- الصور المسجّلة للدكاترة بتتخزن في `data/image_registry.json` (بيتعمل تلقائيًا).
- لو غيّرت بورت أو دومين، حدّث `API_BASE` في `static/index.html` لو هتستضيف الواجهة على سيرفر منفصل.
