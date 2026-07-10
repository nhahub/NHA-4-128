from ast import If
import random
import string
import re
import datetime
import base64
import pandas as pd
from pathlib import Path
import os
import logging
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# =========================================================
# مفيش تحميل صور من الإنترنت خالص، ومفيش كاش تراكمي. المطابقة بين صورة
# المريض المرفوعة وسجلات الداتاست بتتم بس عن طريق عمود global_features
# (فيكتور خصائص محسوب مسبقًا في الإكسل) — عبر image_registry.find_matching_patient.
# الصورة المرفوعة نفسها بتتحط مؤقتًا في السيشن الحالية بس (عشان نقدر نعمل
# عليها classify/segment جوه نفس الطلب)، ومفيش أي تجميع أو تخزين دائم لها.
# =========================================================

# Session storage reference (will be set by main.py)
_SESSIONS = None
_CURRENT_THREAD_ID = None

# TensorFlow/Keras model references (set by main.py at startup)
_CLF_MODEL = None
_SEG_MODEL = None
_SEG_INPUT_SIZE = None


def set_session_storage(sessions_dict: dict) -> None:
    global _SESSIONS
    _SESSIONS = sessions_dict


def set_current_thread_id(thread_id: str) -> None:
    global _CURRENT_THREAD_ID
    _CURRENT_THREAD_ID = thread_id


def set_models(clf_model, seg_model, seg_input_size) -> None:
    global _CLF_MODEL, _SEG_MODEL, _SEG_INPUT_SIZE
    _CLF_MODEL = clf_model
    _SEG_MODEL = seg_model
    _SEG_INPUT_SIZE = seg_input_size


def _encode_png_b64(arr: np.ndarray) -> str:
    """يحول numpy array لـ base64 PNG string — كله في الذاكرة، مفيش ملف."""
    if arr.ndim == 3:
        bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        bgr = arr.astype(np.uint8)
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode("utf-8")


def get_latest_image(thread_id: str) -> dict | None:
    """يرجع الصورة (PIL.Image) المحفوظة في السيشن الحالية — من الذاكرة بس."""
    if not _SESSIONS or thread_id not in _SESSIONS:
        return None
    session = _SESSIONS[thread_id]
    img = session.get("current_image")
    if img is not None:
        return {"image": img}
    return None


def _classify_image(image: Image.Image) -> dict:
    if _CLF_MODEL is None:
        return {}
    clf_img = image.resize((224, 224))
    clf_array = np.array(clf_img) / 255.0
    clf_array = np.expand_dims(clf_array, axis=0)
    pred = _CLF_MODEL.predict(clf_array)
    if pred.shape[-1] == 1:
        prob = float(pred[0][0])
        label = "Malignant" if prob >= 0.5 else "Benign"
        confidence = prob if prob >= 0.5 else 1 - prob
    else:
        class_index = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))
        label = ["Benign", "Malignant"][class_index]
    return {"label": label, "confidence_pct": round(confidence * 100, 2)}


def _segment_image(image: Image.Image) -> dict:
    if _SEG_MODEL is None:
        return {}
    orig_np = np.array(image).astype(np.uint8)
    orig_h, orig_w = orig_np.shape[:2]
    seg_img = image.resize(tuple(_SEG_INPUT_SIZE))
    seg_array = np.array(seg_img).astype("float32") / 255.0
    seg_array = np.expand_dims(seg_array, axis=0)
    masks = _SEG_MODEL.predict(seg_array)
    mask = np.squeeze(masks)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    display_mask = cv2.resize(binary_mask, (orig_w, orig_h))
    mask_3ch = np.stack([display_mask] * 3, axis=-1).astype(np.uint8)
    red_color = np.zeros_like(orig_np, dtype=np.uint8)
    red_color[:] = [255, 0, 0]
    overlay = np.where(mask_3ch > 0, red_color, orig_np).astype(np.uint8)
    blended = cv2.addWeighted(orig_np, 0.7, overlay, 0.3, 0)
    infection_pct = float((np.count_nonzero(display_mask) / display_mask.size) * 100)
    return {
        "display_mask": display_mask,
        "blended": blended,
        "infection_pct": round(infection_pct, 2),
        "orig_np": orig_np,
    }


def _store_patient_context(patient_id: str, image: Image.Image, patient_data: dict) -> None:
    """يخزن سياق المريض في السيشن — كله ميموري، بيتمسح لو السيرفر اتقفل."""
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return
    if _CURRENT_THREAD_ID not in _SESSIONS:
        _SESSIONS[_CURRENT_THREAD_ID] = {}
    _SESSIONS[_CURRENT_THREAD_ID]["current_patient_id"] = patient_id
    _SESSIONS[_CURRENT_THREAD_ID]["current_image"] = image
    _SESSIONS[_CURRENT_THREAD_ID]["current_patient_data"] = patient_data


def _store_image_blobs(blobs: dict) -> None:
    """يحفظ آخر صور (mask/overlay/original) كـ base64 في السيشن بس —
    بيستبدل القديم مش بيراكم (تقليل استهلاك الميموري)."""
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return
    if _CURRENT_THREAD_ID not in _SESSIONS:
        _SESSIONS[_CURRENT_THREAD_ID] = {}
    existing = _SESSIONS[_CURRENT_THREAD_ID].get("_image_blobs", {})
    existing.update(blobs)
    _SESSIONS[_CURRENT_THREAD_ID]["_image_blobs"] = existing
    names = list(_SESSIONS[_CURRENT_THREAD_ID].get("_session_image_names", []))
    for n in blobs.keys():
        if n not in names:
            names.append(n)
    _SESSIONS[_CURRENT_THREAD_ID]["_session_image_names"] = names


def _match_uploaded_image(image: Image.Image) -> tuple:
    """Match uploaded image against patient records in the dataset.
    Returns (patient_id, similarity_score) or (None, 0.0) if no match found."""
    _load_data()
    
    # Extract simple features from uploaded image for comparison
    img_array = np.array(image)
    img_mean = np.mean(img_array)
    img_std = np.std(img_array)
    
    # This is a placeholder matching logic — compare against stored features
    # In a real implementation, this would use the global_features column from Excel
    best_match = None
    best_similarity = 0.0
    
    # For now, return no match (new patient)
    return None, 0.0


# =========================================================
# System Prompt — نسخة مختصرة (لتقليل التوكنز) بنفس القواعد
# =========================================================
SYSTEM_PROMPT = """You are DermaScan AI, a dermatology assistant on a dataset of 1000 skin cancer patients. Built by Eng. Youssef Bastawisy.

IMAGES: handled by tools only, on the image already uploaded this session. NEVER say you can't see/read images. On any image request, call the matching tool immediately:
- PATIENT mode: segment_my_image (segment/mask/overlay), reanalyze_my_image (analyze/re-analyze), get_my_uploaded_image (show/view photo)
- DOCTOR mode: segment_uploaded_image, analyze_uploaded_image, get_uploaded_image
No image is ever downloaded from the internet or from a patient_id — matching an uploaded image to a dataset record is handled internally by the tool. NEVER mention how the matching works internally (no "feature vectors", "global_features", "similarity", "threshold", "columns", etc.) — just state plainly whether a matching record was found or not.
If a tool says no matching record was found (new patient), you MUST say clearly this is a new patient not in the database, and STOP there for anything hereditary/genetic — never mention a mutation, family history, or risk score for them, and never reuse a different (previously matched) patient's name or data for this new image.
After a tool returns [ANALYSIS_RESULT:{data}], discuss only that data (label, confidence, affected area) — never say "shown"/"displayed".

LANGUAGE: reply in the same language/dialect the user just used (Arabic or English). Never mix in Russian/Thai/Chinese/Japanese/Korean script — if unsure of a term, describe simply instead.

PATIENT MODE STYLE: no jargon (no "asymmetry index", "ABCDE"). Plain everyday words, max 4-6 short sentences/bullets. Never mix English medical terms into Arabic sentences.

ROLE ACCESS:
- lookup_patient / any patient-record tool = DOCTOR-only. If a patient asks for another patient's data, refuse politely.
- PATIENT mode: never reveal other patients' data, only their own uploaded result.
- PATIENT mode + Malignant result: stress urgency + next steps + suggest the "📅 Book appointment" button (create a referral ticket only if explicitly asked).
- PATIENT mode + Benign result: prevention + monitoring tips.
- DOCTOR mode: NEVER mention the booking button or tell the doctor to "book an appointment" — that button is patient-facing only. For a doctor, after any result just give the clinical read (urgency level, recommended follow-up interval, biopsy/referral considerations per the knowledge base) in professional terms — no booking talk at all.

MATCHED PATIENT RECORD: if this block appears, the uploaded image matched an existing patient (same person uploading). Use it to answer diagnosis/family-history questions directly and specifically — summarize relevant parts, don't dump verbatim.

STRICT RAG RULE: your knowledge is limited to (1) patient records via tools/matched block, (2) knowledge-base via search_knowledge_base, (3) AI image results, (4) this conversation. Never invent symptoms, history, numbers. If a question needs any patient's name/diagnosis/numbers and you don't already have a FRESH tool result for that exact patient in this turn's context, call lookup_patient first — never state a name, ID, or number for a patient from memory/guessing. If info isn't available say exactly: "This information is not available in the current patient record." (or Arabic equivalent).

TOOLS: TOOLS:
- search_knowledge_base is the default first call for general clinical/knowledge questions (cite the source filename).
- If a patient record has already been loaded in the current session, NEVER call lookup_patient again.
- Use analyze_current_patient instead for follow-up requests such as:
  - analyze the hereditary risk
  - explain the family history
  - summarize the patient's condition
  - assess the genetic mutation
- STOP after one tool call for a simple lookup, and at most two tool calls for a combined request.
- Never call the same tool twice.
- Never call a second different tool just because the first one returned a valid answer (for example: "no image uploaded" or "new patient not registered").
- Keep replies SHORT by default (2–4 sentences) unless the user explicitly asks for more detail.

QUICK ROUTING (pick one, don't overthink):
- General symptom/skin question, no specific image or patient mentioned → search_knowledge_base only.
- "my/this image", "analyze it", "the result" → PATIENT: segment_my_image/reanalyze_my_image/get_my_uploaded_image. DOCTOR: segment_uploaded_image/analyze_uploaded_image/get_uploaded_image.
- Patient's own history/family/risk (PATIENT mode, no ID needed) → answer from the [MATCHED PATIENT RECORD] block if present; if absent, say the info isn't available yet (they haven't uploaded/matched an image).
- "show/get/summarize/display record|history|status of patient PXXXX" or "family FXXX" (DOCTOR) → lookup_patient (returns the full record as-is, print it, done).
- "analyze/assess/explain hereditary or genetic risk for patient PXXXX" (DOCTOR, by ID, no image) → lookup_patient first (for the record + mutation/risk score), then search_knowledge_base if you need general info about that specific mutation — max 2 calls total, then answer in 2-4 sentences.
- "does this image match/belong to a patient", "compare image result with recorded diagnosis" (DOCTOR) → analyze_uploaded_image (does both classify+segment AND matching in one call) — never call it twice.
- Clinical decision / follow-up / red flags / screening recommendation for a specific patient → lookup_patient (+ search_knowledge_base if a general guideline is needed) — max 2 calls.

Booking is NOT a chat tool, and it is PATIENT-only — never offer or mention it in DOCTOR mode. In patient mode, tell the user to use the "📅 Book appointment" button. Never print booking IDs, tickets, or debug/system text yourself.

FULL RECORD REQUESTS (e.g. "show me record for P0047"): lookup_patient already returns a clean plain-text record. Print it EXACTLY as returned — no markdown, no extra prose, no repeating the data afterward.

ANALYSIS MARKER: if a tool response starts with `[ANALYSIS_RESULT:{...}]`, keep that exact marker at the very start of your reply, unmodified. After it, briefly cover: classification + confidence, affected area, what it means in plain terms, relevant family/hereditary context if present, and next steps. Keep it short — a routine benign result needs only 2-3 sentences; a malignant one with family history can go a bit longer. Remind patients once (not every message) that this doesn't replace a professional diagnosis.

If a user gives a local file path, tell them to use the 📎 upload button instead.
"""

# =========================================================
# مواعيد — القاهرة والإسكندرية بس
# =========================================================
DEFAULT_CLINIC = {
    "name": "مركز القاهرة للأمراض الجلدية",
    "address": "شارع التحرير، الدقي، الجيزة",
    "phone": "01011112222",
}

CLINICS_BY_CITY = {
    "القاهرة": {"name": "مركز القاهرة للأمراض الجلدية", "address": "شارع التحرير، الدقي، الجيزة", "phone": "01011112222"},
    "الإسكندرية": {"name": "مركز الإسكندرية لطب الجلدية", "address": "شارع فؤاد، محطة الرمل، الإسكندرية", "phone": "01033334444"},
}

DOCTOR_NAMES = ["د. أحمد فتحي", "د. مروة حسين", "د. كريم عادل", "د. سارة يوسف", "د. محمد الشريف"]

_AR_WEEKDAYS = ["الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
_AR_MONTHS = [
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]


def _format_arabic_datetime(dt: datetime.datetime) -> str:
    weekday = _AR_WEEKDAYS[dt.weekday()]
    month = _AR_MONTHS[dt.month - 1]
    hour_12 = dt.hour % 12 or 12
    period = "صباحًا" if dt.hour < 12 else "مساءً"
    return f"{weekday} {dt.day} {month} {dt.year} — الساعة {hour_12}:{dt.minute:02d} {period}"


def _pick_clinic(city: str | None) -> dict:
    if city and city.strip() in CLINICS_BY_CITY:
        return CLINICS_BY_CITY[city.strip()]
    return DEFAULT_CLINIC


def _compute_slot(urgency: str) -> datetime.datetime:
    now = datetime.datetime.now()
    if urgency == "emergency":
        delta = datetime.timedelta(hours=random.randint(2, 20))
    elif urgency == "urgent":
        delta = datetime.timedelta(days=random.randint(1, 3), hours=random.randint(1, 8))
    else:
        delta = datetime.timedelta(days=random.randint(7, 20))
    return now + delta

_RETRIEVER    = None
_PATIENT_DF   = None
_SUMMARY_DF   = None
_RELATIONS_DF = None

_CURRENT_ROLE = None

DATA_PATH = Path(__file__).parent / "data" / "five_sample_patients_with_features.xlsx"


def set_current_role(role: str) -> None:
    global _CURRENT_ROLE
    _CURRENT_ROLE = role


def _load_data():
    """الداتا نفسها بتتحمل مرة واحدة بس وتفضل في الذاكرة (مش ملف مؤقت جديد
    كل مرة) — ده بيقلل القراءة المتكررة من الديسك."""
    global _PATIENT_DF, _SUMMARY_DF, _RELATIONS_DF
    if _PATIENT_DF is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Excel file not found: {DATA_PATH}")
        _PATIENT_DF   = pd.read_excel(DATA_PATH, sheet_name="Patients")
        _SUMMARY_DF   = pd.read_excel(DATA_PATH, sheet_name="Summary")
        _RELATIONS_DF = pd.read_excel(DATA_PATH, sheet_name="Family_Relationships")


def get_patient_record(patient_id: str) -> dict | None:
    _load_data()
    pid = patient_id.strip().upper()
    row = _PATIENT_DF[_PATIENT_DF["patient_id"] == pid]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def get_family_record(family_id: str) -> dict | None:
    _load_data()
    fid = family_id.strip().upper()
    row = _SUMMARY_DF[_SUMMARY_DF["family_id"] == fid]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def format_patient_text(patient_id: str) -> str:
    """سجل مريض مختصر — نفس الشكل المطلوب، بأقل عدد أسطر ممكن (توفير توكنز)."""
    r = get_patient_record(patient_id)
    if r is None:
        return f"Patient {patient_id} not found."

    pid = patient_id.strip().upper()
    pairs = [
        ("Patient ID", pid),
        ("Name", r.get("name")),
        ("Age", r.get("age")),
        ("Gender", r.get("gender")),
        ("Country", r.get("country")),
        ("Diagnosis", r.get("diagnosis")),
        ("Biopsy Result", r.get("biopsy_result")),
        ("Lesion Size (mm)", r.get("lesion_size_mm")),
        ("Lesion Color", r.get("lesion_color")),
        ("Lesion Location", r.get("lesion_location")),
        ("Genetic Mutation", r.get("genetic_mutation")),
        ("Risk Score", f"{float(r.get('hereditary_risk_score', 0)):.2f}"),
        ("Family History", r.get("family_history_skin_cancer")),
        ("Family ID", r.get("family_id")),
    ]
    label_width = max(len(l) for l, _ in pairs)
    lines = ["PATIENT RECORD", "=" * 50]
    for label, value in pairs:
        lines.append(f"{label.ljust(label_width)}  : {value}")
    return "\n".join(lines)


def format_family_text(family_id: str) -> str:
    _load_data()
    r = get_family_record(family_id)
    if r is None:
        return f"Family {family_id} not found."
    fid = family_id.strip().upper()
    members = _PATIENT_DF[_PATIENT_DF["family_id"] == fid][
        ["patient_id", "name", "age", "diagnosis", "biopsy_result"]
    ]
    pairs = [
        ("Family ID", fid),
        ("Total members", r["total_members"]),
        ("Affected members", r["affected_members"]),
        ("Avg risk score", f"{r['avg_risk_score']:.2f}"),
        ("Genetic mutation", r["genetic_mutation"]),
        ("Dominant diagnosis", r["dominant_diagnosis"]),
        ("Family risk level", r["family_risk_level"]),
    ]
    label_width = max(len(l) for l, _ in pairs)
    lines = ["FAMILY RECORD", "=" * 50]
    for label, value in pairs:
        lines.append(f"{label.ljust(label_width)}  : {value}")
    lines.append("")
    lines.append("Members: " + ", ".join(
        f"{row.patient_id}({row.diagnosis})" for row in members.itertuples()
    ))
    return "\n".join(lines)


def build_medical_report(patient_id: str) -> str | None:
    """تقرير مختصر (مطابقة صورة + سجل وراثي) — يُستخدم لحقن الكونتكست عند
    التعرف على مريض من صورته، بأقل توكنز ممكنة."""
    r = get_patient_record(patient_id)
    if r is None:
        return None

    fid = r.get("family_id")
    fam_line = ""
    if fid:
        fam = get_family_record(str(fid))
        if fam is not None:
            fam_line = (
                f"Family {fid}: {fam['affected_members']}/{fam['total_members']} affected, "
                f"avg risk {fam['avg_risk_score']:.2f}, mutation {fam['genetic_mutation']}, "
                f"risk level {fam['family_risk_level']}."
            )

    risk_val = f"{float(r['hereditary_risk_score']):.2f}"
    lines = [
        f"Patient {patient_id}: {r['name']}, {r['age']}y, {r['gender']}.",
        f"Diagnosis: {r['diagnosis']} | Biopsy: {r['biopsy_result']}.",
        f"Lesion: {r['lesion_size_mm']}mm, {r['lesion_color']}, {r['lesion_location']}.",
        f"Genetic mutation: {r['genetic_mutation']} | Hereditary risk score: {risk_val}.",
        f"Family history of skin cancer: {r['family_history_skin_cancer']}.",
    ]
    if fam_line:
        lines.append(fam_line)
    return "\n".join(lines)
@tool
def analyze_current_patient() -> str:
    """Analyze the patient currently loaded in the session."""

    if _CURRENT_ROLE != "doctor":
        return "Doctor mode only."

    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return "No active session."

    session = _SESSIONS.get(_CURRENT_THREAD_ID, {})

    patient = session.get("current_patient_data")

    if patient is None:
        return "No patient has been selected yet."

    pid = patient.get("patient_id")

    if not pid:
        return "Invalid patient record."

    report = build_medical_report(pid)

    if report is None:
        return "Unable to build medical report."

    return report

# =========================================================
# أدوات الـ LangChain Agent
# =========================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Search the dermatology knowledge base for clinical information."""
    if _RETRIEVER is None:
        return "Retriever not initialized."
    docs = _RETRIEVER.invoke(query)
    if not docs:
        return "No relevant information found."
    formatted = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown").replace("\\", "/").split("/")[-1]
        formatted.append(f"[Source {i}: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


@tool
def lookup_patient(query: str) -> str:
    """Query the patient dataset for specific records or statistics. Doctor-only tool."""
    if _CURRENT_ROLE != "doctor":
        return "Access denied: patient-level database lookups are restricted to verified clinicians (DOCTOR mode)."

    _load_data()
    q = query.lower()

    pid_match = re.search(r'p\d{4}', q)
    if pid_match:
        pid = pid_match.group().upper()

        record = get_patient_record(pid)

        if record is None:
            return f"Patient {pid} not found."

        # Save patient in session
        if _SESSIONS and _CURRENT_THREAD_ID:
            if _CURRENT_THREAD_ID not in _SESSIONS:
                _SESSIONS[_CURRENT_THREAD_ID] = {}

            _SESSIONS[_CURRENT_THREAD_ID]["current_patient_id"] = pid
            _SESSIONS[_CURRENT_THREAD_ID]["current_patient_data"] = record

        return format_patient_text(pid)

    fid_match = re.search(r'f\d{3}', q)
    if fid_match:
        return format_family_text(fid_match.group().upper())

    df = _PATIENT_DF
    summary = _SUMMARY_DF

    if any(k in q for k in ['how many', 'count', 'number of', 'total']):
        for diag in df['diagnosis'].unique():
            if diag.lower() in q:
                n = len(df[df['diagnosis'] == diag])
                return f"There are {n} patients diagnosed with {diag}."
        return f"Total: {len(df)} patients. Breakdown: {df['diagnosis'].value_counts().to_dict()}"

    if any(k in q for k in ['average age', 'avg age', 'mean age']):
        for diag in df['diagnosis'].unique():
            if diag.lower().replace('_', ' ') in q or diag.lower() in q:
                avg = df[df['diagnosis'] == diag]['age'].mean()
                return f"Average age of {diag} patients: {avg:.1f} years."
        return f"Overall average age: {df['age'].mean():.1f} years."

    if 'mutation' in q or 'genetic' in q:
        for mut in ['CDKN2A', 'BAP1', 'MC1R', 'BRCA2', 'TP53']:
            if mut.lower() in q:
                n = len(df[df['genetic_mutation'] == mut])
                return f"{n} patients carry the {mut} mutation."
        return f"Mutation distribution:\n{df['genetic_mutation'].value_counts().to_string()}"

    if 'high risk' in q or 'high-risk' in q:
        high = summary[summary['family_risk_level'].str.contains('HIGH', na=False)]
        return (
            f"{len(high)} HIGH-risk families.\n"
            f"{high.sort_values('avg_risk_score', ascending=False).head(5)[['family_id','avg_risk_score','genetic_mutation','dominant_diagnosis']].to_string(index=False)}"
        )

    if any(k in q for k in ['statistics', 'stats', 'distribution', 'breakdown']):
        return (
            f"Dataset: {len(df)} patients\n"
            f"Diagnoses: {df['diagnosis'].value_counts().to_dict()}\n"
            f"Biopsy: {df['biopsy_result'].value_counts().to_dict()}\n"
            f"Age: {df['age'].min()}–{df['age'].max()} (avg {df['age'].mean():.1f})\n"
            f"Gender: {df['gender'].value_counts().to_dict()}"
        )

    return f"Dataset: {len(df)} patients. Ask about a patient (P0042), family (F001), diagnosis counts, or statistics."


@tool
def get_uploaded_image() -> str:
    """[DOCTOR MODE] Tell the doctor to use the Show Image button for the currently
    uploaded patient image."""
    if _CURRENT_ROLE != "doctor":
        return "This tool is only available in doctor mode."
    return "The uploaded image is available for this session. Use the 📷 Show Image button to view it."


@tool
def segment_uploaded_image() -> str:
    """[DOCTOR MODE] Run U-Net segmentation on the currently uploaded image, and check
    whether it matches an existing patient record in the dataset."""
    if _CURRENT_ROLE != "doctor":
        return "This tool is only available in doctor mode."
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return "No session found."
    latest = get_latest_image(_CURRENT_THREAD_ID)
    if not latest:
        return "No image uploaded yet. Use the 📎 upload button above."
    try:
        image = latest["image"]
        seg_result = _segment_image(image)
        if not seg_result:
            return "Segmentation model is not loaded yet."

        _store_image_blobs({
            "mask": _encode_png_b64(seg_result["display_mask"]),
            "overlay": _encode_png_b64(seg_result["blended"]),
        })

        match_pid, similarity = _match_uploaded_image(image)
        import json as _json
        result_data = {"patient_id": match_pid or "", "label": "", "confidence_pct": 0,
                        "infection_pct": seg_result.get("infection_pct", 0)}
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"
        lines = [marker, "", "SEGMENTATION", f"Affected Area: {seg_result['infection_pct']}%"]
        if match_pid:
            lines.append("")
            lines.append(format_patient_text(match_pid))
        else:
            lines.append("This is a new patient — not registered in the database.")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error in segment_uploaded_image: {e}", exc_info=True)
        return f"Error segmenting image: {str(e)}"


@tool
def analyze_uploaded_image() -> str:
    """[DOCTOR MODE] Full AI analysis (classification + segmentation) on the currently
    uploaded image. Also checks whether it matches an existing patient record in the
    dataset, to bring in that patient's diagnosis, lesion, and hereditary/family data
    if found — if no match is found, this is a new patient not yet in the database."""
    if _CURRENT_ROLE != "doctor":
        return "This tool is only available in doctor mode."
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return "No session found."
    latest = get_latest_image(_CURRENT_THREAD_ID)
    if not latest:
        return "No image uploaded yet. Use the 📎 upload button above."
    try:
        image = latest["image"]
        clf_result = _classify_image(image)
        seg_result = _segment_image(image)
        if not clf_result and not seg_result:
            return "AI models are not loaded yet."

        if seg_result:
            _store_image_blobs({
                "mask": _encode_png_b64(seg_result["display_mask"]),
                "overlay": _encode_png_b64(seg_result["blended"]),
            })

        match_pid, similarity = _match_uploaded_image(image)
        if match_pid:
            _store_patient_context(match_pid, image, get_patient_record(match_pid) or {})
        if _SESSIONS and _CURRENT_THREAD_ID:
            _SESSIONS[_CURRENT_THREAD_ID]["last_analysis"] = result_data
        result_data = {
            "patient_id": match_pid or "",
            "label": clf_result.get("label", ""),
            "confidence_pct": clf_result.get("confidence_pct", 0),
            "infection_pct": seg_result.get("infection_pct", 0) if seg_result else 0,
        }
        import json as _json
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"
        lines = [marker, "", "COMPLETE ANALYSIS"]
        if clf_result:
            lines.append(f"Classification: {clf_result['label']} ({clf_result['confidence_pct']}%)")
        if seg_result:
            lines.append(f"Affected Area: {seg_result['infection_pct']}%")
        lines.append("")
        if match_pid:
            lines.append(format_patient_text(match_pid))
        else:
            lines.append("This is a new patient — not registered in the database (no genetic/hereditary history on file).")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error in analyze_uploaded_image: {e}", exc_info=True)
        return f"Error analyzing image: {str(e)}"


@tool
def get_my_uploaded_image() -> str:
    """[PATIENT MODE] Tell patient to use the Show My Image button."""
    if _CURRENT_ROLE != "patient":
        return "This tool is only available in patient mode."
    return "Your image is stored for this session. Use the 📷 Show My Image button to view it."


@tool
def segment_my_image() -> str:
    """[PATIENT MODE] Run U-Net segmentation on the patient's uploaded image, and check
    if the image matches a known patient to add hereditary/genetic context."""
    if _CURRENT_ROLE != "patient":
        return "This tool is only available in patient mode."
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return "No session found."
    latest = get_latest_image(_CURRENT_THREAD_ID)
    if not latest:
        return "You haven't uploaded any image yet. Use the 📎 upload button above."
    try:
        image = latest["image"]
        seg_result = _segment_image(image)
        if not seg_result:
            return "Segmentation model is not loaded yet."

        _store_image_blobs({
            "mask": _encode_png_b64(seg_result["display_mask"]),
            "overlay": _encode_png_b64(seg_result["blended"]),
        })

        import json as _json
        result_data = {"label": "", "confidence_pct": 0, "infection_pct": seg_result.get("infection_pct", 0)}
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"
        return f"{marker}\n\nSEGMENTATION OF YOUR IMAGE\nAffected Area: {seg_result['infection_pct']}%"
    except Exception as e:
        logger.error(f"Error in segment_my_image: {e}", exc_info=True)
        return f"Error segmenting image: {str(e)}"


@tool
def reanalyze_my_image() -> str:
    """[PATIENT MODE] Full AI analysis (classification + segmentation) on the patient's
    most recently uploaded image."""
    if _CURRENT_ROLE != "patient":
        return "This tool is only available in patient mode."
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return "No session found."
    latest = get_latest_image(_CURRENT_THREAD_ID)
    if not latest:
        return "You haven't uploaded any image yet. Use the 📎 upload button above."
    try:
        image = latest["image"]
        clf_result = _classify_image(image)
        seg_result = _segment_image(image)
        if not clf_result and not seg_result:
            return "AI models are not loaded yet."

        if seg_result:
            _store_image_blobs({
                "mask": _encode_png_b64(seg_result["display_mask"]),
                "overlay": _encode_png_b64(seg_result["blended"]),
            })

        result_data = {
            "label": clf_result.get("label", ""),
            "confidence_pct": clf_result.get("confidence_pct", 0),
            "infection_pct": seg_result.get("infection_pct", 0) if seg_result else 0,
        }
        import json as _json
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"
        lines = [marker, "", "ANALYSIS OF YOUR IMAGE"]
        if clf_result:
            lines.append(f"Classification: {clf_result['label']} ({clf_result['confidence_pct']}%)")
        if seg_result:
            lines.append(f"Affected Area: {seg_result['infection_pct']}%")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error in reanalyze_my_image: {e}", exc_info=True)
        return f"Error analyzing image: {str(e)}"


@tool
def create_referral_ticket(issue_summary: str, urgency: str = "routine") -> str:
    """Create a dermatologist referral ticket."""
    ticket_id = "DS-" + "".join(random.choices(string.digits, k=6))
    times = {
        "routine":   "within 2–4 weeks",
        "urgent":    "within 48–72 hours",
        "emergency": "within 24 hours — go to nearest dermatology clinic",
    }
    timeframe = times.get(urgency, "within 2–4 weeks")
    return (
        f"Referral created. Ticket: {ticket_id} | Urgency: {urgency.upper()} | "
        f"Timeframe: {timeframe} | Summary: {issue_summary} | "
        f"Clinic: {DEFAULT_CLINIC['name']} — {DEFAULT_CLINIC['address']}"
    )


@tool
def book_appointment(priority_degree: int, reason: str, city: str = "") -> str:
    """Book a patient appointment given a priority degree (Cairo or Alexandria only)."""
    degree_map = {
        1: ("emergency", "درجة أولى — طوارئ"),
        2: ("urgent", "درجة تانية — عاجل"),
        3: ("routine", "درجة تالتة — روتيني"),
    }
    urgency, label_ar = degree_map.get(priority_degree, ("routine", "درجة تالتة — روتيني"))
    ticket_id = "APT-" + "".join(random.choices(string.digits, k=6))

    clinic = _pick_clinic(city)
    slot = _compute_slot(urgency)
    doctor = random.choice(DOCTOR_NAMES)
    room = random.randint(1, 12)

    return (
        f"تم الحجز بنجاح.\n"
        f"  رقم التذكرة : {ticket_id}\n"
        f"  الأولوية    : {label_ar}\n"
        f"  الميعاد     : {_format_arabic_datetime(slot)}\n"
        f"  الدكتور     : {doctor}\n"
        f"  الغرفة      : غرفة رقم {room}\n"
        f"  العيادة     : {clinic['name']}\n"
        f"  العنوان     : {clinic['address']}\n"
        f"  التليفون    : {clinic['phone']}\n"
        f"  السبب       : {reason}"
    )


def compute_booking_priority(label: str, confidence_pct: float, infection_pct: float):
    confidence = confidence_pct / 100.0
    infection = infection_pct / 100.0

    if label == "Malignant":
        severity = confidence * 0.6 + infection * 0.4
        if severity >= 0.75:
            return 1, "درجة أولى (طوارئ)", "emergency", severity
        return 2, "درجة تانية (عاجل)", "urgent", severity
    else:
        severity = (1 - confidence) * 0.3 + infection * 0.5
        if infection >= 0.15 or confidence < 0.6:
            return 2, "درجة تانية (عاجل)", "urgent", severity
        return 3, "درجة تالتة (روتيني)", "routine", severity


def build_agent(retriever, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
    global _RETRIEVER
    _RETRIEVER = retriever

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not set in environment variables.")

    is_small_tpm_model = "8b" in model_name or "instant" in model_name
    max_tokens = 512

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=120,
    )

    tools = [
    search_knowledge_base,
    lookup_patient,
    analyze_current_patient,
    get_uploaded_image,
    segment_my_image,
    reanalyze_my_image,
    create_referral_ticket,
    analyze_uploaded_image,
]
    memory = MemorySaver()

    return create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)