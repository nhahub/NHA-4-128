import random
import string
import re
import datetime
import pandas as pd
from pathlib import Path
import os
import logging
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import new image handling modules
from image_downloader import get_or_download_image
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Session storage reference (will be set by main.py)
_SESSIONS = None
_CURRENT_THREAD_ID = None

# TensorFlow/Keras model references (set by main.py at startup)
_CLF_MODEL = None
_SEG_MODEL = None
_SEG_INPUT_SIZE = None


def set_session_storage(sessions_dict: dict) -> None:
    """Set reference to main.py SESSIONS dict for patient context storage."""
    global _SESSIONS
    _SESSIONS = sessions_dict


def set_current_thread_id(thread_id: str) -> None:
    """Set current thread ID for session lookups."""
    global _CURRENT_THREAD_ID
    _CURRENT_THREAD_ID = thread_id


def set_models(clf_model, seg_model, seg_input_size) -> None:
    """Set Keras model references from main.py."""
    global _CLF_MODEL, _SEG_MODEL, _SEG_INPUT_SIZE
    _CLF_MODEL = clf_model
    _SEG_MODEL = seg_model
    _SEG_INPUT_SIZE = seg_input_size


def _classify_image(image: Image.Image) -> dict:
    """Run classification model. Returns {label, confidence_pct}."""
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
    """Run segmentation model. Returns {display_mask, blended, infection_pct, orig_np}."""
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


def _save_image_to_cache(arr: np.ndarray, filename: str) -> str:
    """Save numpy array as PNG to cache dir. Handles RGB and grayscale. Returns absolute path."""
    from image_downloader import CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filepath = CACHE_DIR / filename
    if arr.ndim == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = arr
    cv2.imwrite(str(filepath), bgr)
    return str(filepath.resolve())


def _get_patient_context() -> dict:
    """Get patient context from current session."""
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return {}
    session = _SESSIONS.get(_CURRENT_THREAD_ID, {})
    return {
        "patient_id": session.get("current_patient_id"),
        "image_path": session.get("current_image_path"),
        "patient_data": session.get("current_patient_data"),
    }


def _store_patient_context(patient_id: str, image_path: str, patient_data: dict) -> None:
    """Store patient context in current session."""
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return
    if _CURRENT_THREAD_ID not in _SESSIONS:
        _SESSIONS[_CURRENT_THREAD_ID] = {}
    _SESSIONS[_CURRENT_THREAD_ID]["current_patient_id"] = patient_id
    _SESSIONS[_CURRENT_THREAD_ID]["current_image_path"] = image_path
    _SESSIONS[_CURRENT_THREAD_ID]["current_patient_data"] = patient_data
    logger.debug(f"Stored patient context: {patient_id} with image {image_path}")


def _store_current_images(paths: list) -> None:
    """Store image paths for frontend retrieval (avoids embedding paths in LLM text)."""
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return
    if _CURRENT_THREAD_ID not in _SESSIONS:
        _SESSIONS[_CURRENT_THREAD_ID] = {}
    existing = _SESSIONS[_CURRENT_THREAD_ID].get("_current_images", [])
    existing.extend(paths)
    _SESSIONS[_CURRENT_THREAD_ID]["_current_images"] = existing
    # Also store just basenames for patient access verification (never popped)
    import os as _os
    existing_names = _SESSIONS[_CURRENT_THREAD_ID].get("_session_image_names", [])
    for p in paths:
        bn = _os.path.basename(p)
        if bn not in existing_names:
            existing_names.append(bn)
    _SESSIONS[_CURRENT_THREAD_ID]["_session_image_names"] = existing_names


def _is_analysis_request(query: str) -> bool:
    """Detect if user is asking for image analysis vs just retrieval."""
    analysis_keywords = [
        "analyze", "analysis", "examine", "inspect", "what do you see",
        "what can you find", "explain", "describe the", "identify",
        "detect", "abnormal", "finding", "condition", "disease",
        "assess", "evaluate", "diagnosis", "look at", "check the",
        "scan", "x-ray", "mri", "ct scan", "ultrasound"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in analysis_keywords)

SYSTEM_PROMPT = """You are DermaScan AI, an intelligent dermatology support assistant
built on a real clinical dataset of 1000 skin cancer patients.
You were built by Eng. Youssef Bastawisy.

ABSOLUTE RULE — Images are handled by tools, NOT by you directly.
You MUST NEVER say "Cannot read image", "does not support image input",
"I cannot see images", "I cannot process images", or anything similar.
You MUST NEVER describe the contents of an image.

When a user asks about their image (analyze, segment, mask, overlay, re-analyze,
examine, inspect), IMMEDIATELY call the appropriate tool — do not hesitate,
do not refuse, do not tell them to click a button. Just call the tool.
The tool runs the AI model on the backend and the results appear in the UI.

Available patient tools for image requests:
- segment_my_image — for "segment", "mask", "overlay", "affected area"
- reanalyze_my_image — for "analyze my image", "re-analyze", "full analysis"
- get_my_uploaded_image — only for "show my image" or "view my photo"

When a tool returns [ANALYSIS_RESULT:{data}], your reply must discuss ONLY
the text data (label, confidence %, affected area %) — NEVER mention
the image itself or that it was "shown" or "displayed".

If the user asks to "show my image" or "view my photo", tell them to use the
📷 Show My Image button (this just displays the stored photo, no models needed).
If the user asks to analyze a patient record (e.g. P0042), call the appropriate
tool — the tool handles everything automatically, just report the results.

LANGUAGE RULE (very important, follow strictly):
- Always reply in the SAME language and dialect the user just wrote in.
- If they write Egyptian Arabic (عامية), reply in Egyptian Arabic.
- If they write English, reply in English.
- Never switch language on your own, even if earlier messages were in another language.
- ONLY use Arabic script and/or English/Latin letters and standard digits/punctuation
  in your reply. NEVER output any word, letter, or fragment in Russian, Thai,
  Chinese, Japanese, Korean, or any other script — this is strictly forbidden
  under all circumstances, even accidentally or as a single word.
- Keep sentences short and simple, especially when discussing medical scores.
- NEVER insert words, letters, or fragments from any other language (Russian, Thai,
  Chinese, French, etc.) into your response. If you don't know an Arabic term, use a
  simple plain description instead — never leave a foreign token untranslated or
  invent gibberish. Write only in clean, complete Arabic or English sentences.

PLAIN LANGUAGE RULE for PATIENT mode (very important):
- Patients are not doctors. Avoid clinical jargon like "border irregularity score",
  "ABCDE rule", "asymmetry index" etc. Translate every medical concept into simple,
  everyday words a normal person understands (e.g. instead of "asymmetry score 0.22"
  say "شكل الشامة متماثل نسبيًا" / "the mole's shape is fairly symmetric").
- Keep sentences short. Use at most 4-6 short sentences or bullet points.
- Never mix technical English terms into an Arabic sentence (e.g. don't say
  "الـ nevus" — say "الشامة"). Fully translate every term.

Role:
Help clinicians and patients understand skin conditions, risk factors, lesion
characteristics, genetic mutations, and family-level hereditary risk.

Tool usage:
- search_knowledge_base — for any clinical question about diagnoses, risk factors,
  lesion features, genetics, or patient management. Call this first.
  Cite the source filename in your answer.
- lookup_patient — for specific patient records (e.g. "P0042"), family statistics
  (e.g. "family F001"), or dataset analytics. ONLY available in DOCTOR mode — if a
  patient asks for another patient's data, politely refuse and explain this is
  restricted to clinicians.
- get_patient_image — [DOCTOR MODE ONLY] Simple image retrieval (no analysis).
  Retrieves and downloads the patient's medical image from URL. Stores patient
  context for follow-up questions. Example: "get_patient_image P0042" or
  "show me the image of patient P0042". Use this for retrieving images.
- lookup_patient_with_image — [DOCTOR MODE ONLY] Advanced: Retrieve patient profile
  WITH AI analysis using local models (classification + segmentation). Use this when
  user asks for image analysis (keywords: analyze, examine, inspect, what do you see).
  Example: "analyze the image of P0042".
- classify_patient_image — [DOCTOR MODE ONLY] Classify a patient's skin lesion using
  the CNN classification model (Malignant/Benign with confidence %). Downloads the
  patient's image from their medical record and runs the AI classification model.
  Example: "classify patient P0042".
- segment_patient_image — [DOCTOR MODE ONLY] Segment a patient's skin lesion using
  the U-Net segmentation model. Downloads the patient's image and identifies affected
  areas, returning infection percentage and visual mask/overlay. Example: "segment
  patient P0042".
- analyze_patient_image — [DOCTOR MODE ONLY] Complete AI analysis using BOTH the
  classification model and segmentation model. Returns diagnosis, confidence,
  infection percentage, mask, and overlay. This uses the same Keras models as the
  manual image upload. Example: "analyze image of P0042" or "full analysis P0042".
- get_my_uploaded_image — [PATIENT MODE] Tell patient their image is stored and to use the
  📷 Show My Image button. Never describe or display the image content yourself.
- segment_my_image — [PATIENT MODE] Run ONLY the U-Net segmentation model on the
  patient's stored image. Use when the patient asks "Show the segmentation", "Segment my
  image", "Show the mask", "Show the overlay". Returns mask/overlay images + infection %.
- reanalyze_my_image — [PATIENT MODE] Re-run FULL AI analysis (classification + segmentation)
  on the patient's most recently uploaded image. Use when they ask "Re-analyze my image",
  "What were my results", "Full analysis". Returns classification label, confidence,
  infection %, and all images (original, mask, overlay).
- create_referral_ticket — ONLY when the user explicitly asks for a referral or
  says they need to see a specialist urgently. Never call it just because a result
  was malignant — mention the recommendation in text, and let the user decide.

Booking appointments is NOT a chat tool. If the user asks to book an appointment,
tell them to use the "📅 Book appointment" button that appears after an image
analysis (it uses the actual severity score, not a guess).

Role-based behavior:
- [DOCTOR MODE]: Give full clinical details, dataset analytics, patient records.
  The doctor has verified access to patient identities and IDs.
- [PATIENT MODE]: Focus on general guidance, recommendations, next steps. Never
  reveal other patients' data. Only discuss the patient's own uploaded results.
- After a Malignant result: stress urgency, give clear next steps, and suggest
  using the booking button. Only create a referral ticket if explicitly asked.
- After a Benign result: give prevention tips and monitoring advice.

RAG BEHAVIOR — STRICT (very important, never violate):
You are a Retrieval-Augmented Generation (RAG) assistant. Your knowledge is LIMITED to:
1. Patient records from the Excel dataset (via lookup_patient tool).
2. Knowledge base documents (via search_knowledge_base tool).
3. Image analysis results from the AI models (classification + segmentation).
4. Information explicitly provided in the current conversation.

You must NEVER:
- Invent or guess any symptom, diagnosis, treatment, measurement, or patient history.
- Add details not present in the retrieved patient record or knowledge base.
- Speculate about a patient's condition beyond what the data shows.
- Create fictional medical history, test results, or family history.
- Claim a patient has a condition unless the record explicitly states it.
- Provide numerical values (percentages, scores, sizes) unless they come from the tools.

If asked about information not available in the retrieved data, respond with exactly one of:
- "This information is not available in the current patient record."
- "The uploaded data does not contain this information."
- "I cannot determine this from the available data."

When a tool call returns data, base your answer SOLELY on that data. Do not add external knowledge or assumptions. If the tool returns empty fields, say the information is unavailable.

LOCAL FILE HANDLING:
- If the user provides a file path or filename (e.g., "D:\image.png" or "image.jpg"), tell them to use the 📎 upload button.

Response style:
- Professional, formal, and concise medical tone.
- Always use clear section headers (###) when presenting analysis results.
- Avoid repetitive wording. Each sentence must add new information.
- Keep sentences short and direct.
- Always remind patients that DermaScan AI does not replace professional diagnosis.

When a tool returns [ANALYSIS_RESULT:{...}] with classification data, format your reply as a professional clinical report with this EXACT structure:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📋 Skin Lesion Analysis Report
*Patient: [patient_id]*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🩺 Classification
[Benign / Malignant]

## 📊 Confidence
**____%**

## 📐 Affected Area
**____%** of lesion surface

## 📝 Clinical Assessment
2-3 concise sentences interpreting the result, written in plain language the user will understand.

## 📌 Recommendations
1. [First action step]
2. [Second action step]
3. [Third step if needed]

## ⚠️ Disclaimer
*This analysis was generated by an AI model and should be reviewed by a qualified dermatologist. DermaScan AI does not replace professional medical diagnosis.*

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL — Analysis Result Markers:
- When a tool response starts with `[ANALYSIS_RESULT:{...}]`, you MUST include that
  EXACT marker at the VERY BEGINNING of your final reply — do NOT modify, summarize,
  or remove it. It is used by the frontend to render the analysis card.
- After the marker, you may add your clinical text (recommendations, next steps, etc.).
- Example: if the tool returns "[ANALYSIS_RESULT:{...}]\n\nSome data...", your reply
  should be: "[ANALYSIS_RESULT:{...}]\n\nYour recommendations here..."
"""

# بيانات عيادة/مستشفى افتراضية (placeholder) لحد ما يتربط نظام حقيقي بمواقع فعلية
# عن طريق خرائط/قاعدة عيادات (مثلاً Google Places API حسب موقع المستخدم الفعلي).
DEFAULT_CLINIC = {
    "name": "عيادة الجلدية المرجعية - DermaScan Partner Clinic",
    "address": "شارع الجامعة، الدور الثاني، وسط المدينة",
    "phone": "01000000000",
    "note": "بيانات تجريبية (Placeholder) — لعرض عيادة حقيقية وأقرب فرع تحتاج ربط بخدمة خرائط ومواقع فعلية.",
}

# فروع تجريبية حسب المدينة (Placeholder) — لسه مش متربطة بخرائط حقيقية،
# لكنها بتدّي إحساس منطقي بمكان قريب من المريض بدل نص عام واحد للكل.
CLINICS_BY_CITY = {
    "القاهرة": {"name": "مركز القاهرة للأمراض الجلدية", "address": "شارع التحرير، الدقي، الجيزة", "phone": "01011112222"},
    "الجيزة": {"name": "عيادات الجيزة التخصصية", "address": "شارع الهرم، الجيزة", "phone": "01022223333"},
    "الإسكندرية": {"name": "مركز الإسكندرية لطب الجلدية", "address": "شارع فؤاد، محطة الرمل، الإسكندرية", "phone": "01033334444"},
    "المنصورة": {"name": "عيادة المنصورة الجلدية", "address": "شارع الجمهورية، المنصورة", "phone": "01044445555"},
    "أسوان": {"name": "مركز أسوان الطبي التخصصي", "address": "كورنيش النيل، أسوان", "phone": "01055556666"},
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

# دور المستخدم الحالي (يُحدَّث من app.py قبل كل استدعاء للـ agent)
_CURRENT_ROLE = None  # "doctor" | "patient" | None

DATA_PATH = Path(__file__).parent / "data" / "updated_file_2.xlsx"


def set_current_role(role: str) -> None:
    """يُستدعى من app.py قبل تشغيل الـ agent عشان أدوات الداتا تعرف تفرض الصلاحيات الصح."""
    global _CURRENT_ROLE
    _CURRENT_ROLE = role


def _load_data():
    global _PATIENT_DF, _SUMMARY_DF, _RELATIONS_DF
    if _PATIENT_DF is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Excel file not found: {DATA_PATH}")

        _PATIENT_DF   = pd.read_excel(DATA_PATH, sheet_name="Patients")
        _SUMMARY_DF   = pd.read_excel(DATA_PATH, sheet_name="Summary")
        _RELATIONS_DF = pd.read_excel(DATA_PATH, sheet_name="Family_Relationships")


# =========================================================
# دوال "خام" قابلة لإعادة الاستخدام مباشرة من app.py
# (بدون المرور على الـ LLM) لضمان دقة التقارير الطبية 100%
# =========================================================

def get_patient_record(patient_id: str) -> dict | None:
    """يرجع dict ببيانات المريض أو None لو مش موجود."""
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
    r = get_patient_record(patient_id)
    if r is None:
        return f"Patient {patient_id} not found."
    return (
        f"Patient {patient_id} — {r['name']}, {r['age']} y/o {r['gender']}, {r['country']}\n"
        f"Diagnosis: {r['diagnosis']} | Biopsy: {r['biopsy_result']}\n"
        f"Lesion: {r['lesion_size_mm']}mm, {r['lesion_color']}, location: {r['lesion_location']}\n"
        f"Border irregularity: {r['border_irregularity']:.2f} | Asymmetry: {r['asymmetry']:.2f}\n"
        f"Skin type: {r['skin_type_fitzpatrick']} | UV exposure: {r['UV_exposure_level']}\n"
        f"Genetic mutation: {r['genetic_mutation']} | Hereditary risk: {r['hereditary_risk_score']:.3f}\n"
        f"Family history: {r['family_history_skin_cancer']} | Immunosuppressed: {r['immunosuppressed']}"
    )


def format_family_text(family_id: str) -> str:
    _load_data()
    r = get_family_record(family_id)
    if r is None:
        return f"Family {family_id} not found."
    fid = family_id.strip().upper()
    members = _PATIENT_DF[_PATIENT_DF["family_id"] == fid][
        ["patient_id", "name", "age", "diagnosis", "biopsy_result"]
    ]
    return (
        f"Family {fid}: {r['total_members']} members, {r['affected_members']} affected\n"
        f"Avg risk score: {r['avg_risk_score']:.3f} | Mutation: {r['genetic_mutation']}\n"
        f"Dominant diagnosis: {r['dominant_diagnosis']} | Risk level: {r['family_risk_level']}\n\n"
        f"Members:\n{members.to_string(index=False)}"
    )


def build_medical_report(patient_id: str) -> str | None:
    """يبني تقرير طبي + ملخص إحالة + تاريخ وراثي كامل لمريض معروف. None لو غير موجود."""
    r = get_patient_record(patient_id)
    if r is None:
        return None

    family_block = ""
    fid = r.get("family_id")
    if fid:
        fam = get_family_record(str(fid))
        if fam is not None:
            family_block = (
                f"\n**🧬 التاريخ الوراثي (عائلة {fid}):**\n"
                f"- عدد أفراد العائلة: {fam['total_members']} | المصابون: {fam['affected_members']}\n"
                f"- متوسط درجة الخطورة الوراثية: {fam['avg_risk_score']:.3f}\n"
                f"- الطفرة الجينية السائدة: {fam['genetic_mutation']}\n"
                f"- مستوى خطورة العائلة: **{fam['family_risk_level']}**\n"
            )

    urgency = "عاجلة 🔴" if str(r["diagnosis"]).lower() not in ("benign", "nevus") else "روتينية 🟢"
    border_val = f"{float(r['border_irregularity']):.2f}"
    asym_val = f"{float(r['asymmetry']):.2f}"
    risk_val = f"{float(r['hereditary_risk_score']):.3f}"

    lines = [
        f"### التقرير الطبي — المريض {patient_id}",
        "",
        f"**البيانات الأساسية:** {r['name']}, {r['age']} سنة, {r['gender']}, {r['country']}",
        "",
        f"**التشخيص:** {r['diagnosis']} | **نتيجة الخزعة:** {r['biopsy_result']}",
        "",
        f"**الآفة الجلدية:** {r['lesion_size_mm']} مم | اللون: {r['lesion_color']} | الموضع: {r['lesion_location']}",
        f"**عدم انتظام الحواف:** {border_val} | **عدم التماثل:** {asym_val}",
        "",
        f"**نوع البشرة (Fitzpatrick):** {r['skin_type_fitzpatrick']} | **التعرض للأشعة فوق البنفسجية:** {r['UV_exposure_level']}",
        "",
        f"**الطفرة الجينية:** {r['genetic_mutation']} | **درجة الخطورة الوراثية:** {risk_val}",
        "",
        f"**تاريخ عائلي للإصابة:** {r['family_history_skin_cancer']} | **مثبط مناعة:** {r['immunosuppressed']}",
        family_block,
        f"**ملخص الإحالة:** الحالة مسجّلة مسبقًا في قاعدة البيانات وتمت مطابقة الصورة بنجاح. درجة أولوية المتابعة: **{urgency}**.",
    ]
    return "\n".join(lines)


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
        return (
            "🚫 Access denied: patient-level database lookups are restricted to "
            "verified clinicians (DOCTOR mode). Please identify yourself as a doctor "
            "to unlock this."
        )

    _load_data()
    df = _PATIENT_DF
    summary = _SUMMARY_DF
    q = query.lower()

    pid_match = re.search(r'p\d{4}', q)
    if pid_match:
        return format_patient_text(pid_match.group().upper())

    fid_match = re.search(r'f\d{3}', q)
    if fid_match:
        return format_family_text(fid_match.group().upper())

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

    return (
        f"Dataset: {len(df)} patients, 8 diagnoses. "
        "Ask about a patient (P0042), family (F001), diagnosis counts, or statistics."
    )


@tool
def get_patient_image(patient_id: str) -> str:
    """
    [DOCTOR MODE ONLY] Simple image retrieval tool. Retrieves and downloads a patient's
    medical image without analysis. Stores patient context for follow-up questions.
    
    Returns the local image path. No vision model analysis is performed.
    
    Usage: 'get_patient_image P0042' or 'show me the image of patient P0042'
    """
    if _CURRENT_ROLE != "doctor":
        return (
            "🚫 Access denied: Patient image retrieval is restricted to verified clinicians "
            "(DOCTOR mode). Please identify yourself as a doctor to use this feature."
        )
    
    try:
        # Get patient record from Excel
        patient_record = get_patient_record(patient_id)
        if patient_record is None:
            return f"❌ Patient {patient_id} not found in database."
        
        # Try to get image URL from patient record
        image_url = patient_record.get("image_url")
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            logger.warning(f"No image URL for patient {patient_id}")
            return (
                f"❌ No image URL found for patient {patient_id}.\n"
                f"Patient exists but no medical image is available in the database."
            )
        
        logger.info(f"Retrieving image for {patient_id} from: {image_url[:60]}...")
        
        # Download or retrieve cached image
        image_path = get_or_download_image(image_url)
        if image_path is None:
            logger.warning(f"Failed to download image for {patient_id}")
            return (
                f"❌ Failed to download image for patient {patient_id}.\n"
                f"Network error or invalid URL: {image_url[:80]}"
            )
        
        logger.info(f"Image retrieved successfully: {image_path}")
        
        # Store patient context for follow-up questions
        _store_patient_context(patient_id, str(image_path), dict(patient_record))
        _store_current_images([str(image_path)])

        return (
            f"✅ Image retrieved for patient {patient_id}\n\n"
            f"👤 Patient: {patient_record.get('name', 'N/A')}, "
            f"{patient_record.get('age', 'N/A')} y/o\n"
            f"📋 Diagnosis: {patient_record.get('diagnosis', 'N/A')}\n"
            f"📝 Biopsy Result: {patient_record.get('biopsy_result', 'N/A')}\n\n"
            f"Image is ready for analysis. Ask me to analyze it for lesion details or ask any questions about this patient."
        )
    
    except Exception as e:
        logger.error(f"Error in get_patient_image: {e}", exc_info=True)
        return f"❌ Error retrieving image: {str(e)}"


@tool
def lookup_patient_with_image(patient_id: str) -> str:
    """
    [DOCTOR MODE ONLY] Retrieve patient profile with AI image analysis using local models.
    Downloads the patient's image and runs the classification (Malignant/Benign)
    and segmentation (affected area) models. Returns results with original image,
    mask, overlay, diagnosis, confidence, and infection percentage.
    
    Use this when user explicitly asks for image analysis:
    - "Analyze the image of patient P0042"
    - "What do you see in this patient's image?"
    - "Examine the lesion in P0042's image"
    
    Stores patient context for follow-up questions.
    
    Usage: 'analyze the image of patient P0042'
    """
    if _CURRENT_ROLE != "doctor":
        return (
            "🚫 Access denied: Patient data access is restricted to verified clinicians "
            "(DOCTOR mode). Please identify yourself as a doctor to use this feature."
        )
    
    try:
        patient_record = get_patient_record(patient_id)
        if patient_record is None:
            return f"❌ Patient {patient_id} not found in database."
        patient_text = format_patient_text(patient_id)
        image_url = patient_record.get("image_url")
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            logger.info(f"No image URL for patient {patient_id}")
            return (
                f"✅ Patient record found for {patient_id}.\n\n"
                f"{patient_text}\n\n"
                f"⚠️ Note: No medical image URL available for analysis."
            )
        logger.info(f"Downloading image for {patient_id} from: {image_url[:60]}...")
        image_path = get_or_download_image(image_url)
        if image_path is None:
            logger.warning(f"Failed to download image for {patient_id}")
            return (
                f"✅ Patient record found (text data available).\n\n"
                f"{patient_text}\n\n"
                f"⚠️ Medical image could not be downloaded (network error or invalid URL)."
            )
        logger.info(f"Image downloaded successfully: {image_path}")
        _store_patient_context(patient_id, str(image_path), dict(patient_record))

        # Run local classification and segmentation models
        image = Image.open(image_path).convert("RGB")
        clf_result = _classify_image(image)
        seg_result = _segment_image(image)
        if not clf_result and not seg_result:
            return (
                f"✅ Image retrieved for patient {patient_id}\n\n"
                f"{patient_text}\n\n"
                f"⚠️ AI models are not loaded yet. Image and patient data displayed above."
            )

        stored_images = [str(image_path)]
        if seg_result:
            mask_path = _save_image_to_cache(seg_result["display_mask"], f"{patient_id}_mask.png")
            overlay_path = _save_image_to_cache(seg_result["blended"], f"{patient_id}_overlay.png")
            stored_images.extend([mask_path, overlay_path])
        _store_current_images(stored_images)

        result_data = {
            "patient_id": patient_id,
            "label": clf_result["label"],
            "confidence_pct": clf_result["confidence_pct"],
            "infection_pct": seg_result["infection_pct"] if seg_result else 0,
        }

        import json as _json
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"

        lines = [marker, ""]
        lines.append(f"**📋 PATIENT PROFILE WITH ANALYSIS — {patient_id}**\n")
        lines.append("")
        if clf_result:
            lines.append(f"**Classification:** {clf_result['label']} (Confidence: {clf_result['confidence_pct']}%)")
        if seg_result:
            lines.append(f"**Affected Area:** {seg_result['infection_pct']}%")
        lines.append("")
        lines.append(f"**Patient Data:**\n{patient_text}")
        return "\n".join(lines)
    
    except Exception as e:
        logger.error(f"Error in lookup_patient_with_image: {e}", exc_info=True)
        return f"❌ Error retrieving patient data: {str(e)}"


@tool
def classify_patient_image(patient_id: str) -> str:
    """
    [DOCTOR MODE ONLY] Classify a patient's skin lesion image using the AI classification model.
    Downloads the patient's image and runs the malignant/benign CNN model.
    Returns the classification result (Malignant/Benign) with confidence percentage,
    and displays the original image.

    Usage: 'classify patient P0042' or 'classify the image of patient P0042'
    """
    if _CURRENT_ROLE != "doctor":
        return (
            "🚫 Access denied: Patient image classification is restricted to verified clinicians "
            "(DOCTOR mode). Please identify yourself as a doctor to use this feature."
        )
    try:
        patient_record = get_patient_record(patient_id)
        if patient_record is None:
            return f"❌ Patient {patient_id} not found in database."
        image_url = patient_record.get("image_url")
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            return (
                f"✅ Patient record found for {patient_id}.\n\n"
                f"{format_patient_text(patient_id)}\n\n"
                f"⚠️ No medical image URL available for classification."
            )
        image_path = get_or_download_image(image_url)
        if image_path is None:
            return (
                f"✅ Patient record found.\n\n"
                f"{format_patient_text(patient_id)}\n\n"
                f"⚠️ Medical image could not be downloaded."
            )
        _store_patient_context(patient_id, str(image_path), dict(patient_record))
        _store_current_images([str(image_path)])
        image = Image.open(image_path).convert("RGB")
        clf_result = _classify_image(image)
        if not clf_result:
            return "❌ Classification model is not loaded yet. Try again after server startup."

        import json as _json
        result_data = {
            "patient_id": patient_id,
            "label": clf_result["label"],
            "confidence_pct": clf_result["confidence_pct"],
            "infection_pct": 0,
        }
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"

        return (
            f"{marker}\n\n"
            f"**🔬 CLASSIFICATION RESULT — Patient {patient_id}**\n\n"
            f"**Diagnosis:** {clf_result['label']}\n"
            f"**Confidence:** {clf_result['confidence_pct']}%\n\n"
            f"**Patient Data:**\n"
            f"{format_patient_text(patient_id)}"
        )
    except Exception as e:
        logger.error(f"Error in classify_patient_image: {e}", exc_info=True)
        return f"❌ Error classifying image: {str(e)}"


@tool
def segment_patient_image(patient_id: str) -> str:
    """
    [DOCTOR MODE ONLY] Segment a patient's skin lesion image using the U-Net segmentation model.
    Downloads the patient's image and runs the segmentation model to identify affected areas.
    Returns the infection percentage and displays the original image, mask, and overlay.

    Usage: 'segment patient P0042' or 'segment the image of patient P0042'
    """
    if _CURRENT_ROLE != "doctor":
        return (
            "🚫 Access denied: Patient image segmentation is restricted to verified clinicians "
            "(DOCTOR mode). Please identify yourself as a doctor to use this feature."
        )
    try:
        patient_record = get_patient_record(patient_id)
        if patient_record is None:
            return f"❌ Patient {patient_id} not found in database."
        image_url = patient_record.get("image_url")
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            return (
                f"✅ Patient record found for {patient_id}.\n\n"
                f"{format_patient_text(patient_id)}\n\n"
                f"⚠️ No medical image URL available for segmentation."
            )
        image_path = get_or_download_image(image_url)
        if image_path is None:
            return (
                f"✅ Patient record found.\n\n"
                f"{format_patient_text(patient_id)}\n\n"
                f"⚠️ Medical image could not be downloaded."
            )
        _store_patient_context(patient_id, str(image_path), dict(patient_record))
        image = Image.open(image_path).convert("RGB")
        seg_result = _segment_image(image)
        if not seg_result:
            return "❌ Segmentation model is not loaded yet. Try again after server startup."

        mask_path = _save_image_to_cache(seg_result["display_mask"], f"{patient_id}_mask.png")
        overlay_path = _save_image_to_cache(seg_result["blended"], f"{patient_id}_overlay.png")
        _store_current_images([str(image_path), mask_path, overlay_path])

        import json as _json
        result_data = {
            "patient_id": patient_id,
            "label": "",
            "confidence_pct": 0,
            "infection_pct": seg_result["infection_pct"],
        }
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"

        return (
            f"{marker}\n\n"
            f"**🔬 SEGMENTATION RESULT — Patient {patient_id}**\n\n"
            f"**Affected Area:** {seg_result['infection_pct']}%\n\n"
            f"**Patient Data:**\n"
            f"{format_patient_text(patient_id)}"
        )
    except Exception as e:
        logger.error(f"Error in segment_patient_image: {e}", exc_info=True)
        return f"❌ Error segmenting image: {str(e)}"


@tool
def analyze_patient_image(patient_id: str) -> str:
    """
    [DOCTOR MODE ONLY] Complete AI analysis of a patient's skin lesion image.
    Downloads the patient's image and runs BOTH the classification model (Malignant/Benign)
    and the segmentation model (affected area). Returns the complete analysis with
    original image, mask, overlay, diagnosis, confidence, and infection percentage.

    Usage: 'analyze patient P0042 image' or 'full analysis of patient P0042'
    """
    if _CURRENT_ROLE != "doctor":
        return (
            "🚫 Access denied: Patient image analysis is restricted to verified clinicians "
            "(DOCTOR mode). Please identify yourself as a doctor to use this feature."
        )
    try:
        patient_record = get_patient_record(patient_id)
        if patient_record is None:
            return f"❌ Patient {patient_id} not found in database."
        image_url = patient_record.get("image_url")
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            return (
                f"✅ Patient record found for {patient_id}.\n\n"
                f"{format_patient_text(patient_id)}\n\n"
                f"⚠️ No medical image URL available for analysis."
            )
        image_path = get_or_download_image(image_url)
        if image_path is None:
            return (
                f"✅ Patient record found.\n\n"
                f"{format_patient_text(patient_id)}\n\n"
                f"⚠️ Medical image could not be downloaded."
            )
        _store_patient_context(patient_id, str(image_path), dict(patient_record))
        image = Image.open(image_path).convert("RGB")
        clf_result = _classify_image(image)
        seg_result = _segment_image(image)
        if not clf_result and not seg_result:
            return "❌ Neither classification nor segmentation model is loaded yet."

        stored_images = [str(image_path)]
        if seg_result:
            mask_path = _save_image_to_cache(seg_result["display_mask"], f"{patient_id}_mask.png")
            overlay_path = _save_image_to_cache(seg_result["blended"], f"{patient_id}_overlay.png")
            stored_images.extend([mask_path, overlay_path])
        _store_current_images(stored_images)

        import json as _json
        result_data = {
            "patient_id": patient_id,
            "label": clf_result.get("label", ""),
            "confidence_pct": clf_result.get("confidence_pct", 0),
            "infection_pct": seg_result["infection_pct"] if seg_result else 0,
        }
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"

        lines = [marker, ""]
        lines.append(f"**📋 COMPLETE ANALYSIS — Patient {patient_id}**\n")
        lines.append("")
        if clf_result:
            lines.append(f"**Classification:** {clf_result['label']} (Confidence: {clf_result['confidence_pct']}%)")
        if seg_result:
            lines.append(f"**Affected Area:** {seg_result['infection_pct']}%")
        lines.append("")
        lines.append(f"**Patient Data:**\n{format_patient_text(patient_id)}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error in analyze_patient_image: {e}", exc_info=True)
        return f"❌ Error analyzing image: {str(e)}"


@tool
def get_my_uploaded_image() -> str:
    """
    [PATIENT MODE] Tell the patient their image is stored and available.
    Use when the patient asks: "Show my uploaded image", "Return my last image".
    Just confirm the image exists and tell them to use the Show My Image button.
    """
    if _CURRENT_ROLE != "patient":
        return "🚫 This tool is only available in patient mode."
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return "No session found."
    from patient_storage import get_latest_image
    latest = get_latest_image(_CURRENT_THREAD_ID)
    if not latest:
        return (
            "You haven't uploaded any images yet. Use the 📎 upload button "
            "above to upload a skin image for analysis."
        )
    return (
        "Your image is saved on the server. Click the 📷 **Show My Image** button "
        "above to view it. The button appears below the analysis results."
    )


@tool
def segment_my_image() -> str:
    """
    [PATIENT MODE] Run ONLY the U-Net segmentation model on the patient's stored image.
    Use when the patient asks: "Show me the segmentation", "Segment my stored image",
    "What is the affected area", "Show the mask", "Show the overlay".
    Runs the U-Net model and returns the mask and overlay images with infection percentage.
    The result images appear in the UI automatically.
    """
    if _CURRENT_ROLE != "patient":
        return "🚫 This tool is only available in patient mode."
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return "No session found."
    from patient_storage import get_latest_image
    latest = get_latest_image(_CURRENT_THREAD_ID)
    if not latest:
        return "You haven't uploaded any images yet. Use the 📎 upload button above to upload a skin image for analysis."

    image_path_str = latest["path"]
    image_path = __import__("pathlib").Path(image_path_str)
    if not image_path.exists():
        return "Your stored image file is no longer available. Please upload again."

    try:
        from PIL import Image as _PIL
        image = _PIL.open(image_path).convert("RGB")
        seg_result = _segment_image(image)

        if not seg_result:
            return "⚠️ Segmentation model is not loaded yet. Please try again later."

        mask_path = _save_image_to_cache(seg_result["display_mask"], f"patient_mask.png")
        overlay_path = _save_image_to_cache(seg_result["blended"], f"patient_overlay.png")
        _store_current_images([mask_path, overlay_path])

        import json as _json
        result_data = {
            "label": "",
            "confidence_pct": 0,
            "infection_pct": seg_result.get("infection_pct", 0),
        }
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"

        lines = [marker, ""]
        lines.append("**🎭 SEGMENTATION OF YOUR IMAGE**\n")
        lines.append(f"**Affected Area:** {seg_result['infection_pct']}%")
        lines.append("")
        lines.append("The mask and overlay images appear below.")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Error in segment_my_image: {e}", exc_info=True)
        return f"❌ Error segmenting image: {str(e)}"


@tool
def reanalyze_my_image() -> str:
    """
    [PATIENT MODE] Run FULL AI analysis (classification + segmentation) on the patient's
    most recently uploaded image. Use when the patient asks: "Re-analyze my image",
    "Analyze my previous image", "What were my results?", "Full analysis".
    Runs BOTH the classification (Malignant/Benign) and U-Net segmentation models.
    Displays the original image, mask, and overlay with infection percentage.
    """
    if _CURRENT_ROLE != "patient":
        return "🚫 This tool is only available in patient mode."
    if not _SESSIONS or not _CURRENT_THREAD_ID:
        return "No session found."
    from patient_storage import get_latest_image
    latest = get_latest_image(_CURRENT_THREAD_ID)
    if not latest:
        return "You haven't uploaded any images yet. Use the 📎 upload button above to upload a skin image for analysis."

    image_path_str = latest["path"]
    image_path = __import__("pathlib").Path(image_path_str)
    if not image_path.exists():
        return "Your stored image file is no longer available. Please upload again."

    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        clf_result = _classify_image(image)
        seg_result = _segment_image(image)

        if not clf_result and not seg_result:
            return "⚠️ AI models are not loaded yet. Please try again later."

        stored_images = []
        if seg_result:
            mask_path = _save_image_to_cache(seg_result["display_mask"], f"patient_mask.png")
            overlay_path = _save_image_to_cache(seg_result["blended"], f"patient_overlay.png")
            stored_images.extend([mask_path, overlay_path])
        _store_current_images(stored_images)

        result_data = {
            "label": clf_result.get("label", ""),
            "confidence_pct": clf_result.get("confidence_pct", 0),
            "infection_pct": seg_result.get("infection_pct", 0) if seg_result else 0,
        }

        import json as _json
        marker = f"[ANALYSIS_RESULT:{_json.dumps(result_data)}]"

        lines = [marker, ""]
        lines.append("**🔬 ANALYSIS OF YOUR IMAGE**\n")
        if clf_result:
            lines.append(f"**Classification:** {clf_result['label']} (Confidence: {clf_result['confidence_pct']}%)")
        if seg_result:
            lines.append(f"**Affected Area:** {seg_result['infection_pct']}%")
        lines.append("")
        lines.append("The mask and overlay images appear below.")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error in reanalyze_my_image: {e}", exc_info=True)
        return f"❌ Error analyzing image: {str(e)}"


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
        f"Referral created.\n"
        f"  Ticket ID : {ticket_id}\n"
        f"  Urgency   : {urgency.upper()}\n"
        f"  Timeframe : {timeframe}\n"
        f"  Summary   : {issue_summary}\n"
        f"  Clinic    : {DEFAULT_CLINIC['name']} — {DEFAULT_CLINIC['address']}\n"
        f"A dermatologist will contact you {timeframe}."
    )


@tool
def book_appointment(priority_degree: int, reason: str, city: str = "") -> str:
    """Book a patient appointment given a priority degree (1=highest/emergency,
    2=urgent, 3=routine) and the patient's city, to get a concrete date/time,
    room number, doctor name, and a nearby clinic.
    NOTE: this is invoked directly by the booking button in the UI, not by the
    chat LLM (removed from the agent's toolset — see build_agent)."""
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

    city_note = "" if (city and city.strip() in CLINICS_BY_CITY) else (
        f"\n  ⚠️ '{city}' مش من المدن اللي عندنا فروع مسجّلة فيها حاليًا، "
        f"فتم اختيار أقرب فرع افتراضي." if city else ""
    )

    return (
        f"✅ تم الحجز بنجاح.\n"
        f"  رقم التذكرة : {ticket_id}\n"
        f"  الأولوية    : {label_ar}\n"
        f"  الميعاد     : {_format_arabic_datetime(slot)}\n"
        f"  الدكتور     : {doctor}\n"
        f"  الغرفة      : غرفة رقم {room}\n"
        f"  العيادة     : {clinic['name']}\n"
        f"  العنوان     : {clinic['address']}\n"
        f"  التليفون    : {clinic['phone']}\n"
        f"  السبب       : {reason}{city_note}\n"
        f"  ⚠️ ملاحظة: البيانات دي تجريبية (Demo) — لسه مش متربطة بنظام حجز مستشفيات حقيقي أو خرائط فعلية."
    )


# =========================================================
# دالة حساب أولوية الحجز (خارج LangChain — دقيقة ومباشرة)
# =========================================================

def compute_booking_priority(label: str, confidence_pct: float, infection_pct: float):
    """
    يحسب درجة أولوية الحجز (1/2/3) بناءً على severity score بسيط:
    مزيج من نسبة ثقة التصنيف الخبيث ونسبة المساحة المصابة.
    يرجع (degree:int, label_ar:str, urgency:str, severity_score:float)
    """
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

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=1024,
        request_timeout=120,
    )

    # ملاحظة: book_appointment اتشالت من هنا عمدًا. الحجز بيحصل بس عن طريق
    # زرار "احجز موعد الآن" في الواجهة (استدعاء مباشر للدالة)، مش عن طريق
    # الشات، عشان الموديل ميحجزش لوحده على أي كلام عام.
    tools = [search_knowledge_base, lookup_patient, get_patient_image, lookup_patient_with_image, classify_patient_image, segment_patient_image, analyze_patient_image, get_my_uploaded_image, reanalyze_my_image, create_referral_ticket]
    memory = MemorySaver()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )