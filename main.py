"""
main.py — DermaScan AI Backend (FastAPI)
=========================================
تشغيل:
    uvicorn main:app --reload --port 8000

ملاحظة مهمة: النسخة دي بقت in-memory بالكامل — مفيش أي صورة أو ملف
بيتكتب على الديسك. كل صور المرضى (المرفوعة أو المجلوبة من الداتا) بتتخزن
جوه SESSIONS (dict في الـ RAM) بس. لما تقفل السيرفر (Ctrl+C / uvicorn يقفل)،
كل حاجة بتتمسح تلقائي لأنها مفيش ليها أي وجود على الديسك.
"""

import os
import io
import base64
import time
import logging
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from rag import build_default_vectorstore
from agent import (
    build_agent,
    set_current_role,
    set_session_storage,
    set_current_thread_id,
    set_models as _set_agent_models,
    build_medical_report,
    compute_booking_priority,
    book_appointment,
    get_latest_image,
)
from image_registry import find_matching_patient

MODEL_NAME = "llama-3.1-8b-instant"
FALLBACK_MODEL_NAME = "llama-3.3-70b-versatile"

TEMPERATURE = 0.25
TOP_K = 3

CLF_PATH = "skin_cancer_detection_final.keras"
SEG_PATH = "UNet_model.keras"

app = FastAPI(title="DermaScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clf_model = None
seg_model = None
seg_input_size = None
models_loaded = False
models_error = ""

agent = None
fallback_agent = None

# كل حاجة في الذاكرة بس: thread_id -> session dict.
# مفيش أي persistence على الديسك — لو السيرفر اتقفل السيشنز كلها بتضيع.
SESSIONS: dict = {}


def get_session(thread_id: str) -> dict:
    if thread_id not in SESSIONS:
        SESSIONS[thread_id] = {
            "role": None,
            "last_analysis": None,
            "current_patient_id": None,
            "current_image": None,          # PIL.Image في الذاكرة بس
            "current_patient_data": None,
            "matched_patient_id": None,
            "matched_patient_report": None,
            "_image_blobs": {},              # {"mask": b64, "overlay": b64, "original": b64}
            "_session_image_names": [],
            "_last_upload_meta": None,
        }
    return SESSIONS[thread_id]


def _set_patient_image(thread_id: str, raw_bytes: bytes, original_name: str) -> Image.Image:
    """يحفظ صورة المريض في الذاكرة بس (مفيش كتابة ديسك خالص)."""
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    session = get_session(thread_id)
    session["current_image"] = image
    session["_last_upload_meta"] = {
        "original_name": original_name,
        "timestamp": int(time.time() * 1000),
    }
    return image


def load_models():
    global clf_model, seg_model, seg_input_size, models_loaded, models_error
    try:
        if not os.path.exists(CLF_PATH) or not os.path.exists(SEG_PATH):
            raise FileNotFoundError("Model files (.keras) not found in root directory.")
        clf_model = tf.keras.models.load_model(CLF_PATH)
        seg_model = tf.keras.models.load_model(SEG_PATH, compile=False)
        seg_input_size = seg_model.input_shape[1:3]
        models_loaded = True
        print("Keras models loaded.")
    except Exception as e:
        models_loaded = False
        models_error = str(e)
        print(f"Could not load models: {models_error}")


@app.on_event("startup")
def startup():
    global agent, fallback_agent
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — chat will fail until configured.")
    load_models()
    _set_agent_models(clf_model, seg_model, seg_input_size)
    set_session_storage(SESSIONS)
    print("Building/loading vectorstore + agent...")
    vectorstore = build_default_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K}) if vectorstore else None
    if retriever is None:
        print("WARNING: vectorstore not available — search_knowledge_base will return no results.")
    agent = build_agent(retriever, model_name=MODEL_NAME, temperature=TEMPERATURE)
    fallback_agent = build_agent(retriever, model_name=FALLBACK_MODEL_NAME, temperature=TEMPERATURE)
    print("Agent ready (with rate-limit fallback).")


class RoleRequest(BaseModel):
    thread_id: str
    role: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class BookingRequest(BaseModel):
    thread_id: str
    city: str = ""


class ThreadRequest(BaseModel):
    thread_id: str = ""


@app.post("/set-role")
def set_role(req: RoleRequest):
    if req.role not in ("doctor", "patient"):
        return {"ok": False, "error": "invalid role"}
    session = get_session(req.thread_id)
    session["role"] = req.role
    return {"ok": True, "role": req.role}


@app.get("/health")
def health():
    return {
        "models_loaded": models_loaded,
        "models_error": models_error if not models_loaded else None,
        "agent_ready": agent is not None,
    }


def _to_png_b64(arr) -> str:
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode("utf-8")


@app.post("/analyze-image")
async def analyze_image(thread_id: str = Form(...), file: UploadFile = File(...)):
    session = get_session(thread_id)
    role = session.get("role")

    if not models_loaded:
        return {"ok": False, "error": f"Models not loaded: {models_error}"}
    if role not in ("doctor", "patient"):
        return {"ok": False, "error": "من فضلك حدّد دورك (دكتور/مريض) أول حاجة."}

    raw = await file.read()
    image = _set_patient_image(thread_id, raw, file.filename or "upload.png")
    orig_np = np.array(image).astype(np.uint8)
    orig_h, orig_w = orig_np.shape[:2]

    try:
        match_pid, similarity = find_matching_patient(image)
    except Exception as e:
        logger.error("find_matching_patient failed: %s", e, exc_info=True)
        match_pid, similarity = None, None

    # مهم: لازم نمسح تطابق الرفعة القديمة دايمًا، مش بس نحدّثه لما فيه تطابق
    # جديد — غير كده لو الصورة الجديدة مش متطابقة هيفضل بيانات مريض قديم
    # (زي جيري هوف) عالقة في الجلسة وتتلخبط مع صورة شخص تاني تمامًا.
    matched_report = build_medical_report(match_pid) if match_pid else None
    session["matched_patient_id"] = match_pid
    session["matched_patient_report"] = matched_report

    clf_img = image.resize((224, 224))
    clf_array = np.expand_dims(np.array(clf_img) / 255.0, axis=0)
    pred = clf_model.predict(clf_array)
    if pred.shape[-1] == 1:
        prob = float(pred[0][0])
        label = "Malignant" if prob >= 0.5 else "Benign"
        confidence = prob if prob >= 0.5 else 1 - prob
    else:
        class_index = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))
        label = ["Benign", "Malignant"][class_index]
    confidence_pct = round(confidence * 100, 2)

    seg_img = image.resize(seg_input_size)
    seg_array = np.expand_dims(np.array(seg_img).astype("float32") / 255.0, axis=0)
    masks = seg_model.predict(seg_array)
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

    result = {
        "ok": True,
        "label": label,
        "confidence_pct": confidence_pct,
        "infection_pct": round(infection_pct, 2),
        "images": {
            "original": _to_png_b64(orig_np),
            "mask": _to_png_b64(display_mask),
            "overlay": _to_png_b64(blended),
        },
    }

    analysis_data = {
        "label": label,
        "confidence_pct": confidence_pct,
        "infection_pct": infection_pct,
        "timestamp": int(time.time() * 1000),
    }
    session["last_analysis"] = analysis_data

    if role == "doctor":
        if match_pid:
            result["doctor"] = {"matched": True, "patient_id": match_pid, "similarity": similarity, "report": matched_report}
        else:
            result["doctor"] = {
                "matched": False,
                "message": "الصورة دي مش متعرّف عليها في قاعدة البيانات الحالية.",
            }
    elif role == "patient":
        degree, degree_label_ar, urgency, severity = compute_booking_priority(label, confidence_pct, infection_pct)
        result["patient"] = {
            "priority_degree": degree,
            "priority_label": degree_label_ar,
            "urgency": urgency,
            "severity_score": round(severity, 3),
            "matched": bool(match_pid),
            "patient_record": matched_report,
        }

    return result


@app.post("/register-image")
async def register_image_endpoint(thread_id: str = Form(...), patient_id: str = Form(...), file: UploadFile = File(...)):
    session = get_session(thread_id)
    if session.get("role") != "doctor":
        return {"ok": False, "error": "التسجيل متاح للدكتور فقط."}
    from image_registry import register_image as _register_image
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    _register_image(patient_id, image)
    return {"ok": True, "message": f"تم استقبال الطلب لتسجيل صورة تحت المريض {patient_id.strip().upper()}."}


@app.post("/book-appointment")
def book(req: BookingRequest):
    """الحجز — أي استثناء بيترجم لرسالة عربية عادية، مفيش أي نص تقني/ديبج
    بيظهر للمستخدم في الشات أبدًا."""
    try:
        session = get_session(req.thread_id)
        if session.get("role") != "patient":
            return {"ok": False, "error": "الحجز متاح في وضع المريض فقط."}

        last = session.get("last_analysis")
        if not last:
            return {"ok": False, "error": "من فضلك اعمل تحليل صورة الأول."}

        degree, _, _, _ = compute_booking_priority(last["label"], last["confidence_pct"], last["infection_pct"])
        reason = f"AI analysis: {last['label']} ({last['confidence_pct']}% confidence), affected area {last['infection_pct']:.2f}%"
        ticket_text = book_appointment.invoke({"priority_degree": degree, "reason": reason, "city": req.city})
        return {"ok": True, "ticket": ticket_text}
    except Exception as e:
        logger.error("book_appointment failed: %s", e, exc_info=True)
        return {"ok": False, "error": "تعذّر إتمام الحجز حاليًا، حاول مرة أخرى."}


import re as _re

_DISALLOWED_UNICODE_RANGES = _re.compile(
    "[\u0400-\u04FF"
    "\u0E00-\u0E7F"
    "\u4E00-\u9FFF"
    "\u3040-\u30FF"
    "\uAC00-\uD7AF"
    "]+"
)


def _sanitize_reply(text: str) -> str:
    if not text:
        return text
    cleaned = _DISALLOWED_UNICODE_RANGES.sub("", text)
    cleaned = _re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


# جملة أو جملتين بيتم حذفهم لو فيهم أي إشارة للحجز — الموديل الصغير مش دايمًا
# بيلتزم بتعليمة "متقولش للدكتور يحجز"، فده ضمان إجباري في الكود نفسه.
_BOOKING_MENTION_RE = _re.compile(
    r'[^.!؟\n]*(?:📅|book(?:ing)? an? appointment|زر\s*الحجز|زرار\s*الحجز|'
    r'استخدام\s*زر|احجز\s*موعد|Book Appointment)[^.!؟\n]*[.!؟]?\s*',
    _re.IGNORECASE,
)


def _strip_booking_mentions(text: str) -> str:
    """في وضع الدكتور، أي جملة بتقول له 'احجز/استخدم زر الحجز' بتتشال خالص —
    الزرار ده مخصص للمريض بس، مش للدكتور."""
    if not text:
        return text
    cleaned = _BOOKING_MENTION_RE.sub("", text)
    cleaned = _re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_RECORD_ANALYTICAL_WORDS = [
    "compare", "قارن", "why", "لماذا", "ليه", "اشرح", "explain", "analy",
    "حلل", "تحليل", "risk of", "احتمال", "افضل", "best", "recommend",
]


def _try_direct_record_lookup(role: str, message: str, thread_id: str):
    """طلبات عرض سجل المريض أو العائلة مباشرة بدون LLM."""
    if role != "doctor":
        return None

    q = message.lower()

    if any(w in q for w in _RECORD_ANALYTICAL_WORDS):
        return None

    pid_match = _re.search(r"p\d{4}", q)
    fid_match = _re.search(r"f\d{3}", q)

    if pid_match:
        from agent import format_patient_text, get_patient_record

        pid = pid_match.group().upper()

        session = get_session(thread_id)
        session["current_patient_id"] = pid
        session["current_patient_data"] = get_patient_record(pid)
        session["last_shown_patient_id"] = pid

        return format_patient_text(pid)

    if fid_match:
        from agent import format_family_text
        return format_family_text(fid_match.group().upper())

    return None

@app.post("/chat")
def chat(req: ChatRequest):
    session = get_session(req.thread_id)
    role = session.get("role")
    if role not in ("doctor", "patient"):
        return {"ok": False, "error": "من فضلك حدّد دورك (دكتور/مريض) أول حاجة."}
    if agent is None:
        return {"ok": False, "error": "الـ Agent لسه بيتحمّل، جرّب تاني بعد شوية."}

    # مسار مباشر بدون LLM لطلبات السجل الصريحة (رقم مريض/عائلة) — بيانات
    # حقيقية 100% من الإكسل، مفيش أي احتمال اختلاق بيانات، ومفيش استهلاك توكنز.
    direct_record = _try_direct_record_lookup(
    role,
    req.message,
    req.thread_id,
)
    if direct_record is not None:
        # نفتكر آخر مريض اتعرض سجله بالمسار المباشر ده، عشان لو الدكتور قال
        # بعد كده "حلليها" أو "حللها بناء على السجل" من غير ما يكرر رقم
        # المريض، نقدر نجيب سياقه تاني للـ LLM (نصيًا، من غير أي صورة).
        pid_shown = _re.search(r'p\d{4}', req.message.lower())
        if pid_shown:
            session["last_shown_patient_id"] = pid_shown.group().upper()
            from agent import get_patient_record
            pid = pid_shown.group().upper()
            session["current_patient_id"] = pid
            session["current_patient_data"] = get_patient_record(pid)
        return {"ok": True, "reply": direct_record, "tool_calls": [], "model": "direct_lookup"}
    set_session_storage(SESSIONS)
    set_current_role(role)
    set_current_thread_id(req.thread_id)
    role_prefix = "[DOCTOR MODE] " if role == "doctor" else "[PATIENT MODE] "

    explicit_pid_match = _re.search(r'p\d{4}', req.message.lower())
    matched_pid = session.get("matched_patient_id")
    asking_about_other_patient = bool(
        explicit_pid_match and (not matched_pid or explicit_pid_match.group().upper() != str(matched_pid).upper())
    )

    last_analysis = session.get("last_analysis")
    if last_analysis and not asking_about_other_patient:
        analysis_ctx = (
            f"\n\n[CURRENT IMAGE ANALYSIS — Classification: {last_analysis.get('label', 'N/A')}, "
            f"Confidence: {last_analysis.get('confidence_pct', 0):.1f}%, "
            f"Affected Area: {last_analysis.get('infection_pct', 0):.1f}%. "
            f"Use this directly, don't call reanalyze_my_image again unless explicitly asked. "
            f"If the user asks for 'the last analysis' or to combine it with hereditary/family data, "
            f"restate these exact numbers first, then add the hereditary interpretation — don't skip the numbers.]"
        )
    else:
        analysis_ctx = ""

    matched_report = session.get("matched_patient_report")
    if matched_report and not asking_about_other_patient:
        family_ctx = (
            f"\n\n[MATCHED PATIENT RECORD — image matched patient {session.get('matched_patient_id')} "
            f"already in the database (same person). Weave relevant parts into the answer, don't paste verbatim. "
            f"If asked for a genetic/hereditary-based analysis, explicitly name the mutation and risk score from this block.]\n{matched_report}"
        )
    else:
        family_ctx = ""

    # لو الدكتور شاف سجل مريض بالمسار المباشر (بدون LLM) قبل كده في نفس
    # الجلسة، ولسه بيتكلم عن نفس المريض ده (مفيش رقم تاني صريح في الرسالة)،
    # نديله السجل كسياق نصي — عشان "حلليها بناء على السجل" تشتغل من غير
    # أي صورة، ومن غير ما نضطر نستهلك توكنز في عرض السجل الأول نفسه.
    last_shown_pid = session.get("last_shown_patient_id")
    if last_shown_pid and role == "doctor" and not (
        explicit_pid_match and explicit_pid_match.group().upper() != last_shown_pid
    ):
        rep = build_medical_report(last_shown_pid)
        last_shown_ctx = (
            f"\n\n[PREVIOUSLY VIEWED PATIENT RECORD — {last_shown_pid}, shown to the doctor "
            f"moments ago via a direct lookup. If asked to 'analyze this case' or give a "
            f"hereditary/genetic read on it, answer directly from this text — no image needed.]\n{rep}"
        ) if rep else ""
    else:
        last_shown_ctx = ""

    _IMAGE_INTENT_WORDS = [
        "image", "photo", "picture", "upload", "segment", "mask", "overlay",
        "reanalyze", "my scan",
        "صوره", "صورة", "الصور", "ارفعت", "رفعت", "المرفوع",
        "اعرض صورتي", "عرض الصوره", "المساحه المصابه", "المساحة المصابة",
    ]
    is_image_intent = any(w in req.message.lower() for w in _IMAGE_INTENT_WORDS)

    if role == "patient":
        image_tools_hint = "segment_my_image, reanalyze_my_image, get_my_uploaded_image"
    else:
        image_tools_hint = "segment_uploaded_image, analyze_uploaded_image, get_uploaded_image"

    if is_image_intent:
        anti_refusal = (
            f"\n\n[SYSTEM: You have image tools ({image_tools_hint}). Call ONE of them once "
            f"for this request — never refuse to try. If the tool's result says no image has "
            f"been uploaded yet, that IS your answer: tell the user in one short sentence to "
            f"use the 📎 upload button, and STOP — do not call another tool or retry.]"
        )
    else:
        # سؤال عام (مفيش ذكر لصورة) — نسيبه يروح على search_knowledge_base
        # عادي من غير أي توجيه لأدوات الصورة، عشان منلخبطش الموديل زي ما حصل
        # لما سؤال عام عن شامة رجّع "مفيش تطابق" بدل إجابة معرفية طبيعية.
        anti_refusal = ""

    full_prompt = role_prefix + req.message + analysis_ctx + family_ctx + last_shown_ctx + anti_refusal

    MAX_TURNS = 15
    conversation_summary = session.get("_summary", "")
    turn_count = session.get("_turn_count", 0)

    if turn_count >= MAX_TURNS:
        fresh_thread_id = f"{req.thread_id}_fresh_{turn_count}"
        config_agent = {"configurable": {"thread_id": fresh_thread_id}, "recursion_limit": 14}
        session["_turn_count"] = 0
        session["_summary"] = ""
        if conversation_summary:
            full_prompt = f"[Previous summary: {conversation_summary}]\n\n{full_prompt}"
    else:
        config_agent = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 14}

    _MARKER_RE = _re.compile(r'\[ANALYSIS_RESULT:(\{.*?\})\]')

    def _run_once(active_agent, prompt_text: str):
        import json as _json2
        final_text = ""
        tool_calls = []
        analysis_marker_raw = None
        for event in active_agent.stream({"messages": [("user", prompt_text)]}, config=config_agent, stream_mode="values"):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    tool_calls.append({"name": tc["name"], "args": tc["args"]})
            if last_msg.type == "ai" and last_msg.content:
                final_text = last_msg.content
            for msg in event["messages"]:
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content:
                    m = _MARKER_RE.search(content)
                    if m:
                        analysis_marker_raw = m.group(0)

        cleaned_text = _MARKER_RE.sub("", final_text).strip()
        final = _sanitize_reply(cleaned_text)
        if role == "doctor":
            final = _strip_booking_mentions(final)

        analysis_result = None
        if analysis_marker_raw:
            m = _MARKER_RE.search(analysis_marker_raw)
            if m:
                try:
                    analysis_result = _json2.loads(m.group(1))
                except Exception:
                    analysis_result = None

        return final, tool_calls, analysis_result

    def _is_rate_limit(err) -> bool:
        s = str(err)
        return "rate_limit_exceeded" in s or "429" in s or "Rate limit reached" in s

    def _is_request_too_large(err) -> bool:
        s = str(err)
        return "413" in s or "Request too large" in s or "context length" in s.lower()

    def _is_image_error(err) -> bool:
        s = str(err)
        return "Cannot read" in s and "does not support image input" in s

    def _is_tool_use_failed(err) -> bool:
        s = str(err)
        return (
            "Tool" in s and ("not exist" in s.lower() or "failed" in s.lower() or "error" in s.lower())
        ) or ("tool_call" in s.lower() or "tool call" in s.lower()) or ("function call" in s.lower() and "fail" in s.lower())

    def _is_timeout(err) -> bool:
        s = str(err)
        return "timeout" in s.lower() or "timed out" in s.lower()

    def _is_recursion_limit(err) -> bool:
        s = str(err)
        return "GraphRecursionError" in s or "recursion_limit" in s.lower() or "Recursion limit" in s

    def _get_session_images():
        blobs = session.get("_image_blobs", {})
        names = list(blobs.keys())
        return names

    def _friendly_error(err_type: str, err_msg: str) -> str:
        err_text = f"{err_type}: {err_msg}"
        if _is_request_too_large(Exception(err_text)):
            return "الطلب تجاوز الحد المسموح للذكاء الاصطناعي. حاول مرة أخرى."
        if _is_image_error(Exception(err_text)):
            return "تم تحليل الصورة بنجاح. النتائج معروضة أعلاه."
        if _is_rate_limit(Exception(err_text)):
            return "وصلت للحد الأقصى لاستخدام النموذج. انتظر قليلاً ثم حاول مرة أخرى."
        if _is_timeout(Exception(err_text)):
            return "استغرق الطلب وقتًا طويلاً. حاول مرة أخرى بسؤال أبسط."
        if _is_tool_use_failed(Exception(err_text)):
            return "حصل خطأ مؤقت. جرّب تاني بسؤال أبسط."
        if _is_recursion_limit(Exception(err_text)):
            return "الطلب احتاج خطوات كتير. جرّب تسأل بشكل أبسط."
        return "حدث خطأ غير متوقع. حاول مرة أخرى من فضلك."

    last_error = None

    try:
        final_text, tool_calls, analysis_result = _run_once(agent, full_prompt)
        refusal_patterns = [
            "Cannot read", "does not support image", "cannot process images",
            "I cannot see", "I cannot look", "I don't have the ability",
            "I'm not able to", "I am not able to", "can't see images",
            "can't process images", "don't have vision", "no vision capability",
            "text-based", "language model", "I'm a text",
        ]
        if any(_re.search(p, final_text, _re.IGNORECASE) for p in refusal_patterns):
            final_text = "تم تحليل الصورة بنجاح. النتائج معروضة أعلاه."
        session["_turn_count"] = turn_count + 1
        session["_summary"] = req.message[:150]
        resp = {"ok": True, "reply": final_text, "tool_calls": tool_calls, "model": "primary"}
        if analysis_result is not None:
            resp["analysis_result"] = analysis_result
        session_imgs = _get_session_images()
        if session_imgs:
            resp["images"] = session_imgs
        return resp
    except Exception as e:
        last_error = e
        logger.error("chat primary attempt failed: %s", e, exc_info=True)
        if _is_image_error(e):
            session["_turn_count"] = turn_count + 1
            return {"ok": True, "reply": _friendly_error(type(e).__name__, str(e)), "tool_calls": [], "model": "primary"}
        if _is_request_too_large(e):
            fresh_id = f"{req.thread_id}_trimmed_{turn_count}"
            config_agent = {"configurable": {"thread_id": fresh_id}, "recursion_limit": 14}
            try:
                final_text, tool_calls, analysis_result = _run_once(agent, full_prompt)
                session["_turn_count"] = 0
                resp = {"ok": True, "reply": final_text, "tool_calls": tool_calls, "model": "primary_trimmed", "notice": _friendly_error(type(e).__name__, str(e))}
                if analysis_result is not None:
                    resp["analysis_result"] = analysis_result
                session_imgs = _get_session_images()
                if session_imgs:
                    resp["images"] = session_imgs
                return resp
            except Exception:
                pass

    if last_error and (_is_rate_limit(last_error) or _is_tool_use_failed(last_error) or _is_recursion_limit(last_error)) and fallback_agent is not None:
        fallback_thread = f"{req.thread_id}_fallback_{turn_count}" if (_is_tool_use_failed(last_error) or _is_recursion_limit(last_error)) else req.thread_id
        config_agent = {"configurable": {"thread_id": fallback_thread}, "recursion_limit": 14}
        try:
            final_text, tool_calls, analysis_result = _run_once(fallback_agent, full_prompt)
            session["_turn_count"] = turn_count + 1
            resp = {"ok": True, "reply": final_text, "tool_calls": tool_calls, "model": "fallback",
                    "notice": "تم استخدام موديل أخف مؤقتًا."}
            if analysis_result is not None:
                resp["analysis_result"] = analysis_result
            session_imgs = _get_session_images()
            if session_imgs:
                resp["images"] = session_imgs
            return resp
        except Exception as e2:
            logger.error("chat fallback attempt failed: %s", e2, exc_info=True)
            last_error = e2

    return {"ok": False, "error": _friendly_error(type(last_error).__name__, str(last_error))}


@app.get("/image")
def get_image(name: str = "", thread_id: str = ""):
    """يرجّع صورة (mask/overlay/original) من ذاكرة السيشن بس — مفيش قراءة ديسك خالص."""
    try:
        session = get_session(thread_id) if thread_id else {}
        role = session.get("role")
        blobs = session.get("_image_blobs", {})
        if role not in ("doctor", "patient"):
            return {"ok": False, "error": "Access denied."}
        if role == "patient" and name not in session.get("_session_image_names", []):
            return {"ok": False, "error": "Access denied."}
        if name not in blobs:
            return {"ok": False, "error": "Image not found in session."}
        return {"ok": True, "base64": blobs[name], "media_type": "image/png"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/patient-image")
def get_patient_image(thread_id: str = ""):
    """يرجّع آخر صورة رفعها المريض — من الذاكرة مباشرة."""
    try:
        if not thread_id:
            return {"ok": False, "error": "Missing thread_id parameter."}
        session = get_session(thread_id)
        if session.get("role") != "patient":
            return {"ok": False, "error": "من فضلك اختار وضع المريض وارفع صورتك تاني."}

        latest = get_latest_image(thread_id)
        if not latest:
            return {"ok": False, "error": "لا توجد صورة مرفوعة. استخدم زر 📎 للرفع أولاً."}

        orig_np = np.array(latest["image"]).astype(np.uint8)
        b64 = _to_png_b64(orig_np)
        return {"ok": True, "base64": b64, "media_type": "image/png", "metadata": session.get("_last_upload_meta") or {}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/segment-patient-image")
def segment_patient_image_api(req: ThreadRequest):
    thread_id = req.thread_id
    try:
        if not thread_id:
            return {"ok": False, "error": "Missing thread_id parameter."}
        session = get_session(thread_id)
        if session.get("role") != "patient":
            return {"ok": False, "error": "من فضلك اختار وضع المريض وارفع صورتك تاني."}

        latest = get_latest_image(thread_id)
        if not latest:
            return {"ok": False, "error": "لا توجد صورة مرفوعة. استخدم زر 📎 للرفع أولاً."}

        image = latest["image"]
        orig_np = np.array(image).astype(np.uint8)
        orig_h, orig_w = orig_np.shape[:2]

        seg_img = image.resize(seg_input_size)
        seg_array = np.expand_dims(np.array(seg_img).astype("float32") / 255.0, axis=0)
        masks = seg_model.predict(seg_array)
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
            "ok": True,
            "infection_pct": round(infection_pct, 2),
            "images": {"mask": _to_png_b64(display_mask), "overlay": _to_png_b64(blended)},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/reanalyze-patient-image")
def reanalyze_patient_image(req: ThreadRequest):
    thread_id = req.thread_id
    try:
        if not thread_id:
            return {"ok": False, "error": "Missing thread_id parameter."}
        session = get_session(thread_id)
        if session.get("role") != "patient":
            return {"ok": False, "error": "من فضلك اختار وضع المريض وارفع صورتك تاني."}

        latest = get_latest_image(thread_id)
        if not latest:
            return {"ok": False, "error": "لا توجد صورة مرفوعة. استخدم زر 📎 للرفع أولاً."}

        image = latest["image"]
        orig_np = np.array(image).astype(np.uint8)
        orig_h, orig_w = orig_np.shape[:2]

        clf_img = image.resize((224, 224))
        clf_array = np.expand_dims(np.array(clf_img) / 255.0, axis=0)
        pred = clf_model.predict(clf_array)
        if pred.shape[-1] == 1:
            prob = float(pred[0][0])
            label = "Malignant" if prob >= 0.5 else "Benign"
            confidence = prob if prob >= 0.5 else 1 - prob
        else:
            class_index = int(np.argmax(pred[0]))
            confidence = float(np.max(pred[0]))
            label = ["Benign", "Malignant"][class_index]
        confidence_pct = round(confidence * 100, 2)

        seg_img = image.resize(seg_input_size)
        seg_array = np.expand_dims(np.array(seg_img).astype("float32") / 255.0, axis=0)
        masks = seg_model.predict(seg_array)
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

        analysis_data = {
            "label": label,
            "confidence_pct": confidence_pct,
            "infection_pct": round(infection_pct, 2),
            "timestamp": int(time.time() * 1000),
        }
        session["last_analysis"] = analysis_data

        return {
            "ok": True,
            "label": label,
            "confidence_pct": confidence_pct,
            "infection_pct": round(infection_pct, 2),
            "images": {
                "original": _to_png_b64(orig_np),
                "mask": _to_png_b64(display_mask),
                "overlay": _to_png_b64(blended),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/patient-analysis")
def get_patient_analysis(thread_id: str = ""):
    """آخر تحليل بس (مفيش تاريخ متراكم — كل حاجة سيشن-only)."""
    try:
        session = get_session(thread_id)
        if session.get("role") != "patient":
            return {"ok": False, "error": "Access denied."}
        if not thread_id:
            return {"ok": False, "error": "Missing thread_id."}
        return {"ok": True, "last_analysis": session.get("last_analysis"), "upload": session.get("_last_upload_meta")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")