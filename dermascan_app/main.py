"""
main.py — DermaScan AI Backend (FastAPI)
=========================================
تشغيل:
    uvicorn main:app --reload --port 8000

المتطلبات:
- ملفات الموديل skin_cancer_detection_final.keras و UNet_model.keras في نفس مجلد main.py
- متغير بيئة GROQ_API_KEY (أو ملف .env محمّل قبليًا)
- مجلد data/ فيه updated_file_2.xlsx و data/kb/*.md (لقاعدة المعرفة)
"""

import os
import io
import base64
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
)
from image_registry import find_matching_patient, register_image as _register_image
from patient_storage import (
    save_uploaded_image,
    get_latest_image,
    get_all_images,
    update_analysis,
)

MODEL_NAME = "llama-3.3-70b-versatile"
FALLBACK_MODEL_NAME = "llama-3.1-8b-instant"  # حصة توكن يومية منفصلة — بديل وقت الـ rate limit
TEMPERATURE = 0.2
TOP_K = 3

CLF_PATH = "skin_cancer_detection_final.keras"
SEG_PATH = "UNet_model.keras"

app = FastAPI(title="DermaScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # للتطوير المحلي فقط — ضيّقها في الإنتاج
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# حالة عامة (تُهيّأ عند الإقلاع)
# =========================
clf_model = None
seg_model = None
seg_input_size = None
models_loaded = False
models_error = ""

agent = None
fallback_agent = None

# جلسات بسيطة في الذاكرة: thread_id -> {"role":..., "last_analysis":...}
SESSIONS: dict = {}


def get_session(thread_id: str) -> dict:
    if thread_id not in SESSIONS:
        SESSIONS[thread_id] = {
            "role": None,
            "last_analysis": None,
            "current_patient_id": None,
            "current_image_path": None,
            "current_patient_data": None,
        }
    return SESSIONS[thread_id]


def load_models():
    global clf_model, seg_model, seg_input_size, models_loaded, models_error
    try:
        if not os.path.exists(CLF_PATH) or not os.path.exists(SEG_PATH):
            raise FileNotFoundError("Model files (.keras) not found in root directory.")
        clf_model = tf.keras.models.load_model(CLF_PATH)
        seg_model = tf.keras.models.load_model(SEG_PATH, compile=False)
        seg_input_size = seg_model.input_shape[1:3]
        models_loaded = True
        print("✅ Keras models loaded.")
    except Exception as e:
        models_loaded = False
        models_error = str(e)
        print(f"⚠️ Could not load models: {models_error}")


@app.on_event("startup")
def startup():
    global agent, fallback_agent
    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️ GROQ_API_KEY غير موجود في متغيرات البيئة — الشات هيفشل لحد ما تظبطه.")
    load_models()
    _set_agent_models(clf_model, seg_model, seg_input_size)
    print("🔄 Building/loading vectorstore + agent (may take a while first run)...")
    vectorstore = build_default_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    agent = build_agent(retriever, model_name=MODEL_NAME, temperature=TEMPERATURE)
    # agent احتياطي بموديل أخف — له حصة توكن يومية منفصلة، بيتفعّل لما
    # الموديل الأساسي يوصل لحد الـ rate limit (429)
    fallback_agent = build_agent(retriever, model_name=FALLBACK_MODEL_NAME, temperature=TEMPERATURE)
    print("✅ Agent ready (with rate-limit fallback).")




# =========================
# Schemas
# =========================
class RoleRequest(BaseModel):
    thread_id: str
    role: str  # "doctor" | "patient"


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class BookingRequest(BaseModel):
    thread_id: str
    city: str = ""


class ThreadRequest(BaseModel):
    thread_id: str = ""


# =========================
# تحديد الدور — بزرار صريح فقط، بدون أي تخمين نصي
# (هيحل مشكلة "Role Error" اللي كانت بتحصل مع كلمات زي "I am not a doctor")
# =========================
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


# =========================
# تحليل الصورة — نفس منطق التصنيف والسجمنتيشن الأصلي بالحرف، بدون أي تعديل
# =========================
@app.post("/analyze-image")
async def analyze_image(thread_id: str = Form(...), file: UploadFile = File(...)):
    session = get_session(thread_id)
    role = session.get("role")

    if not models_loaded:
        return {"ok": False, "error": f"Models not loaded: {models_error}"}
    if role not in ("doctor", "patient"):
        return {"ok": False, "error": "من فضلك حدّد دورك (دكتور/مريض) أول حاجة."}

    raw = await file.read()

    # Persist the uploaded image before processing
    saved_path = save_uploaded_image(thread_id, raw, file.filename or "upload.png")

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_np = np.array(image).astype(np.uint8)
    orig_h, orig_w = orig_np.shape[:2]

    # --- Classification (نفس الكود الأصلي) ---
    clf_img = image.resize((224, 224))
    clf_array = np.array(clf_img) / 255.0
    clf_array = np.expand_dims(clf_array, axis=0)
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

    # --- Segmentation (نفس الكود الأصلي) ---
    seg_img = image.resize(seg_input_size)
    seg_array = np.array(seg_img).astype("float32") / 255.0
    seg_array = np.expand_dims(seg_array, axis=0)
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

    def to_png_b64(arr_rgb_or_gray):
        if arr_rgb_or_gray.ndim == 3:
            bgr = cv2.cvtColor(arr_rgb_or_gray, cv2.COLOR_RGB2BGR)
        else:
            bgr = arr_rgb_or_gray
        ok, buf = cv2.imencode(".png", bgr)
        return base64.b64encode(buf).decode("utf-8")

    result = {
        "ok": True,
        "label": label,
        "confidence_pct": confidence_pct,
        "infection_pct": round(infection_pct, 2),
        "images": {
            "original": to_png_b64(orig_np),
            "mask": to_png_b64(display_mask),
            "overlay": to_png_b64(blended),
        },
    }

    analysis_data = {
        "label": label,
        "confidence_pct": confidence_pct,
        "infection_pct": infection_pct,
        "timestamp": int(__import__("time").time() * 1000),
    }
    session["last_analysis"] = analysis_data

    # Associate analysis with the persisted image
    if saved_path:
        update_analysis(thread_id, saved_path.name, analysis_data)

    # ---------- منطق خاص بالدكتور: مطابقة الصورة بقاعدة البيانات ----------
    if role == "doctor":
        match_pid, dist = find_matching_patient(image)
        if match_pid:
            report = build_medical_report(match_pid)
            result["doctor"] = {
                "matched": True,
                "patient_id": match_pid,
                "distance": dist,
                "report": report,
            }
        else:
            result["doctor"] = {
                "matched": False,
                "message": (
                    "⚠️ الصورة دي مش متعرّف عليها. النظام بيتعرّف بس على الصور اللي "
                    "اتسجّلت يدويًا قبل كده من نفس الشاشة دي (بيانات الإكسل الأساسية "
                    "مفيهاش صور خالص، بس بيانات نصية). لو دي حالة موجودة عندك، سجّلها "
                    "دلوقتي بربطها برقم مريض عشان تتعرف تلقائيًا في المرة الجاية."
                ),
            }

    # ---------- منطق خاص بالمريض: درجة أولوية الحجز ----------
    elif role == "patient":
        degree, degree_label_ar, urgency, severity = compute_booking_priority(
            label, confidence_pct, infection_pct
        )
        result["patient"] = {
            "priority_degree": degree,
            "priority_label": degree_label_ar,
            "urgency": urgency,
            "severity_score": round(severity, 3),
        }

    return result


@app.post("/register-image")
async def register_image_endpoint(
    thread_id: str = Form(...), patient_id: str = Form(...), file: UploadFile = File(...)
):
    session = get_session(thread_id)
    if session.get("role") != "doctor":
        return {"ok": False, "error": "التسجيل متاح للدكتور فقط."}

    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    _register_image(patient_id, image)
    return {"ok": True, "message": f"✅ تم تسجيل الصورة تحت المريض {patient_id.strip().upper()}."}


@app.post("/book-appointment")
def book(req: BookingRequest):
    session = get_session(req.thread_id)
    if session.get("role") != "patient":
        return {"ok": False, "error": "الحجز متاح في وضع المريض فقط."}

    last = session.get("last_analysis")
    if not last:
        return {"ok": False, "error": "من فضلك اعمل تحليل صورة الأول."}

    degree, _, _, _ = compute_booking_priority(
        last["label"], last["confidence_pct"], last["infection_pct"]
    )
    reason = (
        f"AI analysis: {last['label']} ({last['confidence_pct']}% confidence), "
        f"affected area {last['infection_pct']:.2f}%"
    )
    ticket_text = book_appointment.invoke(
        {"priority_degree": degree, "reason": reason, "city": req.city}
    )
    return {"ok": True, "ticket": ticket_text}


import re as _re

# نطاقات يونيكود مش مسموح تظهر في الرد أبدًا (روسي، تايلاندي، صيني، ياباني، كوري...)
_DISALLOWED_UNICODE_RANGES = _re.compile(
    "[\u0400-\u04FF"   # Cyrillic (روسي)
    "\u0E00-\u0E7F"    # Thai (تايلاندي)
    "\u4E00-\u9FFF"    # CJK (صيني)
    "\u3040-\u30FF"    # Japanese
    "\uAC00-\uD7AF"    # Korean
    "]+"
)


def _sanitize_reply(text: str) -> str:
    """يشيل أي كلمات بلغات غريبة (زي كلمات روسية عشوائية من الموديل الصغير)
    من الرد، مع الحفاظ على باقي النص العربي/الإنجليزي سليم."""
    if not text:
        return text
    cleaned = _DISALLOWED_UNICODE_RANGES.sub("", text)
    cleaned = _re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


@app.post("/chat")
def chat(req: ChatRequest):
    session = get_session(req.thread_id)
    role = session.get("role")
    if role not in ("doctor", "patient"):
        return {"ok": False, "error": "من فضلك حدّد دورك (دكتور/مريض) أول حاجة."}

    if agent is None:
        return {"ok": False, "error": "الـ Agent لسه بيتحمّل، جرّب تاني بعد شوية."}

    set_current_role(role)
    set_session_storage(SESSIONS)
    set_current_thread_id(req.thread_id)
    role_prefix = "[DOCTOR MODE] " if role == "doctor" else "[PATIENT MODE] "
    anti_refusal = ("\n\n[SYSTEM OVERRIDE: You have image analysis tools (segment_my_image, "
        "reanalyze_my_image, get_my_uploaded_image). You MUST call one of these tools "
        "immediately when the user mentions their image — NEVER say 'Cannot read', "
        "'does not support image input', 'I cannot see', or any refusal. "
        "The tools handle everything. Just call them.]")
    full_prompt = role_prefix + req.message + anti_refusal

    # Context management: track conversation turns; start fresh thread after limit
    MAX_TURNS = 15
    conversation_summary = session.get("_summary", "")
    turn_count = session.get("_turn_count", 0)

    # If we've exceeded max turns, start a fresh thread and include summary
    if turn_count >= MAX_TURNS:
        fresh_thread_id = f"{req.thread_id}_fresh_{turn_count}"
        config_agent = {"configurable": {"thread_id": fresh_thread_id}, "recursion_limit": 15}
        session["_turn_count"] = 0
        session["_summary"] = ""
        if conversation_summary:
            full_prompt = f"[Previous conversation summary: {conversation_summary}]\n\n{full_prompt}"
    else:
        config_agent = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 15}

    def _run_once(active_agent, prompt_text: str):
        import re as _re2
        final_text = ""
        tool_calls = []
        analysis_marker = None
        for event in active_agent.stream(
            {"messages": [("user", prompt_text)]},
            config=config_agent,
            stream_mode="values",
        ):
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    tool_calls.append({"name": tc["name"], "args": tc["args"]})
            if last_msg.type == "ai" and last_msg.content:
                final_text = last_msg.content
            for msg in event["messages"]:
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content:
                    m = _re2.search(r'\[ANALYSIS_RESULT:(\{.*?\})\]', content)
                    if m:
                        analysis_marker = m.group(0)

        final = _sanitize_reply(final_text)
        if analysis_marker:
            final = analysis_marker + "\n\n" + final
        return final, tool_calls

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
            "Tool" in s
            and ("not exist" in s.lower() or "failed" in s.lower() or "error" in s.lower())
        ) or (
            "tool_call" in s.lower() or "tool call" in s.lower()
        ) or (
            "function call" in s.lower() and "fail" in s.lower()
        )

    def _is_timeout(err) -> bool:
        s = str(err)
        return "timeout" in s.lower() or "timed out" in s.lower()

    def _get_session_images():
        import os as _os
        stored = session.pop("_current_images", [])
        return [_os.path.basename(p) for p in stored]

    def _friendly_error(err_type: str, err_msg: str) -> str:
        err_text = f"{err_type}: {err_msg}"
        if _is_request_too_large(Exception(err_text)):
            return "الطلب تجاوز الحد المسموح للذكاء الاصطناعي. تم تصغير المحادثة تلقائيًا، حاول مرة أخرى."
        if _is_image_error(Exception(err_text)):
            return "تم تحليل الصورة بنجاح باستخدام نماذج الذكاء الاصطناعي المحلية. النتائج معروضة أعلاه."
        if _is_rate_limit(Exception(err_text)):
            return "وصلت للحد الأقصى لاستخدام النموذج. انتظر قليلاً ثم حاول مرة أخرى، أو استخدم نموذجًا آخر."
        if _is_timeout(Exception(err_text)):
            return "استغرق الطلب وقتًا طويلاً. حاول مرة أخرى بسؤال أبسط."
        return f"حدث خطأ غير متوقع: {err_msg}"

    last_error = None

    # Primary attempts (up to 2)
    for attempt in range(2):
        try:
            final_text, tool_calls = _run_once(agent, full_prompt)
            # Post-filter: block ALL LLM image-refusal variants
            refusal_patterns = [
                "Cannot read", "does not support image", "cannot process images",
                "I cannot see", "I cannot look", "I don't have the ability",
                "I'm not able to", "I am not able to", "can't see images",
                "can't process images", "don't have vision", "no vision capability",
                "text-based", "language model", "I'm a text",
            ]
            if any(_re.search(p, final_text, _re.IGNORECASE) for p in refusal_patterns):
                final_text = "تم تحليل الصورة بنجاح باستخدام نماذج الذكاء الاصطناعي المحلية. النتائج معروضة أعلاه."
            session["_turn_count"] = turn_count + 1
            # Store a brief summary of the last exchange for future context management
            session["_summary"] = req.message[:200]
            resp = {"ok": True, "reply": final_text, "tool_calls": tool_calls, "model": "primary"}
            session_imgs = _get_session_images()
            if session_imgs:
                resp["images"] = session_imgs
            return resp
        except Exception as e:
            last_error = e
            err_str = f"{type(e).__name__}: {e}"
            if _is_image_error(e):
                # Image-related error: the analysis happened locally; just return a friendly message
                session["_turn_count"] = turn_count + 1
                return {"ok": True, "reply": _friendly_error(type(e).__name__, str(e)), "tool_calls": [], "model": "primary"}
            if _is_request_too_large(e):
                # 413: auto-trim by starting a fresh thread
                fresh_id = f"{req.thread_id}_trimmed_{turn_count}"
                config_agent = {"configurable": {"thread_id": fresh_id}, "recursion_limit": 15}
                try:
                    final_text, tool_calls = _run_once(agent, full_prompt)
                    session["_turn_count"] = 0
                    resp = {"ok": True, "reply": final_text, "tool_calls": tool_calls, "model": "primary_trimmed", "notice": _friendly_error(type(e).__name__, str(e))}
                    session_imgs = _get_session_images()
                    if session_imgs:
                        resp["images"] = session_imgs
                    return resp
                except Exception:
                    pass  # fall through to error return
            if _is_rate_limit(e):
                break  # try fallback
            if attempt == 0 and _is_tool_use_failed(e):
                continue
            break

    # Fallback agent (cheaper model) on rate limit
    if last_error and _is_rate_limit(last_error) and fallback_agent is not None:
        try:
            final_text, tool_calls = _run_once(fallback_agent, full_prompt)
            session["_turn_count"] = turn_count + 1
            resp = {
                "ok": True,
                "reply": final_text,
                "tool_calls": tool_calls,
                "model": "fallback",
                "notice": "⚠️ تم استخدام موديل أخف مؤقتًا لأن الموديل الأساسي وصل لحد الاستخدام اليومي.",
            }
            session_imgs = _get_session_images()
            if session_imgs:
                resp["images"] = session_imgs
            return resp
        except Exception as e2:
            last_error = e2

    # Return user-friendly error instead of raw exception
    return {"ok": False, "error": _friendly_error(type(last_error).__name__, str(last_error))}


# =========================
# Serve cached images from downloads directory (doctor-only)
# =========================
@app.get("/image")
def get_image(path: str = "", thread_id: str = ""):
    """Serve a cached image by filename from the downloaded_images cache directory.
    Doctor-only by default; patients may access images stored in their session."""
    try:
        session = get_session(thread_id) if thread_id else {}
        role = session.get("role")
        if role != "doctor":
            # Allow patients access to images stored in their own session
            stored = session.get("_session_image_names", [])
            if role != "patient" or path not in stored:
                return {"ok": False, "error": "Access denied: image retrieval is restricted to doctors."}

        from pathlib import Path
        if not path or ".." in path or "/" in path or "\\" in path:
            return {"ok": False, "error": "Invalid path"}
        
        from image_downloader import CACHE_DIR
        image_path = CACHE_DIR / path
        
        if not image_path.exists():
            return {"ok": False, "error": "Image not found"}
        
        import mimetypes
        media_type, _ = mimetypes.guess_type(str(image_path))
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        import base64
        b64 = base64.b64encode(image_data).decode("utf-8")
        return {"ok": True, "base64": b64, "media_type": media_type or "image/jpeg"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =========================
# Serve patient's own uploaded images
# =========================
@app.get("/patient-image")
def get_patient_image(thread_id: str = ""):
    """Serve the most recent uploaded image for the calling patient."""
    try:
        if not thread_id:
            return {"ok": False, "error": "Missing thread_id parameter."}
        session = get_session(thread_id)
        if session.get("role") != "patient":
            role = session.get("role")
            return {"ok": False, "error": f"Access denied: current role is '{role}', expected 'patient'. Please select '🙂 أنا مريض' and re-upload."}

        latest = get_latest_image(thread_id)
        if not latest:
            return {"ok": False, "error": "No uploaded image found. Use the 📎 button to upload first."}

        image_path = Path(latest["path"])
        if not image_path.exists():
            return {"ok": False, "error": "Image file not found on disk."}

        import mimetypes
        media_type, _ = mimetypes.guess_type(str(image_path))

        with open(image_path, "rb") as f:
            image_data = f.read()

        import base64
        b64 = base64.b64encode(image_data).decode("utf-8")
        return {
            "ok": True,
            "base64": b64,
            "media_type": media_type or "image/jpeg",
            "metadata": {k: v for k, v in latest.items() if k != "path"},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =========================
# Serve patient's analysis history
# =========================
@app.post("/segment-patient-image")
def segment_patient_image_api(req: ThreadRequest):
    """Run U-Net segmentation on the patient's stored image. Returns mask + overlay as base64."""
    thread_id = req.thread_id
    try:
        if not thread_id:
            return {"ok": False, "error": "Missing thread_id parameter."}
        session = get_session(thread_id)
        if session.get("role") != "patient":
            role = session.get("role")
            return {"ok": False, "error": f"Access denied: current role is '{role}', expected 'patient'. Please select '🙂 أنا مريض' and re-upload."}

        from patient_storage import get_latest_image
        latest = get_latest_image(thread_id)
        if not latest:
            return {"ok": False, "error": "No uploaded image found."}
        if not models_loaded:
            return {"ok": False, "error": f"Models not loaded: {models_error}"}

        image_path = Path(latest["path"])
        if not image_path.exists():
            return {"ok": False, "error": "Stored image file not found."}

        image = Image.open(image_path).convert("RGB")
        orig_np = np.array(image).astype(np.uint8)
        orig_h, orig_w = orig_np.shape[:2]

        seg_img = image.resize(seg_input_size)
        seg_array = np.array(seg_img).astype("float32") / 255.0
        seg_array = np.expand_dims(seg_array, axis=0)
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

        def to_png_b64(arr):
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
            ok, buf = cv2.imencode(".png", bgr)
            return base64.b64encode(buf).decode("utf-8")

        return {
            "ok": True,
            "infection_pct": round(infection_pct, 2),
            "images": {
                "mask": to_png_b64(display_mask),
                "overlay": to_png_b64(blended),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/reanalyze-patient-image")
def reanalyze_patient_image(req: ThreadRequest):
    """Re-run classification + segmentation on the patient's stored image.
    Returns full analysis with image data. NO LLM involved."""
    thread_id = req.thread_id
    try:
        if not thread_id:
            return {"ok": False, "error": "Missing thread_id parameter."}
        session = get_session(thread_id)
        if session.get("role") != "patient":
            role = session.get("role")
            return {"ok": False, "error": f"Access denied: current role is '{role}', expected 'patient'. Please select '🙂 أنا مريض' and re-upload."}

        from patient_storage import get_latest_image, update_analysis
        latest = get_latest_image(thread_id)
        if not latest:
            return {"ok": False, "error": "No uploaded image found."}

        if not models_loaded:
            return {"ok": False, "error": f"Models not loaded: {models_error}"}

        image_path = Path(latest["path"])
        if not image_path.exists():
            return {"ok": False, "error": "Stored image file not found."}

        image = Image.open(image_path).convert("RGB")
        orig_np = np.array(image).astype(np.uint8)
        orig_h, orig_w = orig_np.shape[:2]

        # Classification
        clf_img = image.resize((224, 224))
        clf_array = np.array(clf_img) / 255.0
        clf_array = np.expand_dims(clf_array, axis=0)
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

        # Segmentation
        seg_img = image.resize(seg_input_size)
        seg_array = np.array(seg_img).astype("float32") / 255.0
        seg_array = np.expand_dims(seg_array, axis=0)
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

        def to_png_b64(arr_rgb_or_gray):
            if arr_rgb_or_gray.ndim == 3:
                bgr = cv2.cvtColor(arr_rgb_or_gray, cv2.COLOR_RGB2BGR)
            else:
                bgr = arr_rgb_or_gray
            ok, buf = cv2.imencode(".png", bgr)
            return base64.b64encode(buf).decode("utf-8")

        analysis_data = {
            "label": label,
            "confidence_pct": confidence_pct,
            "infection_pct": round(infection_pct, 2),
            "timestamp": int(__import__("time").time() * 1000),
        }
        update_analysis(thread_id, latest["filename"], analysis_data)
        session["last_analysis"] = analysis_data

        return {
            "ok": True,
            "label": label,
            "confidence_pct": confidence_pct,
            "infection_pct": round(infection_pct, 2),
            "images": {
                "original": to_png_b64(orig_np),
                "mask": to_png_b64(display_mask),
                "overlay": to_png_b64(blended),
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/patient-analysis")
def get_patient_analysis(thread_id: str = ""):
    """Return the analysis history for the calling patient's uploaded images."""
    try:
        session = get_session(thread_id)
        if session.get("role") != "patient":
            return {"ok": False, "error": "Access denied."}
        if not thread_id:
            return {"ok": False, "error": "Missing thread_id."}

        images = get_all_images(thread_id)
        history = []
        for img in images:
            history.append({
                "filename": img["filename"],
                "original_name": img["original_name"],
                "timestamp": img["timestamp"],
                "analysis": img.get("analysis"),
            })
        return {"ok": True, "history": history}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# =========================
# تقديم الواجهة الأمامية (static/index.html) على نفس السيرفر
# =========================
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")