"""
main.py — DermaScan AI Backend (FastAPI)
=========================================
Run:
    uvicorn main:app --reload --port 8000

Requirements:
- Model files skin_cancer_detection_final.keras and UNet_model.keras in the same
  folder as main.py
- Environment variable GROQ_API_KEY (or a pre-loaded .env file)
- A data/ folder containing updated_file_2.xlsx and data/kb/*.md (knowledge base)
"""

import os
import io
import base64
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
    build_medical_report,
    compute_booking_priority,
    book_appointment,
)
from image_registry import find_matching_patient, register_image as _register_image

MODEL_NAME = "openai/gpt-oss-120b"
FALLBACK_MODEL_NAME = "openai/gpt-oss-20b"  # separate daily token quota — used as a fallback during rate limits
TEMPERATURE = 0.2
TOP_K = 3

CLF_PATH = "skin_cancer_detection_final.keras"
SEG_PATH = "UNet_model.keras"

app = FastAPI(title="DermaScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local development only — restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Global state (initialized on startup)
# =========================
clf_model = None
seg_model = None
seg_input_size = None
models_loaded = False
models_error = ""

agent = None
fallback_agent = None

# Simple in-memory sessions: thread_id -> {"role":..., "last_analysis":...}
SESSIONS: dict = {}


def get_session(thread_id: str) -> dict:
    if thread_id not in SESSIONS:
        SESSIONS[thread_id] = {"role": None, "last_analysis": None}
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
        print("⚠️ GROQ_API_KEY not found in environment variables — chat will fail until you set it.")
    load_models()
    print("🔄 Building/loading vectorstore + agent (may take a while first run)...")
    vectorstore = build_default_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    agent = build_agent(retriever, model_name=MODEL_NAME, temperature=TEMPERATURE)
    # Fallback agent on a lighter model — has its own separate daily token
    # quota, kicked in once the primary model hits a rate limit (429)
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


# =========================
# Set the role — explicit button only, no text-based guessing
# (fixes the "Role Error" that used to happen with phrases like
# "I am not a doctor")
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
# Image analysis — same original classification and segmentation logic,
# unchanged.
# =========================
@app.post("/analyze-image")
async def analyze_image(thread_id: str = Form(...), file: UploadFile = File(...)):
    session = get_session(thread_id)
    role = session.get("role")

    if not models_loaded:
        return {"ok": False, "error": f"Models not loaded: {models_error}"}
    if role not in ("doctor", "patient"):
        return {"ok": False, "error": "Please select your role (doctor/patient) first."}

    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    orig_np = np.array(image).astype(np.uint8)
    orig_h, orig_w = orig_np.shape[:2]

    # --- Classification (same original code) ---
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

    # --- Segmentation (same original code) ---
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

    session["last_analysis"] = {
        "label": label,
        "confidence_pct": confidence_pct,
        "infection_pct": infection_pct,
    }

    # ---------- Doctor-specific logic: match the image against the database ----------
    if role == "doctor":
        match_pid, dist = find_matching_patient(image)
        current_analysis = {
            "label": label,
            "confidence_pct": confidence_pct,
            "infection_pct": round(infection_pct, 2),
        }
        if match_pid:
            report = build_medical_report(match_pid, current_analysis=current_analysis, source="image")
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
                    "⚠️ This image is not recognized. The system only recognizes images "
                    "that were manually registered before from this same screen "
                    "(the core Excel data contains no images at all, only text data). "
                    "If this is an existing case on your end, register it now by linking "
                    "it to a patient ID so it's recognized automatically next time."
                ),
            }

    # ---------- Patient-specific logic: booking priority degree ----------
    elif role == "patient":
        degree, degree_label, urgency, severity = compute_booking_priority(
            label, confidence_pct, infection_pct
        )
        result["patient"] = {
            "priority_degree": degree,
            "priority_label": degree_label,
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
        return {"ok": False, "error": "Registration is only available to doctors."}

    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    _register_image(patient_id, image)
    return {"ok": True, "message": f"✅ Image registered under patient {patient_id.strip().upper()}."}


@app.post("/book-appointment")
def book(req: BookingRequest):
    session = get_session(req.thread_id)
    if session.get("role") != "patient":
        return {"ok": False, "error": "Booking is only available in patient mode."}

    last = session.get("last_analysis")
    if not last:
        return {"ok": False, "error": "Please run an image analysis first."}

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

# Unicode ranges not allowed to appear in the reply at all (Cyrillic, Thai,
# Chinese, Japanese, Korean...)
_DISALLOWED_UNICODE_RANGES = _re.compile(
    "[\u0400-\u04FF"   # Cyrillic
    "\u0E00-\u0E7F"    # Thai
    "\u4E00-\u9FFF"    # CJK (Chinese)
    "\u3040-\u30FF"    # Japanese
    "\uAC00-\uD7AF"    # Korean
    "]+"
)


def _sanitize_reply(text: str) -> str:
    """Strips any stray foreign-language words (e.g. random Russian tokens
    from the smaller model) from the reply, while keeping the rest of the
    Arabic/English text intact."""
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
        return {"ok": False, "error": "Please select your role (doctor/patient) first."}

    if agent is None:
        return {"ok": False, "error": "The agent is still loading, please try again shortly."}

    set_current_role(role)
    role_prefix = "[DOCTOR MODE] " if role == "doctor" else "[PATIENT MODE] "
    full_prompt = role_prefix + req.message

    config_agent = {"configurable": {"thread_id": req.thread_id}}

    def _run_once(active_agent, prompt_text: str):
        final_text = ""
        tool_calls = []
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
        return _sanitize_reply(final_text), tool_calls

    def _is_rate_limit(err) -> bool:
        s = str(err)
        return (
            "rate_limit_exceeded" in s
            or "429" in s
            or "Rate limit reached" in s
            or "RESOURCE_EXHAUSTED" in s
            or "quota" in s.lower()
        )

    def _is_tool_use_failed(err) -> bool:
        s = str(err)
        return "tool_use_failed" in s or "Failed to call a function" in s

    last_error = None
    used_fallback = False

    # Attempts 1 and 2: primary model (auto-retry once on tool_use_failed)
    for attempt in range(2):
        try:
            final_text, tool_calls = _run_once(agent, full_prompt)
            return {"ok": True, "reply": final_text, "tool_calls": tool_calls, "model": "primary"}
        except Exception as e:
            last_error = e
            if _is_rate_limit(e):
                break  # no point retrying the same model, go straight to fallback
            if attempt == 0 and _is_tool_use_failed(e):
                continue
            break

    # If the primary model hit its daily usage limit, automatically try the lighter model
    if _is_rate_limit(last_error) and fallback_agent is not None:
        try:
            final_text, tool_calls = _run_once(fallback_agent, full_prompt)
            return {
                "ok": True,
                "reply": final_text,
                "tool_calls": tool_calls,
                "model": "fallback",
                "notice": "⚠️ A lighter model was used temporarily because the primary model hit its daily usage limit.",
            }
        except Exception as e2:
            last_error = e2

    return {"ok": False, "error": f"{type(last_error).__name__}: {last_error}"}


# =========================
# Serve the frontend (static/index.html) from the same server
# =========================
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")