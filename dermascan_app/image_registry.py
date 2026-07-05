"""
image_registry.py
==================
نظام بسيط للتعرف على ما إذا كانت صورة جلدية مرفوعة قد سُجّلت من قبل لمريض معيّن.

الفكرة:
- نحسب "بصمة" إدراكية (perceptual hash / average hash) لكل صورة.
- نخزّن الخريطة patient_id -> hash في ملف JSON محلي (data/image_registry.json).
- عند رفع صورة جديدة، نقارن بصمتها بكل البصمات المخزنة (Hamming distance).
  لو أقرب بصمة أقل من أو تساوي MATCH_THRESHOLD => نعتبرها "نفس الصورة" وبالتالي نفس المريض.

ملاحظة: هذا حل عملي بسيط (average hash) وليس بديلاً عن نظام تعرف حقيقي على الوجه/الجلد،
لكنه كافٍ لمطابقة "نفس الصورة الملتقطة من قبل" بدقة معقولة، وهو ما يحتاجه السيناريو.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

REGISTRY_PATH = Path(__file__).parent / "data" / "image_registry.json"
HASH_SIZE = 8               # 8x8 = 64 bit hash
MATCH_THRESHOLD = 6         # أقصى مسافة هامنج تعتبر "نفس الصورة" (من أصل 64)


def _ahash(image: Image.Image) -> str:
    """يحسب average-hash لصورة PIL ويرجعه كسلسلة بتات '0'/'1'."""
    img = image.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    return "".join("1" if b else "0" for b in bits)


def _hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_registry(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def find_matching_patient(image: Image.Image) -> Tuple[Optional[str], Optional[int]]:
    """
    يبحث عن مريض مسجّل صورته مطابقة للصورة المُدخلة.
    يرجع (patient_id, hamming_distance) أو (None, None) لو مفيش تطابق.
    """
    reg = _load_registry()
    if not reg:
        return None, None

    target = _ahash(image)
    best_pid, best_dist = None, 999
    for pid, h in reg.items():
        d = _hamming(target, h)
        if d < best_dist:
            best_dist, best_pid = d, pid

    if best_pid is not None and best_dist <= MATCH_THRESHOLD:
        return best_pid, best_dist
    return None, None


def register_image(patient_id: str, image: Image.Image) -> None:
    """يسجّل بصمة الصورة تحت رقم المريض (تُستدعى عند تأكيد الدكتور للتسجيل)."""
    reg = _load_registry()
    reg[patient_id.strip().upper()] = _ahash(image)
    _save_registry(reg)


def is_registered(patient_id: str) -> bool:
    reg = _load_registry()
    return patient_id.strip().upper() in reg
