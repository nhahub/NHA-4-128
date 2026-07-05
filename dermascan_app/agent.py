import random
import string
import re
import datetime
import pandas as pd
from pathlib import Path
import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """You are DermaScan AI, an intelligent dermatology support assistant
built on a real clinical dataset of 1000 skin cancer patients.
You were built by Eng. Youssef Bastawisy.

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

Response style:
- Concise: 2-5 sentences for simple queries, bullets for complex ones.
- Professional and empathetic tone.
- Always remind patients that DermaScan AI does not replace professional diagnosis."""

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

    return f"""### 📋 التقرير الطبي — المريض {patient_id}

**البيانات الأساسية:** {r['name']}, {r['age']} سنة, {r['gender']}, {r['country']}

**التشخيص:** {r['diagnosis']} | **نتيجة الخزعة:** {r['biopsy_result']}

**الآفة الجلدية:** {r['lesion_size_mm']} مم | اللون: {r['lesion_color']} | الموضع: {r['lesion_location']}
عدم انتظام الحواف: {r['border_irregularity']:.2f} | عدم التماثل: {r['asymmetry']:.2f}

**نوع البشرة (Fitzpatrick):** {r['skin_type_fitzpatrick']} | **التعرض للأشعة فوق البنفسجية:** {r['UV_exposure_level']}

**الطفرة الجينية:** {r['genetic_mutation']} | **درجة الخطورة الوراثية:** {r['hereditary_risk_score']:.3f}

**تاريخ عائلي للإصابة:** {r['family_history_skin_cancer']} | **مثبط مناعة:** {r['immunosuppressed']}
{family_block}
**📝 ملخص الإحالة:** الحالة مسجّلة مسبقًا في قاعدة البيانات وتمت مطابقة الصورة بنجاح. درجة أولوية المتابعة: **{urgency}**.
"""


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
    )

    # ملاحظة: book_appointment اتشالت من هنا عمدًا. الحجز بيحصل بس عن طريق
    # زرار "احجز موعد الآن" في الواجهة (استدعاء مباشر للدالة)، مش عن طريق
    # الشات، عشان الموديل ميحجزش لوحده على أي كلام عام.
    tools = [search_knowledge_base, lookup_patient, create_referral_ticket]
    memory = MemorySaver()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )