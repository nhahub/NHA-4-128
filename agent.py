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

# Single memory shared between the primary and fallback models, so that if a
# switch happens (rate limit), the new model continues from the same
# conversation history instead of losing context.
_SHARED_MEMORY = MemorySaver()

# Only the last N messages are actually sent to the model (not the entire
# conversation from the start). This is what prevents runaway token usage
# and reduces repeated/confused replies.
_MAX_HISTORY_MESSAGES = 12


def _trim_history(state):
    """Keeps only the most recent messages (plus the system prompt) before
    sending to the model, to stop token bleed as the conversation grows."""
    msgs = state["messages"]
    system_msgs = [m for m in msgs if getattr(m, "type", None) == "system"]
    other_msgs = [m for m in msgs if getattr(m, "type", None) != "system"]
    trimmed = other_msgs[-_MAX_HISTORY_MESSAGES:]
    return {"llm_input_messages": system_msgs + trimmed}

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

Tool usage (only two tools exist — keep it simple):
- search_knowledge_base — for any clinical question about diagnoses, risk factors,
  lesion features, genetics, or patient management. Call this first.
  Cite the source filename in your answer.
- lookup_patient — for specific patient records (e.g. "P0042", "patient P0042",
  "show me patient P0042", "analyze patient P0042"), family statistics
  (e.g. "family F001"), or dataset analytics. ONLY available in DOCTOR mode — if a
  patient asks for another patient's data, politely refuse and explain this is
  restricted to clinicians. This tool already returns a fully formatted clinical
  report with analysis — just present its output as-is, don't rewrite or summarize it.

Booking appointments and specialist referrals are NOT chat tools. If the user asks
to book an appointment or needs a referral, tell them to use the "📅 Book
appointment" button that appears after an image analysis (it uses the actual
severity score, not a guess).

Role-based behavior:
- [DOCTOR MODE]: Give full clinical details, dataset analytics, patient records.
  The doctor has verified access to patient identities and IDs.
- [PATIENT MODE]: Focus on general guidance, recommendations, next steps. Never
  reveal other patients' data. Only discuss the patient's own uploaded results.
- After a Malignant result: stress urgency, give clear next steps, and suggest
  using the booking button.
- After a Benign result: give prevention tips and monitoring advice.

Response style:
- Concise: 2-5 sentences for simple queries, bullets for complex ones.
- Professional and empathetic tone.
- Always remind patients that DermaScan AI does not replace professional diagnosis."""

# Placeholder clinic/hospital data, until a real system is wired up to actual
# locations via a maps/clinic-directory service (e.g. Google Places API based
# on the user's actual location).
DEFAULT_CLINIC = {
    "name": "Reference Dermatology Clinic - DermaScan Partner Clinic",
    "address": "University Street, 2nd Floor, Downtown",
    "phone": "01000000000",
    "note": "Demo/placeholder data — showing a real clinic and nearest branch requires integration with an actual maps/location service.",
}

# Demo branches by city (placeholder) — not yet wired to a real map, but they
# give a more plausible sense of a nearby location instead of one generic
# clinic for everyone.
CLINICS_BY_CITY = {
    "Cairo": {"name": "Cairo Dermatology Center", "address": "Tahrir Street, Dokki, Giza", "phone": "01011112222"},
    "Giza": {"name": "Giza Specialized Clinics", "address": "Haram Street, Giza", "phone": "01022223333"},
    "Alexandria": {"name": "Alexandria Dermatology Center", "address": "Fouad Street, Raml Station, Alexandria", "phone": "01033334444"},
    "Mansoura": {"name": "Mansoura Dermatology Clinic", "address": "Republic Street, Mansoura", "phone": "01044445555"},
    "Aswan": {"name": "Aswan Specialized Medical Center", "address": "Nile Corniche, Aswan", "phone": "01055556666"},
}

DOCTOR_NAMES = ["Dr. Ahmed Fathy", "Dr. Marwa Hussein", "Dr. Kareem Adel", "Dr. Sara Youssef", "Dr. Mohamed El-Sherif"]

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _format_datetime(dt: datetime.datetime) -> str:
    weekday = _WEEKDAYS[dt.weekday()]
    month = _MONTHS[dt.month - 1]
    hour_12 = dt.hour % 12 or 12
    period = "AM" if dt.hour < 12 else "PM"
    return f"{weekday} {dt.day} {month} {dt.year} — {hour_12}:{dt.minute:02d} {period}"


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

# Current user role (updated from main.py before every call to the agent)
_CURRENT_ROLE = None  # "doctor" | "patient" | None

DATA_PATH = Path(__file__).parent / "data" / "five_sample_patients_with_features.xlsx"


def set_current_role(role: str) -> None:
    """Called from main.py before running the agent, so data tools know which
    permissions to enforce."""
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
# "Raw" functions that can be reused directly from main.py (without going
# through the LLM), to guarantee 100% accuracy of medical reports.
# =========================================================

def get_patient_record(patient_id: str) -> dict | None:
    """Returns a dict with the patient's data, or None if not found."""
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


def _interpret_hereditary_risk(score: float) -> str:
    if score < 0.3:
        return "Low — an annual check-up is enough"
    if score < 0.6:
        return "Moderate — a check-up every 6 months is recommended"
    if score < 0.8:
        return "High — a dermatology visit every 3 months is recommended"
    return "Very high — genetic counseling and intensive follow-up recommended"


def build_medical_report(
    patient_id: str,
    current_analysis: dict | None = None,
    source: str = "chat",
) -> str | None:
    """Builds a single medical report that merges: the patient's record +
    (if provided) the current image analysis based on lesion shape + the
    hereditary/genetic history — all combined together.
    source: "image" if coming from an image match, "chat" if coming from a
    text question in the chat.
    Returns None if the patient is not found."""
    r = get_patient_record(patient_id)
    if r is None:
        return None

    family_block = ""
    fid = r.get("family_id")
    if fid:
        fam = get_family_record(str(fid))
        if fam is not None:
            family_block = (
                f"\n**🧬 Hereditary History (Family {fid}):**\n"
                f"- Family members: {fam['total_members']} | Affected: {fam['affected_members']}\n"
                f"- Average hereditary risk score: {fam['avg_risk_score']:.3f}\n"
                f"- Dominant genetic mutation: {fam['genetic_mutation']}\n"
                f"- Family risk level: **{fam['family_risk_level']}**\n"
            )

    risk_note = _interpret_hereditary_risk(float(r["hereditary_risk_score"]))

    is_urgent = str(r["diagnosis"]).lower() not in ("benign", "nevus")
    current_block = ""
    if current_analysis:
        is_urgent = (
            is_urgent
            or current_analysis["label"] == "Malignant"
            or current_analysis["infection_pct"] >= 15
        )
        current_block = (
            f"\n**🔬 Current Uploaded Image Analysis (based on lesion shape):** "
            f"{current_analysis['label']} (confidence {current_analysis['confidence_pct']}%) | "
            f"Affected area: {current_analysis['infection_pct']}%\n"
        )

    urgency = "Urgent 🔴" if is_urgent else "Routine 🟢"
    closing = (
        "The image was successfully matched to the patient's record, and the "
        "analysis above combines the current lesion shape with the recorded "
        "hereditary history."
        if source == "image"
        else "Record retrieved based on patient ID (no new image)."
    )

    return f"""### 📋 Medical Report — Patient {patient_id}

**Basic Info:** {r['name']}, {r['age']} years old, {r['gender']}, {r['country']}

**Recorded Diagnosis:** {r['diagnosis']} | **Biopsy Result:** {r['biopsy_result']}
{current_block}
**Skin Lesion:** {r['lesion_size_mm']} mm | Color: {r['lesion_color']} | Location: {r['lesion_location']}
Border irregularity: {r['border_irregularity']:.2f} | Asymmetry: {r['asymmetry']:.2f}

**Skin Type (Fitzpatrick):** {r['skin_type_fitzpatrick']} | **UV Exposure:** {r['UV_exposure_level']}

**Genetic Mutation:** {r['genetic_mutation']} | **Hereditary Risk Score:** {r['hereditary_risk_score']:.3f} ({risk_note})

**Family History of Skin Cancer:** {r['family_history_skin_cancer']} | **Immunosuppressed:** {r['immunosuppressed']}
{family_block}
**📝 Analysis & Recommendation (lesion shape + heredity combined):** Follow-up priority: **{urgency}**. {closing}
"""


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


# =========================================================
# LangChain Agent tools
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
        return build_medical_report(pid_match.group().upper(), source="chat") or (
            f"Patient {pid_match.group().upper()} not found."
        )

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
def book_appointment(priority_degree: int, reason: str, city: str = "") -> str:
    """Book a patient appointment given a priority degree (1=highest/emergency,
    2=urgent, 3=routine) and the patient's city, to get a concrete date/time,
    room number, doctor name, and a nearby clinic.
    NOTE: this is invoked directly by the booking button in the UI, not by the
    chat LLM (removed from the agent's toolset — see build_agent)."""
    degree_map = {
        1: ("emergency", "Priority 1 — Emergency"),
        2: ("urgent", "Priority 2 — Urgent"),
        3: ("routine", "Priority 3 — Routine"),
    }
    urgency, label = degree_map.get(priority_degree, ("routine", "Priority 3 — Routine"))
    ticket_id = "APT-" + "".join(random.choices(string.digits, k=6))

    clinic = _pick_clinic(city)
    slot = _compute_slot(urgency)
    doctor = random.choice(DOCTOR_NAMES)
    room = random.randint(1, 12)

    city_note = "" if (city and city.strip() in CLINICS_BY_CITY) else (
        f"\n  ⚠️ '{city}' is not one of the cities we currently have registered "
        f"branches in, so the nearest default branch was selected." if city else ""
    )

    return (
        f"✅ Booking confirmed.\n"
        f"  Ticket ID  : {ticket_id}\n"
        f"  Priority   : {label}\n"
        f"  Appointment: {_format_datetime(slot)}\n"
        f"  Doctor     : {doctor}\n"
        f"  Room       : Room {room}\n"
        f"  Clinic     : {clinic['name']}\n"
        f"  Address    : {clinic['address']}\n"
        f"  Phone      : {clinic['phone']}\n"
        f"  Reason     : {reason}{city_note}\n"
        f"  ⚠️ Note: this is demo data — not yet wired to a real hospital booking system or maps."
    )


# =========================================================
# Booking priority calculation (outside LangChain — precise and direct)
# =========================================================

def compute_booking_priority(label: str, confidence_pct: float, infection_pct: float):
    """
    Computes the booking priority degree (1/2/3) based on a simple severity
    score: a blend of the malignant-classification confidence and the
    affected-area percentage.
    Returns (degree: int, label: str, urgency: str, severity_score: float)
    """
    confidence = confidence_pct / 100.0
    infection = infection_pct / 100.0

    if label == "Malignant":
        severity = confidence * 0.6 + infection * 0.4
        if severity >= 0.75:
            return 1, "Priority 1 (Emergency)", "emergency", severity
        return 2, "Priority 2 (Urgent)", "urgent", severity
    else:
        severity = (1 - confidence) * 0.3 + infection * 0.5
        if infection >= 0.15 or confidence < 0.6:
            return 2, "Priority 2 (Urgent)", "urgent", severity
        return 3, "Priority 3 (Routine)", "routine", severity


def build_agent(retriever, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.2):
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

    # Note: book_appointment and create_referral_ticket were deliberately
    # removed from here. Booking only happens through the "Book appointment
    # now" button in the UI (a direct function call), not through the chat,
    # so the model can never book/refer on its own from general conversation.
    tools = [search_knowledge_base, lookup_patient]

    # Same memory shared between the primary and fallback agent, so that if a
    # model switch happens, the conversation continues from the same history
    # instead of losing context.
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=_SHARED_MEMORY,
        pre_model_hook=_trim_history,
    )