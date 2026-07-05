# DermaScan AI — تشغيل المشروع

## الملفات
```
dermascan_app/
├── main.py              ← الباك-إند (FastAPI)
├── agent.py             ← الـ LangGraph agent (نفس المنطق، مع صلاحيات الدور)
├── rag.py               ← بناء قاعدة المعرفة (بدون تغيير)
├── image_registry.py    ← مطابقة الصور المسجّلة سابقًا
├── requirements.txt
└── static/
    └── index.html       ← الواجهة (شات فعلي متصل بالباك-إند)
```

## قبل التشغيل — ضيف الملفات دي جنب main.py:
1. `skin_cancer_detection_final.keras`
2. `UNet_model.keras`
3. مجلد `data/` فيه:
   - `updated_file_2.xlsx`
   - `kb/*.md` (ملفات قاعدة المعرفة)

## التثبيت والتشغيل (Windows PowerShell)
```powershell
cd dermascan_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# ضبط مفتاح Groq (لجلسة الترمينال الحالية فقط)
$env:GROQ_API_KEY="ضع_المفتاح_هنا"

uvicorn main:app --reload --port 8000
```

بعد ما يظهر `Application startup complete`، افتح المتصفح على:
```
http://localhost:8000
```

هتلاقي واجهة الشات شغالة فعليًا: اختيار الدور بزرار، رفع صورة، تحليل حقيقي (تصنيف + سجمنتيشن)، تقرير الدكتور عند مطابقة صورة، درجة أولوية حجز المريض، وشات مع الـ AI.

## ملاحظات
- أول تشغيل بياخد وقت أطول (تحميل الموديلات + بناء الـ vectorstore) — طبيعي.
- الصور المسجّلة للدكاترة بتتخزن في `data/image_registry.json` (بيتعمل تلقائيًا).
- لو غيّرت بورت أو دومين، حدّث `API_BASE` في `static/index.html` لو هتستضيف الواجهة على سيرفر منفصل.
