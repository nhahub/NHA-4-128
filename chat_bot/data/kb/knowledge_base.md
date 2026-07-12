# DermaScan Knowledge Base (Consolidated)

## 1. Diagnoses (1000 patients)
- vascular 146 (14.6%) | benign_keratosis 135 (13.5%) | basal_cell_carcinoma 128 (12.8%)
  | nevus 124 (12.4%) | dermatofibroma 121 (12.1%) | squamous_cell_carcinoma 121 (12.1%)
  | actinic_keratosis 117 (11.7%) | melanoma 108 (10.8%)
- Biopsy results: confirmed_benign 264 | not_done 254 | confirmed_malignant 250 | suspected 232
- Melanoma: most dangerous, needs urgent biopsy. BCC: most common malignant type, usually slow-growing.
  SCC: can metastasize if untreated. Actinic keratosis: pre-cancerous, UV-caused.
  Nevus & benign keratosis: usually non-cancerous but need monitoring. Dermatofibroma: benign nodule,
  rarely malignant. Vascular lesions: mostly benign (hemangiomas etc.).

## 2. Risk Factors
- Avg age by diagnosis: melanoma 43.4 | dermatofibroma 46.5 | benign_keratosis 46.6 | BCC 46.8
  | nevus 47.7 | SCC 48.1 | vascular 48.7 | actinic_keratosis 51.1
- High UV exposure raises risk of melanoma, BCC, SCC, actinic keratosis most.
- Fair skin (Fitzpatrick 1-2), prior skin cancer, immunosuppression (transplant/HIV),
  >50 moles or atypical moles, childhood blistering sunburns, and genetic mutations
  (CDKN2A, BAP1, BRCA2, MC1R, TP53) all raise lifetime risk.

## 3. Lesion Characteristics & ABCDE Rule
- Common colors: amelanotic, mixed, brown, tan, dark_brown, pink, red, black.
- Common locations: chest, neck, abdomen, face, shoulder, arm, back, forearm, scalp, hand, leg.
- Avg size ~17-19mm across diagnoses; avg border irregularity ~0.45-0.53; avg asymmetry ~0.47-0.52
  (melanoma and nevus tend toward the higher end of both).
- **ABCDE**: Asymmetry (score >0.5 is concerning) · Border irregularity (>0.6 suspicious)
  · Color (multiple shades of brown/black/red/pink/white/blue) · Diameter (>6mm eraser-size,
  >10mm needs evaluation) · Evolution (any change in size/shape/color, new bleeding).

## 4. Genetics & Hereditary Risk
- Mutations in dataset: MC1R 140 | BAP1 130 | BRCA2 130 | CDKN2A 110 patients.
- **CDKN2A**: leading hereditary melanoma gene, up to 70% lifetime melanoma risk.
- **BAP1**: linked to uveal melanoma, mesothelioma, cutaneous BAP1-inactivated tumors.
- **MC1R**: red hair/fair skin gene, doubles melanoma risk even without sun exposure.
- **BRCA2**: primarily breast/ovarian cancer gene, also raises melanoma risk.
- **TP53**: Li-Fraumeni syndrome, broad cancer predisposition incl. skin cancers.
- Hereditary risk score: 0.0-0.3 LOW (annual screening) | 0.3-0.6 MODERATE (annual screening) |
  0.6-0.8 HIGH (biannual/every-3-months dermatology visits) | 0.8-1.0 VERY HIGH
  (genetic counseling + prophylactic measures).
- Family risk levels (100 families): MEDIUM 40 | LOW 31 | HIGH 29.
  Highest-risk families: F015 (0.772, MC1R) · F043 (0.770, MC1R) · F016 (0.769, BRCA2)
  · F004 (0.752, BAP1) · F025 (0.745, BRCA2) — all HIGH.

## 5. Patient Management & Follow-Up
- Biopsy indications: asymmetry or border irregularity score >0.7; lesion >10mm that changed
  in color/shape/size; genetic mutation (CDKN2A/BAP1/TP53) with a new lesion; ulceration,
  bleeding, or crusting; clinically suspected melanoma/SCC.
- Follow-up frequency: LOW risk (score <0.3, no prior cancer) → annual. MEDIUM (0.3-0.6 or
  family history) → every 6 months. HIGH (>0.6, known mutation, prior cancer) → every 3 months.
  Post-biopsy confirmed malignant → every 3 months for 2 years, then every 6 months.
- Immunosuppressed patients (transplant, HIV, autoimmune meds): 65x higher SCC risk, need
  visits every 3 months regardless of score, and immediate biopsy for any suspicious lesion.
- Patient education: daily SPF 50+, avoid tanning beds (+75% melanoma risk), monthly ABCDE
  self-exams, protective clothing, shade 10am-4pm, inform family about hereditary mutations.

## 6. Dataset Source
- Built on HAM10000-style dermoscopy data (~1000 images/records), 8 diagnostic categories,
  confirmed via biopsy, expert consensus, or follow-up. Used to train/validate the
  classification and segmentation models behind this app's image analysis.
