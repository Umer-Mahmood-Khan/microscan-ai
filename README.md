# RadiOCR — Clinical Radiology Report Extractor

An end-to-end OCR + LLM pipeline that extracts structured clinical fields from scanned radiology report PDFs. Built for radiologists to summarise reports faster without replacing clinical judgement.

**Live demo:** https://cfxf4o325p6lwux8me8ugz.streamlit.app

---

## What it does

Upload a scanned radiology report PDF. RadiOCR extracts 13 structured clinical fields automatically — patient ID, modality, primary finding, location, impression, follow-up recommendation — in seconds.

**Extracted fields:**
- Patient ID
- Study date
- Modality (CT Brain, MRI, X-Ray etc.)
- Clinical indication
- Primary finding
- Finding location
- Size / extent
- Impression summary
- Urgency
- Follow-up recommended
- Reporting radiologist
- Institution
- OCR confidence (HIGH / PARTIAL / LOW)

---

## Architecture
```
Upload scanned PDF
       |
       v
pdf2image converts pages to images (300 DPI)
       |
       v
Tesseract OCR extracts raw text
       |
       v
GPT-4o-mini structures text into 13 clinical fields
       |
       v
Clean table displayed + downloadable as CSV / JSON
```

---

## Tech stack

| Component | Technology |
|---|---|
| PDF to image | pdf2image + poppler |
| OCR engine | Tesseract (oem 3, psm 6) |
| LLM structuring | GPT-4o-mini |
| Output formats | CSV, JSON |
| UI | Streamlit |
| Language | Python 3.11 |

---

## Why Tesseract over TrOCR

TrOCR is more accurate on handwritten or low-quality scans. For clean printed radiology reports, Tesseract with LSTM engine (oem 3) is faster, requires no GPU, and produces equivalent accuracy. This is a deliberate engineering decision — the right tool for the specific input type.

---

## Ethical design

This tool was built with responsible AI principles from the start:

- The tool never generates a diagnosis — it only extracts what the radiologist already wrote
- No patient data is stored — everything processes in-session only
- An OCR confidence field (HIGH / PARTIAL / LOW) flags low-quality scans so the doctor knows to check the original
- Every output carries a disclaimer: clinical judgement stays with the physician
- The radiologist uploads their own reports — no third party handling of patient data

Responsible AI in healthcare is not about adding disclaimers at the bottom. It is about making the right architectural decisions before writing a single line of code.

---

## Project structure
```
microscan-ai/
├── app.py              # Streamlit UI and main logic
├── requirements.txt    # Python dependencies
└── packages.txt        # System dependencies for Streamlit Cloud
```

---

## Running locally
```bash
# Clone the repo
git clone https://github.com/Umer-Mahmood-Khan/microscan-ai.git
cd microscan-ai

# Install system dependencies (Ubuntu / Debian)
sudo apt-get install tesseract-ocr poppler-utils

# Install Python dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will ask for your OpenAI API key in the UI — no need to set environment variables locally.

---

## Context

Built as part of active research at the National Center for Artificial Intelligence (NCAI), Pakistan. The work connects to ongoing projects in:

- Cerebral microbleed detection from CT scans using fine-tuned YOLOv8 and DETR
- OCR extraction from hospital discharge reports (HDRs)
- Multi-agent crop monitoring systems for Pakistani farmers

RadiOCR demonstrates the document processing layer of the clinical AI pipeline — converting unstructured scanned reports into structured, queryable records.

---

## Companion project

MedRAG — Medical Knowledge Agent (RAG + FAISS + AWS S3):
https://github.com/Umer-Mahmood-Khan/medrag

---

## Built by

**Umer Mahmood Khan**
AI Research Engineer at National Center for Artificial Intelligence (NCAI), Pakistan

GitHub: https://github.com/Umer-Mahmood-Khan
