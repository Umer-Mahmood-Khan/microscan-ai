
import streamlit as st
import pytesseract
import json
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from openai import OpenAI
import tempfile, os

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="RadiOCR — Radiology Report Extractor",
    page_icon="🏥",
    layout="centered"
)

# ── Ethical disclaimer banner ──────────────────────────────
st.warning(
    "**Clinical decision support tool only.** This tool assists radiologists in "
    "summarising existing reports. It does not perform diagnosis, does not store "
    "any patient data, and does not replace clinical judgement. All extracted "
    "output must be reviewed by a qualified physician before any clinical decision."
)

# ── Header ─────────────────────────────────────────────────
st.title("RadiOCR")
st.markdown(
    "Upload a scanned radiology report PDF. The tool extracts structured clinical "
    "fields using OCR + GPT-4o-mini and returns a clean summary table."
)
st.divider()

# ── Sidebar — about ────────────────────────────────────────
with st.sidebar:
    st.markdown("### About this tool")
    st.markdown(
        "Built by an AI Research Engineer at NCAI Pakistan.\n\n"
        "**Pipeline:**\n"
        "1. PDF uploaded by radiologist\n"
        "2. pdf2image converts pages to images\n"
        "3. Tesseract OCR extracts raw text\n"
        "4. GPT-4o-mini structures the text into clinical fields\n"
        "5. Output displayed + downloadable as CSV\n\n"
        "**No data is stored.** Everything is processed in-session only."
    )
    st.divider()
    st.markdown("### Extracted fields")
    for f in [
        "Patient ID","Study date","Modality","Clinical indication",
        "Primary finding","Location","Size / extent","Impression",
        "Urgency","Follow-up","Reporting radiologist","Institution","OCR confidence"
    ]:
        st.markdown(f"- {f}")

# ── API key input ──────────────────────────────────────────
st.markdown("#### OpenAI API key")
st.markdown(
    "Your key is used only for this session and is never stored. "
    "Get a free key at [platform.openai.com](https://platform.openai.com)."
)
api_key = st.text_input("Paste your OpenAI API key", type="password")

# ── File uploader ──────────────────────────────────────────
st.markdown("#### Upload radiology report")
uploaded_file = st.file_uploader(
    "Accepts scanned PDF files",
    type=["pdf"],
    help="Upload a scanned radiology report. Demo works with any printed PDF."
)

# ── OCR function ───────────────────────────────────────────
def ocr_pdf(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    all_text = []
    for page in pages:
        raw = pytesseract.image_to_string(page, config="--oem 3 --psm 6")
        lines = raw.split("\n")
        cleaned, prev_blank = [], False
        for line in lines:
            s = line.strip()
            if s == "" and prev_blank:
                continue
            cleaned.append(s)
            prev_blank = s == ""
        all_text.append("\n".join(cleaned).strip())
    return "\n\n".join(all_text)

# ── Extraction function ────────────────────────────────────
EXTRACTION_PROMPT = """
You are a clinical document processing assistant.
Extract structured information from raw OCR text of a radiology report.

STRICT RULES:
- Extract only what is explicitly stated in the text
- If a field is not present write NOT FOUND — do not guess or infer
- Do not add any medical interpretation or diagnosis of your own
- Do not modify clinical terminology — copy it exactly as written
- Return ONLY a valid JSON object no explanation no markdown no code fences

Extract these fields:
{{
  "patient_id": "patient ID or reference number",
  "study_date": "date the scan was performed",
  "modality": "type of scan e.g. CT Brain MRI X-Ray",
  "clinical_indication": "reason the scan was requested one sentence max",
  "primary_finding": "most important finding from the FINDINGS section",
  "finding_location": "anatomical location of the primary finding",
  "finding_size": "size or extent if mentioned else NOT FOUND",
  "impression_summary": "radiologist conclusion from IMPRESSION section 1-2 sentences",
  "urgency": "urgent / routine / not specified",
  "follow_up": "recommended follow-up scan or action",
  "reporting_radiologist": "name of the radiologist who signed the report",
  "institution": "hospital or clinic name",
  "ocr_confidence": "your assessment of OCR text quality: HIGH / PARTIAL / LOW"
}}

RAW OCR TEXT:
{raw_text}
"""

def extract_fields(raw_text, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a clinical document processing assistant. "
                                          "You extract structured data from radiology reports. "
                                          "You never diagnose. You never infer. You only extract."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(raw_text=raw_text)}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content.strip())

# ── Main logic ─────────────────────────────────────────────
if uploaded_file and api_key:
    st.divider()

    with st.spinner("Step 1/3 — converting PDF to image..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        raw_text = ocr_pdf(tmp_path)
        os.unlink(tmp_path)

    with st.spinner("Step 2/3 — running OCR..."):
        st.success(f"OCR complete — {len(raw_text)} characters extracted")

    with st.spinner("Step 3/3 — extracting structured fields with GPT-4o-mini..."):
        try:
            structured = extract_fields(raw_text, api_key)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    st.success("Extraction complete")
    st.divider()

    # ── Results ────────────────────────────────────────────
    st.markdown("### Extracted clinical fields")

    field_labels = {
        "patient_id":"Patient ID","study_date":"Study date","modality":"Modality",
        "clinical_indication":"Clinical indication","primary_finding":"Primary finding",
        "finding_location":"Location","finding_size":"Size / extent",
        "impression_summary":"Impression","urgency":"Urgency",
        "follow_up":"Follow-up recommended","reporting_radiologist":"Reporting radiologist",
        "institution":"Institution","ocr_confidence":"OCR confidence",
    }

    # Metrics row — highlight key fields at the top
    col1, col2, col3 = st.columns(3)
    col1.metric("Patient ID",   structured.get("patient_id",  "NOT FOUND"))
    col2.metric("Study date",   structured.get("study_date",  "NOT FOUND"))
    col3.metric("OCR quality",  structured.get("ocr_confidence", "NOT FOUND"))

    st.markdown("")

    # Full table
    rows = [{"Field": label, "Extracted value": structured.get(key, "NOT FOUND")}
            for key, label in field_labels.items()]
    df = pd.DataFrame(rows)

    def colour_missing(val):
        return "background-color: #fff3cd" if val == "NOT FOUND" else ""

    st.dataframe(
        df.style.applymap(colour_missing, subset=["Extracted value"]),
        use_container_width=True,
        hide_index=True
    )

    # Raw OCR expander — transparency feature
    with st.expander("View raw OCR text (for verification)"):
        st.text(raw_text)

    # Download buttons
    st.divider()
    st.markdown("### Download results")
    col_a, col_b = st.columns(2)

    with col_a:
        st.download_button(
            label="Download as CSV",
            data=df.to_csv(index=False),
            file_name="extracted_report.csv",
            mime="text/csv"
        )
    with col_b:
        st.download_button(
            label="Download as JSON",
            data=json.dumps(structured, indent=2),
            file_name="extracted_report.json",
            mime="application/json"
        )

    # Ethical footer
    st.divider()
    st.caption(
        "This tool is a clinical decision support aid only. "
        "It does not store data, does not diagnose, and does not replace a physician. "
        "Built with responsible AI principles at NCAI Pakistan."
    )

elif uploaded_file and not api_key:
    st.info("Please enter your OpenAI API key above to run extraction.")

elif api_key and not uploaded_file:
    st.info("Please upload a radiology report PDF to begin.")

else:
    st.markdown("#### How it works")
    col1, col2, col3 = st.columns(3)
    col1.info("**1. Upload** a scanned radiology report PDF")
    col2.info("**2. OCR** extracts the raw text from the scan")
    col3.info("**3. GPT-4o-mini** structures it into clinical fields")
