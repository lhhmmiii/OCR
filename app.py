"""
app.py – Streamlit UI for the Invoice OCR → LLM → DB pipeline.

Run with:
    streamlit run app.py

Features
--------
- Upload an invoice image (JPG, PNG)
- Annotated image preview after OCR
- Editable extracted fields before saving
- One-click save to PostgreSQL
- History table of all saved invoices
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from decimal import Decimal

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config must be first Streamlit call ───────────────────────────────────
st.set_page_config(
    page_title="Invoice OCR",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (heavy ML libs) after page config ────────────────────────────
from src.ocr.pipeline import OCRPipeline
from src.schemas.ocr_config import OCRConfig, DetectionConfig, RecognitionConfig
from src.llm.extractor import BaseInvoiceExtractor
from src.schemas.invoice_schema import InvoiceExtract
from database.database import SessionLocal
from database.models import Invoice
from database.repository import save_invoice

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e2e8f0;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Section headers */
    h1, h2, h3 {
        color: #c4b5fd !important;
    }

    /* Upload widget */
    .stFileUploader > div {
        border: 2px dashed #7c3aed !important;
        border-radius: 12px !important;
        background: rgba(124, 58, 237, 0.08) !important;
    }

    /* Number inputs */
    input[type="number"] {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    /* Primary buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.45);
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.8rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-success { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #34d399; }
    .badge-info    { background: rgba(99,102,241,0.2);  color: #818cf8; border: 1px solid #818cf8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached resource initialisation ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_ocr_pipeline() -> OCRPipeline:
    cfg = OCRConfig(
        detection=DetectionConfig(min_score=0.5),
        recognition=RecognitionConfig(config_name="vgg_transformer", device="cpu"),
    )
    return OCRPipeline(config=cfg)


@st.cache_resource(show_spinner=False)
def get_extractor(
    provider: str,
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
) -> BaseInvoiceExtractor:
    from src.llm.extractor import OllamaInvoiceExtractor, GeminiInvoiceExtractor
    if provider.lower() == "gemini":
        return GeminiInvoiceExtractor(model=model, api_key=api_key)
    else:
        return OllamaInvoiceExtractor(model=model, host=host)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧾 Invoice OCR")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    min_det = st.slider("Min detection score", 0.1, 1.0, 0.5, 0.05)
    device = st.selectbox("OCR device", ["cpu", "cuda:0"])
    
    st.markdown("---")
    st.markdown("### 🤖 LLM Settings")
    llm_provider = st.selectbox("LLM Provider", ["Ollama", "Gemini"], index=0)
    
    if llm_provider == "Ollama":
        ollama_model = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"))
        ollama_host = st.text_input("Ollama Host", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        llm_model = ollama_model
        llm_param = ollama_host
        gemini_api_key = None
    else:
        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-pro-exp"],
            index=0
        )
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        llm_model = gemini_model
        llm_param = gemini_api_key
        ollama_host = None

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "Upload an invoice → OCR extracts raw text → "
        "LLM (Ollama or Gemini) structures the data → save to PostgreSQL."
    )
    if llm_provider == "Ollama":
        st.markdown(
            f'<span class="status-badge badge-info">Powered by Ollama ({llm_model})</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-badge badge-success">Powered by Gemini ({llm_model})</span>',
            unsafe_allow_html=True,
        )


# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <h1 style="font-size:2.8rem; font-weight:700; color:#c4b5fd;">
            🧾 Invoice OCR Pipeline
        </h1>
        <p style="color:#94a3b8; font-size:1.1rem;">
            Extract · Structure · Persist
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_upload, tab_history = st.tabs(["📤 Process Invoice", "📋 Invoice History"])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 – Process Invoice
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    uploaded = st.file_uploader(
        "Drop your invoice image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_img, col_fields = st.columns([1, 1], gap="large")

        # ── Step 1: OCR ────────────────────────────────────────────────────
        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🖼️ Invoice Preview")

            with tempfile.NamedTemporaryFile(suffix=f".{uploaded.name.split('.')[-1]}", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            st.image(tmp_path, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🔍 Run OCR + Extract", type="primary", use_container_width=True):
                with st.spinner("Running OCR…"):
                    pipeline = get_ocr_pipeline()
                    ocr_results = pipeline.run(tmp_path)
                    full_text = pipeline.get_full_text(ocr_results)

                st.session_state["ocr_text"] = full_text
                st.session_state["ocr_results"] = ocr_results
                st.session_state["extraction"] = None  # reset

                spinner_msg = f"Extracting with {llm_provider} ({llm_model})…"
                with st.spinner(spinner_msg):
                    try:
                        extractor = get_extractor(
                            provider=llm_provider,
                            model=llm_model,
                            host=ollama_host,
                            api_key=gemini_api_key,
                        )
                        invoice_extract = extractor.extract(full_text)
                        st.session_state["extraction"] = invoice_extract
                    except Exception as exc:
                        st.error(f"LLM extraction failed: {exc}")

            # Show detected text regions as markdown
            if st.session_state.get("ocr_results"):
                st.markdown("#### 📝 Detected Text Regions")
                rows_md = "| # | Text | Confidence |\n|---|------|------------|\n"
                for i, r in enumerate(st.session_state["ocr_results"], 1):
                    score = r.get("recognition_score")
                    score_str = f"{score:.2f}" if score is not None else "—"
                    rows_md += f"| {i} | {r['text']} | {score_str} |\n"
                st.markdown(rows_md)

        # ── Step 2: Review & Edit ──────────────────────────────────────────
        with col_fields:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🗂️ Extracted Fields")

            if st.session_state.get("extraction") is None:
                st.info("Click **Run OCR + Extract** to see results here.")
            else:
                inv: InvoiceExtract = st.session_state["extraction"]

                st.markdown("#### Header")
                vendor = st.text_input("Vendor", value=inv.vendor)
                date_val = st.date_input(
                    "Invoice Date",
                    value=inv.date.date() if inv.date else datetime.today().date(),
                )

                c1, c2 = st.columns(2)
                with c1:
                    subtotal = st.number_input("Subtotal", value=float(inv.subtotal), step=0.01, format="%.2f")
                    tax      = st.number_input("Tax",      value=float(inv.tax),      step=0.01, format="%.2f")
                    shipping = st.number_input("Shipping", value=float(inv.shipping), step=0.01, format="%.2f")
                with c2:
                    discount = st.number_input("Discount", value=float(inv.discount), step=0.01, format="%.2f")
                    total    = st.number_input("Total",    value=float(inv.total),    step=0.01, format="%.2f")

                st.markdown("#### Line Items")
                for i, item in enumerate(inv.items):
                    with st.expander(f"Item {i+1}: {item.name}", expanded=(i == 0)):
                        ic1, ic2, ic3 = st.columns(3)
                        item.name     = ic1.text_input("Name",     key=f"name_{i}",  value=item.name)
                        item.quantity = Decimal(str(ic2.number_input("Qty",  key=f"qty_{i}",   value=float(item.quantity), step=0.01)))
                        item.price    = Decimal(str(ic3.number_input("Price",key=f"price_{i}", value=float(item.price),    step=0.01, format="%.2f")))

                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                col_save, col_metrics = st.columns([1, 2])

                with col_metrics:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Items",    len(inv.items))
                    m2.metric("Total",    f"${total:,.2f}")
                    m3.metric("Tax",      f"${tax:,.2f}")

                with col_save:
                    if st.button("💾 Save to Database", type="primary", use_container_width=True):
                        updated_extract = InvoiceExtract(
                            vendor=vendor,
                            date=datetime.combine(date_val, datetime.min.time()),
                            subtotal=Decimal(str(subtotal)),
                            discount=Decimal(str(discount)),
                            tax=Decimal(str(tax)),
                            shipping=Decimal(str(shipping)),
                            total=Decimal(str(total)),
                            items=inv.items,
                        )
                        with st.spinner("Saving…"):
                            db = SessionLocal()
                            try:
                                saved = save_invoice(db, updated_extract)
                                st.success(
                                    f"✅ Invoice saved!  "
                                    f"**ID: {saved.invoice_id}** · "
                                    f"Vendor: {saved.vendor}"
                                )
                                st.session_state["extraction"] = None
                            except Exception as exc:
                                st.error(f"DB error: {exc}")
                            finally:
                                db.close()

        # OCR raw text expander
        if st.session_state.get("ocr_text"):
            with st.expander("📄 Raw OCR text"):
                st.text(st.session_state.get("ocr_text", ""))

    else:
        # Empty state
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <div style="font-size:4rem;">📂</div>
                <h3 style="color:#c4b5fd;">Upload an Invoice to Get Started</h3>
                <p style="color:#94a3b8;">Supports JPG and PNG formats</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 – Invoice History
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📋 Saved Invoices")

    if st.button("🔄 Refresh"):
        st.rerun()

    db = SessionLocal()
    try:
        invoices: list[Invoice] = db.query(Invoice).order_by(Invoice.invoice_id.desc()).limit(50).all()
    finally:
        db.close()

    if not invoices:
        st.info("No invoices saved yet. Process one in the **Process Invoice** tab!")
    else:
        import pandas as pd
        rows = [
            {
                "ID":       inv.invoice_id,
                "Vendor":   inv.vendor,
                "Date":     inv.date.strftime("%Y-%m-%d") if inv.date else "—",
                "Subtotal": f"${inv.subtotal:,.2f}",
                "Tax":      f"${inv.tax:,.2f}",
                "Discount": f"${inv.discount:,.2f}",
                "Shipping": f"${inv.shipping:,.2f}",
                "Total":    f"${inv.total:,.2f}",
                "Items":    len(inv.items),
            }
            for inv in invoices
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detail view
        selected_id = st.selectbox("Select invoice to view details", [r["ID"] for r in rows])
        if selected_id:
            db = SessionLocal()
            try:
                inv = db.get(Invoice, selected_id)
                if inv and inv.items:
                    st.markdown(f"#### 🧾 Line Items — Invoice #{inv.invoice_id} ({inv.vendor})")
                    item_rows = [
                        {
                            "Name":     it.name,
                            "Qty":      float(it.quantity),
                            "Price":    f"${it.price:,.2f}",
                            "Savings":  f"${it.savings:,.2f}",
                        }
                        for it in inv.items
                    ]
                    st.dataframe(pd.DataFrame(item_rows), use_container_width=True, hide_index=True)
            finally:
                db.close()


# ── Initialise session state ───────────────────────────────────────────────────
if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""
if "ocr_results" not in st.session_state:
    st.session_state["ocr_results"] = []
if "extraction" not in st.session_state:
    st.session_state["extraction"] = None
