
import os
import re
import io
import hashlib

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
from document_intelligence import ask_groq as validated_ask_groq

st.set_page_config(
    page_title="FinSight AI | Document Intelligence",
    page_icon="📊",
    layout="wide"
)

st.markdown('\n<style>\n/* FinSight Premium UI Theme */\n/* App background and typography */\n.stApp {\n    background: linear-gradient(145deg, #f5f7fb 0%, #eef2f8 100%);\n    color: #17243b;\n}\n.block-container {\n    padding-top: 2rem;\n    padding-bottom: 3rem;\n    max-width: 1500px;\n}\nh1, h2, h3 {\n    color: #142541 !important;\n    letter-spacing: -0.025em;\n}\nh1 { font-weight: 750 !important; }\nh2, h3 { font-weight: 650 !important; }\n\n/* Sidebar */\nsection[data-testid="stSidebar"] {\n    background: linear-gradient(180deg, #101d35 0%, #192c4b 100%);\n    border-right: 1px solid rgba(255,255,255,0.08);\n}\nsection[data-testid="stSidebar"] * {\n    color: #edf3ff;\n}\nsection[data-testid="stSidebar"] [data-testid="stRadio"] label {\n    padding: 0.45rem 0.55rem;\n    border-radius: 8px;\n}\nsection[data-testid="stSidebar"] hr {\n    border-color: rgba(255,255,255,0.16);\n}\n\n/* Metrics */\ndiv[data-testid="stMetric"] {\n    background: #ffffff;\n    border: 1px solid #e3eaf4;\n    border-radius: 14px;\n    padding: 18px 20px;\n    box-shadow: 0 4px 16px rgba(24, 45, 78, 0.045);\n}\ndiv[data-testid="stMetricLabel"] {\n    color: #61718a;\n    font-size: 0.88rem;\n}\ndiv[data-testid="stMetricValue"] {\n    color: #173a67;\n    font-weight: 750;\n}\n\n/* Containers and charts */\ndiv[data-testid="stVerticalBlockBorderWrapper"] {\n    border-radius: 14px;\n}\ndiv[data-testid="stPlotlyChart"],\ndiv[data-testid="stImage"],\ndiv[data-testid="stTable"] {\n    border-radius: 12px;\n}\ndiv[data-testid="stAlert"] {\n    border-radius: 12px;\n}\n\n/* Inputs */\ndiv[data-baseweb="input"] > div,\ndiv[data-baseweb="select"] > div,\ndiv[data-baseweb="textarea"] > div {\n    border-radius: 10px;\n}\n.stTextInput input, .stNumberInput input, .stTextArea textarea {\n    background: #ffffff;\n}\n\n/* Buttons */\n.stButton > button,\n.stDownloadButton > button,\ndiv[data-testid="stFormSubmitButton"] > button {\n    border-radius: 10px;\n    border: 1px solid #244b7b;\n    font-weight: 650;\n    transition: all 0.15s ease;\n}\n.stButton > button[kind="primary"],\ndiv[data-testid="stFormSubmitButton"] > button[kind="primary"] {\n    background: #193d68;\n    color: white;\n}\n.stButton > button:hover,\n.stDownloadButton > button:hover {\n    border-color: #477cb6;\n    box-shadow: 0 4px 12px rgba(25, 61, 104, 0.12);\n}\n\n/* Tabs and dividers */\nbutton[data-baseweb="tab"] {\n    font-weight: 600;\n}\nhr {\n    border-color: #dfe6f0;\n}\n\n/* File uploader */\n[data-testid="stFileUploader"] {\n    background: rgba(255,255,255,0.78);\n    border: 1px dashed #bdcce0;\n    border-radius: 14px;\n    padding: 10px;\n}\n\n/* Make narrow screens comfortable */\n@media (max-width: 768px) {\n    .block-container {\n        padding: 1rem 1rem 2rem;\n    }\n    div[data-testid="stMetric"] {\n        padding: 12px;\n    }\n}\n</style>\n', unsafe_allow_html=True)



@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def process_pdf(pdf_bytes, model, chunk_size=1200, overlap=200):
    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages = []
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        pages.append({
            "page": page_number,
            "text": text
        })

        start = 0
        while start < len(text):
            chunk_text = text[start:start + chunk_size]

            if chunk_text.strip():
                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

            start += chunk_size - overlap

    if not chunks:
        raise ValueError(
            "No readable text found. This PDF may require OCR."
        )

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings
    }


def search_document_data(question, document, model, top_k=5):
    chunks = document["chunks"]
    embeddings = document["embeddings"]

    if not chunks:
        return []

    query_vector = model.encode(
        [question], normalize_embeddings=True
    )[0]
    semantic_scores = embeddings @ query_vector

    stop_words = {
        "what", "was", "were", "the", "how", "much",
        "did", "does", "from", "for", "in", "of",
        "and", "a", "an", "is", "to", "by", "with",
        "company", "report", "fiscal", "year",
        "apple", "2025", "2024", "2023"
    }

    terms = [
        word for word in re.findall(r"[a-z0-9]+", question.lower())
        if word not in stop_words and len(word) > 1
    ]

    keyword_scores = np.array([
        sum(chunk["text"].lower().count(term) for term in terms)
        for chunk in chunks
    ], dtype=float)

    semantic_rank = np.argsort(semantic_scores)[::-1]
    keyword_rank = np.argsort(keyword_scores)[::-1]

    rrf_scores = {}
    for ranking in (semantic_rank, keyword_rank):
        for rank, index in enumerate(ranking, start=1):
            rrf_scores[index] = (
                rrf_scores.get(index, 0) + 1 / (60 + rank)
            )

    q = question.lower()

    if "operating income" in q:
        metric_patterns = [r"\boperating income\b"]
    elif "net income" in q:
        metric_patterns = [r"\bnet income\b"]
    elif "services" in q and ("revenue" in q or "sales" in q):
        metric_patterns = [r"\bservices\b", r"\bnet sales\b"]
    elif "total net sales" in q or "total sales" in q:
        metric_patterns = [r"\btotal net sales\b"]
    else:
        metric_patterns = []

    scored = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"].lower()
        score = rrf_scores.get(i, 0)

        for pattern in metric_patterns:
            if re.search(pattern, text):
                score += 0.025

        if "consolidated statements of operations" in text:
            score += 0.020

        if "products and services performance" in text:
            score += 0.020

        if "in millions" in text or "dollars in millions" in text:
            score += 0.005

        scored.append({
            "page": chunk["page"],
            "text": chunk["text"],
            "score": score,
            "_index": i
        })

    scored.sort(key=lambda item: item["score"], reverse=True)

    # Keep the highest-ranked chunk from each page.
    results = []
    seen_pages = set()

    for item in scored:
        if item["page"] in seen_pages:
            continue

        seen_pages.add(item["page"])
        results.append({
            "page": item["page"],
            "text": item["text"],
            "score": item["score"]
        })

        if len(results) >= top_k:
            break

    return results




def ask_groq(question, results, document=None):
    """Use the shared answer function with year validation."""
    return validated_ask_groq(
        question,
        results,
        document=document
    )


# ---------------- USER INTERFACE ----------------

st.title("📊 FinSight AI")
st.subheader("Financial Document Intelligence")

st.write(
    "Upload a financial report and ask questions about "
    "its revenue, profitability, and other financial metrics."
)

st.info(
    "Prototype: answers depend on extracted PDF text and "
    "the passages retrieved. Always verify important figures "
    "against the cited report pages."
)

uploaded_file = st.file_uploader(
    "Upload a financial PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    st.caption(f"Selected file: {uploaded_file.name}")

    if st.session_state.get("file_hash") != file_hash:
        st.session_state.pop("document", None)
        st.session_state.pop("file_hash", None)

    if st.button("📄 Process PDF", type="primary"):
        try:
            with st.spinner(
                "Extracting text and creating embeddings..."
            ):
                model = load_embedding_model()
                document = process_pdf(pdf_bytes, model)

                st.session_state["document"] = document
                st.session_state["file_hash"] = file_hash
                st.session_state["file_name"] = uploaded_file.name

            st.success("PDF processed successfully!")

        except Exception as error:
            st.error(f"Could not process PDF: {error}")

if st.session_state.get("document") is not None:
    document = st.session_state["document"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Readable pages", len(document["pages"]))
    col2.metric("Text chunks", len(document["chunks"]))
    col3.metric("Embedding dimensions", document["embeddings"].shape[1])

    st.divider()
    st.subheader("Ask your financial report")

    question = st.text_input(
        "Your question",
        placeholder="What was the operating income in 2025?"
    )

    if st.button("🔎 Ask FinSight AI"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                model = load_embedding_model()

                with st.spinner("Searching the report..."):
                    results = search_document_data(
                        question, document, model, top_k=5
                    )

                with st.spinner("Generating an evidence-based answer..."):
                    answer = ask_groq(question, results, document=document)

                st.subheader("Answer")
                st.markdown(answer)

                st.subheader("Retrieved evidence")

                for index, item in enumerate(results, start=1):
                    with st.expander(
                        f"Evidence {index} — PDF page {item['page']}"
                    ):
                        st.write(item["text"])

            except Exception as error:
                st.error(f"Unable to answer the question: {error}")

else:
    st.caption(
        "Upload a PDF and click Process PDF to get started."
    )
