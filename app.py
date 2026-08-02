# ============================================================
# app.py
# Purpose : Streamlit UI for RAG Document Q&A System
# Author  : Venkata Sai Karthik Pyla
# ============================================================

import streamlit as st
import os
import sys
import tempfile
import warnings
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.abspath("src"))

from pdf_loader import load_pdf, chunk_text
from embeddings import generate_embeddings
from vector_store import build_index, search
from llm_handler import get_answer


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)
# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    
    /* Card styling */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: 700;
        font-size: 1rem;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 10px 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255,255,255,0.03);
    }
    
    /* Metric */
    .stMetric {
        background: rgba(0,210,255,0.1);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(0,210,255,0.3);
    }
    
    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(0,255,100,0.1);
        border: 1px solid rgba(0,255,100,0.3);
        border-radius: 10px;
    }
    
    /* Info message */
    .stInfo {
        background: rgba(0,210,255,0.1);
        border: 1px solid rgba(0,210,255,0.3);
        border-radius: 10px;
    }
    
    /* Chat input */
    .stChatInputContainer {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 25px;
    }
    
    /* Subheader */
    h2, h3 {
        color: #00d2ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# App Header
# ============================================================
st.title("📄 RAG Document Q&A System")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔍 Search Engine", "FAISS")
with col2:
    st.metric("🤖 LLM", "Groq Llama3")
with col3:
    st.metric("🧠 Embeddings", "MiniLM-L6")

st.markdown("""
> 🚀 Upload any PDF → Ask questions → Get instant AI answers!
""")

st.divider()


# ============================================================
# Session State — Stores data between interactions!
# Like RAM for our Streamlit app!
# ============================================================
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# PDF Upload Section
# ============================================================
st.subheader("📁 Step 1 — Upload Your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload any PDF document to ask questions about!"
)

if uploaded_file is not None:

    # Process PDF button
    if st.button("🚀 Process PDF", type="primary"):

        with st.spinner("Reading and processing your PDF..."):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Run full pipeline
            raw_text = load_pdf(tmp_path)
            chunks = chunk_text(raw_text)
            embeddings = generate_embeddings(chunks)
            index = build_index(embeddings)

            # Store in session state
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.pipeline_ready = True
            st.session_state.chat_history = []

            # Cleanup temp file
            os.unlink(tmp_path)

        st.success(f"✅ PDF processed! Created {len(chunks)} chunks. Ready for questions!")


# ============================================================
# Q&A Section
# ============================================================
if st.session_state.pipeline_ready:

    st.divider()
    st.subheader("💬 Step 2 — Ask Questions!")

    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Question input
    question = st.chat_input("Ask anything about your document...")

    if question:

        # Show user question
        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Search relevant chunks
                results = search(
                    question,
                    st.session_state.index,
                    st.session_state.chunks
                )

                # Extract chunks
                context_chunks = [r["chunk"] for r in results]

                # Get LLM answer
                answer = get_answer(question, context_chunks)

                st.write(answer)

        # Save to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

else:
    st.info("👆 Upload a PDF and click Process to get started!")


# ============================================================
# Sidebar — Project Info
# ============================================================
with st.sidebar:
    st.header("ℹ️ About This Project")
    st.markdown("""
    **RAG Document Q&A System**
    
    Built by Venkata Sai Karthik Pyla
    
    **Tech Stack:**
    - 🐍 Python
    - 📄 pypdf
    - ✂️ LangChain
    - 🔢 SentenceTransformers
    - 🔍 FAISS
    - 🤖 Groq (Llama3)
    - 🎨 Streamlit
    
    **How it works:**
    1. Upload any PDF
    2. Text extracted & chunked
    3. Chunks converted to embeddings
    4. FAISS indexes embeddings
    5. Questions searched via FAISS
    6. Groq LLM generates answers
    
    **GitHub:**
    [View Source Code](https://github.com/PYLA-VENKATA-SAI-KARTHIK/rag-document-assistant)
    """)

    st.divider()
    
    if st.session_state.pipeline_ready:
        st.success("✅ Pipeline Active")
        st.metric(
            "Chunks Indexed",
            len(st.session_state.chunks)
        )
    else:
        st.warning("⏳ Waiting for PDF...")