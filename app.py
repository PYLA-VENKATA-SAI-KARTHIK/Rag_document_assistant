# ============================================================
# app.py - RAG Document Q&A System
# Author: Venkata Sai Karthik Pyla
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
# Page Config
# ============================================================
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# Beautiful CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: #0d1117;
    }
    
    /* Hero Section */
    .hero {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border-radius: 20px;
        border: 1px solid #30363d;
        margin-bottom: 2rem;
    }
    
    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #bc8cff, #ff7b72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero p {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    
    /* Upload Card */
    .upload-card {
        background: #161b22;
        border: 2px dashed #30363d;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s;
    }
    
    .upload-card:hover {
        border-color: #58a6ff;
    }
    
    /* Stats bar */
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Chat container */
    .chat-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 1.5rem;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* User message */
    .user-msg {
        background: #1f6feb;
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        margin: 8px 0;
        margin-left: 20%;
        font-size: 0.95rem;
    }
    
    /* Bot message */
    .bot-msg {
        background: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        margin: 8px 0;
        margin-right: 20%;
        font-size: 0.95rem;
    }
    
    /* Process button */
    .stButton > button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    
    .stButton > button:hover {
        opacity: 0.85;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Input styling */
    .stChatInput input {
        background: #21262d;
        border: 1px solid #30363d;
        color: #e6edf3;
        border-radius: 8px;
    }

    h2, h3 {
        color: #e6edf3 !important;
    }
    
    .stSuccess {
        background: rgba(35, 134, 54, 0.1);
        border: 1px solid #238636;
        border-radius: 8px;
    }
    
    .stInfo {
        background: rgba(31, 111, 235, 0.1);
        border: 1px solid #1f6feb;
        border-radius: 8px;
    }
    
    .stSpinner {
        color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ============================================================
# Layout — Two Columns like NotebookLM!
# ============================================================
left_col, right_col = st.columns([1, 2])

# ============================================================
# LEFT COLUMN — Upload & Info
# ============================================================
with left_col:

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🧠 DocuMind</h1>
        <p>Your AI Document Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    st.markdown("### 📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown(f"**📄 {uploaded_file.name}**")

        if st.button("⚡ Analyze Document"):
            with st.spinner("Analyzing your document..."):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                raw_text = load_pdf(tmp_path)
                chunks = chunk_text(raw_text)
                embeddings = generate_embeddings(chunks)
                index = build_index(embeddings)

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pipeline_ready = True
                st.session_state.chat_history = []
                st.session_state.pdf_name = uploaded_file.name

                os.unlink(tmp_path)

            st.success("✅ Document analyzed!")

    # Stats
    if st.session_state.pipeline_ready:
        st.markdown("### 📊 Document Stats")
        st.metric("📄 Document", st.session_state.pdf_name[:20] + "...")
        st.metric("🧩 Chunks Created", len(st.session_state.chunks))
        st.metric("🔢 Vector Dimensions", "384")
        st.metric("🤖 LLM Model", "Llama 3.1")

    # How it works
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    1. 📄 **Upload** any PDF document
    2. ✂️ **Chunks** text into pieces
    3. 🔢 **Embeds** chunks as vectors
    4. 🔍 **FAISS** indexes vectors
    5. 💬 **Ask** any question
    6. 🤖 **Groq AI** answers instantly!
    """)

    # Built by
    st.markdown("---")
    st.markdown("""
    **Built by Venkata Sai Karthik Pyla**
    
    [🔗 GitHub](https://github.com/PYLA-VENKATA-SAI-KARTHIK/Rag_document_assistant)
    [💼 LinkedIn](https://www.linkedin.com/in/venkata-sai-karthik-pyla)
    """)

# ============================================================
# RIGHT COLUMN — Chat Interface
# ============================================================
with right_col:

    st.markdown("### 💬 Ask Your Document")

    if not st.session_state.pipeline_ready:
        st.markdown("""
        <div style="
            background: #161b22;
            border: 2px dashed #30363d;
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            color: #8b949e;
            margin-top: 2rem;
        ">
            <h2 style="color: #58a6ff !important;">🧠 DocuMind AI</h2>
            <p style="font-size: 1.1rem;">
                👈 Upload a PDF on the left and click Analyze!
            </p>
            <p>Powered by FAISS + Groq Llama3 + SentenceTransformers</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Show chat history first
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

        # Suggested questions only if no chat yet
        if len(st.session_state.chat_history) == 0:
            st.markdown("**💡 Try asking:**")
            cols = st.columns(2)
            questions = [
                "What is this document about?",
                "Summarize the key points",
                "What are the main skills?",
                "What projects are mentioned?"
            ]
            for i, q in enumerate(questions):
                with cols[i % 2]:
                    if st.button(q, key=f"suggest_{i}"):
                        st.session_state.chat_history.append({
                            "question": q,
                            "answer": "thinking..."
                        })
                        st.rerun()

        # Process "thinking..." placeholder
        if (st.session_state.chat_history and 
            st.session_state.chat_history[-1]["answer"] == "thinking..."):
            
            question = st.session_state.chat_history[-1]["question"]
            
            with st.spinner("Searching document..."):
                results = search(
                    question,
                    st.session_state.index,
                    st.session_state.chunks
                )
                context_chunks = [r["chunk"] for r in results]
                answer = get_answer(question, context_chunks)
            
            st.session_state.chat_history[-1]["answer"] = answer
            st.rerun()

        # Chat input
        question = st.chat_input("Ask anything about your document...")

        if question:
            st.session_state.chat_history.append({
                "question": question,
                "answer": "thinking..."
            })
            st.rerun()