import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c4b5fd !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #e2e8f0 !important;
    }

    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 0.75rem !important;
        border: 1px solid rgba(100, 116, 139, 0.12) !important;
    }

    .status-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .status-card.success {
        background: linear-gradient(135deg, rgba(34,197,94,0.08) 0%, rgba(16,185,129,0.08) 100%);
        border-color: rgba(34,197,94,0.25);
    }
    .status-card.warning {
        background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(234,88,12,0.08) 100%);
        border-color: rgba(245,158,11,0.25);
    }

    .metric-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin: 0.75rem 0;
    }
    .metric-pill {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 0.35rem 0.85rem;
        font-size: 0.82rem;
        color: #818cf8;
        font-weight: 500;
    }

    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        margin: 0.2rem 0.25rem;
        font-size: 0.82rem;
        color: #a5b4fc;
    }

    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
        margin: 1rem 0;
        border: none;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ─── Landing Page Hero ─── */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-18px); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse-ring {
        0%   { transform: scale(0.9); opacity: 0.7; }
        50%  { transform: scale(1.05); opacity: 0.3; }
        100% { transform: scale(0.9); opacity: 0.7; }
    }

    .hero-section {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 20%, rgba(99,102,241,0.06) 0%, transparent 50%),
                    radial-gradient(ellipse at 70% 80%, rgba(168,85,247,0.05) 0%, transparent 50%);
        pointer-events: none;
    }

    .hero-icon {
        font-size: 4.5rem;
        animation: float 4s ease-in-out infinite;
        margin-bottom: 1rem;
        display: inline-block;
        filter: drop-shadow(0 8px 24px rgba(99,102,241,0.25));
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 5s ease infinite;
        margin-bottom: 0.5rem;
        line-height: 1.15;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 300;
        max-width: 550px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 999px;
        padding: 0.4rem 1rem;
        font-size: 0.8rem;
        color: #a5b4fc;
        font-weight: 500;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.6s ease-out;
    }

    /* ─── Feature Cards ─── */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.25rem;
        max-width: 900px;
        margin: 0 auto 2.5rem auto;
        padding: 0 1rem;
    }
    @media (max-width: 768px) {
        .features-grid { grid-template-columns: 1fr; }
    }

    .feature-card {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 16px;
        padding: 1.75rem 1.5rem;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out backwards;
        position: relative;
        overflow: hidden;
    }
    .feature-card:nth-child(1) { animation-delay: 0.1s; }
    .feature-card:nth-child(2) { animation-delay: 0.2s; }
    .feature-card:nth-child(3) { animation-delay: 0.3s; }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        opacity: 0;
        transition: opacity 0.35s ease;
    }
    .feature-card:hover {
        transform: translateY(-6px);
        border-color: rgba(99,102,241,0.3);
        box-shadow: 0 12px 40px rgba(99,102,241,0.12);
    }
    .feature-card:hover::before { opacity: 1; }

    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.75rem;
        display: inline-block;
    }
    .feature-title {
        color: #e2e8f0;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .feature-desc {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.55;
        font-weight: 300;
    }

    /* ─── How It Works ─── */
    .steps-section {
        max-width: 750px;
        margin: 0 auto 2.5rem auto;
        padding: 0 1rem;
    }
    .steps-title {
        text-align: center;
        color: #c4b5fd;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .steps-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        position: relative;
    }
    @media (max-width: 768px) {
        .steps-row { flex-direction: column; align-items: center; }
    }
    .steps-row::before {
        content: '';
        position: absolute;
        top: 28px;
        left: 15%;
        right: 15%;
        height: 2px;
        background: linear-gradient(90deg, rgba(99,102,241,0.15), rgba(168,85,247,0.25), rgba(99,102,241,0.15));
    }
    @media (max-width: 768px) {
        .steps-row::before { display: none; }
    }

    .step-item {
        text-align: center;
        flex: 1;
        position: relative;
        z-index: 1;
    }
    .step-number {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3);
        position: relative;
    }
    .step-number::after {
        content: '';
        position: absolute;
        inset: -5px;
        border-radius: 50%;
        border: 2px solid rgba(99,102,241,0.2);
        animation: pulse-ring 3s ease-in-out infinite;
    }
    .step-label {
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .step-desc {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 300;
    }

    /* ─── Tech Stack Bar ─── */
    .tech-bar {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.6rem;
        max-width: 650px;
        margin: 0 auto 2rem auto;
        padding: 0 1rem;
    }
    .tech-badge {
        background: rgba(30, 27, 75, 0.6);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 8px;
        padding: 0.35rem 0.85rem;
        font-size: 0.75rem;
        color: #818cf8;
        font-weight: 500;
        transition: all 0.25s ease;
    }
    .tech-badge:hover {
        border-color: rgba(99,102,241,0.4);
        background: rgba(99,102,241,0.1);
        transform: translateY(-2px);
    }

    /* ─── CTA ─── */
    .cta-section {
        text-align: center;
        padding: 1rem 0 2.5rem 0;
    }
    .cta-arrow {
        display: inline-block;
        color: #818cf8;
        font-size: 1.5rem;
        animation: float 2.5s ease-in-out infinite;
    }
    .cta-text {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

if "store" not in st.session_state:
    st.session_state.store = {}
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    api_key = st.text_input(
        "🔑 GROQ API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at https://console.groq.com",
    )

    model_choice = st.selectbox(
        "🤖 Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ],
        index=0,
        help="Select the LLM to power your assistant",
    )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📂 Documents")

    session_id = st.text_input(
        "💬 Session ID",
        value="default_session",
        help="Use different session IDs to maintain separate chat histories",
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with",
    )

    if uploaded_files and api_key:
        current_names = sorted([f.name for f in uploaded_files])
        prev_names = sorted(st.session_state.pdf_names)

        if current_names != prev_names:
            with st.spinner("📑 Processing PDFs..."):
                def load_single_pdf(uploaded_file):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    try:
                        return PyMuPDFLoader(tmp_path).load()
                    finally:
                        os.unlink(tmp_path)

                with ThreadPoolExecutor(max_workers=min(4, len(uploaded_files))) as pool:
                    results = pool.map(load_single_pdf, uploaded_files)

                documents = [doc for docs in results for doc in docs]

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                splits = text_splitter.split_documents(documents)

                st.session_state.vector_store = Chroma.from_documents(
                    documents=splits, embedding=embeddings
                )
                st.session_state.pdf_names = current_names
                st.session_state.doc_count = len(documents)
                st.session_state.chunk_count = len(splits)

            st.success(f"✅ Processed {len(current_names)} PDF(s)")

    if st.session_state.pdf_names:
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Knowledge Base")
        for name in st.session_state.pdf_names:
            st.markdown(f'<div class="file-chip">📄 {name}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="metric-row">
                <span class="metric-pill">📃 {st.session_state.doc_count} pages</span>
                <span class="metric-pill">🧩 {st.session_state.chunk_count} chunks</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_display = []
        if session_id in st.session_state.store:
            del st.session_state.store[session_id]
        st.rerun()

if not api_key or not st.session_state.vector_store:
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-badge">✨ Powered by RAG &amp; LangChain</div>
            <br>
            <span class="hero-icon">📄</span>
            <div class="hero-title">PDF Chat Assistant</div>
            <p class="hero-subtitle">
                Upload your PDF documents and have intelligent, context-aware
                conversations — powered by retrieval-augmented generation.
            </p>
        </div>

        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Smart RAG</div>
                <div class="feature-desc">Uses vector embeddings and semantic search to find the most relevant context from your documents.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">Chat History</div>
                <div class="feature-desc">Maintains full conversation context so follow-up questions work naturally and intelligently.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📚</div>
                <div class="feature-title">Multi-PDF</div>
                <div class="feature-desc">Upload and query across multiple PDF files simultaneously for comprehensive answers.</div>
            </div>
        </div>

        <div class="steps-section">
            <div class="steps-title">How It Works</div>
            <div class="steps-row">
                <div class="step-item">
                    <div class="step-number">1</div>
                    <div class="step-label">Add API Key</div>
                    <div class="step-desc">Enter your free GROQ key</div>
                </div>
                <div class="step-item">
                    <div class="step-number">2</div>
                    <div class="step-label">Upload PDFs</div>
                    <div class="step-desc">Drop one or more files</div>
                </div>
                <div class="step-item">
                    <div class="step-number">3</div>
                    <div class="step-label">Start Chatting</div>
                    <div class="step-desc">Ask anything about them</div>
                </div>
            </div>
        </div>

        <div class="tech-bar">
            <span class="tech-badge">🦜 LangChain</span>
            <span class="tech-badge">🚀 GROQ</span>
            <span class="tech-badge">🤗 HuggingFace</span>
            <span class="tech-badge">🔮 ChromaDB</span>
            <span class="tech-badge">🎈 Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_key:
        st.markdown(
            """
            <div class="cta-section">
                <span class="cta-arrow">👈</span>
                <p class="cta-text">Enter your <strong>GROQ API Key</strong> in the sidebar to get started</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="cta-section">
                <span class="cta-arrow">👈</span>
                <p class="cta-text">Upload your <strong>PDF files</strong> in the sidebar to begin</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

st.markdown(
    """
    <div class="main-header">
        <h1>📄 PDF Chat Assistant</h1>
        <p>Upload PDFs and have intelligent conversations with your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

groq = ChatGroq(api_key=api_key, model=model_choice)
retriever = st.session_state.vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20},
)

contextualize_system_prompt = (
    "Given the chat history and the latest user question, "
    "which might reference context in the chat history, "
    "formulate a standalone question that can be understood "
    "without the chat history. Do NOT answer the question — "
    "just reformulate it if needed, otherwise return it as is."
)

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    groq, retriever, contextualize_prompt
)

system_prompt = (
    "You are a knowledgeable assistant that answers questions based on the uploaded PDF documents. "
    "Use the following retrieved context thoroughly to provide accurate, detailed answers. "
    "Always base your answer on the context provided. "
    "If the context contains relevant information, use it to give a comprehensive answer. "
    "Only say you don't know if the context truly contains no relevant information at all."
    "\n\n"
    "Context from the PDF:\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qna_chain = create_stuff_documents_chain(groq, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)

def get_session_history(sid: str) -> BaseChatMessageHistory:
    if sid not in st.session_state.store:
        st.session_state.store[sid] = ChatMessageHistory()
    return st.session_state.store[sid]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

for msg in st.session_state.chat_display:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask something about your PDFs..."):
    st.session_state.chat_display.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                answer = response["answer"]
            except Exception as e:
                answer = f"⚠️ An error occurred: {str(e)}"

        st.markdown(answer)

    st.session_state.chat_display.append({"role": "assistant", "content": answer})
    st.rerun()
