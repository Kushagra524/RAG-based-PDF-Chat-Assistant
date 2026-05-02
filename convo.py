## RAG Q&A CONVERSATIONAL CHATBOT WITH PDF INCLUDING CHAT HISTORY

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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Look ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
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

    /* Sidebar styling */
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

    /* Chat message styling */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 0.75rem !important;
        border: 1px solid rgba(100, 116, 139, 0.12) !important;
    }

    /* Status cards */
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

    /* Metric pills */
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

    /* File chips */
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

    /* Divider */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
        margin: 1rem 0;
        border: none;
    }

    /* Typing animation */
    @keyframes pulse-dot {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    .typing-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #818cf8;
        margin: 0 2px;
    }
    .typing-dot:nth-child(1) { animation: pulse-dot 1.2s infinite 0s; }
    .typing-dot:nth-child(2) { animation: pulse-dot 1.2s infinite 0.2s; }
    .typing-dot:nth-child(3) { animation: pulse-dot 1.2s infinite 0.4s; }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #64748b;
    }
    .empty-state .icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .empty-state h3 {
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .empty-state p {
        color: #64748b;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ─────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = {}
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of {"role": ..., "content": ...}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ─── Cache the embedding model (expensive to load) ────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


embeddings = load_embeddings()


# ─── Sidebar ──────────────────────────────────────────────────────────
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

    # Process uploaded files
    if uploaded_files and api_key:
        current_names = sorted([f.name for f in uploaded_files])
        prev_names = sorted(st.session_state.pdf_names)

        if current_names != prev_names:
            with st.spinner("📑 Processing PDFs..."):
                documents = []
                for uploaded_file in uploaded_files:
                    # Write each file to a unique temp path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        documents.extend(docs)
                    finally:
                        os.unlink(tmp_path)  # clean up temp file

                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=150
                )
                splits = text_splitter.split_documents(documents)

                # Create vector store
                st.session_state.vector_store = Chroma.from_documents(
                    documents=splits, embedding=embeddings
                )
                st.session_state.pdf_names = current_names
                st.session_state.doc_count = len(documents)
                st.session_state.chunk_count = len(splits)

            st.success(f"✅ Processed {len(current_names)} PDF(s)")

    # Show uploaded file info
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

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_display = []
        if session_id in st.session_state.store:
            del st.session_state.store[session_id]
        st.rerun()


# ─── Main Header ──────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📄 PDF Chat Assistant</h1>
        <p>Upload PDFs and have intelligent conversations with your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Status Indicators ────────────────────────────────────────────────
if not api_key:
    st.markdown(
        """
        <div class="status-card warning">
            🔐 <strong>API key required</strong> — Enter your GROQ API key in the sidebar to get started.
            <br><small>Get a free key at <a href="https://console.groq.com" target="_blank">console.groq.com</a></small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if not st.session_state.vector_store:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📤</div>
            <h3>No documents loaded</h3>
            <p>Upload PDF files in the sidebar to begin chatting with your documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ─── Build the RAG Chain ──────────────────────────────────────────────
groq = ChatGroq(api_key=api_key, model=model_choice)
retriever = st.session_state.vector_store.as_retriever()

# History-aware retriever prompt
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

# QA prompt
system_prompt = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep the answer concise — three sentences maximum."
    "\n\n"
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


# ─── Render Chat History ──────────────────────────────────────────────
for msg in st.session_state.chat_display:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ─── Chat Input ───────────────────────────────────────────────────────
if user_input := st.chat_input("Ask something about your PDFs..."):
    # Show user message immediately
    st.session_state.chat_display.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Generate assistant response
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

