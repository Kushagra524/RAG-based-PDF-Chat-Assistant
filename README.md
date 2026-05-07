# 📄 RAG-based PDF Chat Assistant
 
<div align="center">

 
**Upload your PDF documents and have intelligent, context-aware conversations — powered by Retrieval-Augmented Generation.**
 
[🚀 Live Demo](https://pdf-chat-assistant-using-rag.streamlit.app) • [📂 GitHub](https://github.com/Kushagra524/RAG-based-PDF-Chat-Assistant)
 
</div>
---
 
## ✨ What is this?
 
A fully functional **Conversational PDF Chat Assistant** built from scratch using RAG (Retrieval-Augmented Generation). It doesn't just answer questions — it **remembers**, **retrieves**, and **reasons** across your uploaded PDF documents, even across multiple files simultaneously.
 
---
 
## 🧠 Features
 
| Feature | Description |
|---|---|
| 📚 **Multi-PDF Support** | Upload and query across multiple PDF files simultaneously |
| 🔍 **Smart RAG Retrieval** | MMR-based semantic search (k=8, fetch_k=20) for best context |
| 🕰️ **Chat History Awareness** | History-aware retriever reformulates questions using past context |
| 💬 **Session Memory** | Per-session conversation memory using `RunnableWithMessageHistory` |
| ⚡ **Multiple LLM Support** | Switch between Llama 3.1, Llama 3.3, Gemma2, GPT-OSS, Llama 4 Scout |
| 🧩 **Semantic Chunking** | `RecursiveCharacterTextSplitter` for intelligent document chunking |
| 🗂️ **Vector Store** | ChromaDB for fast and persistent vector storage |
| 🤗 **HuggingFace Embeddings** | `all-MiniLM-L6-v2` for high quality sentence embeddings |
 
---
 
## 🏗️ Architecture & RAG Pipeline
 
```
📄 PDF Upload
     │
     ▼
🔪 PyMuPDF Loader + RecursiveCharacterTextSplitter
     │  (chunk_size, chunk_overlap)
     ▼
🧮 HuggingFace Embeddings (all-MiniLM-L6-v2)
     │
     ▼
🗄️ ChromaDB Vector Store
     │
     ▼
🔍 MMR Retriever (k=8, fetch_k=20)
     │
     ▼
🕰️ History-Aware Retriever
     │  (reformulates query using chat history)
     ▼
🤖 Groq LLM (Llama / Gemma / GPT-OSS)
     │
     ▼
💬 Conversational RAG Chain
     │  (RunnableWithMessageHistory)
     ▼
✅ Context-aware Answer
```
 
---
 
## 🛠️ Tech Stack
 
- **Frontend** — [Streamlit](https://streamlit.io/)
- **LLM Provider** — [Groq](https://console.groq.com/) (ultra-fast inference)
- **Orchestration** — [LangChain](https://www.langchain.com/)
- **Embeddings** — [HuggingFace](https://huggingface.co/) (`all-MiniLM-L6-v2`)
- **Vector Store** — [ChromaDB](https://www.trychroma.com/)
- **PDF Loader** — `PyMuPDF` via LangChain community
- **Memory** — `ChatMessageHistory` + `RunnableWithMessageHistory`
---
 
## 🚀 Getting Started
 
### 1. Clone the repository
 
```bash
git clone https://github.com/Kushagra524/RAG-based-PDF-Chat-Assistant.git
cd RAG-based-PDF-Chat-Assistant
```
 
### 2. Create & activate virtual environment
 
```bash
python -m venv rag_doc_qna_env
# Windows
rag_doc_qna_env\Scripts\activate
# Mac/Linux
source rag_doc_qna_env/bin/activate
```
 
### 3. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Set up environment variables
 
Create a `.env` file in the root directory:
 
```env
HF_TOKEN=your_huggingface_token_here
```
 
### 5. Run the app
 
```bash
streamlit run convo.py
```
 
---
 
## 🔑 Configuration
 
| Setting | Description |
|---|---|
| **GROQ API Key** | Get your free key at [console.groq.com](https://console.groq.com) |
| **Model** | Choose from Llama 3.1 8B, Llama 3.3 70B, Gemma2 9B, GPT-OSS 120B, Llama 4 Scout |
| **Session ID** | Separate conversation memory per session |
 
---
 
## 📁 Project Structure
 
```
RAG-based-PDF-Chat-Assistant/
│
├── convo.py              # Main Streamlit app & RAG pipeline
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not committed)
├── .gitignore
└── README.md
```
 
---
 
## 💡 How It Works
 
1. **Upload** one or more PDF files via the sidebar
2. Documents are **loaded, chunked, and embedded** into ChromaDB
3. Enter your **GROQ API Key** and select a model
4. **Ask questions** — the history-aware retriever reformulates your query using past chat context
5. The RAG chain **retrieves relevant chunks** and generates a grounded, accurate answer
6. **Session memory** ensures follow-up questions work naturally
---
 
## 🤖 Supported Models
 
| Model | Provider | Best For |
|---|---|---|
| `llama-3.1-8b-instant` | Meta via Groq | Fast responses |
| `llama-3.3-70b-versatile` | Meta via Groq | Balanced quality |
| `gemma2-9b-it` | Google via Groq | Instruction following |
| `openai/gpt-oss-120b` | OpenAI via Groq | High quality answers |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Meta via Groq | Latest & most capable |
 
---
 
## 📸 Screenshots
 
> **Landing Page**
> Clean UI with drag-and-drop PDF upload and model selection
 
> **In Action**
> 227-page document processed into 525 chunks, answering complex questions about Deep Learning & Neural Networks with full context
 
---
 
## 🙏 Acknowledgements
 
> **Special Thanks to [Krish Naik](https://www.youtube.com/@krishnaik06)** 👑🔥
>
> The absolute GOAT of AI & ML education. This project exists because you made
> LangChain, RAG, and the hardest AI concepts feel simple and accessible.
> Legend in every sense of the word. 🙏
 
---
 
 
---
 
<div align="center">
Built with ❤️ by **Kushagra**
 
⭐ Star this repo if you found it helpful!
 
</div>
 
