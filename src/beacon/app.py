"""
Beacon — RAG Knowledge Assistant
Streamlit UI for uploading documents, asking questions, and getting cited answers.

Run with: streamlit run src/beacon/app.py
"""

import os
import sys
import tempfile
import streamlit as st

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beacon.config import ANTHROPIC_API_KEY, TOP_K
from beacon.ingestion.loader import load_document
from beacon.ingestion.chunker import chunk_document
from beacon.retrieval.embedder import Embedder
from beacon.retrieval.store import VectorStore
from beacon.generation.answerer import generate_answer


# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Beacon — RAG Knowledge Assistant",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .source-tag {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 2px;
    }
    .chunk-preview {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .stats-card {
        background: #f1f5f9;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
@st.cache_resource
def init_store():
    """Initialise the embedding model and vector store (cached across reruns)."""
    embedder = Embedder()
    store = VectorStore(embedder=embedder)
    return store

# Check for API key
if not ANTHROPIC_API_KEY:
    st.error("⚠️ ANTHROPIC_API_KEY not found. Add it to your .env file.")
    st.stop()

store = init_store()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar — Document Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, Markdown, Plain Text",
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                total_chunks = 0
                for uploaded_file in uploaded_files:
                    # Save to temp file for loading
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(uploaded_file.name)[1],
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        # Load and chunk
                        doc = load_document(tmp_path)
                        doc["source"] = uploaded_file.name  # use original filename
                        chunks = chunk_document(doc)

                        # Add to vector store
                        added = store.add_chunks(chunks)
                        total_chunks += added
                        st.success(f"✅ {uploaded_file.name}: {added} chunks")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_path)

                st.success(f"**Ingested {total_chunks} chunks total**")

    st.divider()

    # Collection stats
    stats = store.stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", stats["documents"])
    with col2:
        st.metric("Chunks", stats["total_chunks"])

    if stats["sources"]:
        st.markdown("**Loaded sources:**")
        for source in stats["sources"]:
            st.markdown(f"- `{source}`")

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 1, 10, TOP_K)
    show_sources = st.toggle("Show retrieved chunks", value=True)

    st.divider()

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        store.clear()
        st.session_state.messages = []
        st.rerun()

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main — Chat Interface
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">🔦 Beacon</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Ask questions about your documents. Every answer is grounded and cited.</p>',
    unsafe_allow_html=True,
)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and "sources" in msg and show_sources:
            with st.expander("📎 Retrieved context"):
                for chunk in msg["sources"]:
                    st.markdown(
                        f'<div class="chunk-preview">'
                        f'<strong>{chunk["source"]}</strong> '
                        f'(score: {chunk["score"]})<br/>'
                        f'{chunk["text"][:300]}...'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Check if documents are loaded
    if store.stats()["total_chunks"] == 0:
        st.warning("Upload and ingest some documents first using the sidebar.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve and generate
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            # Retrieve relevant chunks
            chunks = store.query(question, top_k=top_k)

            # Build conversation history for multi-turn context
            history = []
            for msg in st.session_state.messages[:-1]:  # exclude current question
                if msg["role"] in ("user", "assistant"):
                    history.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

            # Generate answer
            result = generate_answer(
                question=question,
                chunks=chunks,
                conversation_history=history[-6:],  # last 3 turns for context
            )

        # Display answer
        st.markdown(result["answer"])

        # Show token usage
        tokens = result["tokens_used"]
        st.caption(
            f"Model: {result['model']} · "
            f"Chunks retrieved: {result['chunks_provided']} · "
            f"Sources cited: {len(result['sources_cited'])} · "
            f"Tokens: {tokens['input']}→{tokens['output']}"
        )

        # Show retrieved chunks
        if show_sources and chunks:
            with st.expander("📎 Retrieved context"):
                for chunk in chunks:
                    st.markdown(
                        f'<div class="chunk-preview">'
                        f'<strong>{chunk["source"]}</strong> '
                        f'(score: {chunk["score"]})<br/>'
                        f'{chunk["text"][:300]}...'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": chunks,
        })


def main():
    """Entry point for the beacon CLI command."""
    import subprocess
    subprocess.run(["streamlit", "run", __file__])
