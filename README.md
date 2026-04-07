## Author

**Kirpal Sachdev**
[LinkedIn](https://linkedin.com/in/kirpalsachdev) · [GitHub](https://github.com/kirpalsachdev) · Singapore

# Beacon — RAG Knowledge Assistant

**Upload your documents. Ask questions. Get grounded, cited answers.**

Beacon is a Retrieval-Augmented Generation (RAG) system that turns any collection of documents into a searchable knowledge base with AI-powered answers. Every claim is grounded in the source material and cited — no hallucination, no guesswork.

---

## What It Does

Upload PDFs, Markdown, or text files, and Beacon:

1. **Chunks documents intelligently** — paragraph-first splitting preserves semantic coherence, with sentence-level fallback for long sections
2. **Embeds chunks locally** — vector embeddings run entirely on-device using sentence-transformers (no API calls, no data leaving your machine)
3. **Stores in ChromaDB** — persistent local vector store with cosine similarity search
4. **Retrieves relevant context** — finds the most semantically similar chunks for your question
5. **Generates cited answers** — Claude produces grounded responses with inline `[Source: filename]` citations

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  📄 Documents     →    📦 Chunker      →    🧮 Embedder         │
│  (PDF/MD/TXT)          (paragraph-first     (sentence-transformers│
│                         token-aware)         all-MiniLM-L6-v2)   │
│                                                    │              │
│                                                    ▼              │
│                                              ┌──────────┐        │
│                                              │ ChromaDB │        │
│                                              │ (local)  │        │
│                                              └────┬─────┘        │
│                                                   │              │
└───────────────────────────────────────────────────┼──────────────┘
                                                    │
┌───────────────────────────────────────────────────┼──────────────┐
│                        QUERY PIPELINE             │              │
│                                                   │              │
│  ❓ Question   →   🧮 Embed Query   →   🔍 Retrieve  ──────┘   │
│                                          (top-k cosine)          │
│                                              │                   │
│                                              ▼                   │
│                                     ┌─────────────────┐         │
│                                     │  Claude (Gen)   │         │
│                                     │  + Context      │         │
│                                     │  + System Prompt│         │
│                                     └────────┬────────┘         │
│                                              │                   │
│                                              ▼                   │
│                                     📝 Cited Answer              │
│                                     [Source: doc.pdf]            │
└──────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Local embeddings** | sentence-transformers runs entirely on-device. No API calls for embedding means no cost, no latency, and no data leaving the machine. This is the right trust architecture for enterprise documents. |
| **Paragraph-first chunking** | Chunks that contain complete thoughts match queries better than arbitrary character splits. The chunker respects natural document structure — paragraphs first, sentences as fallback. |
| **Token-aware splitting** | Using cl100k_base tokenizer ensures chunks align with how the LLM processes text, avoiding mid-token cuts and optimising context window usage. |
| **Inline citations** | Every claim in the answer includes a `[Source: filename]` tag. This makes grounding verifiable — the user can trace any statement back to its source document. |
| **Similarity threshold** | Low-relevance chunks are filtered out before generation. This prevents Claude from being distracted by marginally related content, improving answer precision. |
| **Honest uncertainty** | The system prompt explicitly instructs Claude to say "I don't have enough information" rather than guess. This is a guardrail against the most common RAG failure mode — hallucination when context is insufficient. |

## The RAG Stack

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Loading** | pypdf + text reader | Extracts text from PDF, Markdown, and plain text files |
| **Chunking** | Custom token-aware chunker | Splits documents preserving semantic boundaries |
| **Embedding** | sentence-transformers (MiniLM) | Generates 384-dimensional vectors locally |
| **Storage** | ChromaDB | Persistent local vector store with cosine similarity |
| **Retrieval** | Cosine similarity + threshold | Finds top-k relevant chunks above quality threshold |
| **Generation** | Anthropic Claude | Produces grounded, cited answers from context |
| **Interface** | Streamlit | Upload docs, ask questions, view cited answers |

## How to Run

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) for Claude

### Setup

```bash
# Clone the repo
git clone https://github.com/kirpalsachdev/beacon-rag-assistant.git
cd beacon-rag-assistant

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the app
streamlit run src/beacon/app.py
```

The first run will download the embedding model (~80MB). After that, startup is instant.

### Quick Demo

Sample documents are included in `docs/sample/` — product FAQ and return policy for a fictional outdoor gear brand. Upload them through the Streamlit interface and try:

```
"What is the return window for defective items?"
```
→ Answer with citations from the return policy

```
"How should I care for my Ridgeline jacket?"
```
→ Care instructions cited from the product FAQ

```
"Can I return a gift without the original receipt?"
```
→ Gift return policy with store credit details, cited from source

```
"What's the best restaurant in Singapore?"
```
→ "I don't have enough information in the loaded documents to answer this question."

## What This Demonstrates

RAG is the knowledge layer that powers every enterprise AI agent. This prototype covers the full pipeline:

| RAG Concept | Beacon Implementation | Enterprise Equivalent |
|-------------|----------------------|----------------------|
| Document ingestion | Multi-format loader + chunker | Knowledge base connector |
| Embedding | Local sentence-transformers | Embedding service (OpenAI, Cohere, Voyage) |
| Vector storage | ChromaDB (local, persistent) | Pinecone, Weaviate, Qdrant |
| Semantic search | Cosine similarity + threshold | Retrieval pipeline |
| Grounded generation | Claude with citation prompt | RAG-augmented agent |
| Source attribution | Inline [Source: file] tags | Citation and audit trail |
| Hallucination prevention | "I don't know" guardrail | Confidence thresholds |

## Enterprise Considerations

This prototype is designed for local, single-user use. In production, you'd add:

- **Authentication and access control** — who can query which documents
- **Chunk metadata enrichment** — page numbers, section headers, document version
- **Hybrid search** — combine semantic search with keyword (BM25) for better recall
- **Re-ranking** — use a cross-encoder to re-score retrieved chunks before generation
- **Evaluation framework** — measure retrieval quality (precision@k, recall) and answer accuracy
- **Streaming responses** — progressive answer display for better UX
- **Document versioning** — track which version of a document was used for each answer

## Stack

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2, local)
- **Vector Store:** ChromaDB (local, persistent)
- **LLM:** Anthropic Claude Sonnet
- **Interface:** Streamlit
- **Language:** Python 3.12



---

*Built as a portfolio prototype demonstrating RAG architecture for enterprise knowledge management. Sample documents use a fictional brand — no real company data is included.*
