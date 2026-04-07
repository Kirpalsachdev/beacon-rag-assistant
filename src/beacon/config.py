"""
Configuration for Beacon RAG Assistant.

All settings are loaded from environment variables with sensible defaults.
Copy .env.example to .env and add your Anthropic API key.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
# Using a local embedding model — no API calls, no cost, runs on-device.
# all-MiniLM-L6-v2 is a good balance of speed and quality (384 dimensions).
# For higher quality, switch to all-mpnet-base-v2 (768 dimensions).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Chunking Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))       # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))   # overlap between chunks

# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))                    # chunks to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "beacon_knowledge")
