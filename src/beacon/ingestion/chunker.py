"""
Document Chunker — splits documents into retrieval-friendly chunks.

Uses token-aware splitting with configurable size and overlap.
Preserves source metadata so every chunk can be traced back to its document.

Chunking Strategy:
  1. Split on paragraph boundaries first (natural breakpoints)
  2. If a paragraph exceeds chunk_size, split on sentence boundaries
  3. Merge small paragraphs up to chunk_size
  4. Maintain overlap between chunks for context continuity

This is a deliberate design choice over naive character splitting.
Paragraph-first chunking preserves semantic coherence, which directly
impacts retrieval quality — a chunk that contains a complete thought
will match queries better than one that cuts mid-sentence.
"""

import re
from typing import List, Dict

import tiktoken

from beacon.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_document(document: Dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split a document into chunks with metadata.

    Args:
        document: Dict with 'content', 'source', and 'doc_type' keys
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of chunk dicts, each with 'text', 'source', 'chunk_index', and 'token_count'
    """
    content = document["content"]
    source = document["source"]

    # Use cl100k_base tokenizer (same family as Claude's tokenizer)
    enc = tiktoken.get_encoding("cl100k_base")

    # Step 1: Split into paragraphs
    paragraphs = _split_paragraphs(content)

    # Step 2: Build chunks by merging paragraphs up to chunk_size
    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))

        # If a single paragraph exceeds chunk_size, split it into sentences
        if para_tokens > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split the large paragraph into sentence-level chunks
            sentence_chunks = _split_long_paragraph(para, enc, chunk_size)
            chunks.extend(sentence_chunks)
            continue

        # If adding this paragraph would exceed chunk_size, flush
        if current_tokens + para_tokens > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))

            # Keep last paragraph for overlap if it fits
            if overlap > 0 and current_chunk:
                last = current_chunk[-1]
                last_tokens = len(enc.encode(last))
                if last_tokens <= overlap:
                    current_chunk = [last]
                    current_tokens = last_tokens
                else:
                    current_chunk = []
                    current_tokens = 0
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Step 3: Build chunk dicts with metadata
    chunk_dicts = []
    for i, text in enumerate(chunks):
        token_count = len(enc.encode(text))
        chunk_dicts.append({
            "text": text,
            "source": source,
            "chunk_index": i,
            "token_count": token_count,
        })

    return chunk_dicts


def chunk_documents(documents: List[Dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Chunk multiple documents and return a flat list of all chunks."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on double newlines, filtering empty ones."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_long_paragraph(text: str, enc, chunk_size: int) -> List[str]:
    """Split a long paragraph into sentence-level chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = len(enc.encode(sentence))

        if current_tokens + s_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += s_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks
