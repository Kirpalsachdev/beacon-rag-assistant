"""
Document Loader — reads PDF, Markdown, and plain text files.

Each document is returned as a dict with:
  - content: the raw text
  - source: the filename
  - doc_type: pdf | markdown | text
"""

import os
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def load_document(file_path: str) -> Dict:
    """Load a single document and return its content with metadata."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".md", ".markdown"):
        return _load_text(path, doc_type="markdown")
    elif suffix in (".txt", ".text"):
        return _load_text(path, doc_type="text")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .md, or .txt")


def load_directory(dir_path: str) -> List[Dict]:
    """Load all supported documents from a directory."""
    supported = {".pdf", ".md", ".markdown", ".txt", ".text"}
    documents = []

    for root, _, files in os.walk(dir_path):
        for file in sorted(files):
            if Path(file).suffix.lower() in supported:
                try:
                    doc = load_document(os.path.join(root, file))
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")

    return documents


def _load_pdf(path: Path) -> Dict:
    """Extract text from a PDF file."""
    reader = PdfReader(str(path))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())

    return {
        "content": "\n\n".join(pages),
        "source": path.name,
        "doc_type": "pdf",
        "page_count": len(reader.pages),
    }


def _load_text(path: Path, doc_type: str) -> Dict:
    """Read a plain text or markdown file."""
    content = path.read_text(encoding="utf-8")
    return {
        "content": content,
        "source": path.name,
        "doc_type": doc_type,
    }
