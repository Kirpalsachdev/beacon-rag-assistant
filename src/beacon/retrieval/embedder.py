"""
Embedder — generates vector embeddings using a local model.

Uses sentence-transformers to run embeddings entirely on-device.
No API calls, no cost, no data leaving the machine.

Default model: all-MiniLM-L6-v2 (384 dimensions)
  - Fast inference, good quality for semantic search
  - ~80MB model, downloads on first run

For higher quality (at the cost of speed):
  - Set EMBEDDING_MODEL=all-mpnet-base-v2 (768 dimensions)
"""

from typing import List

from sentence_transformers import SentenceTransformer

from beacon.config import EMBEDDING_MODEL


class Embedder:
    """Generates embeddings for text using a local model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model ready ({self.dimension} dimensions)")

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts efficiently."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
            batch_size=32,
        )
        return embeddings.tolist()
