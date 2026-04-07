"""
Vector Store — ChromaDB-backed storage for document chunks and embeddings.

ChromaDB runs entirely locally with persistent storage.
No cloud service, no API keys, no data leaving the machine.

Operations:
  - add_chunks: Ingest chunked documents with their embeddings
  - query: Semantic search for relevant chunks
  - clear: Reset the collection
  - stats: Get collection statistics
"""

from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

from beacon.config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K, SIMILARITY_THRESHOLD
from beacon.retrieval.embedder import Embedder


class VectorStore:
    """ChromaDB vector store for document chunks."""

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()

        # Initialise ChromaDB with persistent local storage
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add document chunks to the vector store.

        Args:
            chunks: List of dicts with 'text', 'source', 'chunk_index', 'token_count'

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Generate embeddings for all chunks
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Prepare data for ChromaDB
        ids = [f"{c['source']}::chunk-{c['chunk_index']}" for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
            }
            for c in chunks
        ]

        # Upsert (add or update)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a question.

        Args:
            question: The user's question
            top_k: Number of chunks to retrieve
            threshold: Minimum similarity score (0-1, higher = more similar)

        Returns:
            List of result dicts with 'text', 'source', 'score', 'chunk_index'
        """
        # Embed the question
        query_embedding = self.embedder.embed_text(question)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns cosine distance, convert to similarity
        # cosine_distance = 1 - cosine_similarity
        chunks = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # convert distance to similarity

            if similarity >= threshold:
                chunks.append({
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "chunk_index": results["metadatas"][0][i]["chunk_index"],
                    "score": round(similarity, 4),
                })

        return chunks

    def clear(self):
        """Delete all documents from the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count()

        # Get unique sources
        sources = set()
        if count > 0:
            all_metadata = self.collection.get(include=["metadatas"])
            for m in all_metadata["metadatas"]:
                sources.add(m["source"])

        return {
            "total_chunks": count,
            "documents": len(sources),
            "sources": sorted(sources),
        }
