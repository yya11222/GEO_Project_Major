"""
GEO System - Vector Store Module (Sprint 3 extension)
--------------------------------------------------------
Persists SBERT embeddings for analyzed pages to disk using FAISS,
so embeddings don't need to be recomputed every run, and so we can
do similarity search across previously analyzed pages -- this is
also the foundation for the "Competitor Comparison" feature.

Storage cost reference (asked about in viva):
    Each embedding is 384-dim, stored as float32 -> 384 x 4 bytes
    = 1536 bytes (~1.5 KB) per page. 1,000 pages ~= 1.5 MB.

Usage:
    python vector_store.py   # runs a small self-test
"""

import os
import json
import numpy as np
import faiss

INDEX_PATH = "geo_vector_index.faiss"
METADATA_PATH = "geo_vector_metadata.json"
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2 output size


class VectorStore:
    """
    Wraps a FAISS index + a parallel metadata store (FAISS itself only
    stores vectors and integer IDs, not arbitrary metadata like URLs).
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        # IndexFlatIP = exact inner-product search. Since SBERT embeddings
        # are normalized, inner product is equivalent to cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []  # list of dicts, position i matches vector i in the index

    def add(self, embedding: np.ndarray, metadata: dict):
        """Add one embedding + its metadata (e.g. url, title, target_query)."""
        vec = self._normalize(embedding).reshape(1, -1).astype("float32")
        self.index.add(vec)
        self.metadata.append(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Return the top-k most similar stored pages to the query embedding."""
        if self.index.ntotal == 0:
            return []
        vec = self._normalize(query_embedding).reshape(1, -1).astype("float32")
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"similarity": float(score), **self.metadata[idx]})
        return results

    def save(self, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load(self, index_path: str = INDEX_PATH, metadata_path: str = METADATA_PATH):
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            return True
        return False

    def size_estimate_bytes(self) -> int:
        """Rough storage size: n_vectors x dim x 4 bytes (float32)."""
        return self.index.ntotal * self.dim * 4

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype="float32")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


if __name__ == "__main__":
    # Small self-test with fake random embeddings (in real use, these
    # come from nlp_features.get_sbert_model().encode(...))
    store = VectorStore(dim=EMBEDDING_DIM)

    np.random.seed(42)
    pages = [
        {"url": "https://example.com/geo-article", "title": "About GEO"},
        {"url": "https://example.com/cooking-blog", "title": "Pasta Recipes"},
        {"url": "https://example.com/seo-guide", "title": "SEO Basics"},
    ]

    for page in pages:
        fake_embedding = np.random.rand(EMBEDDING_DIM)
        store.add(fake_embedding, page)

    store.save()
    print(f"Stored {store.index.ntotal} vectors")
    print(f"Estimated storage size: {store.size_estimate_bytes()} bytes")

    # reload from disk to confirm persistence works
    reloaded = VectorStore(dim=EMBEDDING_DIM)
    reloaded.load()
    print(f"Reloaded {reloaded.index.ntotal} vectors from disk")

    query = np.random.rand(EMBEDDING_DIM)
    results = reloaded.search(query, k=2)
    print("Search results:")
    for r in results:
        print(f"  {r['title']} (similarity: {r['similarity']:.4f})")
