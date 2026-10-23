import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import CACHE_DIR


class VectorStore:
    """In-memory high performance vector store with metadata filtering and disk persistence."""

    def __init__(self, cache_file: str = "vector_index.npz", metadata_file: str = "vector_metadata.json"):
        self.cache_path = CACHE_DIR / cache_file
        self.meta_path = CACHE_DIR / metadata_file
        self.embeddings: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.metadata: List[Dict[str, Any]] = []
        self.load()

    def add_chunks(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add vectors and their associated chunk metadata to index."""
        if len(vectors) == 0:
            return

        if self.embeddings.shape[0] == 0:
            self.embeddings = vectors.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, vectors.astype(np.float32)])

        self.metadata.extend(metadata_list)
        self.save()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_year_min: Optional[int] = None,
        filter_year_max: Optional[int] = None,
        filter_sections: Optional[List[str]] = None,
        filter_paper_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Cosine similarity search with metadata filtering."""
        if self.embeddings.shape[0] == 0 or len(self.metadata) == 0:
            return []

        # Compute cosine similarities (assuming normalized vectors)
        scores = np.dot(self.embeddings, query_vector)

        # Apply filtering
        matching_indices = []
        filter_secs = [s.lower() for s in filter_sections] if filter_sections else None

        for idx, meta in enumerate(self.metadata):
            # Year filter
            year = meta.get("year")
            if filter_year_min is not None and year is not None and year < filter_year_min:
                continue
            if filter_year_max is not None and year is not None and year > filter_year_max:
                continue

            # Section filter
            if filter_secs is not None:
                sec = meta.get("section_name", "").lower()
                if sec not in filter_secs:
                    continue

            # Paper ID filter
            if filter_paper_ids is not None:
                pid = meta.get("paper_id")
                if pid not in filter_paper_ids:
                    continue

            matching_indices.append((idx, scores[idx]))

        if not matching_indices:
            return []

        # Sort by score descending
        matching_indices.sort(key=lambda x: x[1], reverse=True)
        top_matches = matching_indices[:top_k]

        results = []
        for idx, score in top_matches:
            item = dict(self.metadata[idx])
            item["similarity_score"] = float(score)
            results.append(item)

        return results

    def clear(self):
        """Reset the vector store."""
        self.embeddings = np.empty((0, 384), dtype=np.float32)
        self.metadata = []
        if self.cache_path.exists():
            self.cache_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def remove_by_paper_id(self, paper_id: str):
        """Remove all chunks associated with a specific paper_id."""
        if len(self.metadata) == 0:
            return
        keep_indices = [i for i, m in enumerate(self.metadata) if m.get("paper_id") != paper_id]
        if len(keep_indices) == len(self.metadata):
            return
        if len(keep_indices) == 0:
            self.clear()
        else:
            self.embeddings = self.embeddings[keep_indices]
            self.metadata = [self.metadata[i] for i in keep_indices]
            self.save()

    def save(self):
        """Persist embeddings matrix and metadata JSON to disk."""
        np.savez_compressed(str(self.cache_path), embeddings=self.embeddings)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

    def load(self):
        """Load from disk if files exist."""
        if self.cache_path.exists() and self.meta_path.exists():
            try:
                data = np.load(str(self.cache_path))
                self.embeddings = data["embeddings"]
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"[VectorStore] Loaded {len(self.metadata)} chunks from cache.")
            except Exception as e:
                print(f"[VectorStore] Failed to load index from cache: {e}")


# Global vector store singleton
vector_store = VectorStore()
