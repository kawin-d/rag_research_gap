import hashlib
import numpy as np
from typing import List
from app.config import settings

_model_instance = None


class EmbeddingService:
    """Computes dense vector representations for text chunks and queries using sentence-transformers."""

    @classmethod
    def get_model(cls):
        global _model_instance
        if _model_instance is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[EmbeddingService] Loading model: {settings.EMBEDDING_MODEL_NAME} on {settings.EMBEDDING_DEVICE}")
                _model_instance = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=settings.EMBEDDING_DEVICE)
            except Exception as e:
                print(f"[EmbeddingService] Failed to load SentenceTransformer ({e}). Using deterministic semantic fallback.")
                _model_instance = "fallback"
        return _model_instance

    @classmethod
    def embed_texts(cls, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings into a 2D numpy array of vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = cls.get_model()
        if model != "fallback":
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
            return embeddings.astype(np.float32)

        # Fallback: Deterministic TF-IDF / hashing vectorizer (384 dimensions)
        return cls._fallback_embed(texts)

    @classmethod
    def embed_query(cls, text: str) -> np.ndarray:
        """Embed a single query string into a 1D vector."""
        vectors = cls.embed_texts([text])
        return vectors[0]

    @classmethod
    def _fallback_embed(cls, texts: List[str]) -> np.ndarray:
        """Deterministic hash-based projection producing normalized 384-d vectors."""
        dim = 384
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            vec = np.zeros(dim, dtype=np.float32)
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if (h // dim) % 2 == 0 else -1.0
                vec[idx] += sign
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec /= norm
            out[i] = vec
        return out
