from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)
