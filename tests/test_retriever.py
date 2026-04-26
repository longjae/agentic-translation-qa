from pathlib import Path

import numpy as np

from app.rag.retriever import TermRetriever


class _FakeEmbedder:
    def encode(self, texts):
        base = np.zeros((len(texts), 4), dtype="float32")
        for i, text in enumerate(texts):
            base[i, i % 4] = float(len(text)) or 1.0
        return base


def test_retriever_returns_results(monkeypatch) -> None:
    monkeypatch.setattr("app.rag.retriever.get_embedder", lambda: _FakeEmbedder())
    retriever = TermRetriever(Path("data/glossaries"), Path("data/faiss_index"))
    results = retriever.retrieve(["혈압"], top_k=2)
    assert len(results) > 0
    assert "recommended_translation" in results[0]
