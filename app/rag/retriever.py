from pathlib import Path

import faiss
import numpy as np

from app.models.embedder import get_embedder
from app.rag.glossary_loader import load_glossary_files


class TermRetriever:
    def __init__(self, glossary_dir: Path, index_dir: Path) -> None:
        self.embedder = get_embedder()
        self.glossary = load_glossary_files(glossary_dir)
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "terms.index"
        self.index = self._load_or_build_index()

    def _load_or_build_index(self) -> faiss.IndexFlatL2:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        texts = [item["ko_term"] for item in self.glossary]
        embeddings = self.embedder.encode(texts).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        return index

    def retrieve(self, query_terms: list[str], top_k: int = 3) -> list[dict]:
        if not query_terms or not self.glossary:
            return []
        query_embeddings = self.embedder.encode(query_terms).astype("float32")
        distances, indices = self.index.search(query_embeddings, top_k)
        results: list[dict] = []
        for i, query in enumerate(query_terms):
            for j in range(top_k):
                idx = int(indices[i][j])
                if idx < 0 or idx >= len(self.glossary):
                    continue
                entry = self.glossary[idx]
                score = float(1 / (1 + distances[i][j]))
                results.append(
                    {
                        "query": query,
                        "term": entry["ko_term"],
                        "recommended_translation": entry["en_term"],
                        "score": score,
                        "source": entry["source"],
                    }
                )

        unique: dict[str, dict] = {}
        for row in results:
            term = row["term"]
            if term not in unique or row["score"] > unique[term]["score"]:
                unique[term] = row
        return sorted(unique.values(), key=lambda x: x["score"], reverse=True)[:top_k]
