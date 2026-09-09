"""
FAISS vector store implementation behind the VectorStore interface.
Uses IndexFlatIP for exact nearest neighbor search on L2-normalized embeddings.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from hiver_agent.retrieval.interfaces import VectorStore


class FAISSVectorStore(VectorStore):
    """
    VectorStore implementation using FAISS IndexFlatIP.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Add document payloads and embedding vectors to the FAISS index.
        """
        if embeddings.shape[0] == 0:
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")

        # Ensure float32 contiguous array
        vecs = np.ascontiguousarray(embeddings.astype(np.float32))
        self.index.add(vecs)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top_k most similar document payloads.
        Returns document payload dicts enriched with '_score' field.
        """
        if self.count() == 0:
            return []

        q_vec = np.ascontiguousarray(query_embedding.astype(np.float32))
        if q_vec.ndim == 1:
            q_vec = np.expand_dims(q_vec, axis=0)

        actual_k = min(top_k, self.count())
        distances, indices = self.index.search(q_vec, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc_copy = dict(self.documents[idx])
            doc_copy["_score"] = float(dist)
            results.append(doc_copy)

        return results

    def count(self) -> int:
        return self.index.ntotal

    def save(self, index_dir: str) -> None:
        """
        Save FAISS binary index, document mapping JSON, and index metadata JSON.
        """
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        faiss_file = str(path / "faiss_index.bin")
        doc_map_file = str(path / "document_map.json")
        meta_file = str(path / "index_meta.json")

        faiss.write_index(self.index, faiss_file)

        with open(doc_map_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        meta = {
            "dimension": self.dimension,
            "total_documents": self.count(),
            "index_type": "IndexFlatIP"
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"FAISS index successfully saved to: {path}", flush=True)

    @classmethod
    def load(cls, index_dir: str) -> "FAISSVectorStore":
        """
        Load FAISS index and document mappings from directory.
        """
        path = Path(index_dir)
        faiss_file = str(path / "faiss_index.bin")
        doc_map_file = str(path / "document_map.json")
        meta_file = str(path / "index_meta.json")

        if not path.exists() or not os.path.exists(faiss_file):
            raise FileNotFoundError(f"FAISS index directory or binary file not found at {index_dir}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        store = cls(dimension=meta["dimension"])
        store.index = faiss.read_index(faiss_file)

        with open(doc_map_file, "r", encoding="utf-8") as f:
            store.documents = json.load(f)

        return store
