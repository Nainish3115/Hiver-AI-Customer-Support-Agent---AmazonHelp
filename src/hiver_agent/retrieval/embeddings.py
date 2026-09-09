"""
SentenceTransformer embedding provider implementation.
Supports configurable model selection, batch encoding, L2 normalization, and caching.
"""
import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from hiver_agent.retrieval.interfaces import EmbeddingProvider

# Set 4 threads to avoid CPU core thread contention / lock overhead on Windows
torch.set_num_threads(4)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Concrete EmbeddingProvider backed by SentenceTransformers.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 128,
        normalize: bool = True,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading embedding model: {model_name} on device: {self.device} (CPU threads: {torch.get_num_threads()})...", flush=True)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = 128
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into a 2D numpy array of shape (N, dimension).
        Uses caching and deduplication to avoid re-encoding identical strings.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Identify unique texts not in cache
        to_encode_set = set()
        for txt in texts:
            if txt not in self._cache:
                to_encode_set.add(txt)

        if to_encode_set:
            unique_list = list(to_encode_set)
            print(f"Encoding {len(unique_list):,} unique texts (batch size: {self.batch_size})...", flush=True)
            encoded_vecs = self.model.encode(
                unique_list,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            for txt, vec in zip(unique_list, encoded_vecs):
                self._cache[txt] = vec.astype(np.float32)

        # Assemble result array
        res = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for idx, txt in enumerate(texts):
            res[idx] = self._cache[txt]

        return res

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string into a 2D numpy array of shape (1, dimension).
        """
        vec = self.embed_texts([query])
        return vec
