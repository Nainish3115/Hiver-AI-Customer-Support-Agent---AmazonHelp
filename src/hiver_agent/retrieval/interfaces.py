"""
Modular abstract interfaces for RAG vector retrieval.
Provides decoupled abstractions for EmbeddingProvider, VectorStore, and Retriever.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class EmbeddingProvider(ABC):
    """
    Abstract interface for embedding text into vector representations.
    """
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings into a 2D numpy array of shape (N, dimension).
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query string into a 1D or 2D numpy array of shape (1, dimension).
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimension of the output vector space.
        """
        pass


class VectorStore(ABC):
    """
    Abstract interface for storing and searching vector representations.
    """
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Add document payloads and their corresponding embedding vectors to the store.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for top_k most similar document payloads given a query vector embedding.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return total number of indexed documents in the vector store.
        """
        pass


class Retriever(ABC):
    """
    Abstract interface for high-level historical case retrieval.
    """
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k historical cases relevant to the customer query.
        """
        pass
