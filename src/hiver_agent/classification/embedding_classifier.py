"""
Embedding nearest-centroid prototype classifier for AmazonHelp intents.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from hiver_agent.classification.interface import IntentClassifier, IntentPrediction
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.taxonomy import AMAZON_INTENTS


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class EmbeddingPrototypeClassifier(IntentClassifier):
    """
    Nearest-centroid prototype classifier based on sentence embeddings.
    """
    def __init__(self, embedding_provider: SentenceTransformerEmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.centroids: Dict[str, np.ndarray] = {}
        self.classes: List[str] = AMAZON_INTENTS

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Compute normalized mean vector centroids for each intent category.
        """
        if not texts or not labels:
            return

        embeddings = self.embedding_provider.embed_texts(texts)
        class_embeddings = {c: [] for c in set(labels)}

        for emb, lbl in zip(embeddings, labels):
            class_embeddings[lbl].append(emb)

        for c, embs in class_embeddings.items():
            mean_vec = np.mean(embs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            self.centroids[c] = mean_vec

        self.classes = sorted(list(self.centroids.keys()))

    def predict(self, text: str) -> IntentPrediction:
        if not self.centroids:
            # Fallback if un-fitted
            probs = {c: 1.0 / len(self.classes) for c in self.classes}
            return IntentPrediction(
                text=text,
                intent=self.classes[0],
                confidence=float(1.0 / len(self.classes)),
                probabilities=probs
            )

        q_vec = self.embedding_provider.embed_query(text).flatten()

        sims = {}
        for c in self.classes:
            centroid = self.centroids[c]
            sim = float(np.dot(q_vec, centroid))
            sims[c] = sim

        sim_values = np.array([sims[c] for c in self.classes])
        # Temperature scaling for probability calibration
        prob_values = softmax(sim_values * 10.0)

        probs = {c: float(p) for c, p in zip(self.classes, prob_values)}
        best_intent = max(sims.items(), key=lambda x: x[1])[0]
        confidence = float(probs[best_intent])

        return IntentPrediction(
            text=text,
            intent=best_intent,
            confidence=confidence,
            probabilities=probs
        )

    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        """
        Batch-predict intents for a list of texts using efficient matrix multiplication.
        """
        if not texts:
            return []

        if not self.centroids:
            probs = {c: 1.0 / len(self.classes) for c in self.classes}
            return [
                IntentPrediction(text=t, intent=self.classes[0], confidence=probs[self.classes[0]], probabilities=probs)
                for t in texts
            ]

        # Embed all texts in a single batch
        q_vecs = self.embedding_provider.embed_texts(texts) # Shape: (N, dim)
        centroid_matrix = np.array([self.centroids[c] for c in self.classes]) # Shape: (C, dim)

        # Vectorized cosine similarity: (N, dim) @ (dim, C) -> (N, C)
        sim_matrix = np.dot(q_vecs, centroid_matrix.T)

        results = []
        for idx, text in enumerate(texts):
            sims = {c: float(sim_matrix[idx, i]) for i, c in enumerate(self.classes)}
            sim_values = sim_matrix[idx]
            prob_values = softmax(sim_values * 10.0)
            probs = {c: float(p) for c, p in zip(self.classes, prob_values)}
            best_intent = self.classes[int(np.argmax(sim_values))]
            confidence = float(probs[best_intent])

            results.append(
                IntentPrediction(
                    text=text,
                    intent=best_intent,
                    confidence=confidence,
                    probabilities=probs
                )
            )

        return results
