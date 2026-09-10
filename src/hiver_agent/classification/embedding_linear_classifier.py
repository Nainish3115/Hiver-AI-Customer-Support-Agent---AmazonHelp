"""
SentenceTransformer Embedding + Linear Classifier (Logistic Regression) for AmazonHelp intent classification.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from hiver_agent.classification.interface import IntentClassifier, IntentPrediction
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.taxonomy import AMAZON_INTENTS


class EmbeddingLinearClassifier(IntentClassifier):
    """
    Supervised classifier that uses pre-trained SentenceTransformer embeddings
    as features for a multi-class Logistic Regression model.
    """
    def __init__(
        self,
        embedding_provider: SentenceTransformerEmbeddingProvider,
        c_param: float = 1.0
    ):
        self.embedding_provider = embedding_provider
        self.c_param = c_param
        self.model: Optional[LogisticRegression] = None
        self.classes: List[str] = AMAZON_INTENTS

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Extract embeddings and fit Logistic Regression model strictly on training data.
        """
        if not texts or not labels:
            return

        embeddings = self.embedding_provider.embed_texts(texts)

        self.model = LogisticRegression(
            C=self.c_param,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        self.model.fit(embeddings, labels)
        self.classes = [str(c) for c in self.model.classes_]

    def predict(self, text: str) -> IntentPrediction:
        if self.model is None:
            probs = {c: 1.0 / len(self.classes) for c in self.classes}
            return IntentPrediction(
                text=text,
                intent=self.classes[0],
                confidence=float(1.0 / len(self.classes)),
                probabilities=probs
            )

        q_vec = self.embedding_provider.embed_query(text).reshape(1, -1)
        prob_arr = self.model.predict_proba(q_vec)[0]

        probs = {c: float(p) for c, p in zip(self.classes, prob_arr)}
        best_intent = self.model.predict(q_vec)[0]
        confidence = float(probs[best_intent])

        return IntentPrediction(
            text=text,
            intent=best_intent,
            confidence=confidence,
            probabilities=probs
        )

    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        if not texts:
            return []

        if self.model is None:
            probs = {c: 1.0 / len(self.classes) for c in self.classes}
            return [
                IntentPrediction(text=t, intent=self.classes[0], confidence=probs[self.classes[0]], probabilities=probs)
                for t in texts
            ]

        embeddings = self.embedding_provider.embed_texts(texts)
        prob_matrix = self.model.predict_proba(embeddings)
        pred_labels = self.model.predict(embeddings)

        results = []
        for idx, text in enumerate(texts):
            prob_arr = prob_matrix[idx]
            probs = {c: float(p) for c, p in zip(self.classes, prob_arr)}
            best_intent = str(pred_labels[idx])
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
