"""
TF-IDF + Linear Classifier baseline for AmazonHelp intent classification.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from hiver_agent.classification.interface import IntentClassifier, IntentPrediction
from hiver_agent.classification.taxonomy import AMAZON_INTENTS


class TfidfIntentClassifier(IntentClassifier):
    """
    Supervised classical NLP classifier combining Word + Char n-gram TF-IDF
    features with a Logistic Regression linear classifier.
    """
    def __init__(
        self,
        c_param: float = 1.0,
        max_features: int = 5000,
        solver: str = "lbfgs"
    ):
        self.c_param = c_param
        self.max_features = max_features
        self.solver = solver
        self.vectorizer: Optional[FeatureUnion] = None
        self.model: Optional[LogisticRegression] = None
        self.classes: List[str] = AMAZON_INTENTS

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Fit TF-IDF vocabulary and Logistic Regression classifier strictly on training data.
        """
        if not texts or not labels:
            return

        word_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            sublinear_tf=True
        )
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=self.max_features,
            sublinear_tf=True
        )

        self.vectorizer = FeatureUnion([
            ("word", word_vec),
            ("char", char_vec)
        ])

        X_train = self.vectorizer.fit_transform(texts)

        self.model = LogisticRegression(
            C=self.c_param,
            max_iter=1000,
            solver=self.solver,
            random_state=42
        )
        self.model.fit(X_train, labels)
        self.classes = [str(c) for c in self.model.classes_]

    def predict(self, text: str) -> IntentPrediction:
        if self.vectorizer is None or self.model is None:
            probs = {c: 1.0 / len(self.classes) for c in self.classes}
            return IntentPrediction(
                text=text,
                intent=self.classes[0],
                confidence=float(1.0 / len(self.classes)),
                probabilities=probs
            )

        X_q = self.vectorizer.transform([text])
        prob_arr = self.model.predict_proba(X_q)[0]

        probs = {c: float(p) for c, p in zip(self.classes, prob_arr)}
        best_intent = self.model.predict(X_q)[0]
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

        if self.vectorizer is None or self.model is None:
            probs = {c: 1.0 / len(self.classes) for c in self.classes}
            return [
                IntentPrediction(text=t, intent=self.classes[0], confidence=probs[self.classes[0]], probabilities=probs)
                for t in texts
            ]

        X_batch = self.vectorizer.transform(texts)
        prob_matrix = self.model.predict_proba(X_batch)
        pred_labels = self.model.predict(X_batch)

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
