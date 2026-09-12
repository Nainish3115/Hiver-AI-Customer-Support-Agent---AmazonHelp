"""
Abstract base interface and prediction schemas for intent classification.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IntentPrediction(BaseModel):
    """
    Schema for intent classification prediction outputs.
    """
    text: str = Field(..., description="Input text query")
    intent: str = Field(..., description="Predicted intent label string")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Per-class probability distribution")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional prediction metadata")


class IntentClassifier(ABC):
    """
    Abstract base interface for intent classification algorithms.
    """
    @abstractmethod
    def predict(self, text: str) -> IntentPrediction:
        """
        Predict intent for a single text utterance.
        """
        pass

    def predict_batch(self, texts: List[str]) -> List[IntentPrediction]:
        """
        Predict intent for a batch of text utterances.
        """
        return [self.predict(t) for t in texts]
