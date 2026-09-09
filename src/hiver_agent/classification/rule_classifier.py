"""
Rule-based keyword baseline classifier for AmazonHelp intents.
"""
from typing import List, Dict, Any, Optional
from hiver_agent.classification.interface import IntentClassifier, IntentPrediction
from hiver_agent.analysis.intent_discovery import classify_single_text_intent, INTENT_PATTERNS
from hiver_agent.classification.taxonomy import AMAZON_INTENTS


class RuleBasedIntentClassifier(IntentClassifier):
    """
    Baseline classifier using keyword pattern matching.
    """
    def __init__(self):
        self.intents = AMAZON_INTENTS

    def predict(self, text: str) -> IntentPrediction:
        intent = classify_single_text_intent(text)
        confidence = 0.85 if intent != "general_inquiry_other" else 0.40

        # Uniform or baseline probabilities
        probs = {i: (0.85 if i == intent else 0.15 / (len(self.intents) - 1)) for i in self.intents}

        return IntentPrediction(
            text=text,
            intent=intent,
            confidence=confidence,
            probabilities=probs
        )
