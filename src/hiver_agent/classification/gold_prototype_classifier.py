"""
Gold Prototype Intent Classifier and Variants for AmazonHelp taxonomy.

Phase 12 Day 4 Implementation:
- GoldPrototypeIntentClassifier: Clean nearest-centroid prototype trained ONLY on Gold Dev examples.
- Variants:
  - Variant A: Simple mean centroid.
  - Variant B: Medoid centroid (most representative Dev example per intent).
  - Variant C: Multi-prototype (sub-clustering per intent if Dev support >= 4).
- CalibratedRulePrototypeHybrid: Controlled hybrid combining calibrated rules with prototype semantic search.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans

from hiver_agent.classification.interface import IntentClassifier, IntentPrediction
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from hiver_agent.classification.calibrated_rule_classifier import CalibratedRuleIntentClassifier


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


class GoldPrototypeIntentClassifier(IntentClassifier):
    """
    Clean Dev-only Prototype Classifier using all-MiniLM-L6-v2 embeddings.
    Variant modes:
    - 'mean': Normalized mean vector centroid per intent.
    - 'medoid': Medoid vector per intent (Dev example with highest mean similarity to class members).
    - 'multi': Sub-centroid multi-prototype per intent (KMeans with k=2 if dev support >= 4).
    """
    def __init__(
        self,
        embedding_provider: SentenceTransformerEmbeddingProvider,
        variant_mode: str = "mean"
    ):
        self.embedding_provider = embedding_provider
        self.variant_mode = variant_mode.lower()
        self.all_intents = AMAZON_INTENTS

        # Centroids storage
        # Key: intent string -> Value: list of 384-d normalized vectors (1 for mean/medoid, multiple for multi)
        self.centroids: Dict[str, List[np.ndarray]] = {}
        self.dev_support: Dict[str, int] = {c: 0 for c in self.all_intents}

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Fit prototypes strictly on Gold Dev texts and labels.
        """
        if not texts or not labels:
            return

        embeddings = self.embedding_provider.embed_texts(texts) # Shape: (N, 384)
        
        # Group embeddings by intent label
        class_embs: Dict[str, List[np.ndarray]] = {c: [] for c in self.all_intents}
        for emb, lbl in zip(embeddings, labels):
            if lbl in class_embs:
                class_embs[lbl].append(normalize_vector(emb))

        for intent in self.all_intents:
            embs = class_embs[intent]
            self.dev_support[intent] = len(embs)

            if not embs:
                # If intent has 0 Dev examples, centroid is zero vector (fallback)
                self.centroids[intent] = [np.zeros(384)]
                continue

            emb_matrix = np.array(embs) # (N_c, 384)

            if self.variant_mode == "mean":
                mean_vec = normalize_vector(np.mean(emb_matrix, axis=0))
                self.centroids[intent] = [mean_vec]

            elif self.variant_mode == "medoid":
                if len(embs) == 1:
                    self.centroids[intent] = [emb_matrix[0]]
                else:
                    # Cosine sim matrix within class
                    sim_matrix = np.dot(emb_matrix, emb_matrix.T)
                    mean_sims = np.mean(sim_matrix, axis=1)
                    best_medoid_idx = int(np.argmax(mean_sims))
                    self.centroids[intent] = [emb_matrix[best_medoid_idx]]

            elif self.variant_mode == "multi":
                if len(embs) < 4:
                    mean_vec = normalize_vector(np.mean(emb_matrix, axis=0))
                    self.centroids[intent] = [mean_vec]
                else:
                    # Simple 2-means clustering in pure NumPy
                    c1 = emb_matrix[0]
                    c2 = emb_matrix[len(embs) // 2]
                    for _ in range(5):
                        sim1 = np.dot(emb_matrix, c1)
                        sim2 = np.dot(emb_matrix, c2)
                        group1 = emb_matrix[sim1 >= sim2]
                        group2 = emb_matrix[sim1 < sim2]
                        if len(group1) > 0:
                            c1 = normalize_vector(np.mean(group1, axis=0))
                        if len(group2) > 0:
                            c2 = normalize_vector(np.mean(group2, axis=0))
                    self.centroids[intent] = [c1, c2]
            else:
                raise ValueError(f"Unknown variant_mode: {self.variant_mode}")

    def predict(self, text: str) -> IntentPrediction:
        q_vec = normalize_vector(self.embedding_provider.embed_query(text).flatten())

        intent_sims: Dict[str, float] = {}
        for intent in self.all_intents:
            centroids_list = self.centroids.get(intent, [np.zeros(384)])
            # Max similarity over all sub-centroids for this intent
            max_sim = max(float(np.dot(q_vec, c_vec)) for c_vec in centroids_list)
            intent_sims[intent] = max_sim

        sorted_sims = sorted(intent_sims.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_sim = sorted_sims[0]
        second_intent, second_sim = sorted_sims[1]
        margin = round(top_sim - second_sim, 4)

        # Min-max scale similarities for probabilities display
        sim_array = np.array([intent_sims[i] for i in self.all_intents])
        exp_sims = np.exp((sim_array - np.max(sim_array)) * 10.0)
        prob_dist = {i: float(p) for i, p in zip(self.all_intents, exp_sims / np.sum(exp_sims))}

        metadata = {
            "top_similarity": round(top_sim, 4),
            "second_similarity": round(second_sim, 4),
            "margin": margin,
            "variant_mode": self.variant_mode,
            "dev_support": self.dev_support.get(top_intent, 0)
        }

        return IntentPrediction(
            text=text,
            intent=top_intent,
            confidence=round(top_sim, 4),
            probabilities=prob_dist,
            metadata=metadata
        )


class CalibratedRulePrototypeHybrid(IntentClassifier):
    """
    Controlled Hybrid combining Calibrated Rules with Gold Prototype Semantic Search.
    Policy:
    1. If Calibrated Rule fires with HIGH specificity (>= 0.85 conf & no conflict), trust rule.
    2. Else if Prototype margin >= 0.08, trust Prototype semantic match.
    3. Otherwise blend rule and prototype distributions.
    """
    def __init__(
        self,
        rule_clf: CalibratedRuleIntentClassifier,
        proto_clf: GoldPrototypeIntentClassifier,
        rule_threshold: float = 0.85,
        margin_threshold: float = 0.08
    ):
        self.rule_clf = rule_clf
        self.proto_clf = proto_clf
        self.rule_threshold = rule_threshold
        self.margin_threshold = margin_threshold

    def predict(self, text: str) -> IntentPrediction:
        rule_pred = self.rule_clf.predict(text)
        rule_meta = rule_pred.metadata or {}

        # 1. High-specificity rule without conflict -> trust rule directly
        if rule_pred.confidence >= self.rule_threshold and not rule_meta.get("conflict_detected", False) and rule_pred.intent != "general_inquiry_other":
            return IntentPrediction(
                text=text,
                intent=rule_pred.intent,
                confidence=rule_pred.confidence,
                probabilities=rule_pred.probabilities,
                metadata={"decision_source": "high_specificity_rule", "rule_confidence": rule_pred.confidence}
            )

        # 2. Prototype prediction
        proto_pred = self.proto_clf.predict(text)
        proto_meta = proto_pred.metadata or {}
        proto_margin = proto_meta.get("margin", 0.0)

        # If rule had zero keyword matches and prototype has strong margin, use prototype
        if rule_meta.get("rule_evidence") == "none" or (proto_margin >= self.margin_threshold and proto_pred.confidence >= 0.45):
            return IntentPrediction(
                text=text,
                intent=proto_pred.intent,
                confidence=proto_pred.confidence,
                probabilities=proto_pred.probabilities,
                metadata={"decision_source": "prototype_semantic", "proto_margin": proto_margin, "proto_sim": proto_pred.confidence}
            )

        # 3. Blend rule and prototype probability distributions
        blended_probs = {}
        r_weight = rule_pred.confidence
        p_weight = max(0.30, 1.0 - r_weight)

        for intent in AMAZON_INTENTS:
            r_p = rule_pred.probabilities.get(intent, 0.0) if rule_pred.probabilities else 0.0
            p_p = proto_pred.probabilities.get(intent, 0.0) if proto_pred.probabilities else 0.0
            blended_probs[intent] = r_weight * r_p + p_weight * p_p

        total_b = sum(blended_probs.values())
        norm_probs = {k: float(v / total_b) for k, v in blended_probs.items()}
        best_intent = max(norm_probs.items(), key=lambda x: x[1])[0]
        best_conf = round(float(norm_probs[best_intent]), 4)

        return IntentPrediction(
            text=text,
            intent=best_intent,
            confidence=best_conf,
            probabilities=norm_probs,
            metadata={
                "decision_source": "blended_rule_prototype",
                "rule_intent": rule_pred.intent,
                "proto_intent": proto_pred.intent,
                "rule_confidence": rule_pred.confidence,
                "proto_margin": proto_margin
            }
        )
