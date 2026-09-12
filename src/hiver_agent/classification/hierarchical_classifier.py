"""
Two-Stage Hierarchical Intent Classifier for AmazonHelp taxonomy.

Stage 1: Predicts 1 of 7 parent domains.
Stage 2: Predicts 1 of 17 child intents conditional on parent domain.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from hiver_agent.classification.interface import IntentClassifier, IntentPrediction
from hiver_agent.classification.taxonomy import AMAZON_INTENTS


# Authoritative 7 Parent Domains and Intent Mappings
PARENT_HIERARCHY: Dict[str, List[str]] = {
    "Delivery_Order": [
        "order_tracking_delivery_status",
        "late_delivery_shipping_delay",
        "missing_package_delivered_not_received",
        "order_cancellation_request",
        "address_delivery_instructions"
    ],
    "Returns_Product": [
        "return_exchange_inquiry",
        "damaged_defective_item",
        "wrong_item_received"
    ],
    "Financial": [
        "refund_status_request",
        "payment_billing_issue",
        "promotions_discounts_coupons"
    ],
    "Account_Membership": [
        "account_access_login",
        "prime_membership_subscription"
    ],
    "Content": [
        "digital_content_kindle_video"
    ],
    "Seller": [
        "third_party_seller_issue"
    ],
    "General_Complaint": [
        "customer_service_complaint",
        "general_inquiry_other"
    ]
}

# Reverse Mapping: Intent -> Parent Domain
INTENT_TO_PARENT: Dict[str, str] = {}
for parent_name, children in PARENT_HIERARCHY.items():
    for child in children:
        INTENT_TO_PARENT[child] = parent_name


class HierarchicalIntentClassifier(IntentClassifier):
    """
    Two-Stage Hierarchical Classifier:
    1. Parent Classifier: Maps query to one of 7 parent domains.
    2. Child Classifiers: Dedicated classifier per parent domain mapping query to exact child intent.
    """
    def __init__(
        self,
        c_param: float = 1.0,
        max_features: int = 3000,
        fallback_parent: str = "General_Complaint"
    ):
        self.c_param = c_param
        self.max_features = max_features
        self.fallback_parent = fallback_parent
        self.all_intents = AMAZON_INTENTS

        self.parent_vectorizer: Optional[FeatureUnion] = None
        self.parent_model: Optional[LogisticRegression] = None
        self.parent_classes: List[str] = list(PARENT_HIERARCHY.keys())

        # Child vectorizers and models per parent domain
        self.child_vectorizers: Dict[str, Optional[FeatureUnion]] = {}
        self.child_models: Dict[str, Optional[LogisticRegression]] = {}
        self.child_classes: Dict[str, List[str]] = {}

    def _build_vectorizer(self) -> FeatureUnion:
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
        return FeatureUnion([
            ("word", word_vec),
            ("char", char_vec)
        ])

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """
        Fit Parent Classifier and Child Classifiers strictly on training data.
        """
        if not texts or not labels:
            return

        # 1. Map training labels to Parent Domains
        parent_labels = [INTENT_TO_PARENT.get(l, self.fallback_parent) for l in labels]

        # 2. Fit Stage 1 Parent Classifier
        # Check unique parent domains present in training data
        unique_parents = list(set(parent_labels))
        if len(unique_parents) < 2:
            self.parent_vectorizer = None
            self.parent_model = None
            self.parent_classes = unique_parents if unique_parents else list(PARENT_HIERARCHY.keys())
        else:
            self.parent_vectorizer = self._build_vectorizer()
            X_parent = self.parent_vectorizer.fit_transform(texts)
            self.parent_model = LogisticRegression(
                C=self.c_param,
                max_iter=1000,
                solver="lbfgs",
                random_state=42
            )
            self.parent_model.fit(X_parent, parent_labels)
            self.parent_classes = [str(c) for c in self.parent_model.classes_]

        # 3. Fit Stage 2 Child Classifiers for each parent domain
        for parent_domain, child_intents in PARENT_HIERARCHY.items():
            self.child_classes[parent_domain] = child_intents

            if len(child_intents) <= 1:
                # Single-child domain pass-through
                self.child_vectorizers[parent_domain] = None
                self.child_models[parent_domain] = None
                continue

            # Extract training subset for this parent domain
            parent_indices = [idx for idx, p in enumerate(parent_labels) if p == parent_domain]
            child_texts = [texts[i] for i in parent_indices]
            child_labels = [labels[i] for i in parent_indices]

            unique_labels = list(set(child_labels))

            if len(unique_labels) <= 1:
                # Insufficient variety in training subset
                self.child_vectorizers[parent_domain] = None
                self.child_models[parent_domain] = None
            else:
                child_vec = self._build_vectorizer()
                X_child = child_vec.fit_transform(child_texts)

                child_model = LogisticRegression(
                    C=self.c_param,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42
                )
                child_model.fit(X_child, child_labels)

                self.child_vectorizers[parent_domain] = child_vec
                self.child_models[parent_domain] = child_model

    def predict_parent(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict Stage 1 parent domain.
        """
        all_parents = list(PARENT_HIERARCHY.keys())

        if self.parent_vectorizer is None or self.parent_model is None:
            default_p = self.parent_classes[0] if self.parent_classes else self.fallback_parent
            probs = {p: (1.0 if p == default_p else 0.0) for p in all_parents}
            return default_p, 1.0, probs

        X_q = self.parent_vectorizer.transform([text])
        prob_arr = self.parent_model.predict_proba(X_q)[0]
        seen_probs = {c: float(p) for c, p in zip(self.parent_classes, prob_arr)}

        probs = {p: seen_probs.get(p, 0.0) for p in all_parents}

        best_parent = str(self.parent_model.predict(X_q)[0])
        confidence = float(probs.get(best_parent, 0.0))
        return best_parent, confidence, probs

    def predict(self, text: str) -> IntentPrediction:
        """
        End-to-End Hierarchical Prediction:
        Stage 1 (Parent Domain) -> Stage 2 (Child Intent).
        """
        best_parent, parent_conf, parent_probs = self.predict_parent(text)
        allowed_children = PARENT_HIERARCHY.get(best_parent, self.all_intents)

        child_vec = self.child_vectorizers.get(best_parent)
        child_model = self.child_models.get(best_parent)

        if child_vec is None or child_model is None:
            # Single-child domain or single-class training subset
            best_intent = allowed_children[0]
            child_conf = 1.0
            child_probs = {c: (1.0 if c == best_intent else 0.0) for c in allowed_children}
        else:
            X_q = child_vec.transform([text])
            prob_arr = child_model.predict_proba(X_q)[0]
            model_classes = [str(c) for c in child_model.classes_]
            child_probs = {c: float(p) for c, p in zip(model_classes, prob_arr)}

            best_intent = child_model.predict(X_q)[0]
            child_conf = float(child_probs[best_intent])

        # Combined transparent confidence: P(parent) * P(child|parent)
        combined_confidence = round(parent_conf * child_conf, 4)

        # Build full 17-class global probability distribution
        global_probs = {intent: 0.0 for intent in self.all_intents}
        for parent_name, p_prob in parent_probs.items():
            children = PARENT_HIERARCHY.get(parent_name, [])
            c_vec = self.child_vectorizers.get(parent_name)
            c_mod = self.child_models.get(parent_name)

            if c_vec is not None and c_mod is not None:
                X_q_p = c_vec.transform([text])
                c_prob_arr = c_mod.predict_proba(X_q_p)[0]
                c_model_classes = [str(c) for c in c_mod.classes_]
                for c_class, c_prob in zip(c_model_classes, c_prob_arr):
                    global_probs[c_class] += p_prob * c_prob
            else:
                if children:
                    for ch in children:
                        global_probs[ch] += p_prob / len(children)

        # Normalize global probabilities
        total_p = sum(global_probs.values())
        if total_p > 0:
            global_probs = {k: float(v / total_p) for k, v in global_probs.items()}

        pred = IntentPrediction(
            text=text,
            intent=best_intent,
            confidence=combined_confidence,
            probabilities=global_probs
        )

        return pred
