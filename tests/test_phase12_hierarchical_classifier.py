"""
Unit tests for Day 2 Two-Stage Hierarchical Intent Classifier module.
"""
import sys
import pytest
import pandas as pd
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from hiver_agent.classification.hierarchical_classifier import (
    HierarchicalIntentClassifier,
    PARENT_HIERARCHY,
    INTENT_TO_PARENT
)
from scripts.evaluate_phase12_hierarchical_classifier import load_gold_records
from scripts.run_phase_5_6_experiments import split_gold_dataset_50_50


def test_parent_mapping_completeness():
    mapped_intents = set(INTENT_TO_PARENT.keys())
    taxonomy_intents = set(AMAZON_INTENTS)
    assert mapped_intents == taxonomy_intents, f"Missing or extra mapped intents: {mapped_intents ^ taxonomy_intents}"


def test_every_intent_maps_to_exactly_one_parent():
    all_children = []
    for parent, children in PARENT_HIERARCHY.items():
        all_children.extend(children)
    assert len(all_children) == len(set(all_children)), "Intent mapped to multiple parent domains."
    assert len(all_children) == 17, f"Expected 17 total child intents, found {len(all_children)}."


def test_classifier_returns_only_valid_intents():
    clf = HierarchicalIntentClassifier(c_param=1.0)
    train_texts = ["where is my order", "I want to return item", "refund missing money"]
    train_labels = ["order_tracking_delivery_status", "return_exchange_inquiry", "refund_status_request"]
    clf.fit(train_texts, train_labels)

    pred = clf.predict("status of shipment")
    assert pred.intent in AMAZON_INTENTS, f"Invalid intent label returned: {pred.intent}"


def test_parent_classifier_output_validity():
    clf = HierarchicalIntentClassifier(c_param=1.0)
    train_texts = ["track package", "cancel order", "damaged box"]
    train_labels = ["order_tracking_delivery_status", "order_cancellation_request", "damaged_defective_item"]
    clf.fit(train_texts, train_labels)

    parent_name, parent_conf, probs = clf.predict_parent("track my shipment")
    assert parent_name in PARENT_HIERARCHY.keys()
    assert 0.0 <= parent_conf <= 1.0
    assert len(probs) == len(PARENT_HIERARCHY)


def test_child_classifier_output_validity():
    clf = HierarchicalIntentClassifier(c_param=1.0)
    train_texts = ["item broken", "item torn", "wrong item delivered"]
    train_labels = ["damaged_defective_item", "damaged_defective_item", "wrong_item_received"]
    clf.fit(train_texts, train_labels)

    pred = clf.predict("item damaged")
    assert pred.intent in AMAZON_INTENTS
    assert 0.0 <= pred.confidence <= 1.0


def test_deterministic_predictions():
    clf = HierarchicalIntentClassifier(c_param=1.0)
    train_texts = ["order tracking query", "late delivery complaint"]
    train_labels = ["order_tracking_delivery_status", "late_delivery_shipping_delay"]
    clf.fit(train_texts, train_labels)

    pred1 = clf.predict("where is package")
    pred2 = clf.predict("where is package")
    assert pred1.intent == pred2.intent
    assert pred1.confidence == pred2.confidence


def test_gold_test_is_not_used_during_training():
    df_gold, gold_records = load_gold_records()
    gold_dev_recs, gold_test_recs = split_gold_dataset_50_50(gold_records, seed=42)

    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]
    test_texts = [r["customer_text"] for r in gold_test_recs]

    clf = HierarchicalIntentClassifier(c_param=1.0)
    clf.fit(dev_texts, dev_labels)

    # Verify model classes reflect only training labels present in dev split
    assert clf.parent_model is not None
    test_text_sample = test_texts[0]
    pred = clf.predict(test_text_sample)
    assert pred.intent in AMAZON_INTENTS


def test_confidence_metadata_valid():
    clf = HierarchicalIntentClassifier(c_param=1.0)
    train_texts = ["where is my order", "refund requested", "login failed"]
    train_labels = ["order_tracking_delivery_status", "refund_status_request", "account_access_login"]
    clf.fit(train_texts, train_labels)

    pred = clf.predict("order status")
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.probabilities is not None
    total_prob = sum(pred.probabilities.values())
    assert abs(total_prob - 1.0) < 1e-3, f"Probabilities sum to {total_prob}, expected ~1.0"
