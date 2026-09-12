"""
Unit tests for Phase 12 Intent Classification Diagnostic module and artifacts.
"""
import sys
import pytest
import pandas as pd
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from scripts.run_phase12_intent_diagnostic import PARENT_HIERARCHY, INTENT_TO_PARENT


def test_gold_annotation_dataset_schema():
    gold_csv = Path("data/golden/amazonhelp_gold_200_annotation.csv")
    assert gold_csv.exists(), "AmazonHelp Gold 200 CSV file missing."
    df = pd.read_csv(gold_csv)
    assert len(df) == 200, f"Expected 200 rows, found {len(df)}."
    assert "case_id" in df.columns
    assert "customer_text" in df.columns
    assert "final_human_intent" in df.columns
    
    unique_intents = set(df["final_human_intent"].dropna())
    for intent in unique_intents:
        assert intent in AMAZON_INTENTS, f"Unknown intent found in gold dataset: {intent}"


def test_parent_hierarchy_taxonomy_mapping():
    mapped_intents = set(INTENT_TO_PARENT.keys())
    taxonomy_intents = set(AMAZON_INTENTS)
    
    assert mapped_intents == taxonomy_intents, f"Hierarchy missing or extra intents: {mapped_intents ^ taxonomy_intents}"
    assert len(PARENT_HIERARCHY) == 7, "Expected 7 parent categories."


def test_diagnostic_tables_generated():
    tables_dir = Path("reports/tables")
    cm_csv = tables_dir / "phase_12_intent_confusion_matrix.csv"
    err_csv = tables_dir / "phase_12_intent_error_analysis.csv"
    pairs_csv = tables_dir / "phase_12_intent_confusion_pairs.csv"
    
    assert cm_csv.exists(), "phase_12_intent_confusion_matrix.csv missing."
    assert err_csv.exists(), "phase_12_intent_error_analysis.csv missing."
    assert pairs_csv.exists(), "phase_12_intent_confusion_pairs.csv missing."
    
    df_cm = pd.read_csv(cm_csv)
    df_err = pd.read_csv(err_csv)
    df_pairs = pd.read_csv(pairs_csv)
    
    assert len(df_cm) == 17, "Confusion matrix should be 17x17."
    assert len(df_err) == 200, "Error analysis should contain all 200 cases."
    assert len(df_pairs) > 0, "Confusion pairs should be non-empty."


def test_confidence_bucketing_partition():
    from scripts.run_phase12_intent_diagnostic import analyze_confidence_buckets
    mock_evals = [
        {"confidence": 0.45, "is_correct": True},
        {"confidence": 0.55, "is_correct": False},
        {"confidence": 0.85, "is_correct": True},
        {"confidence": 0.85, "is_correct": False},
    ]
    df_buckets = analyze_confidence_buckets(mock_evals)
    total_counted = df_buckets["count"].sum()
    assert total_counted == 4, f"Expected 4 cases, got {total_counted}"
