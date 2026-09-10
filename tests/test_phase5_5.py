"""
Unit tests for Phase 5.5 Human Gold Evaluation Dataset and Workflow.
"""
import sys
import pytest
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
scripts_dir = project_root / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from annotate_amazonhelp_gold import save_annotation_progress


def test_200_annotation_dataset_schema_and_integrity():
    parquet_path = Path("data/golden/amazonhelp_gold_200_annotation.parquet")
    csv_path = Path("data/golden/amazonhelp_gold_200_annotation.csv")

    assert parquet_path.exists()
    assert csv_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 200

    expected_cols = ["case_id", "customer_text", "suggested_intent", "final_human_intent", "annotator_notes"]
    assert list(df.columns) == expected_cols

    # Verify all suggested intents are valid 17 taxonomy intents
    unique_sugg = df["suggested_intent"].unique()
    for s in unique_sugg:
        assert s in AMAZON_INTENTS, f"Invalid suggested intent found: {s}"

    # Verify final_human_intent has 200 non-empty valid human gold labels
    assert df["final_human_intent"].dropna().count() == 200
    for h_label in df["final_human_intent"]:
        assert str(h_label).strip() in AMAZON_INTENTS, f"Invalid human gold intent label: {h_label}"


def test_intent_id_validation():
    for intent in AMAZON_INTENTS:
        assert isinstance(intent, str)
        assert len(intent) > 0
        assert intent.islower()


def test_incomplete_annotation_detection():
    df = pd.DataFrame([
        {"case_id": "C1", "customer_text": "text1", "suggested_intent": "general_inquiry_other", "final_human_intent": "refund_status_request", "annotator_notes": ""},
        {"case_id": "C2", "customer_text": "text2", "suggested_intent": "general_inquiry_other", "final_human_intent": None, "annotator_notes": ""},
        {"case_id": "C3", "customer_text": "text3", "suggested_intent": "general_inquiry_other", "final_human_intent": "", "annotator_notes": ""}
    ])

    annotated = [v for v in df["final_human_intent"] if v is not None and str(v).strip() != ""]
    unannotated = [v for v in df["final_human_intent"] if v is None or str(v).strip() == ""]

    assert len(annotated) == 1
    assert len(unannotated) == 2


def test_save_and_resume_behavior(tmp_path):
    csv_tmp = tmp_path / "test_annotation.csv"
    parquet_tmp = tmp_path / "test_annotation.parquet"

    df_initial = pd.DataFrame([
        {"case_id": "C1", "customer_text": "text1", "suggested_intent": "general_inquiry_other", "final_human_intent": None, "annotator_notes": ""},
        {"case_id": "C2", "customer_text": "text2", "suggested_intent": "general_inquiry_other", "final_human_intent": None, "annotator_notes": ""}
    ])

    save_annotation_progress(df_initial, csv_tmp, parquet_tmp)
    assert csv_tmp.exists()
    assert parquet_tmp.exists()

    # Simulate annotating 1 row and saving
    df_loaded = pd.read_csv(csv_tmp, dtype=str)
    df_loaded.fillna("", inplace=True)
    df_loaded.at[0, "final_human_intent"] = "refund_status_request"
    df_loaded.at[0, "annotator_notes"] = "Verified by human"

    save_annotation_progress(df_loaded, csv_tmp, parquet_tmp)

    # Resume check
    df_resumed = pd.read_csv(csv_tmp, dtype=str)
    df_resumed.fillna("", inplace=True)
    assert df_resumed.at[0, "final_human_intent"] == "refund_status_request"
    assert df_resumed.at[0, "annotator_notes"] == "Verified by human"
    assert df_resumed.at[1, "final_human_intent"] == ""
