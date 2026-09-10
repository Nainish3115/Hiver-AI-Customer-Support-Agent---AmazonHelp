"""
Unit tests for Phase 7.1 Human Escalation Validation & Evaluation Correction.
"""
import pytest
import pandas as pd
from pathlib import Path
from hiver_agent.decision.interface import DecisionResult
from hiver_agent.config import load_config
from scripts.evaluate_phase7_1 import (
    validate_and_load_50_human_dataset,
    evaluate_human_verified_metrics,
    compute_cohens_kappa
)


def test_50_human_dataset_validation(tmp_path):
    """Test loading valid 50-case human dataset and rejecting invalid row counts or missing escalation reasons."""
    valid_csv = Path("data/golden/amazonhelp_escalation_validation_50.csv")
    df = validate_and_load_50_human_dataset(valid_csv)
    assert len(df) == 50
    assert "parsed_should_auto_handle" in df.columns

    # Test invalid count
    bad_csv = tmp_path / "short_50.csv"
    df.head(10).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Expected exactly 50 validation cases"):
        validate_and_load_50_human_dataset(bad_csv)

    # Test missing escalation reason for FALSE
    bad_false_csv = tmp_path / "missing_reason.csv"
    df_bad = df.copy()
    df_bad.loc[df_bad["parsed_should_auto_handle"] == False, "escalation_reason"] = ""
    df_bad.to_csv(bad_false_csv, index=False)
    with pytest.raises(ValueError, match="missing escalation_reason"):
        validate_and_load_50_human_dataset(bad_false_csv)


def test_confusion_matrix_and_metrics_calculation():
    """
    Test confusion matrix, false auto-handle, false escalation calculations across all 4 quadrants:
    1. Human AUTO + Model AUTO (TP)
    2. Human ESCALATE + Model AUTO (FP -> False Auto-Handle)
    3. Human AUTO + Model ESCALATE (FN -> False Escalation)
    4. Human ESCALATE + Model ESCALATE (TN)
    """
    # 4 cases covering all quadrants
    human_labels = [True, False, True, False]
    dec_results = [
        DecisionResult(decision="AUTO_HANDLE", risk_score=0.1, reason_code="SAFE", reason="Safe", signals={}, recommended_action="auto_handle"),       # TP
        DecisionResult(decision="AUTO_HANDLE", risk_score=0.2, reason_code="SAFE", reason="Safe", signals={}, recommended_action="auto_handle"),       # FP
        DecisionResult(decision="ESCALATE_TO_HUMAN", risk_score=0.8, reason_code="RISK", reason="Risk", signals={}, recommended_action="human_review"), # FN
        DecisionResult(decision="ESCALATE_TO_HUMAN", risk_score=0.9, reason_code="RISK", reason="Risk", signals={}, recommended_action="human_review"), # TN
    ]

    m = evaluate_human_verified_metrics(dec_results, human_labels)

    assert m["total"] == 4
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["accuracy"] == 0.50

    # FP / 4 total = 0.25; FP / 2 model auto = 0.50
    assert m["false_auto_handle_total"] == 0.25
    assert m["false_auto_handle_in_auto"] == 0.50

    # FN / 4 total = 0.25; FN / 2 model escalate = 0.50
    assert m["false_escalation_total"] == 0.25
    assert m["false_escalation_in_esc"] == 0.50


def test_cohens_kappa_perfect_agreement():
    """Test Cohen's kappa for identical annotator labels."""
    a1 = [True, False, True, False, True]
    a2 = [True, False, True, False, True]
    kappa = compute_cohens_kappa(a1, a2)
    assert kappa == 1.0


def test_deterministic_50_case_sampling_file_exists():
    """Test that data/golden/amazonhelp_escalation_validation_50.csv exists with 50 rows."""
    file_path = Path("data/golden/amazonhelp_escalation_validation_50.csv")
    assert file_path.exists(), "50-case escalation validation file must exist."
    df = pd.read_csv(file_path)
    assert len(df) == 50, f"Expected 50 cases, found {len(df)}."
    required_cols = {"case_id", "customer_text", "gold_intent", "should_auto_handle", "escalation_reason", "annotator_notes"}
    assert required_cols.issubset(set(df.columns))


def test_canonical_configuration_alignment():
    """Test that loaded config matches canonical executable settings exactly."""
    cfg = load_config()
    assert cfg.decision.risk_threshold == 0.35
    assert cfg.decision.confidence_threshold == 0.60
    assert cfg.decision.min_retrieval_score == 0.70
    assert cfg.decision.weights.get("intent_uncertainty") == 0.40
    assert cfg.decision.weights.get("retrieval_quality") == 0.30
    assert cfg.decision.weights.get("category_risk") == 0.30
