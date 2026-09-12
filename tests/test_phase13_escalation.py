"""
Unit tests for Day 5 Human Escalation Re-Evaluation module.
"""
import sys
import pytest
import pandas as pd
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.classification.calibrated_rule_classifier import CalibratedRuleIntentClassifier
from hiver_agent.classification.gold_prototype_classifier import GoldPrototypeIntentClassifier, CalibratedRulePrototypeHybrid
from hiver_agent.generation.generator import TemplateResponseGenerator
from hiver_agent.pipeline import CustomerSupportPipeline
from hiver_agent.decision.interface import DecisionInput
from hiver_agent.decision.policy import RiskAwarePolicy

from scripts.evaluate_phase12_hierarchical_classifier import load_gold_records
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier, split_gold_dataset_50_50
from scripts.evaluate_phase7_1 import validate_and_load_50_human_dataset, evaluate_human_verified_metrics


@pytest.fixture(scope="module")
def setup_eval_environment():
    val_50_path = Path("data/golden/amazonhelp_escalation_validation_50.csv")
    df_50 = validate_and_load_50_human_dataset(val_50_path)
    human_auto_labels = df_50["parsed_should_auto_handle"].tolist()

    df_gold, gold_records = load_gold_records()
    gold_dev_recs, _ = split_gold_dataset_50_50(gold_records, seed=42)
    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]

    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    faiss_dir = Path("data/processed/faiss_index_15k")
    vector_store = FAISSVectorStore.load(str(faiss_dir))
    base_retriever = FAISSRetriever(embedder, vector_store)
    generator = TemplateResponseGenerator()

    # Day 4 Classifier
    calib_rule = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    gold_proto = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    gold_proto.fit(dev_texts, dev_labels)
    day4_clf = CalibratedRulePrototypeHybrid(calib_rule, gold_proto)

    pipeline_day4 = CustomerSupportPipeline(
        classifier=day4_clf,
        base_retriever=base_retriever,
        generator=generator,
        confidence_threshold=0.60
    )

    return df_50, human_auto_labels, pipeline_day4


def test_human_escalation_dataset_loading():
    val_50_path = Path("data/golden/amazonhelp_escalation_validation_50.csv")
    df = validate_and_load_50_human_dataset(val_50_path)
    assert len(df) == 50
    assert "parsed_should_auto_handle" in df.columns
    assert sum(df["parsed_should_auto_handle"]) == 42  # 84% Human Auto


def test_day4_escalation_evaluation_safety(setup_eval_environment):
    df_50, human_auto_labels, pipeline_day4 = setup_eval_environment
    policy_035 = RiskAwarePolicy(risk_threshold=0.35)

    dec_day4 = []
    for idx, row in df_50.iterrows():
        cid = str(row["case_id"])
        txt = str(row["customer_text"])
        gen_res = pipeline_day4.process_message(txt, case_id=cid, exclude_case_id=cid)
        dec_inp = DecisionInput.from_generation_result(gen_res)
        dec_res = policy_035.evaluate(dec_inp)
        dec_day4.append(dec_res)

    m = evaluate_human_verified_metrics(dec_day4, human_auto_labels)
    assert m["total"] == 50
    assert m["accuracy"] == 0.74  # 74% agreement
    assert m["false_auto_handle_total"] == 0.06  # 6% false auto overall (down from 12%)


def test_cost_sensitivity_calculation():
    # Verify cost calculation logic: Cost = (False Auto * c_fa) + (False Esc * c_fe)
    c_fa = 5
    c_fe = 1
    fp = 3
    fn = 10
    cost = (fp * c_fa) + (fn * c_fe)
    assert cost == 25


def test_disagreement_classification_logic():
    # Positive: Human=False (Escalate), Baseline=AUTO, Day4=ESCALATE
    human = False
    base_dec = "AUTO_HANDLE"
    day4_dec = "ESCALATE_TO_HUMAN"

    b_corr = (base_dec == ("AUTO_HANDLE" if human else "ESCALATE_TO_HUMAN"))
    d_corr = (day4_dec == ("AUTO_HANDLE" if human else "ESCALATE_TO_HUMAN"))

    assert not b_corr and d_corr  # Positive change
