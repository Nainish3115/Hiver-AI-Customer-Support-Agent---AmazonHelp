"""
Phase 13 Day 5: Human Escalation Re-Evaluation Script.

Evaluates Baseline Intent Classifier vs Day 4 Calibrated Hybrid Classifier inside the exact same Risk-Aware Escalation Framework against 50 human escalation validation labels.
"""
import sys
import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.classification.calibrated_rule_classifier import CalibratedRuleIntentClassifier
from hiver_agent.classification.gold_prototype_classifier import GoldPrototypeIntentClassifier, CalibratedRulePrototypeHybrid
from hiver_agent.generation.generator import TemplateResponseGenerator
from hiver_agent.pipeline import CustomerSupportPipeline
from hiver_agent.decision.interface import DecisionInput, DecisionResult
from hiver_agent.decision.policy import RiskAwarePolicy

from scripts.evaluate_phase12_hierarchical_classifier import load_gold_records
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier, split_gold_dataset_50_50
from scripts.evaluate_phase7_1 import validate_and_load_50_human_dataset, evaluate_human_verified_metrics


def run_experiment():
    print("=== Phase 13 Day 5: Human Escalation Re-Evaluation ===", flush=True)

    # 1. Load 50 Human Validation set
    val_50_path = Path("data/golden/amazonhelp_escalation_validation_50.csv")
    df_50 = validate_and_load_50_human_dataset(val_50_path)
    human_auto_labels = df_50["parsed_should_auto_handle"].tolist()

    # 2. Load Gold Dev for classifier fitting
    df_gold, gold_records = load_gold_records()
    gold_dev_recs, _ = split_gold_dataset_50_50(gold_records, seed=42)
    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]

    # Initialize RAG components
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    faiss_dir = Path("data/processed/faiss_index_15k")
    vector_store = FAISSVectorStore.load(str(faiss_dir))
    base_retriever = FAISSRetriever(embedder, vector_store)
    generator = TemplateResponseGenerator()

    # 3. Classifiers
    rule_base = RuleBasedIntentClassifier()
    proto_legacy = EmbeddingPrototypeClassifier(embedder)
    proto_legacy.fit(dev_texts, dev_labels)
    baseline_clf = HybridRuleEmbeddingClassifier(rule_base, proto_legacy)

    calib_rule = CalibratedRuleIntentClassifier(use_ambiguity_resolver=True)
    gold_proto = GoldPrototypeIntentClassifier(embedder, variant_mode="mean")
    gold_proto.fit(dev_texts, dev_labels)
    day4_clf = CalibratedRulePrototypeHybrid(calib_rule, gold_proto)

    pipeline_baseline = CustomerSupportPipeline(classifier=baseline_clf, base_retriever=base_retriever, generator=generator, confidence_threshold=0.60)
    pipeline_day4 = CustomerSupportPipeline(classifier=day4_clf, base_retriever=base_retriever, generator=generator, confidence_threshold=0.60)

    # 4. Generate RAG outputs for 50 cases
    print("\nProcessing 50 cases through Baseline and Day 4 Pipelines...", flush=True)
    gen_baseline = []
    gen_day4 = []
    for idx, row in df_50.iterrows():
        cid = str(row["case_id"])
        txt = str(row["customer_text"])
        gen_b = pipeline_baseline.process_message(txt, case_id=cid, exclude_case_id=cid)
        gen_d = pipeline_day4.process_message(txt, case_id=cid, exclude_case_id=cid)
        gen_baseline.append(gen_b)
        gen_day4.append(gen_d)

    inputs_baseline = [DecisionInput.from_generation_result(g) for g in gen_baseline]
    inputs_day4 = [DecisionInput.from_generation_result(g) for g in gen_day4]

    # Primary Comparison (Risk Threshold = 0.35)
    policy_035 = RiskAwarePolicy(risk_threshold=0.35)
    dec_base_035 = [policy_035.evaluate(inp) for inp in inputs_baseline]
    dec_day4_035 = [policy_035.evaluate(inp) for inp in inputs_day4]

    m_base = evaluate_human_verified_metrics(dec_base_035, human_auto_labels)
    m_day4 = evaluate_human_verified_metrics(dec_day4_035, human_auto_labels)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {
            "System": "Baseline Intent Classifier + RiskPolicy (th=0.35)",
            "Agreement_Pct": f"{m_base['accuracy']*100:.1f}%",
            "Auto_Rate": f"{m_base['model_auto_rate']*100:.1f}%",
            "Escalation_Rate": f"{m_base['model_escalate_rate']*100:.1f}%",
            "False_Auto_Overall": f"{m_base['false_auto_handle_total']*100:.1f}%",
            "False_Auto_in_Auto": f"{m_base['false_auto_handle_in_auto']*100:.1f}%",
            "False_Escalation_Overall": f"{m_base['false_escalation_total']*100:.1f}%",
            "False_Escalation_in_Esc": f"{m_base['false_escalation_in_esc']*100:.1f}%"
        },
        {
            "System": "Day 4 Calibrated Hybrid + RiskPolicy (th=0.35)",
            "Agreement_Pct": f"{m_day4['accuracy']*100:.1f}%",
            "Auto_Rate": f"{m_day4['model_auto_rate']*100:.1f}%",
            "Escalation_Rate": f"{m_day4['model_escalate_rate']*100:.1f}%",
            "False_Auto_Overall": f"{m_day4['false_auto_handle_total']*100:.1f}%",
            "False_Auto_in_Auto": f"{m_day4['false_auto_handle_in_auto']*100:.1f}%",
            "False_Escalation_Overall": f"{m_day4['false_escalation_total']*100:.1f}%",
            "False_Escalation_in_Esc": f"{m_day4['false_escalation_in_esc']*100:.1f}%"
        }
    ]
    df_comp = pd.DataFrame(summary_rows)
    df_comp.to_csv(tables_dir / "phase_13_escalation_comparison.csv", index=False)
    print("\n--- Primary Decision Comparison (th=0.35) ---")
    print(df_comp.to_string())

    # 5. Diagnostic Threshold Sweep (0.20 to 0.50)
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep_rows = []
    for th in thresholds:
        pol = RiskAwarePolicy(risk_threshold=th)
        b_res = [pol.evaluate(inp) for inp in inputs_baseline]
        d_res = [pol.evaluate(inp) for inp in inputs_day4]
        mb = evaluate_human_verified_metrics(b_res, human_auto_labels)
        md = evaluate_human_verified_metrics(d_res, human_auto_labels)

        sweep_rows.append({
            "threshold": th,
            "baseline_agreement": f"{mb['accuracy']*100:.1f}%",
            "baseline_false_auto": f"{mb['false_auto_handle_total']*100:.1f}%",
            "baseline_false_esc": f"{mb['false_escalation_total']*100:.1f}%",
            "day4_agreement": f"{md['accuracy']*100:.1f}%",
            "day4_false_auto": f"{md['false_auto_handle_total']*100:.1f}%",
            "day4_false_esc": f"{md['false_escalation_total']*100:.1f}%"
        })
    df_sweep = pd.DataFrame(sweep_rows)
    df_sweep.to_csv(tables_dir / "phase_13_threshold_sweep.csv", index=False)
    print("\n--- Threshold Sweep (0.20 to 0.50) ---")
    print(df_sweep.to_string())

    # 6. Illustrative Cost Sensitivity Analysis
    # Cost = (False Auto Count * Cost_FA) + (False Escalation Count * Cost_FE)
    cost_scenarios = [
        ("Scenario A (FA=2, FE=1)", 2, 1),
        ("Scenario B (FA=5, FE=1)", 5, 1),
        ("Scenario C (FA=10, FE=1)", 10, 1)
    ]
    cost_rows = []
    for th in [0.30, 0.35, 0.40]:
        pol = RiskAwarePolicy(risk_threshold=th)
        b_res = [pol.evaluate(inp) for inp in inputs_baseline]
        d_res = [pol.evaluate(inp) for inp in inputs_day4]
        mb = evaluate_human_verified_metrics(b_res, human_auto_labels)
        md = evaluate_human_verified_metrics(d_res, human_auto_labels)

        for sc_name, c_fa, c_fe in cost_scenarios:
            b_cost = (mb["fp"] * c_fa) + (mb["fn"] * c_fe)
            d_cost = (md["fp"] * c_fa) + (md["fn"] * c_fe)
            cost_rows.append({
                "threshold": th,
                "scenario": sc_name,
                "baseline_cost": b_cost,
                "day4_cost": d_cost,
                "cost_savings": b_cost - d_cost
            })
    df_cost = pd.DataFrame(cost_rows)
    df_cost.to_csv(tables_dir / "phase_13_cost_sensitivity.csv", index=False)
    print("\n--- Illustrative Cost Sensitivity Table ---")
    print(df_cost.to_string())

    # 7. Detailed Disagreement Analysis (Case Level)
    disagreement_rows = []
    pos_change = 0
    neg_change = 0
    neutral_change = 0

    for idx, row in df_50.iterrows():
        cid = str(row["case_id"])
        txt = str(row["customer_text"])
        h_label = "AUTO_HANDLE" if human_auto_labels[idx] else "ESCALATE_TO_HUMAN"
        
        b_dec = dec_base_035[idx].decision
        d_dec = dec_day4_035[idx].decision
        
        b_corr = (b_dec == h_label)
        d_corr = (d_dec == h_label)

        if not b_corr and d_corr:
            change_type = "POSITIVE (Baseline Wrong -> Day 4 Correct)"
            pos_change += 1
        elif b_corr and not d_corr:
            change_type = "NEGATIVE (Baseline Correct -> Day 4 Wrong)"
            neg_change += 1
        else:
            change_type = "NEUTRAL (No Change in Correctness)"
            neutral_change += 1

        disagreement_rows.append({
            "case_id": cid,
            "customer_text": txt[:80] + "...",
            "human_label": h_label,
            "baseline_decision": b_dec,
            "baseline_risk_score": dec_base_035[idx].risk_score,
            "day4_decision": d_dec,
            "day4_risk_score": dec_day4_035[idx].risk_score,
            "day4_intent": gen_day4[idx].predicted_intent,
            "day4_intent_conf": gen_day4[idx].intent_confidence,
            "change_classification": change_type
        })

    df_disagreements = pd.DataFrame(disagreement_rows)
    df_disagreements.to_csv(tables_dir / "phase_13_disagreements.csv", index=False)

    print(f"\n--- Disagreement Classification ---")
    print(f"Positive Changes (Baseline Wrong -> Day 4 Correct): {pos_change}")
    print(f"Negative Changes (Baseline Correct -> Day 4 Wrong): {neg_change}")
    print(f"Neutral Changes: {neutral_change}")

    print(f"\nAll Day 5 tables saved under {tables_dir}/")


if __name__ == "__main__":
    run_experiment()
