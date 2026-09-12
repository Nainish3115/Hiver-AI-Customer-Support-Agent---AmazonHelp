"""
Phase 12: Intent Classification Diagnostic Script.

Executes comprehensive diagnostic analysis on the 200 Human Gold intent benchmark:
1. E2E Baseline Reproduction & Per-Class Precision/Recall/F1
2. Full 17x17 Confusion Matrix Generation
3. Error Analysis Case Extraction & Top Confusion Pairs
4. Confidence Calibration Bucketing (0.0-0.49 to 0.90-1.00)
5. Experimental Hierarchical Classification Analysis (7 parent domains)
6. Embedding Prototype Centroid Analysis & Leakage Inspection

Generates:
- reports/tables/phase_12_intent_confusion_matrix.csv
- reports/tables/phase_12_intent_error_analysis.csv
- reports/tables/phase_12_intent_confusion_pairs.csv
"""
import sys
import json
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)
sys.path.insert(0, str(Path(root_dir) / "src"))

from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from hiver_agent.classification.taxonomy import AMAZON_INTENTS
from hiver_agent.classification.interface import IntentPrediction
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier, split_gold_dataset_50_50


# Parent domain mapping for Step 5 Hierarchical Analysis
PARENT_HIERARCHY = {
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

INTENT_TO_PARENT = {}
for parent, children in PARENT_HIERARCHY.items():
    for child in children:
        INTENT_TO_PARENT[child] = parent


def load_gold_records():
    gold_csv = Path(root_dir) / "data/golden/amazonhelp_gold_200_annotation.csv"
    df_gold = pd.read_csv(gold_csv, encoding="utf-8")
    records = df_gold.to_dict(orient="records")
    for r in records:
        r["case_id"] = str(r["case_id"])
        r["customer_text"] = str(r["customer_text"])
        r["intent"] = str(r["final_human_intent"])
    return df_gold, records


def build_pipeline_classifier(gold_records):
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    gold_dev_recs, gold_test_recs = split_gold_dataset_50_50(gold_records, seed=42)
    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]

    rule_clf = RuleBasedIntentClassifier()
    proto_clf = EmbeddingPrototypeClassifier(embedder)
    proto_clf.fit(dev_texts, dev_labels)
    hybrid_clf = HybridRuleEmbeddingClassifier(rule_clf, proto_clf)
    
    return hybrid_clf, embedder, rule_clf, proto_clf, gold_dev_recs, gold_test_recs


def run_baseline_evaluation(hybrid_clf, gold_records):
    results = []
    for r in gold_records:
        txt = r["customer_text"]
        pred: IntentPrediction = hybrid_clf.predict(txt)
        
        rule_pred = hybrid_clf.rule_clf.predict(txt)
        rule_used = (rule_pred.confidence >= 0.80 and rule_pred.intent != "general_inquiry_other")
        
        sorted_probs = sorted(pred.probabilities.items(), key=lambda x: x[1], reverse=True)
        top2 = sorted_probs[:2]
        competing_intent = top2[1][0] if len(top2) > 1 else "none"
        competing_prob = round(top2[1][1], 4) if len(top2) > 1 else 0.0
        
        results.append({
            "case_id": r["case_id"],
            "customer_text": txt,
            "gold_intent": r["intent"],
            "predicted_intent": pred.intent,
            "confidence": round(pred.confidence, 4),
            "is_correct": pred.intent == r["intent"],
            "classifier_stage": "rule" if rule_used else "embedding_prototype",
            "competing_intent": competing_intent,
            "competing_confidence": competing_prob
        })
    return results


def analyze_per_class_performance(eval_results):
    y_true = [r["gold_intent"] for r in eval_results]
    y_pred = [r["predicted_intent"] for r in eval_results]
    
    intents = AMAZON_INTENTS
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=intents, zero_division=0)
    
    per_class = []
    for i, intent in enumerate(intents):
        per_class.append({
            "intent": intent,
            "support": int(supp[i]),
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1_score": round(float(f1[i]), 4)
        })
    return pd.DataFrame(per_class)


def analyze_confusion_matrix(eval_results):
    y_true = [r["gold_intent"] for r in eval_results]
    y_pred = [r["predicted_intent"] for r in eval_results]
    intents = AMAZON_INTENTS
    
    cm = pd.crosstab(
        pd.Series(y_true, name="Gold Intent"),
        pd.Series(y_pred, name="Predicted Intent")
    ).reindex(index=intents, columns=intents, fill_value=0)
    
    # Top confusion pairs (where gold != predicted)
    pairs = Counter()
    for r in eval_results:
        if not r["is_correct"]:
            pairs[(r["gold_intent"], r["predicted_intent"])] += 1
            
    pair_rows = []
    for (gold_i, pred_i), count in pairs.most_common():
        pair_rows.append({
            "gold_intent": gold_i,
            "predicted_intent": pred_i,
            "gold_parent": INTENT_TO_PARENT.get(gold_i, "Unknown"),
            "predicted_parent": INTENT_TO_PARENT.get(pred_i, "Unknown"),
            "is_cross_parent": INTENT_TO_PARENT.get(gold_i) != INTENT_TO_PARENT.get(pred_i),
            "error_count": count,
            "percentage_of_all_errors": round((count / max(1, len(eval_results) - sum(r['is_correct'] for r in eval_results))) * 100, 2)
        })
    df_pairs = pd.DataFrame(pair_rows)
    return cm, df_pairs


def analyze_confidence_buckets(eval_results):
    buckets = [
        ("0.00–0.49", 0.00, 0.4999),
        ("0.50–0.59", 0.50, 0.5999),
        ("0.60–0.69", 0.60, 0.6999),
        ("0.70–0.79", 0.70, 0.7999),
        ("0.80–0.89", 0.80, 0.8999),
        ("0.90–1.00", 0.90, 1.0000),
    ]
    
    total = len(eval_results)
    bucket_stats = []
    for name, low, high in buckets:
        sub = [r for r in eval_results if low <= r["confidence"] <= high]
        cnt = len(sub)
        corr = sum(1 for r in sub if r["is_correct"])
        acc = round((corr / cnt), 4) if cnt > 0 else 0.0
        pct = round((cnt / total) * 100, 2)
        bucket_stats.append({
            "confidence_bucket": name,
            "count": cnt,
            "percentage_of_total": pct,
            "correct_count": corr,
            "accuracy": acc
        })
    return pd.DataFrame(bucket_stats)


def run_hierarchical_experiment(eval_results):
    """
    Evaluates experimental hierarchical routing:
    1. Parent Category Classification Accuracy
    2. Within-Parent Child Intent Accuracy
    """
    parent_true = [INTENT_TO_PARENT[r["gold_intent"]] for r in eval_results]
    parent_pred = [INTENT_TO_PARENT[r["predicted_intent"]] for r in eval_results]
    
    parent_acc = accuracy_score(parent_true, parent_pred)
    parent_f1 = f1_score(parent_true, parent_pred, average="macro", zero_division=0)
    
    within_parent_correct = 0
    within_parent_total = 0
    for r in eval_results:
        gold_p = INTENT_TO_PARENT[r["gold_intent"]]
        pred_p = INTENT_TO_PARENT[r["predicted_intent"]]
        if gold_p == pred_p:
            within_parent_total += 1
            if r["gold_intent"] == r["predicted_intent"]:
                within_parent_correct += 1
                
    within_child_acc = (within_parent_correct / within_parent_total) if within_parent_total > 0 else 0.0
    
    return {
        "parent_accuracy": round(parent_acc, 4),
        "parent_macro_f1": round(parent_f1, 4),
        "correct_parent_count": sum(1 for gt, pt in zip(parent_true, parent_pred) if gt == pt),
        "cross_parent_error_count": sum(1 for gt, pt in zip(parent_true, parent_pred) if gt != pt),
        "within_parent_child_accuracy": round(within_child_acc, 4),
        "within_parent_child_error_count": within_parent_total - within_parent_correct
    }


def run_prototype_leakage_analysis(embedder, gold_records, gold_dev_recs, gold_test_recs):
    """
    Evaluates embedding prototype centroids under three protocol settings:
    1. Untouched Test Set (100 cases, centroids fit on 100 dev cases) -> Leakage-Free Test Baseline
    2. Full Gold Evaluation (200 cases, centroids fit on 100 dev cases) -> Pipeline Baseline
    3. Full Gold Fitting (200 cases, centroids fit on ALL 200 cases) -> Diagnostic Upper Bound (100% Data Reuse)
    """
    # 1. Untouched Test Set (fit on 100 dev, test on 100 test)
    proto_dev = EmbeddingPrototypeClassifier(embedder)
    proto_dev.fit([r["customer_text"] for r in gold_dev_recs], [r["intent"] for r in gold_dev_recs])
    
    y_test_true = [r["intent"] for r in gold_test_recs]
    y_test_pred = [proto_dev.predict(r["customer_text"]).intent for r in gold_test_recs]
    acc_untouched = accuracy_score(y_test_true, y_test_pred)
    f1_untouched = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)

    # 2. Hybrid Rule + Proto fit on 100 dev, test on untouched 100 test
    rule_clf = RuleBasedIntentClassifier()
    hybrid_dev = HybridRuleEmbeddingClassifier(rule_clf, proto_dev)
    y_test_hybrid = [hybrid_dev.predict(r["customer_text"]).intent for r in gold_test_recs]
    acc_untouched_hybrid = accuracy_score(y_test_true, y_test_hybrid)
    f1_untouched_hybrid = f1_score(y_test_true, y_test_hybrid, average="macro", zero_division=0)

    # 3. Full Gold Fitting (fit on all 200 gold cases, test on all 200 cases)
    proto_full = EmbeddingPrototypeClassifier(embedder)
    proto_full.fit([r["customer_text"] for r in gold_records], [r["intent"] for r in gold_records])
    
    y_full_true = [r["intent"] for r in gold_records]
    y_full_proto = [proto_full.predict(r["customer_text"]).intent for r in gold_records]
    acc_full_proto = accuracy_score(y_full_true, y_full_proto)
    f1_full_proto = f1_score(y_full_true, y_full_proto, average="macro", zero_division=0)
    
    hybrid_full = HybridRuleEmbeddingClassifier(rule_clf, proto_full)
    y_full_hybrid = [hybrid_full.predict(r["customer_text"]).intent for r in gold_records]
    acc_full_hybrid = accuracy_score(y_full_true, y_full_hybrid)
    f1_full_hybrid = f1_score(y_full_true, y_full_hybrid, average="macro", zero_division=0)

    return {
        "untouched_test_proto_acc": round(acc_untouched, 4),
        "untouched_test_proto_f1": round(f1_untouched, 4),
        "untouched_test_hybrid_acc": round(acc_untouched_hybrid, 4),
        "untouched_test_hybrid_f1": round(f1_untouched_hybrid, 4),
        "full_fitted_proto_acc": round(acc_full_proto, 4),
        "full_fitted_proto_f1": round(f1_full_proto, 4),
        "full_fitted_hybrid_acc": round(acc_full_hybrid, 4),
        "full_fitted_hybrid_f1": round(f1_full_hybrid, 4)
    }


def main():
    print("=== Phase 12: Comprehensive Intent Diagnostic Pipeline ===", flush=True)
    df_gold, gold_records = load_gold_records()
    print(f"Loaded {len(gold_records)} Human Gold benchmark cases.", flush=True)

    hybrid_clf, embedder, rule_clf, proto_clf, gold_dev_recs, gold_test_recs = build_pipeline_classifier(gold_records)
    eval_results = run_baseline_evaluation(hybrid_clf, gold_records)
    
    df_err = pd.DataFrame(eval_results)
    
    # Per-Class Analysis
    df_per_class = analyze_per_class_performance(eval_results)
    
    # Confusion Matrix & Pairs
    cm, df_pairs = analyze_confusion_matrix(eval_results)
    
    # Confidence Buckets
    df_conf = analyze_confidence_buckets(eval_results)
    
    # Hierarchical Routing Analysis
    hier_res = run_hierarchical_experiment(eval_results)
    
    # Prototype Leakage & Benchmark Analysis
    proto_res = run_prototype_leakage_analysis(embedder, gold_records, gold_dev_recs, gold_test_recs)

    # Save Tables
    tables_dir = Path(root_dir) / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    cm.to_csv(tables_dir / "phase_12_intent_confusion_matrix.csv")
    df_err.to_csv(tables_dir / "phase_12_intent_error_analysis.csv", index=False)
    df_pairs.to_csv(tables_dir / "phase_12_intent_confusion_pairs.csv", index=False)
    
    print("\n==================================================", flush=True)
    print("PHASE 12 INTENT DIAGNOSTIC RESULTS", flush=True)
    print("==================================================", flush=True)
    
    y_true = [r["gold_intent"] for r in eval_results]
    y_pred = [r["predicted_intent"] for r in eval_results]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print(f"\n1. Current E2E Baseline Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}", flush=True)
    
    print("\n2. Top 5 Confusion Pairs:", flush=True)
    for idx, row in df_pairs.head(5).iterrows():
        print(f"   {idx+1}. {row['gold_intent']} -> {row['predicted_intent']} ({row['error_count']} cases, {row['percentage_of_all_errors']:.1f}% of errors) [{'Cross-Parent' if row['is_cross_parent'] else 'Intra-Parent'}]", flush=True)

    print("\n3. Worst 5 Intents by Recall:", flush=True)
    df_worst_rec = df_per_class.sort_values(by="recall", ascending=True).head(5)
    for idx, row in df_worst_rec.iterrows():
        print(f"   - {row['intent']}: Recall = {row['recall']*100:.1f}%, Precision = {row['precision']*100:.1f}%, F1 = {row['f1_score']:.4f} (Support: {row['support']})", flush=True)

    print("\n4. Worst 5 Intents by Precision (Over-predicted):", flush=True)
    df_worst_prec = df_per_class.sort_values(by="precision", ascending=True).head(5)
    for idx, row in df_worst_prec.iterrows():
        print(f"   - {row['intent']}: Precision = {row['precision']*100:.1f}%, Recall = {row['recall']*100:.1f}%, F1 = {row['f1_score']:.4f} (Support: {row['support']})", flush=True)

    print("\n5. Confidence Calibration Buckets:", flush=True)
    for idx, row in df_conf.iterrows():
        print(f"   - Bucket [{row['confidence_bucket']}]: {row['count']} cases ({row['percentage_of_total']}%) | Accuracy = {row['accuracy']*100:.1f}%", flush=True)

    high_conf_wrong = [r for r in eval_results if r["confidence"] >= 0.70 and not r["is_correct"]]
    low_conf_correct = [r for r in eval_results if r["confidence"] < 0.60 and r["is_correct"]]
    print(f"\n   - High-Confidence Incorrect Cases (conf >= 0.70): {len(high_conf_wrong)} cases ({len(high_conf_wrong)/len(eval_results)*100:.1f}%)", flush=True)
    print(f"   - Low-Confidence Correct Cases (conf < 0.60): {len(low_conf_correct)} cases ({len(low_conf_correct)/len(eval_results)*100:.1f}%)", flush=True)

    print("\n6. Hierarchical Analysis (Parent Categories):", flush=True)
    print(f"   - Parent Domain Accuracy: {hier_res['parent_accuracy']*100:.2f}% (Macro F1: {hier_res['parent_macro_f1']:.4f})", flush=True)
    print(f"   - Correct Parent Domain Cases: {hier_res['correct_parent_count']}/200 ({hier_res['correct_parent_count']/200*100:.1f}%)", flush=True)
    print(f"   - Cross-Parent Errors: {hier_res['cross_parent_error_count']}/200 ({hier_res['cross_parent_error_count']/200*100:.1f}%)", flush=True)
    print(f"   - Within-Parent Child Accuracy (when parent correct): {hier_res['within_parent_child_accuracy']*100:.2f}%", flush=True)

    print("\n7. Prototype Analysis & Data Split Diagnostics:", flush=True)
    print(f"   - Untouched Test Set (N=100) - Proto Only Acc: {proto_res['untouched_test_proto_acc']*100:.2f}% (Macro F1: {proto_res['untouched_test_proto_f1']:.4f})", flush=True)
    print(f"   - Untouched Test Set (N=100) - Hybrid Rule+Proto Acc: {proto_res['untouched_test_hybrid_acc']*100:.2f}% (Macro F1: {proto_res['untouched_test_hybrid_f1']:.4f})", flush=True)
    print(f"   - Full Gold Fitted (N=200, fit on 200) - Proto Only Acc: {proto_res['full_fitted_proto_acc']*100:.2f}% (Macro F1: {proto_res['full_fitted_proto_f1']:.4f})", flush=True)
    print(f"   - Full Gold Fitted (N=200, fit on 200) - Hybrid Rule+Proto Acc: {proto_res['full_fitted_hybrid_acc']*100:.2f}% (Macro F1: {proto_res['full_fitted_hybrid_f1']:.4f})", flush=True)

    print("\nDiagnostic tables written to reports/tables/.", flush=True)


if __name__ == "__main__":
    main()
