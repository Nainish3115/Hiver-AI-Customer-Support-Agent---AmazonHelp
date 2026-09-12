"""
Phase 14 Day 6: Lightweight Reranking Benchmark Evaluation Script.

Evaluates 4 Retrieval Ablation Configurations on N=200 Human Gold Benchmark:
- Config A: Existing Hybrid + Intent Baseline
- Config B: Hybrid + Intent + Lexical Overlap
- Config C: Hybrid + Intent + Normalized Dense Score
- Config D: Hybrid + Intent + Lexical + Dense

Outputs:
- reports/tables/phase_14_retrieval_ablation.csv
- reports/tables/phase_14_latency.csv
- reports/phase_14_lightweight_reranking.md
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
from hiver_agent.retrieval.retriever import FAISSRetriever, IntentAwareRetriever, SearchResult
from hiver_agent.retrieval.bm25_retriever import BM25Retriever
from hiver_agent.retrieval.hybrid_retriever import HybridRetriever
from hiver_agent.retrieval.lightweight_reranker import LightweightReranker
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from scripts.evaluate_phase12_hierarchical_classifier import load_gold_records
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier, split_gold_dataset_50_50


def run_experiment():
    print("=== Phase 14 Day 6: Lightweight Reranking Benchmark Evaluation ===", flush=True)

    # 1. Load Gold Records (N=200)
    df_gold, gold_records = load_gold_records()
    gold_dev_recs, _ = split_gold_dataset_50_50(gold_records, seed=42)
    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]

    # 2. Setup Base Retrievers
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    faiss_dir = Path("data/processed/faiss_index_15k")
    vector_store = FAISSVectorStore.load(str(faiss_dir))
    dense_retriever = FAISSRetriever(embedder, vector_store)
    bm25_retriever = BM25Retriever.from_vector_store(vector_store)
    hybrid_rrf = HybridRetriever(dense_retriever=dense_retriever, bm25_retriever=bm25_retriever, rrf_k=60, dense_top_k=50, bm25_top_k=50)

    # 3. Fit Intent Classifier (Dev 100)
    rule_clf = RuleBasedIntentClassifier()
    proto_clf = EmbeddingPrototypeClassifier(embedder)
    proto_clf.fit(dev_texts, dev_labels)
    intent_clf = HybridRuleEmbeddingClassifier(rule_clf, proto_clf)

    # 4. Define Ablation Configurations
    reranker_b = LightweightReranker(config_name="Hybrid + Intent + Lexical", weight_rrf=0.50, weight_intent=0.30, weight_dense=0.0, weight_lexical=0.20)
    reranker_c = LightweightReranker(config_name="Hybrid + Intent + Dense", weight_rrf=0.40, weight_intent=0.30, weight_dense=0.30, weight_lexical=0.0)
    reranker_d = LightweightReranker(config_name="Hybrid + Intent + Lexical + Dense", weight_rrf=0.35, weight_intent=0.25, weight_dense=0.20, weight_lexical=0.20)

    ablation_configs = [
        ("Config A: Existing Hybrid + Intent Baseline", None),
        ("Config B: Hybrid + Intent + Lexical Overlap", reranker_b),
        ("Config C: Hybrid + Intent + Normalized Dense Score", reranker_c),
        ("Config D: Hybrid + Intent + Lexical + Dense", reranker_d)
    ]

    # Baseline R@1 rank lookup for tracking upward/downward movement
    baseline_first_ranks = []

    # Process all cases and record performance per config
    results_table = []
    latency_table = []

    for name, reranker in ablation_configs:
        ranks = []
        r1_c = r3_c = r5_c = r10_c = 0
        latencies_ms = []
        moved_up = 0
        moved_down = 0

        for idx_record, r in enumerate(gold_records):
            cid = str(r["case_id"])
            txt = str(r["customer_text"])
            gold_intent = str(r["intent"])

            t0 = time.perf_counter()

            # Predict intent
            pred = intent_clf.predict(txt)

            # Retrieve candidate pool (30 candidates)
            raw_candidates = hybrid_rrf.retrieve(txt, top_k=30)

            # Filter self case
            filtered_cands = [x for x in raw_candidates if str(x.case_id).strip() != cid]

            if reranker is None:
                # Baseline Config A: IntentAware boost over RRF
                if pred.confidence >= 0.60:
                    for cand in filtered_cands:
                        if cand.intent == pred.intent:
                            cand.score += 0.15
                    filtered_cands.sort(key=lambda x: x.score, reverse=True)
                final_candidates = filtered_cands[:10]
            else:
                # Attach raw dense similarity score to metadata for normalization
                q_vec = embedder.embed_query(txt).flatten()
                for cand in filtered_cands:
                    # Compute dense similarity directly against cand.customer_problem / doc
                    c_doc = cand.support_interaction or cand.customer_problem
                    c_vec = embedder.embed_query(c_doc[:300]).flatten()
                    cand.metadata["dense_score"] = float(np.dot(q_vec, c_vec))

                final_candidates = reranker.rerank(txt, pred.intent, pred.confidence, filtered_cands, top_k=10)

            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            # Find rank of first gold-intent matching document
            match_rank = None
            for rank_i, item in enumerate(final_candidates, start=1):
                if str(item.intent).strip() == gold_intent:
                    match_rank = rank_i
                    break

            if name.startswith("Config A"):
                baseline_first_ranks.append(match_rank)
            else:
                base_r = baseline_first_ranks[idx_record]
                if match_rank is not None and base_r is not None:
                    if match_rank < base_r:
                        moved_up += 1
                    elif match_rank > base_r:
                        moved_down += 1
                elif match_rank is not None and base_r is None:
                    moved_up += 1
                elif match_rank is None and base_r is not None:
                    moved_down += 1

            if match_rank is not None:
                ranks.append(1.0 / match_rank)
                if match_rank == 1:
                    r1_c += 1
                if match_rank <= 3:
                    r3_c += 1
                if match_rank <= 5:
                    r5_c += 1
                if match_rank <= 10:
                    r10_c += 1
            else:
                ranks.append(0.0)

        n = len(gold_records)
        r1 = round(r1_c / n, 4)
        r3 = round(r3_c / n, 4)
        r5 = round(r5_c / n, 4)
        r10 = round(r10_c / n, 4)
        mrr = round(float(np.mean(ranks)), 4)
        mean_lat = round(float(np.mean(latencies_ms)), 2)
        p95_lat = round(float(np.percentile(latencies_ms, 95)), 2)

        results_table.append({
            "Configuration": name,
            "R@1": r1,
            "R@3": r3,
            "R@5": r5,
            "R@10": r10,
            "MRR": mrr,
            "Moved_Upward": moved_up,
            "Moved_Downward": moved_down,
            "Mean_Latency_ms": mean_lat
        })

        latency_table.append({
            "Configuration": name,
            "Mean_Latency_ms": mean_lat,
            "P95_Latency_ms": p95_lat,
            "Delta_vs_Baseline_ms": round(mean_lat - results_table[0]["Mean_Latency_ms"], 2)
        })

    df_results = pd.DataFrame(results_table)
    df_latency = pd.DataFrame(latency_table)

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(tables_dir / "phase_14_retrieval_ablation.csv", index=False)
    df_latency.to_csv(tables_dir / "phase_14_latency.csv", index=False)

    print("\n--- Ablation Results ---")
    print(df_results.to_string())
    print("\n--- Latency Breakdown ---")
    print(df_latency.to_string())


if __name__ == "__main__":
    run_experiment()
