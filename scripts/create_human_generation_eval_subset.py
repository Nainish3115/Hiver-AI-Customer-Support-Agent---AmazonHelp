"""
Script to create a 50-case human generation evaluation dataset template and annotation guide.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, root_dir)
sys.path.insert(0, str(Path(root_dir) / "src"))

from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore
from hiver_agent.retrieval.retriever import FAISSRetriever
from hiver_agent.classification.rule_classifier import RuleBasedIntentClassifier
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier
from scripts.run_phase_5_6_experiments import HybridRuleEmbeddingClassifier, split_gold_dataset_50_50
from hiver_agent.generation.generator import MockLLMGenerator
from hiver_agent.pipeline import CustomerSupportPipeline


def main():
    print("=== Step 13: Creating 50-Case Human Evaluation Dataset Template ===", flush=True)

    gold_csv = Path("data/golden/amazonhelp_gold_200_annotation.csv")
    if not gold_csv.exists():
        raise FileNotFoundError(f"Gold benchmark dataset not found at {gold_csv}")

    df_gold = pd.read_csv(gold_csv, encoding="utf-8")
    gold_records = df_gold.to_dict(orient="records")
    for r in gold_records:
        r["customer_text"] = str(r["customer_text"])
        r["intent"] = str(r["final_human_intent"])

    # Split to get 100 Dev and 100 Test, then sample 50 from Test
    gold_dev_recs, gold_test_recs = split_gold_dataset_50_50(gold_records, seed=42)

    df_test = pd.DataFrame(gold_test_recs)
    # Stratified 50-case sample
    df_sample = df_test.groupby("intent", group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(1, int(round(50 * len(x) / len(df_test))))), random_state=42),
        include_groups=False
    ).head(50)

    # Load 15K Benchmark FAISS Store & Embedder
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=128)
    faiss_dir = Path("data/processed/faiss_index_15k")

    if not faiss_dir.exists():
        print(f"15K FAISS index not found at {faiss_dir}. Initializing in-memory vector store...")
        vector_store = FAISSVectorStore(dimension=embedder.dimension)
        # Load docs and add
        jsonl_path = Path("data/processed/amazonhelp_rag_documents_15k.jsonl")
        if jsonl_path.exists():
            import json
            docs = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    docs.append(json.loads(line))
                embs = embedder.embed_texts([d.get("document_text", d.get("customer_text", "")) for d in docs])
                vector_store.add_documents(docs, embs)
                # Save FAISS index so it doesn't need to be recomputed
                vector_store.save(str(faiss_dir))
    else:
        vector_store = FAISSVectorStore.load(str(faiss_dir))

    base_retriever = FAISSRetriever(embedder, vector_store)

    # Fit Intent Classifier on Gold Dev Set
    dev_texts = [r["customer_text"] for r in gold_dev_recs]
    dev_labels = [r["intent"] for r in gold_dev_recs]

    rule_clf = RuleBasedIntentClassifier()
    proto_clf = EmbeddingPrototypeClassifier(embedder)
    proto_clf.fit(dev_texts, dev_labels)
    hybrid_clf = HybridRuleEmbeddingClassifier(rule_clf, proto_clf)

    generator = MockLLMGenerator()

    pipeline = CustomerSupportPipeline(
        classifier=hybrid_clf,
        base_retriever=base_retriever,
        generator=generator,
        confidence_threshold=0.60,
        top_k_retrieve=10,
        top_k_evidence=3
    )

    eval_rows = []
    for _, row in df_sample.iterrows():
        cid = str(row["case_id"])
        txt = str(row["customer_text"])

        res = pipeline.process_message(txt, case_id=cid, exclude_case_id=cid)

        eval_rows.append({
            "case_id": cid,
            "customer_message": txt,
            "predicted_intent": res.predicted_intent,
            "intent_confidence": round(res.intent_confidence, 4),
            "retrieval_mode": res.retrieval_mode,
            "retrieved_case_ids": ",".join(res.retrieved_case_ids),
            "generated_response": res.draft_response,
            "human_groundedness": "",
            "human_helpfulness": "",
            "human_correctness": "",
            "human_hallucination": "",
            "human_notes": ""
        })

    df_eval_out = pd.DataFrame(eval_rows)
    out_csv = Path("data/golden/amazonhelp_generation_human_eval_50.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_eval_out.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"Successfully saved 50-case human evaluation template to: {out_csv}", flush=True)

    # Generate Human Annotation Guide
    guide_md = f"""# Phase 6: Human Generation Evaluation Annotation Guide

> **Dataset Path**: [`data/golden/amazonhelp_generation_human_eval_50.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_generation_human_eval_50.csv)  
> **Total Cases**: **50 Stratified Cases**  
> **Status**: **READY FOR HUMAN ANNOTATION**  

---

## 1. Overview & Guidelines

This document provides standardized scoring rules for human annotators evaluating AI-generated customer support responses for `AmazonHelp`.

Annotators should review each row in `amazonhelp_generation_human_eval_50.csv` and populate the blank rating columns on a **1–5 scale**.

---

## 2. Evaluation Rating Dimensions (1–5 Scale)

| Rating Column | Description | 1 (Worst) | 3 (Acceptable) | 5 (Best) |
| :--- | :--- | :--- | :--- | :--- |
| `human_groundedness` | Response strictly supported by historical evidence without invented facts. | Fabricated policies, refunds, or dates. | Minor extrapolations, mostly faithful. | 100% grounded in retrieved evidence. |
| `human_helpfulness` | Direct usefulness to customer in resolving their problem. | Unhelpful or generic refusal. | Basic helpful guidance. | Direct, clear resolution or next steps. |
| `human_correctness` | Factual accuracy relative to customer query context. | Factually incorrect guidance. | Partially accurate. | Factually flawless relative to evidence. |
| `human_hallucination` | Hallucination risk assessment. | High hallucination (unsupported claims). | Moderate risk. | Zero hallucination (completely faithful). |

---

## 3. Special Rule: `INSUFFICIENT_EVIDENCE` Handling
- If `generated_response` is `"INSUFFICIENT_EVIDENCE"`, rate `human_groundedness` as **5** (safe abstention) if historical evidence was genuinely missing or ambiguous.
- Rate `human_helpfulness` as **2-3** depending on whether asking for info was appropriate.

---

## 4. Annotation Instructions
1. Open [`data/golden/amazonhelp_generation_human_eval_50.csv`](file:///D:/hiver-support-agent/data/golden/amazonhelp_generation_human_eval_50.csv).
2. For each row, inspect `customer_message`, `retrieved_case_ids`, and `generated_response`.
3. Assign integer scores (1 to 5) in `human_groundedness`, `human_helpfulness`, `human_correctness`, `human_hallucination`.
4. Add concise notes in `human_notes` for any flagged failures.
"""
    guide_file = Path("reports/phase_6_human_eval_guide.md")
    guide_file.parent.mkdir(parents=True, exist_ok=True)
    with open(guide_file, "w", encoding="utf-8") as f:
        f.write(guide_md)

    print(f"Saved Human Annotation Guide to: {guide_file}", flush=True)


if __name__ == "__main__":
    main()
