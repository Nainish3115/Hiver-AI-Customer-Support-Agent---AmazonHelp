"""
CLI script to build historical AmazonHelp RAG corpus and JSONL documents.
Usage:
    python scripts/build_rag_corpus.py
"""
import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from hiver_agent.data.rag_builder import build_amazonhelp_rag_corpus


def main():
    print("=== Step 2: Building Historical AmazonHelp RAG Corpus ===", flush=True)
    t0 = time.time()

    threads_parquet = Path("data/processed/amazonhelp_threads.parquet")
    if not threads_parquet.exists():
        print("Error: Reconstructed threads parquet not found.", flush=True)
        return

    df_rag, docs_jsonl, stats = build_amazonhelp_rag_corpus(
        threads_parquet_path=str(threads_parquet),
        output_dir="data/processed"
    )

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "rag_corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Generate RAG Analysis Report
    reports_dir = Path("reports")
    report_file = reports_dir / "rag_corpus_analysis.md"

    rag_md = f"""# Historical Customer Support RAG Corpus Analysis Report

> **Project Phase**: Phase 3 — RAG Corpus Construction & Document Representation  
> **Target Brand**: `AmazonHelp`  
> **Input Threads Dataset**: `data/processed/amazonhelp_threads.parquet`

---

## 1. RAG Corpus Summary Statistics

- **Total Eligible Input Conversations**: **{stats['total_eligible_conversations']:,}**
- **Cases with Valid Resolution Actions**: **{stats['conversations_with_resolution']:,}** (**{stats['resolution_rate_pct']}%**)
- **Cases Filtered Out (DM Redirection / Truncated / No Resolution)**: **{stats['conversations_without_resolution']:,}**
- **Unique Customer Problem Cases**: **{stats['unique_cases_count']:,}**
- **Duplicate / Near-Duplicate Queries Identified**: **{stats['duplicate_problem_cases']:,}**
- **Average Support Resolution Character Length**: **{stats['avg_resolution_char_length']}** characters

---

## 2. Retrieval Document Format (`data/processed/amazonhelp_rag_documents.jsonl`)

Each RAG document is rendered into a clean, deterministic text representation optimized for vector embedding and context augmentation:

```text
INTENT: <intent_category>
CUSTOMER PROBLEM: <initial_customer_utterance>
SUPPORT INTERACTION:
[author_1]: <tweet_1>
[AmazonHelp]: <tweet_2>
RESOLUTION: <final_support_response_action>
```

---

## 3. High-Quality Resolution Corpus Storage

- **Parquet Dataset**: [`data/processed/amazonhelp_rag_corpus.parquet`](file:///d:/hiver-support-agent/data/processed/amazonhelp_rag_corpus.parquet)
- **JSONL Document Index**: [`data/processed/amazonhelp_rag_documents.jsonl`](file:///d:/hiver-support-agent/data/processed/amazonhelp_rag_documents.jsonl)
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(rag_md)

    print(f"Saved {len(df_rag)} RAG cases and {len(docs_jsonl)} documents in {time.time()-t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
