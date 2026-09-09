# Historical Customer Support RAG Corpus Analysis Report

> **Project Phase**: Phase 3 — RAG Corpus Construction & Document Representation  
> **Target Brand**: `AmazonHelp`  
> **Input Threads Dataset**: `data/processed/amazonhelp_threads.parquet`

---

## 1. RAG Corpus Summary Statistics

- **Total Eligible Input Conversations**: **81,186**
- **Cases with Valid Resolution Actions**: **81,178** (**99.99%**)
- **Cases Filtered Out (DM Redirection / Truncated / No Resolution)**: **8**
- **Unique Customer Problem Cases**: **81,126**
- **Duplicate / Near-Duplicate Queries Identified**: **60**
- **Average Support Resolution Character Length**: **123.83** characters

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
