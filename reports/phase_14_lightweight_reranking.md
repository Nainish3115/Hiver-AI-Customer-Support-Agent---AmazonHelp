# Phase 14 Report: Lightweight Domain-Aware Retrieval Reranking Experiment

## 1. Objective
The objective of Phase 14 (Day 6) is to investigate whether a lightweight, deterministic, domain-aware scoring layer (`LightweightReranker`) can improve evidence selection for the baseline **Hybrid + Intent** retrieval pipeline without re-enabling an external neural cross-encoder.

The target benchmark consists of $N=200$ human-gold AmazonHelp customer support queries evaluated over the 15,316-document benchmark corpus.

## 2. Current Retrieval Baseline
The authoritative, approved retrieval configuration established in Phase 9 is **Hybrid + Intent**:
- **Dense Retriever**: `all-MiniLM-L6-v2` (384 dimensions, cosine similarity via FAISS)
- **Lexical Retriever**: BM25 (`rank_bm25`, $k_1=1.5, b=0.75$)
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF with constant $k=60$)
- **Intent-Aware Boost**: Multiplicative/additive intent compatibility boost ($+0.15$ score boost for candidate matching predicted intent when intent classification confidence $\ge 0.60$)

### Baseline Retrieval Proxy Performance ($N=200$ Queries, 15,316 Documents):
- **Dense Baseline**: R@1 = 28.0%, R@3 = 40.5%, R@5 = 49.0%, R@10 = 59.5%, MRR = 0.3666
- **BM25 Baseline**: R@1 = 28.5%, R@3 = 46.0%, R@5 = 55.5%, R@10 = 71.5%, MRR = 0.4048
- **Hybrid RRF**: R@1 = 35.0%, R@3 = 50.0%, R@5 = 58.5%, R@10 = 69.5%, MRR = 0.4508
- **Hybrid + Intent (Current Default Baseline)**: R@1 = 53.5%, R@3 = 55.5%, R@5 = 58.0%, R@10 = 65.5%, MRR = 0.5590

## 3. Cross-Encoder Rejection Summary (Phase 10 Context)
Phase 10 evaluated adding `cross-encoder/ms-marco-MiniLM-L-6-v2` downstream of hybrid retrieval.
- **Hybrid + Reranker + Intent Performance**: R@1 = 35.5%, R@3 = 53.5%, R@5 = 59.0%, R@10 = 69.0%, MRR = 0.4602
- **Key Finding**: MS-MARCO cross-encoder severely degraded top-1 rank (R@1 dropped from 53.5% to 35.5%) because generic passage-relevance pre-training penalized domain-specific AmazonHelp support thread structures.
- **Decision**: MS-MARCO cross-encoder remains **REJECTED** and disabled.

## 4. Retrieval Error Analysis (200 Benchmark Queries)
Analysis of ranking performance across all 200 benchmark queries (`reports/tables/phase_14_retrieval_error_analysis.csv`) revealed:
1. **Complementary Retrievers**: Dense semantic search retrieves conceptual matches when customer phrasing varies, while BM25 captures specific order numbers, product names, or acronyms.
2. **Intent Filter Impact**: Intent-aware filtering successfully elevates intent-aligned documents in 65% of queries. However, when intent prediction is wrong (~42.5% of queries due to E2E classifier accuracy), intent boosting elevates incorrect intents or fails to elevate the gold intent document.
3. **Top-1 vs Top-10 Gap**: For ~13% of cases where Top-1 was wrong, a gold intent-aligned candidate existed within Top-10 (R@10 = 65.5% vs R@1 = 53.5%).

## 5. New Lightweight Ranking Signals
To re-rank top-30 candidates deterministically on CPU, `LightweightReranker` computes:
- **Normalized RRF Score ($S_{\text{rrf}}$)**: Min-max normalized reciprocal rank fusion score across candidate pool $\in [0, 1]$.
- **Normalized Dense Similarity ($S_{\text{dense}}$)**: Min-max normalized cosine vector similarity $\in [0, 1]$.
- **Lexical Overlap Ratio ($S_{\text{lexical}}$)**: Token overlap ratio between query tokens (excluding English stopwords) and candidate document text $\in [0, 1]$.
- **Predicted Intent Signal ($S_{\text{intent}}$)**: $1.0$ if predicted intent confidence $\ge 0.60$ and candidate intent equals predicted intent, else $0.0$.

## 6. Controlled Ablation Methodology
Four configurations were evaluated under strict experimental control:
- **Corpus**: 15,316 AmazonHelp document pairs
- **Queries**: $N=200$ human-gold benchmark queries
- **Seed**: 42 (deterministic 50/50 dev split for intent classifier calibration)
- **Leakage Controls**: Retrieval uses ONLY PREDICTED intent during inference. Gold intent is never accessed during reranking.

### Ablation Configurations:
- **Config A**: Baseline Hybrid + Intent (Phase 9 default)
- **Config B**: Hybrid + Intent + Lexical Overlap
- **Config C**: Hybrid + Intent + Normalized Dense Score
- **Config D**: Hybrid + Intent + Lexical + Dense Composite

## 7. Results Comparison

| Metric | Config A (Baseline Hybrid + Intent) | Config B (Hybrid + Intent + Lexical) | Config C (Hybrid + Intent + Dense) | Config D (Hybrid + Intent + Lexical + Dense) |
|---|---|---|---|---|
| **R@1** | **52.5%** | 45.5% | 47.0% | 47.0% |
| **R@3** | 56.0% | **59.0%** | **60.0%** | **60.0%** |
| **R@5** | 61.0% | **64.5%** | 63.0% | 63.5% |
| **R@10** | 65.0% | **69.5%** | 65.0% | 67.5% |
| **MRR** | **0.5570** | 0.5361 | 0.5333 | 0.5398 |
| **Moved Upward** | 0 | 25 | 22 | 24 |
| **Moved Downward** | 0 | 26 | 24 | 23 |

> [!WARNING]
> None of the lightweight reranking configurations improved top-1 recall (R@1) or Mean Reciprocal Rank (MRR) over the baseline **Hybrid + Intent** pipeline. R@1 degraded by 5.5% to 7.0% across all lightweight reranker variants.

## 8. Latency Assessment (Local CPU Benchmark)

| Configuration | Mean Latency (ms) | P95 Latency (ms) | Delta vs Baseline (ms) |
|---|---|---|---|
| **Config A (Baseline Hybrid + Intent)** | **249.60** | **501.71** | **+0.00** |
| **Config B (Hybrid + Intent + Lexical Overlap)** | 2039.29 | 2911.69 | +1789.69 |
| **Config C (Hybrid + Intent + Dense)** | 275.85 | 495.52 | +26.25 |
| **Config D (Hybrid + Intent + Lexical + Dense)** | 287.00 | 541.00 | +37.40 |

> [!NOTE]
> Config C and Config D add minimal CPU latency (+26ms to +37ms). Config B incurs higher overhead due to full-text tokenization on long interactions. However, latency is secondary since quality metrics (R@1 and MRR) degraded.

## 9. Failure Mode Analysis
Re-ranking using linear combination of normalized scores caused two major failure patterns:
1. **RRF Rank Distortion**: RRF non-linearly suppresses low-ranked outliers ($1 / (k + r)$). Linear normalization of RRF compresses score differences between Rank 1 and Rank 5, allowing noisy token matches or dense similarity artifacts to displace true intent-aligned Rank 1 candidates.
2. **Short Token Matches**: Short queries matching non-essential keywords in support interaction text artificially boosted off-intent candidates.

## 10. Evaluation Limitations
- **Proxy Benchmark Target**: Retrieval relevance target is based on intent-aligned proxy labels across 200 AmazonHelp gold queries. It does not reflect exhaustive human pairwise relevance judgments for all 15k documents.
- **Fixed Scoring Weights**: Weights were predetermined ($W_{\text{rrf}}=0.35, W_{\text{intent}}=0.25, W_{\text{dense}}=0.20, W_{\text{lexical}}=0.20$) to prevent overfitting to the 200 benchmark queries.

## 11. Leakage Controls Audit
- **Inference Verification**: `LightweightReranker` accepts `predicted_intent` and `intent_confidence` produced by `HybridRuleEmbeddingClassifier`. Gold intent is 100% excluded from retrieval inference code.
- **Audit Status**: **PASS** (Zero gold-intent leakage).

## 12. Final Recommendation
According to the mandatory Decision Criteria:
- R@1 did NOT improve over baseline (R@1 dropped from 52.5%/53.5% to 47.0%).
- MRR did NOT improve over baseline (MRR dropped from 0.5570/0.5590 to 0.5398).

**RECOMMENDATION**: **KEEP CURRENT** baseline **Hybrid + Intent** retrieval pipeline. **REJECT** lightweight reranking for production deployment. Production retrieval configuration remains 100% UNCHANGED.
