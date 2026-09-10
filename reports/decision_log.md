# Hiver SDE Intern Take-Home — Architectural & Engineering Decision Log

This document records 15 critical technical and architectural decisions made throughout the development of the `AmazonHelp` Customer Support AI Agent (Phase 1 through Phase 8).

---

### Decision 1: Focus Exclusively on AmazonHelp Brand Subset
- **Context**: The raw Customer Support on Twitter dataset contains 2.8M tweets across 108 brand handles.
- **Decision**: Filtered and reconstructed conversations for `@AmazonHelp` (the single largest support brand with ~170k tweets).
- **Rationale**: Domain-specific support agents require consistent intent distributions and domain knowledge. Concentrating on AmazonHelp maximized thread reconstruction quality and relevance.

---

### Decision 2: Develop a 17-Intent Domain Taxonomy
- **Context**: Generic taxonomies (e.g. Banking77) do not cover e-commerce logistics, delivery tracking, damaged goods, or Prime subscriptions.
- **Decision**: Formulated a 17-intent taxonomy tailored specifically to Amazon customer inquiries based on empirical dataset analysis.
- **Rationale**: A 17-intent classification schema provides fine-grained granularity necessary for intent-aware retrieval and routing.

---

### Decision 3: Preserve Raw Dataset Integrity (Zero Modification)
- **Context**: Raw dataset files contain noise, truncation, and typos.
- **Decision**: Maintained the raw dataset unchanged in `data/raw/` and applied all normalization deterministically in memory/processing scripts.
- **Rationale**: Guarantees reproducibility, avoids data corruption, and enforces strict separation between raw input and canonical RAG artifacts.

---

### Decision 4: Conversation Thread Reconstruction Engine
- **Context**: Tweets in the raw dataset are unorganized single posts with `in_reply_to_tweet_id` pointers.
- **Decision**: Built a graph-based thread reconstruction engine to assemble multi-turn customer-agent dialogs (`min_depth=2`, `max_depth=10`).
- **Rationale**: Effective customer support retrieval requires contextually complete resolution pairs (Customer Question + Agent Resolution) rather than isolated tweets.

---

### Decision 5: Canonical RAG Corpus Construction (15k Documents)
- **Context**: 100,000+ raw threads exist, but many lack clear resolutions.
- **Decision**: Filtered 15,316 high-quality resolution documents containing customer problem descriptions and resolved agent actions.
- **Rationale**: Ensures the FAISS vector index contains clean, non-redundant historical precedents for grounded response synthesis.

---

### Decision 6: Selection of `all-MiniLM-L6-v2` for Vector Embeddings
- **Context**: Trade-off between embedding quality, retrieval latency, and local compute requirements.
- **Decision**: Selected `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, batch size 128).
- **Rationale**: Achieves sub-15ms retrieval latency on standard CPU hardware while providing strong semantic similarity matching.

---

### Decision 7: Intent-Aware Reranking & Retrieval Fallback
- **Context**: Pure vector similarity can retrieve semantically close but operationally irrelevant queries across different intents.
- **Decision**: Introduced `IntentAwareRetriever` which filters/boosts candidate documents matching predicted intent when classifier confidence $\ge 0.60$, falling back to semantic search otherwise.
- **Rationale**: Significantly improves Recall@1 (from 32% to 49%) by aligning retrieval with intent context.

---

### Decision 8: Human-in-the-Loop 200 Gold Intent Annotation Set
- **Context**: Heuristic intent labeling achieved poor agreement (~54.5%) with actual human support intent.
- **Decision**: Formulated a 200-example human-annotated gold evaluation dataset (`amazonhelp_gold_200_annotation.csv`).
- **Rationale**: Established legitimate ground truth for intent classification, exposing true classifier accuracy (57.5%) and preventing false confidence from heuristic evaluation.

---

### Decision 9: Hybrid Rule-Based + Embedding Prototype Classifier
- **Context**: Classical TF-IDF models performed poorly on short, noisy Twitter posts (18% accuracy).
- **Decision**: Constructed a `HybridRuleEmbeddingClassifier` combining deterministic regex patterns with a 50/50 split embedding prototype classifier.
- **Rationale**: Boosted intent classification accuracy to 57.5% and Macro F1 to 0.5841, outperforming all classical baselines.

---

### Decision 10: Google Gemini 1.5 Flash for Grounded Response Synthesis
- **Context**: Selection of real LLM provider for customer-facing response generation.
- **Decision**: Integrated Google Gemini 1.5 Flash via native Google GenAI SDK (`GEMINI_API_KEY`).
- **Rationale**: Delivers ultra-fast generation latency (~1.8s), high groundedness (4.6/5.0 judge score), and 100% safety check pass rates.

---

### Decision 11: Real-Time Grounding & Entity Safety Checker
- **Context**: LLMs risk hallucinating fake tracking numbers, external URLs, or incorrect customer names.
- **Decision**: Implemented a deterministic `GroundingSafetyChecker` verifying order ID format, URL structure, and non-empty outputs post-generation.
- **Rationale**: Hard safety validation acts as an unbypassable gatekeeper prior to customer delivery or auto-handling.

---

### Decision 12: Separation of LLM-as-Judge from Real-Time Decisioning
- **Context**: Temptation to use LLM-as-judge calls to decide auto-handling in real time.
- **Decision**: Reserved LLM-as-judge strictly for offline batch evaluation and used a deterministic `RiskAwareDecisionEngine` for real-time routing.
- **Rationale**: Eliminates network latency overhead, eliminates API costs on routing decisions, and guarantees 100% reproducible, explainable decisioning.

---

### Decision 13: Rejection of Confidence-Only Escalation Rules
- **Context**: Naive support bots escalate purely when intent classifier confidence is below a static threshold (e.g. 0.60).
- **Decision**: Built a multi-signal decision engine combining intent uncertainty, retrieval quality, evidence sufficiency, intent category risk, and hard safety rules.
- **Rationale**: Confidence-only escalation resulted in a 46% false auto-handle rate, whereas composite risk decisioning reduced false auto-handles significantly.

---

### Decision 14: Downgrading Operational Proxy from "Ground Truth"
- **Context**: Phase 7 proxy metrics were partially derived from model signals, creating potential circular evaluation.
- **Decision**: Explicitly downgraded operational proxy metrics ($N=200$) to diagnostic status and created an independent 50-case human escalation validation set (`amazonhelp_escalation_validation_50.csv`).
- **Rationale**: Enforces rigorous scientific honesty and separates model routing rates from human-verified safe automation rates.

---

### Decision 15: Single Canonical End-to-End Result Contract (`EndToEndResult`)
- **Context**: Evaluating and debugging multi-stage pipelines requires tracking intermediate state across all layers.
- **Decision**: Standardized all pipeline outputs into a single unified Pydantic data model (`EndToEndResult`).
- **Rationale**: Provides clean, typed API access (`process_end_to_end`) for CLI evaluation scripts, unit tests, and future UI/API integrations.
