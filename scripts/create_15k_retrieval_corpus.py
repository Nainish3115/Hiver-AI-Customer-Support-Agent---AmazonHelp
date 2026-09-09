"""
CLI script to create a deterministic, stratified, deduplicated 15,000-document
AmazonHelp retrieval corpus subset from the 81,186 canonical document corpus.

Usage:
    python scripts/create_15k_retrieval_corpus.py
"""
import sys
import os
import json
import random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def main():
    print("=== Step 1: Creating Deterministic 15K Stratified RAG Corpus ===", flush=True)
    seed = 42
    random.seed(seed)

    input_jsonl = Path("data/processed/amazonhelp_rag_documents.jsonl")
    output_jsonl = Path("data/processed/amazonhelp_rag_documents_15k.jsonl")

    if not input_jsonl.exists():
        print(f"Error: Source corpus {input_jsonl} not found.", flush=True)
        return

    # Load and deduplicate
    documents = []
    seen_texts = set()
    dup_count = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            txt = doc.get("document_text", "")
            if txt in seen_texts:
                dup_count += 1
            else:
                seen_texts.add(txt)
                documents.append(doc)

    print(f"Loaded {len(documents):,} unique documents ({dup_count} exact text duplicates removed).", flush=True)

    # Group by intent
    intent_groups = defaultdict(list)
    for doc in documents:
        intent = doc.get("metadata", {}).get("intent") or doc.get("intent") or "general_inquiry_other"
        intent_groups[intent].append(doc)

    total_unique = len(documents)
    target_count = 15000

    print(f"Stratifying across {len(intent_groups)} intent classes for target size {target_count:,}...", flush=True)

    # Calculate proportional allocation
    sampled_docs = []
    allocation = {}
    remaining_budget = target_count

    # First pass: proportional allocation with minimum floor of 10
    sorted_intents = sorted(list(intent_groups.keys()))
    for intent in sorted_intents:
        group = intent_groups[intent]
        prop = len(group) / total_unique
        alloc = max(10, int(round(prop * target_count)))
        alloc = min(alloc, len(group))
        allocation[intent] = alloc

    # Adjust sum to exactly 15,000
    alloc_sum = sum(allocation.values())
    diff = target_count - alloc_sum

    if diff != 0:
        # Sort intents by group size descending to distribute difference
        by_size = sorted(sorted_intents, key=lambda i: len(intent_groups[i]), reverse=True)
        idx = 0
        while diff != 0:
            target_i = by_size[idx % len(by_size)]
            if diff > 0 and allocation[target_i] < len(intent_groups[target_i]):
                allocation[target_i] += 1
                diff -= 1
            elif diff < 0 and allocation[target_i] > 10:
                allocation[target_i] -= 1
                diff += 1
            idx += 1

    # Perform deterministic sampling
    print("\nIntent Distribution in 15K Headline Corpus:", flush=True)
    print(f"{'Intent Category':<35} | {'Source Count':<12} | {'15K Sampled':<12} | {'Proportion %':<12}", flush=True)
    print("-" * 78, flush=True)

    for intent in sorted_intents:
        group = intent_groups[intent]
        n_sample = allocation[intent]
        # Sort group by case_id for deterministic shuffle
        group_sorted = sorted(group, key=lambda d: d["case_id"])
        # Deterministic sample using seed
        rng = random.Random(seed + hash(intent) % 1000000)
        sampled = rng.sample(group_sorted, n_sample)
        sampled_docs.extend(sampled)

        pct = (n_sample / target_count) * 100.0
        print(f"{intent:<35} | {len(group):<12,} | {n_sample:<12,} | {pct:.2f}%", flush=True)

    # Sort final sampled documents deterministically by case_id
    sampled_docs = sorted(sampled_docs, key=lambda d: d["case_id"])

    # Output 15K JSONL
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for doc in sampled_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully created deterministic 15K corpus: {output_jsonl} ({len(sampled_docs):,} documents)", flush=True)


if __name__ == "__main__":
    main()
