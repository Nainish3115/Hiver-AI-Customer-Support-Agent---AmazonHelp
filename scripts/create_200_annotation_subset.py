"""
CLI script to create a deterministic, stratified 200-example human annotation dataset
from data/golden/amazonhelp_intent_annotation_ready.parquet.
"""
import sys
import random
from collections import defaultdict
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from hiver_agent.classification.dataset import normalize_text_for_grouping
from hiver_agent.classification.taxonomy import AMAZON_INTENTS


def main():
    print("=== Step 1: Creating Deterministic 200-Example Human Annotation Dataset ===", flush=True)

    input_parquet = Path("data/golden/amazonhelp_intent_annotation_ready.parquet")
    output_parquet = Path("data/golden/amazonhelp_gold_200_annotation.parquet")
    output_csv = Path("data/golden/amazonhelp_gold_200_annotation.csv")

    if not input_parquet.exists():
        print(f"Error: {input_parquet} not found.", flush=True)
        return

    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df):,} candidate rows from {input_parquet.name}")

    seed = 42
    random.seed(seed)

    # Deduplicate exact/near-duplicate texts
    seen_norm_texts = set()
    unique_rows = []

    for _, r in df.iterrows():
        txt = str(r["customer_text"])
        norm = normalize_text_for_grouping(txt)
        if norm in seen_norm_texts:
            continue
        seen_norm_texts.add(norm)
        unique_rows.append(r.to_dict())

    print(f"Unique non-duplicate candidate cases: {len(unique_rows):,}")

    # Group by proposed_intent
    intent_groups = defaultdict(list)
    for r in unique_rows:
        intent = str(r.get("proposed_intent", "general_inquiry_other"))
        intent_groups[intent].append(r)

    # Proportional allocation targeting 200 samples
    target_total = 200
    allocation = {}
    total_unique = len(unique_rows)

    sorted_intents = sorted(list(intent_groups.keys()))
    for intent in sorted_intents:
        grp = intent_groups[intent]
        prop = len(grp) / total_unique
        alloc = max(8, int(round(prop * target_total)))
        alloc = min(alloc, len(grp))
        allocation[intent] = alloc

    # Adjust sum to exactly 200
    alloc_sum = sum(allocation.values())
    diff = target_total - alloc_sum

    if diff != 0:
        by_size = sorted(sorted_intents, key=lambda i: len(intent_groups[i]), reverse=True)
        idx = 0
        while diff != 0:
            target_i = by_size[idx % len(by_size)]
            if diff > 0 and allocation[target_i] < len(intent_groups[target_i]):
                allocation[target_i] += 1
                diff -= 1
            elif diff < 0 and allocation[target_i] > 8:
                allocation[target_i] -= 1
                diff += 1
            idx += 1

    # Perform deterministic sampling per intent
    selected_rows = []
    print("\nTarget Intent Allocation for 200 Golden Dataset:")
    print(f"{'Intent ID':<40} | {'Candidate Count':<16} | {'Sampled (N=200)':<16} | {'Proportion %':<12}")
    print("-" * 90)

    for intent in sorted_intents:
        grp = intent_groups[intent]
        n_sample = allocation[intent]
        # Sort group by case_id for deterministic shuffle
        grp_sorted = sorted(grp, key=lambda d: str(d["case_id"]))
        rng = random.Random(seed + hash(intent) % 1000000)
        sampled = rng.sample(grp_sorted, n_sample)
        selected_rows.extend(sampled)

        pct = (n_sample / target_total) * 100.0
        print(f"{intent:<40} | {len(grp):<16} | {n_sample:<16} | {pct:.2f}%")

    # Sort final 200 rows deterministically by case_id
    selected_rows = sorted(selected_rows, key=lambda d: str(d["case_id"]))
    assert len(selected_rows) == 200, f"Expected 200 samples, got {len(selected_rows)}"

    # Format final schema
    final_data = []
    for r in selected_rows:
        final_data.append({
            "case_id": str(r["case_id"]),
            "customer_text": str(r["customer_text"]),
            "suggested_intent": str(r.get("proposed_intent", "general_inquiry_other")),
            "final_human_intent": None,  # MUST BE NULL
            "annotator_notes": ""        # MUST BE EMPTY
        })

    df_200 = pd.DataFrame(final_data)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_200.to_parquet(output_parquet, index=False)
    df_200.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\nSuccessfully created 200-example annotation dataset:\n- {output_parquet}\n- {output_csv}", flush=True)


if __name__ == "__main__":
    main()
