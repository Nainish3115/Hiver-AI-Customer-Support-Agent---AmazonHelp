"""
CLI script to generate stratified review sample of AmazonHelp conversations.
Usage:
    python scripts/sample_amazonhelp.py
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
from hiver_agent.data.sampler import generate_stratified_sample


def main():
    print("=== Step 3: Generating Stratified AmazonHelp Review Sample ===", flush=True)
    t0 = time.time()

    parquet_path = Path("data/processed/amazonhelp_threads.parquet")
    if not parquet_path.exists():
        print("Error: amazonhelp_threads.parquet not found. Run build_amazonhelp_threads.py first.", flush=True)
        return

    df_threads = pd.read_parquet(parquet_path)
    df_sampled = generate_stratified_sample(df_threads, target_sample_size=800, random_seed=42)

    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = samples_dir / "amazonhelp_intent_review_sample.parquet"
    out_csv = samples_dir / "amazonhelp_intent_review_sample.csv"

    # Select clean display columns for review sample
    review_cols = [
        "conversation_id", "discovered_intent", "num_tweets", "depth",
        "duration_seconds", "contains_both", "is_rag_eligible",
        "first_customer_text", "participants"
    ]
    review_cols = [c for c in review_cols if c in df_sampled.columns]

    df_sampled.to_parquet(out_parquet, index=False)
    df_sampled[review_cols].to_csv(out_csv, index=False)

    print(f"Saved stratified sample ({len(df_sampled)} conversations) to:", flush=True)
    print(f"  - Parquet: {out_parquet}", flush=True)
    print(f"  - CSV:     {out_csv}", flush=True)

    elapsed = time.time() - t0
    print(f"Sampling script finished in {elapsed:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
