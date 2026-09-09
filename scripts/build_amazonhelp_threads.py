"""
CLI script to extract AmazonHelp tweets and build reconstructed threads parquet.
Usage:
    python scripts/build_amazonhelp_threads.py
"""
import sys
import os
import time
import json
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from hiver_agent.config import load_config
from hiver_agent.data.amazon_builder import extract_amazonhelp_raw_df, build_amazonhelp_threads
from hiver_agent.data.filters import apply_amazonhelp_quality_filters


def main():
    print("=== Step 1: Building AmazonHelp Conversation Threads ===", flush=True)
    t0 = time.time()
    config = load_config()

    raw_csv = config.datasets.twitter.raw_path
    df_amazon_raw = extract_amazonhelp_raw_df(raw_csv)
    df_threads, stats = build_amazonhelp_threads(df_amazon_raw)
    _, filter_breakdown = apply_amazonhelp_quality_filters(df_threads)

    # Ensure processed directory exists
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "amazonhelp_threads.parquet"

    # Save to parquet
    df_threads.to_parquet(out_parquet, index=False)
    print(f"\nSaved reconstructed threads to: {out_parquet}", flush=True)

    # Save stats summary JSON
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(tables_dir / "amazonhelp_thread_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(tables_dir / "amazonhelp_filter_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(filter_breakdown, f, indent=2)

    elapsed = time.time() - t0
    print(f"Extraction & thread building finished in {elapsed:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
