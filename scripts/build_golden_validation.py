"""
CLI script to build taxonomy validation golden sample under data/golden/
Usage:
    python scripts/build_golden_validation.py
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from hiver_agent.data.golden_builder import build_taxonomy_validation_sample


def main():
    print("=== Step 1: Building Golden Taxonomy Validation Sample ===", flush=True)
    t0 = time.time()

    sample_parquet = Path("data/samples/amazonhelp_intent_review_sample.parquet")
    if not sample_parquet.exists():
        print("Error: Review sample parquet not found.", flush=True)
        return

    df_golden, _ = build_taxonomy_validation_sample(
        sample_parquet_path=str(sample_parquet),
        output_dir="data/golden",
        target_count=200
    )

    print(f"Saved {len(df_golden)} golden validation records to data/golden/ in {time.time()-t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
