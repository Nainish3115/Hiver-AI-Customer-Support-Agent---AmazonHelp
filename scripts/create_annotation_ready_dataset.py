"""
CLI script to audit human gold labels and generate a clean, annotation-ready dataset
file for AmazonHelp intent classification.
"""
import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from hiver_agent.classification.taxonomy import AMAZON_INTENTS


def main():
    print("=== Step 1: Human Gold Label Audit & Annotation File Generation ===", flush=True)

    golden_parquet = Path("data/golden/taxonomy_validation_sample.parquet")
    review_parquet = Path("data/samples/amazonhelp_intent_review_sample.parquet")
    output_parquet = Path("data/golden/amazonhelp_intent_annotation_ready.parquet")
    output_csv = Path("data/golden/amazonhelp_intent_annotation_ready.csv")

    rows = []

    # 1. Inspect taxonomy validation sample (200 cases)
    if golden_parquet.exists():
        df_gold = pd.read_parquet(golden_parquet)
        print(f"Loaded {len(df_gold)} cases from {golden_parquet.name}")
        for _, r in df_gold.iterrows():
            cid = int(r["conversation_id"])
            txt = str(r.get("conversation_text", ""))
            # Extract customer problem statement
            if "[cust" in txt.lower():
                prob = txt.split(":")[1].split("\n")[0] if ":" in txt else txt
            else:
                prob = txt.split("\n")[0] if "\n" in txt else txt

            prop_intent = str(r.get("proposed_intent", "general_inquiry_other"))
            human_intent = r.get("annotation_intent")
            if pd.isna(human_intent) or human_intent is None or str(human_intent).strip() in ["", "None"]:
                human_intent = None
            else:
                human_intent = str(human_intent).strip()

            rows.append({
                "case_id": f"RAG-AMAZON-{cid}",
                "conversation_id": cid,
                "customer_text": prob,
                "proposed_intent": prop_intent,
                "final_human_intent": human_intent,
                "annotator_notes": str(r.get("annotation_notes", ""))
            })

    # 2. Inspect review sample (800 cases)
    if review_parquet.exists():
        df_rev = pd.read_parquet(review_parquet)
        print(f"Loaded {len(df_rev)} cases from {review_parquet.name}")
        seen_cids = {r["conversation_id"] for r in rows}
        for _, r in df_rev.iterrows():
            cid = int(r["conversation_id"])
            if cid in seen_cids:
                continue
            txt = str(r.get("first_customer_text", ""))
            prop_intent = str(r.get("discovered_intent", "general_inquiry_other"))
            rows.append({
                "case_id": f"RAG-AMAZON-{cid}",
                "conversation_id": cid,
                "customer_text": txt,
                "proposed_intent": prop_intent,
                "final_human_intent": None,
                "annotator_notes": ""
            })

    df_out = pd.DataFrame(rows)

    human_verified_count = df_out["final_human_intent"].dropna().count()
    print(f"\nAudit Summary:")
    print(f"- Total Candidate Examples: {len(df_out):,}")
    print(f"- Explicit Human-Verified Labels: {human_verified_count}")
    print(f"- Heuristic / Proposed Labels: {len(df_out) - human_verified_count:,}")

    if human_verified_count == 0:
        print("\n[CRITICAL FINDING]: Human gold labels are not yet sufficient for final AmazonHelp classifier evaluation.")
        print("Heuristic rule-discovered labels will be used strictly as DEVELOPMENT PROXIES during modeling experiments.")

    # Write annotation-ready files
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_parquet, index=False)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Saved annotation-ready dataset to:\n- {output_parquet}\n- {output_csv}", flush=True)


if __name__ == "__main__":
    main()
