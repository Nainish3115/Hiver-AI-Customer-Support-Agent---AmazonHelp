"""
Interactive CLI Annotation Tool for Human Gold Labeling of 200 AmazonHelp Cases.

Usage:
    python scripts/annotate_amazonhelp_gold.py [--input-csv data/golden/amazonhelp_gold_200_annotation.csv]
"""
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from hiver_agent.classification.taxonomy import AMAZON_INTENTS


def parse_args():
    parser = argparse.ArgumentParser(description="Human Gold Annotation Tool for AmazonHelp Intents.")
    parser.add_argument("--input-csv", type=str, default="data/golden/amazonhelp_gold_200_annotation.csv", help="Input annotation CSV file.")
    parser.add_argument("--re-edit", action="store_true", help="Re-edit previously annotated cases.")
    return parser.parse_args()


def display_intent_menu():
    print("\nAvailable 17 AmazonHelp Intent Categories:")
    print("=" * 60)
    for idx, intent in enumerate(AMAZON_INTENTS, 1):
        print(f"  {idx:2d}. {intent}")
    print("=" * 60)


def save_annotation_progress(df: pd.DataFrame, csv_path: Path, parquet_path: Path):
    """
    Safely save current dataframe progress to CSV and Parquet formats.
    """
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_parquet(parquet_path, index=False)


def main():
    args = parse_args()
    csv_path = Path(args.input_csv)
    parquet_path = csv_path.with_suffix(".parquet")

    if not csv_path.exists():
        print(f"Error: Annotation file {csv_path} not found. Run scripts/create_200_annotation_subset.py first.", flush=True)
        return

    df = pd.read_csv(csv_path, dtype=str)
    df.fillna("", inplace=True)

    total_cases = len(df)
    annotated_cnt = sum(1 for v in df["final_human_intent"] if str(v).strip() != "")

    print("\n" + "=" * 70)
    print(" Hiver AmazonHelp Human Gold Intent Annotation Interface")
    print("=" * 70)
    print(f"Loaded: {csv_path} ({total_cases} total cases)")
    print(f"Current Progress: {annotated_cnt}/{total_cases} completed ({(annotated_cnt/total_cases)*100:.1f}%)")
    print("Commands: Enter number 1-17 | Type intent name | [s]kip | [q]uit & save")
    print("=" * 70)

    for idx, row in df.iterrows():
        case_id = str(row["case_id"])
        cust_text = str(row["customer_text"])
        sugg_intent = str(row["suggested_intent"])
        curr_human = str(row["final_human_intent"]).strip()

        # Skip if already annotated unless --re-edit flag is active
        if curr_human != "" and not args.re-edit:
            continue

        annotated_cnt = sum(1 for v in df["final_human_intent"] if str(v).strip() != "")

        print(f"\n--- Case {idx+1}/{total_cases} [{case_id}] (Progress: {annotated_cnt}/{total_cases}) ---")
        print(f"CUSTOMER TEXT:\n\"{cust_text}\"")
        print(f"\n[Reference Suggested Intent]: {sugg_intent}")
        if curr_human:
            print(f"[Current Human Label]: {curr_human}")

        display_intent_menu()

        while True:
            try:
                choice = input("\nSelect Intent (1-17, s to skip, q to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nSession interrupted. Saving progress cleanly...")
                save_annotation_progress(df, csv_path, parquet_path)
                print(f"Saved {annotated_cnt}/{total_cases} annotations to {csv_path}")
                return

            if choice.lower() in ["q", "quit", "exit"]:
                save_annotation_progress(df, csv_path, parquet_path)
                print(f"\nProgress saved cleanly ({annotated_cnt}/{total_cases} cases). Goodbye!")
                return

            if choice.lower() in ["s", "skip"]:
                print("Skipped.")
                break

            selected_intent = None
            if choice.isdigit():
                num = int(choice)
                if 1 <= num <= len(AMAZON_INTENTS):
                    selected_intent = AMAZON_INTENTS[num - 1]
            elif choice in AMAZON_INTENTS:
                selected_intent = choice

            if selected_intent is not None:
                notes = input("Optional Annotator Notes (Press Enter to skip): ").strip()
                df.at[idx, "final_human_intent"] = selected_intent
                df.at[idx, "annotator_notes"] = notes

                # Immediately save progress after each decision
                save_annotation_progress(df, csv_path, parquet_path)
                print(f"✓ Saved annotation: '{selected_intent}' for case {case_id}")
                break
            else:
                print("Invalid input! Please enter a number between 1 and 17, or type an exact intent string.")

    annotated_cnt = sum(1 for v in df["final_human_intent"] if str(v).strip() != "")
    print(f"\n=== Annotation Session Complete! ({annotated_cnt}/{total_cases} completed) ===")
    save_annotation_progress(df, csv_path, parquet_path)


if __name__ == "__main__":
    main()
