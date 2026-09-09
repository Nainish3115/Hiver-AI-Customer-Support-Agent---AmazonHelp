"""
Golden taxonomy validation dataset builder.
Selects 200 diverse conversations from the 800-example review sample.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


def build_taxonomy_validation_sample(
    sample_parquet_path: str,
    output_dir: str = "data/golden",
    target_count: int = 200,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select 200 diverse conversations from 800 review sample for human taxonomy validation.
    """
    path = Path(sample_parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Review sample parquet not found at {sample_parquet_path}")

    df_sample = pd.read_parquet(path)
    np.random.seed(random_seed)

    # Stratified selection across proposed_intent
    if 'discovered_intent' in df_sample.columns:
        df_sample['proposed_intent'] = df_sample['discovered_intent']

    # Stratify by proposed_intent and depth
    groups = df_sample.groupby('proposed_intent')
    selected_rows = []

    # Ensure at least 5-15 examples per intent category
    per_intent_target = max(5, target_count // max(1, len(groups)))

    for intent_name, group in groups:
        n = min(len(group), per_intent_target)
        sampled_group = group.sample(n=n, random_state=random_seed)
        selected_rows.append(sampled_group)

    df_selected = pd.concat(selected_rows, ignore_index=True)

    # Fill up to target_count if needed
    if len(df_selected) < target_count and len(df_sample) > len(df_selected):
        remaining = df_sample[~df_sample['conversation_id'].isin(df_selected['conversation_id'])]
        needed = min(target_count - len(df_selected), len(remaining))
        df_top_up = remaining.sample(n=needed, random_state=random_seed)
        df_selected = pd.concat([df_selected, df_top_up], ignore_index=True)

    df_selected = df_selected.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Format fields for human validation dataset
    golden_rows = []
    for _, row in df_selected.iterrows():
        # Format full conversation text from tweets_json if available
        full_text = ""
        if 'tweets_json' in row and pd.notnull(row['tweets_json']):
            try:
                tw_list = json.loads(row['tweets_json'])
                full_text = "\n".join([f"[{t['author_id']}]: {t['text']}" for t in tw_list])
            except Exception:
                full_text = str(row.get('first_customer_text', ''))
        else:
            full_text = str(row.get('first_customer_text', ''))

        has_res = bool(row.get('contains_both', True))

        golden_rows.append({
            "conversation_id": int(row['conversation_id']),
            "conversation_text": full_text,
            "proposed_intent": str(row.get('proposed_intent', 'general_inquiry_other')),
            "annotation_intent": None,
            "annotation_confidence": None,
            "is_multi_intent": False,
            "should_escalate": False,
            "has_resolution": has_res,
            "annotation_notes": ""
        })

    df_golden = pd.DataFrame(golden_rows)

    # Ensure output directory exists
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    out_parquet = out_p / "taxonomy_validation_sample.parquet"
    out_csv = out_p / "taxonomy_validation_sample.csv"

    df_golden.to_parquet(out_parquet, index=False)
    df_golden.to_csv(out_csv, index=False)

    return df_golden, df_selected
