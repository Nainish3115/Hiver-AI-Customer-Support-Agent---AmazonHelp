"""
Eligibility and quality filtering rules for AmazonHelp RAG candidate conversations.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def apply_amazonhelp_quality_filters(df_threads: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply explicit quality criteria to determine eligible RAG candidate conversations.
    Calculates impact before and after filtering.
    """
    total_before = len(df_threads)

    # Filter 1: Contains both customer and support messages
    c1 = df_threads['contains_both']

    # Filter 2: At least 2 total tweets (1 query + 1 support response)
    c2 = df_threads['num_tweets'] >= 2

    # Filter 3: Customer initial text has sufficient length (>= 15 chars) to form a valid query
    c3 = df_threads['first_customer_text'].astype(str).str.len() >= 15

    # Filter 4: Valid duration (0 to 7 days)
    c4 = (df_threads['duration_seconds'] >= 0) & (df_threads['duration_seconds'] <= 604800)

    # Combined eligible mask
    eligible_mask = c1 & c2 & c3 & c4

    df_threads['is_rag_eligible'] = eligible_mask
    df_eligible = df_threads[eligible_mask].copy()

    # Calculate step-by-step breakdown
    breakdown = {
        "total_conversations_before": total_before,
        "failed_both_participants": int((~c1).sum()),
        "failed_min_length_2": int((~c2).sum()),
        "failed_customer_text_length_15": int((~c3).sum()),
        "failed_valid_duration": int((~c4).sum()),
        "total_eligible_after": len(df_eligible),
        "retention_rate_pct": round((len(df_eligible) / total_before) * 100, 2) if total_before > 0 else 0.0
    }

    df_threads['is_rag_eligible'] = eligible_mask

    return df_eligible, breakdown
