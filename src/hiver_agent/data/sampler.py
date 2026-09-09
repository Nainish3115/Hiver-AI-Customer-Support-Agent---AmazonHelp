"""
Stratified sampling engine for AmazonHelp human review sample creation.
"""
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def generate_stratified_sample(
    df_threads: pd.DataFrame,
    target_sample_size: int = 800,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate a stratified review sample of AmazonHelp conversations covering:
    - Intent categories
    - Thread depth levels
    - RAG eligibility & edge cases
    """
    if len(df_threads) == 0:
        return df_threads

    np.random.seed(random_seed)

    # Ensure intent column exists
    if 'discovered_intent' not in df_threads.columns:
        df_threads['discovered_intent'] = 'general_inquiry_other'

    # Compute sampling strata: intent + depth_bucket
    def get_depth_bucket(depth):
        if depth <= 2:
            return "short_1_2"
        elif depth <= 4:
            return "med_3_4"
        else:
            return "deep_5_plus"

    df_threads['depth_bucket'] = df_threads['depth'].apply(get_depth_bucket)
    df_threads['stratum'] = df_threads['discovered_intent'] + "__" + df_threads['depth_bucket']

    strata_counts = df_threads['stratum'].value_counts()
    num_strata = len(strata_counts)

    # Allocate per stratum
    base_per_stratum = max(1, target_sample_size // num_strata)

    sampled_dfs = []
    for stratum_name, group in df_threads.groupby('stratum'):
        n_sample = min(len(group), base_per_stratum)
        sampled_group = group.sample(n=n_sample, random_state=random_seed)
        sampled_dfs.append(sampled_group)

    df_sampled = pd.concat(sampled_dfs, ignore_index=True)

    # Top-up if total is less than target
    if len(df_sampled) < target_sample_size and len(df_threads) > len(df_sampled):
        remaining = df_threads[~df_threads['conversation_id'].isin(df_sampled['conversation_id'])]
        needed = min(target_sample_size - len(df_sampled), len(remaining))
        top_up = remaining.sample(n=needed, random_state=random_seed)
        df_sampled = pd.concat([df_sampled, top_up], ignore_index=True)

    df_sampled = df_sampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df_sampled
