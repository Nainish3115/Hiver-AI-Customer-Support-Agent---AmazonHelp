"""
Dataset utilities for leakage-resistant dataset splitting and grouping.
"""
import re
import json
import random
import hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import pandas as pd
from pathlib import Path

from hiver_agent.classification.taxonomy import AMAZON_INTENTS


def normalize_text_for_grouping(text: str) -> str:
    """
    Normalize text by removing URLs, handles, digits, order numbers, punctuation,
    and extra whitespace to identify near-duplicate templated queries.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\d+", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_near_duplicate_groups(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Assign a unique group_id to each record based on normalized text hash
    to ensure near-duplicate templated messages stay in the same split.
    """
    text_to_group = {}
    group_counter = 0

    grouped_records = []
    for r in records:
        norm_txt = normalize_text_for_grouping(r["customer_text"])
        if not norm_txt:
            norm_txt = r["customer_text"].strip().lower()

        if norm_txt not in text_to_group:
            group_counter += 1
            text_to_group[norm_txt] = group_counter

        group_id = text_to_group[norm_txt]
        r_copy = dict(r)
        r_copy["group_id"] = group_id
        r_copy["normalized_text"] = norm_txt
        grouped_records.append(r_copy)

    stats = {
        "total_records": len(records),
        "unique_groups": group_counter
    }
    return grouped_records, stats


def split_grouped_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Perform a leakage-resistant split of grouped records into Train, Val, and Test splits.
    All records sharing the same group_id are assigned together to one split.
    """
    grouped_records, group_stats = create_near_duplicate_groups(records)

    # Group records by group_id
    group_map = defaultdict(list)
    for r in grouped_records:
        group_map[r["group_id"]].append(r)

    unique_groups = sorted(list(group_map.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = int(round(train_ratio * n_groups))
    n_val = int(round(val_ratio * n_groups))

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    train_records = []
    val_records = []
    test_records = []

    for gid, group_items in group_map.items():
        if gid in train_groups:
            train_records.extend(group_items)
        elif gid in val_groups:
            val_records.extend(group_items)
        else:
            test_records.extend(group_items)

    # Verify 0 leakage between splits
    train_g = {r["group_id"] for r in train_records}
    val_g = {r["group_id"] for r in val_records}
    test_g = {r["group_id"] for r in test_records}

    assert len(train_g.intersection(val_g)) == 0, "Leakage detected: Train/Val group overlap!"
    assert len(train_g.intersection(test_g)) == 0, "Leakage detected: Train/Test group overlap!"
    assert len(val_g.intersection(test_g)) == 0, "Leakage detected: Val/Test group overlap!"

    split_meta = {
        "seed": seed,
        "total_records": len(records),
        "unique_groups": n_groups,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_intent_dist": dict(Counter([r["intent"] for r in train_records])),
        "val_intent_dist": dict(Counter([r["intent"] for r in val_records])),
        "test_intent_dist": dict(Counter([r["intent"] for r in test_records]))
    }

    return train_records, val_records, test_records, split_meta
