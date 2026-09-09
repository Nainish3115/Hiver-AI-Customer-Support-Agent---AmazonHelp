"""
Safe, memory-efficient data loading and thread reconstruction infrastructure.
"""
from typing import Generator, List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, deque

from hiver_agent.data.schema import TweetRecord, ConversationThread, Banking77Record


def read_twitter_chunks(
    csv_path: str,
    chunksize: int = 250000,
    usecols: Optional[List[str]] = None
) -> Generator[pd.DataFrame, None, None]:
    """
    Memory-efficient generator yielding chunks of the Customer Support on Twitter dataset.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw Twitter dataset not found at {csv_path}")

    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False):
        yield chunk


def load_twitter_sample(
    csv_path: str,
    nrows: int = 1000
) -> List[TweetRecord]:
    """
    Load a safe sample of tweets and validate against the Pydantic TweetRecord schema.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw Twitter dataset not found at {csv_path}")

    df = pd.read_csv(path, nrows=nrows)
    validated_records = []
    for _, row in df.iterrows():
        rec_dict = row.to_dict()
        record = TweetRecord(**rec_dict)
        validated_records.append(record)

    return validated_records


def load_banking77(
    train_path: str,
    test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Banking77 train and test datasets.
    """
    tr_p = Path(train_path)
    te_p = Path(test_path)
    if not tr_p.exists():
        raise FileNotFoundError(f"Banking77 train dataset not found at {train_path}")
    if not te_p.exists():
        raise FileNotFoundError(f"Banking77 test dataset not found at {test_path}")

    df_train = pd.read_csv(tr_p)
    df_test = pd.read_csv(te_p)
    return df_train, df_test


def reconstruct_thread_sample(
    df: pd.DataFrame,
    root_tweet_id: int
) -> Optional[ConversationThread]:
    """
    Reconstruct a multi-turn conversation thread tree starting from a root tweet ID.
    Does not alter raw files.
    """
    # Create fast index lookup
    if 'tweet_id' not in df.columns or 'in_response_to_tweet_id' not in df.columns:
        raise ValueError("DataFrame missing required tweet ID fields for thread reconstruction")

    row_by_id = {}
    children_by_id = defaultdict(list)

    for _, row in df.iterrows():
        tid = int(row['tweet_id'])
        row_by_id[tid] = row.to_dict()
        pid_val = row['in_response_to_tweet_id']
        if pd.notnull(pid_val):
            try:
                pid = int(float(pid_val))
                children_by_id[pid].append(tid)
            except (ValueError, TypeError):
                pass

    if root_tweet_id not in row_by_id:
        return None

    # Graph traversal to collect thread tweets
    stack = [root_tweet_id]
    thread_tweets = []
    brands_in_thread = []
    inb_count = 0
    outb_count = 0
    max_depth = 1

    depth_map = {root_tweet_id: 1}

    while stack:
        curr = stack.pop()
        raw_row = row_by_id[curr]
        tweet_rec = TweetRecord(**raw_row)
        thread_tweets.append(tweet_rec)

        d = depth_map[curr]
        if d > max_depth:
            max_depth = d

        if tweet_rec.inbound:
            inb_count += 1
        else:
            outb_count += 1
            brands_in_thread.append(tweet_rec.author_id)

        for child in children_by_id.get(curr, []):
            if child in row_by_id:
                depth_map[child] = d + 1
                stack.append(child)

    primary_brand = brands_in_thread[0] if brands_in_thread else "unknown"

    return ConversationThread(
        thread_id=root_tweet_id,
        brand_account=primary_brand,
        tweets=thread_tweets,
        depth=max_depth,
        total_tweets=len(thread_tweets),
        inbound_count=inb_count,
        outbound_count=outb_count,
        is_multi_turn=len(thread_tweets) >= 3
    )
