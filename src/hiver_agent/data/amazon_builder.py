"""
AmazonHelp conversation extractor and thread graph builder.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

from hiver_agent.data.schema import TweetRecord, ConversationThread


def extract_amazonhelp_raw_df(csv_path: str) -> pd.DataFrame:
    """
    Extract all tweets associated with AmazonHelp from raw Twitter dataset.
    Associated tweets include:
    1. Tweets authored by AmazonHelp (outbound)
    2. Inbound customer tweets that AmazonHelp responded to
    3. Inbound tweets that mention AmazonHelp or are part of AmazonHelp thread graphs
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV dataset not found at {csv_path}")

    print("Loading raw CSV for AmazonHelp extraction...", flush=True)
    df_raw = pd.read_csv(csv_path, low_memory=False)

    # Convert ID columns
    df_raw['tweet_id'] = df_raw['tweet_id'].astype(np.int64)
    df_raw['in_response_to_tweet_id_clean'] = df_raw['in_response_to_tweet_id'].fillna(-1).astype(np.int64)

    # 1. Direct AmazonHelp outbound
    amazon_outbound_mask = (df_raw['author_id'] == 'AmazonHelp') & (~df_raw['inbound'].astype(bool))
    amazon_outbound_tids = set(df_raw[amazon_outbound_mask]['tweet_id'])
    print(f"Direct AmazonHelp outbound tweets: {len(amazon_outbound_tids)}", flush=True)

    # 2. Parents of AmazonHelp outbound
    amazon_parents = set(df_raw[amazon_outbound_mask]['in_response_to_tweet_id_clean'])
    amazon_parents.discard(-1)

    # 3. Children of AmazonHelp outbound
    amazon_children = set(df_raw[df_raw['in_response_to_tweet_id_clean'].isin(amazon_outbound_tids)]['tweet_id'])

    # 4. Inbound mentioning AmazonHelp
    mentions_amazon_mask = df_raw['inbound'].astype(bool) & df_raw['text'].str.contains('AmazonHelp', case=False, na=False)
    mentions_tids = set(df_raw[mentions_amazon_mask]['tweet_id'])

    # Initial set of relevant tweet IDs
    relevant_tids = amazon_outbound_tids | amazon_parents | amazon_children | mentions_tids
    print(f"Initial seed AmazonHelp tweet IDs: {len(relevant_tids)}", flush=True)

    # Expand 2 hops to capture multi-turn conversation context
    parent_map = dict(zip(df_raw['tweet_id'], df_raw['in_response_to_tweet_id_clean']))
    
    # Adjacency children
    children_map = defaultdict(list)
    for tid, pid in parent_map.items():
        if pid != -1:
            children_map[pid].append(tid)

    expanded_tids = set(relevant_tids)
    for tid in list(relevant_tids):
        # Add parent
        p = parent_map.get(tid, -1)
        if p != -1:
            expanded_tids.add(p)
            gp = parent_map.get(p, -1)
            if gp != -1:
                expanded_tids.add(gp)
        # Add children
        for c in children_map.get(tid, []):
            expanded_tids.add(c)
            for gc in children_map.get(c, []):
                expanded_tids.add(gc)

    print(f"Expanded AmazonHelp tweet IDs (multi-turn context): {len(expanded_tids)}", flush=True)

    df_amazon = df_raw[df_raw['tweet_id'].isin(expanded_tids)].copy()
    return df_amazon


def build_amazonhelp_threads(df_amazon: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reconstruct multi-turn conversations from AmazonHelp tweet subset.
    Returns:
      - DataFrame of reconstructed conversation threads
      - Dictionary of graph & data quality metrics
    """
    print("Reconstructing AmazonHelp conversation threads...", flush=True)

    # Parse timestamps
    df_amazon['dt'] = pd.to_datetime(df_amazon['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')

    # Build graph mappings
    tweet_dict = {}
    parent_map = {}
    children_map = defaultdict(list)

    tids = df_amazon['tweet_id'].values
    pids = df_amazon['in_response_to_tweet_id_clean'].values
    authors = df_amazon['author_id'].astype(str).values
    inbounds = df_amazon['inbound'].astype(bool).values
    texts = df_amazon['text'].astype(str).values
    dts = df_amazon['dt'].values

    for tid, pid, auth, inb, txt, dt in zip(tids, pids, authors, inbounds, texts, dts):
        tweet_dict[tid] = {
            "tweet_id": int(tid),
            "author_id": auth,
            "inbound": bool(inb),
            "created_at": str(pd.Timestamp(dt)),
            "dt": pd.Timestamp(dt),
            "text": txt,
            "parent_id": int(pid)
        }
        parent_map[tid] = int(pid)
        if pid != -1:
            children_map[int(pid)].append(int(tid))

    # Identify roots: A tweet is a root if pid == -1 or parent not in dataset
    all_tids = set(tweet_dict.keys())
    roots = []
    missing_parent_count = 0
    orphan_count = 0

    for tid, pid in parent_map.items():
        if pid == -1:
            roots.append(tid)
        elif pid not in all_tids:
            roots.append(tid)
            missing_parent_count += 1

    print(f"Identified {len(roots)} conversation roots ({missing_parent_count} with missing external parents)", flush=True)

    threads_data = []

    for r_tid in roots:
        # Traverse tree starting from root
        stack = [r_tid]
        thread_tids = []
        has_missing_p = (parent_map.get(r_tid, -1) != -1 and parent_map.get(r_tid, -1) not in all_tids)

        while stack:
            curr = stack.pop()
            thread_tids.append(curr)

            # Check if any child has missing references
            for child in children_map.get(curr, []):
                stack.append(child)

        # Retrieve tweets and sort chronologically
        thread_tweets = [tweet_dict[t] for t in thread_tids if t in tweet_dict]
        thread_tweets.sort(key=lambda x: x['dt'])

        num_tweets = len(thread_tweets)
        if num_tweets == 1 and not children_map.get(r_tid, []):
            orphan_count += 1

        inb_cnt = sum(1 for t in thread_tweets if t['inbound'])
        outb_cnt = sum(1 for t in thread_tweets if not t['inbound'])
        participants = sorted(list(set(t['author_id'] for t in thread_tweets)))
        contains_both = (inb_cnt > 0 and outb_cnt > 0)

        start_dt = thread_tweets[0]['dt']
        end_dt = thread_tweets[-1]['dt']
        duration_sec = (end_dt - start_dt).total_seconds() if pd.notnull(start_dt) and pd.notnull(end_dt) else 0.0

        # Calculate max tree depth
        depth_dict = {r_tid: 1}
        st_depth = [r_tid]
        max_d = 1
        while st_depth:
            c = st_depth.pop()
            d = depth_dict[c]
            if d > max_d:
                max_d = d
            for ch in children_map.get(c, []):
                if ch in tweet_dict:
                    depth_dict[ch] = d + 1
                    st_depth.append(ch)

        # Format JSON serialization of tweets for parquet storage
        serialized_tweets = [
            {
                "tweet_id": t["tweet_id"],
                "author_id": t["author_id"],
                "inbound": t["inbound"],
                "created_at": t["created_at"],
                "text": t["text"]
            }
            for t in thread_tweets
        ]

        # First customer query text
        cust_texts = [t["text"] for t in thread_tweets if t["inbound"]]
        first_customer_text = cust_texts[0] if cust_texts else ""

        threads_data.append({
            "conversation_id": int(r_tid),
            "num_tweets": num_tweets,
            "num_inbound": inb_cnt,
            "num_outbound": outb_cnt,
            "start_timestamp": str(start_dt),
            "end_timestamp": str(end_dt),
            "duration_seconds": float(duration_sec),
            "participants": ",".join(participants),
            "contains_both": contains_both,
            "depth": max_d,
            "has_missing_parent": has_missing_p,
            "is_orphan": (num_tweets == 1),
            "is_broken_chain": has_missing_p or (num_tweets == 1 and inb_cnt > 0 and outb_cnt == 0),
            "first_customer_text": first_customer_text,
            "tweets_json": json.dumps(serialized_tweets, ensure_ascii=False)
        })

    df_threads = pd.DataFrame(threads_data)

    stats = {
        "total_extracted_tweets": len(df_amazon),
        "total_reconstructed_conversations": len(df_threads),
        "conversations_with_both_participants": int(df_threads['contains_both'].sum()),
        "single_tweet_orphans": orphan_count,
        "threads_with_missing_parents": missing_parent_count,
        "multi_turn_conversations_cnt": int((df_threads['num_tweets'] >= 3).sum()),
        "avg_thread_depth": float(df_threads['depth'].mean()),
        "max_thread_depth": int(df_threads['depth'].max())
    }

    return df_threads, stats
