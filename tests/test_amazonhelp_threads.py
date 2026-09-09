"""
Deterministic unit tests for AmazonHelp thread reconstruction, graph mechanics,
quality filtering, schema validation, and stratified sampling.
"""
import pytest
import json
import pandas as pd
import numpy as np

from hiver_agent.data.amazon_builder import build_amazonhelp_threads
from hiver_agent.data.filters import apply_amazonhelp_quality_filters
from hiver_agent.analysis.intent_discovery import classify_single_text_intent, discover_amazonhelp_intents
from hiver_agent.data.sampler import generate_stratified_sample


@pytest.fixture
def sample_amazon_df():
    """
    Synthetic fixture representing an AmazonHelp tweet interaction graph.
    Contains:
    - Thread 100: Complete 3-turn multi-turn conversation (Root 100 -> Reply 101 -> Reply 102)
    - Thread 200: Single-tweet orphan (Root 200, no responses)
    - Thread 300: Thread with missing external parent pointer
    """
    data = [
        # Thread 100
        {
            "tweet_id": 100,
            "author_id": "cust_1",
            "inbound": True,
            "created_at": "Wed Nov 01 10:00:00 +0000 2017",
            "text": "@AmazonHelp Where is my order? Tracking says delayed.",
            "in_response_to_tweet_id_clean": -1
        },
        {
            "tweet_id": 101,
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Wed Nov 01 10:05:00 +0000 2017",
            "text": "@cust_1 We'd like to check this! Please DM us your order number.",
            "in_response_to_tweet_id_clean": 100
        },
        {
            "tweet_id": 102,
            "author_id": "cust_1",
            "inbound": True,
            "created_at": "Wed Nov 01 10:10:00 +0000 2017",
            "text": "@AmazonHelp DM sent, thanks.",
            "in_response_to_tweet_id_clean": 101
        },

        # Thread 200 (Orphan)
        {
            "tweet_id": 200,
            "author_id": "cust_2",
            "inbound": True,
            "created_at": "Wed Nov 01 11:00:00 +0000 2017",
            "text": "@AmazonHelp hello",
            "in_response_to_tweet_id_clean": -1
        },

        # Thread 300 (Missing external parent 999)
        {
            "tweet_id": 300,
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Wed Nov 01 12:00:00 +0000 2017",
            "text": "@cust_3 Please let us know if you still need help.",
            "in_response_to_tweet_id_clean": 999  # 999 not in dataset
        }
    ]
    df = pd.DataFrame(data)
    df['dt'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    return df


def test_thread_reconstruction_chronological_ordering(sample_amazon_df):
    df_threads, stats = build_amazonhelp_threads(sample_amazon_df)
    assert len(df_threads) == 3
    assert stats["total_reconstructed_conversations"] == 3

    # Find thread 100
    t100 = df_threads[df_threads['conversation_id'] == 100].iloc[0]
    assert t100['num_tweets'] == 3
    assert t100['num_inbound'] == 2
    assert t100['num_outbound'] == 1
    assert bool(t100['contains_both']) is True
    assert t100['depth'] == 3
    assert t100['duration_seconds'] == 600.0  # 10 minutes

    # Verify chronological sorting in JSON payload
    tweets = json.loads(t100['tweets_json'])
    assert len(tweets) == 3
    assert tweets[0]['tweet_id'] == 100
    assert tweets[1]['tweet_id'] == 101
    assert tweets[2]['tweet_id'] == 102


def test_orphan_and_missing_parent_handling(sample_amazon_df):
    df_threads, stats = build_amazonhelp_threads(sample_amazon_df)

    t200 = df_threads[df_threads['conversation_id'] == 200].iloc[0]
    assert bool(t200['is_orphan']) is True
    assert bool(t200['contains_both']) is False

    t300 = df_threads[df_threads['conversation_id'] == 300].iloc[0]
    assert bool(t300['has_missing_parent']) is True


def test_quality_filtering(sample_amazon_df):
    df_threads, stats = build_amazonhelp_threads(sample_amazon_df)
    df_eligible, breakdown = apply_amazonhelp_quality_filters(df_threads)

    # Only Thread 100 satisfies all criteria:
    # - Thread 100: contains both (True), num_tweets (3 >= 2), text len (>= 15) -> ELIGIBLE
    # - Thread 200: text len < 15, contains both False -> INELIGIBLE
    # - Thread 300: contains both False -> INELIGIBLE
    assert len(df_eligible) == 1
    assert df_eligible.iloc[0]['conversation_id'] == 100
    assert breakdown['total_eligible_after'] == 1


def test_intent_classification_rules():
    t1 = "@AmazonHelp Where is my order? Tracking says delayed."
    intent1 = classify_single_text_intent(t1)
    assert intent1 == "order_tracking_delivery_status"

    t2 = "@AmazonHelp My Prime Video app is throwing error code 5001"
    intent2 = classify_single_text_intent(t2)
    assert intent2 == "digital_content_kindle_video"

    t3 = "@AmazonHelp why was I charged $139 for Prime membership?"
    intent3 = classify_single_text_intent(t3)
    assert intent3 == "prime_membership_subscription"


def test_stratified_sampler(sample_amazon_df):
    df_threads, stats = build_amazonhelp_threads(sample_amazon_df)
    df_threads, summary = discover_amazonhelp_intents(df_threads)
    df_threads, _ = apply_amazonhelp_quality_filters(df_threads)

    df_sampled = generate_stratified_sample(df_threads, target_sample_size=2, random_seed=42)
    assert len(df_sampled) > 0
    assert "discovered_intent" in df_sampled.columns
