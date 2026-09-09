"""
Unit tests for data loader utilities and thread reconstruction logic.
"""
import pytest
import pandas as pd
from pathlib import Path
from hiver_agent.config import load_config
from hiver_agent.data.loader import load_twitter_sample, load_banking77, reconstruct_thread_sample


def test_load_twitter_sample():
    config = load_config()
    tweets = load_twitter_sample(config.datasets.twitter.raw_path, nrows=50)
    assert len(tweets) == 50
    assert tweets[0].tweet_id == 1
    assert tweets[0].author_id == "sprintcare"


def test_load_banking77():
    config = load_config()
    df_train, df_test = load_banking77(
        config.datasets.banking77.train_path,
        config.datasets.banking77.test_path
    )
    assert len(df_train) == 9993
    assert len(df_test) == 3076
    assert set(df_train.columns) == {"text", "label", "label_text"}


def test_reconstruct_thread_sample():
    sample_data = [
        {"tweet_id": 8, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 21:45:10 +0000 2017", "text": "@AmazonHelp issue", "response_tweet_id": "6", "in_response_to_tweet_id": None},
        {"tweet_id": 6, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 31 21:46:24 +0000 2017", "text": "@cust1 please DM us", "response_tweet_id": "5", "in_response_to_tweet_id": 8.0},
        {"tweet_id": 5, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 21:49:35 +0000 2017", "text": "@AmazonHelp sent DM", "response_tweet_id": None, "in_response_to_tweet_id": 6.0}
    ]
    df = pd.DataFrame(sample_data)
    thread = reconstruct_thread_sample(df, root_tweet_id=8)
    assert thread is not None
    assert thread.thread_id == 8
    assert thread.brand_account == "AmazonHelp"
    assert thread.total_tweets == 3
    assert thread.depth == 3
    assert thread.is_multi_turn is True
    assert thread.inbound_count == 2
    assert thread.outbound_count == 1
