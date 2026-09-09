"""
Unit tests for data schema validation.
"""
import pytest
from hiver_agent.data.schema import TweetRecord, ConversationThread, Banking77Record


def test_tweet_record_validation_valid():
    data = {
        "tweet_id": 1,
        "author_id": "sprintcare",
        "inbound": False,
        "created_at": "Tue Oct 31 22:10:47 +0000 2017",
        "text": "Hello @115712, how can we help?",
        "response_tweet_id": "2",
        "in_response_to_tweet_id": 3.0
    }
    tweet = TweetRecord(**data)
    assert tweet.tweet_id == 1
    assert tweet.author_id == "sprintcare"
    assert tweet.inbound is False
    assert tweet.in_response_to_tweet_id == 3
    assert tweet.response_tweet_id == "2"


def test_tweet_record_null_parent_id():
    data = {
        "tweet_id": 8,
        "author_id": "115712",
        "inbound": True,
        "created_at": "Tue Oct 31 21:45:10 +0000 2017",
        "text": "@sprintcare bad service",
        "response_tweet_id": "9,6,10",
        "in_response_to_tweet_id": None
    }
    tweet = TweetRecord(**data)
    assert tweet.in_response_to_tweet_id is None
    assert tweet.response_tweet_id == "9,6,10"


def test_banking77_record_validation():
    data = {
        "text": "How do I locate my card?",
        "label": 11,
        "label_text": "card_arrival"
    }
    rec = Banking77Record(**data)
    assert rec.label == 11
    assert rec.label_text == "card_arrival"
    assert rec.text == "How do I locate my card?"
