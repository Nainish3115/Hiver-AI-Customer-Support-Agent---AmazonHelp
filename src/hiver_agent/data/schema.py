"""
Data Schemas and Pydantic Models for Customer Support and Banking77 Datasets.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TweetRecord(BaseModel):
    """
    Schema representing a raw or processed tweet from Customer Support on Twitter dataset.
    """
    tweet_id: int = Field(..., description="Unique tweet identifier")
    author_id: str = Field(..., description="Author ID (anonymized user number or brand handle)")
    inbound: bool = Field(..., description="True if customer message, False if company support")
    created_at: str = Field(..., description="Timestamp of tweet creation")
    text: str = Field(..., description="Text content of the tweet")
    response_tweet_id: Optional[str] = Field(None, description="Comma-separated tweet IDs that responded")
    in_response_to_tweet_id: Optional[int] = Field(None, description="Parent tweet ID that this tweet replied to")

    @field_validator('in_response_to_tweet_id', mode='before')
    @classmethod
    def parse_parent_id(cls, v):
        if v is None or (isinstance(v, float) and float('nan') == v) or v == "" or v == "nan":
            return None
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None

    @field_validator('response_tweet_id', mode='before')
    @classmethod
    def parse_response_id(cls, v):
        if v is None or (isinstance(v, float) and float('nan') == v) or v == "" or v == "nan":
            return None
        return str(v)


class ConversationThread(BaseModel):
    """
    Schema representing a reconstructed multi-turn customer support conversation.
    """
    thread_id: int = Field(..., description="Root tweet ID starting the conversation")
    brand_account: str = Field(..., description="Primary support brand handle involved")
    tweets: List[TweetRecord] = Field(..., description="Ordered list of tweets in the thread tree")
    depth: int = Field(..., description="Maximum depth of the thread tree")
    total_tweets: int = Field(..., description="Total number of tweets in the conversation")
    inbound_count: int = Field(..., description="Number of customer tweets")
    outbound_count: int = Field(..., description="Number of brand support tweets")
    is_multi_turn: bool = Field(..., description="True if thread contains 3 or more total tweets")


class Banking77Record(BaseModel):
    """
    Schema representing an intent classification record from Banking77 dataset.
    """
    text: str = Field(..., description="Customer utterance text")
    label: int = Field(..., description="Integer label index (0 to 76)")
    label_text: str = Field(..., description="Category label name string")
