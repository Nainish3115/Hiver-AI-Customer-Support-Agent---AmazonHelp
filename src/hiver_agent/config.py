"""
Configuration manager for Hiver Support Agent.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    raw_path: str
    chunk_size: int = 250000
    date_format: str = "%a %b %d %H:%M:%S %z %Y"


class Banking77Config(BaseModel):
    train_path: str
    test_path: str


class DatasetsConfig(BaseModel):
    twitter: DatasetConfig
    banking77: Banking77Config


class ProjectConfig(BaseModel):
    name: str = "hiver-support-agent"
    version: str = "0.1.0"
    target_brand: str = "AmazonHelp"


class PreprocessingConfig(BaseModel):
    sample_size: int = 10000
    min_thread_depth: int = 2
    max_thread_depth: int = 10


class GenerationConfig(BaseModel):
    top_k_evidence: int = 3
    confidence_threshold: float = 0.60
    llm_provider: str = "gemini"
    model_name: str = "gemini-1.5-flash"
    max_response_length: int = 500
    temperature: float = 0.2


class DecisionConfig(BaseModel):
    risk_threshold: float = 0.35
    confidence_threshold: float = 0.60
    min_retrieval_score: float = 0.70
    high_risk_intents: list[str] = Field(
        default_factory=lambda: [
            "payment_billing_issue",
            "account_access_login",
            "third_party_seller_issue",
            "refund_status_request",
            "damaged_defective_item"
        ]
    )
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "intent_uncertainty": 0.40,
            "retrieval_quality": 0.30,
            "category_risk": 0.30
        }
    )



class RetrievalConfig(BaseModel):
    mode: str = "hybrid"
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rrf_k: int = 60
    intent_boost_factor: float = 0.15


class RerankingConfig(BaseModel):
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 30
    final_top_k: int = 3


class AppConfig(BaseModel):
    project: ProjectConfig
    datasets: DatasetsConfig
    preprocessing: PreprocessingConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)




def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load and parse configuration YAML into Pydantic models.
    """
    if config_path is None:
        # Default to configs/config.yaml relative to project root
        root_dir = Path(__file__).resolve().parent.parent.parent
        config_path = str(root_dir / "configs" / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)
