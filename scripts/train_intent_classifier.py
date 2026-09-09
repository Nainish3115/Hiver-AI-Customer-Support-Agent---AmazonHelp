"""
CLI script to build leakage-resistant classification splits and train prototype classifier.
Usage:
    python scripts/train_intent_classifier.py
"""
import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

from hiver_agent.config import load_config
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.classification.embedding_classifier import EmbeddingPrototypeClassifier


def main():
    print("=== Step 1: Building Classification Dataset & Training Prototype Classifier ===", flush=True)
    t0 = time.time()
    config = load_config()

    # Source dataset
    golden_parquet = Path("data/golden/taxonomy_validation_sample.parquet")
    threads_parquet = Path("data/processed/amazonhelp_threads.parquet")

    if golden_parquet.exists():
        df_source = pd.read_parquet(golden_parquet)
        text_col = "conversation_text"
        label_col = "proposed_intent"
    else:
        df_source = pd.read_parquet(threads_parquet)
        text_col = "first_customer_text"
        label_col = "discovered_intent"

    # Filter out empty texts
    df_source = df_source[df_source[text_col].astype(str).str.strip() != ""].copy()

    # Leakage-resistant split grouping by conversation_id
    np.random.seed(42)
    unique_convs = df_source['conversation_id'].unique()
    np.random.shuffle(unique_convs)

    n_total = len(unique_convs)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_cids = set(unique_convs[:n_train])
    val_cids = set(unique_convs[n_train:n_train + n_val])
    test_cids = set(unique_convs[n_train + n_val:])

    df_train = df_source[df_source['conversation_id'].isin(train_cids)].copy()
    df_val = df_source[df_source['conversation_id'].isin(val_cids)].copy()
    df_test = df_source[df_source['conversation_id'].isin(test_cids)].copy()

    out_dir = Path("data/processed/classification")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(out_dir / "train.parquet", index=False)
    df_val.to_parquet(out_dir / "val.parquet", index=False)
    df_test.to_parquet(out_dir / "test.parquet", index=False)

    print(f"Created leakage-resistant splits in {out_dir}:", flush=True)
    print(f"  - Train: {len(df_train)} cases", flush=True)
    print(f"  - Val:   {len(df_val)} cases", flush=True)
    print(f"  - Test:  {len(df_test)} cases", flush=True)

    # Train EmbeddingPrototypeClassifier
    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    clf = EmbeddingPrototypeClassifier(embedder)

    train_texts = df_train[text_col].astype(str).tolist()
    train_labels = df_train[label_col].astype(str).tolist()

    clf.fit(train_texts, train_labels)

    # Save prototypes JSON
    proto_data = {c: vec.tolist() for c, vec in clf.centroids.items()}
    with open(out_dir / "prototypes.json", "w", encoding="utf-8") as f:
        json.dump(proto_data, f, indent=2)

    elapsed = time.time() - t0
    print(f"Classifier fitting complete in {elapsed:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
