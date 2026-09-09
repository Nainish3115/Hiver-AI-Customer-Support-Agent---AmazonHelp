"""
Diagnostic benchmark script for SentenceTransformer embedding provider.
Evaluates model loading, batch throughput, memory, and max_seq_length impact on a 1,000-document sample.
Usage:
    python scripts/benchmark_embedding.py
"""
import sys
import os
import time
import json
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import torch
import numpy as np
from sentence_transformers import SentenceTransformer


def measure_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def main():
    print("=== Step 2 & 3: Bounded 1,000-Document Embedding Benchmark ===", flush=True)
    t_start = time.time()
    mem_initial = measure_memory_mb()

    model_name = "all-MiniLM-L6-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_available = torch.cuda.is_available()

    # Load 1,000 documents
    jsonl_path = Path("data/processed/amazonhelp_rag_documents.jsonl")
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} does not exist.", flush=True)
        return

    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 1000:
                break
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    sample_texts = [d["document_text"] for d in documents]
    print(f"Loaded {len(sample_texts):,} sample documents for benchmark.", flush=True)

    # Benchmark Model Load
    t0_load = time.time()
    torch.set_num_threads(4)
    model = SentenceTransformer(model_name, device=device)
    model_load_time = time.time() - t0_load

    embedding_dim = model.get_sentence_embedding_dimension()
    mem_after_load = measure_memory_mb()

    # Test Configurations
    configs_to_test = [
        {"batch_size": 128, "max_seq_length": 128, "num_threads": 4},
        {"batch_size": 256, "max_seq_length": 128, "num_threads": 4},
        {"batch_size": 256, "max_seq_length": 256, "num_threads": 4},
    ]

    bench_results = []

    for cfg in configs_to_test:
        bs = cfg["batch_size"]
        msl = cfg["max_seq_length"]
        nth = cfg["num_threads"]

        torch.set_num_threads(nth)
        model.max_seq_length = msl

        # Warmup
        _ = model.encode(sample_texts[:10], batch_size=bs, show_progress_bar=False, normalize_embeddings=True)

        t0_enc = time.time()
        mem_before_enc = measure_memory_mb()
        embeddings = model.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
        enc_time = time.time() - t0_enc
        mem_after_enc = measure_memory_mb()

        docs_per_sec = len(sample_texts) / enc_time
        est_full_sec = 81186 / docs_per_sec
        est_full_min = est_full_sec / 60.0

        res_entry = {
            "batch_size": bs,
            "max_seq_length": msl,
            "num_threads": nth,
            "encoding_time_seconds": round(enc_time, 3),
            "docs_per_second": round(docs_per_sec, 2),
            "est_full_corpus_minutes": round(est_full_min, 2),
            "est_full_corpus_seconds": round(est_full_sec, 1),
            "rss_mem_mb": round(mem_after_enc, 2),
            "embedding_shape": list(embeddings.shape)
        }
        bench_results.append(res_entry)
        print(f"Config (BS={bs}, SeqLen={msl}, Threads={nth}): Encoded {len(sample_texts)} in {enc_time:.3f}s | {docs_per_sec:.1f} docs/sec | Est Full Corpus: {est_full_min:.2f} min", flush=True)

    summary = {
        "model_name": model_name,
        "embedding_dimension": embedding_dim,
        "device": device,
        "cuda_available": cuda_available,
        "cpu_count": os.cpu_count(),
        "model_load_time_seconds": round(model_load_time, 3),
        "initial_memory_mb": round(mem_initial, 2),
        "post_load_memory_mb": round(mem_after_load, 2),
        "benchmark_sample_size": len(sample_texts),
        "configurations": bench_results,
        "total_benchmark_time_seconds": round(time.time() - t_start, 3)
    }

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "embedding_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDiagnostic benchmark complete! Output saved to {tables_dir / 'embedding_benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
