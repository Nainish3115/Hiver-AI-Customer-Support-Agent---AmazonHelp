"""
CLI script to build FAISS vector index with deterministic chunk checkpointing and resumability.
Supports interrupting, resuming, and bounded testing (--max-docs N).

Usage:
    python scripts/build_retrieval_index.py [--max-docs 1000] [--chunk-size 5000]
"""
import sys
import os
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import torch
import psutil

from hiver_agent.config import load_config
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore


def measure_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index with chunk checkpointing.")
    parser.add_argument("--input-jsonl", type=str, default="data/processed/amazonhelp_rag_documents.jsonl", help="Input JSONL documents corpus.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output index directory.")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit maximum documents to process.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Number of documents per checkpoint chunk.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding provider.")
    parser.add_argument("--max-seq-length", type=int, default=128, help="Max sequence token length.")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== Resumable FAISS Retrieval Index Builder ===", flush=True)
    t0 = time.time()
    mem_initial = measure_memory_mb()

    # Apply PyTorch thread limit
    torch.set_num_threads(args.threads)

    jsonl_path = Path(args.input_jsonl)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found.", flush=True)
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif "15k" in jsonl_path.name.lower():
        out_dir = Path("data/processed/retrieval_index_15k")
    else:
        out_dir = Path("data/processed/retrieval_index")

    # Load source documents
    documents = []
    seen_texts = set()
    exact_dup_cnt = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if args.max_docs and len(documents) >= args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            txt = doc.get("document_text", "")
            if txt in seen_texts:
                exact_dup_cnt += 1
            else:
                seen_texts.add(txt)
            documents.append(doc)

    total_docs = len(documents)
    print(f"Loaded {total_docs:,} input documents ({exact_dup_cnt} duplicate document texts).", flush=True)
    if args.max_docs:
        print(f"Notice: Bounded execution active (--max-docs {args.max_docs}).", flush=True)

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize EmbeddingProvider
    model_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformerEmbeddingProvider(model_name=model_name, batch_size=args.batch_size)
    embedder.model.max_seq_length = args.max_seq_length

    chunk_size = args.chunk_size
    num_chunks = (total_docs + chunk_size - 1) // chunk_size

    print(f"Processing in {num_chunks} chunk(s) of size {chunk_size:,}...", flush=True)

    chunk_files = []
    t_embed_start = time.time()

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_docs)
        chunk_docs = documents[start_idx:end_idx]

        chunk_file = checkpoint_dir / f"chunk_{chunk_idx:03d}_{start_idx}_{end_idx}.npy"
        chunk_files.append(chunk_file)

        # Resumability check: if chunk file exists and has correct shape, load it
        if chunk_file.exists():
            try:
                arr = np.load(chunk_file)
                if arr.shape == (len(chunk_docs), embedder.dimension):
                    print(f"Chunk {chunk_idx+1}/{num_chunks} [{start_idx}:{end_idx}] already encoded. Loaded checkpoint: {chunk_file.name}", flush=True)
                    continue
                else:
                    print(f"Chunk {chunk_idx+1}/{num_chunks} checkpoint size mismatch. Re-encoding...", flush=True)
            except Exception as e:
                print(f"Chunk {chunk_idx+1}/{num_chunks} checkpoint corrupt ({e}). Re-encoding...", flush=True)

        print(f"Encoding Chunk {chunk_idx+1}/{num_chunks} [{start_idx}:{end_idx}] ({len(chunk_docs):,} docs)...", flush=True)
        chunk_texts = [d["document_text"] for d in chunk_docs]
        t_chunk_start = time.time()
        chunk_vecs = embedder.embed_texts(chunk_texts)
        chunk_time = time.time() - t_chunk_start

        np.save(chunk_file, chunk_vecs)
        print(f"Chunk {chunk_idx+1}/{num_chunks} encoded in {chunk_time:.2f}s and saved to {chunk_file.name}", flush=True)

    embed_total_time = time.time() - t_embed_start

    # Assemble all chunk embeddings
    print("\nConcatenating chunk embeddings...", flush=True)
    all_chunks = [np.load(f) for f in chunk_files]
    final_embeddings = np.vstack(all_chunks).astype(np.float32)
    print(f"Final embeddings assembled. Shape: {final_embeddings.shape}", flush=True)

    # Build FAISS VectorStore
    t_idx_start = time.time()
    store = FAISSVectorStore(dimension=embedder.dimension)
    store.add_documents(documents, final_embeddings)
    index_building_time = time.time() - t_idx_start

    # Save FAISS index
    store.save(str(out_dir))

    index_bytes = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    index_mb = round(index_bytes / (1024 * 1024), 2)
    mem_final = measure_memory_mb()

    stats = {
        "source_jsonl": str(jsonl_path),
        "total_input_documents": total_docs,
        "exact_duplicate_documents": exact_dup_cnt,
        "indexed_documents_count": store.count(),
        "embedding_model": model_name,
        "embedding_dimension": embedder.dimension,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "pytorch_threads": args.threads,
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "bounded_max_docs": args.max_docs,
        "embedding_runtime_seconds": round(embed_total_time, 2),
        "index_building_runtime_seconds": round(index_building_time, 2),
        "total_runtime_seconds": round(time.time() - t0, 2),
        "index_disk_size_mb": index_mb,
        "initial_rss_mb": round(mem_initial, 2),
        "final_rss_mb": round(mem_final, 2)
    }

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "faiss_index_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nFAISS Index successfully built & saved to {out_dir} ({index_mb} MB) in {time.time()-t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
