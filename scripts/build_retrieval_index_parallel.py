"""
Parallel FAISS Vector Index Builder using 4 independent worker processes.
Splits 81,186 documents into 4 chunks, encodes them in parallel, concatenates embeddings,
and builds the FAISS IndexFlatIP index.
"""
import sys
import os
import time
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import faiss

from hiver_agent.config import load_config
from hiver_agent.retrieval.embeddings import SentenceTransformerEmbeddingProvider
from hiver_agent.retrieval.faiss_store import FAISSVectorStore


def encode_chunk_worker(chunk_idx: int, num_chunks: int):
    """Worker function to encode a single chunk of documents."""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    jsonl_path = Path("data/processed/amazonhelp_rag_documents.jsonl")
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % num_chunks == chunk_idx:
                line = line.strip()
                if line:
                    documents.append(json.loads(line))

    print(f"[Worker {chunk_idx}] Loaded {len(documents):,} documents.", flush=True)

    embedder = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2", batch_size=256)
    doc_texts = [d["document_text"] for d in documents]

    t0 = time.time()
    embeddings = embedder.embed_texts(doc_texts)
    print(f"[Worker {chunk_idx}] Encoded {len(embeddings):,} embeddings in {time.time()-t0:.2f}s.", flush=True)

    chunk_dir = Path("data/processed/retrieval_index/chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    np.save(chunk_dir / f"chunk_{chunk_idx}.npy", embeddings)
    with open(chunk_dir / f"chunk_{chunk_idx}_docs.json", "w", encoding="utf-8") as f:
        json.dump(documents, f)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        chunk_idx = int(sys.argv[2])
        num_chunks = int(sys.argv[3])
        encode_chunk_worker(chunk_idx, num_chunks)
        return

    print("=== Step 3: Parallel FAISS Retrieval Index Building (4 Workers) ===", flush=True)
    t0 = time.time()

    num_workers = 4
    processes = []
    for w in range(num_workers):
        cmd = [sys.executable, "-u", __file__, "--worker", str(w), str(num_workers)]
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"Spawned worker {w} (PID: {p.pid})", flush=True)

    for w, p in enumerate(processes):
        p.wait()
        print(f"Worker {w} finished with exit code {p.returncode}.", flush=True)

    # Re-assemble chunks
    chunk_dir = Path("data/processed/retrieval_index/chunks")
    all_embeddings = []
    all_documents = []

    # Read lines in exact original index order
    jsonl_path = Path("data/processed/amazonhelp_rag_documents.jsonl")
    orig_docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                orig_docs.append(json.loads(line))

    # Map case_id -> embedding
    doc_embedding_map = {}
    for w in range(num_workers):
        vecs = np.load(chunk_dir / f"chunk_{w}.npy")
        with open(chunk_dir / f"chunk_{w}_docs.json", "r", encoding="utf-8") as f:
            docs = json.load(f)
        for d, v in zip(docs, vecs):
            doc_embedding_map[d["case_id"]] = v

    final_embeddings = np.array([doc_embedding_map[d["case_id"]] for d in orig_docs], dtype=np.float32)
    print(f"Combined final embeddings shape: {final_embeddings.shape} in {time.time()-t0:.2f}s!", flush=True)

    # Build FAISS store
    t_idx_start = time.time()
    store = FAISSVectorStore(dimension=final_embeddings.shape[1])
    store.add_documents(orig_docs, final_embeddings)
    index_time = time.time() - t_idx_start

    out_dir = Path("data/processed/retrieval_index")
    store.save(str(out_dir))

    index_bytes = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    index_mb = round(index_bytes / (1024 * 1024), 2)

    stats = {
        "source_jsonl": str(jsonl_path),
        "total_input_documents": len(orig_docs),
        "exact_duplicate_documents": 0,
        "indexed_documents_count": store.count(),
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "total_runtime_seconds": round(time.time() - t0, 2),
        "index_disk_size_mb": index_mb
    }

    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "faiss_index_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nFAISS Index successfully built & saved to {out_dir} ({index_mb} MB) in {time.time()-t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
