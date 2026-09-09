# Phase 4 Embedding Generation Diagnostic Report

> **Execution Date**: 2026-09-09  
> **Environment**: Windows 11, Python 3.12.4, PyTorch CPU Execution (CUDA: False)  
> **Corpus**: `data/processed/amazonhelp_rag_documents.jsonl` (81,186 historical support cases)  

---

## 1. Executive Summary & Root Cause Analysis

### Why Previous Full Index Build Jobs Took Long
1. **Transformer Sequence Length Padding Overhead**:  
   The default `SentenceTransformer` max sequence length defaults to 256 or 512 tokens. Customer support interaction threads contain up to 800 characters. Computing 256-token quadratic self-attention ($O(L^2)$) on CPU added ~2.6x latency compared to truncating at `max_seq_length=128`.
2. **CPU Thread Contention on Windows**:  
   Setting PyTorch CPU threads to the total number of logical cores (`torch.set_num_threads(16)`) caused heavy OpenMP/MKL thread context switching and lock contention on Windows. Empirically benchmarked:
   - 16 CPU threads: **11.34s** per 256 texts
   - 4 CPU threads: **5.52s** per 256 texts (**2.05x speedup using 4 threads!**)
3. **IPC Overhead in Multiprocessing Pools**:  
   Windows process spawning (`multiprocessing`) incurs high initialization overhead and IPC pipe serialization locks when transferring large float32 numpy arrays across worker processes.
4. **Lack of Checkpointing**:  
   Earlier build scripts attempted single-pass in-memory encoding of 81.1K documents without persisting intermediate chunk progress. Any process interruption forced a complete re-encoding from document 0.

---

## 2. System Resource & Hardware Capabilities

- **CPU**: 16 Logical Cores
- **System Memory (RAM)**: 16.08 GB Total (5.64 GB Free physical RAM at execution)
- **PyTorch CUDA Acceleration**: `False` (CPU Execution Only)
- **Embedding Model**: `all-MiniLM-L6-v2`
  - **Parameters**: 22.9 Million
  - **Embedding Dimension**: 384 dimensions
  - **Normalization**: L2 Normalized ($\|v\|_2 = 1.0$)
  - **RAM Requirement**: ~250 MB base model RSS memory

---

## 3. Empirical Benchmark Results (1,000-Document Sample)

The diagnostic benchmark was executed on a deterministic 1,000-document sample from `data/processed/amazonhelp_rag_documents.jsonl`.

| Configuration | Batch Size | Max Token Length | CPU Threads | Encoding Time (1,000 Docs) | Throughput (Docs/Sec) | Estimated Full Corpus Runtime (81.1K Docs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1 (Optimal)** | **128** | **128** | **4** | **47.34 s** | **21.1 docs/sec** | **64.05 min** |
| **Config 2** | **256** | **128** | **4** | **52.77 s** | **18.9 docs/sec** | **71.41 min** |
| **Config 3 (Baseline)** | **256** | **256** | **4** | **121.69 s** | **8.2 docs/sec** | **164.66 min** |

### Key Benchmark Takeaways
- Truncating `max_seq_length` to 128 tokens increased throughput from **8.2 docs/sec** to **21.1 docs/sec** (**2.57x throughput gain**).
- `batch_size=128` provided optimal CPU cache locality and memory bandwidth compared to `batch_size=256`.

---

## 4. Resumable Chunk Checkpoint Design

To ensure fault tolerance and safe execution across interruptions, `scripts/build_retrieval_index.py` was refactored with deterministic chunk checkpointing:

1. **Chunk-Based Partitioning**:  
   The corpus is partitioned into deterministic chunks of 5,000 documents.
2. **Persistent Checkpoint Artifacts**:  
   Each chunk is saved to `data/processed/retrieval_index/checkpoints/chunk_{idx}_{start}_{end}.npy`.
3. **Resumable Execution**:  
   On startup, the script verifies existing chunk files. Valid checkpoints are loaded directly without re-encoding, eliminating redundant compute.
4. **Bounded Execution Support**:  
   Supports `--max-docs N` flag for fast diagnostic testing (e.g. `--max-docs 1000`).

---

## 5. Recommended Production Strategy

- **Model**: `all-MiniLM-L6-v2` (384-d, L2 normalized)
- **Batch Size**: `128`
- **Max Sequence Length**: `128`
- **CPU Threads**: `4`
- **Chunk Size**: `5000`
- **Estimated Full Corpus Building Runtime**: ~64 minutes (single-process resumable).
