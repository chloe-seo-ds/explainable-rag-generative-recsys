# Preliminary Results — RecSys 2026 Tutorial

**Dataset:** Amazon Books 2023, 20K users, 10-core filtered  
**Model for LLM inference:** Qwen/Qwen2.5-3B-Instruct (float16, local GPU)  
**Evaluation:** Leave-last-two-out temporal split; shared-pool uses 100 candidates (1 positive + 99 negatives) per user

---

## Paradigm 1 — Explainable (BPR-MF + LLM Explanation)

**Anchor paper:** XRec (Ma et al., EMNLP 2024)  
**Run date:** 2026-05-11  
**Status:** Complete

### Ranking — Full Catalog

| Metric | Score |
|---|---|
| HR@5 | 0.0158 |
| NDCG@5 | 0.0101 |
| HR@10 | 0.0220 |
| NDCG@10 | 0.0121 |
| HR@20 | 0.0325 |
| NDCG@20 | 0.0148 |

### Ranking — Shared Pool (100 candidates/user)

| Metric | Score |
|---|---|
| HR@5 | 0.1679 |
| NDCG@5 | 0.1197 |
| HR@10 | 0.2440 |
| NDCG@10 | 0.1440 |
| HR@20 | 0.3670 |
| NDCG@20 | 0.1749 |

### Explanation Quality (100 users, Qwen2.5-3B)

| Metric | Score | Notes |
|---|---|---|
| Relevance (lexical) | 0.264 | Lexical overlap between explanation and history |
| Relevance (embedding) | 0.554 | Semantic similarity via sentence-transformers |
| Specificity | 0.710 | Fraction of explanations referencing specific titles |
| Consistency | 0.358 | Pairwise Jaccard across 3 runs (20 users) |
| Consistency (embedding) | 0.861 | Embedding-based consistency |
| Hallucination rate | 0.321 | Fraction mentioning titles not in user history |

### System Metrics

| Metric | Value |
|---|---|
| Ranking latency (mean) | 1.4 ms/user |
| Explanation latency (mean) | 20,449 ms/explanation (~20s) |
| Explanation latency (p50) | 22,816 ms |
| Explanation latency (p95) | 45,428 ms |
| Total explanation time (100 users) | ~34 min |

**Notes:**
- Full-catalog HR@10 of 2.2% is expected for BPR-MF over ~50K items without reranking
- Shared-pool HR@10 of 24.4% is the controlled comparison figure for the proposal table
- Hallucination rate of 32% is a meaningful tutorial talking point — 1 in 3 explanations invents a title
- Explanation latency of ~20s/user is due to Qwen2.5-3B on 8GB VRAM; outputs are cached so this is a one-time cost
- Consistency embedding score (0.86) is high while lexical consistency (0.36) is lower — reflects paraphrase variation rather than factual inconsistency

---

## Paradigm 2 — RAG (FAISS Retrieval + LLM Reranking)

**Anchor paper:** K-RagRec (Wang et al., ACL 2025)  
**Run date:** 2026-05-12  
**Status:** Complete

### Ranking — Full Catalog (200 users)

| Metric | Score |
|---|---|
| HR@5 | 0.0500 |
| NDCG@5 | 0.0293 |
| HR@10 | 0.0500 |
| NDCG@10 | 0.0293 |
| HR@20 | 0.0500 |
| NDCG@20 | 0.0293 |

### Ranking — Shared Pool (100 candidates/user)

| Metric | Score |
|---|---|
| HR@5 | 0.2091 |
| NDCG@5 | 0.1584 |
| HR@10 | 0.2829 |
| NDCG@10 | 0.1820 |
| HR@20 | 0.4108 |
| NDCG@20 | 0.2140 |

### Explanation Quality (100 users, template-based consistency)

| Metric | Score | Notes |
|---|---|---|
| Relevance (lexical) | 0.222 | Lexical overlap between explanation and history |
| Relevance (embedding) | 0.602 | Semantic similarity via sentence-transformers |
| Specificity | 0.631 | Fraction of explanations referencing specific titles |
| Consistency | 1.000 | Artifact — deterministic template used for consistency runs |
| Consistency (embedding) | 1.000 | Same artifact |
| Hallucination rate | 0.369 | Fraction mentioning titles not in user history |

### System Metrics

| Metric | Value |
|---|---|
| Latency (mean) | 30,122 ms/user (~30s) |
| Latency (p50) | 28,682 ms |
| Latency (p95) | 61,708 ms |
| Total pipeline time (200 users) | ~100 min |

**Notes:**
- Full-catalog HR@K is flat (same at K=5, 10, 20) because the LLM reranker outputs a truncated JSON list (~5 items); no items are added beyond rank 5 in many cases
- Shared-pool HR@10 of 28.3% outperforms NB01 (24.4%), showing LLM reranking adds value over BPR-MF
- Consistency = 1.0 is an artifact: the consistency runs use the deterministic `generate_rag_explanation` template, not the LLM; not a meaningful signal
- Hallucination rate of 37% is slightly higher than NB01 (32%), likely because the retrieval-based explanations reference the top retrieved title regardless of user history overlap

---

## Paradigm 3 — Generative (LLM-Native, GenRec-style)

**Anchor paper:** GenRec (Ji et al., ECIR 2024)  
**Status:** Pending

---

## Cross-Paradigm Summary (to be filled after NB02 and NB03)

| Paradigm | Model | HR@10 (pool) | NDCG@10 (pool) | Expl. Relevance | Hallucination | Ranking Latency |
|---|---|---|---|---|---|---|
| Explainable | BPR-MF + Qwen2.5-3B | 0.2440 | 0.1440 | 0.554 | 0.321 | 1.4 ms |
| RAG | FAISS + Qwen2.5-3B rerank | 0.2829 | 0.1820 | 0.602 | 0.369 | 30,122 ms |
| Generative | Qwen2.5-3B zero-shot | — | — | N/A | N/A | — |
