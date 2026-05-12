# Preliminary Results — RecSys 2026 Tutorial

**Dataset:** Amazon Books 2023, 20K users, 10-core filtered
**Model for LLM inference:** Qwen/Qwen2.5-3B-Instruct
**Evaluation:** Leave-last-two-out temporal split; shared-pool uses 100 candidates (1 positive + 99 negatives) per user
**Hardware:** NB01–03 on RTX 5050 (8GB, float16); NB03.1 fine-tuning on Colab A100 (40GB, 4-bit QLoRA)

---

## Paradigm 1 — Explainable (BPR-MF + LLM Explanation)

**Anchor paper:** XRec (Ma et al., EMNLP 2024)
**Run date:** 2026-05-11
**Notebook:** `01_explainable_recsys.ipynb`
**Users evaluated:** 7,288 (ranking), 100 (explanations)

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
| Explanation latency (p95) | 45,428 ms |

**Notes:**
- Full-catalog HR@10 of 2.2% is expected for BPR-MF over ~6K items without reranking
- Shared-pool HR@10 of 24.4% is the controlled comparison figure
- Hallucination rate of 32% — 1 in 3 explanations invents a title not in the user's history
- Consistency: embedding score (0.86) is high while lexical (0.36) is lower — reflects paraphrase variation rather than factual inconsistency

---

## Paradigm 2 — RAG (FAISS Retrieval + LLM Reranking)

**Anchor paper:** K-RagRec (Wang et al., ACL 2025)
**Run date:** 2026-05-12
**Notebook:** `02_rag_recsys.ipynb`
**Users evaluated:** 200 (ranking + reranking), 100 (explanations)

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

### Explanation Quality (100 users)

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
| Latency (p95) | 61,708 ms |
| Total pipeline time (200 users) | ~100 min |

**Notes:**
- Full-catalog HR@K is flat at K=5/10/20 because the LLM reranker outputs a truncated list (~5 items)
- Shared-pool HR@10 of 28.3% outperforms Paradigm 1 (24.4%), showing LLM reranking adds value over BPR-MF
- Consistency = 1.0 is an artifact: consistency runs use a deterministic template, not the LLM
- Hallucination rate of 37% is slightly higher than Paradigm 1 (32%)

---

## Paradigm 3 — Generative (LLM-Native, GenRec-style)

**Anchor paper:** GenRec (Ji et al., ECIR 2024)

### 3a. Zero-Shot (no fine-tuning)

**Notebook:** `03_generative_recsys.ipynb`
**Run date:** 2026-05-12
**Model:** Qwen2.5-3B-Instruct, zero-shot, float16 on RTX 5050
**Users evaluated:** 200

| Metric | Full Catalog | Shared Pool |
|---|---|---|
| HR@5 | 0.0050 | 0.0150 |
| NDCG@5 | 0.0032 | 0.0150 |
| HR@10 | 0.0050 | 0.0150 |
| NDCG@10 | 0.0032 | 0.0150 |

- **Match rate:** 71.5% (143/200 users had at least one generated title matching the catalog)
- **Latency:** 24,487 ms/example (RTX 5050, max_new_tokens=200, do_sample=True)

### 3b. QLoRA Fine-Tuned

**Notebook:** `03_1_genrec_qlora_finetuning.ipynb`
**Run date:** 2026-05-12
**Model:** Qwen2.5-3B-Instruct, QLoRA (r=8, alpha=16), 4-bit on A100
**Training:** 7,288 examples, 3 epochs, batch=4, lr=3e-4
**Users evaluated:** 7,288 (full test set)

**Training loss:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.1691 | 0.0626 |
| 2 | 0.0613 | 0.0596 |
| 3 | 0.0562 | 0.0549 |

**Ranking:**

| Metric | Full Catalog | Shared Pool |
|---|---|---|
| HR@1 | 0.0266 | 0.0445 |
| HR@5 | 0.0266 | 0.0445 |
| HR@10 | 0.0266 | 0.0445 |
| NDCG@10 | 0.0266 | 0.0445 |

- **Match rate:** 93.4% (6,807/7,288 users had a generated title matching the catalog)
- **Latency:** 1,257 ms/example (A100, max_new_tokens=30, greedy decoding)

**Zero-shot → Fine-tuned improvement:**

| Metric | Zero-shot | Fine-tuned | Improvement |
|---|---|---|---|
| HR@10 (full catalog) | 0.0050 | 0.0266 | 5.3x |
| HR@10 (shared pool) | 0.0150 | 0.0445 | 3.0x |
| Catalog match rate | 71.5% | 93.4% | +21.9pp |

**Notes:**
- Fine-tuning improved HR@10 by 5.3x over zero-shot — the model learned to generate titles that exist in the catalog
- Match rate jumped from 71.5% to 93.4% — the model learned the catalog vocabulary
- HR is flat across K=1/5/10/20 because the model generates a single title (GenRec design); it either hits or misses
- Even fine-tuned, HR@10 of 2.7% (full catalog) is substantially below BPR-MF (2.2%) and RAG (5.0%) on full catalog — but this is expected for a 3B model on a 6K-item catalog; larger models and more training data would improve this
- Shared-pool HR@10 of 4.5% is below BPR-MF (24.4%) and RAG (28.3%) — the generative approach generates one title per user while the other paradigms score all 100 pool items, making the comparison structurally different
- Latency difference vs zero-shot (1.3s vs 24.5s) is partly due to max_new_tokens (30 vs 200) and hardware (A100 vs RTX 5050), not purely the paradigm

---

## Cross-Paradigm Summary

### Ranking Quality (Shared Pool — controlled comparison)

| Paradigm | Model | HR@10 | NDCG@10 |
|---|---|---|---|
| **RAG** | FAISS + Qwen2.5-3B rerank | **0.2829** | **0.1820** |
| **Explainable** | BPR-MF + Qwen2.5-3B | 0.2440 | 0.1440 |
| **Generative (fine-tuned)** | Qwen2.5-3B QLoRA | 0.0445 | 0.0445 |
| **Generative (zero-shot)** | Qwen2.5-3B | 0.0150 | 0.0150 |

### Explanation Quality

| Paradigm | Relevance (emb) | Specificity | Hallucination Rate |
|---|---|---|---|
| **RAG** | **0.602** | 0.631 | 0.369 |
| **Explainable** | 0.554 | **0.710** | **0.321** |
| **Generative** | N/A | N/A | N/A |

### System Metrics

| Paradigm | Ranking Latency | LLM Latency | Hardware |
|---|---|---|---|
| **Explainable** | **1.4 ms** | 20,449 ms/expl | RTX 5050 |
| **RAG** | — | 30,122 ms/user | RTX 5050 |
| **Generative (fine-tuned)** | — | 1,257 ms/user | A100 |
| **Generative (zero-shot)** | — | 24,487 ms/user | RTX 5050 |

---

## Key Takeaways

1. **RAG achieves the best ranking quality** (HR@10 28.3% on shared pool) — LLM reranking over retrieved candidates outperforms both collaborative filtering and pure generation.

2. **Explainable (BPR-MF) is the fastest for ranking** at 1.4 ms/user — orders of magnitude faster than LLM-based approaches. The LLM is only used for explanation generation, not ranking.

3. **Fine-tuning matters for generative recommendation.** QLoRA improved HR@10 by 5.3x over zero-shot and catalog match rate from 71.5% to 93.4%. Without fine-tuning, the LLM has no knowledge of the item catalog.

4. **Generative remains weakest on ranking** even after fine-tuning (HR@10 4.5% vs 28.3% for RAG). This is partly structural — GenRec generates a single title while the other paradigms score all candidates. It is also a function of model size (3B) and the inherent difficulty of generating exact titles from a large catalog.

5. **Hallucination is a real concern** across all LLM-using paradigms — 32-37% of explanations reference titles not in the user's history. This motivates the tutorial's emphasis on grounded explanation evaluation.

6. **Latency comparisons are not apples-to-apples** across paradigms — different hardware (RTX 5050 vs A100), different generation settings (max_new_tokens, sampling vs greedy), and different pipeline structures (score-all vs generate-one). The ranking latency for BPR-MF (1.4 ms, no LLM) is the only paradigm-intrinsic measurement.
