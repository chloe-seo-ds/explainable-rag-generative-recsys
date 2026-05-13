# Choosing Between Explainable, Retrieval-Augmented, and LLM-Native Recommenders in Text-Rich Domains

**CSCI E-222: Foundations of Large Language Models — Final Project Report**

**Authors:** Yejin Seo, Sarthak Baral

**Date:** May 2026

**Repository:** [github.com/chloe-seo-ds/explainable-rag-generative-recsys](https://github.com/chloe-seo-ds/explainable-rag-generative-recsys)

**Video Presentation:** [PLACEHOLDER]

---

## Abstract

Large language models are reshaping the design space of recommender systems, yet no practitioner-oriented comparison exists that places competing LLM-based paradigms side by side under a common evaluation framework. We present a controlled empirical comparison of three paradigms for text-rich recommendation: (1) collaborative-evidence explainable recommenders, which pair BPR matrix factorization with LLM-generated explanations; (2) retrieval-augmented recommenders, which combine dense FAISS retrieval with LLM reranking; and (3) LLM-native generative recommenders, which instruction-tune a language model to directly generate item titles. All three paradigms are benchmarked on a shared Amazon Books 2023 dataset with common evaluation code, identical temporal splits, and matched candidate pools. We evaluate ranking quality (HR@K, NDCG@K), explanation quality (relevance, specificity, consistency, hallucination rate), and system-level metrics (latency, hardware requirements). Our results show that the RAG paradigm achieves the best ranking quality (28.3% HR@10), BPR-MF offers the fastest ranking latency (1.4 ms/user), QLoRA fine-tuning improves the generative paradigm by 5.3x over zero-shot, and hallucination rates of 32--37% across LLM-using paradigms remain a deployment concern. A key methodological finding: the generative paradigm's HR@10 on the shared pool jumps 6× (from 4.45% to 26.67%) when we switch the evaluation protocol from title generation to log-likelihood scoring on the same model and same candidates, making it competitive with — and slightly above — BPR-MF. We conclude that paradigm choice depends on operational constraints: latency-sensitive settings favor collaborative filtering, quality-sensitive settings favor RAG, and the generative approach requires fine-tuning and a score-all-candidates evaluation protocol to be competitive.

---

## 1. Introduction and Problem Statement

Text-rich recommendation is a growing subfield of recommender systems. In domains such as books, news, scientific papers, and e-commerce, item descriptions, user reviews, and metadata carry rich signals that go beyond the sparse user-item interaction matrices that traditional collaborative filtering relies on. The emergence of large language models has created multiple viable ways to exploit these textual signals, but the practical choice among model families remains unclear.

Three distinct paradigms have emerged:

1. **Explainable recommenders** grounded in collaborative evidence, where a traditional collaborative filtering model produces rankings and a downstream LLM generates natural-language explanations conditioned on the user's history and item metadata.
2. **Retrieval-augmented recommenders**, which combine semantic dense retrieval with LLM reasoning or reranking — the LLM operates over a retrieved candidate set rather than the full catalog.
3. **LLM-native generative recommenders**, which use instruction-following and language generation as the primary recommendation interface — the model directly generates the recommended item title given a user's interaction history.

These paradigms differ not only in offline ranking performance but also in explanation quality, controllability, serving cost, and deployment complexity. Existing work typically evaluates each paradigm in isolation, making it difficult for practitioners to reason about trade-offs. Each paradigm's anchor paper uses different datasets, different evaluation protocols, and different candidate construction strategies, making direct comparison impossible without reimplementation.

**Our contribution.** We implement all three paradigms on a shared Amazon Books 2023 benchmark with common evaluation code, identical data splits, and matched candidate pools. We evaluate ranking quality, explanation quality, and system-level metrics under controlled conditions. We anchor our implementations on three recent papers:

- **XRec** (Ma et al., EMNLP 2024) — collaborative ranking with LLM-generated explanations
- **K-RagRec** (Wang et al., ACL 2025) — knowledge-graph retrieval-augmented generation for LLM-based recommendation
- **GenRec** (Ji et al., ECIR 2024) — instruction-tuned LLM for direct next-item generation

This report presents the implementation details, experimental results, and practical lessons learned from this comparison.

---

## 2. Data Description and Preprocessing

### Source

We use the **Amazon Reviews 2023** dataset (Hou et al., 2024), specifically the Books category, accessed via HuggingFace (`McAuley-Lab/Amazon-Reviews-2023`). This dataset contains user reviews, ratings, timestamps, and item metadata (titles) for the Amazon Books catalog. We chose this dataset because it is natively text-rich (book titles and reviews provide strong textual signals) and is supported by all three anchor systems.

### Scale and Filtering

From the full Books category, we sample 20,000 users and apply iterative k-core filtering:

- **Minimum interactions:** 5 per user and 5 per item
- **Iterations:** 10 rounds of alternating user and item filtering
- **Final scale:** ~6,117 unique items after filtering

After filtering, the dataset is re-indexed to contiguous integer user and item indices, sorted by timestamp within each user.

### Temporal Split

We use a **leave-last-two-out** temporal split, following standard practice in sequential recommendation evaluation:

- **Training set:** All interactions except each user's last two
- **Validation set:** Each user's second-to-last interaction
- **Test set:** Each user's last interaction

This yields 7,288 test users (users with at least 3 interactions after filtering).

### Shared Candidate Pool

For controlled cross-paradigm comparison, we construct a shared candidate pool for each test user: **1 ground-truth item + 99 randomly sampled negative items**. All paradigms rank or score items from this same pool, ensuring that differences in HR@K and NDCG@K reflect model quality rather than candidate set construction.

### GenRec Instruction-Tuning Format

For the generative paradigm, we construct instruction-tuning examples from the training split. Each example consists of:

- An instruction drawn from a set of 10 paraphrased templates (e.g., "What book would complement this reading history")
- The user's reading history (book titles, chronologically ordered) as input
- The next item's title as the target output

This produces 7,288 training and 7,288 test examples.

---

## 3. Models and Methods

### 3.1 Paradigm 1 — Explainable (BPR-MF + LLM Explanation)

**Anchor paper:** XRec (Ma et al., EMNLP 2024)

This paradigm separates ranking from explanation. A traditional collaborative filtering model handles ranking; a downstream LLM generates human-readable explanations for the recommendations.

**Ranking model — BPR-MF.** We train a Bayesian Personalized Ranking matrix factorization model with 64-dimensional embeddings. Training uses pairwise ranking loss with random negative sampling: for each observed (user, positive item) pair, a negative item is sampled uniformly from the catalog. We train with validation-based early stopping on HR@10.

An implementation detail: the original negative sampler called `rng.choice(python_list)`, triggering a list-to-array conversion on every call. We vectorized this by pre-converting to numpy arrays, reducing training time from ~5 hours to ~15 minutes.

**LLM explanation — Qwen2.5-3B-Instruct.** For each recommended item, we prompt the LLM with the user's reading history (recent book titles) and the recommended item title, asking it to generate a natural-language explanation for why the user would enjoy this item. The model runs in float16 on an RTX 5050 (8 GB VRAM).

**Evaluation.** Ranking is evaluated on both the full catalog (~6,117 items) and the shared pool (100 candidates/user). Explanation quality is evaluated on 100 users using relevance, specificity, consistency, and hallucination metrics.

### 3.2 Paradigm 2 — RAG (FAISS Retrieval + LLM Reranking)

**Anchor paper:** K-RagRec (Wang et al., ACL 2025)

This paradigm uses dense retrieval to generate candidates and an LLM to rerank them.

**Dense retrieval.** We encode all item titles using the `all-MiniLM-L6-v2` sentence-transformer model and index the resulting 384-dimensional embeddings in a FAISS `IndexFlatIP` (inner-product) index. For each test user, we construct a query by mean-pooling the embeddings of the user's recent history titles, then retrieve the top-50 nearest items from the index.

**LLM reranking — Qwen2.5-3B-Instruct.** The top-50 retrieved candidates are formatted as a numbered list in a prompt. The LLM is asked to rerank them by relevance to the user's reading history, outputting a JSON-style ranked list of candidate indices. The response is parsed via regex; if parsing fails, the original retrieval order is used as a fallback.

**Evaluation.** Same metrics as Paradigm 1. We evaluate 200 users for ranking and 100 users for explanation quality.

### 3.3 Paradigm 3 — Generative (GenRec-style)

**Anchor paper:** GenRec (Ji et al., ECIR 2024)

This paradigm uses the LLM as both the recommender and the ranking model — it directly generates the recommended item title given the user's interaction history.

**Zero-shot.** Qwen2.5-3B-Instruct is prompted with the user's reading history and asked to generate a book recommendation. The generated title is extracted and matched against the item catalog using `difflib.get_close_matches` with a 0.7 similarity threshold. The model runs in float16 on an RTX 5050 with `max_new_tokens=200` and sampling enabled. We evaluate 200 users.

**QLoRA fine-tuned.** We fine-tune Qwen2.5-3B-Instruct using QLoRA (4-bit NF4 quantization, LoRA r=8, alpha=16) on the q_proj and v_proj attention matrices. Training uses 7,288 examples over 3 epochs with batch size 4, gradient accumulation 8, and learning rate 3e-4. The fine-tuned model runs with `max_new_tokens=30` and greedy decoding on a Colab A100 (40 GB). Training follows a manual loop with AdamW optimizer and per-epoch validation loss tracking. We evaluate all 7,288 test users.

**Log-likelihood scoring.** The zero-shot and fine-tuned generative models produce a single title per user, creating a structural disadvantage compared to paradigms that score all 100 pool items. To enable a fairer comparison, we implement forward-pass log-likelihood scoring: for each test user, the model computes the conditional log-probability of each candidate title given the user's history, and candidates are ranked by their scores.

> **[PENDING]** Log-likelihood evaluation results (notebook 03_2) are currently running and will be added to Section 5 when available.

**Title matching.** Generated titles are matched to the item catalog using `difflib.get_close_matches` with a 0.7 similarity threshold. If no match is found, the prediction is treated as a miss.

### 3.4 Evaluation Framework

**Ranking metrics.** We use Hit Rate at K (HR@K) and Normalized Discounted Cumulative Gain at K (NDCG@K), following standard leave-last-one-out evaluation. For each test user, there is a single ground-truth item; HR@K is 1 if the ground-truth item appears in the top-K predictions, and 0 otherwise. NDCG@K additionally accounts for the rank position.

**Explanation quality metrics.** We evaluate four dimensions of explanation quality, operationalized as follows:

- **Relevance (lexical):** Fraction of user history titles mentioned verbatim in the explanation.
- **Relevance (embedding):** Cosine similarity between the sentence-transformer encoding of the explanation and the concatenated user history titles.
- **Specificity:** A composite score measuring whether the explanation references the recommended item by name and provides sufficient detail (penalizing short or generic explanations).
- **Consistency (lexical):** Mean pairwise Jaccard similarity across 3 repeated explanations for the same input (evaluated on 20 users).
- **Consistency (embedding):** Mean pairwise cosine similarity of sentence-transformer encodings across repeated explanations.
- **Hallucination rate:** Fraction of book titles mentioned in the explanation that are neither the recommended item nor in the user's interaction history. Uses word-boundary matching with a minimum title length of 4 characters to reduce false positives.

These metrics draw on the evaluation dimensions surveyed by Zhang and Chen (2020) and operationalized with embedding-based evaluation (Zhang et al., 2020).

**System metrics.** Latency per user (mean and p95), measured per-component. We note the hardware context for each measurement, as the paradigms were evaluated on different hardware.

---

## 4. Implementation Details

### Repository Structure

```
explainable-rag-generative-recsys/
├── Notebooks/
│   ├── 00_data_preparation.ipynb      # Download, filter, and split Amazon Books
│   ├── 01_explainable_recsys.ipynb    # Paradigm 1: BPR-MF + LLM explanations
│   ├── 02_rag_recsys.ipynb            # Paradigm 2: FAISS retrieval + LLM reranking
│   ├── 03_generative_recsys.ipynb     # Paradigm 3: Zero-shot generation
│   ├── 03_1_genrec_qlora_finetuning.ipynb  # Paradigm 3: QLoRA fine-tuning (Colab)
│   ├── 03_2_genrec_likelihood_eval.ipynb   # Paradigm 3: Log-likelihood scoring
│   ├── 04_unified_evaluation.ipynb    # Cross-paradigm comparison tables and charts
│   └── tutorial_utils.py              # Shared metrics, data loading, and I/O utilities
├── recsys26_tutorial_proposal.tex     # ACM-formatted tutorial proposal
├── RESULTS.md                         # All experimental results
├── LLM_INTEGRATION.md                 # Hardware/model selection design record
└── README.md
```

### Dependencies

Core libraries: `torch`, `transformers`, `peft` (QLoRA), `sentence-transformers` (dense retrieval and embedding-based metrics), `faiss-cpu` (approximate nearest-neighbor search), `datasets` (Amazon Reviews 2023 loading), `bitsandbytes` (4-bit quantization for QLoRA), `accelerate` (device mapping).

### Hardware

| Component | Hardware | Use |
|---|---|---|
| NB 00--03 | NVIDIA RTX 5050 Laptop GPU (8 GB, Blackwell) | Data prep, BPR training, inference |
| NB 03.1 | Google Colab A100 (40 GB) | QLoRA fine-tuning and fine-tuned inference |

### Model Selection: Why Qwen2.5-3B-Instruct

Our original plan was to use LLaMA-3-8B-Instruct with 4-bit quantization. Two blocking issues forced a model change:

1. **bitsandbytes/Blackwell incompatibility.** `bitsandbytes` 0.49.2 has no CUDA kernels compiled for the Blackwell architecture (sm_120). Attempting 4-bit quantization on the RTX 5050 triggers a CUDA kernel crash surfacing as a spurious out-of-memory error. The 8B model requires ~16 GB in fp16 and cannot fit in 8 GB VRAM without quantization.

2. **LLaMA-3.2-3B license gate.** Meta introduced a separate license agreement for the 3.2 family. Existing LLaMA 3.0/3.1 access does not carry over — a separate form submission is required and HuggingFace returns HTTP 403 until approved.

We selected **Qwen2.5-3B-Instruct** (Apache 2.0 license) because it fits in 8 GB VRAM in float16 (~6 GB loaded, ~2 GB headroom), requires no quantization, downloads without approval, and uses the same `apply_chat_template()` interface.

### How to Run

Notebooks execute in order: `00 → 01 → 02 → 03 → (03.1 on Colab) → (03.2) → 04`. Two execution modes are supported:

- **Full mode:** Requires GPU. Runs all LLM inference and training from scratch. Results are cached after first run.
- **Demo mode:** Uses precomputed checkpoints and cached LLM outputs. Each notebook checks for cached outputs automatically.

### Caching

Pre-computed embeddings, LLM outputs, and inference checkpoints are cached using pickle serialization. The QLoRA notebook implements periodic checkpointing every 500 examples, enabling resilience to kernel restarts during long inference runs.

---

## 5. Experiments and Results

### 5.1 Cross-Paradigm Ranking (Shared Pool — Main Comparison)

The shared candidate pool (1 ground-truth + 99 negatives per user) provides a controlled comparison across all paradigms.

| Paradigm | Model | Eval Method | N | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|---|
| **RAG** | FAISS + Qwen2.5-3B rerank | reranking | 200 | 0.2091 | 0.1584 | **0.2829** | **0.1820** | **0.4108** | **0.2140** |
| **Generative (fine-tuned)** | Qwen2.5-3B QLoRA | log-likelihood | 2,820 | 0.1706 | 0.1221 | 0.2667 | 0.1530 | 0.4085 | 0.1885 |
| **Explainable** | BPR-MF | dot-product scoring | 7,288 | 0.1679 | 0.1197 | 0.2440 | 0.1440 | 0.3670 | 0.1749 |
| **Generative (fine-tuned)** | Qwen2.5-3B QLoRA | title generation | 7,288 | 0.0445 | 0.0445 | 0.0445 | 0.0445 | 0.0445 | 0.0445 |
| **Generative (zero-shot)** | Qwen2.5-3B | title generation | 200 | 0.0150 | 0.0150 | 0.0150 | 0.0150 | 0.0150 | 0.0150 |

Under a fair evaluation (all three paradigms scoring the same 100 candidates), the ranking is **RAG > Generative (log-likelihood) > Explainable (BPR-MF)**. The fine-tuned generative model with log-likelihood scoring (HR@10 = 0.2667) actually *beats* BPR-MF (0.2440), and only RAG retains a clear advantage. The two generative rows together make the central methodological point: the same model produces a 6× difference in HR@10 (0.0445 → 0.2667) depending solely on whether we evaluate it by title generation or by log-likelihood scoring. The "title generation" row's flat scores across K reflect that the model emits a single title — it either hits or misses regardless of K.

### 5.2 Full-Catalog Ranking (Per-Paradigm)

| Paradigm | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| **Explainable** (BPR-MF, 7,288 users) | 0.0158 | 0.0101 | 0.0220 | 0.0121 | 0.0325 | 0.0148 |
| **RAG** (200 users) | 0.0500 | 0.0293 | 0.0500 | 0.0293 | 0.0500 | 0.0293 |
| **Generative fine-tuned** (7,288 users) | 0.0266 | 0.0266 | 0.0266 | 0.0266 | 0.0266 | 0.0266 |
| **Generative zero-shot** (200 users) | 0.0050 | 0.0032 | 0.0050 | 0.0032 | — | — |

Note: The RAG paradigm's full-catalog HR is flat at K=5/10/20 because the LLM reranker outputs a truncated list (~5 items). BPR-MF's full-catalog HR@10 of 2.2% is expected for a matrix factorization model over ~6K items without reranking.

We do not extend log-likelihood scoring to the full catalog. Doing so would require ~6,117 forward passes per user (~75 seconds per user even with prompt-KV reuse, ~150 hours total for the full test set), which is not tractable. More importantly, title generation against the full catalog is the standard generative-recommender evaluation in the GenRec anchor paper, and each paradigm has a different natural full-catalog mechanism (dot-product, retrieve-then-rerank, generate). Cross-paradigm full-catalog comparison is therefore inherently asymmetric; the shared-pool result is where the controlled comparison lives.

### 5.3 Explanation Quality

| Metric | Explainable (BPR-MF + LLM) | RAG (FAISS + LLM) | Generative |
|---|---|---|---|
| Relevance (lexical) | 0.264 | 0.222 | N/A |
| Relevance (embedding) | 0.554 | **0.602** | N/A |
| Specificity | **0.710** | 0.631 | N/A |
| Consistency (lexical) | 0.358 | 1.000* | N/A |
| Consistency (embedding) | 0.861 | 1.000* | N/A |
| Hallucination rate | **0.321** | 0.369 | N/A |

\* RAG consistency = 1.0 is an artifact: consistency runs use a deterministic template rather than the LLM, so all repeated outputs are identical.

The explainable paradigm produces more specific explanations (0.710 vs 0.631) and has a lower hallucination rate (32.1% vs 36.9%). The RAG paradigm achieves higher embedding-based relevance (0.602 vs 0.554). The generative paradigm does not produce per-item explanations and is not evaluated on these metrics.

### 5.4 System Metrics

| Paradigm | Ranking Latency | LLM Latency | Hardware |
|---|---|---|---|
| **Explainable** | **1.4 ms/user** | 20,449 ms/explanation (p95: 45,428 ms) | RTX 5050 |
| **RAG** | — | 30,122 ms/user (p95: 61,708 ms) | RTX 5050 |
| **Generative (fine-tuned, title gen)** | — | 1,257 ms/user | A100 |
| **Generative (fine-tuned, log-likelihood)** | — | 15,070 ms/user | L4 |
| **Generative (zero-shot)** | — | 24,487 ms/user | RTX 5050 |

Log-likelihood scoring is ~12× slower per user than title generation on the same model because it runs 100 forward passes (one per candidate) instead of a single 30-token greedy generation. This is the fair-comparison cost: the structural fix that unlocks the 6× HR@10 improvement comes with a ~12× latency hit.

**Important caveat:** Latency comparisons are not apples-to-apples across paradigms. Different hardware (RTX 5050 vs A100), different generation settings (max_new_tokens: 200 vs 30, sampling vs greedy), and different pipeline structures (score-all vs generate-one) all affect the numbers. The BPR-MF ranking latency of 1.4 ms is the only paradigm-intrinsic measurement — it does not involve any LLM call.

### 5.5 Fine-Tuning Impact

QLoRA fine-tuning dramatically improves the generative paradigm:

| Metric | Zero-shot | Fine-tuned (QLoRA) | Improvement |
|---|---|---|---|
| HR@10 (full catalog) | 0.0050 | 0.0266 | **5.3x** |
| HR@10 (shared pool) | 0.0150 | 0.0445 | **3.0x** |
| Catalog match rate | 71.5% | 93.4% | +21.9 pp |

The match rate improvement (71.5% → 93.4%) shows that fine-tuning teaches the model the item catalog's vocabulary — the zero-shot model frequently generates plausible but non-existent titles, while the fine-tuned model generates titles that exist in the catalog 93% of the time.

### 5.6 Training Curves

QLoRA fine-tuning converges within 3 epochs:

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.1691 | 0.0626 |
| 2 | 0.0613 | 0.0596 |
| 3 | 0.0562 | 0.0549 |

The large drop from epoch 1 to epoch 2 (0.1691 → 0.0613) indicates rapid adaptation to the recommendation task. The small gap between train and validation loss by epoch 3 (0.0562 vs 0.0549) suggests the model is not overfitting — the val loss is actually slightly lower than train loss, likely due to the limited 500-example validation sample.

### 5.7 Qualitative Examples

Sample outputs from the fine-tuned generative model:

| User | Generated Title | Ground Truth | Hit? |
|---|---|---|---|
| 2 | The Nightingale: A Novel | The Library at Mount Char: A Novel | No |
| 4 | The 7 Habits of Highly Effective People | Essentialism: The Disciplined Pursuit of Less | No |
| 6 | The Very Hungry Caterpillar | The Book with No Pictures | No |
| 16 | The Silent Patient | Lincoln in the Bardo: A Novel | No |
| 18 | The Last Word (A Jack Noble Thriller) | Noble Intentions: A Jack Noble Thriller | No |
| 19 | The Last Man: A Novel | Tier One (Tier One Thrillers Book 1) | No |

The model generates genre-appropriate titles (thrillers for thriller readers, children's books for children's book readers) but struggles to predict the exact next item. This is consistent with the 2.66% full-catalog HR@10 — the model has learned meaningful genre preferences but the task of predicting the exact next title from a 6K-item catalog is inherently difficult for a 3B-parameter model.

### 5.8 Log-Likelihood Evaluation (Generative, Shared Pool)

To address the structural unfairness of comparing single-title generation against score-all-candidates ranking, we re-evaluate the fine-tuned generative model with **log-likelihood scoring** over the same 100-candidate shared pool. For each user and each candidate, we compute the model's conditional log-probability of the candidate title given the user's history (a single forward pass per candidate, no autoregressive generation). Candidates are ranked by mean per-token log-probability.

Evaluated on **N = 2,820 users** (subset of the 7,288-user test set; user ordering is effectively random with respect to the recommendation task — see Section 7 limitations). Binomial CI half-width at this N is ≈±1.6 percentage points for HR@10, tighter than every other LLM-paradigm evaluation in the report.

**Hardware:** NVIDIA L4 (Colab), 15,070 ms/user including 100 forward passes.

**Results:**

| Metric | Log-likelihood (N=2,820) | Title generation (N=7,288) | Ratio |
|---|---|---|---|
| HR@1 | 0.0745 | 0.0445 | 1.7× |
| HR@5 | 0.1706 | 0.0445 | 3.8× |
| HR@10 | **0.2667** | 0.0445 | **6.0×** |
| HR@20 | 0.4085 | 0.0445 | 9.2× |
| NDCG@5 | 0.1221 | 0.0445 | — |
| NDCG@10 | 0.1530 | 0.0445 | — |
| NDCG@20 | 0.1885 | 0.0445 | — |

The contrast between the two evaluation methods is the headline finding: the same fine-tuned model produces **6× higher HR@10** on the same pool depending on how we ask it to rank. When asked to *generate* the correct title from a 100-item pool, it succeeds 4.45% of the time. When asked to *score* the same 100 titles by log-likelihood and rank them, it succeeds 26.67% of the time. This confirms that the generative paradigm's apparent weakness on shared-pool ranking metrics is largely a measurement artifact of the title-generation evaluation protocol — the model can reliably identify good candidates even when it cannot reliably generate the exact title.

Under log-likelihood scoring, the generative paradigm (0.2667) is competitive with — and slightly above — BPR-MF (0.2440), though still below RAG (0.2829). The full shared-pool ranking is **RAG > Generative (LL) > BPR-MF**, a meaningfully different conclusion from the title-generation-based ranking that placed generative far behind both baselines.

**Why we did not extend this to the full catalog.** Log-likelihood scoring over the full 6,117-item catalog would require ~6,117 forward passes per user (~75 seconds/user even with prompt-KV reuse, ~150 hours total). This was not feasible within the project timeline. Full-catalog title generation (Section 5.2) remains the natural full-catalog metric for generative recommenders and is consistent with the GenRec anchor paper.

---

## 6. Discussion

### What Worked Well

**RAG achieves the best ranking quality.** The combination of dense retrieval (FAISS with sentence-transformer embeddings) and LLM reranking produces the highest HR@10 (28.3%) on the shared pool. This suggests that LLM reranking adds meaningful value over both pure collaborative filtering (24.4%) and pure generation (4.5%). The retrieval stage narrows the candidate space to semantically relevant items, and the LLM applies nuanced reasoning over this focused set.

**BPR-MF ranking is extremely fast.** At 1.4 ms per user, collaborative filtering is orders of magnitude faster than any LLM-based approach. In latency-constrained production environments, this matters — the LLM can be reserved for explanation generation (a post-hoc, potentially asynchronous step) rather than ranking.

**Fine-tuning dramatically improves generative recommendation.** QLoRA fine-tuning improved HR@10 by 5.3x over zero-shot and raised the catalog match rate from 71.5% to 93.4%. This confirms that without fine-tuning, the LLM has no knowledge of the item catalog — it generates plausible titles that often do not exist in the catalog. Fine-tuning on (history → next item) pairs teaches the model item transition patterns and catalog vocabulary.

**Shared candidate pools enable controlled comparison.** Constructing a common pool of 100 candidates per user was essential for fair cross-paradigm comparison. Without it, differences in candidate set construction would confound the ranking metric comparison.

**Caching infrastructure makes notebooks reproducible.** Pre-computed embeddings, LLM outputs, and inference checkpoints allow all notebooks to run in "demo mode" without GPU hardware, enabling reproducibility verification.

### What Did Not Work / Challenges

**Generative weakness on shared pool was largely a measurement artifact.** Under the standard title-generation evaluation, the fine-tuned generative model achieves only 4.5% HR@10 on the shared pool — far below RAG (28.3%) and BPR-MF (24.4%). But when re-evaluated with log-likelihood scoring over the same 100-candidate pool (Section 5.8), HR@10 rises to **26.7%** — a 6× improvement. With this fair evaluation, the generative paradigm actually outperforms BPR-MF and is only narrowly behind RAG. The structural mismatch between "generate one title" and "score all candidates" was driving most of the apparent weakness, not the underlying recommendation ability.

**Hallucination rates of 32--37% are concerning.** Across both LLM-using paradigms (explainable and RAG), roughly one in three explanations mentions book titles not in the user's actual history. This would be unacceptable in a user-facing deployment without post-hoc filtering or grounding mechanisms.

**bitsandbytes/Blackwell GPU incompatibility.** The RTX 5050's Blackwell architecture (sm_120) is not supported by bitsandbytes 0.49.2, blocking our original plan to use LLaMA-3-8B-Instruct with 4-bit quantization. This forced a switch to the smaller Qwen2.5-3B-Instruct model. The spurious out-of-memory error (rather than a clear incompatibility message) added debugging time.

**Latency comparison is not apples-to-apples.** Different hardware (RTX 5050 vs A100), generation settings (max_new_tokens, sampling vs greedy), and pipeline structures make cross-paradigm latency comparison approximate rather than definitive. A fair latency comparison would require all paradigms on the same hardware with the same generation budget.

**RAG consistency = 1.0 is an artifact.** The consistency metric for the RAG paradigm uses a deterministic template for repeated runs rather than re-invoking the LLM. This produces perfect consistency but does not measure actual LLM consistency. A proper measurement would require multiple stochastic LLM runs, which was infeasible within the compute budget.

### Lessons Learned

**Paradigm choice depends on constraints.** There is no single best paradigm — the right choice depends on the deployment context:

- **Latency-constrained settings:** BPR-MF ranking (1.4 ms) with optional asynchronous LLM explanations
- **Quality-constrained settings:** RAG retrieval + LLM reranking (28.3% HR@10)
- **Flexibility-constrained settings:** Generative approach with fine-tuning, especially when the system needs natural-language interaction or when the item catalog changes frequently

**Evaluation methodology matters — sometimes more than the model.** Our generative paradigm went from 4.45% HR@10 to 26.67% on the same data with the same model by changing only the evaluation protocol (title generation → log-likelihood scoring). This is a methodological finding as much as an empirical one: shared-pool comparisons across paradigms must use a scoring protocol the model can actually exploit, not impose an interface (single-title generation) on a model that can naturally produce scores for any candidate. Log-likelihood is the structurally fair generative analogue of BPR-MF's dot-product score and RAG's reranker logit. A practitioner who only saw the title-generation number would conclude that fine-tuned generative recommendation is non-viable; a practitioner who saw the log-likelihood number would conclude it is competitive with classical CF.

**Fine-tuning is not optional for generative recommendation.** Zero-shot performance is effectively unusable (HR@10 = 0.5% on full catalog). Fine-tuning is a prerequisite, not an enhancement.

---

## 7. Limitations and Responsible Use

**Hallucination.** 32--37% of LLM-generated explanations cite book titles not in the user's actual history. In a production setting, serving these explanations without filtering could mislead users and erode trust. Any deployment of LLM-generated explanations should include post-hoc hallucination checking against the user's actual interaction data.

**Bias.** The Amazon Books dataset reflects popularity bias inherent to the Amazon marketplace. Bestsellers and mainstream titles dominate the interaction data. Recommendations for underrepresented genres, authors, and languages will be systematically worse. Any deployment should monitor recommendation diversity and consider fairness-aware re-ranking.

**Model size.** Qwen2.5-3B-Instruct is a small model by current standards. Production systems would likely use larger models (7B--70B) with correspondingly better instruction following, reduced hallucination, and improved ranking quality. Our results represent a lower bound on what LLM-based recommenders can achieve.

**Single dataset.** All results are from Amazon Books 2023. The relative performance of paradigms may differ in other domains (e.g., movies, music, news, e-commerce products with shorter text descriptions). Cross-domain validation was beyond the scope of this project.

**Log-likelihood evaluation subset (N=2,820).** The shared-pool log-likelihood result for the generative paradigm is computed on 2,820 of the 7,288 test users due to compute budget (~19 additional hours of L4 GPU time would have been needed to complete the full set; at 15.07 s/user × 4,468 remaining ≈ 18.7 hours). The 2,820 users are not a strictly randomized sample — they are the first 2,820 user_idx values, which were assigned in alphabetical order of the underlying Amazon `user_id` hash strings after a random 20K-user subsample. There is no obvious mechanism by which this ordering would bias recommendation difficulty, but we cannot rule out subtle effects. The 95% binomial CI half-width for HR@10 at N=2,820 is ≈±1.6 percentage points — so the 0.2667 result is statistically distinguishable from BPR-MF's 0.2440 (computed on 7,288 users) but not from RAG's 0.2829 (computed on 200 users, CI half-width ≈±6.2 pp).

**No user study.** All evaluation is offline and automatic. We do not measure whether users find the generated explanations helpful, whether they trust the recommendations more with explanations, or whether the explanations change user behavior. Offline explanation quality metrics are imperfect proxies for real-world utility.

**Evaluation gaps.** No unified benchmark standard exists for explanation quality in recommendation. Our metrics (relevance, specificity, consistency, hallucination rate) are operationalizations of reasonable desiderata, but the field lacks consensus on how to measure explanation quality. Different operationalizations may yield different conclusions.

---

## 8. References

1. Ma, Q., Ren, X., and Huang, C. (2024). XRec: Large Language Models for Explainable Recommendation. In *Proceedings of EMNLP 2024*.

2. Wang, S., Dang, Y., Zhang, G., Wang, Y., Liu, Q., Wu, S., and Wang, L. (2025). Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation. In *Proceedings of ACL 2025*.

3. Ji, J., Li, Z., Xu, S., Hua, W., Ge, Y., Tan, J., and Zhang, Y. (2024). GenRec: Large Language Model for Generative Recommendation. In *Proceedings of ECIR 2024*.

4. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., and McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. *arXiv preprint arXiv:2403.03952*.

5. Zhang, Y. and Chen, X. (2020). Explainable Recommendation: A Survey and New Perspectives. *Foundations and Trends in Information Retrieval*, 14(1), 1--101.

6. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. In *Proceedings of ICLR 2020*.

7. Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of EMNLP 2019*.

8. Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In *NeurIPS 2020*.

---

## Appendix

### A. Video Presentation

[PLACEHOLDER — link to be added]

### B. GitHub Repository

[github.com/chloe-seo-ds/explainable-rag-generative-recsys](https://github.com/chloe-seo-ds/explainable-rag-generative-recsys)

### C. Notebook Execution Order

| Step | Notebook | Description |
|---|---|---|
| 1 | `00_data_preparation.ipynb` | Download, filter, and split Amazon Books |
| 2 | `01_explainable_recsys.ipynb` | Paradigm 1: BPR-MF + LLM explanations |
| 3 | `02_rag_recsys.ipynb` | Paradigm 2: FAISS retrieval + LLM reranking |
| 4 | `03_generative_recsys.ipynb` | Paradigm 3: Zero-shot generation |
| 5 | `03_1_genrec_qlora_finetuning.ipynb` | Paradigm 3: QLoRA fine-tuning (Colab A100) |
| 6 | `03_2_genrec_likelihood_eval.ipynb` | Paradigm 3: Log-likelihood scoring |
| 7 | `04_unified_evaluation.ipynb` | Cross-paradigm comparison tables and charts |

### D. Scale Caps for LLM Inference

| Notebook | Variable | Value | Rationale |
|---|---|---|---|
| NB01 | `N_EXPLAIN` | 100 users | ~8 min at ~5s/explanation |
| NB02 | `N_RERANK` | 200 users | ~20 min at ~6s/rerank call |
| NB03 | `N_GENERATE` | 200 examples | ~20 min at ~6s/generation call |
