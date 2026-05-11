# Repository Analysis & Conference Tutorial Expert Recommendations

## Context

This document captures a full analysis of the `explainable-rag-generative-recsys` repository — a tutorial proposal and implementation for RecSys 2026 — and provides next-step recommendations from the perspective of an experienced conference tutorial instructor.

The proposal is well-motivated and structurally sound. The primary gap is that the implementation is a **template skeleton**, not a runnable system: LLM integration is commented out across all three paradigms, no results have been generated, and several infrastructure pieces needed for a public tutorial are missing.

---

## Repository State: What Exists

### Proposal Document (`recsys26_tutorial_proposal.tex`)
**Status: Complete and polished.**

- Title: "Choosing Between Explainable, Retrieval-Augmented, and LLM-Native Recommenders in Text-Rich Domains"
- Authors: Yejin Seo (LegalZoom / Harvard), Sarthak Baral (Harvard), Kaylee Vo (Harvard)
- Format: Half-day (3 hours + coffee break), two 90-minute blocks
- Anchor systems: XRec (EMNLP 2024), K-RagRec (ACL 2025), GenRec (ECIR 2024)
- Benchmark: Amazon Books 2023
- Clear differentiation from 5 prior RecSys tutorials via feature comparison table

### Notebooks

| Notebook | Purpose | Status |
|---|---|---|
| `00_data_preparation.ipynb` | Download, k-core filter, split Amazon Books | Complete |
| `01_explainable_recsys.ipynb` | BPR-MF ranking + LLM explanation (XRec) | Partial — LLM disabled (placeholder) |
| `02_rag_recsys.ipynb` | FAISS retrieval + LLM reranking (K-RagRec) | Partial — LLM disabled (retrieval-only fallback) |
| `03_generative_recsys.ipynb` | QLoRA fine-tuning / API generation (GenRec) | Stub — random placeholder output |
| `04_unified_evaluation.ipynb` | Cross-paradigm comparison tables | Complete (needs upstream results) |

### `tutorial_utils.py`
**Status: Complete.** Contains fully implemented ranking metrics (HR@K, NDCG@K, Recall@K), explanation quality metrics (relevance, specificity, consistency, hallucination rate), latency tracking, and results I/O utilities.

### Missing Infrastructure
- No `requirements.txt` or `environment.yml`
- No `data/` or `results/` directories committed
- No precomputed checkpoints or cached LLM outputs
- Dependencies scattered across individual notebook `!pip install` cells with no version pinning

---

## Expert Analysis: Strengths

1. **Strong conceptual differentiation.** The multi-paradigm comparative framing is genuinely rare in the RecSys tutorial space; the proposal correctly identifies the gap left by single-paradigm tutorials at RecSys 2023–2025.

2. **Rigorous evaluation design.** Covering ranking metrics, explanation quality metrics (with 4 dimensions including hallucination rate), AND system metrics (latency, cost, complexity) in a unified framework is the core intellectual contribution. Few tutorials go this deep on operationalizing explanation quality.

3. **Reproducibility-first architecture.** The shared data splits, common candidate pools, and modular utility module show strong engineering instincts. The design — where LLM calls are opt-in and placeholders exist — is smart for a demo environment.

4. **Data pipeline is solid.** Notebook 00 uses k-core filtering, leave-last-two-out temporal splitting, and generates all downstream artifacts in a single reproducible run. This is tutorial-grade data hygiene.

5. **Unified evaluation notebook (04) is well-designed.** LaTeX table export and graceful missing-file handling is exactly what you need for live demos.

---

## Expert Recommendations: Next Steps

These are ordered by priority for RecSys 2026 submission and tutorial delivery.

---

### Priority 1 — Generate Real Results (Critical for Submission)

The proposal claims benchmarks on Amazon Books across all three paradigms, but no results exist yet. Reviewers may ask for preliminary numbers.

**Paradigm 1 (Explainable):** BPR-MF is already implemented. Run it end-to-end, save `results/explainable_results.json`. For explanations, enable one of the two LLM paths (OpenAI API is the fastest path; `gpt-4o-mini` cost for ~100 users is negligible). Even 100-user pilot results are enough for a proposal.

**Paradigm 2 (RAG):** FAISS retrieval is already implemented. The retrieval-only baseline already produces ranking metrics. Enable `rerank_api()` for a subset of users to show the ranking gain from LLM reranking.

**Paradigm 3 (Generative):** This is the highest-effort item. The QLoRA path requires a GPU. The fastest path is to enable the OpenAI zero-shot generation (`generate_recommendation_api()`) on a 50–100 user sample. Random placeholder output will produce near-zero HR@10; without real results, Paradigm 3 cannot be compared fairly.

**Recommendation:** Run all three paradigms with OpenAI API (gpt-4o-mini) on 200 test users. This produces publishable preliminary numbers, costs ~$5–15, and takes one afternoon.

---

### Priority 2 — Create `requirements.txt` (Critical for Reproducibility)

Dependencies are currently scattered across notebook cells. Attendees at a conference tutorial have 5 minutes to get set up; a missing package at minute 3 kills the session.

**Packages to include (with version pins):**
```
torch>=2.1.0
transformers>=4.37.0
sentence-transformers>=2.5.0
faiss-cpu>=1.7.4
pandas>=2.0.0
numpy>=1.26.0
datasets>=2.16.0
tqdm>=4.66.0
openai>=1.10.0
bitsandbytes>=0.41.0
peft>=0.7.0
jupyter>=1.0.0
ipykernel>=6.25.0
```

Also add an `environment.yml` for conda users.

---

### Priority 3 — Complete Paradigm 3 Notebook (Critical for Tutorial Credibility)

Notebook 03 currently returns random items. This is the paradigm the audience will be most curious about. Even a zero-shot OpenAI baseline with real output — even if it underperforms BPR-MF — demonstrates the approach honestly.

**Minimum viable fix:** Uncomment and wire up the OpenAI generation path. Add a clear comment explaining the QLoRA path requires GPU and is pre-trained for the tutorial demo. Provide a checkpoint download link or precomputed `generative_predictions.pkl`.

**Longer-term fix:** Run QLoRA fine-tuning on LLaMA-2-7B (or LLaMA-3-8B) and upload the adapter weights to HuggingFace Hub. Attendees then load the adapter without needing to train.

---

### Priority 4 — Precompute Artifacts for Tutorial Day (Critical for Live Demo)

The proposal says training-heavy steps are precomputed and cached outputs are provided. This infrastructure does not yet exist.

**What to precompute and host (HuggingFace Hub or Zenodo — too large for git):**
- `data/amazon_books_processed.parquet` + `data/splits.pkl` + `data/shared_data.pkl`
- `data/faiss_index.bin` — pre-built FAISS index for item embeddings
- `results/bprmf_checkpoint.pt` — trained BPR-MF model weights
- `results/genrec_qlora/final/` — QLoRA adapter weights
- `results/cached_llm_outputs/` — pre-generated explanations and reranker outputs for 200 test users

**Implementation pattern:** Each notebook should check for cached outputs first and fall back to live generation only if cache is missing.

---

### Priority 5 — Tighten Explanation Metrics (Important for Academic Rigor)

The current implementations are heuristic-based (lexical overlap, title length). For a RecSys tutorial, these should be stronger.

**Relevance:** Switch from lexical overlap to BERTScore (already cited in the proposal bibliography). Compute BERTScore between the explanation and the concatenated user history.

**Hallucination rate:** The current implementation checks if mentioned book titles exist in user history. This misses factual hallucinations (wrong author, wrong genre). Consider adding an NLI-based entailment check (e.g., `cross-encoder/nli-deberta-v3-small`) as an optional stronger metric.

**Consistency:** The `explanation_consistency()` function in `tutorial_utils.py` is implemented but never called in any notebook. Wire it up in Notebooks 01 and 02 — generate 3 explanations per user for 20 users and compute pairwise Jaccard. This produces a compelling cross-paradigm insight.

---

### Priority 6 — Slide Deck Preparation (Required Before Tutorial)

The proposal promises a PDF slide deck. No slides exist yet. A half-day tutorial at RecSys needs ~80–100 slides across two blocks.

**Recommended structure:**

Block A (slides 1–50):
- Title + instructor intro (5 slides)
- Why text-rich recommendation matters — motivating examples (8 slides)
- Where CF breaks — sparse data / cold start (5 slides)
- Embedding and retrieval foundations — with diagrams (12 slides)
- Paradigm 1: XRec — architecture diagram, data flow, live notebook teaser (15 slides)
- Coffee break

Block B (slides 51–100):
- Paradigm 2: K-RagRec — two-stage pipeline diagram, reranking demo (15 slides)
- Paradigm 3: GenRec — instruction-tuning format, generation output examples (12 slides)
- Results: Comparison tables + interpretation (10 slides)
- Decision framework — "when to use which" flowchart (5 slides)
- Open challenges and Q&A (5 slides)

**Key visuals to create:**
- Architecture diagrams for each paradigm side by side
- A "when to use which" decision flowchart
- Comparison tables showing real metric numbers across all three paradigms

---

### Priority 7 — Stretch Goals (Nice-to-Have)

**A. Cold-start experiment.** Filter test users to those with fewer than 10 interactions. Show how BPR-MF degrades while RAG and GenRec hold up better. This directly supports the decision framework section.

**B. Cost tracking in Notebook 04.** Log token usage from OpenAI API calls in Notebooks 01–03 and include a "cost per 1K recommendations" row in the comparison table. This is the most practically relevant system metric for ML engineers.

**C. Smaller/cheaper LLM option.** `gpt-4o-mini` is the fast path, but some attendees prefer open-weight models. `Mistral-7B-Instruct-v0.3` or `Qwen2.5-7B-Instruct` run well on 4-bit on consumer GPUs.

**D. Domain transfer notebook.** Show how to swap Amazon Books for a different text-rich dataset (e.g., Goodreads, MovieLens with plot summaries) using the shared utility module. This directly supports the learning objective of adapting templates to new datasets.

**E. Publish dataset splits on HuggingFace Hub.** Rather than requiring attendees to run Notebook 00 (~2GB download), pre-publish the processed splits so attendees load them in one line.

---

## Execution Roadmap

| Phase | Action | Timeline |
|---|---|---|
| Now | Enable OpenAI API path in Notebooks 01–03; run on 200 users; save results | 1 day |
| Now | Add `requirements.txt` | 1 day |
| Soon | Wire up BERTScore for explanation relevance metric | 2 days |
| Soon | Wire up consistency metric in Notebooks 01 and 02 | 1 day |
| Before submission | Precompute FAISS index, BPR-MF checkpoint, cached LLM outputs | 3 days |
| Before submission | Add preliminary results table to proposal if reviewers ask | 1 day |
| After acceptance | QLoRA fine-tuning; upload adapter to HuggingFace Hub | 1 week |
| After acceptance | Build slide deck (80–100 slides) | 3–4 weeks |
| After acceptance | Cold-start experiment, cost tracking, domain-transfer demo | 2 weeks |
| 1 month before tutorial | Full dry run of all notebooks end-to-end on a clean environment | 1 day |
| 1 week before tutorial | Finalize cached artifacts; test on attendee-class hardware (no GPU) | 1 day |
