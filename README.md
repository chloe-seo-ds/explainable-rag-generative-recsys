# Choosing Between Explainable, Retrieval-Augmented, and LLM-Native Recommenders in Text-Rich Domains

Tutorial proposal and implementation for **RecSys 2026**.

**Authors:** Yejin Seo (LegalZoom / Harvard), Sarthak Baral (Harvard), Kaylee Vo (Harvard)

---

## Overview

This repository contains the code, notebooks, and proposal for a half-day RecSys 2026 tutorial on choosing among three modern paradigms for text-rich recommendation:

1. **Explainable Recommenders** grounded in collaborative evidence (XRec-inspired: BPR-MF ranking + LLM-generated explanations)
2. **Retrieval-Augmented Recommenders** combining dense retrieval with LLM reranking (K-RagRec-inspired)
3. **LLM-Native Generative Recommenders** using instruction-tuned generation (GenRec-inspired)

All three paradigms are benchmarked on a shared **Amazon Books 2023** dataset under a common evaluation framework covering ranking quality, explanation quality, and system metrics (latency, cost, complexity).

---

## Repository Structure

```
explainable-rag-generative-recsys/
├── Notebooks/
│   ├── 00_data_preparation.ipynb      # Download, filter, and split Amazon Books
│   ├── 01_explainable_recsys.ipynb    # Paradigm 1: BPR-MF + LLM explanations
│   ├── 02_rag_recsys.ipynb            # Paradigm 2: FAISS retrieval + LLM reranking
│   ├── 03_generative_recsys.ipynb     # Paradigm 3: Instruction-tuned generation
│   ├── 04_unified_evaluation.ipynb    # Cross-paradigm comparison tables
│   └── tutorial_utils.py              # Shared metrics, data loading, and I/O utilities
├── recsys26_tutorial_proposal.tex     # ACM-formatted tutorial proposal
├── NOTES.md                           # Development notes and expert recommendations
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is optional. Training-heavy steps (BPR-MF, QLoRA) have precomputed checkpoints; all notebooks run inference-only without a GPU.
- An OpenAI API key is required for LLM explanation and reranking steps (optional: local LLaMA via 4-bit quantization is also supported).

### Installation

```bash
git clone https://github.com/chloe-seo-ds/explainable-rag-generative-recsys.git
cd explainable-rag-generative-recsys
pip install -r requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-key-here"
```

---

## Running the Notebooks

Run notebooks in order. Each notebook reads artifacts produced by the previous one.

| Step | Notebook | Description | Runtime (CPU) |
|---|---|---|---|
| 1 | `00_data_preparation.ipynb` | Downloads and preprocesses Amazon Books; creates shared splits | ~10 min (first run) |
| 2 | `01_explainable_recsys.ipynb` | Trains BPR-MF, generates rankings, and optionally generates LLM explanations | ~5 min (ranking) + API calls |
| 3 | `02_rag_recsys.ipynb` | Builds FAISS index, retrieves candidates, optionally reranks with LLM | ~5 min (retrieval) + API calls |
| 4 | `03_generative_recsys.ipynb` | Runs zero-shot or fine-tuned LLM generation | API calls or GPU required |
| 5 | `04_unified_evaluation.ipynb` | Loads all results and produces comparison tables | < 1 min |

### Two Execution Modes

**Full mode (requires OpenAI API key or local GPU):** Uncomment the LLM generation/reranking cells in Notebooks 01–03. Real explanations and reranked results will be generated and cached.

**Demo mode (no API key or GPU needed):** Use precomputed checkpoints and cached LLM outputs. Each notebook checks for cached outputs automatically and loads them if present.

---

## Evaluation Framework

### Ranking Metrics
- HR@5, HR@10, HR@20
- NDCG@5, NDCG@10, NDCG@20
- Recall@K

### Explanation Quality Metrics
- **Relevance:** How well the explanation references items and attributes from the user's actual history
- **Specificity:** Whether the explanation goes beyond generic praise to cite concrete item features
- **Consistency:** Semantic stability of explanations across multiple runs on the same input
- **Hallucination rate:** Fraction of explanation content not supported by the input data

### System Metrics
- End-to-end latency per user (tracked per component)
- Token cost per recommendation (OpenAI API usage)
- Implementation complexity and reproducibility burden

---

## Anchor Papers

| Paradigm | System | Paper |
|---|---|---|
| Explainable | XRec | Ma et al., EMNLP 2024 |
| Retrieval-Augmented | K-RagRec | Wang et al., ACL 2025 |
| LLM-Native Generative | GenRec | Ji et al., ECIR 2024 |

---

## Tutorial Proposal

The full tutorial proposal (`recsys26_tutorial_proposal.tex`) is formatted for ACM and submitted to RecSys 2026. It includes:
- Detailed 2-block outline (90 minutes each)
- Learning objectives for attendees
- Differentiation from 5 prior RecSys tutorials
- Feature comparison table

---

## Citation

If you use this code or build on this tutorial framework, please cite the proposal:

```bibtex
@misc{seo2026recsystutorial,
  title = {Choosing Between Explainable, Retrieval-Augmented, and LLM-Native Recommenders in Text-Rich Domains},
  author = {Seo, Yejin and Baral, Sarthak and Vo, Kaylee},
  year = {2026},
  note = {Tutorial proposal, ACM RecSys 2026}
}
```

---

## License

This repository is intended for educational use as part of the RecSys 2026 tutorial.
