# Local LLM Integration — Design Record

## Summary

All three paradigm notebooks (01, 02, 03) now run real LLM inference locally using
`meta-llama/Llama-3.2-3B-Instruct` in float16. This document records the hardware
constraints encountered, models evaluated, the final choice, and every code change made.

---

## Hardware Context

| Property | Value |
|---|---|
| GPU | NVIDIA RTX 5050 Laptop GPU |
| VRAM | 8 GB |
| GPU Architecture | Blackwell (compute capability 12.0 / sm_120) |
| OS | Windows 11 |

**Critical constraint discovered:** `bitsandbytes` 0.49.2 (the latest release as of May 2026)
has no CUDA kernels compiled for Blackwell (sm_120). Attempting 4-bit quantization on this
GPU triggers a CUDA kernel crash that surfaces as a spurious out-of-memory error, not a
clear incompatibility message. The 8B model is ~15 GB in bf16/fp16 and cannot fit in 8 GB
VRAM without quantization, so the original plan (LLaMA-3-8B-Instruct + 4-bit) was blocked.

A secondary issue on Windows: HuggingFace's Xet storage backend (introduced for faster
large-file downloads) crashes the download thread with `RuntimeError: Background writer
channel closed`. Fixed globally across all three notebooks with:
```python
os.environ["HF_HUB_DISABLE_XET"] = "1"
```

---

## Models Evaluated

| Model | VRAM (float16) | Gated? | Verdict |
|---|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | ~16 GB | Yes (Meta license) | **Rejected** — requires 4-bit quant; bitsandbytes incompatible with RTX 5050 Blackwell |
| `meta-llama/Llama-3.2-3B-Instruct` | ~6 GB | Yes — separate gate from 3.0/3.1 | **Rejected (initially chosen, then blocked)** — requires a separate license acceptance; HF returns 403 even with LLaMA-3-8B access |
| `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | No (Apache 2.0) | **Chosen** — fully open, no approval needed, same API |
| `microsoft/Phi-3.5-mini-instruct` | ~7.6 GB | No | Borderline fit; slightly larger |

### Why Qwen2.5-3B-Instruct

1. **No license gate.** Apache 2.0. Downloads immediately without any HuggingFace access
   request or Meta account requirement.

2. **Fits cleanly in 8 GB VRAM in float16.** ~6 GB loaded, leaving ~2 GB headroom. No
   quantization library needed.

3. **Same prompt interface.** `apply_chat_template()` works identically to the LLaMA-3
   interface — zero changes to the generation functions.

4. **Strong instruction following.** Qwen 2.5 outperforms LLaMA 3.2-3B on most instruction
   following benchmarks; it is a suitable model for explanation and ranking tasks.

**Note on LLaMA 3.2:** Access was requested by the user concurrently. Once approved,
switching back requires only changing `MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"` in
each notebook's model loading cell — no other code changes needed.

**Root cause of LLaMA 3.2 gate:** Meta introduced a separate license agreement for the
3.2 family (which includes multimodal variants) in late 2024. Existing LLaMA 3.0/3.1
license holders are NOT automatically granted 3.2 access — a separate form submission
at huggingface.co/meta-llama/Llama-3.2-3B-Instruct is required.

---

## Loading Pattern (all three notebooks)

```python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"   # Windows Xet crash workaround

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

Generation uses `apply_chat_template()` (the correct LLaMA-3 interface; not the LLaMA-2
`[INST]...[/INST]` format) and sets `pad_token_id=tokenizer.eos_token_id` to suppress a
common padding warning during generation.

---

## Code Changes by Notebook

### NB01 — `01_explainable_recsys.ipynb`

**cell-1 (pip install):** Added `accelerate bitsandbytes transformers` (accelerate is
required by `device_map="auto"`; bitsandbytes kept for future compatibility even though
quantization is not used).

**cell-5 (BPR sampler):** Vectorized the negative sampler. The original `sample_batch()`
called `rng.choice(python_list)`, which triggers a list-to-array conversion on every call
and made BPR training take ~5 hours. Fixed by pre-converting to numpy arrays before the
training loop:
```python
user_arr = np.array(list(user_histories.keys()))
all_items_arr = np.arange(n_items)
```
Training time dropped to ~15 minutes.

**cell-13 (LLM loading):** Replaced the original commented-out LLaMA-2 block (which used
`[INST]` format and targeted `Meta-Llama-3-8B-Instruct` with `BitsAndBytesConfig`) with
the new float16 loading pattern above. Removed the `del model` call that was present here —
the BPR model is needed downstream in cell-18 for shared-pool evaluation, and at ~18 MB
it does not compete meaningfully for VRAM.

**cell-14 (generation dispatch):** Removed the placeholder `generate_explanation()`
function and set `generate_explanation = generate_explanation_local` directly.

---

### NB02 — `02_rag_recsys.ipynb`

**cell-11 (LLM loading + `rerank_local`):** Replaced `Meta-Llama-3-8B-Instruct` +
`BitsAndBytesConfig` with the float16 loading pattern. The `rerank_local()` function
parses the LLM's output for a JSON-style ranked list (e.g. `[3, 1, 7, ...]`) via regex
and falls back to retrieval order if parsing fails.

**cell-13 (deleted):** Removed the old retrieval-only pipeline cell that ran over the
entire test set using `rerank_retrieval_only`. If left in, it would have populated the
`"rag_pipeline_outputs"` cache before cell-14's local LLM pipeline ran, causing cell-14
to silently skip LLM reranking and load stale retrieval-only results.

**cell-14 (capped pipeline):** Kept intact. Runs `rerank_local` over the first
`N_RERANK = 200` test users and saves to cache.

---

### NB03 — `03_generative_recsys.ipynb`

**cell-10 (LLM loading + `generate_recommendation_local`):** Same float16 pattern.
Generates a numbered list of book titles; titles are extracted via `re.findall(r'\d+\.\s*(.+)', text)`
and matched against the item catalog using fuzzy string matching.

**cell-12 (deleted):** Removed the old placeholder pipeline cell. Same cache-poisoning
risk as NB02 — if left in, it would have saved random-item predictions to
`"generative_pipeline_outputs"` before the local LLM ran.

**cell-13 (capped pipeline):** Kept intact. Runs `generate_recommendation_local` over
the first `N_GENERATE = 200` test examples.

**cell-14 (inserted — evaluation):** Added missing evaluation cell after the pipeline.
The save cell (cell-17) references `ranking_results`, `latency`, and `explanation_results`;
none of these were defined anywhere in the notebook before this insertion.
`explanation_results` is set to `{}` — GenRec is a pure generation paradigm with no
per-explanation quality metrics.

**cell-15 (pool eval):** Changed the loop from `for example in genrec_test:` (all ~40K
examples, would call `generate_fn` on each — hours of inference) to
`for example in genrec_test[:N_GENERATE]:` to match the main pipeline's cap.

**cell-16 (deleted):** Removed a broken markdown-type cell that contained Python code —
a leftover from an earlier failed insert that created a duplicate cell with the wrong
cell type.

---

### NB04 — `04_unified_evaluation.ipynb`

No changes required. The notebook already handles missing upstream results gracefully
(try/except around each paradigm load), reads the correct result keys, and produces
both console tables and a LaTeX export. The empty `explanation_results` from NB03 will
produce blank explanation metric columns for the generative paradigm, which is accurate.

---

## Scale Caps

LLM inference over the full test set would take days on a single consumer GPU. Caps set:

| Notebook | Variable | Value | Rationale |
|---|---|---|---|
| NB01 | `N_EXPLAIN` | 100 users | ~8 min at ~5s/explanation |
| NB02 | `N_RERANK` | 200 users | ~20 min at ~6s/rerank call |
| NB03 | `N_GENERATE` | 200 examples | ~20 min at ~6s/generation call |

Results are cached after first run (`save_cache` / `load_cache`) so subsequent notebook
executions are instant.

---

## Deleted Artifacts

- `~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct` — 15 GB,
  deleted after switching to the 3B model. The model had been fully downloaded (4 shards,
  ~3.7 GB each) but was never successfully loaded due to the bitsandbytes/Blackwell
  incompatibility.
- HuggingFace datasets Arrow cache (`McAuley-Lab___amazon-reviews-2023/` and
  `downloads/extracted/`) — ~61 GB deleted in an earlier session. The processed parquet
  files in `Notebooks/data/` are the only dataset artifacts needed going forward.
