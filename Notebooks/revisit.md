# Revisit List

## Consistency evaluation sample size
- Currently: 3 runs x 20 users across all three notebooks
- Consider bumping to 50 users if results look unstable with real LLM outputs
- 3 runs per user is standard (SelfCheckGPT uses 3-5); increase to 5 if budget allows
- Placeholder generators will always show 1.0 — only meaningful with a real LLM (temperature > 0)

## Evaluation split: leave-last-one-out vs multi-item test set
- Currently using leave-last-one-out (1 ground-truth item per user), which is standard for sequential recommendation and matches the anchor papers (BPR, GenRec, XRec)
- Under this protocol, Recall@K = HR@K (redundant), so Recall@K was dropped
- Consider adding a multi-item test set (e.g., hold out last 3-5 items) as a second evaluation mode
  - Makes Recall@K meaningful and distinct from HR@K
  - More realistic for top-K recommendation evaluation
  - More common in industry settings
  - Would need to update: data splits, GenRec training data construction, evaluation logic
  - GenRec generates one title per prompt — would need multiple prompts or multi-title generation
- Could report both protocols side by side (like we do with full-catalog vs shared-pool)
