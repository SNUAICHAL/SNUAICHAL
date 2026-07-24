# Final competition strategy

Last updated: 2026-07-24

## Objective hierarchy

1. Qualify for the final round using Public+Private performance and a complete report.
2. Maximize final-round award probability through a defensible, efficient method.
3. Preserve exact code/weight reproducibility for organizer reruns and the one-shot
   external evaluation.
4. Avoid leaderboard-only complexity that does not improve the frozen system.

## Frozen final system

- Base: `Qwen/Qwen3.6-27B`
- Revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Training data: all 9,535 official labeled rows
- Adapter: QLoRA checkpoint 2726, rank/alpha 32/32, dropout 0.05
- Runtime: NF4/BF16, SDPA, 512² image-area budget, batch 1
- TTA: `1234`, `2341`, `3412`, `4123`
- Mapping: transform every view prediction back to original input-slot coordinates
- Aggregation: hard majority, deterministic lexicographic exact-tie rule
- Public score: 0.93542
- Physical RTX 3090 peak: 22,710,861,824 bytes, approximately 21.15 GiB

The old 8B/Qwen3.5 plan was an exploration-stage strategy and is superseded by this
document. Its useful role is limited to baseline and ablation evidence.

## Why this candidate

| Candidate | Public | Views | Decision |
|---|---:|---:|---|
| Q36-2726 Latin4 confidence | 0.92670 | 4 | confidence removed |
| **Q36-2726 Latin4 hard** | **0.93542** | **4** | **frozen final** |
| Q36-2726 TTA24 hard | 0.93193 | 24 | more compute, lower observed Public |
| Q36-3400 TTA12 hard | 0.93542 | 12 | tied, three times the views |
| Q35-1073 Latin4 confidence | 0.92844 | 4 | emergency baseline only |

The 2726-vs-3400 comparison is an end-to-end candidate comparison, not a pure
checkpoint causal ablation, because view count also changes. The selection rule is
nevertheless clear: among tied observed Public candidates, keep the earlier
checkpoint and lower inference cost.

## Method contribution boundary

Qwen, QLoRA, index shuffling and TTA are established components. The team-specific
contribution is their task-consistent integration:

1. Treat input permutation as a coordinate transform, not an ordinary image
   augmentation.
2. Transform the training Answer with the same group action used for image slots.
3. Invert each TTA prediction to original-slot coordinates before aggregation.
4. Use a four-view cyclic design that balances the first-order
   frame-by-displayed-position exposure.
5. Remove confidence weighting after a direct calibration ablation.
6. Bind the model tree, adapter, runtime contract and scored source to SHA-256
   manifests and raw-output audits.

Latin4 does not prove that every higher-order positional interaction or model bias is
zero. It only removes the stated first-order exposure imbalance.

## Generalization policy

Public uses only 70% of the test set and repeated candidate selection may overfit it.
The final Q36 model also uses all labeled training rows, so it has no leakage-free
same-model holdout. These limitations are reported rather than hidden.

For the organizer external dataset:

- use the exact frozen checkpoint-2726 release;
- do not train, tune, select, rescale, ensemble or change the prompt;
- reject an `Answer` column and unsafe/missing image paths before CUDA load;
- run the fixed Latin4-hard wrapper once in a fresh output directory;
- preserve `submission.csv`, raw audit, metrics and reproduction receipt;
- do not alter the rule after observing any external score.

## Final evidence

- [Five-page report](final_report_5page_ko.pdf)
- [Machine-readable results](final_results.json)
- [Reviewer reproduction guide](reviewer_quickstart.md)
- [Final inference contract](../configs/final_inference.json)
- [Base manifest](../configs/weights/qwen36-27b-final.manifest.json)
- [Adapter manifest](../configs/weights/qwen36-checkpoint2726-adapter.manifest.json)

No external commercial API, external training dataset, pseudo-labeling or
multi-model ensemble was used for model development.
