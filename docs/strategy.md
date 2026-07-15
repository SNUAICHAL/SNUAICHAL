# Competition strategy

Last updated: 2026-07-14

## Objective

Maximize exact-match accuracy for four-frame temporal ordering while remaining reproducible on one RTX 3090 24 GB and compliant with the single-model/no-external-data rules.

## Measured baseline

- Training rows: 9,535
- Label space: all 24 permutations
- Identity label `[1, 2, 3, 4]`: 1,478 rows (15.50%)
- Kaggle identity submission: 0.15532 public accuracy
- Mean sentence length: 24.15 whitespace-delimited words; maximum 69
- Duplicate sentences: one duplicated pair only
- Qwen2-VL-2B LoRA smoke test: batch sizes 1 and 2 both completed on RTX 3090

The identity public score closely matches its training prior. Input order therefore has a strong and transferable class bias. A useful model must beat this prior on a fixed validation split, not merely reduce language-model loss.

## Primary approach

1. Use a deterministic label-stratified 20% holdout containing only rows whose exact image bytes never occur in another row; never augment validation rows.
2. Randomly permute the four input-image slots in each training row and transform the target accordingly. This preserves the task while flattening the accidental identity-label prior.
3. Fine-tune Qwen2-VL-2B-Instruct with LoRA on the chronological image-label answer.
4. Evaluate generated answers with the competition's exact-match metric and record parse-failure rate in addition to loss.
5. Load the selected LoRA adapter during test inference; never evaluate only the untouched base model.
6. Guarantee a valid submission permutation by parsing model output and using the measured identity prior only as a parse-failure fallback.

## Experiment order

| Stage | Change | Gate |
|---|---|---|
| A | Base Qwen2-VL zero-shot clean validation | Establish exact-match and parse-failure baseline |
| B | LoRA, stratified split, no augmentation | Must improve validation exact match over A |
| C | One random slot permutation per training row | Keep only if it improves exact match, especially non-identity labels |
| D | Compare a 24-way permutation classifier or 24-candidate scoring | Prefer it if exact match is at least tied with free generation |
| E | Tune image resolution and LoRA rank within 24 GB | Keep best single checkpoint; no fold/model ensemble |
| F | Full test inference and Kaggle submission | Validate 819 rows, zero invalid answers, runtime under 24 hours |

## Initial training configuration

- Model: `Qwen/Qwen2-VL-2B-Instruct`
- Precision: BF16 when supported, otherwise FP16
- LoRA: rank 16, alpha 32, dropout 0.05
- Targets: language attention and MLP projections
- Image budget: 56 to 256 visual patches (`43,904` to `200,704` pixels)
- Micro batch: 2 after successful local smoke test
- Gradient accumulation: 4 (effective batch 8)
- Epochs: 1 initially
- Learning rate: 2e-4 with 3% warmup
- Validation: deterministic, label-stratified 20% clean holdout (1,907 rows, zero exact-image overlap)
- Selection metric: validation exact-match accuracy, then parse-failure rate

## Risks and controls

- **Label-prior overfitting:** report identity and non-identity validation accuracy separately.
- **Noisy/ambiguous samples:** start with one epoch; inspect high-loss and disagreement cases before filtering.
- **Generation format errors:** parse strictly and report failures; fallback is identity because it is the measured majority prior.
- **Validation leakage:** exact image reuse affects 46.3% of images, so select validation only from globally unique-image rows and split before augmentation.
- **Checkpoint mismatch:** preserve all adapters and rank them with generated exact match; do not select by teacher-forcing loss.
- **Prohibited test-set analysis:** do not inspect, hash-match, cluster, or otherwise use evaluation-set characteristics to design preprocessing or the model. The official rule explicitly forbids this; use test rows only through the fixed final inference path unless organizers grant written approval.
- **Rule violation:** use one model/checkpoint only, no external or generated data, and record all parameters and runtime.
- **Compute overrun:** smoke-test every configuration, record peak VRAM, and retain resumable checkpoints.

## Success criteria

The implementation is ready for a competitive submission when:

- tests and lint pass;
- a LoRA adapter can be trained, saved, reloaded, and used for inference;
- validation reports exact match, identity accuracy, non-identity accuracy, and parse failures;
- a full 819-row submission contains only valid permutations;
- measured validation accuracy exceeds the 15.5% identity-prior baseline before using a Kaggle submission slot.
