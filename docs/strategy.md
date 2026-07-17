# Competition strategy

Last updated: 2026-07-15

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

The reproduction target is the public 0.91972 result with one
`Qwen/Qwen3-VL-8B-Instruct` checkpoint. Qwen3.5-27B is a secondary dense challenger,
not an ensemble member.

1. Split a deterministic, label-stratified, image-disjoint 10% holdout before any augmentation and save all split IDs.
2. At each training epoch, deterministically shuffle each row's four input slots from `SHA-256(seed, epoch, Id)` and transform the answer. Never augment validation.
3. Load the 8B model with bitsandbytes NF4 double quantization and BF16 compute. Freeze the vision tower and apply LoRA to language attention/MLP projections.
4. Train with micro batch 1, accumulation 8, and a six-epoch cosine scheduler horizon. Stop and save after four epochs at global step 4292; do not shorten the scheduler with `max_steps`.
5. Evaluate every epoch checkpoint by generated exact match, parse failure, 24-way confusion, speed, peak VRAM, and cyclic 4-TTA consistency.
6. For TTA, canonicalize every permuted-view prediction back to original input-slot coordinates before voting. Use identity only when all four parses fail.
7. Run Qwen3.5-27B only after an 8B two-step smoke and pipeline validation. Keep one final model/checkpoint.

## Experiment order

| Stage | Change | Gate |
|---|---|---|
| A | Qwen3-VL-8B NF4 two-step smoke at 512² | No OOM; record visual tokens, trainable parameters, loss, LR, peak VRAM |
| B | 8B LoRA, 10% clean split, no augmentation | Establish generated exact-match baseline |
| C | Deterministic epoch-aware slot augmentation | Improve non-identity and overall exact match |
| D | Evaluate checkpoints 1073/2146/3219/4292 with cyclic 4-TTA | Select by exact match, then failures/speed; no ensemble |
| E | Qwen3.5-27B rank-8 NF4 smoke and one challenger run | Keep only if 24 GB feasible and validation improves |
| F | Full test inference and Kaggle submission | Validate 819 rows, zero invalid answers, runtime under 24 hours |

## Reproduction training configuration

- Model: `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Quantization: bitsandbytes NF4, double quantization, BF16 compute
- LoRA: rank 16, alpha 32, dropout 0.05; language projections only; frozen vision tower
- Image budget: 512×512 area (`262,144` pixels), while preserving aspect ratio
- Micro batch: 1; gradient accumulation: 8 (effective batch 8)
- Learning rate: 2e-4, 3% warmup, cosine horizon 6438 updates
- Stop: callback at global step 4292; checkpoints every 1073 updates
- Validation: deterministic label-stratified 10% clean holdout, never augmented
- Generation: thinking disabled for Qwen3.5, strict final-answer parsing, max 16 new tokens
- Selection metric: validation exact match, then parse-failure/TTA consistency/speed
- Challenger: `Qwen/Qwen3.5-27B`, revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, default LoRA rank 8

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
