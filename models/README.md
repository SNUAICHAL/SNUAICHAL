# Final model weights

대회 데이터와 base weights는 Git에 커밋하지 않습니다. 최종 시스템은 아래 한
snapshot과 한 adapter만 사용합니다.

## Qwen3.6-27B base

- Repository: `Qwen/Qwen3.6-27B`
- Revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- License: Apache-2.0; pinned snapshot의 `LICENSE` 확인
- Inventory: 29 files, 15 safetensors shards, 55,586,107,940 bytes
- Portable tree SHA-256:
  `e4107e6508793261ca372faf4b560dcb55a5b6ba79a5ab921bfe1b25a207ec07`

자동 다운로드와 byte verification:

```bash
python -B scripts/download_weights.py \
  --manifest configs/weights/qwen36-27b-final.manifest.json \
  --output models/Qwen3.6-27B
```

이미 받은 snapshot의 오프라인 검증:

```bash
python -B scripts/download_weights.py \
  --manifest configs/weights/qwen36-27b-final.manifest.json \
  --output models/Qwen3.6-27B \
  --verify-only
```

## checkpoint-2726 QLoRA adapter

- rank/alpha: 32/32
- adapter SHA-256:
  `189f6c1be09bce1a9b71afeb4807b255b4c144fd2ddfc495ba0109d08ca9f1f6`
- Release archive SHA-256:
  `f89df01d00d3b4808881abd2abb2d48df35213c1b1506dd856ac66c62cd5a054`

```bash
python -B scripts/download_final_adapter.py \
  --output weights/qwen36-checkpoint2726
```

private repository에서는 `gh auth login` 또는 read 권한이 있는
`GH_TOKEN`/`GITHUB_TOKEN`이 필요합니다.

## 검증 원칙

- manifest revision을 branch name이나 `main`으로 바꾸지 않습니다.
- adapter merge/dequantization을 하지 않습니다.
- inference는 기본적으로 network-disabled local snapshot을 사용합니다.
- 운영진 외부 평가 전 아래처럼 base, adapter, source와 무라벨 data 계약을 함께
  확인합니다.

```bash
python -B -m scripts.verify_evaluation_package \
  --data-dir data \
  --model-path models/Qwen3.6-27B \
  --adapter-path weights/qwen36-checkpoint2726 \
  --require-all
```

과거 Qwen3-VL-8B와 Qwen3.5-27B는 baseline/ablation이며 최종 외부 평가에
사용하지 않습니다. license attribution은 `docs/model_licenses.md`에 있습니다.
