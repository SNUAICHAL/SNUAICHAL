# 심사위원용 재현·평가 가이드

이 문서는 코드 검증과 운영진 외부 데이터 1회 평가를 위한 최단 경로입니다. 최종
시스템은 **Qwen3.6-27B checkpoint-2726 QLoRA + canonical Latin4 + hard
majority** 하나로 고정되어 있습니다. 외부 평가 결과를 본 뒤 checkpoint, prompt,
TTA, 집계 규칙을 바꾸지 않습니다.

## 1. 2분 정적 검증

GPU나 대회 데이터 없이 최종 계약, 두 manifest의 self-hash, 29개 base-model
inventory, 15개 shard, adapter identity, 점수를 만든 핵심 source SHA를 확인합니다.

```bash
python -B -m scripts.verify_evaluation_package
```

예상 결과는 `status: PARTIAL_PASS`, `checkpoint_step: 2726`, `model_files: 29`,
`model_shards: 15`, `scored_source_files: 4`, `release_source_files: 5`입니다.
가중치와 데이터를 생략한
정적 검사에서는 최상위 `complete: false`가 정상입니다. 모든 byte를 확인한
3절의 `--require-all` 실행만 `complete: true`가 됩니다.

## 2. 환경과 가중치

Python 3.10과 CUDA 12.4 호환 드라이버가 필요합니다.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
```

고정 base revision과 최종 adapter를 내려받아 모든 byte를 검증합니다.

```bash
python -B scripts/download_weights.py \
  --manifest configs/weights/qwen36-27b-final.manifest.json \
  --output models/Qwen3.6-27B
python -B scripts/download_final_adapter.py \
  --output weights/qwen36-checkpoint2726
```

저장소가 private이면 GitHub collaborator 계정으로 `gh auth login`을 먼저
실행하거나 read 권한이 있는 `GH_TOKEN`을 설정합니다.

## 3. 무라벨 평가 데이터 배치

```text
data/
├── test.csv
└── test/
    └── <Id>/
        └── <Input_1 ... Input_4 파일>
```

`test.csv` 열은 정확히 다음 순서여야 합니다.

```text
Id,Input_1,Input_2,Input_3,Input_4,Sentence
```

공식 wrapper는 `Answer` 열, 중복 ID, path traversal, symlink image, 누락 image를
CUDA model load 전에 거부합니다. 전체 CPU preflight는 다음과 같습니다.

```bash
python -B -m scripts.verify_evaluation_package \
  --data-dir data \
  --model-path models/Qwen3.6-27B \
  --adapter-path weights/qwen36-checkpoint2726 \
  --require-all \
  --output outputs/evaluation-preflight.json
```

## 4. 외부 데이터 단일 accepted run

외부 평가 전에 별도 학습, 모델 선택, prompt 변경, multi-scale 탐색을 하지
않습니다. 새 output directory에서 fresh run을 실행합니다.

```bash
python -B -m scripts.run_final_inference \
  --data-dir data \
  --output-dir outputs/external-evaluation
```

공식 wrapper는 의도적으로 row resume을 사용하지 않습니다. 이는 다른 model,
adapter, prompt 또는 TTA로 만들어진 과거 row가 섞일 가능성을 없애기
위함입니다. 실패한 실행을 다시 해야 한다면 기존 output directory를 재사용하지
말고 원인을 기록한 뒤 새 directory에서 전체를 다시 실행합니다.

생성되는 파일:

```text
outputs/external-evaluation/
├── submission.csv
├── audit.jsonl
├── metrics.json
└── reproduction-receipt.json
```

`reproduction-receipt.json`은 정적 계약, 무라벨 CSV·image-tree 계약, adapter,
exact child command, release source, raw audit와 최종 CSV 검증을 연결합니다.
독립 확인 명령:

```bash
python -B scripts/validate_submission_artifacts.py \
  --test-csv data/test.csv \
  --submission outputs/external-evaluation/submission.csv \
  --audit outputs/external-evaluation/audit.jsonl \
  --metrics outputs/external-evaluation/metrics.json \
  --expected-tta 4 \
  --aggregation-mode hard
```

공식 외부 Kaggle 제출은 이 저장소가 자동 수행하지 않습니다. 운영진 계정에서
competition과 남은 1회 quota를 확인하고, 위에서 검증한 `submission.csv`의 SHA-256을
기록한 뒤 수동으로 정확히 한 번 제출합니다. 자동 retry를 사용하지 않고 Kaggle
receipt와 제출 시각을 보존합니다.

## 5. 최종 계약과 예상 자원

| 항목 | 고정값 |
|---|---|
| Base | `Qwen/Qwen3.6-27B` |
| Revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Adapter | checkpoint 2726, rank/alpha 32/32 |
| Runtime | NF4, BF16, SDPA, image area 512², batch 1 |
| Views | `1234`, `2341`, `3412`, `4123` |
| 집계 | original-slot canonicalization, hard majority, lexicographic tie |
| 실측 GPU | NVIDIA RTX 3090 24 GB |
| 물리 peak | 22,710,861,824 bytes, 약 21.15 GiB |
| generation loop | 16.4693 s/row, 819행 환산 약 3 h 45 m |
| end-to-end | 13,990.456 s, 약 3 h 53 m |

## 6. 심사 항목별 증거 지도

| 심사 항목 | 저장소 증거 |
|---|---|
| 전략의 논리·독창성 | 보고서 §1·§3; 순열 좌표 역변환, 최소 1차 주변균형 Latin4 |
| 데이터 활용 | 보고서 §2; official 9,535행, deterministic index shuffle |
| 모델·학습 | `src/snuaichal/training.py`, final manifest, report §2 |
| 최적화 | NF4/BF16/SDPA, 4-view ablation, `configs/final_inference.json` |
| 자원 효율 | external NVML measurement와 `docs/final_results.json` |
| 구축 비용 | API 0원 선언, 단일 A6000 학습·단일 RTX 3090 추론, report §4 |
| 재현성 | 두 downloader, static preflight, raw audit, reproduction receipt, CI |

CI는 GPU가 없는 lightweight source/unit 검증입니다. 실제 55.6 GB model과 RTX 3090
실행 증거는 보고서와 `docs/final_results.json`에 별도로 기록했습니다. 최종
release source의 fresh one-row model-load/generation/auditor 재검증은 입력·예측을
제외한 `docs/cleanroom_smoke_receipt.json`으로 보존했습니다.

## 7. 해석 한계

- Public score 0.93542는 test 70% 관찰이며 Private/외부 정확도를 보장하지 않습니다.
- final full-data QLoRA에는 leakage-free 동일-model holdout이 없습니다.
- Latin4가 제거하는 것은 각 원본 frame과 displayed position 사이의 **1차 주변
  노출 불균형**입니다. 모든 고차 순서 상호작용이나 model bias가 0이라는 뜻은
  아닙니다.
- checkpoint-2726 Latin4와 checkpoint-3400 TTA12 비교는 checkpoint와 view 수가
  함께 달라지는 end-to-end 후보 비교이며, 순수한 학습-step causal ablation이
  아닙니다.
- 외부 데이터는 개발·학습·선택·튜닝에 사용하지 않으며, 운영진 평가용 frozen
  inference에만 사용합니다.
