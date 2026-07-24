# SNU AI Challenge 2026 final solution

문장과 뒤섞인 네 장의 프레임으로 원래 시간 순서를 복원하는 SNU AI Challenge
2026 제출 시스템입니다. 최종 재현 대상은 **Qwen3.6-27B + checkpoint-2726
QLoRA + canonical Latin4 hard vote**입니다.

이 저장소는 다음을 한 번에 재현하도록 고정했습니다.

- 공개일 제한 이전의 Qwen3.6-27B base snapshot을 revision과 파일 SHA-256으로 검증
- checkpoint-2726 LoRA adapter를 GitHub Release에서 내려받아 SHA-256 검증
- RTX 3090 24 GB 한 장에서 NF4/BF16 추론
- 네 개의 균형 잡힌 cyclic view를 원래 입력 좌표로 되돌린 뒤 hard majority 집계
- 819행 제출 CSV, raw-output audit, metrics receipt 생성

최종 보고서는 [한국어 PDF](docs/final_report_5page_ko.pdf)와
[원문 Markdown](docs/final_report_5page_ko.md)으로 제공합니다.

## 최종 결과와 선택

| 모델·추론 | Public | 비용 | 최종 사용 |
|---|---:|---:|---|
| Qwen3.6-27B ckpt2726, Latin4 confidence | 0.92670 | 4 views | 아니오 |
| **Qwen3.6-27B ckpt2726, Latin4 hard** | **0.93542** | **4 views** | **예** |
| Qwen3.6-27B ckpt2726, TTA24 hard | 0.93193 | 24 views | 아니오 |
| Qwen3.6-27B ckpt3400, TTA12 hard | 0.93542 | 12 views | 아니오 |
| Qwen3.5-27B ckpt1073, Latin4 confidence | 0.92844 | 4 views | fallback |

checkpoint-3400은 세 배의 추론 view와 더 긴 학습에도 checkpoint-2726을 넘지
못했습니다. TTA24와 confidence tie-break도 각각 점수를 낮췄습니다. 따라서 동일
최고점 중 가장 작고 빠르며 재현 가능한 checkpoint-2726 Latin4 hard를 최종
시스템으로 선택했습니다.

Public leaderboard는 test의 70%만 반영하므로 위 비교가 Private 또는 외부 데이터의
우위를 보장하지는 않습니다. 이 한계를 숨기지 않고, 운영진의 외부 데이터 평가는
아래의 단일 고정 모델과 코드로 수행합니다.

## 최종 계약

| 항목 | 고정값 |
|---|---|
| Base | `Qwen/Qwen3.6-27B` |
| Revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Portable tree SHA-256 | `e4107e6508793261ca372faf4b560dcb55a5b6ba79a5ab921bfe1b25a207ec07` |
| 학습 | 전체 9,535행, 2,726 optimizer steps, 2.28694 epochs |
| Adapter | QLoRA rank 32 / alpha 32 / dropout 0.05 |
| Adapter SHA-256 | `189f6c1be09bce1a9b71afeb4807b255b4c144fd2ddfc495ba0109d08ca9f1f6` |
| Runtime | NF4, BF16 compute, SDPA, image size 512, batch 1 |
| Generation | `max_new_tokens=64`, seed 42, thinking disabled |
| Views | `1234`, `2341`, `3412`, `4123` |
| Aggregation | original-slot canonicalization → hard majority → lexicographic tie |

채점에 사용한 `inference.py`, prompt, parser, TTA와 aggregation 코드는 이후 점수에
맞춰 다시 쓰지 않았습니다. 설정의 기계 판독본은
[configs/final_inference.json](configs/final_inference.json)에 있습니다.

## 요구 환경

- Python 3.10
- Linux 또는 Windows
- NVIDIA RTX 3090 24 GB 이상
- CUDA 12.4 호환 driver
- 약 80 GB의 디스크 여유

검증 환경:

```text
torch 2.6.0+cu124
torchvision 0.21.0+cu124
transformers 5.13.1
accelerate 1.7.0
peft 0.19.1
bitsandbytes 0.48.2
qwen-vl-utils 0.0.14
```

설치:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell에서는 활성화 명령만 다음으로 바꿉니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

## 데이터 배치

대회 데이터는 재배포하지 않습니다. 운영진 데이터를 아래처럼 둡니다.

```text
data/
├── test.csv
├── test/
│   └── <Id>/<image files>
├── train.csv
└── train/
```

`test.csv`는 `Id, Input_1, Input_2, Input_3, Input_4, Sentence` 열을 가져야
합니다. 외부 데이터나 외부 상용 API를 학습·전처리·추론에 사용하지 않았습니다.

## 가중치 준비

### 1. Base model

고정 manifest에는 29개 파일(15개 safetensors shard)의 크기와 SHA-256이 모두
포함됩니다.

```bash
python -B scripts/download_weights.py \
  --manifest configs/weights/qwen36-27b-final.manifest.json \
  --output models/Qwen3.6-27B
```

이미 snapshot이 있다면 다운로드 없이 검증할 수 있습니다.

```bash
python -B scripts/download_weights.py \
  --manifest configs/weights/qwen36-27b-final.manifest.json \
  --output models/Qwen3.6-27B \
  --verify-only
```

### 2. checkpoint-2726 adapter

adapter는 GitHub Release asset으로 제공됩니다. downloader는 archive의
591,369,852 bytes와 SHA-256
`f89df01d00d3b4808881abd2abb2d48df35213c1b1506dd856ac66c62cd5a054`
및 내부 두 파일을 모두 검증한 뒤 설치합니다.

저장소가 private인 동안에는 먼저 `gh auth login`으로 collaborator 계정을
인증하거나 `GH_TOKEN`/`GITHUB_TOKEN`을 설정해야 합니다.

```bash
python -B scripts/download_final_adapter.py \
  --output weights/qwen36-checkpoint2726
```

수동 다운로드 시:

```text
https://github.com/SNUAICHAL/SNUAICHAL/releases/download/final-q36-2726-v1/q36-checkpoint2726-adapter.tar.gz
```

## 최종 추론

다른 CUDA workload가 없는지 확인한 뒤 실행합니다.

```bash
python -B -m scripts.run_final_inference --data-dir data --resume
```

기본 출력:

```text
outputs/final-inference/
├── submission.csv
├── audit.jsonl
└── metrics.json
```

`--resume`은 완전한 row audit만 재사용합니다. 최종 제출 전에는 다음을 확인합니다.

```bash
python -m pytest
python -m ruff check src scripts tests
```

실측 checkpoint-2726 Latin4 run은 RTX 3090에서 다음과 같았습니다.

- 819/819 rows, parse failure 0
- 16.4693 seconds/row, 약 3시간 45분
- physical peak VRAM 22,710,861,824 bytes(약 21.15 GiB)
- Public score 0.93542

## 방법 개요

### 순열-aware 학습

각 epoch의 입력 slot은 `SHA-256(seed, epoch, Id)`로 결정론적으로 섞고 정답
permutation도 동일 좌표 변환을 적용했습니다. 입력 위치 자체의 shortcut을 줄이고
test-time canonicalization과 학습 표현을 맞추는 것이 목적입니다.

### Canonical Latin4

네 cyclic permutation은 각 원본 이미지가 네 displayed position에 정확히 한 번씩
나타나는 Latin square입니다. 각 view의 예측은 먼저 원래 slot 좌표로 역변환한 뒤
vote합니다. 이 단계를 생략하면 서로 다른 좌표계의 답을 집계하게 됩니다.

### 적은 view를 선택한 이유

24-view는 더 많은 계산으로 permutation coverage를 늘리지만, 이미 균형인 Latin4
이후에는 편향 감소보다 noisy view의 추가가 커질 수 있습니다. 실제로 동일
checkpoint-2726에서 TTA24 hard는 0.93193으로 Latin4 hard 0.93542보다 낮았습니다.
confidence tie-break도 0.92670으로 하락해 최종 시스템에서는 제거했습니다.

### 재현성과 자원 효율

base model, adapter, command, raw output과 physical-memory evidence를 SHA-256
receipt로 묶었습니다. adapter merge와 multi-model ensemble은 쓰지 않았고,
단일 resident model과 batch 1로 24 GB 제약을 만족했습니다.

## 학습 재현

최종 adapter는 full-data QLoRA run의 optimizer step 2,726입니다. 학습 코드는
`snu-train` 진입점과 [src/snuaichal/training.py](src/snuaichal/training.py)에
있습니다. 장시간 학습 전에 2-step smoke와 checkpoint reload를 먼저 수행하세요.

```bash
snu-train \
  --model-path models/Qwen3.6-27B \
  --model-repository Qwen/Qwen3.6-27B \
  --model-family qwen3_5 \
  --model-revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --model-manifest configs/weights/qwen36-27b-final.manifest.json \
  --load-in-4bit \
  --validation-fraction 0 \
  --epochs 6 \
  --image-size 512 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --seed 42 \
  --stop-after-steps 2726 \
  --output-dir outputs/q36-r32-full2726
```

장비와 library minor version에 따라 kernel 선택과 속도는 달라질 수 있습니다.
checkpoint의 optimizer/scheduler/RNG까지 포함한 완전 재개를 권장합니다.

## 저장소 구조

```text
configs/
├── final_inference.json
└── weights/
    ├── qwen36-27b-final.manifest.json
    └── qwen36-checkpoint2726-adapter.manifest.json
docs/
├── final_report_5page_ko.pdf
└── model_licenses.md
scripts/
├── download_weights.py
├── download_final_adapter.py
└── run_final_inference.py
src/snuaichal/
├── inference.py
├── training.py
├── tta.py
├── submission.py
└── physical_memory.py
tests/
```

## 한계

- Qwen3.6 최종 run은 전체 labeled data로 학습했으므로 그 데이터에서의 validation
  정확도를 모델 선택 근거로 사용할 수 없습니다.
- 제출 비교는 Public 70%에 노출되어 있어 반복 선택에 따른 leaderboard overfitting
  가능성이 있습니다.
- Private 및 외부 데이터 성능은 제출 시 알 수 없으며, 최종 고정 코드로 별도
  평가받아야 합니다.
- semantic error 유형에 대한 대규모 수동 annotation과 다중 seed 비교는 완료하지
  못했습니다.

## 라이선스와 데이터

Qwen3.6-27B는 원 배포자의 Apache-2.0 라이선스를 따릅니다. 상세 attribution은
[docs/model_licenses.md](docs/model_licenses.md)를 확인하세요. 대회 데이터와
이미지는 이 저장소 또는 Release에 포함하지 않습니다.

프로젝트 코드에는 별도의 최상위 `LICENSE`가 아직 없으므로, 저장소 소유자가
라이선스를 승인하기 전까지 이 저장소 자체가 재사용 권한을 부여한다고 해석해서는
안 됩니다.
