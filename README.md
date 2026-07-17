# SNU AI Challenge 2026 baseline

> 텍스트로 풀어보는 장면의 재구성 — 단서를 활용해 4장의 이미지를 재구성하라

[SNU AI Challenge 2026](https://snuaichallenge.github.io/) 참가를 위한 재현 가능한
베이스라인 저장소입니다. 주어진 문장(`Sentence`)과 뒤섞인 이미지 프레임 4장을
바탕으로 원본 비디오의 시간 순서를 복원합니다.

운영진 제공 노트북의 `Qwen2-VL-2B-Instruct` zero-shot 베이스라인을 기반으로,
학습과 추론 진입점을 분리하고 인터넷이 차단된 검증 환경에서도 로컬 가중치만으로
실행할 수 있게 구성했습니다.

- [공식 홈페이지](https://snuaichallenge.github.io/)
- [참가 안내](https://snuaichallenge.github.io/participation/)
- [대회 규칙](https://snuaichallenge.github.io/rules/)
- [FAQ](https://snuaichallenge.github.io/faq/)

대회는 Kaggle에서 진행되며, 참가 신청과 자격 확인 후 접근 권한이 부여됩니다.

## 과제와 평가

각 샘플에는 고유한 `Id`, 문장 하나, 뒤섞인 이미지 4장이 주어집니다. 답은 이미지
번호를 시간순으로 나열하는 형식이 아니라, **입력된 각 이미지가 원본 영상에서 몇 번째
위치인지** 나타내는 순열입니다.

```text
원본 순서:       a, b, c, d
입력 이미지:     a, d, b, c
정답 Answer:    [1, 4, 2, 3]
```

- 평가 지표: Exact Match Accuracy
- Public leaderboard: 전체 테스트 데이터의 70%
- 한 위치라도 틀리면 해당 샘플 전체가 오답
- 별도 validation 데이터는 없으므로 `train` 일부를 검증용으로 분리
- 학습 데이터에는 모호하거나 텍스트와 무관한 프레임 등 정제되지 않은 샘플이 포함될 수 있음

데이터 원본이나 수정본을 재배포할 수 없으므로 이 저장소에는 데이터 파일을 포함하지
않습니다. 자세한 내용은 공식 [Data](https://snuaichallenge.github.io/data/) 페이지를
확인하세요. 저장소 공개 여부와 코드·가중치 공유 범위 역시 공식 Agreement 및 각 모델
라이선스를 먼저 확인해야 합니다.

## 주요 일정

| 일정 | 날짜 및 시간 |
|---|---|
| 참가 신청 | 2026-06-22 ~ 2026-07-17 |
| 온라인 예선·제출 | 2026-06-29 10:00 ~ 2026-07-24 23:59 KST |
| 최종 리더보드 | 2026-07-25 |
| 상위팀 코드·보고서 제출 | 2026-07-25 ~ 2026-07-28 |
| 본선 진출팀 발표 | 2026-08-03 |
| 발표 자료 제출 마감 | 2026-08-06 |
| 오프라인 본선 | 2026-08-07 예정 |

일정은 변경될 수 있으므로 공식 [Timeline](https://snuaichallenge.github.io/timeline/)을
우선합니다.

## 저장소 구조

```text
.
├── .github/workflows/ci.yml       # 가벼운 정적 검사와 단위 테스트
├── data/                          # 대회 데이터(커밋 금지)
├── docs/
│   ├── competition_checklist.md  # 규정 준수 체크리스트
│   └── experiment_log.md         # 실험·비용 기록 양식
├── models/                        # 로컬 모델 가중치(커밋 금지)
├── notebooks/                     # 운영진 제공 원본 베이스라인
├── outputs/                       # 제출 파일과 원시 출력(커밋 금지)
├── scripts/train.py               # Qwen2-VL LoRA 학습 진입점
├── src/snuaichal/
│   ├── evaluation.py             # exact-match 및 실패율 측정
│   ├── inference.py              # base/LoRA 검증 및 제출 추론
│   ├── submission.py             # 출력 파싱·순서 변환·검증
│   └── training.py               # 층화 split·순열 증강·LoRA 학습
└── tests/                         # 제출 형식 단위 테스트
```

## 환경

대회 검증 서버 사양은 AMD EPYC 7502 32-Core Processor 2개, RAM 512 GB,
NVIDIA RTX 3090 24 GB 1장, NVIDIA driver 550.54.15, CUDA 12.4입니다.
운영체제는 제공된 규칙에 명시되지 않았으므로 최종 안내를 확인해야 합니다. 이 저장소의
기준 Python은 공식 노트북과 같은 3.10.20이며 라이브러리는
`requirements.txt`에 고정했습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell에서는 활성화 명령만 `.\.venv\Scripts\Activate.ps1`로
바꾸면 됩니다. `requirements.txt`의 일반 PyPI PyTorch가 환경과 맞지 않는 경우
위 CUDA 12.4 명령으로 먼저 설치하세요.

## 데이터와 모델 준비

운영진 제공 데이터는 [data/README.md](data/README.md), 모델 스냅샷은
[models/README.md](models/README.md)의 구조로 배치합니다. 데이터와 가중치는
용량·라이선스·누수 방지를 위해 `.gitignore`로 제외됩니다.

기본 경로는 모두 저장소 루트 기준 상대 경로입니다.

```text
data/train.csv
data/train/<Id>/<image file>
data/test.csv
data/test/<Id>/<image file>
models/Qwen2-VL-2B-Instruct/
```

## 실행

Kaggle에서 팀 생성 권한을 얻기 위한 첫 제출은 모델 없이 즉시 만들 수 있습니다.

```bash
snu-baseline-submit --test-csv data/test.csv --output outputs/baseline_submission.csv
```

이 파일은 모든 샘플에 유효한 기본 순열 `[1, 2, 3, 4]`를 사용합니다. 성능 확인용
모델 베이스라인이 아니라 제출 형식과 팀 생성 절차를 확인하기 위한 파일입니다.

### 재현 대상 모델

| 모델 | local path | 공식 revision | 최초 공개 | 라이선스 | snapshot 크기 |
|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | `models/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 2025-10-11 | Apache-2.0 | 약 17.55 GB |
| Qwen3.5-27B | `models/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` | 2026-02-24 | Apache-2.0 | 약 55.58 GB |

둘 다 2026-05-31 cutoff 이전 공개 가중치입니다. `Qwen3.5-27B`는 dense
`Qwen3_5ForConditionalGeneration` 모델입니다. 모델 파일이 없을 때 학습 코드가
자동 다운로드하지 않습니다. 정확한 다운로드 명령은 [models/README.md](models/README.md)에
기록되어 있습니다.

### Qwen3-VL-8B 0.91972 레시피

split은 augmentation보다 먼저 수행합니다. 9,535행의 10% validation은 약 954행,
train은 약 8,581행입니다. effective batch 8일 때 `ceil(8581/8)=1073`
update/epoch이므로 4 epoch 종료점은 4,292입니다. cosine scheduler는 6 epoch,
즉 6,438 step horizon을 유지합니다. `max_steps=4292`를 사용하지 않으며 callback이
4,292에서 저장 후 종료합니다.

2-step VRAM smoke:

```bash
snu-train \
  --model-path models/Qwen3-VL-8B-Instruct \
  --load-in-4bit \
  --validation-fraction 0.10 \
  --image-size 512 \
  --epochs 6 \
  --stop-after-steps 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --output-dir outputs/qwen3-vl-8b-smoke
```

전체 학습은 smoke의 peak VRAM, loss, visual token 수를 확인한 뒤에만 실행합니다.

```bash
snu-train \
  --model-path models/Qwen3-VL-8B-Instruct \
  --load-in-4bit \
  --validation-fraction 0.10 \
  --image-size 512 \
  --epochs 6 \
  --stop-after-steps 4292 \
  --save-steps 1073 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --output-dir outputs/qwen3-vl-8b-aug
```

학습 입력은 매 epoch마다 `SHA-256(seed, epoch, Id)`로 결정된 permutation을
on-the-fly 적용합니다. validation은 원본 순서를 유지합니다. split Id는
`split_manifest.json`, scheduler 계획은 `schedule.json`, LoRA target과 trainable
parameter는 `model_manifest.json`에 저장됩니다. resume 시 scheduler/global step은
Trainer checkpoint에서 복구되며 4,292 이상 checkpoint의 재개는 거부됩니다.

### Validation과 cyclic 4-TTA

4-TTA 입력 순서는 `[1,2,3,4]`, `[2,3,4,1]`, `[3,4,1,2]`, `[4,1,2,3]`입니다.
각 view를 순차 실행하고 prediction을 원본 slot 좌표로 역변환한 뒤 canonical Answer에서
다수결합니다. 동률은 lexicographic minimum으로 결정합니다. 모든 view 파싱 실패일 때만
identity `[1,2,3,4]`를 사용합니다.

```bash
snu-infer \
  --test-csv data/train.csv \
  --image-dir data/train \
  --model-path models/Qwen3-VL-8B-Instruct \
  --adapter-path outputs/qwen3-vl-8b-aug/checkpoint-4292 \
  --load-in-4bit --image-size 512 --tta 4 \
  --validation-fraction 0.10 \
  --output outputs/validation_predictions.csv \
  --metrics-output outputs/validation_metrics.json
```

최종 제출:

```bash
snu-infer \
  --data-dir data \
  --model-path models/Qwen3-VL-8B-Instruct \
  --adapter-path outputs/qwen3-vl-8b-aug/checkpoint-4292 \
  --load-in-4bit --image-size 512 --tta 4 \
  --output outputs/submission_v5_8b_aug_checkpoint-4292_tta4.csv \
  --audit-log outputs/submission_v5_8b_aug_checkpoint-4292_tta4.jsonl
```

### Public 리더보드 감시

Kaggle public 리더보드의 신규 팀, 점수·순위 및 TOP7/10/16 컷 변동을 터미널에서
감시할 수 있습니다. 표 상단에는 공식 예선 기간(`2026-06-29 10:00`부터
`2026-07-24 23:59 KST`)의 진행률과 실시간 마감 카운트다운이 표시됩니다. 기본 우리
팀명은 `밥먹을돈으로3090사서거지됨`이며 `--team` 또는 `SNUAICHAL_TEAM` 환경변수로
바꿀 수 있습니다.

```powershell
# 한 번 조회해 인증과 팀명을 확인
snu-leaderboard-watch --once

# 기본 30초 간격으로 감시하고 변동 시 터미널 벨 출력
snu-leaderboard-watch --bell

# 10초 간격, 매번 표 출력
snu-leaderboard-watch --interval 10 --table
```

인증은 `~/.kaggle/kaggle.json`의 `username`/`key`, 환경변수
`KAGGLE_USERNAME`+`KAGGLE_KEY`, 또는 `KAGGLE_API_TOKEN` 중 하나를 사용합니다.
인증값은 출력이나 상태 파일에 기록하지 않습니다. 마지막 정상 스냅샷과 변동 이력은
각각 `outputs/leaderboard_watch/state.json`, `events.jsonl`에 저장되어 프로세스를
재시작해도 중간 변동을 확인할 수 있습니다. 상태를 무시하려면 `--fresh`, 파일을 전혀
남기지 않으려면 `--no-state`를 사용합니다. 원본 호환 진입점
`python tools/leaderboard_watch.py`도 제공됩니다.

운영진이 일정을 변경하면 ISO 형식의 `--start`, `--deadline`으로 덮어쓸 수 있습니다.

```powershell
snu-leaderboard-watch `
  --start 2026-06-29T10:00+09:00 `
  --deadline 2026-07-24T23:59+09:00
```

Qwen3.5-27B challenger는 같은 명령에서 `--model-path models/Qwen3.5-27B`와
`--output-dir`만 바꿉니다. 기본 LoRA rank는 8입니다. native Windows에서는 pure
PyTorch fallback을 사용하며, Linux/WSL2 fast kernel은
`requirements-linux-kernels.txt`의 선택 의존성을 사용합니다. 27B는 먼저 2-step
VRAM smoke만 수행하고 8B pipeline 검증 전에는 장시간 학습하지 않습니다.

모델/processor 로딩은 기본 `local_files_only=True`입니다. 완전 오프라인 검증은
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`을 설정해 실행합니다. VRAM은 학습의
`training_summary.json`과 validation의 `peak_vram_mib`로 확인하며 필요하면 별도로
`nvidia-smi --query-compute-apps=used_memory --format=csv`를 기록합니다.

validation metric에는 Exact Match, identity/non-identity accuracy, parse failure,
TTA consistency, seconds/sample, peak VRAM, 24×24 permutation confusion이 포함됩니다.
audit JSONL에는 TTA별 raw prediction과 canonical prediction이 기록됩니다. 최종 CSV는
Id 순서·행 수·모든 Answer의 1~4 순열 여부를 저장 전에 검증합니다.

## 테스트

```bash
pip install -r requirements-dev.txt
pip install -e .
ruff check src tests scripts tools
pytest
```

단위 테스트는 `[4, 2, 1, 3]` 같은 시간순 모델 출력을 대회 제출 정의인
`[3, 2, 4, 1]`로 정확히 변환하는지, 잘못된 순열을 거부하는지 검증합니다.

## 규정 준수

현재 구현은 외부 API, 외부 데이터, 앙상블을 사용하지 않는 단일 모델 zero-shot
베이스라인이며 외부 API 비용은 `0 KRW`입니다. 공식 규칙상 상용 API는 학습과
추론에 사용할 수 없고, 데이터 전처리에만 총 30,000원 한도로 사용할 수 있습니다.
또한 제공 데이터 자체의 증강은 가능하지만 생성형 모델을 이용한 데이터 생성·변형은
허용되지 않습니다. 이후 방법을 변경할 때는
[규정 체크리스트](docs/competition_checklist.md)와
[실험 기록 양식](docs/experiment_log.md)을 함께 갱신하세요.

특히 다음은 자동화만으로 보장할 수 없으므로 제출자가 직접 확인해야 합니다.

- 모델 가중치의 최초 공개일과 라이선스 근거
- RTX 3090 24 GB에서 전체 테스트 추론 24시간 이내 완료
- 평가 데이터 정보가 학습·전처리·모델 설계에 사용되지 않았는지 여부
- 코드와 가중치를 합친 최종 크기가 80 GB 이하인지 여부
- 운영진의 최신 공지와 Kaggle Discussion 답변

공식 [Rules](https://snuaichallenge.github.io/rules/)는 대회 중 추가될 수 있으므로
제출 전마다 다시 확인하세요. 제출 횟수는 FAQ 기준 UTC 하루 최대 2회입니다.

## 원본 베이스라인

운영진 제공 노트북은 `notebooks/SNU_AI_Challenge_Baseline_Code.ipynb`에 원형으로
보관합니다. 실제 재현성 검증에는 노트북이 아니라 `src/snuaichal/inference.py`를
사용하세요.
