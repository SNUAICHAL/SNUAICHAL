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

경쟁용 학습은 이미지 SHA-256이 train에 재등장하지 않는 20% clean validation을
층화 추출한 뒤, 학습 입력 슬롯을 재배열해 24개 순열 라벨을 균형화합니다. 기본값은
RTX 3090에서 검증한 micro batch 2, gradient accumulation 4, LoRA rank 16입니다.

```bash
snu-train --output-dir outputs/qwen2-vl-lora
```

학습 중 teacher-forcing validation loss는 대회 exact match와 맞지 않아 계산하지
않습니다. `checkpoint-*` 어댑터를 모두 보존하므로 아래 평가 명령의
`--adapter-path`를 각 checkpoint로 바꿔 exact match가 가장 높은 하나를 선택합니다.

학습 전 짧은 smoke test는 다음처럼 실행합니다.

```bash
snu-train --limit 48 --max-steps 3 --gradient-accumulation-steps 1 \
  --save-steps 3 --output-dir outputs/strategy-smoke
```

저장된 LoRA를 동일한 추론 코드로 held-out validation에 평가합니다.

```bash
snu-infer \
  --test-csv data/train.csv \
  --image-dir data/train \
  --adapter-path outputs/qwen2-vl-lora/final \
  --validation-fraction 0.2 \
  --output outputs/validation_predictions.csv \
  --metrics-output outputs/validation_metrics.json
```

먼저 소수 샘플로 입출력과 VRAM 사용량을 확인합니다.

```bash
snu-infer --limit 5
```

전체 테스트 추론은 다음과 같습니다.

```bash
snu-infer \
  --data-dir data \
  --model-path models/Qwen2-VL-2B-Instruct \
  --adapter-path outputs/qwen2-vl-lora/final \
  --output outputs/submission.csv \
  --audit-log outputs/raw_predictions.jsonl
```

모델 로딩은 기본적으로 네트워크를 차단합니다. `--allow-network`는 개발 중 명시적으로
필요한 경우에만 사용하고, 최종 제출 전에는 반드시 옵션 없이 오프라인 재현을 확인하세요.

생성물은 다음 파일입니다.

- `outputs/submission.csv`: `Id,Answer` 형식의 제출 파일
- `outputs/raw_predictions.jsonl`: 원시 모델 출력, 파싱 성공 여부, 최종 답변 감사 로그
- `outputs/validation_metrics.json`: exact match, identity/non-identity 정확도, 파싱 실패율

모델이 올바른 순열을 출력하지 못하면 베이스라인과 동일하게 `[1, 2, 3, 4]`를
사용하되 실패 건수를 마지막에 표시하고 감사 로그에 `parse_ok=false`로 남깁니다.

## 테스트

```bash
pip install -r requirements-dev.txt
pip install -e .
ruff check src tests scripts
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
