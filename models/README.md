# Model weights

모델 가중치는 Git에 올리지 않으며 학습/추론 코드도 자동 다운로드하지 않습니다.
인터넷이 허용된 준비 환경에서 필요한 모델 하나만 immutable revision으로 받습니다.

Qwen3-VL-8B-Instruct (Apache-2.0, 2025-10-11 공개, 약 17.55 GB):

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --local-dir models/Qwen3-VL-8B-Instruct
```

Qwen3.5-27B dense challenger (Apache-2.0, 2026-02-24 공개, 약 55.58 GB):

```bash
hf download Qwen/Qwen3.5-27B \
  --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --local-dir models/Qwen3.5-27B
```

Qwen3.5-27B는 8B pipeline과 2-step smoke가 검증된 뒤에만 받는 것을 권장합니다.
4-bit loading은 메모리에서 quantize할 뿐 원본 snapshot 다운로드 크기를 줄이지 않습니다.

스냅샷 구조 예시:

```text
models/Qwen3-VL-8B-Instruct/
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── model-*.safetensors
└── ...
```

오프라인 검증:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
snu-infer --model-path models/Qwen3-VL-8B-Instruct --limit 1
```

추론기는 기본적으로 `local_files_only=True`로 로드합니다. 제출 전 다음을 확인하세요.

- 가중치 공개일이 대회 기준일(2026-05-31) 이전인지 근거 링크와 함께 기록
- 모델 라이선스 및 재배포 조건 확인
- 코드와 가중치를 합친 전체 크기가 80 GB 이하인지 확인
- 깨끗한 오프라인 환경에서 실제 로딩 및 전체 추론 완료 확인

