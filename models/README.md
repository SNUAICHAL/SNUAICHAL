# Model weights

모델 가중치는 Git에 올리지 않습니다. 인터넷이 차단된 평가 환경에 들어가기 전에
`Qwen/Qwen2-VL-2B-Instruct`의 전체 스냅샷을 아래 경로에 복사하세요.

```text
models/Qwen2-VL-2B-Instruct/
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── tokenizer_config.json
├── model-*.safetensors
└── ...
```

추론기는 기본적으로 `local_files_only=True`로 로드합니다. 제출 전 다음을 확인하세요.

- 가중치 공개일이 대회 기준일(2026-05-31) 이전인지 근거 링크와 함께 기록
- 모델 라이선스 및 재배포 조건 확인
- 코드와 가중치를 합친 전체 크기가 80 GB 이하인지 확인
- 깨끗한 오프라인 환경에서 실제 로딩 및 전체 추론 완료 확인

