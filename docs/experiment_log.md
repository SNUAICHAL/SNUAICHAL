# Experiment log

실험마다 아래 블록을 복사해 기록하세요. 비용이 0원이어도 `0 KRW`로 명시합니다.

## EXP-YYYYMMDD-001

- 목적:
- Git commit:
- 실행 명령:
- 데이터 버전/해시:
- 모델 이름 및 로컬 경로:
- 원본 가중치 공개일/출처:
- 시드:
- 환경: OS / CPU / RAM / GPU / driver / CUDA / Python
- 라이브러리: `pip freeze` 파일 경로
- 학습 시간 / 추론 시간:
- 최대 VRAM:
- 검증 방식 및 점수:
- 전처리용 외부 API 이름/사용 방식 (학습·추론 사용 금지):
- 외부 API 호출 수/비용: 0 KRW
- 산출물 경로/해시:
- 비고:

## EXP-20260714-001

- 목적: 층화 split, 24순열 균형 증강, LoRA 저장·재로딩, exact-match 평가의 end-to-end smoke test
- Git commit: `f9b339d` 위 작업 트리 변경 포함
- 실행 명령: `snu-train --limit 48 --max-steps 3 --gradient-accumulation-steps 1 --save-steps 3 --logging-steps 1 --output-dir outputs/strategy-smoke`
- 데이터: 공식 train 9,535행 중 첫 48행을 split 전 선택
- 모델: `Qwen/Qwen2-VL-2B-Instruct`, `models/Qwen2-VL-2B-Instruct`
- 시드: 42
- 환경: Windows / RTX 3090 24,576 MiB / driver 610.62 / CUDA 12.4 / Python 3.10.11
- 라이브러리: torch 2.5.1+cu124, transformers 4.49.0, peft 0.14.0
- 학습 시간: 535.84초 (GPU가 별도 그래픽 프로세스와 경쟁 중이어서 성능 기준으로 사용 불가)
- 검증 추론: held-out 5개 제한, exact match 0.20, identity 1.00, non-identity 0.00, parse failure 0.00
- 외부 API 호출 수/비용: 0 KRW
- 어댑터: `outputs/strategy-smoke/final/adapter_model.safetensors`
- 어댑터 SHA-256: `fe647c1e617a74ad38cd7c228d3ed759499784200d4d4ac8ce89ce7ee64020ba`
- 지표 SHA-256: `db83c8ad466fe201035c09defa6af8e4276e29f389edf5f8965749cabe3bd2c8`
- 비고: 기능 검증용 3-step 결과이며 모델 선택이나 성능 비교에 사용하지 않음. 이 결과는 clean holdout 도입 전 5% split을 사용했으므로 폐기 대상임. 실행 중 다른 GPU 작업이 약 11 GiB와 99% GPU를 점유해 전체 학습을 보류함.
