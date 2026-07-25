# 최종 코드·보고서 제출 체크리스트

기준 공지: 2026-07-24 확인본. 보고서·코드 제출 마감은 2026-07-28,
보고서는 PDF 최대 5페이지이며 Public 상위 30팀은 제출 필수입니다. 최종 판단은
운영진 최신 공지를 우선합니다.

## 완료된 release gate

- [x] 실행 가능한 학습·추론 Python source 포함
- [x] Python 3.10 및 pinned library 환경 기록
- [x] 설치, base/adapter 다운로드, inference, CSV audit 명령을 README에 기록
- [x] Qwen3.6 base revision과 29-file/15-shard SHA-256 manifest 포함
- [x] checkpoint-2726 adapter를 GitHub Release asset으로 제공
- [x] adapter archive·내부 파일 SHA-256 검증 downloader 포함
- [x] 최종 모델을 Q36-2726 rank32/alpha32 하나로 고정
- [x] 최종 추론을 Latin4 hard-majority 하나로 고정
- [x] 819행, 3,276 views, parse failure 0 기록
- [x] RTX 3090 physical peak 약 21.15 GiB 기록
- [x] generation 약 3 h 45 m, end-to-end 약 3 h 53 m 기록
- [x] Public 0.93542 CSV SHA를 `docs/final_results.json`에 기록
- [x] 외부 상용 API·외부 학습 데이터·pseudo-label·model ensemble 미사용 선언
- [x] 대회 데이터와 test prediction 원문을 GitHub에 미포함
- [x] 한국어 최종 보고서 PDF 정확히 5페이지
- [x] LaTeX report source, machine-readable result, model license attribution 포함
- [x] lightweight CI에서 Ruff, compile, full unit test 실행
- [x] 외부 평가용 무라벨 data/path preflight와 semantic CSV auditor 포함
- [x] 외부 평가 exactly-once frozen inference 절차 기록

## 운영진 외부 데이터 1회 평가

- [ ] frozen Git commit/tag와 adapter release SHA를 receipt에 기록
- [ ] 별도 학습·tuning·model selection을 수행하지 않음
- [ ] `Answer` 없는 공식 external `test.csv`를 새 `data/`에 배치
- [ ] `scripts/verify_evaluation_package.py --require-all` PASS
- [ ] 새 output directory에서 `scripts.run_final_inference` exactly once 실행
- [ ] `reproduction-receipt.json`과 독립 CSV auditor PASS
- [ ] 결과를 본 뒤 checkpoint/prompt/TTA/aggregation을 변경하지 않음
- [ ] 운영진 Kaggle에 정확히 1회 제출하고 receipt 보존

## GitHub·Google Form 제출 전 사용자 확인

- [ ] GitHub 초대가 pending이 아니라 `SNUAIchallenge: read`로 수락됐는지 확인
- [ ] Google Form에 대표자 정보, GitHub URL과 최종 PDF 업로드
- [ ] Form에 제출한 Git commit/tag와 Release URL 별도 보존
- [ ] private Release asset을 운영진 collaborator 계정이 다운로드 가능한지 확인
- [ ] 제출 완료 화면 또는 이메일 receipt 보존

## 심사 해석 주의

- Public 0.93542는 test 70% 관찰이며 Private/외부 점수 보장이 아닙니다.
- final full-data QLoRA에는 leakage-free 동일-model holdout이 없습니다.
- Latin4는 first-order frame-position exposure를 균형화하지만 모든 position bias를
  제거했다고 주장하지 않습니다.
- Q36-2726 Latin4와 Q36-3400 TTA12는 checkpoint와 view가 함께 다른 operational
  후보 비교입니다.
- 실제 RunPod 총 연구비는 billing receipt 없이 추정값을 확정 금액으로 쓰지 않습니다.
