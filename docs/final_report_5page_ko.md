# 순열-등변 Latin4와 24 GB QLoRA를 이용한 4-프레임 시간 순서 추론

**SNU AI Challenge 2026 최종 보고서**

**팀 SNUAICHAL · 대표 한지후 · github.com/SNUAICHAL/SNUAICHAL**

## 1. 문제 정의와 최종 시스템

본 과제는 문장 하나와 무작위 순서로 주어진 네 장의 이미지로 원래 시간 순서를
복원하는 문제다. 정답은 원본 프레임 번호의 나열이 아니라, **현재 입력된 각 이미지가
시간축의 몇 번째인지**를 나타내는 4-순열이다. 한 위치라도 틀리면 전체 샘플이
오답인 Exact Match이므로, 국소적인 frame 비교뿐 아니라 네 장의 전역적인 일관성이
필요하다.

우리의 최종 시스템은 다음과 같다.

`deterministic index shuffle + answer transform → Qwen3.6-27B rank32 QLoRA →
balanced cyclic Latin4 → original-slot canonicalization → hard majority →
lexicographic tie`

### 최종 모델 계약

- Base: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- 학습: 제공된 9,535행 전체, optimizer step 2,726, 2.28694 epochs
- Adapter: QLoRA rank 32, alpha 32, dropout 0.05, unmerged
- 추론: NF4 base, BF16 compute, SDPA, image size 512, batch 1
- 생성: `max_new_tokens=64`, seed 42, thinking disabled, 순열 하나만 출력
- TTA: `1234, 2341, 3412, 4123`, hard vote, lexicographic tie

### 핵심 기여

1. **순열-등변 좌표 처리.** 입력 순서를 바꿔 얻은 예측을 그대로 투표하지 않고 원래
   slot 좌표로 역변환한 뒤 집계한다.
2. **학습–추론 대칭성.** 학습 중 입력 index를 결정론적으로 섞고 정답도 같은 좌표계로
   변환해 위치 shortcut을 줄였다.
3. **최소 균형 TTA.** 네 cyclic view만으로 각 원본 이미지가 네 displayed position에
   정확히 한 번씩 등장한다. 24-view보다 6배 저렴하면서 실측 점수도 높았다.
4. **실제 24 GB 배치.** 27B base를 NF4로, adapter를 merge하지 않은 채 단일 RTX
   3090에서 실행하고 process 외부 NVML sampler로 물리 peak를 측정했다.
5. **재현 가능한 선택.** model tree, adapter, exact 명령, 입력 image tree,
   release source, raw output과 CSV를 SHA-256 receipt로 연결하고, 더 많은
   epoch/view가 항상 낫다는 가정을 실제 ablation으로 반박했다.

최종 Public score는 **0.93542**다. Public은 test의 70%만 사용하므로 이 점수를
Private 또는 외부 데이터 성능의 보장으로 해석하지 않는다. 그 대신 외부 평가에도
동일한 checkpoint-2726과 동일 추론 코드로 단일 accepted run을 수행한다.

<div class="page-break"></div>

## 2. 데이터 활용과 학습 방법

### 결정론적 index-shuffle augmentation

네 이미지의 의미는 같아도 입력 slot은 임의적이다. 고정 slot로만 학습하면 모델이
내용이 아니라 위치 prior를 이용할 수 있다. 각 `(seed, epoch, Id)`를 SHA-256으로
hash해 입력 permutation을 결정하고, 이미지 배열과 Answer를 동시에 변환했다. 같은
seed와 data에서 매번 동일하게 재현되며, test-time에 적용하는 좌표 canonicalization과
수학적으로 같은 변환을 사용한다.

### QLoRA 학습

Qwen3.6-27B의 vision-language 사전학습 능력을 유지하면서 24 GB 추론 제약에
맞추기 위해 4-bit QLoRA를 사용했다. vision encoder는 freeze하고 language의
`q/k/v/o`와 `gate/up/down` projection에 rank32 adapter를 적용했으며 base는
quantized 상태로 유지했다. 3 epochs horizon, learning
rate 1e-4, cosine scheduler, batch 1, accumulation 8, BF16 compute, image area
512×512 상당, seed 42를 사용했다. 최종 checkpoint는 global step 2,726이며 adapter
tensor 512개가 모두 finite임을 확인했다.

최종 run은 labeled 9,535행 전체를 학습에 사용했다. 따라서 이 모델을 같은 데이터의
holdout으로 사후 평가하면 누수가 된다. Qwen3.6 checkpoint 선택에는 그 수치를 쓰지
않았으며, Public 제출과 비용·안정성 ablation만 사용했다. 초기 설계 단계의 clean954
split은 8B/Qwen3.5 파이프라인과 permutation 방법을 검증하는 surrogate로만 사용했다.

### Checkpoint-axis 선택

epoch 증가가 단조 성능 향상을 보장하지 않는다는 관찰을 반영해 checkpoint를 하나의
명시적인 탐색 축으로 취급했다. 마지막 구간에서 2,726과 3,400을 end-to-end 제출
후보로 보존했다. checkpoint-3400 TTA12는 0.93542로 checkpoint-2726 Latin4와
동점이었다. checkpoint와 view 수가 함께 달라지는 operational comparison이므로
순수 학습-step causal ablation은 아니다. 동점 후보 중 계산량이 작은 2,726을
사전 규칙으로 채택했다.

### 외부 자원과 데이터

모델 개발·학습·선택·튜닝에는 운영진 제공 학습 데이터만 사용했다. 외부 상용 API,
외부 labeling API, 외부 학습 데이터셋, 다른 모델의 pseudo-label, row-level model
mixing은 사용하지 않았다. 운영진 별도 외부 평가는 freeze 후 inference에만 쓴다.
Base는 cutoff 이전 공개된 Apache-2.0 Qwen3.6-27B 한 개다. RunPod A6000은 학습과
checkpoint-3400 ablation 추론에, 로컬 3090은 최종 checkpoint-2726 추론에 썼다.

<div class="page-break"></div>

## 3. Canonical Latin4 추론과 자원 최적화

### 좌표계 문제

예를 들어 view `2341`에서 모델이 출력한 label은 그 view 안의 위치를 의미한다.
이를 원본 `1234` view의 label과 직접 투표하면 서로 다른 좌표계를 합치는 오류가
생긴다. 각 view 출력 `p`와 입력 permutation `v`로부터 원래 네 이미지의 시간
순서를 다시 계산한 뒤 canonical Answer로 변환했다. parser는 `[1,2,3,4]`의 순열만
허용하고, 설명문이나 중복 숫자는 실패로 기록한다.

view order를 `g=(g_1,...,g_4)`, view-slot prediction을 `p`라 하면 역변환은
`C_g(p)[g_j]=p[j]`다. 이 식을 학습 Answer 변환과 TTA audit 양쪽에서 같은 함수로
검사했다.

### 왜 Latin4인가

`1234, 2341, 3412, 4123`은 각 원본 이미지가 각 displayed position에 정확히 한
번씩 나타나는 Latin square다. 따라서 원본 frame과 displayed position 사이의 1차
주변 노출은 네 view만으로 균형이다. 이는 고차 순서 상호작용이나 model bias가
0이라는 뜻은 아니다. 노출 행렬은 `N(i,j)=Σ_g 1[g_j=i]=1`이다. 24개 모든
permutation은 coverage를 늘리지만 계산량도 6배다.

| 단일 checkpoint 비교 | Public | views | 결과 |
|---|---:|---:|---|
| ckpt2726 Latin4 confidence tie-break | 0.92670 | 4 | hard보다 낮음 |
| **ckpt2726 Latin4 hard** | **0.93542** | **4** | **최종 선택** |
| ckpt2726 full TTA24 hard | 0.93193 | 24 | Latin4 대비 -0.00349 |
| ckpt3400 TTA12 hard | 0.93542 | 12 | 동점, 비용 3배 |

confidence tie-break은 생성 token confidence가 정답 확률로 잘 보정됐다는 보장이
없어 오히려 점수를 낮췄다. 최종 규칙은 top vote가 하나면 그것을, exact tie이면
canonical permutation의 lexicographic minimum을 고른다. Public의 개별 문항 정답을
역추정하거나 수동으로 행을 고치지 않았다.

### 24 GB 실행

Base를 NF4, compute를 BF16, attention을 SDPA로 두고 batch 1, 단일 resident
model로 순차 처리했다. adapter merge, dequantization, multi-model ensemble은
사용하지 않았다. checkpoint-2726 Latin4 전체 819행의 결과는 다음과 같다.

- 819/819 완료, 3,276 views, parse failure 0
- generation loop 평균 16.4693초/row, 819행 환산 약 3시간 45분
- model load·검증·종료 포함 end-to-end 13,990.456초, 약 3시간 53분
- physical peak 22,710,861,824 bytes(약 21.15 GiB)
- 예상 밖 CUDA process 0

PyTorch allocator 값만으로는 driver와 non-PyTorch allocation을 놓칠 수 있어,
model load 이전부터 작업 종료 이후까지 NVML device-used memory를 0.5초 간격으로
측정했다. 이는 대회 단일 RTX 3090 24 GB 제약 안에서 실제 실행 가능함을 보인다.

<div class="page-break"></div>

## 4. 실험 결과, 오류 분석, 비용

### Ablation

| 축 | 비교 | 관찰 | 결론 |
|---|---|---:|---|
| 모델/초기 checkpoint | Q35-1073 Latin4 confidence | 0.92844 | 27B fallback |
| Q36 초기 probe | step1074 Latin4 confidence | 0.93019 | 최종 trend 단독 근거 아님 |
| 집계 | step2726 confidence → hard | 0.92670 → 0.93542 | confidence 제거 |
| view 수 | step2726 TTA4 → TTA24 hard | 0.93542 → 0.93193 | 추가 view가 악화 |
| 학습·view | step2726 TTA4 → step3400 TTA12 | 0.93542 → 0.93542 | 추가 비용 무이득 |

위 표는 Public 70%에 대한 제한된 관찰이다. 여러 제출 중 최댓값을 고르는 과정 자체가
Public에 과적합될 수 있다. 따라서 최종 코드는 최댓값을 만든 복잡한 조합이 아니라,
동점 후보 중 더 이른 checkpoint와 더 적은 view를 고정했다. Private와 외부
데이터에서는 이 사전 고정 규칙을 변경하지 않는다.

### 오류 분석

가장 분명한 오류 축은 **입력 permutation sensitivity**였다. 초기 모델에서 view별
예측이 달라지는 행이 많았고, canonical multi-view는 이 변동을 줄였다. 남은 오류는
미세한 상태 변화, 반복 동작, 장면 전환과 카메라 이동, 문장과 시각 단서의 불일치가
섞여 있을 것으로 예상한다. 다만 test label이 없으므로 개별 test 행의 오류 유형을
정답인 것처럼 단정하지 않았다.

추론 자체의 technical failure는 최종 run에서 없었다. 819행 모두 네 view가 유효한
순열로 parse됐고 fallback identity가 사용되지 않았다. TTA24가 16개 행에서 Latin4
hard와 다른 답을 만들었으나 Public score는 0.00349 낮았다. 이는 더 많은 변환이
항상 robustness를 높이지 않으며, 균형 이후의 marginal view quality가 중요하다는
실험적 근거다.

### 구축 비용

- API 비용: 0원. 외부 상용 API를 사용하지 않음
- 학습: 사비 RunPod RTX A6000, 표시 단가 USD 0.53/hour
- 추론: 보유 RTX 3090 한 장
- 최종 adapter: 약 638 MB; base snapshot: 약 55.6 GB
- 최종 inference: generation 약 3.75 GPU-hours, end-to-end 약 3.89 hours

Provider invoice와 가정하지 않은 전기·감가상각 비용은 정확한 총액으로 환산하지
않았으며, 존재하지 않는 비용 수치를 추정해 성과로 제시하지 않았다. 계산 절약의
핵심은 Latin4, single checkpoint, unmerged adapter,
checkpoint-2726 조기 종료다. TTA24 대신 Latin4를 쓰면 최종 추론량을 83.3%
줄이면서 Public 점수는 오히려 높아졌다.

<div class="page-break"></div>

## 5. 재현성, 장점과 한계

### 재현 절차

GitHub 저장소는 base model manifest, adapter manifest, 최종 wrapper와 단위
테스트를 포함한다. Base는 29개 파일/15개 shard의 size와 SHA-256을 확인하고,
adapter release는 archive와 내부 두 파일을 모두 검증한다.

- Base: `python -B scripts/download_weights.py --manifest
  configs/weights/qwen36-27b-final.manifest.json --output models/Qwen3.6-27B`
- Adapter: `python -B scripts/download_final_adapter.py --output
  weights/qwen36-checkpoint2726`
- Preflight: `python -B -m scripts.verify_evaluation_package`
- Inference: `python -B -m scripts.run_final_inference --data-dir data
  --output-dir outputs/external-evaluation`

최종 wrapper가 NF4/BF16/SDPA/512/64/seed42, Latin4 orders, original-coordinate
canonicalization, hard vote를 명시한다. 결과는 `submission.csv`, `audit.jsonl`,
`metrics.json`, `reproduction-receipt.json`으로 저장된다. 공식 wrapper는 과거
row의 resume을 허용하지 않고 Answer 열을 CUDA load 전에 거부한다. raw audit로
parser와 집계를 독립 재계산하며 exact child command, 입력 image tree와 release
source도 receipt에 결합한다. release reproduction 핵심 source SHA-256은
inference `7ab46f47…d414`, TTA
`c3a3df4f…7bc2`, parser `e2f9844b…71c8`, modeling `3fcd1e73…4778`이다.

### 장점

- 입력 순열 변화에 수학적으로 일관된 canonicalization
- 네 view만으로 완전한 position balance
- 27B 모델을 실제 24 GB GPU에서 실행
- 단일 checkpoint·단일 model로 규정과 외부 평가 재현성이 명확
- 성능이 나빠진 confidence/TTA24/추가 epoch도 결과에 포함한 정직한 ablation

### 한계

- 최종 Q36은 full-data 학습이라 leakage-free 동일 모델 holdout 비교가 없다.
- Public 70%의 반복 관찰 때문에 모델 선택 overfitting 가능성이 있다.
- 다중 seed, fold별 안정성, 대규모 수동 semantic error annotation이 없다.
- rank32와 rank8의 동일 checkpoint·동일 split 대조가 없다.
- Private 및 외부 데이터 성능은 최종 평가 전까지 알 수 없다.

### 결론

최종 시스템은 가장 큰 모델이나 가장 많은 TTA를 단순히 택한 것이 아니다. **학습과
추론의 좌표 변환을 일치시키고, position balance를 만족하는 최소 Latin4를 사용하며,
실제 RTX 3090 메모리 안에서 재현 가능한 checkpoint-2726을 고정**했다. 더 긴
checkpoint와 24-view는 추가 비용에도 개선되지 않았고 confidence 집계는 악화됐다.
이 결과를 바탕으로 성능, 논리적 타당성, 자원 효율성과 재현성을 함께 갖춘 단일
모델을 외부 데이터 평가와 코드 검증에 제출한다.

---

상세 수치·SHA·artifact 출처는 `docs/final_results.json`, 가중치 저작권은
`docs/model_licenses.md`, 실행 지침은 저장소 `README.md`에 기록했다.

참고: Qwen Team, *Qwen3.5/Qwen3.6 model card* (2026); Dettmers et al.,
*QLoRA* (NeurIPS 2023); Hu et al., *LoRA* (ICLR 2022).
