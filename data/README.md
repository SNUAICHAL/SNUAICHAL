# Data directory

대회 데이터는 Git에 올리지 않습니다. 운영진이 제공한 파일을 다음과 같이 배치하세요.

```text
data/
├── test.csv
├── train.csv                 # 제공되는 경우
├── test/
│   └── <Id>/
│       ├── <Input_1 filename>
│       └── ...
└── train/                    # 제공되는 경우
```

코드에서는 저장소 루트 기준 상대 경로만 사용합니다. 평가 데이터의 내용이나 특성을
학습·전처리·모델 설계에 반영하지 마세요.

`test.csv` 열은 정확히 다음 순서여야 하며 `Answer`를 포함하면 안 됩니다.

```text
Id,Input_1,Input_2,Input_3,Input_4,Sentence
```

운영진 외부 평가 데이터도 같은 구조로 배치합니다. 이 데이터는 frozen
checkpoint-2726 추론에만 사용하며 추가 학습, tuning, model selection이나
전처리 설계에 사용하지 않습니다.

CUDA를 사용하기 전 schema, unique ID, 4개 image 존재와 path containment를
검증합니다.

```bash
python -B -m scripts.verify_evaluation_package --data-dir data
```
