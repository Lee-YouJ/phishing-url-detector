# Phishing URL Detector

URL을 정상(0)과 피싱(1)으로 분류하는 이진 분류 머신러닝 baseline 프로젝트입니다.

---

## 프로젝트 구조

```
phishing-url-detector/
├── data/
│   ├── final_url_dataset.csv        # 원본 데이터셋 (수정 금지)
│   └── final_url_dataset_v2.csv     # 보강 데이터셋 (학습에 사용)
├── notebooks/
│   └── preprocess.ipynb             # 전처리 참고 노트북
├── src/
│   ├── feature_extractor.py         # URL 문자열 → feature 추출
│   ├── train.py                     # 모델 학습 및 평가
│   ├── predict.py                   # URL 예측 및 위험도 점수 출력
│   ├── utils.py                     # 공통 함수
│   └── experiment_no_https.py       # has_https 제거 ablation 실험
├── models/                          # 학습된 모델 (.pkl) — Git 미포함
├── reports/
│   ├── model_comparison_v2.csv      # 최종 모델 성능 비교표
│   ├── model_comparison_canonical.csv
│   ├── threshold_analysis.csv       # threshold별 성능 분석
│   ├── url_prediction_test_v2.csv   # 9개 실사용 URL 테스트 결과
│   ├── model_report.md              # 성능 보고서 (threshold 선택 근거 포함)
│   └── images/                      # Confusion Matrix, Feature Importance, threshold 그래프
├── requirements.txt
└── README.md
```

> **`models/` 와 `data/` 디렉터리는 `.gitignore`에 포함되어 GitHub에 업로드되지 않습니다.**
> 팀원은 아래 학습 명령어를 실행해 모델을 직접 생성해야 합니다.

---

## 데이터 설명

### 원본 데이터셋 (`data/final_url_dataset.csv`)

| 항목 | 내용 |
|------|------|
| 전체 행 수 | 73,328 |
| 클래스 분포 | 정상(0): 36,664 / 피싱(1): 36,664 (균형) |
| 결측치 / 중복 | 없음 |

### 보강 데이터셋 (`data/final_url_dataset_v2.csv`) — 학습에 사용

원본 데이터의 정상 URL은 대부분 `https://domain.com` 형태(path 없음, www 없음)로만 구성되어 있어, 아래 세 가지 패턴의 정상 샘플이 구조적으로 결핍되어 있었습니다.

| 결핍 패턴 | 원본 정상 샘플 | 보강 후 |
|-----------|--------------|---------|
| path 있는 URL (`num_slashes ≥ 3`) | 0건 | 추가됨 |
| subdomain 있는 URL (`subdomain_count ≥ 1`) | 매우 적음 | 추가됨 |
| `login`·`account` 경로 포함 정상 URL | 0건 | 추가됨 |

정상 URL 99건 + 피싱 99건을 추가해 클래스 균형을 유지했습니다 (최종: 정상 36,763 / 피싱 36,763).

---

## feature 추출 (`src/feature_extractor.py`)

`feature_extractor.py`는 URL 문자열 하나를 받아 19개 숫자형 feature를 추출하는 핵심 모듈입니다.
학습(`train.py`)과 예측(`predict.py`) 모두 이 모듈을 사용하여 feature 추출 방식을 일치시킵니다.

### 주요 처리 로직

**1. www 정규화 (`_canonical_url`)**
- host의 leading `www.`를 제거한 canonical URL 기준으로 feature를 계산합니다.
- `www.naver.com`과 `naver.com`이 동일한 feature 값을 가집니다.

**2. `has_suspicious_keyword` — path 부분만 검사**
- `login`, `verify`, `update`, `secure`, `account` 키워드를 URL의 path에서만 탐지합니다.
- 도메인 이름 자체에 keyword가 포함된 경우(`accounts.google.com`)를 피싱으로 오탐하지 않습니다.

### feature 목록 (19개)

| feature | 설명 |
|---------|------|
| `url_length` | URL 전체 길이 (www 정규화 기준) |
| `domain_length` | 도메인 길이 |
| `path_length` | 경로 길이 |
| `query_length` | 쿼리 문자열 길이 |
| `num_digits` | 숫자 문자 개수 |
| `digit_ratio` | 숫자 비율 |
| `num_dots` | 점(.) 개수 |
| `num_slashes` | 슬래시(/) 개수 |
| `num_hyphens` | 하이픈(-) 개수 |
| `num_at` | @ 기호 개수 |
| `num_question_marks` | 물음표(?) 개수 |
| `num_ampersand` | & 기호 개수 |
| `num_special_chars` | 특수 문자 개수 |
| `subdomain_count` | 서브도메인 수 (www 제외) |
| `tld_length` | TLD(최상위 도메인) 길이 |
| `has_punycode` | 퓨니코드 포함 여부 (0/1) |
| `has_suspicious_keyword` | path에 의심 키워드 포함 여부 (0/1) |
| `has_https` | HTTPS 사용 여부 (0/1) |
| `has_ip` | IP 주소 포함 여부 (0/1) |

---

## 모델 설명

| 모델 | 특징 |
|------|------|
| **Logistic Regression** | 선형 분류기, StandardScaler 적용, 빠른 학습 |
| **Random Forest** | 앙상블 트리, feature importance 제공, **기본 사용 모델** |
| **XGBoost** | 그래디언트 부스팅 (미설치 시 HistGradientBoosting으로 대체) |

---

## 실행 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
python src/train.py
```

학습 완료 후 생성 파일:
- `models/*.pkl` — 학습된 모델 (Random_Forest, Logistic_Regression, XGBoost)
- `reports/model_comparison_v2.csv` — 성능 비교표
- `reports/model_report.md` — 성능 보고서
- `reports/images/cm_*.png` — Confusion Matrix
- `reports/images/fi_Random_Forest.png` — Feature Importance

### 3. 실제 URL 예측

#### 단일 URL 예측

```bash
python src/predict.py --url "https://www.naver.com"
```

#### 여러 URL 동시 예측

```bash
python src/predict.py --url "https://github.com/login" "http://login-verify.xyz/update"
```

#### feature 값 함께 출력

```bash
python src/predict.py --url "https://accounts.google.com" --show-features
```

#### CSV 파일 일괄 예측

```bash
python src/predict.py --input data/final_url_dataset_v2.csv --output reports/predictions.csv
```

#### 모델 지정 (기본값: Random_Forest)

```bash
python src/predict.py --url "https://example.com" --model XGBoost
```

#### 인수 없이 실행 (데모)

```bash
python src/predict.py
```

---

## 평가 지표 설명

| 지표 | 설명 |
|------|------|
| **Accuracy** | 전체 샘플 중 올바르게 분류한 비율 |
| **Precision** | 피싱으로 예측한 것 중 실제 피싱인 비율 |
| **Recall** | 실제 피싱 URL 중 피싱으로 탐지한 비율 **(핵심 지표)** |
| **F1-score** | Precision과 Recall의 조화 평균 |

> **Recall이 핵심 지표입니다.**
> 피싱 URL을 정상으로 잘못 예측하는 **False Negative**를 최소화하는 것이 목표입니다.

---

## 위험도 점수

모델의 `predict_proba`를 이용해 피싱일 확률을 계산하고 0~100점의 위험도 점수로 변환합니다.

```
risk_score = phishing_probability × 100
```

| 위험도 점수 | 등급 |
|------------|------|
| 0 ~ 30 | 안전 |
| 31 ~ 70 | 주의 |
| 71 ~ 100 | 위험 |

---

## 분류 Threshold 선택 근거 (threshold = 0.6)

기본 threshold 0.5는 학습 데이터 편향(정상 URL에 www/path/subdomain 패턴 결핍)으로 인해
`accounts.google.com` 같은 실제 정상 URL을 피싱으로 오탐하는 문제가 있었습니다.

threshold를 0.3~0.8 구간에서 0.05 간격으로 분석한 결과:

| Threshold | Precision | Recall | FP (오탐) | FN (미탐) | 비고 |
|-----------|-----------|--------|-----------|-----------|------|
| 0.50 | 0.9988 | 0.9959 | 9 | 30 | 기본값 |
| **0.60** | **0.9995** | **0.9952** | **4** | **35** | **선택값** |
| 0.65 | 0.9995 | 0.9952 | 4 | 35 | 실사용 테스트 8/9 |
| 0.70 | 0.9997 | 0.9952 | 2 | 35 | — |

> 전체 분석 결과: `reports/threshold_analysis.csv`
> 시각화 그래프: `reports/images/threshold_analysis.png`

**threshold = 0.6을 선택한 이유:**

- FP(정상 → 피싱 오탐)가 9건 → 4건으로 **55% 감소**해 서비스 신뢰도를 높임
- Recall 하락은 0.9959 → 0.9952로 **0.0007에 불과**해 피싱 탐지 성능을 사실상 유지
- 0.65 이상은 FP·FN이 동일하거나 실사용 테스트(9개 URL)에서 피싱 URL을 놓치는 케이스가 발생
- 9개 실사용 URL 테스트에서 **유일하게 9/9 전부 통과**한 threshold

---

## 결과 파일 설명

| 파일 | 내용 |
|------|------|
| `reports/model_comparison_v2.csv` | 최종 모델 성능 비교 (v2 데이터 기준) |
| `reports/threshold_analysis.csv` | threshold 0.30~0.80 구간별 성능 분석 |
| `reports/url_prediction_test_v2.csv` | 9개 실사용 URL 예측 결과 (9/9 통과) |
| `reports/model_report.md` | 데이터 요약, 성능 비교, threshold 선택 근거 |
| `reports/images/cm_*.png` | 모델별 Confusion Matrix |
| `reports/images/fi_Random_Forest.png` | Random Forest 상위 10개 Feature Importance |
| `reports/images/threshold_analysis.png` | Threshold별 Precision/Recall/F1 + FP/FN 그래프 |

---

## Git 포함 여부

| 경로 | Git 포함 | 이유 |
|------|----------|------|
| `src/` | O | 코드 |
| `reports/` | O | 분석 결과 |
| `data/` | X | 대용량 데이터 파일 |
| `models/*.pkl` | X | 대용량 바이너리, `train.py`로 재생성 가능 |
| `.venv/` | X | 가상환경 |

> 팀원은 저장소 클론 후 `pip install -r requirements.txt` → `python src/train.py` 순서로 모델을 생성하세요.
