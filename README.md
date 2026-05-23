# Phishing URL Detector

URL을 정상(0)과 피싱(1)으로 분류하는 이진 분류 머신러닝 baseline 프로젝트입니다.

---

## 프로젝트 구조

```
phishing-url-detector/
├── data/
│   └── final_url_dataset.csv       # 원본 데이터셋
├── notebooks/
│   └── preprocess.ipynb            # 전처리 참고 노트북
├── src/
│   ├── train.py                    # 모델 학습 및 평가
│   ├── predict.py                  # 예측 및 위험도 점수 출력
│   └── utils.py                    # 공통 함수
├── models/                         # 학습된 모델 저장 (.pkl)
├── reports/
│   ├── model_comparison.csv        # 모델 성능 비교표
│   ├── model_report.md             # 모델 성능 보고서
│   └── images/                     # Confusion Matrix, Feature Importance 이미지
├── requirements.txt
└── README.md
```

---

## 데이터 설명

| 항목 | 내용 |
|------|------|
| 파일 | `data/final_url_dataset.csv` |
| 전체 행 수 | 73,328 |
| 클래스 분포 | 정상(0): 36,664 / 피싱(1): 36,664 (균형 데이터) |
| 결측치 | 없음 |
| 중복 | 없음 |

### 학습에 사용하는 feature (숫자형 19개)

| feature | 설명 |
|---------|------|
| `url_length` | URL 전체 길이 |
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
| `subdomain_count` | 서브도메인 수 |
| `tld_length` | TLD(최상위 도메인) 길이 |
| `has_punycode` | 퓨니코드 포함 여부 (0/1) |
| `has_suspicious_keyword` | 의심 키워드 포함 여부 (0/1) |
| `has_https` | HTTPS 사용 여부 (0/1) |
| `has_ip` | IP 주소 포함 여부 (0/1) |

> `url` 컬럼(문자열)은 학습 feature에서 제외합니다.

---

## 모델 설명

3가지 모델을 baseline으로 비교합니다.

| 모델 | 특징 |
|------|------|
| **Logistic Regression** | 선형 분류기, StandardScaler 적용, 빠른 학습 |
| **Random Forest** | 앙상블 트리, feature importance 제공 |
| **XGBoost** | 그래디언트 부스팅, 높은 성능 기대 (없으면 HistGradientBoosting 사용) |

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
- `models/*.pkl` — 학습된 모델
- `reports/model_comparison.csv` — 성능 비교표
- `reports/model_report.md` — 성능 보고서
- `reports/images/cm_*.png` — Confusion Matrix
- `reports/images/fi_*.png` — Feature Importance

### 3. 예측

#### 단일 샘플 예측 (feature 값 직접 입력)

```bash
python src/predict.py --model Random_Forest --values 23 10 5 0 3 0.13 2 1 0 0 0 0 4 1 3 0 1 1 0
```

값 순서: `url_length domain_length path_length query_length num_digits digit_ratio num_dots num_slashes num_hyphens num_at num_question_marks num_ampersand num_special_chars subdomain_count tld_length has_punycode has_suspicious_keyword has_https has_ip`

#### CSV 파일 일괄 예측

```bash
python src/predict.py --model Random_Forest --input data/final_url_dataset.csv --output reports/predictions.csv
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

## 위험도 점수 계산 방식

모델의 `predict_proba`를 이용해 피싱일 확률을 계산하고, 0~100점 사이의 위험도 점수로 변환합니다.

```
risk_score = phishing_probability × 100
```

| 위험도 점수 | 등급 |
|------------|------|
| 0 ~ 30 | 안전 |
| 31 ~ 70 | 주의 |
| 71 ~ 100 | 위험 |

---

## 결과 파일 설명

| 파일 | 내용 |
|------|------|
| `reports/model_comparison.csv` | 모델별 Accuracy / Precision / Recall / F1-score 비교 |
| `reports/model_report.md` | 데이터 요약, 성능 비교, 지표 설명 |
| `reports/images/cm_*.png` | 모델별 Confusion Matrix 히트맵 |
| `reports/images/fi_Random_Forest.png` | Random Forest 상위 10개 Feature Importance |
| `models/*.pkl` | joblib으로 저장된 학습 모델 |
