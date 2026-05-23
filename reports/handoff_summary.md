# Handoff Summary — Phishing URL Detector

작성일: 2026-05-23

---

## 1. 프로젝트 개요

URL을 정상(0) / 피싱(1)으로 분류하는 이진 분류 머신러닝 baseline 파이프라인.

- 데이터: `data/final_url_dataset.csv` (73,328행, 정상 36,664 / 피싱 36,664, 완전 균형)
- 학습 feature: 숫자형 19개 (`url` 문자열 컬럼 제외)
- 모델: Logistic Regression / Random Forest / XGBoost

---

## 2. 구현된 파일 목록

### src/

| 파일 | 역할 |
|------|------|
| `utils.py` | 데이터 로드, 평가 함수, 위험도 점수 변환, 모델 저장/로드 공통 함수 |
| `feature_extractor.py` | URL 문자열 → 19개 feature dict 추출. `FEATURE_ORDER` 상수로 순서 고정. www. leading 제거 후 subdomain_count 계산 로직 포함 |
| `train.py` | 데이터 로드 → `extract_features(url)`로 feature 재계산 → 3개 모델 학습 → 평가/저장/보고서 생성 |
| `predict.py` | `--url URL` 인수로 실시간 URL 예측. feature 추출 → 모델 예측 → 위험도 점수/등급 출력 |
| `experiment_no_https.py` | `has_https` 제거 ablation 실험 스크립트 |

### models/

| 파일 | 설명 |
|------|------|
| `Logistic_Regression.pkl` | v2 재학습 모델 (StandardScaler Pipeline 포함) |
| `Random_Forest.pkl` | v2 재학습 모델 |
| `XGBoost.pkl` | v2 재학습 모델 |

> 현재 저장된 모델은 `train.py`를 `extract_features` 방식으로 수정한 후 재학습한 **v2 모델**입니다.

### reports/

| 파일 | 설명 |
|------|------|
| `model_comparison.csv` | v1 성능 (CSV 원본 feature 사용) |
| `model_comparison_v2.csv` | v2 성능 (`extract_features` 재계산, www 보정 포함) |
| `model_comparison_without_https.csv` | has_https 제거 ablation 실험 결과 |
| `model_report.md` | 데이터 요약 + v2 성능 비교 + has_https 제거 실험 비교 |
| `url_prediction_test_v2.csv` | 테스트 URL 9개 예측 결과 (현재 3/9 통과) |
| `images/cm_*.png` | 모델별 Confusion Matrix (v1 기준 3개, no_https 실험 3개) |
| `images/fi_Random_Forest.png` | v2 Random Forest Feature Importance 상위 10 |
| `images/fi_Random_Forest_no_https.png` | has_https 제거 후 Feature Importance |

---

## 3. 모델 성능 요약

### v1 vs v2 (extract_features 재계산) 비교

| 모델 | v1 Recall | v2 Recall | 변화 |
|------|----------|----------|------|
| Logistic Regression | 0.9853 | 0.9850 | -0.0003 |
| Random Forest | 0.9969 | 0.9969 | ±0.0000 |
| XGBoost | 0.9969 | 0.9969 | ±0.0000 |

> v2에서 feature 추출 방식을 `extract_features(url)`로 통일한 후에도 성능이 거의 동일하게 유지됨.  
> Random Forest / XGBoost: Precision 1.0000, Recall 0.9969, F1 0.9984.

### has_https 제거 실험 결과

| 모델 | Recall (전체) | Recall (no-https) | 변화 |
|------|-------------|-----------------|------|
| Logistic Regression | 0.9853 | 0.9819 | -0.0034 |
| Random Forest | 0.9969 | 0.9952 | -0.0017 |
| XGBoost | 0.9969 | 0.9952 | -0.0017 |

> `has_https` 제거 후 Recall 하락 폭이 0.002~0.003으로 매우 작음.  
> 모델이 HTTPS 여부에 과도하게 의존하지 않으며, 나머지 feature만으로도 실용 수준의 성능 유지.

### Feature Importance 상위 5 (Random Forest v2)

| 순위 | Feature | 중요도 |
|------|---------|--------|
| 1 | has_https | 0.2414 |
| 2 | digit_ratio | 0.1360 |
| 3 | num_special_chars | 0.1259 |
| 4 | num_digits | 0.1251 |
| 5 | path_length | 0.1073 |

---

## 4. 발견된 데이터 편향 문제 (미해결)

### 현상

`https://www.naver.com`, `https://accounts.google.com` 등 실제 정상 URL이 피싱으로 잘못 예측됨.

### 원인 분석

학습 데이터의 정상 URL이 **`https://domain.com` 형태(path 없음, www 없음)** 로만 구성되어, 아래 세 가지 패턴의 정상 샘플이 구조적으로 결핍됨.

| 결핍 패턴 | 정상 샘플 수 | 피싱 샘플 수 | 영향받는 URL |
|-----------|-------------|-------------|-------------|
| `path 있는 URL` (`num_slashes ≥ 3`) | **0건 (0%)** | 21,545건 (59%) | `github.com/login`, `microsoft.com/ko-kr` |
| `num_dots=2` + `subdomain_count=0` | **5건** | 15,347건 | `www.naver.com`, `www.google.com` |
| `has_suspicious_keyword=1` (login·account 등 경로 포함) | **0건** | 수천 건 | `accounts.google.com`, `github.com/login` |

### 시도한 조치 및 결과

- **`feature_extractor.py`에서 `www.` leading 제거** → `www.naver.com`의 `subdomain_count`가 1→0으로 보정됨.  
  그러나 `num_dots=2` 기준 정상 샘플이 5건뿐이라 Random Forest가 여전히 피싱으로 판정.
- `has_suspicious_keyword=1` 정상 샘플 자체가 0건이므로 feature 보정만으로는 해결 불가.
- **결론: feature 보정만으로는 해결 불가능. 학습 데이터 보강이 필수.**

### 테스트 결과 (v2 모델 기준)

| URL | 기대 | 예측 | 통과 |
|-----|------|------|------|
| https://www.naver.com | 정상 | 피싱 | ✗ |
| https://www.google.com | 정상 | 피싱 | ✗ |
| https://www.github.com | 정상 | 피싱 | ✗ |
| https://github.com/login | 정상 | 피싱 | ✗ |
| https://accounts.google.com | 정상 | 피싱 | ✗ |
| https://www.microsoft.com/ko-kr | 정상 | 피싱 | ✗ |
| http://login-secure-update-example.com/account/verify | 피싱 | 피싱 | ✓ |
| http://naver-login-security.example.com/verify | 피싱 | 피싱 | ✓ |
| http://192.168.0.1/login | 피싱 | 피싱 | ✓ |

**현재 통과율: 3/9 (피싱 탐지는 완벽, 정상 URL 오탐이 문제)**

---

## 5. 다음 작업 TODO

### [필수] 학습 데이터 보강 (`data/final_url_dataset_v2.csv` 생성)

아래 세 유형의 정상 URL 샘플을 보강해 학습 데이터 편향을 해소한다.  
보강 후 피싱 데이터도 동일한 수를 추가해 클래스 균형을 유지한다.

| 보강 유형 | 생성 방법 | 해결하는 편향 |
|-----------|----------|--------------|
| `www.domain.com` 형태 | 기존 정상 URL 중 일부에 `www.` 접두사 추가 | `num_dots=2`, `subdomain_count=0` 결핍 |
| path 포함 정상 URL | 신뢰 도메인에 `/about`, `/products` 등 일반 경로 추가 | `num_slashes≥3`, `path_length>0` 결핍 |
| `login`·`account` 경로 포함 정상 URL | 신뢰 도메인에 `/login`, `/account` 경로 추가 (subdomain 형태도 포함) | `has_suspicious_keyword=1` 정상 샘플 결핍 |

**균형 유지 원칙**: 정상 추가 N건 → 피싱도 N건 추가 (또는 정상을 추가하지 않고 피싱 일부를 www/path 형태로 재구성).

### [필수] 재학습 및 검증

1. `data/final_url_dataset_v2.csv` 생성 (위 보강 적용)
2. `train.py`로 v2 데이터셋 기반 재학습 → 모델 덮어쓰기
3. `reports/url_prediction_test_v2.csv` 재생성 → 9/9 통과 목표
4. `reports/model_comparison_v2.csv` 업데이트

### [선택] 추가 개선 고려사항

- `has_suspicious_keyword` 판단 기준을 URL 전체가 아닌 **path 부분에만** 적용하도록 변경 검토  
  (현재는 `accounts.google.com` 도메인의 `account`도 키워드로 잡힘)
- 신뢰 도메인 화이트리스트 기반 후처리 레이어 추가 검토  
  (단기 회피책이나 발표용 데모에는 유효)

---

## 6. 실행 방법 요약

```bash
# 학습 (feature_extractor 기반, www 보정 포함)
python src/train.py

# 단일 URL 예측
python src/predict.py --url "https://www.naver.com"

# 여러 URL 동시 예측
python src/predict.py --url "https://github.com" "http://login-verify.xyz/update"

# feature 값 함께 출력
python src/predict.py --url "https://example.com" --show-features

# has_https 제거 ablation 실험
python src/experiment_no_https.py
```
