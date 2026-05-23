"""
저장된 모델을 불러와 URL을 예측합니다.

사용 예시:
  # URL 직접 입력 (단일)
  python src/predict.py --url "http://login-verify.secure-account.xyz/update"

  # URL 여러 개 동시 입력
  python src/predict.py --url "https://github.com" "http://192.168.1.1/login"

  # CSV 파일 일괄 예측 (url 컬럼 필요)
  python src/predict.py --input data/final_url_dataset.csv --output reports/predictions.csv

  # 모델 지정 (기본값: Random_Forest)
  python src/predict.py --url "https://example.com" --model XGBoost
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, to_risk_score, to_risk_level
from feature_extractor import extract_features, FEATURE_ORDER


THRESHOLD = 0.6  # 기본 분류 threshold (0.5 → 0.6: 정상 URL 오탐 감소)


def predict_from_url(model, url: str, threshold: float = THRESHOLD) -> dict:
    """URL 문자열 하나를 받아 예측 결과 dict를 반환한다."""
    features = extract_features(url)
    X = pd.DataFrame([features], columns=FEATURE_ORDER)

    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
    pred = int(prob >= threshold) if prob is not None else int(model.predict(X)[0])

    risk_score = to_risk_score(prob) if prob is not None else None
    risk_level = to_risk_level(risk_score) if risk_score is not None else "알 수 없음"

    return {
        "url": url,
        "prediction": pred,
        "label": "피싱" if pred == 1 else "정상",
        "phishing_probability": round(float(prob), 4) if prob is not None else None,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "features": features,
    }


def predict_batch_csv(model, input_path: str, threshold: float = THRESHOLD) -> pd.DataFrame:
    """url 컬럼을 가진 CSV를 읽어 전체 행을 예측한다."""
    df = pd.read_csv(input_path)
    if "url" not in df.columns:
        raise ValueError(f"입력 CSV에 'url' 컬럼이 없습니다: {input_path}")

    feature_rows = df["url"].apply(extract_features).tolist()
    X = pd.DataFrame(feature_rows, columns=FEATURE_ORDER)

    probs = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else np.full(len(X), np.nan)
    )
    preds = (probs >= threshold).astype(int)

    result = df.copy()
    result["prediction"] = preds
    result["predicted_label"] = ["피싱" if p == 1 else "정상" for p in preds]
    result["phishing_probability"] = np.round(probs, 4)
    result["risk_score"] = [to_risk_score(p) for p in probs]
    result["risk_level"] = [to_risk_level(to_risk_score(p)) for p in probs]
    return result


def print_result(result: dict, show_features: bool = False):
    print("\n" + "=" * 50)
    print(f"  입력 URL       : {result['url']}")
    print(f"  예측 클래스    : {result['prediction']} ({result['label']})")
    if result["phishing_probability"] is not None:
        print(f"  피싱 확률      : {result['phishing_probability'] * 100:.2f}%")
        print(f"  위험도 점수    : {result['risk_score']:.1f} / 100")
        print(f"  위험도 등급    : {result['risk_level']}")
    if show_features:
        print("\n  [추출된 feature]")
        for k, v in result["features"].items():
            print(f"    {k:<28s}: {v}")
    print("=" * 50)
    # 학습 데이터 구성 한계 안내: 정상 데이터에 www. 포함 URL이 거의 없어
    # subdomain_count=1 패턴이 피싱으로 편향될 수 있음
    feats = result.get("features", {})
    if result["prediction"] == 1 and feats.get("subdomain_count", 0) == 1 and feats.get("has_https") == 1:
        print("  ※ 참고: 학습 데이터의 정상 URL은 대부분 www. 없는 형태(domain.com)입니다.")
        print("           www.domain.com 형태는 모델이 피싱으로 판단하기 쉬운 데이터 편향이 존재합니다.")


def main():
    parser = argparse.ArgumentParser(description="피싱 URL 탐지 예측기")
    parser.add_argument("--url", nargs="+", type=str, help="예측할 URL (1개 이상)")
    parser.add_argument("--input", type=str, help="일괄 예측용 CSV 경로 (url 컬럼 필요)")
    parser.add_argument("--output", type=str, help="일괄 예측 결과 저장 CSV 경로")
    parser.add_argument("--model", type=str, default="Random_Forest",
                        help="사용할 모델 이름 (기본값: Random_Forest)")
    parser.add_argument("--show-features", action="store_true",
                        help="추출된 feature 값도 함께 출력")
    args = parser.parse_args()

    print(f"모델 로드 중: {args.model}")
    model = load_model(args.model)

    if args.url:
        for url in args.url:
            result = predict_from_url(model, url)
            print_result(result, show_features=args.show_features)

    elif args.input:
        print(f"입력 파일: {args.input}")
        result_df = predict_batch_csv(model, args.input)
        cols = ["url", "prediction", "predicted_label",
                "phishing_probability", "risk_score", "risk_level"]
        # label 컬럼이 있으면 실제값도 함께 표시
        if "label" in result_df.columns:
            cols.insert(2, "label")
        print(f"\n예측 완료: {len(result_df)}건")
        print(result_df[cols].head(10).to_string(index=False))

        if args.output:
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            result_df.to_csv(args.output, index=False)
            print(f"\n결과 저장: {args.output}")

    else:
        # 인수 없이 실행 시 — 대표 URL 샘플로 동작 확인
        demo_urls = [
            "https://github.com",
            "https://www.naver.com",
            "http://192.168.0.1/login/verify",
            "http://xn--e1afmapc.xn--p1ai/secure/update?account=1",
            "http://login-verify.secure-account.xyz/update",
        ]
        print("\n[데모] 샘플 URL 예측:")
        for url in demo_urls:
            result = predict_from_url(model, url)
            print_result(result)


if __name__ == "__main__":
    main()
