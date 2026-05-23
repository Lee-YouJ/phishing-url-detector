"""
저장된 모델을 불러와 단일 샘플 또는 CSV 파일에 대해 예측합니다.

사용 예시:
  # 모델 이름과 feature 값 직접 입력
  python src/predict.py --model Random_Forest --values 23 10 5 0 3 0.13 2 1 0 0 0 0 4 1 3 0 1 1 0

  # CSV 파일 일괄 예측
  python src/predict.py --model Random_Forest --input data/final_url_dataset.csv --output reports/predictions.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_model, select_features, to_risk_score, to_risk_level, load_data


def predict_single(model, feature_names: list[str], values: list[float]) -> dict:
    if len(values) != len(feature_names):
        raise ValueError(
            f"feature 수 불일치: 모델은 {len(feature_names)}개 필요, "
            f"입력은 {len(values)}개"
        )
    X = pd.DataFrame([values], columns=feature_names)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

    risk_score = to_risk_score(prob) if prob is not None else None
    risk_level = to_risk_level(risk_score) if risk_score is not None else "알 수 없음"

    return {
        "prediction": int(pred),
        "label": "피싱" if pred == 1 else "정상",
        "phishing_probability": round(float(prob), 4) if prob is not None else None,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


def predict_batch(model, feature_names: list[str], df: pd.DataFrame) -> pd.DataFrame:
    # 문자열 컬럼 보존, X만 숫자형으로 구성
    X = df[feature_names] if all(c in df.columns for c in feature_names) else df.select_dtypes(include="number")
    X = X[feature_names]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.full(len(X), np.nan)

    result = df.copy()
    result["prediction"] = preds
    result["predicted_label"] = ["피싱" if p == 1 else "정상" for p in preds]
    result["phishing_probability"] = np.round(probs, 4)
    result["risk_score"] = [to_risk_score(p) for p in probs]
    result["risk_level"] = [to_risk_level(to_risk_score(p)) for p in probs]
    return result


def get_feature_names(model_name: str) -> list[str]:
    """학습 데이터에서 feature 이름 목록을 가져온다."""
    df = load_data()
    X, _ = select_features(df)
    return X.columns.tolist()


def print_result(result: dict):
    print("\n" + "=" * 40)
    print("예측 결과")
    print("=" * 40)
    print(f"  예측 클래스    : {result['prediction']} ({result['label']})")
    if result["phishing_probability"] is not None:
        print(f"  피싱 확률      : {result['phishing_probability'] * 100:.2f}%")
        print(f"  위험도 점수    : {result['risk_score']} / 100")
        print(f"  위험도 등급    : {result['risk_level']}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="피싱 URL 탐지 예측기")
    parser.add_argument("--model", type=str, default="Random_Forest", help="모델 이름 (pkl 파일명 기준, 공백→_)")
    parser.add_argument("--values", nargs="+", type=float, help="단일 예측용 feature 값 목록 (순서대로 입력)")
    parser.add_argument("--input", type=str, help="일괄 예측용 입력 CSV 경로")
    parser.add_argument("--output", type=str, help="일괄 예측 결과 저장 CSV 경로")
    args = parser.parse_args()

    print(f"\n모델 로드 중: {args.model}")
    model = load_model(args.model)
    feature_names = get_feature_names(args.model)
    print(f"feature 목록 ({len(feature_names)}개): {feature_names}")

    if args.values:
        result = predict_single(model, feature_names, args.values)
        print_result(result)

    elif args.input:
        print(f"\n입력 파일: {args.input}")
        df = pd.read_csv(args.input)
        result_df = predict_batch(model, feature_names, df)

        print(f"\n예측 완료: {len(result_df)}건")
        print(result_df[["prediction", "predicted_label", "phishing_probability", "risk_score", "risk_level"]].head(10).to_string(index=False))

        if args.output:
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
            result_df.to_csv(args.output, index=False)
            print(f"\n결과 저장: {args.output}")

    else:
        # 인수 없이 실행 시 데모: 테스트셋 첫 번째 샘플로 예측
        print("\n[데모] 데이터셋의 첫 번째 샘플로 예측합니다.")
        df = load_data()
        sample = df[feature_names].iloc[0].tolist()
        actual = df["label"].iloc[0]
        result = predict_single(model, feature_names, sample)
        print(f"실제 레이블: {actual} ({'피싱' if actual == 1 else '정상'})")
        print_result(result)


if __name__ == "__main__":
    main()
