import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "final_url_dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
IMAGES_DIR = os.path.join(REPORTS_DIR, "images")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def select_features(df: pd.DataFrame, label_col: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    """숫자형 컬럼만 feature로 사용하고 label 컬럼은 제외한다."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c != label_col]
    X = df[feature_cols]
    y = df[label_col]
    return X, y


def evaluate_model(name: str, y_true, y_pred, y_prob=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["정상(0)", "피싱(1)"])

    print(f"\n{'='*50}")
    print(f"[{name}] 평가 결과")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4),
    }


def to_risk_score(prob: float) -> float:
    """피싱 확률(0~1)을 위험도 점수(0~100)로 변환한다."""
    return round(prob * 100, 2)


def to_risk_level(score: float) -> str:
    if score <= 30:
        return "안전"
    elif score <= 70:
        return "주의"
    else:
        return "위험"


def save_model(model, name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"모델 저장: {path}")
    return path


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)


def save_comparison(records: list[dict]) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"성능 비교표 저장: {path}")
    return path
