"""
has_https 제거 실험: has_https 없이 학습했을 때 성능 변화를 측정합니다.
결과는 reports/model_comparison_without_https.csv 와 reports/model_report.md 에 저장됩니다.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_data, select_features, evaluate_model, REPORTS_DIR, IMAGES_DIR
from train import build_models, plot_confusion_matrix, plot_feature_importance, set_korean_font


EXCLUDE_COL = "has_https"


def run_experiment(exclude: str | None = None) -> tuple[list[dict], object, list[str]]:
    """학습 + 평가를 실행하고 (성능 레코드, RF 모델, feature 목록)을 반환한다."""
    df = load_data()
    X, y = select_features(df)

    if exclude and exclude in X.columns:
        X = X.drop(columns=[exclude])

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    records = []
    rf_model = None

    for name, model in models:
        print(f"\n{'─'*50}")
        print(f"[{name}] 학습 시작...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_model(name, y_test, y_pred)
        records.append(metrics)

        suffix = f"no_https_{name.replace(' ', '_')}"
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, suffix)

        if name == "Random Forest":
            rf_model = model

    return records, rf_model, feature_names


def save_comparison_no_https(records: list[dict]) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "model_comparison_without_https.csv")
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"\n성능 비교표 저장: {path}")
    return path


def load_baseline() -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def append_report(baseline_df: pd.DataFrame, no_https_df: pd.DataFrame):
    """model_report.md 끝에 has_https 제거 실험 결과 섹션을 추가한다."""
    report_path = os.path.join(REPORTS_DIR, "model_report.md")

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

    # 비교 테이블 구성: 모델별로 기존 vs 제거 나란히
    rows = []
    for _, row_new in no_https_df.iterrows():
        model = row_new["Model"]
        base_row = baseline_df[baseline_df["Model"] == model]
        entry = {"Model": model}
        for m in metrics:
            base_val = base_row[m].values[0] if not base_row.empty else float("nan")
            new_val = row_new[m]
            diff = new_val - base_val
            sign = "+" if diff >= 0 else ""
            entry[f"{m} (기존)"] = base_val
            entry[f"{m} (no-https)"] = new_val
            entry[f"{m} 변화"] = f"{sign}{diff:.4f}"
        rows.append(entry)

    compare_df = pd.DataFrame(rows)

    section = "\n\n---\n\n"
    section += "## has_https 제거 실험\n\n"
    section += (
        "> `has_https`는 Feature Importance 1위(0.2380) 컬럼입니다.\n"
        "> 모델이 HTTPS 여부에 과도하게 의존하는지 확인하기 위해 해당 컬럼을 제거하고 재학습했습니다.\n\n"
    )
    section += "### 성능 변화 비교\n\n"
    section += compare_df.to_markdown(index=False)
    section += "\n\n### 해석\n\n"

    # 자동 해석: Recall 기준 최대 하락 모델
    max_drop_model = ""
    max_drop = 0.0
    for _, row_new in no_https_df.iterrows():
        model = row_new["Model"]
        base_row = baseline_df[baseline_df["Model"] == model]
        if not base_row.empty:
            drop = base_row["Recall"].values[0] - row_new["Recall"]
            if drop > max_drop:
                max_drop = drop
                max_drop_model = model

    if max_drop > 0.001:
        section += (
            f"- `has_https` 제거 후 **{max_drop_model}**의 Recall이 "
            f"{max_drop:.4f} 하락했습니다.\n"
        )
        section += "- 모델이 HTTPS 여부에 일정 수준 의존하고 있음을 나타냅니다.\n"
    else:
        section += "- `has_https` 제거 후 Recall 변화가 매우 작습니다.\n"
        section += "- 모델이 HTTPS 여부에 과도하게 의존하지 않으며, 나머지 feature만으로도 충분한 탐지 성능을 유지합니다.\n"

    section += "- 피싱 탐지 실무에서 HTTPS를 신뢰 근거로 삼기 어렵기 때문에, `has_https` 제거 후 성능 유지 여부는 모델 신뢰도 판단에 중요한 근거가 됩니다.\n"

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"model_report.md 업데이트: {report_path}")


def main():
    set_korean_font()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print(f"실험: {EXCLUDE_COL} 제거 후 재학습")
    print("=" * 60)

    records, rf_model, feature_names = run_experiment(exclude=EXCLUDE_COL)

    # Feature Importance (has_https 제외 버전)
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        plot_feature_importance(
            feature_names, rf_model.feature_importances_,
            "Random_Forest_no_https"
        )

    no_https_df = pd.DataFrame(records)
    print(f"\n{'='*60}")
    print("has_https 제거 후 성능 비교표")
    print("=" * 60)
    print(no_https_df.to_string(index=False))

    save_comparison_no_https(records)

    baseline_df = load_baseline()
    if not baseline_df.empty:
        print(f"\n{'='*60}")
        print("기존 결과와 비교")
        print("=" * 60)
        metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
        for _, row_new in no_https_df.iterrows():
            model = row_new["Model"]
            base_row = baseline_df[baseline_df["Model"] == model]
            if base_row.empty:
                continue
            print(f"\n[{model}]")
            for m in metrics:
                base_val = base_row[m].values[0]
                new_val = row_new[m]
                diff = new_val - base_val
                sign = "+" if diff >= 0 else ""
                print(f"  {m:<12s}: {base_val:.4f} → {new_val:.4f}  ({sign}{diff:.4f})")

    append_report(baseline_df, no_https_df)
    print("\n실험 완료.")


if __name__ == "__main__":
    main()
