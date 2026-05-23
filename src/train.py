import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_data, select_features, evaluate_model,
    save_model, save_comparison, IMAGES_DIR, REPORTS_DIR
)

# 한글 폰트 설정
def set_korean_font():
    font_candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(cm, model_name: str):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["정상(0)", "피싱(1)"],
        yticklabels=["정상(0)", "피싱(1)"],
        ax=ax,
    )
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    path = os.path.join(IMAGES_DIR, f"cm_{model_name.replace(' ', '_')}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion Matrix 저장: {path}")


def plot_feature_importance(feature_names, importances, model_name: str, top_n: int = 10):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    print(f"\n[{model_name}] 상위 {top_n} Feature Importance:")
    for rank, (feat, val) in enumerate(zip(top_features, top_values), 1):
        print(f"  {rank:2d}. {feat:<30s} {val:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), top_values[::-1], align="center")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} - Top {top_n} Feature Importance")
    fig.tight_layout()
    path = os.path.join(IMAGES_DIR, f"fi_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Feature Importance 그래프 저장: {path}")


def build_models():
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_name = "XGBoost"
        print("XGBoost 사용")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        xgb_model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=42
        )
        xgb_name = "HistGradientBoosting"
        print("XGBoost 없음 → HistGradientBoostingClassifier 사용")

    models = [
        (
            "Logistic Regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
            ]),
        ),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        ),
        (xgb_name, xgb_model),
    ]
    return models


def main():
    set_korean_font()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 60)
    print("데이터 로드 중...")
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"Label 분포:\n{df['label'].value_counts().to_string()}")
    print(f"결측치: {df.isnull().sum().sum()}")
    print(f"중복 행: {df.duplicated().sum()}")

    X, y = select_features(df)
    feature_names = X.columns.tolist()
    print(f"\n학습 feature ({len(feature_names)}개): {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    models = build_models()
    records = []
    rf_model = None
    xgb_model_obj = None
    xgb_name = None

    for name, model in models:
        print(f"\n{'─'*50}")
        print(f"[{name}] 학습 시작...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_model(name, y_test, y_pred, y_prob)
        records.append(metrics)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, name)

        save_model(model, name.replace(" ", "_"))

        if name == "Random Forest":
            rf_model = model

        if "XGBoost" in name or "Gradient" in name:
            xgb_model_obj = model
            xgb_name = name

    # Feature Importance (Random Forest 우선, 없으면 XGBoost 계열)
    fi_model = rf_model if rf_model is not None else xgb_model_obj
    fi_name = "Random Forest" if rf_model is not None else xgb_name

    if fi_model is not None:
        if hasattr(fi_model, "feature_importances_"):
            importances = fi_model.feature_importances_
        elif hasattr(fi_model, "named_steps"):
            importances = fi_model.named_steps.get("clf", fi_model).feature_importances_
        else:
            importances = None

        if importances is not None:
            plot_feature_importance(feature_names, importances, fi_name)

    # 성능 비교표
    comparison_df = pd.DataFrame(records)
    print(f"\n{'='*60}")
    print("모델 성능 비교표")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    save_comparison(records)

    # model_report.md
    report_path = os.path.join(REPORTS_DIR, "model_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 모델 성능 보고서\n\n")
        f.write("## 데이터 요약\n\n")
        f.write(f"- 전체 데이터: {df.shape[0]}행 × {df.shape[1]}열\n")
        f.write(f"- 학습 feature 수: {len(feature_names)}\n")
        f.write(f"- 정상(0): {(y == 0).sum()}건\n")
        f.write(f"- 피싱(1): {(y == 1).sum()}건\n\n")
        f.write("## 성능 비교\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n## 평가 지표 설명\n\n")
        f.write("- **Accuracy**: 전체 정확도\n")
        f.write("- **Precision**: 피싱으로 예측한 것 중 실제 피싱 비율\n")
        f.write("- **Recall**: 실제 피싱 중 피싱으로 탐지한 비율 (핵심 지표)\n")
        f.write("- **F1-score**: Precision과 Recall의 조화 평균\n")
    print(f"\n모델 보고서 저장: {report_path}")

    best = max(records, key=lambda x: x["Recall"])
    print(f"\n최고 Recall 모델: {best['Model']} (Recall={best['Recall']:.4f})")
    print("\n학습 완료.")


if __name__ == "__main__":
    main()
