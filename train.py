from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import joblib

DATA = Path("data/raw/creditcard.csv")
OUT = Path("reports"); MODELS = Path("models")
OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA.exists():
        raise SystemExit("Dataset missing. Download creditcard.csv from the Kaggle URL in README and place it in data/raw/.")
    df = pd.read_csv(DATA)
    if "Class" not in df or not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Expected binary target column named Class (0=legitimate, 1=fraud).")
    y = df.pop("Class").astype(int)
    X = df.drop(columns=["Time"], errors="ignore")
    X = X.select_dtypes(include="number")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    positive = max(int(y_train.sum()), 1)
    negative = max(int((y_train == 0).sum()), 1)
    candidates = {
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                min_samples_leaf=2, n_jobs=-1, random_state=42))
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                scale_pos_weight=negative/positive, eval_metric="logloss",
                tree_method="hist", n_jobs=-1, random_state=42))
        ])
    }
    scores = {}
    fitted = {}
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        candidate_prob = candidate.predict_proba(X_test)[:, 1]
        scores[name] = float(average_precision_score(y_test, candidate_prob))
        fitted[name] = (candidate, candidate_prob)
    best_name = max(scores, key=scores.get)
    model, prob = fitted[best_name]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "average_precision": float(average_precision_score(y_test, prob)),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "threshold": 0.5,
        "selected_model": best_name,
        "average_precision_by_model": scores,
        "test_rows": int(len(y_test)),
        "fraud_test_rows": int(y_test.sum())
    }
    joblib.dump({"model": model, "features": list(X.columns)}, MODELS / "fraud_pipeline.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred))
    disp.plot(); plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=160); plt.close()
    from sklearn.metrics import PrecisionRecallDisplay
    PrecisionRecallDisplay.from_predictions(y_test, prob)
    plt.tight_layout(); plt.savefig(OUT / "precision_recall_curve.png", dpi=160); plt.close()
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    print("Saved model and evaluation artifacts.")

if __name__ == "__main__": main()
