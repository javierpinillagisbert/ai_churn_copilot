"""
Train Semrush-like churn model V10.

CSM ownership fields are deliberately excluded from model training because they are
workflow/assignment metadata, not customer health signals. This avoids turning CSM
assignment into a predictive driver.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic_semrush_churn_dataset_v10_csm.csv"
MODEL_PATH = BASE_DIR / "best_churn_model_semrush_v10_csm.pkl"
IMPORTANCE_PATH = BASE_DIR / "model_feature_importance_semrush_v10_csm.csv"

EXCLUDED_COLUMNS = [
    "account_id", "account_name", "churn_label", "synthetic_true_churn_probability",
    "csm_name", "csm_region", "csm_timezone", "csm_focus_area", "csm_seniority",
    "csm_manager", "book_tier", "csm_portfolio_size", "csm_assigned_account_rank",
    "target_book_size",
]


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns], errors="ignore")
    y = df["churn_label"]
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", round(roc_auc, 4))

    joblib.dump(
        {
            "model": model,
            "columns": X.columns.tolist(),
            "excluded_columns": EXCLUDED_COLUMNS,
            "roc_auc": float(roc_auc),
            "notes": "CSM assignment fields are excluded from model scoring and used only for workflow filtering/storytelling.",
        },
        MODEL_PATH,
    )

    pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False) \
        .to_csv(IMPORTANCE_PATH, index=False)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature importance to {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
