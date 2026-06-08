
"""
Train the AI Churn Copilot V3 model on the Semrush-like synthetic dataset.

The model package stores both the trained RandomForestClassifier and the exact
one-hot encoded training columns required by the Streamlit app.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic_semrush_churn_dataset_v3.csv"
MODEL_PATH = BASE_DIR / "best_churn_model_semrush_v3.pkl"

EXCLUDED_COLUMNS = [
    "account_id",
    "account_name",
    "churn_label",
    "synthetic_true_churn_probability",
]


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    if "churn_label" not in df.columns:
        raise ValueError("Dataset must contain churn_label")

    X = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns], errors="ignore")
    y = df["churn_label"]

    X = pd.get_dummies(X, drop_first=True)
    print(f"Training feature count after one-hot encoding: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nModel evaluation:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))

    package = {
        "model": model,
        "columns": X.columns.tolist(),
        "excluded_columns": EXCLUDED_COLUMNS,
        "target": "churn_label",
        "model_type": "RandomForestClassifier",
        "dataset": DATA_PATH.name,
    }
    joblib.dump(package, MODEL_PATH)
    print(f"\nSaved model package to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
