import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "synthetic_saas_churn_dataset_v2.csv"
MODEL_PATH = BASE_DIR / "best_churn_model_v2.pkl"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    if "churn_label" not in df.columns:
        raise ValueError("The dataset must contain a 'churn_label' column.")

    excluded_cols = ["account_id", "churn_label"]

    X = df.drop(columns=excluded_cols, errors="ignore")
    y = df["churn_label"]

    print("Preparing features...")
    X = pd.get_dummies(X, drop_first=True)

    print(f"Training with {X.shape[1]} model features")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=5
    )

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nModel evaluation:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))

    model_package = {
        "model": model,
        "columns": X.columns.tolist()
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
