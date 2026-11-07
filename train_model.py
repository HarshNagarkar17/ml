"""Train 5 classifiers on the Dry Bean dataset and save models + metrics."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 42
TARGET = "Class"
DATA_PATH = "data/dry_bean.csv"
MODEL_DIR = "model"


def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    from ucimlrepo import fetch_ucirepo  # uv pip install ucimlrepo (local only)
    ds = fetch_ucirepo(id=602)  # Dry Bean Dataset
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_data().dropna().drop_duplicates()
    print(f"Dataset shape: {df.shape} | classes: {df[TARGET].nunique()}")

    feature_cols = [c for c in df.columns if c != TARGET]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    X = df[feature_cols]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Scaling is part of each pipeline, so the saved model handles preprocessing itself.
    models = {
        "Logistic Regression": ("logistic_regression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])),
        "Decision Tree": ("decision_tree", Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=12, random_state=SEED))])),
        "KNN": ("knn", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7))])),
        "Naive Bayes": ("naive_bayes", Pipeline([
            ("clf", GaussianNB())])),
        "Support Vector Machine": ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=5, probability=True, random_state=SEED))])),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    rows = []
    for name, (fname, pipe) in models.items():
        print(f"Training {name} ...")
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)
        pred = pipe.predict(X_te)
        res = evaluate(y_te, pred, proba)
        rows.append({"ML Model Name": name, **res})
        joblib.dump(pipe, f"{MODEL_DIR}/{fname}.joblib")

    joblib.dump({"label_encoder": le, "feature_cols": feature_cols,
                 "target": TARGET, "models": {n: f for n, (f, _) in models.items()}},
                f"{MODEL_DIR}/artifacts.joblib")

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(f"{MODEL_DIR}/metrics.csv", index=False)

    # Test CSV keeps the original string labels in the target column
    test_df = X_te.copy()
    test_df[TARGET] = le.inverse_transform(y_te)
    test_df.to_csv("test_data.csv", index=False)

    print("\n" + metrics.to_string(index=False))
    # README-ready markdown table
    print("\n| " + " | ".join(metrics.columns) + " |")
    print("|" + "---|" * len(metrics.columns))
    for _, r in metrics.iterrows():
        print("| " + " | ".join(str(v) for v in r.values) + " |")


if __name__ == "__main__":
    main()