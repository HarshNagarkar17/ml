import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")


@st.cache_resource
def load_artifacts():
    art = joblib.load("model/artifacts.joblib")
    models = {name: joblib.load(f"model/{fname}.joblib")
              for name, fname in art["models"].items()}
    return art, models


def compute_metrics(y_true, y_pred, y_proba):
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                            labels=np.arange(y_proba.shape[1]))
    except ValueError:  # e.g. a class missing from the uploaded file
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


art, models = load_artifacts()
le, feature_cols, target = art["label_encoder"], art["feature_cols"], art["target"]

st.title("🫘 Dry Bean Classifier Lab")
st.caption("Compare five pre-trained classifiers on your own test CSV. "
           "Models are loaded from disk, with no retraining.")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    st.markdown("The CSV needs these columns:")
    st.code(", ".join(feature_cols + [target]), language=None)

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin. Use the file below if you need one.")
    try:
        with open("test_data.csv", "rb") as f:
            st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")
    except FileNotFoundError:
        pass
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in feature_cols + [target] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

unknown = set(df[target].unique()) - set(le.classes_)
if unknown:
    st.error(f"Unknown labels in target column: {unknown}. Expected: {list(le.classes_)}")
    st.stop()

X = df[feature_cols]
y_true = le.transform(df[target])

st.subheader("Data preview")
st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)

# Selected model
pipe = models[model_name]
y_pred = pipe.predict(X)
y_proba = pipe.predict_proba(X)
m = compute_metrics(y_true, y_pred, y_proba)

st.subheader(f"Evaluation metrics: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(le.classes_)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y_true, y_pred, labels=np.arange(len(le.classes_)),
                                target_names=le.classes_, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# All models side by side
st.subheader("All models on this uploaded data")
rows = []
for name, p in models.items():
    r = compute_metrics(y_true, p.predict(X), p.predict_proba(X))
    rows.append({"ML Model Name": name, **r})
st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)