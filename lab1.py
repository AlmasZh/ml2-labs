import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ---------------------------------------------------------------------------
# 1. SETUP & DATA PREPARATION
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000/")
EXPERIMENT_NAME = "Lab1_Heart_Disease_Tracking"
mlflow.set_experiment(EXPERIMENT_NAME)

print("Fetching Heart Disease dataset from OpenML...")
data = fetch_openml("heart-statlog", version=1, as_frame=True)
df = data.frame

if "present" in df["class"].values:
    df["target"] = df["class"].map({"absent": 0, "present": 1})
else:
    df["target"] = df["class"].astype(str).str.strip().map({"1": 0, "2": 1})

if df["target"].isna().any():
    df["target"] = LabelEncoder().fit_transform(df["class"])

df = df.drop(columns=["class"])

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data ready. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------------------------------------------------------------------------
# 2. PART 1: MLFLOW AUTOLOG DEMONSTRATION
# ---------------------------------------------------------------------------
print("\n--- Running Part 1: Autolog with Random Forest ---")
mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

with mlflow.start_run(run_name="Baseline_Autolog_RandomForest"):
    mlflow.set_tag("logging_mode", "automatic")
    mlflow.set_tag("phase", "lab1_autolog_demo")
    
    rf_auto = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_auto.fit(X_train, y_train)
    preds = rf_auto.predict(X_test)
    
    print(f"Autolog RF Test Accuracy: {accuracy_score(y_test, preds):.4f}")

# Disable autolog to prevent interference with custom logging
mlflow.sklearn.autolog(disable=True)

# ---------------------------------------------------------------------------
# 3. PART 2: CUSTOM LOGGING (4 ALGORITHMS)
# ---------------------------------------------------------------------------
def log_evaluation_plots(y_true, y_pred, y_proba, model_name):
    """Generates and logs Confusion Matrix and ROC Curve into MLflow artifacts."""
    # 1. Confusion Matrix
    cm_fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    mlflow.log_figure(cm_fig, f"plots/confusion_matrix_{model_name.lower()}.png")
    plt.close(cm_fig)
    
    # 2. ROC Curve
    if y_proba is not None:
        roc_fig, ax = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name} - ROC Curve")
        ax.legend()
        plt.tight_layout()
        mlflow.log_figure(roc_fig, f"plots/roc_curve_{model_name.lower()}.png")
        plt.close(roc_fig)


models = {
    "Logistic_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42))
    ]),
    "Support_Vector_Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))
    ]),
    "Random_Forest": RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_split=4, random_state=42
    ),
    "Gradient_Boosting": GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42
    )
}

print("\n--- Running Part 2: Custom Logging across 4 Algorithms ---")

for model_name, pipeline in models.items():
    with mlflow.start_run(run_name=f"Custom_{model_name}"):
        mlflow.set_tag("logging_mode", "custom")
        mlflow.set_tag("model_family", model_name)
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        
        # 1. Log Parameters (clean primitive types)
        params = pipeline.named_steps["clf"].get_params() if isinstance(pipeline, Pipeline) else pipeline.get_params()
        filtered_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
        mlflow.log_params(filtered_params)
        
        # 2. Log Metrics
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1_score": f1,
            "test_roc_auc": roc
        })
        
        # 3. Log Artifacts (Plots)
        log_evaluation_plots(y_test, y_pred, y_proba, model_name)
        
        # 4. Model Signature & Model Artifact
        signature = infer_signature(X_train, pipeline.predict(X_train))
        input_example = X_train.iloc[:3]
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle"
        )
        
        print(f"Logged {model_name:<23} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")

print("\nLab 1 completed successfully! Check your MLflow UI at http://localhost:5000/")