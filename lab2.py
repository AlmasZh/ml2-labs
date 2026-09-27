import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, roc_curve
)

# Algorithms: Classical vs Gradient Boosting / Ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)

# ---------------------------------------------------------------------------
# 1. SETUP & DATA PREPARATION
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000/")
EXPERIMENT_NAME = "Lab2_Hyperparameter_Tuning_Optuna"
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

# ---------------------------------------------------------------------------
# 2. PHASE 1: CLASSICAL VS GRADIENT BOOSTING COMPARISON
# ---------------------------------------------------------------------------
print("\n--- Phase 1: Benchmark Classical ML vs. Boosting Algorithms ---")

comparison_models = {
    # Classical algorithms
    "Logistic_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ]),
    # Ensemble / Gradient Boosting algorithms
    "Random_Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
    "Hist_Gradient_Boosting": HistGradientBoostingClassifier(random_state=42)
}

benchmark_results = {}

for model_name, model in comparison_models.items():
    with mlflow.start_run(run_name=f"Benchmark_{model_name}"):
        mlflow.set_tag("phase", "model_comparison")
        mlflow.set_tag("model_family", "Gradient_Boosting" if "Boost" in model_name else "Classical/Bagging")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        f1 = f1_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba) if proba is not None else 0.0

        mlflow.log_metrics({"test_f1": f1, "test_accuracy": acc, "test_roc_auc": auc})
        benchmark_results[model_name] = {"F1": f1, "Accuracy": acc, "ROC-AUC": auc}
        print(f"Benchmark: {model_name:<24} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Generate and log a comparative bar plot
fig, ax = plt.subplots(figsize=(10, 5))
results_df = pd.DataFrame(benchmark_results).T
results_df.plot(kind="bar", ax=ax)
ax.set_title("Classical ML vs. Boosting Algorithms Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0.5, 1.0)
plt.xticks(rotation=25)
plt.tight_layout()

with mlflow.start_run(run_name="Benchmark_Summary"):
    mlflow.set_tag("phase", "benchmark_summary")
    mlflow.log_figure(fig, "plots/algorithm_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. PHASE 2: OPTUNA HYPERPARAMETER TUNING (GRADIENT BOOSTING)
# ---------------------------------------------------------------------------
print("\n--- Phase 2: Hyperparameter Tuning with Optuna & Nested MLflow Runs ---")

N_TRIALS = 25
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with mlflow.start_run(run_name="Optuna_Parent_GradientBoosting_Tuning") as parent_run:
    mlflow.set_tag("phase", "hyperparameter_tuning")
    mlflow.set_tag("optimizer", "optuna")
    mlflow.set_tag("tuned_algorithm", "GradientBoostingClassifier")

    def objective(trial):
        # Sample hyperparameter space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "random_state": 42
        }

        # Track each trial as an MLflow nested run
        with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
            mlflow.log_params(params)

            clf = GradientBoostingClassifier(**params)
            # Evaluate using 5-fold cross validation F1 score
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1")
            mean_cv_f1 = float(np.mean(cv_scores))
            std_cv_f1 = float(np.std(cv_scores))

            mlflow.log_metrics({
                "cv_mean_f1": mean_cv_f1,
                "cv_std_f1": std_cv_f1
            })

            return mean_cv_f1

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nOptuna search finished!")
    print(f"Best Trial CV F1: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 1. Log best summary to parent run
    mlflow.log_metric("best_cv_mean_f1", study.best_value)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

    # 2. Train final model with the best parameters on full train set
    best_clf = GradientBoostingClassifier(**study.best_params, random_state=42)
    best_clf.fit(X_train, y_train)

    # Evaluate on holdout test set
    test_preds = best_clf.predict(X_test)
    test_proba = best_clf.predict_proba(X_test)[:, 1]

    test_metrics = {
        "final_test_accuracy": accuracy_score(y_test, test_preds),
        "final_test_precision": precision_score(y_test, test_preds),
        "final_test_recall": recall_score(y_test, test_preds),
        "final_test_f1": f1_score(y_test, test_preds),
        "final_test_roc_auc": roc_auc_score(y_test, test_proba)
    }
    mlflow.log_metrics(test_metrics)

    # 3. Log Evaluation Visualizations
    cm_fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, test_preds, ax=ax, cmap="Greens")
    ax.set_title("Best Tuned GB - Confusion Matrix")
    plt.tight_layout()
    mlflow.log_figure(cm_fig, "plots/best_model_confusion_matrix.png")
    plt.close(cm_fig)

    roc_fig, ax = plt.subplots(figsize=(5, 4))
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    ax.plot(fpr, tpr, label=f"AUC = {test_metrics['final_test_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7)
    ax.set_title("Best Tuned GB - ROC Curve")
    ax.legend()
    plt.tight_layout()
    mlflow.log_figure(roc_fig, "plots/best_model_roc_curve.png")
    plt.close(roc_fig)

    # 4. Log Optuna Visualizations (Optimization History & Parameter Importances)
    try:
        opt_history_fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        mlflow.log_figure(opt_history_fig.figure, "plots/optuna_optimization_history.png")
        plt.close(opt_history_fig.figure)

        param_importances_fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        mlflow.log_figure(param_importances_fig.figure, "plots/optuna_param_importances.png")
        plt.close(param_importances_fig.figure)
    except Exception as e:
        print(f"Note: Could not log Optuna Matplotlib plots: {e}")

    # 5. Register Best Model to MLflow Model Registry
    signature = infer_signature(X_train, best_clf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=best_clf,
        artifact_path="optuna_best_gradient_boosting",
        registered_model_name="HeartDisease_Optuna_GradientBoosting",
        signature=signature,
        input_example=X_train.iloc[:3],
        serialization_format="cloudpickle"
    )
    print(f"\nTuned model successfully registered to MLflow Model Registry as 'HeartDisease_Optuna_GradientBoosting'!")

print("\nLab 2 completed successfully! Check your MLflow UI at http://localhost:5000/")