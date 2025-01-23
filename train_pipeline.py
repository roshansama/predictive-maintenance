import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc
)

# **Step 1: Load Updated Data for Retraining**
df = pd.read_csv("updated_training_data.csv")  # Use the latest combined dataset

# **Step 2: Define Features and Target**
FEATURES = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", 
            "Torque [Nm]", "Tool wear [min]", "Type"]
TARGET = "Machine failure"

X = df[FEATURES].copy()  # ✅ Ensure X is a separate copy to avoid SettingWithCopyWarning
y = df[TARGET]

# **Fix for XGBoost & MLflow: Remove special characters from feature names**
X.columns = [col.replace("[", "").replace("]", "").replace("<", "").replace(" ", "_") for col in X.columns]

# **Step 3: Identify Numerical & Categorical Features**
num_features = ["Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"]
cat_features = ["Type"]

# **Step 4: Handle Missing Values Separately for Numeric & Categorical Features**
X.loc[:, num_features] = X[num_features].apply(pd.to_numeric, errors="coerce")  # Ensure numeric columns are correct
X.loc[:, num_features] = X[num_features].fillna(X[num_features].median())  # ✅ Fill missing values for numerical columns
X.loc[:, cat_features] = X[cat_features].fillna("Unknown")  # ✅ Fill missing values for categorical columns

# **Step 5: Create Preprocessing Pipeline**
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# **Step 6: Initialize Models with Pipelines**
models = {
    "RandomForest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "DecisionTree": Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),
    "CatBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", CatBoostClassifier(verbose=0))
    ]),
    "LogisticRegression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression())
    ]),
    "XGBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
    ]),
    "GradientBoosting": Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
    "SVM": Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVC(probability=True))
    ])
}

# **Step 7: Define Model Evaluation Function**
def evaluate_and_log_model(model_name, model):
    """Train, evaluate, and log model performance in MLflow."""
    with mlflow.start_run(run_name=model_name):
        print(f"🚀 Training {model_name}...")

        # Train Model
        model.fit(X_train, y_train)

        # Make Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model.named_steps["model"], "predict_proba") else None

        # Compute Metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=1),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        }
        if y_prob is not None:
            metrics["AUC"] = roc_auc_score(y_test, y_prob)

        # Log Metrics in MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Save Precision-Recall Curve
        if y_prob is not None:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall_vals, precision_vals)
            mlflow.log_metric("PR AUC", pr_auc)

            plt.figure(figsize=(6, 6))
            plt.plot(recall_vals, precision_vals, marker='.', label=f"PR Curve (AUC={pr_auc:.2f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.legend()
            plt.grid()

            pr_curve_path = f"pr_curve_{model_name}.png"
            plt.savefig(pr_curve_path)
            plt.close()

            mlflow.log_artifact(pr_curve_path)

        # Save and Log Model
        model_filename = f"best_model_{model_name}.pkl"
        joblib.dump(model, model_filename)
        mlflow.log_artifact(model_filename)

        print(f"✅ {model_name} logged successfully in MLflow!\n")


# **Step 8: Split Data & Train Models**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Predictive Maintenance Models")

best_model = None
best_score = 0.0

for model_name, model in models.items():
    evaluate_and_log_model(model_name, model)

    # Fix: Ensure active run exists before accessing metrics
    active_run = mlflow.active_run()
    if active_run:
        f1 = mlflow.get_run(active_run.info.run_id).data.metrics.get("F1-Score", 0)
        if f1 > best_score:
            best_score = f1
            best_model = model_name
    else:
        print(f"⚠️ No active MLflow run found for {model_name}. Skipping best model selection.")

print(f"\n🏆 Best Model Selected: {best_model} with F1-Score {best_score:.4f}")

print("🎉 Training complete! Check MLflow UI for results.")
