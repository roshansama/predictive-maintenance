import pandas as pd
import mlflow
import json
import os
import subprocess  # Used to trigger retraining scripts
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from evidently.report import Report
from evidently.metrics import DataDriftTable, ColumnDriftMetric

# **Step 1: Load Original Training Data (Baseline)**
df_original = pd.read_csv("ai4i2020.csv").drop(columns=['UDI', 'Product ID'])

# **Step 2: Load Live Data**
df_live = pd.read_csv("live_data.csv")

# **Step 3: Sort Columns Alphabetically to Ensure Order Matches**
df_original_sorted = df_original.reindex(sorted(df_original.columns), axis=1)
df_live_sorted = df_live.reindex(sorted(df_live.columns), axis=1)

# **Step 4: Ensure Column Names Match**
assert list(df_original_sorted.columns) == list(df_live_sorted.columns), "❌ Column mismatch between train & live data!"

# **Step 5: Define Data Drift Report (Using Evidently)**
drift_report = Report(metrics=[DataDriftTable()])
drift_report.run(reference_data=df_original_sorted, current_data=df_live_sorted)

# **Step 6: Extract Drift Results**
drift_json = drift_report.as_dict()

# **Step 7: Save Drift Report as JSON**
json_filename = "data_drift_results.json"
with open(json_filename, "w") as f:
    json.dump(drift_json, f, indent=4)

print(f"📁 Data Drift Report Saved: {json_filename}")

# **Step 8: Generate Feature-wise Drift Graphs & Compute JSD**
output_dir = "drift_graphs"
os.makedirs(output_dir, exist_ok=True)

numerical_features = df_original_sorted.select_dtypes(include=["int64", "float64"]).columns.tolist()
feature_drift_scores = {}
jsd_scores = {}

for feature in numerical_features:
    # **Compute Drift Score using Evidently**
    drift_metric = Report(metrics=[ColumnDriftMetric(column_name=feature)])
    drift_metric.run(reference_data=df_original_sorted, current_data=df_live_sorted)
    drift_result = drift_metric.as_dict()
    drift_score = drift_result["metrics"][0]["result"].get("drift_score", 0.0)
    feature_drift_scores[feature] = drift_score

    # **Compute JSD (Jensen-Shannon Divergence)**
    p = df_original_sorted[feature].value_counts(normalize=True, bins=20).sort_index()
    q = df_live_sorted[feature].value_counts(normalize=True, bins=20).sort_index()
    
    # Align bins to avoid mismatched distributions
    common_index = p.index.union(q.index)
    p = p.reindex(common_index, fill_value=0)
    q = q.reindex(common_index, fill_value=0)

    jsd_score = jensenshannon(p, q)
    jsd_scores[feature] = jsd_score

    # **Plot histogram of feature distribution**
    plt.figure(figsize=(8, 5))
    plt.hist(df_original_sorted[feature], bins=30, alpha=0.5, label="Training Data", color="blue")
    plt.hist(df_live_sorted[feature], bins=30, alpha=0.5, label="Live Data", color="red")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Drift in {feature} (Evidently: {drift_score:.2f}, JSD: {jsd_score:.2f})")
    plt.legend()

    # Save graph
    safe_feature_name = feature.replace("[", "").replace("]", "").replace(" ", "_").replace("/", "_")
    graph_filename = f"{output_dir}/{safe_feature_name}_drift.png"
    plt.savefig(graph_filename)
    plt.close()

    print(f"📊 Feature Drift Graph Saved: {graph_filename}")

# **Step 9: Log Drift Metrics & Graphs in MLflow**
mlflow.set_experiment("Data Drift Monitoring")

trigger_retrain = False  # **Flag to decide retraining**

with mlflow.start_run():
    # Log Number of Drifted Columns
    mlflow.log_metric("num_drifted_columns", drift_json["metrics"][0]["result"]["number_of_drifted_columns"])
    
    # **Log Individual Feature Drift Scores**
    for feature, score in feature_drift_scores.items():
        safe_feature_name = feature.replace("[", "").replace("]", "").replace(" ", "_").replace("/", "_")
        mlflow.log_metric(f"Evidently_Drift_{safe_feature_name}", score)

    # **Log JSD Scores & Check if Retraining Needed**
    for feature, score in jsd_scores.items():
        safe_feature_name = feature.replace("[", "").replace("]", "").replace(" ", "_").replace("/", "_")
        mlflow.log_metric(f"JSD_{safe_feature_name}", score)
        
        if score > 0.3:
            trigger_retrain = True  # **Set flag if any feature exceeds threshold**
            print(f"⚠️ {feature} has significant drift (JSD = {score:.2f})! Retraining needed.")

    # **Log Drift Report JSON**
    mlflow.log_artifact(json_filename)

    # **Log All Feature Drift Graphs**
    for feature in numerical_features:
        safe_feature_name = feature.replace("[", "").replace("]", "").replace(" ", "_").replace("/", "_")
        graph_path = f"{output_dir}/{safe_feature_name}_drift.png"
        mlflow.log_artifact(graph_path)

    print("✅ Data Drift Report & Graphs Logged in MLflow!")

# **Step 10: Trigger Model Retraining if Needed**
if trigger_retrain:
    print("\n🔄 Retraining Pipeline Triggered Due to High Drift...")

    # **1️⃣ Combine Training & Live Data**
    subprocess.run(["python", "combining.py"], check=True)
    
    # **2️⃣ Retrain Model Using Updated Data**
    subprocess.run(["python", "train_pipeline.py"], check=True)

    # **3️⃣ Log Retraining in MLflow**
    with mlflow.start_run():
        mlflow.log_param("Retraining Triggered", True)
        print("📢 Model Retrained and Logged in MLflow!")

print("\n🎯 Drift Monitoring Complete! Check MLflow for results.")
