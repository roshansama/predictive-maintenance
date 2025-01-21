import pandas as pd
import mlflow
import json
from evidently.report import Report
from evidently.metrics import DataDriftTable

# **Step 1: Load Training Data (Baseline)**
df_train = pd.read_csv("synthetic_data.csv")

# **Step 2: Load Live Data (Simulated)**
df_live = pd.read_csv("live_data.csv")  # Replace with actual streaming data source

# **Step 3: Define Data Drift Report**
drift_report = Report(metrics=[DataDriftTable()])
drift_report.run(reference_data=df_train, current_data=df_live)

# **Step 4: Extract Drift Results**
drift_json = drift_report.as_dict()

# **Step 5: Log Drift Metrics in MLflow**
mlflow.set_experiment("Data Drift Monitoring")

with mlflow.start_run():
    mlflow.log_metric("num_drifted_columns", drift_json["metrics"][0]["result"]["number_of_drifted_columns"])
    mlflow.log_text(json.dumps(drift_json, indent=4), "data_drift_report.json")

    print("✅ Data Drift Report Logged in MLflow!")
