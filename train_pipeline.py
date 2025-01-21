import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# **Step 1: Load Dataset**
df = pd.read_csv("synthetic_data.csv")

# **Step 2: Define Features & Target Variable**
target = "Machine failure"
features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", 
            "Torque [Nm]", "Tool wear [min]", "Type", "TWF", "HDF", "PWF", "OSF", "RNF"]

X = df[features]
y = df[target]

# **Step 3: Identify Numerical & Categorical Features**
num_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", 
                "Torque [Nm]", "Tool wear [min]"]
cat_features = ["Type"]

# **Step 4: Create Preprocessing Pipeline**
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# **Step 5: Define Models to Train**
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# **Step 6: Train & Evaluate Models**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"🔹 Training {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = pipeline
        best_model_name = name

# **Step 7: Save Best Model**
if best_model:
    joblib.dump(best_model, f"best_model_{best_model_name}.pkl")
    print(f"🚀 Best Model ({best_model_name}) Saved as best_model_{best_model_name}.pkl")
