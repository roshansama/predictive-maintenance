import pandas as pd
import numpy as np

# Load real dataset for reference
df_real = pd.read_csv("ai4i2020.csv")

# Define sample size
num_samples = 500  

# Generate synthetic data using real statistics
synthetic_data = {
    "Air temperature [K]": np.random.normal(df_real["Air temperature [K]"].mean(), df_real["Air temperature [K]"].std(), num_samples),
    "Process temperature [K]": np.random.normal(df_real["Process temperature [K]"].mean(), df_real["Process temperature [K]"].std(), num_samples),
    "Rotational speed [rpm]": np.random.normal(df_real["Rotational speed [rpm]"].mean(), df_real["Rotational speed [rpm]"].std(), num_samples),
    "Torque [Nm]": np.random.normal(df_real["Torque [Nm]"].mean(), df_real["Torque [Nm]"].std(), num_samples),
    "Tool wear [min]": np.random.normal(df_real["Tool wear [min]"].mean(), df_real["Tool wear [min]"].std(), num_samples),
    "Type": np.random.choice(df_real["Type"].unique(), num_samples, p=df_real["Type"].value_counts(normalize=True))  
}

# Convert to DataFrame
df_synthetic = pd.DataFrame(synthetic_data)

# **Fix: Ensure at least some failures exist**
failure_rate = df_real["Machine failure"].mean()  # 3.39% failure cases
df_synthetic["Machine failure"] = np.random.choice([0, 1], size=num_samples, p=[1 - failure_rate, failure_rate])

# **Force at least 5 failures to avoid all 0s**
num_forced_failures = 5
df_synthetic.loc[df_synthetic.sample(num_forced_failures).index, "Machine failure"] = 1

# Detect failure reason columns dynamically
failure_reason_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# **Ensure failure modes exist in dataset**
for col in failure_reason_columns:
    if col not in df_real.columns:
        print(f"⚠️ Warning: Column {col} not found in ai4i2020.csv")
        df_real[col] = 0  # Default to 0 if missing

# **Assign failure reasons correctly**
failure_indices = df_synthetic[df_synthetic["Machine failure"] == 1].index

for col in failure_reason_columns:
    failure_prob = df_real[col].mean()  # Get probability of this failure mode occurring

    df_synthetic[col] = 0  # Default all to 0  
    df_synthetic.loc[failure_indices, col] = np.random.choice(
        [0, 1], 
        size=len(failure_indices), 
        p=[1 - failure_prob, failure_prob]
    )

# **Ensure every failed machine has at least one failure reason**
for i in failure_indices:
    if df_synthetic.loc[i, failure_reason_columns].sum() == 0:
        random_failure_col = np.random.choice(failure_reason_columns)
        df_synthetic.loc[i, random_failure_col] = 1

# Save synthetic data
df_synthetic.to_csv("synthetic_data.csv", index=False)
print("✅ Synthetic data generated successfully with failure modes included!")
