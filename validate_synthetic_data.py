import pandas as pd
import numpy as np

# Load real dataset for reference
df_real = pd.read_csv("ai4i2020.csv")

# Define sample size (adjustable)
num_samples = 500  

# Generate synthetic data with real distributions
synthetic_data = {
    "Air temperature [K]": np.random.normal(df_real["Air temperature [K]"].mean(), df_real["Air temperature [K]"].std(), num_samples),
    "Process temperature [K]": np.random.normal(df_real["Process temperature [K]"].mean(), df_real["Process temperature [K]"].std(), num_samples),
    "Rotational speed [rpm]": np.random.normal(df_real["Rotational speed [rpm]"].mean(), df_real["Rotational speed [rpm]"].std(), num_samples),
    "Torque [Nm]": np.random.normal(df_real["Torque [Nm]"].mean(), df_real["Torque [Nm]"].std(), num_samples),
    "Tool wear [min]": np.random.normal(df_real["Tool wear [min]"].mean(), df_real["Tool wear [min]"].std(), num_samples),
    "Type": np.random.choice(df_real["Type"].unique(), num_samples, p=df_real["Type"].value_counts(normalize=True))  # Preserve type distribution
}

# Convert to DataFrame
df_synthetic = pd.DataFrame(synthetic_data)

# Generate failure labels with the same probability as real data (~3.39%)
failure_rate = df_real["Machine failure"].mean()  # 3.39% failure cases
df_synthetic["Machine failure"] = np.random.choice([0, 1], size=num_samples, p=[1 - failure_rate, failure_rate])

# Save synthetic data
df_synthetic.to_csv("synthetic_data.csv", index=False)
print("✅ Synthetic data generated and saved as synthetic_data.csv")
