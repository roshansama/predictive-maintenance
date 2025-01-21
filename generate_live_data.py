import pandas as pd
import numpy as np

# Load original training data
df_train = pd.read_csv("synthetic_data.csv")

# Generate new "live" data with some drift
df_live = df_train.copy()
df_live["Air temperature [K]"] += np.random.normal(1, 0.5, len(df_live))  # Introduce slight drift
df_live["Rotational speed [rpm]"] *= np.random.uniform(0.9, 1.1, len(df_live))  # Vary RPM

df_live.to_csv("live_data.csv", index=False)
print("✅ Simulated live data saved as live_data.csv")
