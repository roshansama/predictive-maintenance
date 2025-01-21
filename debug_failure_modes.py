import pandas as pd

df_real = pd.read_csv("ai4i2020.csv")

# Get mean occurrence of each failure mode
failure_mode_means = df_real[["TWF", "HDF", "PWF", "OSF", "RNF"]].mean()

print("\n📊 Failure Mode Probabilities in Real Data:")
print(failure_mode_means)
