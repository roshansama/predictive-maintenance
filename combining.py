import pandas as pd

# Load original dataset
df_original = pd.read_csv("ai4i2020.csv").drop(columns=['UDI', 'Product ID'])

# Load live data
df_live = pd.read_csv("live_data.csv")

# Merge both datasets for retraining
df_combined = pd.concat([df_original, df_live], ignore_index=True)

# Save the new dataset for model training
df_combined.to_csv("updated_training_data.csv", index=False)

print("✅ Updated training dataset created!")
