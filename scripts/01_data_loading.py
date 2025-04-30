# scripts/01_data_loading.py

import pandas as pd

# Load CSVs
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Save basic info
print("\nTrain Data Preview:")
print(train_df.head())

print("\nTrain Description:")
print(train_df.describe(include='all'))
