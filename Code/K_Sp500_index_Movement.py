# ===============================
#Sp500 index movement Prediction Dataset Preparation
# ===============================

import pandas as pd
import os


# Project Path Definition

PROJECT_PATH = "C:\Research\Finance_Research\Data"

INPUT_FILE = os.path.join(PROJECT_PATH, "preprocessed_data", "preprocessed_sp500_index.csv")
OUTPUT_FILE = os.path.join(PROJECT_PATH, "Model_Training", "sp500_index_movement.csv")


# Load Preprocessed Dataset

df = pd.read_csv(INPUT_FILE)


# Calculate Daily Index Change

df['index_change'] = df['index'].diff()        
df['index_change'].fillna(0, inplace=True)     


# Calculate Binary Index Movement
# 1 if index_change > 0, else 0

df['index_movement'] = (df['index_change'] > 0).astype(int)


# Save Final Dataset

df.to_csv(OUTPUT_FILE, index=False)


# Verification

print("✓ Dataset with index change and movement saved to:", OUTPUT_FILE)
print(df.head())