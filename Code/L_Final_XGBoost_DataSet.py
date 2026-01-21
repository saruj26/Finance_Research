import pandas as pd
import os

# Project Paths

PROJECT_PATH = "C:\Research\Finance_Research\Data"

# X variables (already merged: FinBERT + LDA + TF-IDF)
X_FILE = os.path.join(PROJECT_PATH, "Model_Training", "XGBoost_merged_input.csv")

# y variable (index_movement)
Y_FILE = os.path.join(PROJECT_PATH, "Model_Training", "sp500_index_movement.csv")

# Output final dataset
OUTPUT_FILE = os.path.join(PROJECT_PATH, "Model_Training", "XGBoost_final_dataset.csv")


# Load datasets

df_X = pd.read_csv(X_FILE)
df_y = pd.read_csv(Y_FILE)


# Ensure date columns are datetime

df_X['date'] = pd.to_datetime(df_X['date'])
df_y['date'] = pd.to_datetime(df_y['date'])


# Keep only date and target from y

df_y = df_y[['date', 'index_movement']]


# Merge X and y by date (inner join ensures only common days)

df_final = df_X.merge(df_y, on='date', how='inner')


# Sort by date

df_final = df_final.sort_values('date').reset_index(drop=True)


# Save final merged dataset

df_final.to_csv(OUTPUT_FILE, index=False)

print("✓ Final dataset saved:", OUTPUT_FILE)
print("Shape:", df_final.shape)
print(df_final.head())