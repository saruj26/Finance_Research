import pandas as pd
import os

# --------------------------------------------------------
# Project Path Definition
# --------------------------------------------------------
PROJECT_PATH = "/content/drive/MyDrive/SP500_Project"

RAW_DATA_PATH = os.path.join(PROJECT_PATH, "Data", "sp500_index.csv")
SAVE_PATH = os.path.join(PROJECT_PATH, "preprocessed_data")

os.makedirs(SAVE_PATH, exist_ok=True)

# --------------------------------------------------------
# Load Raw S&P 500 Index Data
# --------------------------------------------------------
df = pd.read_csv(RAW_DATA_PATH)

# --------------------------------------------------------
# Select Required Columns
# --------------------------------------------------------
df_clean = df[['Date', 'Adj Close']].copy()
df_clean.rename(columns={'Adj Close': 'index'}, inplace=True)

# --------------------------------------------------------
# Date Handling
# --------------------------------------------------------
df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
df_clean.dropna(subset=['Date'], inplace=True)

# --------------------------------------------------------
# Sort Chronologically
# --------------------------------------------------------
df_clean.sort_values('Date', inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# --------------------------------------------------------
# Handle Missing Values
# --------------------------------------------------------
df_clean['index'] = df_clean['index'].ffill()
df_clean['index'] = df_clean['index'].bfill()

# --------------------------------------------------------
# Remove Duplicate Dates
# --------------------------------------------------------
df_clean.drop_duplicates(subset=['Date'], keep='first', inplace=True)

# --------------------------------------------------------
# Save Cleaned Dataset
# --------------------------------------------------------
cleaned_path = os.path.join(
    SAVE_PATH,
    "preprocessed_sp500_index.csv"
)

df_clean.to_csv(cleaned_path, index=False)

# --------------------------------------------------------
# Verification Output
# --------------------------------------------------------
print("✓ Cleaned dataset saved to:", cleaned_path)
print("\nFirst 5 rows:")
print(df_clean.head())

print("\nLast 5 rows:")
print(df_clean.tail())
