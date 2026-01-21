# =========================
# TF-IDF DAILY AGGREGATION Day-by-Day
# =========================


import os
import pandas as pd


# 2. PATH SETUP

PROJECT_PATH = "C:\Research\Finance_Research\Data"

INPUT_DIR = os.path.join(PROJECT_PATH, "topic_results", "TF_IDF")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "topic_results", "TF_IDF")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(INPUT_DIR, "top11_financial_level.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "top11_financial_daily_aggregated.csv")


# 3. LOAD DATA

df = pd.read_csv(INPUT_FILE)
print("Input dataset shape:", df.shape)

# Ensure date column is datetime and remove time
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
df = df.dropna(subset=["date"])


# 4. IDENTIFY TF-IDF COLUMNS

tfidf_columns = [col for col in df.columns if col.startswith("tf_")]

print("TF-IDF feature columns:")
print(tfidf_columns)


# 5. DAILY AGGREGATION


# 5.1 Article count per day (IMPORTANT feature)
daily_count = df.groupby("date").size().reset_index(name="article_count")

# 5.2 Sum TF-IDF per day
daily_sum = df.groupby("date")[tfidf_columns].sum().reset_index()

# 5.3 Merge sum + count
daily_df = daily_sum.merge(daily_count, on="date")

# 5.4 Normalized TF-IDF = SUM / article_count
for col in tfidf_columns:
    daily_df[f"norm_{col}"] = daily_df[col] / daily_df["article_count"]


# 6. FINAL DATASET


# Keep only normalized features + article count
final_columns = (
    ["date"]
    + [f"{col}" for col in tfidf_columns]
)

daily_final_df = daily_df[final_columns]


# 7. SAVE OUTPUT

daily_final_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Daily Aggregated TF-IDF dataset saved to:")
print(OUTPUT_FILE)
print("Final dataset shape:", daily_final_df.shape)
print("Final columns:")
print(daily_final_df.columns.tolist())
