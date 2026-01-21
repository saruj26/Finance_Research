# ========================================================
# FinBERT Sentiment Day-by-Day Aggregation
# ========================================================

import pandas as pd
import os


# Project Paths

PROJECT_PATH = "C:\Research\Finance_Research\Data"

INPUT_DIR = os.path.join(PROJECT_PATH, "sentiment_results")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "sentiment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input file (FinBERT article-level sentiment)
INPUT_FILE = os.path.join(INPUT_DIR, "finbert_title_sentiment.csv")

# Output file (day-by-day aggregated)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "finbert_daily_sentiment.csv")


# Load FinBERT output

df = pd.read_csv(INPUT_FILE)
print("✓ Loaded article-level sentiment:", df.shape)


# Ensure 'date' is datetime type

df['date'] = pd.to_datetime(df['date'])


# Day-by-day aggregation

# Average sentiment_score per day
daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()



# Save aggregated results

daily_sentiment.to_csv(OUTPUT_FILE, index=False)
print("✓ Daily aggregated FinBERT sentiment saved to:", OUTPUT_FILE)
print(daily_sentiment.head())