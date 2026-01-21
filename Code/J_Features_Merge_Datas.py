# ===============================
# Merge Datasets for XGBoost Model Training
# ===============================

import os
import pandas as pd


# Paths

PROJECT_PATH = r"C:\Research\Finance_Research\Data"

OUTPUT_DIR = os.path.join(PROJECT_PATH, "Model_Training")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- FinBERT sentiment ----
FIN_DIR = os.path.join(PROJECT_PATH, "sentiment_results")
FIN_FILE = os.path.join(FIN_DIR, "finbert_daily_sentiment.csv")

# ---- LDA topics ----
LDA_DIR = os.path.join(PROJECT_PATH, "topic_results/LDA")
LDA_FILE = os.path.join(LDA_DIR, "lda_topic_10_daily_aggregated.csv")

# ---- TF-IDF keywords ----
TF_DIR = os.path.join(PROJECT_PATH, "topic_results/TF_IDF")
TF_FILE = os.path.join(TF_DIR, "top11_financial_daily_aggregated.csv")


# Load datasets

df_fin = pd.read_csv(FIN_FILE)
df_lda = pd.read_csv(LDA_FILE)
df_tf = pd.read_csv(TF_FILE)


# Convert date column to datetime

df_fin['date'] = pd.to_datetime(df_fin['date'])
df_lda['date'] = pd.to_datetime(df_lda['date'])
df_tf['date'] = pd.to_datetime(df_tf['date'])


# Keep required columns only


# FinBERT (sentiment)
fin_cols = ['date', 'sentiment_score']
df_fin = df_fin[fin_cols]

# LDA topics (rename consistent with your labels)
lda_cols = [
    'date',
    'credit_ratings_risk',
    'monetary_policy_inflation',
    'banking_financial_markets',
    'us_politics_geopolitics',
    'economic_data_releases',
    'stock_market_performance',
    'trade_war_oil',
    'corporate_business_activity',
    'elections_fiscal_policy',
    'index_earnings'
]
df_lda = df_lda[lda_cols]

# TF-IDF financial keywords
tf_cols = [
    'date',
    'tf_market',
    'tf_economy',
    'tf_bank',
    'tf_oil_energy',
    'tf_trade',
    'tf_stock',
    'tf_fed',
    'tf_rate',
    'tf_inflation',
    'tf_earnings',
    'tf_debt'
]
df_tf = df_tf[tf_cols]


# Merge datasets by date

df_merged = df_fin.merge(df_lda, on='date', how='inner')
df_merged = df_merged.merge(df_tf, on='date', how='inner')


# Sort by date (important for time series)

df_merged = df_merged.sort_values('date').reset_index(drop=True)


# Save final merged dataset

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "XGBoost_merged_input.csv")
df_merged.to_csv(OUTPUT_FILE, index=False)

print("Final merged dataset created:")
print(OUTPUT_FILE)
print("Shape:", df_merged.shape)