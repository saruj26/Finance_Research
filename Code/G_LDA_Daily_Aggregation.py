# ========================================================
# LDA Daily Aggregation + Topic Renaming
# ========================================================

import os
import pandas as pd


# Project Path (CHANGE ONLY THIS if needed)

PROJECT_PATH = "C:\Research\Finance_Research\Data"

INPUT_DIR = os.path.join(PROJECT_PATH, "topic_results/LDA")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "topic_results/LDA")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Input: article-level LDA output with topic probabilities

input_file = os.path.join(INPUT_DIR, "lda_topic_10_filtered.csv")
df = pd.read_csv(input_file)


# Ensure date format

# If your date is like '1/2/2016' or '2016-01-02', this handles both
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Use only date (remove time) for proper grouping
df["date"] = df["date"].dt.date


# Identify topic probability columns

topic_cols = [c for c in df.columns if c.startswith("topic_") and c.endswith("_prob")]
topic_cols = sorted(topic_cols, key=lambda x: int(x.split("_")[1]))  # topic_0_prob ... topic_9_prob


# Day-by-day aggregation (mean topic probabilities per date)

daily_df = df.groupby("date")[topic_cols].mean().reset_index()


# ✅ Rename topics based on your Excel top_keywords (topic_id → title)

# From your table:
# 0 = Fitch, ratings, credit, debt ...
# 1 = Fed, rate, inflation, economy ...
# 2 = Banks, markets, bonds, crisis ...
# 3 = Trump, US government, Iran, sanctions ...
# 4 = Italy, data, treasury, minister, meeting ...
# 5 = Dollar, index, stocks, oil, gold, yields, yen ...
# 6 = China, trade, tariffs, oil, imports/exports ...
# 7 = Company, business, firms, industry ...
# 8 = Election, tax, democrats, republicans, budget ...
# 9 = SP500, earnings, nasdaq, dow, market moves ...

topic_rename_map = {
    "topic_0_prob": "credit_ratings_risk",
    "topic_1_prob": "monetary_policy_inflation",
    "topic_2_prob": "banking_financial_markets",
    "topic_3_prob": "us_politics_geopolitics",
    "topic_4_prob": "economic_data_releases",
    "topic_5_prob": "stock_market_performance",
    "topic_6_prob": "trade_war_oil",
    "topic_7_prob": "corporate_business_activity",
    "topic_8_prob": "elections_fiscal_policy",
    "topic_9_prob": "index_earnings"
}

daily_df = daily_df.rename(columns=topic_rename_map)


# Save outputs

daily_output_file = os.path.join(OUTPUT_DIR, "lda_topic_10_daily_aggregated.csv")
daily_df.to_csv(daily_output_file, index=False)

print("✓ Saved:", daily_output_file)


# Optional: Save the mapping (for thesis / documentation)

mapping_df = pd.DataFrame(
    [{"topic_id": int(k.split("_")[1]), "old_col": k, "new_name": v}
     for k, v in topic_rename_map.items()]
).sort_values("topic_id")

mapping_file = os.path.join(OUTPUT_DIR, "lda_topic_10_rename.csv")
mapping_df.to_csv(mapping_file, index=False)
print("✓ Mapping saved:", mapping_file)

print("✓ Day-by-day aggregated LDA dataset columns renamed successfully.")
print("Columns now are:")
print(daily_df.columns.tolist())
