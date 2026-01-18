# ========================================================
# LDA Daily Aggregation + Topic Renaming
# ========================================================

import os
import pandas as pd

# --------------------------------------------------------
# Project Path (CHANGE ONLY THIS if needed)
# --------------------------------------------------------
PROJECT_PATH = "/content/drive/MyDrive/SP500_Index"
INPUT_DIR = os.path.join(PROJECT_PATH, "topic_results")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "topic_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# Input: article-level LDA output with topic probabilities
# --------------------------------------------------------
input_file = os.path.join(INPUT_DIR, "lda_output_filtered.csv")
df = pd.read_csv(input_file)

# --------------------------------------------------------
# Ensure date format
# --------------------------------------------------------
# If your date is like '1/2/2016' or '2016-01-02', this handles both
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Use only date (remove time) for proper grouping
df["date"] = df["date"].dt.date

# --------------------------------------------------------
# Identify topic probability columns
# --------------------------------------------------------
topic_cols = [c for c in df.columns if c.startswith("topic_") and c.endswith("_prob")]
topic_cols = sorted(topic_cols, key=lambda x: int(x.split("_")[1]))  # topic_0_prob ... topic_9_prob

# --------------------------------------------------------
# Day-by-day aggregation (mean topic probabilities per date)
# --------------------------------------------------------
daily_df = df.groupby("date")[topic_cols].mean().reset_index()

# --------------------------------------------------------
# ✅ Rename topics based on your Excel top_keywords (topic_id → title)
# --------------------------------------------------------
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
    "topic_0_prob": "credit_ratings_fitch",
    "topic_1_prob": "fed_rates_macro",
    "topic_2_prob": "banks_markets_bonds_risk",
    "topic_3_prob": "us_politics_geopolitics_trump",
    "topic_4_prob": "italy_europe_data_treasury",
    "topic_5_prob": "markets_fx_commodities",
    "topic_6_prob": "china_trade_tariffs_energy",
    "topic_7_prob": "corporate_business_activity",
    "topic_8_prob": "election_tax_budget_us_politics",
    "topic_9_prob": "sp500_earnings_us_equities"
}

daily_df = daily_df.rename(columns=topic_rename_map)

# --------------------------------------------------------
# Save outputs
# --------------------------------------------------------
daily_output_file = os.path.join(OUTPUT_DIR, "lda_daily_aggregated.csv")
daily_df.to_csv(daily_output_file, index=False)

print("✓ Saved:", daily_output_file)

# --------------------------------------------------------
# Optional: Save the mapping (for thesis / documentation)
# --------------------------------------------------------
mapping_df = pd.DataFrame(
    [{"topic_id": int(k.split("_")[1]), "old_col": k, "new_name": v}
     for k, v in topic_rename_map.items()]
).sort_values("topic_id")

mapping_file = os.path.join(OUTPUT_DIR, "lda_topic_name_mapping.csv")
mapping_df.to_csv(mapping_file, index=False)
print("✓ Mapping saved:", mapping_file)

print("✓ Day-by-day aggregated LDA dataset columns renamed successfully.")
print("Columns now are:")
print(daily_df.columns.tolist())
