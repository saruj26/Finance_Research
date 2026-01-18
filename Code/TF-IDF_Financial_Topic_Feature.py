# =========================
# 1. IMPORT LIBRARIES
# =========================
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 2. PATH SETUP
# =========================
PROJECT_PATH = "/content/drive/MyDrive/SP500_Index"

INPUT_DIR = os.path.join(PROJECT_PATH, "preprocessed_data")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "topic_results/TF_IDF")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(INPUT_DIR, "preprocessed_financial_news.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "top11_financial_article_level.csv")

# =========================
# 3. LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

df["cleaned_article"] = df["cleaned_article"].fillna("").astype(str)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

print("Dataset shape:", df.shape)

# =========================
# 4. FINANCIAL VOCABULARY (SET)
# =========================
FINANCIAL_VOCAB = {
    # Macro / Indicators
    "inflation", "deflation", "cpi", "ppi", "gdp", "economy", "economic",
    "recession", "slowdown", "growth", "unemployment", "jobs", "payroll",

    # Rates / Central bank
    "fed", "federal reserve", "interest", "interest rate", "rates", "rate hike", "rate cut",
    "monetary policy", "fiscal policy", "policy",

    # Markets / Equities
    "market", "markets", "stock", "stocks", "equity", "equities", "shares",
    "sp500", "s&p", "dow", "nasdaq", "index", "indices",
    "volatility", "risk", "selloff", "rally", "crash", "surge", "plunge",

    # Bonds / Credit
    "treasury", "credit", "debt", "default",
    "downgrade", "upgrade", "rating", "ratings",

    # Corporate
    "earnings", "revenue", "profit", "loss", "guidance", "forecast",

    # FX / Trade / Geopolitics
    "exchange", "currency", "dollar", "yuan", "euro", "yen",
    "trade", "tariff", "sanction", "import", "export",

    # Commodities / Energy
    "oil", "crude", "brent", "wti", "gold", "commodity", "commodities",

    # Banking
    "bank", "banks", "banking", "central bank"
}

print("Financial vocabulary size:", len(FINANCIAL_VOCAB))

# =========================
# 5. COMPUTE TF-IDF (FULL CORPUS)
# =========================
vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=5,
    ngram_range=(1, 2),
    vocabulary=sorted(FINANCIAL_VOCAB)
)

tfidf_matrix = vectorizer.fit_transform(df["cleaned_article"])
feature_names = np.array(vectorizer.get_feature_names_out())

print("Filtered TF-IDF feature count:", len(feature_names))

# =========================
# 6. BUILD FULL TF-IDF DF (all vocab features)
# =========================
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# =========================
# 7. GROUP WORD VARIANTS INTO ONE FEATURE
# =========================
GROUPS = {
    "topic_economy": ["economy", "economic"],
    "topic_bank": ["bank", "banks"],
    "topic_market": ["market", "markets"],
    "topic_stock": ["stock", "stocks"],
    "topic_equity": ["equity", "equities"],
    "topic_rate": ["rate", "rates"],
    "topic_bond": ["bond", "bonds"],
    "topic_yield": ["yield", "yields"],
    "topic_investor": ["investor", "investors", "investment"],
    "topic_oil_energy": ["oil", "crude"]
}

grouped_features = pd.DataFrame(index=tfidf_df.index)

for new_col, cols in GROUPS.items():
    existing = [c for c in cols if c in tfidf_df.columns]
    grouped_features[new_col] = tfidf_df[existing].sum(axis=1) if existing else 0.0

# Optional: keep some important single features (not grouped)
SINGLE_FEATURES = [
    "fed", "federal reserve", "interest rate", "rate hike", "rate cut",
    "inflation", "cpi", "ppi", "unemployment", "payroll",
    "earnings", "guidance", "forecast",
    "trade", "tariff", "sanction",
    "brent", "wti",
    "credit", "debt", "default", "treasury",
    "sp500", "dow", "nasdaq"
]
for term in SINGLE_FEATURES:
    col_name = f"topic_{term.replace(' ', '_').replace('&','and')}"
    grouped_features[col_name] = tfidf_df[term] if term in tfidf_df.columns else 0.0

# =========================
# 8. SELECT TOP-10 "TOPIC/FEATURE" COLUMNS (by global mean)
# =========================
avg_scores = grouped_features.mean(axis=0).values
top_11_indices = np.argsort(avg_scores)[::-1][:11]
top_11_cols = grouped_features.columns[top_11_indices]

print("\nTop 10 Grouped Financial TF-IDF Features:")
for i, col in enumerate(top_11_cols, 1):
    print(f"{i}. {col}")

top10_df = grouped_features[top_11_cols].reset_index(drop=True)

# =========================
# 9. FINAL DATASET
# =========================
final_df = pd.concat(
    [
        df[["date", "cleaned_article"]].reset_index(drop=True),
        top10_df
    ],
    axis=1
)

# =========================
# 10. SAVE OUTPUT
# =========================
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Saved file to:")
print(OUTPUT_FILE)
print("Final dataset shape:", final_df.shape)
print("Final columns:", final_df.columns.tolist())
