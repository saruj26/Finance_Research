# ========================================================
# FinBERT Sentiment Analysis on Preprocessed Financial News Titles
# ========================================================

# Step 0: Install Dependencies (Run only once if not installed)
# !pip install -q transformers torch

# Step 1: Import Libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

# Step 2: Project Path Definition
PROJECT_PATH = "/content/drive/MyDrive/SP500_Project"

INPUT_DIR = os.path.join(PROJECT_PATH, "preprocessed_data")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "sentiment_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input file (as you requested)
INPUT_FILE = os.path.join(INPUT_DIR, "preprocessed_financial_news.csv")

# Output file
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "finbert_title_sentiment.csv")

# Step 3: Load Preprocessed Dataset
df = pd.read_csv(INPUT_FILE)
print("✓ Loaded dataset:", df.shape)

# Step 4: Load FinBERT Model
FINBERT_MODEL = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
model.eval()

# Step 5: Function to compute FinBERT sentiment
labels = ["negative", "neutral", "positive"]

def get_finbert_sentiment(text):
    # Return 5 values always (to avoid errors)
    if not isinstance(text, str) or text.strip() == "":
        return None, None, None, None, None

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    sentiment = labels[int(np.argmax(probs))]

    # Financial sentiment score = positive - negative
    sentiment_score = float(probs[2] - probs[0])

    return sentiment, float(probs[0]), float(probs[1]), float(probs[2]), sentiment_score

# Step 6: Apply FinBERT to title column
# (If your file uses 'title' instead of 'cleaned_title', use df['title'])
results = df["title"].apply(get_finbert_sentiment)

df["sentiment_label"] = results.apply(lambda x: x[0])
df["neg_prob"] = results.apply(lambda x: x[1])
df["neu_prob"] = results.apply(lambda x: x[2])
df["pos_prob"] = results.apply(lambda x: x[3])
df["sentiment_score"] = results.apply(lambda x: x[4])

# Step 7: Select required output columns (minimal)
finbert_output = df[["date", "title", "sentiment_score"]].copy()

# Step 8: Save Output
finbert_output.to_csv(OUTPUT_FILE, index=False)

print("✓ FinBERT sentiment saved to:", OUTPUT_FILE)
print(finbert_output.head())
