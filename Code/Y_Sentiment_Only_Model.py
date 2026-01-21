# =========================================================
# Sentiment-only model for S&P 500 index movement
# Uses only FinBERT sentiment_score vs index_movement
# Time-series-safe split with simple logistic regression baseline
# =========================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..")
INPUT_FILE = os.path.join(PROJECT_PATH, "Data/Model_Training", "XGBoost_final_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "Data", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "sentiment_score"])

y = df["index_movement"].astype(int)
X = df[["sentiment_score"]]

print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Time-series safe split: 70/15/15
train_size = int(0.70 * len(X))
val_size = int(0.15 * len(X))

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]

X_test = X.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

print(f"Splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
results = {}
for split_name, X_split, y_split in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test),
]:
    proba = model.predict_proba(X_split)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_split, pred)
    auc = roc_auc_score(y_split, proba)
    ll = log_loss(y_split, proba)
    results[split_name] = {"acc": acc, "auc": auc, "logloss": ll, "pred": pred, "proba": proba}
    print(f"\n{split_name.upper()} -> Accuracy: {acc:.4f} | AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    print(classification_report(y_split, pred, digits=4))

# Confusion matrices
# Confusion matrices
cm_map = {
    "Validation": "val",
    "Test": "test",
}

for split_name, X_split, y_split in [
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    key = cm_map[split_name]
    pred = results[key]["pred"]
    cm = confusion_matrix(y_split, pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{split_name} Confusion Matrix (sentiment-only)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"sentiment_only_cm_{split_name.lower()}.png"), dpi=200)
    plt.close()

# Save summary
summary = pd.DataFrame([{
    "model": "sentiment_only_logreg",
    "train_accuracy": results["train"]["acc"],
    "val_accuracy": results["val"]["acc"],
    "test_accuracy": results["test"]["acc"],
    "train_auc": results["train"]["auc"],
    "val_auc": results["val"]["auc"],
    "test_auc": results["test"]["auc"],
    "train_logloss": results["train"]["logloss"],
    "val_logloss": results["val"]["logloss"],
    "test_logloss": results["test"]["logloss"],
    "train_val_gap": results["train"]["acc"] - results["val"]["acc"],
    "train_test_gap": results["train"]["acc"] - results["test"]["acc"],
}])

summary_file = os.path.join(OUTPUT_DIR, "sentiment_only_summary.csv")
summary.to_csv(summary_file, index=False)
print(f"\nSaved summary: {summary_file}")
print(f"Saved model not needed (linear weights): coef={model.coef_.ravel()[0]:.6f}, intercept={model.intercept_[0]:.6f}")
