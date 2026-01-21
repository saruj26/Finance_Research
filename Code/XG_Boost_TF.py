# XGBoost_training.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, log_loss
)

import xgboost as xgb
import joblib


# -----------------------------
# Paths
# -----------------------------
PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..")
INPUT_FILE = os.path.join(PROJECT_PATH, "Data/Model_Training", "XGBoost_final_dataset_tf.csv")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "Data", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

y = df["index_movement"].astype(int)
X = df.drop(columns=["date", "index_movement"])

print("✓ Loaded dataset:", df.shape)
print("✓ Features:", X.shape, " Target:", y.shape)
print("✓ Target distribution:\n", y.value_counts())


# -----------------------------
# Split (time-series safe)
# -----------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, shuffle=False
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# -----------------------------
# Model (regularized)
# -----------------------------
model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=2000,          # large, because early stopping will cut it
    learning_rate=0.006,        # SLOWER (reduce overfitting)
    max_depth=2,                # SIMPLER trees (reduce capacity)
    min_child_weight=8,         # MORE regularization
    gamma=0.3,                  # STRICTER splits
    subsample=0.6,              # MORE aggressive row sampling
    colsample_bytree=0.6,       # MORE aggressive column sampling
    colsample_bylevel=0.6,      # Feature sampling per level
    reg_alpha=0.3,              # INCREASED L1 regularization
    reg_lambda=3.0,             # INCREASED L2 regularization
    scale_pos_weight=0.81,      # Adjust for class imbalance (476/587)
    random_state=42,
    eval_metric=["logloss", "error"],  # ✅ track both loss + error (accuracy = 1-error)
    early_stopping_rounds=150   # MORE patience
)

# -----------------------------
# Train with early stopping
# -----------------------------
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50
)

# -----------------------------
# Get training history
# -----------------------------
evals = model.evals_result()

train_logloss = evals["validation_0"]["logloss"]
val_logloss   = evals["validation_1"]["logloss"]

train_error = evals["validation_0"]["error"]
val_error   = evals["validation_1"]["error"]

train_acc = [1 - e for e in train_error]
val_acc   = [1 - e for e in val_error]

best_epoch = int(np.argmin(val_logloss)) + 1
print(f"\n✓ Best epoch: {best_epoch}")
print(f"✓ Best val logloss: {min(val_logloss):.4f}")
print(f"✓ Val accuracy at best epoch: {val_acc[best_epoch-1]:.4f}")


# -----------------------------
# Final predictions
# -----------------------------
train_proba = model.predict_proba(X_train)[:, 1]
val_proba   = model.predict_proba(X_val)[:, 1]
test_proba  = model.predict_proba(X_test)[:, 1]

train_pred = (train_proba >= 0.5).astype(int)
val_pred   = (val_proba >= 0.5).astype(int)
test_pred  = (test_proba >= 0.5).astype(int)


# -----------------------------
# Metrics
# -----------------------------
def print_metrics(name, y_true, pred, proba):
    acc = accuracy_score(y_true, pred)
    ll = log_loss(y_true, proba)
    print(f"\n{name} Accuracy: {acc:.4f} | Loss: {ll:.4f}")
    print(classification_report(y_true, pred, digits=4))
    return acc, ll

train_acc_final, train_ll_final = print_metrics("TRAIN", y_train, train_pred, train_proba)
val_acc_final, val_ll_final     = print_metrics("VAL", y_val, val_pred, val_proba)
test_acc_final, test_ll_final   = print_metrics("TEST", y_test, test_pred, test_proba)


# -----------------------------
# Graphs: Loss + Accuracy
# -----------------------------
epochs = list(range(1, len(train_logloss) + 1))

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_logloss, label="Train loss")
plt.plot(epochs, val_logloss, label="Val loss")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label="Train accuracy")
plt.plot(epochs, val_acc, label="Val accuracy")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# Confusion matrices (Val + Test)
# -----------------------------
def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",        
        xticklabels=[0, 1],
        yticklabels=[0, 1]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()

plot_cm(y_val, val_pred, "Validation Confusion Matrix", "cm_val.png")
plot_cm(y_test, test_pred, "Test Confusion Matrix", "cm_test.png")



# -----------------------------
# Save model + summary
# -----------------------------
MODEL_FILE = os.path.join(OUTPUT_DIR, "xgboost_sp500_model.pkl")
joblib.dump(model, MODEL_FILE)

summary = pd.DataFrame([{
    "best_epoch": best_epoch,
    "train_accuracy": train_acc_final,
    "val_accuracy": val_acc_final,
    "test_accuracy": test_acc_final,
    "train_logloss": train_ll_final,
    "val_logloss": val_ll_final,
    "test_logloss": test_ll_final
}])

summary.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"), index=False)
print(f"\n✓ Saved model: {MODEL_FILE}")
print(f"✓ Saved results in: {OUTPUT_DIR}")
