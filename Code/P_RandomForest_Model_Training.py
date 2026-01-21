# =========================================================
# RandomForest Classifier for S&P 500 Index Movement Prediction
# (time-series safe + tuning to avoid overfitting)
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Paths
# -----------------------------
PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..")
INPUT_FILE = os.path.join(PROJECT_PATH, "Data/Model_Training", "XGBoost_final_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "Data/Model_Training", "results_rf")
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


# =========================================================
# 1) Random Forest with TimeSeriesSplit Tuning
# =========================================================
print("\n==============================")
print("RANDOM FOREST TRAINING (TUNED)")
print("==============================")

# Base model (strong defaults) - regularized for overfitting reduction
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
    max_depth=8,                  # Limit tree depth
    min_samples_split=15,         # Require more samples to split
    min_samples_leaf=5,           # Require more samples in leaves
    max_features='sqrt',          # Reduce feature randomness
    max_samples=0.8,              # Bootstrap samples limit
    warm_start=False
)

# Parameter search space (focused on regularization)
param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [5, 6, 7, 8],
    "min_samples_split": [10, 15, 20],
    "min_samples_leaf": [4, 5, 6],
    "max_features": ["sqrt"],
    "max_samples": [0.7, 0.8, 0.9],
    "bootstrap": [True]
}

# TimeSeries CV (no leakage)
tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=25,
    scoring="accuracy",
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit tuning ONLY on train data
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("\n✓ Best RF Params:", search.best_params_)
print("✓ Best CV Accuracy:", search.best_score_)


# =========================================================
# 2) Evaluate on Train / Val / Test with tracking
# =========================================================
def evaluate(name, model, X_, y_):
    pred = model.predict(X_)
    acc = accuracy_score(y_, pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_, pred, digits=4))
    return acc, pred

train_acc, train_pred = evaluate("TRAIN", best_model, X_train, y_train)
val_acc, val_pred     = evaluate("VAL", best_model, X_val, y_val)
test_acc, test_pred   = evaluate("TEST", best_model, X_test, y_test)

# Track CV fold accuracies
cv_fold_accuracies = search.cv_results_['mean_test_score']
cv_std = search.cv_results_['std_test_score']

print(f"\n✓ Cross-Validation Fold Mean Accuracy: {cv_fold_accuracies.mean():.4f} (+/- {cv_std.mean():.4f})")
print(f"✓ Overfitting Gap (Train-Test): {(train_acc - test_acc):.4f}")


# =========================================================
# 3) Accuracy & Loss Graphs
# =========================================================
# Create comprehensive accuracy comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Train vs Val vs Test Accuracy
ax = axes[0, 0]
stages = ['Train', 'Val', 'Test']
accuracies = [train_acc, val_acc, test_acc]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(stages, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Train vs Val vs Test Accuracy', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f'{acc:.4f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 2) Overfitting Gap Analysis (Loss visualization)
ax = axes[0, 1]
gaps = [0, val_acc - train_acc, test_acc - train_acc]
gap_stages = ['Train Baseline', 'Val Gap', 'Test Gap']
colors_gap = ['#2ecc71', '#e67e22', '#e74c3c']
bars = ax.bar(gap_stages, gaps, color=colors_gap, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Accuracy Difference', fontsize=12, fontweight='bold')
ax.set_title('Overfitting Gap (Negative = Overfitting)', fontsize=13, fontweight='bold')
for i, (bar, gap) in enumerate(zip(bars, gaps)):
    ax.text(bar.get_x() + bar.get_width()/2, gap + 0.01 if gap > 0 else gap - 0.01, 
            f'{gap:.4f}', ha='center', va='bottom' if gap > 0 else 'top', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 3) Cross-Validation Accuracy per Iteration
ax = axes[1, 0]
iterations = range(1, len(search.cv_results_['mean_test_score']) + 1)
ax.plot(iterations, search.cv_results_['mean_test_score'], 'o-', color='#3498db', linewidth=2, markersize=6, label='CV Accuracy')
ax.fill_between(iterations, 
                search.cv_results_['mean_test_score'] - search.cv_results_['std_test_score'],
                search.cv_results_['mean_test_score'] + search.cv_results_['std_test_score'],
                alpha=0.2, color='#3498db', label='±1 Std Dev')
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('CV Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation Accuracy Across Iterations', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

# 4) Loss Metrics (1 - Accuracy)
ax = axes[1, 1]
loss_stages = ['Train', 'Val', 'Test']
losses = [1 - train_acc, 1 - val_acc, 1 - test_acc]
colors_loss = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(loss_stages, losses, color=colors_loss, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Loss (1 - Accuracy)', fontsize=12, fontweight='bold')
ax.set_title('Model Loss Across Stages', fontsize=13, fontweight='bold')
ax.set_ylim([0, max(losses) * 1.2])
for i, (bar, loss) in enumerate(zip(bars, losses)):
    ax.text(bar.get_x() + bar.get_width()/2, loss + 0.005, f'{loss:.4f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_accuracy_loss_analysis.png"), dpi=300, bbox_inches="tight")
print(f"\n✓ Saved accuracy/loss graph: rf_accuracy_loss_analysis.png")
plt.show()


# =========================================================
# 4) Confusion Matrices
# -----------------------------
# =========================================================
# 4) Confusion Matrices
# =========================================================
def plot_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.show()

plot_cm(y_val, val_pred, "RF Validation Confusion Matrix", "rf_cm_val.png")
plot_cm(y_test, test_pred, "RF Test Confusion Matrix", "rf_cm_test.png")


# =========================================================
# 5) Feature Importance
# -----------------------------
importances = best_model.feature_importances_
feat_names = X.columns
idx = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(10, 6))
plt.barh(feat_names[idx][::-1], importances[idx][::-1])
plt.xlabel("Importance")
plt.title("Random Forest - Top 20 Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_feature_importance_top20.png"), dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# 6) Save Model & Summary
# -----------------------------
MODEL_FILE = os.path.join(OUTPUT_DIR, "random_forest_sp500_model.pkl")
joblib.dump(best_model, MODEL_FILE)

summary = pd.DataFrame([{
    "best_params": str(search.best_params_),
    "cv_best_accuracy": float(search.best_score_),
    "train_accuracy": float(train_acc),
    "val_accuracy": float(val_acc),
    "test_accuracy": float(test_acc)
}])

summary_file = os.path.join(OUTPUT_DIR, "rf_model_summary.csv")
summary.to_csv(summary_file, index=False)

print(f"\n✓ Saved RF model: {MODEL_FILE}")
print(f"✓ Saved RF summary: {summary_file}")
print(f"✓ All results saved in: {OUTPUT_DIR}")
