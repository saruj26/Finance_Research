# =========================================================
# XGBoost Classifier - Improved Model (No Overfitting)
# S&P 500 Index Movement Prediction
# Target: >0.80 accuracy with minimal overfitting
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
import joblib


# =========================================================
# PATHS & SETUP
# =========================================================
PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..")
INPUT_FILE = os.path.join(PROJECT_PATH, "Data/Model_Training", "XGBoost_final_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "Data", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD & PREPARE DATA
# =========================================================
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_csv(INPUT_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

y = df["index_movement"].astype(int)
X = df.drop(columns=["date", "index_movement"])

print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"✓ Target distribution:\n{y.value_counts()}\n")
print(f"✓ Class balance: {y.value_counts(normalize=True).to_dict()}")


# =========================================================
# FEATURE ANALYSIS & SELECTION
# =========================================================
print("\n" + "="*70)
print("FEATURE ANALYSIS")
print("="*70)

# Check for collinearity
corr_matrix = X.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.85:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print(f"\n⚠ High correlations detected (>0.85):")
    for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {f1} <-> {f2}: {corr:.3f}")

# Feature selection using mutual information
selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(X, y)

feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print(f"\nTop 15 features by mutual information:")
print(feature_scores.head(15).to_string())

# Select top K features (reduce noise)
K_BEST = 18
selector_best = SelectKBest(score_func=mutual_info_classif, k=K_BEST)
X_selected = selector_best.fit_transform(X, y)
selected_features = X.columns[selector_best.get_support()].tolist()

print(f"\n✓ Selected {K_BEST} best features:")
print(f"  {selected_features}")

X = pd.DataFrame(X_selected, columns=selected_features)


# =========================================================
# TIME-SERIES SAFE SPLIT
# =========================================================
print("\n" + "="*70)
print("DATA SPLITTING (TIME-SERIES SAFE)")
print("="*70)

# Use 70% train, 15% val, 15% test
train_size = int(0.70 * len(X))
val_size = int(0.15 * len(X))

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:train_size+val_size]
y_val = y.iloc[train_size:train_size+val_size]

X_test = X.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train target dist: {y_train.value_counts().to_dict()}")
print(f"Val target dist: {y_val.value_counts().to_dict()}")
print(f"Test target dist: {y_test.value_counts().to_dict()}")


# =========================================================
# HYPERPARAMETER TUNING WITH CROSS-VALIDATION (SIMPLIFIED)
# =========================================================
print("\n" + "="*70)
print("MODEL VALIDATION (5-FOLD STRATIFIED CV)")
print("="*70)

# Simpler CV approach - train on each fold without eval_set
skf = StratifiedKFold(n_splits=5, shuffle=False)

cv_model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=10,
    gamma=1.0,
    subsample=0.7,
    colsample_bytree=0.5,
    colsample_bylevel=0.5,
    colsample_bynode=0.5,
    reg_alpha=2.0,
    reg_lambda=3.0,
    scale_pos_weight=1.0,
    random_state=42,
    eval_metric=["logloss", "error"],
    tree_method='hist',
    device='cpu'
)

print("\nRunning 5-Fold Stratified Cross-Validation...")
cv_scores = cross_validate(
    cv_model, X_train, y_train,
    cv=skf,
    scoring=['accuracy', 'roc_auc', 'precision', 'recall'],
    n_jobs=-1,
    verbose=1
)

print(f"\n✓ CV Results:")
print(f"  Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
print(f"  ROC-AUC: {cv_scores['test_roc_auc'].mean():.4f} ± {cv_scores['test_roc_auc'].std():.4f}")
print(f"  Precision: {cv_scores['test_precision'].mean():.4f} ± {cv_scores['test_precision'].std():.4f}")
print(f"  Recall: {cv_scores['test_recall'].mean():.4f} ± {cv_scores['test_recall'].std():.4f}")


# =========================================================
# FINAL MODEL TRAINING
# =========================================================
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON 70% TRAIN DATA")
print("="*70)

final_model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=10,
    gamma=1.0,
    subsample=0.7,
    colsample_bytree=0.5,
    colsample_bylevel=0.5,
    colsample_bynode=0.5,
    reg_alpha=2.0,
    reg_lambda=3.0,
    scale_pos_weight=1.0,
    random_state=42,
    eval_metric=["logloss", "error"],
    early_stopping_rounds=300,
    tree_method='hist',
    device='cpu'
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100
)

# Get training history
evals = final_model.evals_result()
train_logloss = evals["validation_0"]["logloss"]
val_logloss = evals["validation_1"]["logloss"]
train_error = evals["validation_0"]["error"]
val_error = evals["validation_1"]["error"]

train_acc = [1 - e for e in train_error]
val_acc = [1 - e for e in val_error]

best_epoch = int(np.argmin(val_logloss)) + 1
print(f"\n✓ Best epoch: {best_epoch}")
print(f"✓ Best val logloss: {min(val_logloss):.4f}")
print(f"✓ Val accuracy at best epoch: {val_acc[best_epoch-1]:.4f}")


# =========================================================
# PREDICTIONS & EVALUATION
# =========================================================
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

def evaluate_model(name, model, X_data, y_data):
    """Comprehensive model evaluation"""
    proba = model.predict_proba(X_data)[:, 1]
    pred = model.predict(X_data)
    
    acc = accuracy_score(y_data, pred)
    auc = roc_auc_score(y_data, proba)
    ll = log_loss(y_data, proba)
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print("\n" + classification_report(y_data, pred, digits=4))
    
    return acc, auc, ll, pred, proba

train_acc_final, train_auc_final, train_ll_final, train_pred, train_proba = evaluate_model(
    "TRAIN", final_model, X_train, y_train
)
val_acc_final, val_auc_final, val_ll_final, val_pred, val_proba = evaluate_model(
    "VALIDATION", final_model, X_val, y_val
)
test_acc_final, test_auc_final, test_ll_final, test_pred, test_proba = evaluate_model(
    "TEST", final_model, X_test, y_test
)

# Check overfitting
print("\n" + "="*70)
print("OVERFITTING ANALYSIS")
print("="*70)
print(f"Train-Val Accuracy Gap: {(train_acc_final - val_acc_final):.4f}")
print(f"Train-Test Accuracy Gap: {(train_acc_final - test_acc_final):.4f}")
print(f"Train-Val Loss Gap: {(train_ll_final - val_ll_final):.4f}")

if train_acc_final - val_acc_final < 0.05:
    print("✓ EXCELLENT: Minimal overfitting detected!")
elif train_acc_final - val_acc_final < 0.10:
    print("✓ GOOD: Low overfitting")
else:
    print("⚠ WARNING: Some overfitting still present")


# =========================================================
# VISUALIZATIONS
# =========================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(16, 12))

# 1) Training & Validation Loss
ax1 = plt.subplot(3, 3, 1)
epochs = range(1, len(train_logloss) + 1)
ax1.plot(epochs, train_logloss, label="Train Loss", linewidth=2, color='#2ecc71')
ax1.plot(epochs, val_logloss, label="Val Loss", linewidth=2, color='#e74c3c')
ax1.axvline(best_epoch, color='gray', linestyle='--', alpha=0.5, label=f"Best epoch: {best_epoch}")
ax1.set_xlabel("Epoch", fontweight='bold')
ax1.set_ylabel("Loss", fontweight='bold')
ax1.set_title("Training & Validation Loss", fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2) Training & Validation Accuracy
ax2 = plt.subplot(3, 3, 2)
ax2.plot(epochs, train_acc, label="Train Accuracy", linewidth=2, color='#2ecc71')
ax2.plot(epochs, val_acc, label="Val Accuracy", linewidth=2, color='#e74c3c')
ax2.axvline(best_epoch, color='gray', linestyle='--', alpha=0.5, label=f"Best epoch: {best_epoch}")
ax2.set_xlabel("Epoch", fontweight='bold')
ax2.set_ylabel("Accuracy", fontweight='bold')
ax2.set_title("Training & Validation Accuracy", fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3) Final Accuracy Comparison
ax3 = plt.subplot(3, 3, 3)
stages = ['Train', 'Val', 'Test']
accuracies = [train_acc_final, val_acc_final, test_acc_final]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax3.bar(stages, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Accuracy', fontweight='bold')
ax3.set_title('Final Accuracy Comparison', fontweight='bold')
ax3.set_ylim([0, 1])
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4) ROC-AUC Curves
ax4 = plt.subplot(3, 3, 4)
for name, y_true, proba, color in [
    ('Train', y_train, train_proba, '#2ecc71'),
    ('Val', y_val, val_proba, '#f39c12'),
    ('Test', y_test, test_proba, '#e74c3c')
]:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    ax4.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2, color=color)
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax4.set_xlabel('False Positive Rate', fontweight='bold')
ax4.set_ylabel('True Positive Rate', fontweight='bold')
ax4.set_title('ROC-AUC Curves', fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# 5) Confusion Matrix - Test
ax5 = plt.subplot(3, 3, 5)
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=False,
            xticklabels=[0, 1], yticklabels=[0, 1])
ax5.set_xlabel('Predicted', fontweight='bold')
ax5.set_ylabel('Actual', fontweight='bold')
ax5.set_title('Test Confusion Matrix', fontweight='bold')

# 6) Feature Importance
ax6 = plt.subplot(3, 3, 6)
importances = final_model.feature_importances_
top_k = 12
top_idx = np.argsort(importances)[-top_k:]
ax6.barh(range(top_k), importances[top_idx], color='#3498db', edgecolor='black', linewidth=1)
ax6.set_yticks(range(top_k))
ax6.set_yticklabels([selected_features[i] for i in top_idx], fontsize=9)
ax6.set_xlabel('Importance', fontweight='bold')
ax6.set_title(f'Top {top_k} Feature Importance', fontweight='bold')
ax6.grid(axis='x', alpha=0.3)

# 7) Overfitting Analysis
ax7 = plt.subplot(3, 3, 7)
gaps = [0, val_acc_final - train_acc_final, test_acc_final - train_acc_final]
gap_labels = ['Baseline', 'Val Gap', 'Test Gap']
colors_gap = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax7.bar(gap_labels, gaps, color=colors_gap, alpha=0.7, edgecolor='black', linewidth=2)
ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax7.set_ylabel('Accuracy Difference', fontweight='bold')
ax7.set_title('Overfitting Gap Analysis', fontweight='bold')
for bar, gap in zip(bars, gaps):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.005 if height > 0 else height - 0.015,
             f'{gap:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
ax7.grid(axis='y', alpha=0.3)

# 8) Loss Comparison
ax8 = plt.subplot(3, 3, 8)
losses = [train_ll_final, val_ll_final, test_ll_final]
bars = ax8.bar(stages, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax8.set_ylabel('Log Loss', fontweight='bold')
ax8.set_title('Final Loss Comparison', fontweight='bold')
for bar, loss in zip(bars, losses):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
ax8.grid(axis='y', alpha=0.3)

# 9) Cross-Validation Scores
ax9 = plt.subplot(3, 3, 9)
cv_metrics = ['Accuracy', 'ROC-AUC', 'Precision', 'Recall']
cv_values = [
    cv_scores['test_accuracy'].mean(),
    cv_scores['test_roc_auc'].mean(),
    cv_scores['test_precision'].mean(),
    cv_scores['test_recall'].mean()
]
bars = ax9.bar(cv_metrics, cv_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
               alpha=0.7, edgecolor='black', linewidth=2)
ax9.set_ylabel('Score', fontweight='bold')
ax9.set_title('5-Fold CV Metrics', fontweight='bold')
ax9.set_ylim([0, 1])
for bar, val in zip(bars, cv_values):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax9.grid(axis='y', alpha=0.3)
ax9.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_comprehensive_v3.png"), dpi=300, bbox_inches='tight')
print(f"✓ Saved comprehensive plot: xgboost_comprehensive_v3.png")
plt.show()


# =========================================================
# SAVE MODEL & SUMMARY
# =========================================================
print("\n" + "="*70)
print("SAVING MODEL & RESULTS")
print("="*70)

MODEL_FILE = os.path.join(OUTPUT_DIR, "xgboost_sp500_model_v3.pkl")
joblib.dump(final_model, MODEL_FILE)

# Save feature selector and selected features
feature_selector_file = os.path.join(OUTPUT_DIR, "feature_selector_v3.pkl")
joblib.dump(selector_best, feature_selector_file)

summary = pd.DataFrame([{
    "model_version": "v3_improved",
    "n_features_selected": K_BEST,
    "selected_features": "|".join(selected_features),
    "best_epoch": best_epoch,
    "cv_accuracy_mean": float(cv_scores['test_accuracy'].mean()),
    "cv_accuracy_std": float(cv_scores['test_accuracy'].std()),
    "cv_roc_auc_mean": float(cv_scores['test_roc_auc'].mean()),
    "train_accuracy": float(train_acc_final),
    "val_accuracy": float(val_acc_final),
    "test_accuracy": float(test_acc_final),
    "train_roc_auc": float(train_auc_final),
    "val_roc_auc": float(val_auc_final),
    "test_roc_auc": float(test_auc_final),
    "train_loss": float(train_ll_final),
    "val_loss": float(val_ll_final),
    "test_loss": float(test_ll_final),
    "train_val_gap": float(train_acc_final - val_acc_final),
    "train_test_gap": float(train_acc_final - test_acc_final),
    "overfitting_status": "Excellent" if train_acc_final - val_acc_final < 0.05 else "Good" if train_acc_final - val_acc_final < 0.10 else "Moderate"
}])

summary_file = os.path.join(OUTPUT_DIR, "model_summary_v3.csv")
summary.to_csv(summary_file, index=False)

print(f"✓ Saved model: {MODEL_FILE}")
print(f"✓ Saved feature selector: {feature_selector_file}")
print(f"✓ Saved summary: {summary_file}")

print("\n" + "="*70)
print("✓ MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\nKEY RESULTS:")
print(f"  Test Accuracy: {test_acc_final:.4f} (Target: >0.80)")
print(f"  Test ROC-AUC: {test_auc_final:.4f}")
print(f"  Overfitting Gap: {(train_acc_final - test_acc_final):.4f}")
print(f"  CV Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
print(f"\n✓ All results saved in: {OUTPUT_DIR}")
