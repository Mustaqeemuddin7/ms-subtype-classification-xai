"""
Generate SHAP Explainability Notebook for XGBoost MS Subtype Classification.
"""
import json, os

def _s(src):
    lines = src.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _s(src)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": _s(src), "outputs": [], "execution_count": None}

cells = []

# ── SECTION 1: INTRO ──
cells.append(md("""# SHAP Explainability — XGBoost MS Subtype Classification

## 1. Introduction

### What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to explain model predictions. It assigns each feature an importance value (SHAP value) for every prediction, answering:

> *"How much did each feature contribute to this specific prediction?"*

### SHAP Values — Intuition

Imagine a prediction as a team effort. SHAP values assign credit to each player (feature) based on their **marginal contribution** across all possible team compositions. This comes from Shapley values in cooperative game theory:

$$\\phi_i = \\sum_{S \\subseteq N \\setminus \\{i\\}} \\frac{|S|! \\cdot (|N| - |S| - 1)!}{|N|!} \\left[ f(S \\cup \\{i\\}) - f(S) \\right]$$

### Why SHAP for Clinical ML?

1. **Patient-level explanations** — Why was *this* patient classified as SPMS?
2. **Global feature ranking** — Which features matter most overall?
3. **Feature interactions** — Do EDSS and disease duration interact?
4. **Trust and transparency** — Clinicians need to understand predictions before trusting them
5. **Regulatory compliance** — Explainability is increasingly required for clinical decision support

### Why XGBoost + SHAP?

SHAP's `TreeExplainer` provides **exact** Shapley values for tree-based models in polynomial time — no approximation, no sampling. XGBoost is the model SHAP was originally designed for (same author: Scott Lundberg).

---"""))

# ── SECTION 2: IMPORTS ──
cells.append(md("## 2. Setup and Configuration"))

cells.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120, 'font.size': 11, 'axes.titlesize': 13,
    'figure.figsize': (10, 6), 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = {'RRMS': '#2196F3', 'SPMS': '#FF5722', 'PPMS': '#4CAF50', 'CIS': '#9C27B0'}
ORDER = ['RRMS', 'SPMS', 'PPMS', 'CIS']

print(f"SHAP version: {shap.__version__}")
print("Setup complete.")"""))

# ── SECTION 3: DATA + MODEL ──
cells.append(md("""## 3. Data Preparation and Model Training

We reproduce the same preprocessing and XGBoost model used in the baseline notebook to ensure SHAP explanations are for the exact same model."""))

cells.append(code("""# Load and prepare data
df = pd.read_csv('ms_dataset.csv')
feature_cols = [c for c in df.columns if c != 'subtype']
X = df[feature_cols]
y = df['subtype']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
n_classes = len(class_names)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# Impute (no scaling)
imputer = SimpleImputer(strategy='median')
X_train_ready = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_ready = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

# Sample weights for class balance
sample_weights = compute_sample_weight('balanced', y_train)

print(f"Train: {X_train_ready.shape}, Test: {X_test_ready.shape}")
print(f"Classes: {list(class_names)}")"""))

cells.append(code("""# Train XGBoost (same config as baseline)
model = XGBClassifier(
    objective='multi:softprob', n_estimators=300,
    learning_rate=0.1, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='mlogloss', use_label_encoder=False,
    random_state=RANDOM_STATE, n_jobs=-1
)
model.fit(X_train_ready, y_train, sample_weight=sample_weights, verbose=False)

# Verify performance
y_pred = model.predict(X_test_ready)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Macro F1: {f1:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))"""))

# ── SECTION 4: SHAP EXPLAINER ──
cells.append(md("""## 4. SHAP TreeExplainer

### How TreeExplainer Works

For tree-based models, SHAP computes exact Shapley values by:
1. Tracing every possible path through each tree
2. Calculating each feature's marginal contribution at every split
3. Aggregating across all trees in the ensemble

This produces a **SHAP value matrix** of shape `(n_samples, n_features, n_classes)` — one value per feature per class per sample.

### Interpretation Guide

- **Positive SHAP value** for class $k$ → feature pushes prediction toward class $k$
- **Negative SHAP value** for class $k$ → feature pushes prediction away from class $k$
- **SHAP values sum to the model output** (log-odds minus expected value)"""))

cells.append(code("""# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values for test set
shap_values_raw = explainer.shap_values(X_test_ready)

# SHAP v0.51+ returns 3D array (samples, features, classes)
# Convert to list of 2D arrays [class_0_array, class_1_array, ...] for compatibility
if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    print(f"SHAP values (3D -> list): {len(shap_values)} classes, shape per class: {shap_values[0].shape}")
else:
    shap_values = shap_values_raw
    print(f"SHAP values (list): {len(shap_values)} classes, shape per class: {shap_values[0].shape}")

# Handle expected_value format
base_values = explainer.expected_value
if isinstance(base_values, float):
    base_values = np.array([base_values] * n_classes)
elif isinstance(base_values, np.ndarray) and base_values.ndim == 0:
    base_values = np.array([base_values.item()] * n_classes)

print(f"Classes: {list(class_names)}")
print(f"Base values: {[f'{v:.4f}' for v in base_values]}")""")
)

# ── SECTION 5: GLOBAL EXPLANATIONS ──
cells.append(md("""## 5. Global Feature Importance (SHAP)

### Mean Absolute SHAP Values

Unlike built-in feature importance (gain-based), SHAP importance is:
- **Consistent** — if a feature's true contribution increases, its SHAP importance never decreases
- **Unbiased** — not affected by feature cardinality or scale
- **Additive** — importance values sum to the total model output

This makes SHAP the gold standard for feature importance in ML."""))

cells.append(code("""# Global feature importance: mean |SHAP| across all classes
shap_abs_mean = np.zeros(len(feature_cols))
for cls_shap in shap_values:
    shap_abs_mean += np.abs(cls_shap).mean(axis=0)
shap_abs_mean /= len(shap_values)

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Mean |SHAP|': shap_abs_mean
}).sort_values('Mean |SHAP|', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.magma(np.linspace(0.3, 0.85, len(importance_df)))
ax.barh(importance_df['Feature'], importance_df['Mean |SHAP|'], color=colors, edgecolor='white')
ax.set_xlabel('Mean |SHAP Value|')
ax.set_title('Global Feature Importance (SHAP)')
plt.tight_layout()
plt.show()

print("\\nFeature Importance Ranking (SHAP):")
for rank, (_, row) in enumerate(importance_df.iloc[::-1].iterrows(), 1):
    print(f"  {rank:2d}. {row['Feature']:30s} {row['Mean |SHAP|']:.4f}")"""))

# ── SECTION 6: SHAP SUMMARY PLOTS ──
cells.append(md("""## 6. SHAP Summary Plots Per Class

Summary plots combine feature importance with **directionality** — they show whether high/low feature values push predictions toward or away from each class.

- Each dot is one patient
- X-axis: SHAP value (impact on model output)
- Color: Feature value (red = high, blue = low)
- Features ordered by importance (top = most important)"""))

cells.append(code("""# Summary plot for each class
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for i, name in enumerate(class_names):
    plt.sca(axes[i])
    shap.summary_plot(shap_values[i], X_test_ready, feature_names=feature_cols,
                      show=False, max_display=12, plot_size=None)
    axes[i].set_title(f'{name}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('SHAP Value')

plt.suptitle('SHAP Summary Plots — Per Subtype', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""### Clinical Interpretation of Summary Plots

- **RRMS:** High relapse count (red dots on the right) pushes toward RRMS — consistent with the relapsing phenotype
- **SPMS:** Higher EDSS and longer disease duration push toward SPMS — reflecting the progressive disability accumulation
- **PPMS:** Low/zero relapse count strongly pushes toward PPMS — the defining clinical characteristic
- **CIS:** Lower EDSS and shorter disease duration push toward CIS — early-stage, minimal disability"""))

# ── SECTION 7: SHAP BAR PLOTS PER CLASS ──
cells.append(md("""## 7. Mean SHAP Values Per Class

Bar plots show the average SHAP value for each feature, separately for each class."""))

cells.append(code("""fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for i, name in enumerate(class_names):
    mean_shap = pd.Series(np.abs(shap_values[i]).mean(axis=0), index=feature_cols)
    mean_shap = mean_shap.sort_values(ascending=True)
    top = mean_shap.tail(10)
    
    ax = axes[i]
    color = COLORS[name]
    ax.barh(top.index, top.values, color=color, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(f'{name} — Top 10 Features', fontweight='bold')

plt.suptitle('Feature Importance Per Subtype (SHAP)', fontsize=15, y=1.01)
plt.tight_layout()
plt.show()"""))

# ── SECTION 8: INDIVIDUAL EXPLANATIONS ──
cells.append(md("""## 8. Individual Patient Explanations (Waterfall Plots)

### Why Individual Explanations Matter

In clinical practice, a doctor doesn't just want to know the prediction — they want to know **why**. SHAP waterfall plots show:
- Starting point: the average prediction (base value)
- Each feature's contribution (red pushes up, blue pushes down)
- Final point: the model's actual output for this patient

This enables clinicians to validate whether the model's reasoning aligns with clinical knowledge."""))

cells.append(code("""# Select representative patients (one per subtype — correct predictions)
y_pred_test = model.predict(X_test_ready)

print("Individual Patient Explanations:")
print("="*60)

for class_idx, name in enumerate(class_names):
    # Find a correctly classified patient of this subtype
    mask = (y_test == class_idx) & (y_pred_test == class_idx)
    if mask.sum() > 0:
        idx = np.where(mask)[0][0]
        print(f"\\n--- {name} Patient (test index {idx}) ---")
        
        # Get SHAP values for this prediction
        patient_shap = shap_values[class_idx][idx]
        base_val = base_values[class_idx]
        
        # Show top contributing features
        contrib = pd.Series(patient_shap, index=feature_cols)
        top_pos = contrib.nlargest(3)
        top_neg = contrib.nsmallest(3)
        
        print(f"  Prediction: {name} (correct)")
        print(f"  Base value: {base_val:.4f}")
        print(f"  Top features pushing TOWARD {name}:")
        for feat, val in top_pos.items():
            print(f"    {feat}: {val:+.4f} (value={X_test_ready.iloc[idx][feat]:.2f})")
        print(f"  Top features pushing AWAY from {name}:")
        for feat, val in top_neg.items():
            print(f"    {feat}: {val:+.4f} (value={X_test_ready.iloc[idx][feat]:.2f})")"""))

cells.append(code("""# Waterfall plots for one patient per subtype
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes_flat = axes.flatten()

for class_idx, name in enumerate(class_names):
    mask = (y_test == class_idx) & (y_pred_test == class_idx)
    if mask.sum() > 0:
        idx = np.where(mask)[0][0]
        
        shap_exp = shap.Explanation(
            values=shap_values[class_idx][idx],
            base_values=base_values[class_idx],
            data=X_test_ready.iloc[idx].values,
            feature_names=feature_cols
        )
        
        plt.sca(axes_flat[class_idx])
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        axes_flat[class_idx].set_title(f'{name} Patient', fontsize=13, fontweight='bold')

plt.suptitle('Waterfall Plots — Individual Patient Explanations', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()"""))

# ── SECTION 9: FORCE PLOTS ──
cells.append(md("""## 9. Force Plots

Force plots provide a compact visualization of individual predictions. Features in **red** push the prediction higher (toward this class), features in **blue** push it lower (away from this class)."""))

cells.append(code("""# Force plots for one patient per subtype
for class_idx, name in enumerate(class_names):
    mask = (y_test == class_idx) & (y_pred_test == class_idx)
    if mask.sum() > 0:
        idx = np.where(mask)[0][0]
        print(f"\\n--- Force Plot: {name} Patient ---")
        shap.force_plot(
            base_values[class_idx],
            shap_values[class_idx][idx],
            X_test_ready.iloc[idx],
            feature_names=feature_cols,
            matplotlib=True,
            show=True
        )
        plt.title(f'{name} — Force Plot', fontsize=12)
        plt.tight_layout()
        plt.show()"""))

# ── SECTION 10: DEPENDENCE PLOTS ──
cells.append(md("""## 10. Feature Dependence Plots

### What Are Dependence Plots?

Dependence plots show how a single feature's value affects the model output, accounting for interactions with other features:
- X-axis: Feature value
- Y-axis: SHAP value (contribution to prediction)
- Color: Interaction feature (auto-detected)

These reveal **nonlinear effects** and **feature interactions** that simple importance rankings miss."""))

cells.append(code("""# Dependence plots for top 4 features (for the dominant class RRMS)
top4 = importance_df.tail(4)['Feature'].values[::-1]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Use RRMS class (index 0 or wherever it is)
rrms_idx = list(class_names).index('RRMS') if 'RRMS' in class_names else 0

for i, feat in enumerate(top4):
    ax = axes[i]
    feat_idx = feature_cols.index(feat)
    ax.scatter(
        X_test_ready[feat], shap_values[rrms_idx][:, feat_idx],
        c=X_test_ready[feat], cmap='coolwarm', alpha=0.6, s=20
    )
    ax.set_xlabel(feat)
    ax.set_ylabel(f'SHAP Value (RRMS)')
    ax.set_title(f'Dependence: {feat}')
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

plt.suptitle('Feature Dependence Plots — RRMS Class', fontsize=14)
plt.tight_layout()
plt.show()"""))

# ── SECTION 11: SHAP INTERACTION VALUES ──
cells.append(md("""## 11. Feature Interaction Analysis

### SHAP Interaction Values

SHAP can decompose predictions into **main effects** and **pairwise interactions**:

$$f(x) = \\phi_0 + \\sum_i \\phi_i + \\sum_{i < j} \\phi_{ij}$$

This reveals which feature pairs jointly influence predictions beyond their individual effects."""))

cells.append(code("""# Compute interaction values (for a subset to save time)
n_sample = min(50, len(X_test_ready))
X_sample = X_test_ready.iloc[:n_sample]

# Use RRMS class for interaction analysis
interaction_raw = explainer.shap_interaction_values(X_sample)

# Handle different array formats
if isinstance(interaction_raw, np.ndarray):
    if interaction_raw.ndim == 4:  # (samples, features, features, classes)
        rrms_interactions = interaction_raw[:, :, :, rrms_idx]
    elif interaction_raw.ndim == 3:  # (samples, features, features) single class
        rrms_interactions = interaction_raw
    else:
        rrms_interactions = interaction_raw
elif isinstance(interaction_raw, list):
    rrms_interactions = interaction_raw[rrms_idx]
else:
    rrms_interactions = interaction_raw

# Mean absolute interaction matrix
mean_interactions = np.abs(rrms_interactions).mean(axis=0)
np.fill_diagonal(mean_interactions, 0)  # Remove main effects

interaction_df = pd.DataFrame(mean_interactions, index=feature_cols, columns=feature_cols)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(interaction_df, cmap='YlOrRd', square=True, ax=ax,
            linewidths=0.3, annot=True, fmt='.3f', annot_kws={'size': 7})
ax.set_title('Feature Interaction Heatmap (SHAP — RRMS class)', fontsize=13)
plt.tight_layout()
plt.show()

# Top interactions
interactions_flat = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        interactions_flat.append((feature_cols[i], feature_cols[j], mean_interactions[i,j]))
interactions_flat.sort(key=lambda x: x[2], reverse=True)

print("\\nTop 5 Feature Interactions:")
for feat1, feat2, val in interactions_flat[:5]:
    print(f"  {feat1} x {feat2}: {val:.4f}")"""))

# ── SECTION 12: MISCLASSIFICATION ANALYSIS ──
cells.append(md("""## 12. Explaining Misclassifications

### Clinical Value

Understanding **why** the model makes errors is as important as understanding correct predictions. SHAP reveals whether misclassifications are due to:
- Ambiguous feature values (patient doesn't fit neatly into one subtype)
- Model limitations (features that should matter don't get enough weight)
- Genuine clinical overlap between subtypes"""))

cells.append(code("""# Find misclassified samples
misclassified = np.where(y_test != y_pred_test)[0]
print(f"Total misclassified: {len(misclassified)} out of {len(y_test)} ({100*len(misclassified)/len(y_test):.1f}%)")
print()

# Analyze up to 5 misclassifications
for idx in misclassified[:5]:
    true_label = class_names[y_test[idx]]
    pred_label = class_names[y_pred_test[idx]]
    pred_class_idx = y_pred_test[idx]
    
    print(f"Patient {idx}: True={true_label}, Predicted={pred_label}")
    
    # Top features for the WRONG prediction
    wrong_shap = shap_values[pred_class_idx][idx]
    contrib = pd.Series(wrong_shap, index=feature_cols)
    top3 = contrib.nlargest(3)
    
    print(f"  Features pushing toward {pred_label} (wrong):")
    for feat, val in top3.items():
        print(f"    {feat}: SHAP={val:+.4f}, value={X_test_ready.iloc[idx][feat]:.2f}")
    print()"""))

# ── SECTION 13: SUMMARY ──
cells.append(md("""## 13. Summary

### What SHAP Revealed

1. **Feature Importance:** SHAP provides a consistent, unbiased ranking of feature importance — often different from gain-based importance
2. **Per-Subtype Drivers:** Each subtype has distinct feature patterns:
   - RRMS: relapse-related features
   - SPMS: disability progression features
   - PPMS: absence of relapses + progressive markers
   - CIS: low disability scores
3. **Individual Explanations:** Waterfall and force plots explain each prediction for clinical transparency
4. **Feature Interactions:** SHAP reveals which features jointly influence predictions
5. **Misclassification Insights:** Errors often occur at subtype boundaries where clinical features overlap

### Clinical Implications

- SHAP enables **evidence-based trust** in ML predictions
- Individual explanations can support **clinical decision-making**
- Interaction analysis reveals **complex clinical relationships** between features
- Misclassification analysis identifies **improvement opportunities**"""))

# ── SAVE ──
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

path = os.path.join(os.path.dirname(__file__), 'SHAP_Explainability.ipynb')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook: {path}")
print(f"Cells: {len(cells)}")
