"""
Script to generate a publication-ready Jupyter Notebook for
Extra Trees Classifier on MS Subtype Classification.
"""
import json, os

def _split_source(source):
    lines = source.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(source)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": _split_source(source), "outputs": [], "execution_count": None}

cells = []

# =====================================================================
# SECTION 1
# =====================================================================
cells.append(md("""# Extra Trees Classifier for Multi-Class MS Subtype Classification
## (RRMS, SPMS, PPMS, CIS)

---

**Research Objective:** Develop and evaluate an Extremely Randomized Trees (Extra Trees) ensemble model for classifying Multiple Sclerosis (MS) subtypes using structured clinical and MRI-derived features.

**Subtypes:**
- **RRMS** — Relapsing-Remitting MS
- **SPMS** — Secondary Progressive MS
- **PPMS** — Primary Progressive MS
- **CIS** — Clinically Isolated Syndrome

---"""))

cells.append(md("""## 1. Imports and Configuration

### Reproducibility and Deterministic Control

Extra Trees introduces **additional randomness** beyond Random Forest — split thresholds are chosen randomly rather than optimally. Controlling `random_state` ensures:

1. **Identical random thresholds** across runs for every split in every tree
2. **Consistent bootstrap behavior** (when enabled) or sample usage
3. **Deterministic feature subset selection** at each node
4. **Reproducible performance metrics** for peer verification"""))

cells.append(code("""# ── Core Libraries ──
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ── Scikit-learn Modules ──
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ── Configuration ──
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')

# Publication-quality plot settings
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

CLASS_PALETTE = {'RRMS': '#2196F3', 'SPMS': '#FF5722', 'PPMS': '#4CAF50', 'CIS': '#9C27B0'}

print("Environment configured successfully.")
print(f"NumPy: {np.__version__}, Pandas: {pd.__version__}")
print(f"Random seed: {RANDOM_STATE}")"""))

# =====================================================================
# SECTION 2
# =====================================================================
cells.append(md("""---
## 2. Data Loading and Exploratory Data Analysis

We load the dataset and perform comprehensive exploration to understand patterns, distributions, and potential challenges before modeling."""))

cells.append(code("""# Load dataset
df = pd.read_csv('ms_dataset.csv')

print(f"Dataset shape: {df.shape[0]} samples × {df.shape[1]} features")
print("\\n" + "="*60)
print("First 5 rows:")
df.head()"""))

cells.append(code("""# Data types and summary
print("Data Types:")
print(df.dtypes)
print("\\n" + "="*60)
print("\\nDescriptive Statistics:")
df.describe().round(3)"""))

cells.append(code("""# Missing values and duplicates
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing, 'Percentage (%)': missing_pct})
print("Missing Values:")
print(missing_df[missing_df['Count'] > 0])
print(f"\\nTotal rows with any missing: {df.isnull().any(axis=1).sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")"""))

cells.append(code("""# Class distribution
print("Subtype Distribution:")
class_dist = df['subtype'].value_counts()
print(class_dist)
print(f"\\nClass ratio (max/min): {class_dist.max() / class_dist.min():.2f}")"""))

cells.append(md("""### 2.1 Visualizations

#### Subtype Distribution"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(8, 5))
order = ['RRMS', 'SPMS', 'PPMS', 'CIS']
colors = [CLASS_PALETTE[s] for s in order]
counts = df['subtype'].value_counts().reindex(order)
bars = ax.bar(order, counts, color=colors, edgecolor='white', linewidth=1.2)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(c), ha='center', va='bottom', fontweight='bold')
ax.set_xlabel('MS Subtype')
ax.set_ylabel('Count')
ax.set_title('Distribution of MS Subtypes')
plt.tight_layout()
plt.show()"""))

cells.append(md("""#### Histograms of Key Clinical Variables"""))

cells.append(code("""numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
n = len(numeric_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    df[col].hist(ax=axes[i], bins=30, color='#5C6BC0', edgecolor='white', alpha=0.85)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel('')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Feature Distributions', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""#### Boxplots Grouped by Subtype

Boxplots reveal inter-subtype feature differences. Extra Trees, like Random Forest, can exploit even subtle distributional differences through its ensemble of diverse decision boundaries."""))

cells.append(code("""fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x='subtype', y=col, order=order,
                palette=CLASS_PALETTE, ax=axes[i], fliersize=2, linewidth=0.8)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Feature Distributions by MS Subtype', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""#### Correlation Heatmap

Tree-based ensembles are **immune to multicollinearity** — correlated features do not cause numerical instability. However, correlated features will share Gini importance, which is important for interpretation."""))

cells.append(code("""corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()"""))

cells.append(md("""#### Feature Distribution Comparison Across Subtypes

Violin plots provide a richer view than boxplots, showing the full density shape of each feature by subtype."""))

cells.append(code("""key_features = ['edss', 'disease_duration', 'relapse_count', 'lesion_volume',
                'brain_volume', 'edss_progression_rate']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    sns.violinplot(data=df, x='subtype', y=col, order=order,
                   palette=CLASS_PALETTE, ax=axes[i], inner='box', linewidth=0.8)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel('')
fig.suptitle('Feature Distribution Comparison Across Subtypes', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 2.2 Clinical Interpretation

- **EDSS:** Progressive subtypes (SPMS, PPMS) show higher disability scores, reflecting cumulative neurodegeneration.
- **Disease Duration:** SPMS patients typically have the longest duration, consistent with the RRMS → SPMS conversion pathway.
- **Relapse Count:** Discriminates relapsing (RRMS, CIS) from progressive (PPMS) subtypes — PPMS has zero relapses by definition.
- **MRI Volumes:** Brain and gray matter atrophy is more pronounced in progressive forms.
- **Class Imbalance:** RRMS dominates the dataset; `class_weight='balanced'` compensates during training."""))

# =====================================================================
# SECTION 3
# =====================================================================
cells.append(md("""---
## 3. Data Preprocessing

### Why Scaling is Unnecessary for Extra Trees

Extra Trees makes splits based on **random thresholds within feature value ranges**. The algorithm:
- Never computes distances or dot products across features
- Only compares values within a single feature at each node
- Is invariant to monotonic transformations

Therefore, **standardization or normalization is unnecessary** and would not change the model's behavior.

### Data Leakage Prevention

Strict protocol:
1. **Split before preprocessing** — train-test split performed first
2. **Fit imputer on training data only** — test set never influences imputation parameters
3. **Transform both sets** — using training-derived parameters"""))

cells.append(code("""# Separate features and target
feature_cols = [c for c in df.columns if c != 'subtype']
X = df[feature_cols].copy()
y = df['subtype'].copy()

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"Classes: {list(class_names)}")
print(f"Encoded: {dict(zip(class_names, le.transform(class_names)))}")"""))

cells.append(code("""# Stratified train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")

# Verify stratification
for label, name in enumerate(class_names):
    tr_pct = (y_train == label).sum() / len(y_train) * 100
    te_pct = (y_test == label).sum() / len(y_test) * 100
    print(f"  {name}: Train {tr_pct:.1f}% | Test {te_pct:.1f}%")"""))

cells.append(code("""# Imputation — fit ONLY on training data (NO scaling)
imputer = SimpleImputer(strategy='median')
X_train_processed = pd.DataFrame(
    imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index
)
X_test_processed = pd.DataFrame(
    imputer.transform(X_test), columns=feature_cols, index=X_test.index
)

print(f"Missing after imputation — Train: {X_train_processed.isnull().sum().sum()}, "
      f"Test: {X_test_processed.isnull().sum().sum()}")
print("\\nNote: No scaling applied — tree-based models are scale-invariant.")"""))

# =====================================================================
# SECTION 4
# =====================================================================
cells.append(md("""---
## 4. Model Building — Extra Trees Classifier

### Extra Trees vs. Random Forest: Key Differences

| Aspect | Random Forest | Extra Trees |
|--------|--------------|-------------|
| **Bootstrap** | Yes (samples with replacement) | No (uses full training set by default) |
| **Split threshold** | Best threshold among candidates | **Random threshold** per candidate feature |
| **Bias** | Low | Slightly higher (random splits less optimal) |
| **Variance** | Reduced by bagging | **Further reduced** by extreme randomization |
| **Overfitting** | Moderate resistance | **Stronger resistance** |
| **Speed** | Slower (evaluates many thresholds) | **Faster** (random thresholds) |

### The Extreme Randomization Principle

At each node, Extra Trees:
1. Selects a random subset of $k$ features (same as RF, typically $k = \\sqrt{p}$)
2. For each selected feature, draws a **random split threshold** uniformly from the feature's range
3. Chooses the best (feature, threshold) pair among these random candidates

This additional randomness:
- **Reduces variance** more aggressively than Random Forest
- **Increases bias** slightly (suboptimal individual splits)
- **Net effect:** often comparable or better generalization, especially on noisy data

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | `300` | Sufficient for stable ensemble predictions |
| `criterion` | `gini` | Standard impurity-based splitting |
| `class_weight` | `balanced` | Compensates for subtype imbalance |
| `max_depth` | `None` | Full-depth trees for maximum expressivity |
| `random_state` | `42` | Reproducibility |"""))

cells.append(code("""# Build the Extra Trees model
model = ExtraTreesClassifier(
    n_estimators=300,
    criterion='gini',
    max_depth=None,
    class_weight='balanced',
    n_jobs=-1,
    random_state=RANDOM_STATE
)

print("Model Configuration:")
print(model)"""))

# =====================================================================
# SECTION 5
# =====================================================================
cells.append(md("""---
## 5. Stratified 5-Fold Cross-Validation

### Model Stability

Extra Trees typically shows **lower variance across folds** compared to Random Forest due to its stronger randomization. This translates to more consistent performance estimates.

### Macro F1 for Imbalanced Subtypes

- **Macro F1** gives equal weight to every subtype regardless of prevalence
- Ensures clinically important but rare subtypes (PPMS, CIS) are not overlooked
- A model with high accuracy but low Macro F1 is merely predicting the majority class"""))

cells.append(code("""# Stratified 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'f1_weighted': 'f1_weighted'
}

cv_results = cross_validate(
    model, X_train_processed, y_train,
    cv=cv, scoring=scoring, return_train_score=False
)

print("Cross-Validation Results (5-Fold):")
print(f"  Accuracy:       {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
print(f"  Macro F1:       {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")
print(f"  Weighted F1:    {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")"""))

cells.append(code("""# Bar chart of CV metrics
metrics_names = ['Accuracy', 'Macro F1', 'Weighted F1']
means = [cv_results[f'test_{m}'].mean() for m in ['accuracy', 'f1_macro', 'f1_weighted']]
stds = [cv_results[f'test_{m}'].std() for m in ['accuracy', 'f1_macro', 'f1_weighted']]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics_names, means, yerr=stds, capsize=6,
              color=['#2196F3', '#FF5722', '#4CAF50'], edgecolor='white', linewidth=1.2, alpha=0.9)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
            f'{m:.3f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Cross-Validation Performance Summary — Extra Trees')
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Fold-wise performance
fig, ax = plt.subplots(figsize=(10, 5))
folds = range(1, 6)
for key, label, color in [('test_accuracy', 'Accuracy', '#2196F3'),
                           ('test_f1_macro', 'Macro F1', '#FF5722'),
                           ('test_f1_weighted', 'Weighted F1', '#4CAF50')]:
    ax.plot(folds, cv_results[key], 'o-', label=label, color=color, linewidth=2, markersize=7)
ax.set_xlabel('Fold')
ax.set_ylabel('Score')
ax.set_title('Fold-wise Cross-Validation Performance')
ax.set_xticks(list(folds))
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()"""))

# =====================================================================
# SECTION 6
# =====================================================================
cells.append(md("""---
## 6. Final Model Training

### Why Retrain on the Full Training Set?

Cross-validation used only ~80% of training data per fold. Final retraining on **all training samples**:

1. **Maximizes available information** for learning complex subtype boundaries
2. **Produces the definitive model** for test evaluation and feature importance
3. **Leverages every sample** — Extra Trees uses the full dataset (no bootstrap by default), making this especially impactful"""))

cells.append(code("""# Train on full training set
model.fit(X_train_processed, y_train)

print("Model trained on full training set.")
print(f"  Training samples: {X_train_processed.shape[0]}")
print(f"  Features: {X_train_processed.shape[1]}")
print(f"  Number of trees: {model.n_estimators}")
print(f"  Classes: {list(class_names)}")"""))

# =====================================================================
# SECTION 7
# =====================================================================
cells.append(md("""---
## 7. Evaluation on Test Set

We evaluate the trained Extra Trees model on the held-out test set using comprehensive metrics."""))

cells.append(code("""# Predictions
y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)

# Overall metrics
acc = accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average='macro')
f1_wt = f1_score(y_test, y_pred, average='weighted')

print("Test Set Performance:")
print(f"  Accuracy:       {acc:.4f}")
print(f"  Macro F1:       {f1_mac:.4f}")
print(f"  Weighted F1:    {f1_wt:.4f}")

# Per-class metrics
print("\\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))"""))

cells.append(code("""# OvR ROC-AUC
y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
try:
    roc_auc_ovr = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro')
    print(f"One-vs-Rest ROC-AUC (macro): {roc_auc_ovr:.4f}")
except ValueError as e:
    print(f"ROC-AUC could not be computed: {e}")
    roc_auc_ovr = None"""))

cells.append(md("""### 7.1 Confusion Matrix

**Clinical Implications of Misclassifications:**
- **CIS → RRMS:** May lead to premature long-term therapy initiation
- **RRMS → SPMS:** Could trigger inappropriate switch to progressive treatment protocols
- **PPMS → SPMS:** Both progressive, but treatment strategies differ
- Minimizing false negatives for progressive subtypes is clinically critical"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix — Test Set (Extra Trees)')
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 7.2 ROC Curves (One-vs-Rest)"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 7))
for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc_i = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc_i:.3f})',
            color=CLASS_PALETTE.get(name, None))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — One-vs-Rest (Extra Trees)')
ax.legend(loc='lower right')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 7.3 Precision-Recall Curves"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 7))
for i, name in enumerate(class_names):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
    ax.plot(rec, prec, linewidth=2, label=f'{name} (AP = {ap:.3f})',
            color=CLASS_PALETTE.get(name, None))
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Per Class (Extra Trees)')
ax.legend(loc='lower left')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 7.4 Feature Importance (Built-in)"""))

cells.append(code("""# Feature importance bar plot
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
ax.barh(importance_df['Feature'], importance_df['Importance'],
        color=colors_imp, edgecolor='white', height=0.7)
ax.set_xlabel('Gini Importance')
ax.set_title('Feature Importance — Extra Trees')
plt.tight_layout()
plt.show()"""))

# =====================================================================
# SECTION 8
# =====================================================================
cells.append(md("""---
## 8. Feature Importance Analysis

### Gini Importance in Extra Trees

Gini importance measures the **total reduction in impurity** contributed by each feature across all trees:

$$\\text{Importance}(j) = \\sum_{t=1}^{T} \\sum_{n \\in \\text{nodes using } j} w_n \\cdot \\Delta G(n)$$

where $w_n$ is the fraction of samples reaching node $n$ and $\\Delta G(n)$ is the Gini impurity decrease.

### Comparison with Linear Model Coefficients

| Aspect | Extra Trees Importance | Logistic Regression Coefficients |
|--------|----------------------|--------------------------------|
| **Sign** | Always ≥ 0 (magnitude only) | Positive or negative (directional) |
| **Nonlinearity** | Captures nonlinear and interaction effects | Assumes linearity in log-odds |
| **Collinearity** | Splits importance among correlated features | Inflates coefficient variance |
| **Interpretation** | "How much does this feature reduce impurity?" | "How does a unit change affect log-odds?" |

### Clinical Relevance

In MS subtype classification, we expect high importance for:
- **EDSS / progression rate** — direct disability measures differentiating progressive from relapsing forms
- **Disease duration** — SPMS evolves from long-standing RRMS
- **Relapse count** — zero in PPMS, present in RRMS/CIS
- **Lesion metrics** — MRI biomarkers of disease burden"""))

cells.append(code("""# Detailed importance table
importance_table = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
importance_table.index += 1
importance_table.index.name = 'Rank'
importance_table['Cumulative (%)'] = (importance_table['Importance'].cumsum() / 
                                       importance_table['Importance'].sum() * 100).round(1)
importance_table['Importance'] = importance_table['Importance'].round(6)
print("Feature Importance Ranking:")
print(importance_table.to_string())"""))

cells.append(code("""# Top 10 most important features
top10 = importance_df.tail(10)

fig, ax = plt.subplots(figsize=(10, 6))
colors_top = ['#FF5722' if v >= top10['Importance'].quantile(0.75) else '#2196F3' 
              for v in top10['Importance']]
ax.barh(top10['Feature'], top10['Importance'], color=colors_top, edgecolor='white', height=0.6)
ax.set_xlabel('Gini Importance')
ax.set_title('Top 10 Most Important Features — Extra Trees')
for i, (feat, imp) in enumerate(zip(top10['Feature'], top10['Importance'])):
    ax.text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.show()"""))

# =====================================================================
# SECTION 9
# =====================================================================
cells.append(md("""---
## 9. Model Diagnostics

### Ensemble Robustness

Extra Trees achieves robustness through **extreme randomization**:
- Random split thresholds prevent individual trees from memorizing training patterns
- Aggregation across 300 diverse trees smooths out noise
- Using the full dataset (no bootstrap) means every tree sees all available information

### Impact of Extreme Randomization on Variance

Compared to Random Forest:
- **Lower variance** — random thresholds create more diverse trees, making the ensemble more stable
- **Slightly higher bias** — individual splits are suboptimal by design
- **Net effect** — often similar or better generalization, particularly on noisy or high-dimensional data

### Overfitting Assessment

Extra Trees with `max_depth=None` can still perfectly fit training data. However:
- The ensemble averaging mitigates individual tree overfitting
- A moderate train-test gap is expected and acceptable
- Large gaps warrant tuning `max_depth` or `min_samples_leaf`"""))

cells.append(code("""# Overfitting analysis: Train vs Test
y_train_pred = model.predict(X_train_processed)

train_acc = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='macro')
test_acc = acc
test_f1 = f1_mac

print("Overfitting Diagnostic:")
print(f"  Training Accuracy:  {train_acc:.4f}")
print(f"  Test Accuracy:      {test_acc:.4f}")
print(f"  Gap (Accuracy):     {train_acc - test_acc:.4f}")
print(f"")
print(f"  Training Macro F1:  {train_f1:.4f}")
print(f"  Test Macro F1:      {test_f1:.4f}")
print(f"  Gap (Macro F1):     {train_f1 - test_f1:.4f}")
print()
if train_acc - test_acc > 0.15:
    print("  ⚠ Notable gap between train and test — potential overfitting.")
    print("  Consider tuning max_depth or min_samples_leaf.")
else:
    print("  ✓ Train-test gap is within acceptable range.")"""))

cells.append(code("""# Train vs Test visualization
fig, ax = plt.subplots(figsize=(8, 5))
metrics = ['Accuracy', 'Macro F1']
train_vals = [train_acc, train_f1]
test_vals_plot = [test_acc, test_f1]

x = np.arange(len(metrics))
w = 0.3
ax.bar(x - w/2, train_vals, w, label='Train', color='#7E57C2', edgecolor='white')
ax.bar(x + w/2, test_vals_plot, w, label='Test', color='#26A69A', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_title('Train vs Test Performance — Extra Trees')
ax.legend()
ax.set_ylim(0, 1.15)
for i in range(len(metrics)):
    ax.text(x[i] - w/2, train_vals[i] + 0.02, f'{train_vals[i]:.3f}', ha='center', fontsize=9)
    ax.text(x[i] + w/2, test_vals_plot[i] + 0.02, f'{test_vals_plot[i]:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()"""))

# =====================================================================
# SECTION 10
# =====================================================================
cells.append(md("""---
## 10. Model Performance Summary

A consolidated dashboard of the Extra Trees model's performance."""))

cells.append(code("""# Comprehensive performance dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# (a) Per-class precision, recall, F1
prec_per = precision_score(y_test, y_pred, average=None)
rec_per = recall_score(y_test, y_pred, average=None)
f1_per = f1_score(y_test, y_pred, average=None)

x_pos = np.arange(len(class_names))
w = 0.25
axes[0, 0].bar(x_pos - w, prec_per, w, label='Precision', color='#2196F3', edgecolor='white')
axes[0, 0].bar(x_pos, rec_per, w, label='Recall', color='#FF5722', edgecolor='white')
axes[0, 0].bar(x_pos + w, f1_per, w, label='F1-Score', color='#4CAF50', edgecolor='white')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(class_names)
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('(a) Per-Class Metrics')
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1.15)

# (b) CV vs Test
cv_means = [cv_results['test_accuracy'].mean(), cv_results['test_f1_macro'].mean(), cv_results['test_f1_weighted'].mean()]
test_vals_b = [acc, f1_mac, f1_wt]
metric_labels = ['Accuracy', 'Macro F1', 'Weighted F1']
x2 = np.arange(len(metric_labels))
axes[0, 1].bar(x2 - 0.15, cv_means, 0.3, label='CV (mean)', color='#7E57C2', edgecolor='white')
axes[0, 1].bar(x2 + 0.15, test_vals_b, 0.3, label='Test', color='#26A69A', edgecolor='white')
axes[0, 1].set_xticks(x2)
axes[0, 1].set_xticklabels(metric_labels)
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('(b) CV vs Test Performance')
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1.15)

# (c) Feature importance (top 8)
top8 = importance_df.tail(8)
axes[1, 0].barh(top8['Feature'], top8['Importance'],
                color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top8))), edgecolor='white')
axes[1, 0].set_xlabel('Gini Importance')
axes[1, 0].set_title('(c) Top 8 Feature Importances')

# (d) Per-class AUC
auc_per_class = []
for i in range(len(class_names)):
    fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    auc_per_class.append(auc(fpr_i, tpr_i))
axes[1, 1].bar(class_names, auc_per_class, color=[CLASS_PALETTE[n] for n in class_names], edgecolor='white')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].set_title('(d) Per-Class ROC-AUC')
axes[1, 1].set_ylim(0, 1.15)
for i, v in enumerate(auc_per_class):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

fig.suptitle('Model Performance Dashboard — Extra Trees Classifier', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""---
## Summary

This notebook established an **Extra Trees (Extremely Randomized Trees) baseline** for MS subtype classification. Key findings:

1. **Extreme Randomization:** Random split thresholds (vs. optimal in RF) provide stronger variance reduction, often yielding more stable predictions.
2. **Feature Insights:** Gini importance identifies clinical and MRI features most relevant for subtype discrimination.
3. **No Scaling Required:** Like Random Forest, Extra Trees is invariant to feature scales.
4. **Class Imbalance:** `class_weight='balanced'` ensures minority subtypes are equitably represented in the loss function.
5. **Overfitting Resistance:** The combination of extreme randomization and ensemble averaging provides inherent regularization.

**Comparison Note:** Extra Trees typically shows:
- **Lower variance** than Random Forest (more stable across data perturbations)
- **Slightly higher bias** (random splits are suboptimal individually)
- **Faster training** (no threshold optimization per feature)
- **Comparable generalization** on most real-world datasets

**Next Steps:** Compare with gradient boosting methods (XGBoost) and conduct a final model comparison across all baselines."""))

# =====================================================================
# ASSEMBLE NOTEBOOK
# =====================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

output_path = os.path.join(os.path.dirname(__file__), 'Extra_Trees.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {output_path}")
print(f"Total cells: {len(cells)}")
