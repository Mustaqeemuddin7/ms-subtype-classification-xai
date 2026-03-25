"""
Script to generate a publication-ready Jupyter Notebook for
CatBoost Classifier on MS Subtype Classification.
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
cells.append(md("""# CatBoost Classifier for Multi-Class MS Subtype Classification
## (RRMS, SPMS, PPMS, CIS)

---

**Research Objective:** Develop and evaluate a CatBoost (Categorical Boosting) model for classifying Multiple Sclerosis (MS) subtypes using structured clinical and MRI-derived features.

**Subtypes:**
- **RRMS** — Relapsing-Remitting MS
- **SPMS** — Secondary Progressive MS
- **PPMS** — Primary Progressive MS
- **CIS** — Clinically Isolated Syndrome

---"""))

cells.append(md("""## 1. Imports and Configuration

### Reproducibility

CatBoost uses a deterministic training mode by default when `random_seed` is set, ensuring identical results across runs on the same hardware.

### The Ordered Boosting Principle

CatBoost introduces **ordered boosting** — a novel modification to gradient boosting that:
- Uses a permutation-based scheme to compute residuals, preventing **target leakage** during training
- Each data point's gradient is calculated using a model trained on a different subset, reducing overfitting
- This technique addresses the **prediction shift** problem inherent in traditional gradient boosting

### Why CatBoost for Tabular Medical Data

CatBoost is particularly suited for structured clinical datasets because:
1. **Native categorical support** — encodes categorical features using ordered target statistics, avoiding the information loss of one-hot encoding
2. **Robust to overfitting** — ordered boosting and built-in regularization
3. **Missing value handling** — processes NaN values internally without explicit imputation
4. **Symmetric trees** — balanced tree structure reduces prediction time and improves generalization
5. **Minimal hyperparameter tuning** — strong out-of-the-box performance"""))

cells.append(code("""# ── Core Libraries ──
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ── Scikit-learn Modules ──
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ── CatBoost ──
from catboost import CatBoostClassifier, Pool

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

Comprehensive exploration of the dataset before modeling."""))

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

cells.append(md("""#### Histograms of EDSS and Lesion-Related Features"""))

cells.append(code("""severity_features = ['edss', 'edss_progression_rate', 'lesion_count', 'lesion_volume']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(severity_features):
    for subtype in order:
        subset = df[df['subtype'] == subtype][col].dropna()
        axes[i].hist(subset, bins=20, alpha=0.5, label=subtype, color=CLASS_PALETTE[subtype], edgecolor='white')
    axes[i].set_title(f'{col} Distribution by Subtype', fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].legend(fontsize=8)
fig.suptitle('EDSS and Lesion Feature Distributions', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""#### Boxplots Grouped by Subtype"""))

cells.append(code("""numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
n = len(numeric_cols)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
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

CatBoost handles correlated features effectively through its ordered boosting mechanism. Unlike linear models, collinearity does not cause instability. However, understanding correlations helps interpret feature importance (correlated features share predictive credit)."""))

cells.append(code("""corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()"""))

cells.append(md("""#### Feature Distribution Comparison Across Subtypes"""))

cells.append(code("""key_features = ['edss', 'disease_duration', 'relapse_count', 'brain_volume',
                'lesion_volume', 'edss_progression_rate']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    for subtype in order:
        subset = df[df['subtype'] == subtype][col].dropna()
        if len(subset) > 1:
            subset.plot.kde(ax=axes[i], label=subtype, color=CLASS_PALETTE[subtype], linewidth=2)
    axes[i].set_title(col, fontsize=11)
    axes[i].legend(fontsize=8)
    axes[i].set_ylabel('Density')
fig.suptitle('Feature Density Distributions by Subtype', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 2.2 Clinical Interpretation

- **EDSS & Progression Rate:** Progressive subtypes (SPMS, PPMS) cluster at higher disability scores. CatBoost's symmetric tree structure can efficiently capture these nonlinear thresholds.
- **Lesion Metrics:** Lesion count and volume are key MRI biomarkers. CatBoost can exploit complex interactions between lesion burden and brain volumes.
- **Disease Duration:** The RRMS → SPMS conversion is a gradual process — duration combined with other features creates nonlinear interaction patterns that gradient boosting excels at modeling.
- **Relapse Count:** A near-binary discriminator (PPMS has zero relapses by definition).
- **Class Imbalance:** Addressed through CatBoost's `auto_class_weights='Balanced'` parameter."""))

# =====================================================================
# SECTION 3
# =====================================================================
cells.append(md("""---
## 3. Data Preprocessing

### Why Scaling is Unnecessary for CatBoost

Like all tree-based models, CatBoost makes decisions via **threshold comparisons within individual features**. Feature magnitudes never affect split quality — the algorithm is inherently scale-invariant.

### How CatBoost Handles Categorical Features

CatBoost uses **ordered target statistics** to encode categorical features:

$$x_i^{cat} \\rightarrow \\frac{\\sum_{j=1}^{p-1} [x_{\\sigma_j} = x_{\\sigma_p}] \\cdot y_{\\sigma_j} + a \\cdot P}{\\sum_{j=1}^{p-1} [x_{\\sigma_j} = x_{\\sigma_p}] + a}$$

where $\\sigma$ is a random permutation, $P$ is the prior, and $a$ is the smoothing parameter. This prevents target leakage during encoding.

### Data Leakage Prevention

- **Split before any processing** — train-test separation first
- **CatBoost handles missing values internally** — no explicit imputation needed (uses minimum/maximum value boundaries)
- **Categorical encoding is learned during training** — automatically respects train-test boundaries"""))

cells.append(code("""# Separate features and target
feature_cols = [c for c in df.columns if c != 'subtype']
X = df[feature_cols].copy()
y = df['subtype'].copy()

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
n_classes = len(class_names)
print(f"Classes: {list(class_names)}")
print(f"Encoded: {dict(zip(class_names, le.transform(class_names)))}")
print(f"Number of classes: {n_classes}")"""))

cells.append(code("""# Identify categorical feature indices
# sex_encoded and treatment_status are binary-coded categorical features
cat_feature_names = ['sex_encoded', 'treatment_status']
cat_feature_indices = [feature_cols.index(c) for c in cat_feature_names if c in feature_cols]
print(f"Categorical features: {cat_feature_names}")
print(f"Categorical indices: {cat_feature_indices}")

# Convert categorical columns to integer type for CatBoost
for col in cat_feature_names:
    if col in X.columns:
        X[col] = X[col].astype(int)"""))

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

cells.append(code("""# Create CatBoost Pool objects (native data format)
train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)

print(f"Train Pool: {train_pool.num_row()} samples, {train_pool.num_col()} features")
print(f"Test Pool:  {test_pool.num_row()} samples, {test_pool.num_col()} features")
print(f"Missing values in train: {X_train.isnull().sum().sum()}")
print("Note: CatBoost handles missing values internally — no explicit imputation needed.")
print("Note: No scaling applied — tree-based models are scale-invariant.")"""))

# =====================================================================
# SECTION 4
# =====================================================================
cells.append(md("""---
## 4. Model Building — CatBoost Classifier

### Ordered Boosting Mechanism

Traditional gradient boosting computes gradients using the same data points the model was trained on, causing **prediction shift** — a form of target leakage at the algorithm level.

CatBoost's ordered boosting solves this:
1. Generate a random permutation of the training data
2. For each sample $i$, train a separate model on samples $1, \\ldots, i-1$
3. Use this model to compute the gradient for sample $i$
4. This ensures no data point's gradient depends on itself

### Target Statistics Encoding

For categorical features, CatBoost replaces categories with target-based statistics computed in an ordered fashion — preventing the target leakage that label encoding or target encoding would introduce.

### CatBoost vs. XGBoost

| Aspect | CatBoost | XGBoost |
|--------|----------|---------|
| **Boosting** | Ordered boosting (reduced overfitting) | Traditional gradient boosting |
| **Categorical features** | Native target statistics encoding | Requires manual preprocessing |
| **Missing values** | Internal handling (boundaries) | Internal handling (default direction) |
| **Tree structure** | Symmetric (oblivious) trees | Asymmetric trees |
| **Regularization** | L2 on leaf values + ordered boosting | L1/L2 + subsampling |
| **Default performance** | Strong out-of-the-box | Requires more tuning |
| **Training speed** | Slightly slower (permutation overhead) | Faster on small datasets |

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `loss_function` | `MultiClass` | Softmax cross-entropy for multi-class |
| `iterations` | `500` | Sufficient boosting rounds |
| `learning_rate` | `0.1` | Moderate shrinkage |
| `depth` | `6` | Symmetric tree depth |
| `l2_leaf_reg` | `3.0` | L2 regularization on leaf values |
| `auto_class_weights` | `Balanced` | Compensates for subtype imbalance |
| `random_seed` | `42` | Reproducibility |"""))

cells.append(code("""# Build the CatBoost model
model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',
    iterations=500,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3.0,
    auto_class_weights='Balanced',
    random_seed=RANDOM_STATE,
    verbose=0
)

print("Model Configuration:")
print(f"  Loss function: {model.get_param('loss_function')}")
print(f"  Iterations: {model.get_param('iterations')}")
print(f"  Learning rate: {model.get_param('learning_rate')}")
print(f"  Depth: {model.get_param('depth')}")
print(f"  L2 leaf reg: {model.get_param('l2_leaf_reg')}")
print(f"  Auto class weights: {model.get_param('auto_class_weights')}")"""))

# =====================================================================
# SECTION 5
# =====================================================================
cells.append(md("""---
## 5. Stratified 5-Fold Cross-Validation

### Why Macro F1 is Critical

In MS subtype classification:
- **Macro F1** treats all subtypes equally, regardless of sample size
- Rare subtypes (PPMS, CIS) are clinically just as important as common ones (RRMS)
- A model predicting only RRMS would have decent accuracy but abysmal Macro F1

### Stability Interpretation

CatBoost's ordered boosting typically produces **lower variance across folds** compared to traditional gradient boosting, as the permutation-based gradient computation reduces overfitting to specific fold compositions."""))

cells.append(code("""# Stratified 5-Fold CV (manual loop for CatBoost Pool support)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_accuracy = []
cv_f1_macro = []
cv_f1_weighted = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train[val_idx]
    
    fold_train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_feature_indices)
    
    fold_model = CatBoostClassifier(
        loss_function='MultiClass', iterations=500,
        learning_rate=0.1, depth=6, l2_leaf_reg=3.0,
        auto_class_weights='Balanced', random_seed=RANDOM_STATE, verbose=0
    )
    fold_model.fit(fold_train_pool)
    
    y_fold_pred = fold_model.predict(X_fold_val).flatten().astype(int)
    
    cv_accuracy.append(accuracy_score(y_fold_val, y_fold_pred))
    cv_f1_macro.append(f1_score(y_fold_val, y_fold_pred, average='macro'))
    cv_f1_weighted.append(f1_score(y_fold_val, y_fold_pred, average='weighted'))
    
    print(f"  Fold {fold}: Acc={cv_accuracy[-1]:.4f}, Macro F1={cv_f1_macro[-1]:.4f}, Weighted F1={cv_f1_weighted[-1]:.4f}")

cv_results = {
    'accuracy': np.array(cv_accuracy),
    'f1_macro': np.array(cv_f1_macro),
    'f1_weighted': np.array(cv_f1_weighted)
}

print("\\nCross-Validation Results (5-Fold):")
print(f"  Accuracy:       {cv_results['accuracy'].mean():.4f} ± {cv_results['accuracy'].std():.4f}")
print(f"  Macro F1:       {cv_results['f1_macro'].mean():.4f} ± {cv_results['f1_macro'].std():.4f}")
print(f"  Weighted F1:    {cv_results['f1_weighted'].mean():.4f} ± {cv_results['f1_weighted'].std():.4f}")"""))

cells.append(code("""# Bar chart of CV metrics
metrics_names = ['Accuracy', 'Macro F1', 'Weighted F1']
means = [cv_results[k].mean() for k in ['accuracy', 'f1_macro', 'f1_weighted']]
stds = [cv_results[k].std() for k in ['accuracy', 'f1_macro', 'f1_weighted']]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics_names, means, yerr=stds, capsize=6,
              color=['#2196F3', '#FF5722', '#4CAF50'], edgecolor='white', linewidth=1.2, alpha=0.9)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01,
            f'{m:.3f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Cross-Validation Performance Summary — CatBoost')
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Fold-wise stability
fig, ax = plt.subplots(figsize=(10, 5))
folds = range(1, 6)
for vals, label, color in [(cv_accuracy, 'Accuracy', '#2196F3'),
                            (cv_f1_macro, 'Macro F1', '#FF5722'),
                            (cv_f1_weighted, 'Weighted F1', '#4CAF50')]:
    ax.plot(folds, vals, 'o-', label=label, color=color, linewidth=2, markersize=7)
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

### Early Stopping Rationale

CatBoost monitors performance on a validation set during training. When no improvement is observed for a specified number of rounds (`early_stopping_rounds`), training halts automatically.

Benefits:
- **Prevents over-boosting** — stops before the model memorizes noise
- **Data-driven selection** of optimal iteration count
- **Saves computation** — no need to train for the full `iterations` budget

### Generalization Control

CatBoost's ordered boosting combined with early stopping provides dual protection:
1. **Algorithmic level:** Ordered boosting reduces prediction shift
2. **Training level:** Early stopping prevents excessive iterations"""))

cells.append(code("""# Split training data for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15,
    random_state=RANDOM_STATE, stratify=y_train
)

tr_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)

# Train with early stopping
model_es = CatBoostClassifier(
    loss_function='MultiClass', eval_metric='MultiClass',
    iterations=1000, learning_rate=0.1, depth=6, l2_leaf_reg=3.0,
    auto_class_weights='Balanced', random_seed=RANDOM_STATE,
    early_stopping_rounds=50, verbose=0
)

model_es.fit(tr_pool, eval_set=val_pool)

print(f"Early stopping result:")
print(f"  Best iteration: {model_es.best_iteration_}")
print(f"  Trees used: {model_es.tree_count_}")"""))

cells.append(code("""# Retrain on full training data
model_final = CatBoostClassifier(
    loss_function='MultiClass', eval_metric='MultiClass',
    iterations=500, learning_rate=0.1, depth=6, l2_leaf_reg=3.0,
    auto_class_weights='Balanced', random_seed=RANDOM_STATE, verbose=0
)

model_final.fit(train_pool)

print("Final model trained on full training set.")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Categorical features: {cat_feature_names}")
print(f"  Total trees: {model_final.tree_count_}")
print(f"  Classes: {list(class_names)}")"""))

# =====================================================================
# SECTION 7
# =====================================================================
cells.append(md("""---
## 7. Evaluation on Test Set

Comprehensive evaluation of the CatBoost model on held-out test data."""))

cells.append(code("""# Predictions
y_pred = model_final.predict(test_pool).flatten().astype(int)
y_proba = model_final.predict_proba(test_pool)

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

**Clinical Implications:**
- **CIS → RRMS:** Premature long-term DMT for a potentially isolated event
- **RRMS → SPMS:** Falsely signaling disease progression
- **PPMS → RRMS:** Masking a progressive course with relapse-focused treatment
- CatBoost's ordered boosting often improves discrimination on borderline cases by reducing overfitting to noisy training gradients"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix — Test Set (CatBoost)')
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
ax.set_title('ROC Curves — One-vs-Rest (CatBoost)')
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
ax.set_title('Precision-Recall Curves — Per Class (CatBoost)')
ax.legend(loc='lower left')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 7.4 Feature Importance (Built-in)"""))

cells.append(code("""# Feature importance (PredictionValuesChange — default)
importances = model_final.get_feature_importance()
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_imp = plt.cm.inferno(np.linspace(0.25, 0.85, len(importance_df)))
ax.barh(importance_df['Feature'], importance_df['Importance'],
        color=colors_imp, edgecolor='white', height=0.7)
ax.set_xlabel('Feature Importance (PredictionValuesChange)')
ax.set_title('Feature Importance — CatBoost')
plt.tight_layout()
plt.show()"""))

# =====================================================================
# SECTION 8
# =====================================================================
cells.append(md("""---
## 8. Feature Importance Analysis

### CatBoost Feature Importance Types

CatBoost provides multiple importance measures:

| Type | Description | Best For |
|------|-------------|----------|
| **PredictionValuesChange** (default) | Average change in prediction when the feature is permuted | Overall feature contribution |
| **LossFunctionChange** | How much the loss increases when the feature is excluded | Direct impact on model accuracy |
| **InternalFeatureImportance** | Based on the number of times used in splits and split depth | Feature selection frequency |

### Comparison Across Models

| Method | CatBoost | XGBoost | Random Forest |
|--------|----------|---------|---------------|
| **Primary metric** | PredictionValuesChange | Gain | Gini Importance |
| **What it measures** | Prediction sensitivity | Loss reduction per split | Impurity reduction |
| **Bias** | Less biased (ordered) | Biased toward high-cardinality | Biased toward high-cardinality |
| **Interactions** | Captures interactions | Captures interactions | Individual splits only |

### Clinical Relevance

Expected top predictors in MS classification:
- **EDSS and progression rate** — direct disability quantification
- **Disease duration** — temporal signature of progressive conversion
- **Relapse count** — discriminates relapsing from progressive courses
- **MRI volumes and ratios** — structural neurodegeneration markers"""))

cells.append(code("""# Detailed importance table
importance_table = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
importance_table.index += 1
importance_table.index.name = 'Rank'
importance_table['Cumulative (%)'] = (importance_table['Importance'].cumsum() / 
                                       importance_table['Importance'].sum() * 100).round(1)
importance_table['Importance'] = importance_table['Importance'].round(4)
print("Feature Importance Ranking (PredictionValuesChange):")
print(importance_table.to_string())"""))

cells.append(code("""# Top 10 most important features
top10 = importance_df.tail(10)

fig, ax = plt.subplots(figsize=(10, 6))
colors_top = ['#FF5722' if v >= top10['Importance'].quantile(0.75) else '#2196F3' 
              for v in top10['Importance']]
ax.barh(top10['Feature'], top10['Importance'], color=colors_top, edgecolor='white', height=0.6)
ax.set_xlabel('Feature Importance')
ax.set_title('Top 10 Most Important Features — CatBoost')
for i, (feat, imp) in enumerate(zip(top10['Feature'], top10['Importance'])):
    ax.text(imp + 0.1, i, f'{imp:.2f}', va='center', fontsize=9)
plt.tight_layout()
plt.show()"""))

# =====================================================================
# SECTION 9
# =====================================================================
cells.append(md("""---
## 9. Model Diagnostics

### Ordered Boosting Advantage

Traditional gradient boosting suffers from **prediction shift**: gradients are computed using a model that was trained on the same data points, leading to biased updates. CatBoost's ordered boosting eliminates this by:
- Using random permutations to ensure each gradient is computed with an independent model
- Reducing the effective overfitting without sacrificing model capacity

### Reduced Prediction Shift

The prediction shift problem manifests as:
- Overly optimistic training performance
- Large train-test gap
- Poor calibration of predicted probabilities

CatBoost's approach mitigates all three, typically producing **smaller train-test gaps** than XGBoost or traditional GBDT.

### Depth vs. Learning Rate Tradeoff

CatBoost's symmetric (oblivious) trees constrain complexity differently than asymmetric trees:
- **Symmetric depth 6** ≈ asymmetric depth 8–10 in terms of model capacity
- Symmetric structure ensures all leaves at the same level use the same splits, improving generalization
- Lower depth with higher iterations often outperforms deeper trees with fewer rounds"""))

cells.append(code("""# Overfitting analysis: Train vs Test
y_train_pred = model_final.predict(train_pool).flatten().astype(int)

train_acc = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred, average='macro')
test_acc = acc
test_f1 = f1_mac

print("Overfitting Diagnostic:")
print(f"  Training Accuracy:  {train_acc:.4f}")
print(f"  Test Accuracy:      {test_acc:.4f}")
print(f"  Gap (Accuracy):     {train_acc - test_acc:.4f}")
print()
print(f"  Training Macro F1:  {train_f1:.4f}")
print(f"  Test Macro F1:      {test_f1:.4f}")
print(f"  Gap (Macro F1):     {train_f1 - test_f1:.4f}")
print()
if train_acc - test_acc > 0.15:
    print("  ⚠ Notable train-test gap — consider reducing depth or increasing l2_leaf_reg.")
else:
    print("  ✓ Train-test gap is within acceptable range.")
    print("  (Ordered boosting typically produces smaller gaps than traditional GBDT.)")"""))

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
ax.set_title('Train vs Test Performance — CatBoost')
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

A consolidated dashboard of CatBoost's performance across all key dimensions."""))

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
cv_means = [cv_results['accuracy'].mean(), cv_results['f1_macro'].mean(), cv_results['f1_weighted'].mean()]
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
                color=plt.cm.inferno(np.linspace(0.25, 0.85, len(top8))), edgecolor='white')
axes[1, 0].set_xlabel('Feature Importance')
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

fig.suptitle('Model Performance Dashboard — CatBoost Classifier', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""---
## Summary

This notebook established a **CatBoost baseline** for MS subtype classification. Key findings:

1. **Ordered Boosting:** CatBoost's permutation-based gradient computation reduces prediction shift and overfitting compared to traditional gradient boosting.
2. **Native Categorical Support:** Target statistics encoding handles `sex_encoded` and `treatment_status` without information loss from one-hot encoding.
3. **Missing Value Handling:** CatBoost internally processes NaN values without explicit imputation.
4. **Symmetric Trees:** Oblivious decision trees provide built-in regularization through structural constraints.
5. **Class Imbalance:** `auto_class_weights='Balanced'` adjusts the loss function to ensure equitable learning across all subtypes.
6. **Early Stopping:** Validation-based stopping prevents over-boosting.

**Comparison with Previous Models:**
- CatBoost often matches or exceeds XGBoost's performance with less hyperparameter tuning
- Ordered boosting typically produces smaller train-test performance gaps
- Native categorical support eliminates preprocessing errors

**Next Steps:** Conduct a comprehensive model comparison across all baselines (Logistic Regression, Random Forest, Extra Trees, XGBoost, CatBoost)."""))

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

output_path = os.path.join(os.path.dirname(__file__), 'CatBoost.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {output_path}")
print(f"Total cells: {len(cells)}")
