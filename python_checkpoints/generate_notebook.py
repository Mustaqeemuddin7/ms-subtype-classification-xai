"""
Script to generate a publication-ready Jupyter Notebook for
Multinomial Logistic Regression on MS Subtype Classification.
"""
import json, os

def _split_source(source):
    lines = source.split("\n")
    # ipynb requires each line except the last to end with \n
    return [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(source)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": _split_source(source), "outputs": [], "execution_count": None}

cells = []

# =====================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# =====================================================================
cells.append(md("""# Multinomial Logistic Regression for Multi-Class MS Subtype Classification
## (RRMS, SPMS, PPMS, CIS)

---

**Research Objective:** Develop and evaluate a Multinomial Logistic Regression baseline model for classifying Multiple Sclerosis (MS) subtypes using structured clinical and MRI-derived features.

**Subtypes:**
- **RRMS** — Relapsing-Remitting MS
- **SPMS** — Secondary Progressive MS
- **PPMS** — Primary Progressive MS
- **CIS** — Clinically Isolated Syndrome

---"""))

cells.append(md("""## 1. Imports and Configuration

### Reproducibility and Deterministic Setup

Reproducibility is a cornerstone of rigorous scientific research. Setting a fixed random seed ensures that:

1. **Data splits** remain identical across runs, enabling fair comparison of models.
2. **Stochastic optimization** (e.g., solver initialization) produces consistent results.
3. **Third-party reviewers** can exactly replicate all reported metrics.

We suppress warnings to maintain clean notebook output and configure matplotlib for publication-quality figures with high-DPI rendering."""))

cells.append(code("""# ── Core Libraries ──
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# ── Scikit-learn Modules ──
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# ── Multicollinearity ──
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
# SECTION 2: DATA LOADING AND EXPLORATION
# =====================================================================
cells.append(md("""---
## 2. Data Loading and Initial Exploration

We load the dataset and perform a comprehensive initial inspection covering:
- **Shape and schema** — number of samples, features, and data types
- **Missing values** — identify features requiring imputation
- **Duplicates** — check for repeated observations that could bias training
- **Class distribution** — assess balance across MS subtypes"""))

cells.append(code("""# Load dataset
df = pd.read_csv('../datasets/ms_dataset.csv')

print(f"Dataset shape: {df.shape[0]} samples × {df.shape[1]} features")
print("\\n" + "="*60)
print("First 5 rows:")
df.head()"""))

cells.append(code("""# Data types
print("Data Types:")
print(df.dtypes)
print("\\n" + "="*60)
print("\\nDescriptive Statistics:")
df.describe().round(3)"""))

cells.append(code("""# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing, 'Percentage (%)': missing_pct})
print("Missing Values:")
print(missing_df[missing_df['Count'] > 0])
print(f"\\nTotal rows with any missing: {df.isnull().any(axis=1).sum()}")"""))

cells.append(code("""# Duplicates
n_dup = df.duplicated().sum()
print(f"Duplicate rows: {n_dup}")"""))

cells.append(code("""# Class distribution
print("Subtype Distribution:")
class_dist = df['subtype'].value_counts()
print(class_dist)
print(f"\\nClass ratio (max/min): {class_dist.max() / class_dist.min():.2f}")"""))

cells.append(md("""### 2.1 Visualizations

#### Subtype Distribution
The bar plot below reveals the class balance. Significant imbalance can bias the model toward majority classes, necessitating techniques such as class weighting."""))

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

cells.append(md("""#### Histograms of Numeric Features
Histograms reveal the distributional shape (skewness, modality) of each feature, informing preprocessing decisions."""))

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
Boxplots stratified by subtype highlight differences in feature medians, spreads, and outliers across classes — guiding clinical interpretation."""))

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
The heatmap exposes linear relationships among features. High correlations (|r| > 0.8) signal potential multicollinearity, which can inflate coefficient variance in logistic regression."""))

cells.append(code("""corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 2.2 Clinical Interpretation of Key Observations

- **EDSS (Expanded Disability Status Scale):** Higher EDSS scores are expected in progressive subtypes (SPMS, PPMS) compared to RRMS and CIS.
- **Disease Duration & Age at Onset:** SPMS patients typically have longer disease durations, reflecting the natural transition from RRMS.
- **Relapse Count:** RRMS is characterized by relapses, while PPMS typically shows no relapses — a key discriminative feature.
- **Brain/GM/WM Volumes:** Progressive subtypes often exhibit greater atrophy (lower volumes).
- **Lesion Metrics:** Higher lesion counts and volumes are associated with more advanced disease.
- **Class Imbalance:** RRMS is the majority class, which can bias model predictions. We address this with `class_weight='balanced'`."""))

# =====================================================================
# SECTION 3: DATA PREPROCESSING
# =====================================================================
cells.append(md("""---
## 3. Data Preprocessing

### Data Leakage Prevention

**Data leakage** occurs when information from the test set influences model training, leading to overly optimistic performance estimates that do not generalize. To prevent this:

1. **Split first, preprocess second:** We perform the train-test split before any imputation or scaling.
2. **Fit on training data only:** Imputers and scalers learn parameters (mean, std) exclusively from the training set.
3. **Transform both sets:** The same learned parameters are applied to both training and test data.

### Why Scaling is Required for Logistic Regression

Logistic Regression uses gradient-based optimization (L-BFGS). Features on vastly different scales (e.g., `brain_volume` ~1.5M vs. `edss` ~0–9) cause:

- **Elongated loss contours** → slow, oscillating convergence
- **Disproportionate regularization** — L2 penalty treats all coefficients equally, so unscaled features receive unequal effective regularization

**StandardScaler** transforms each feature $x$ to:

$$z = \\frac{x - \\mu}{\\sigma}$$

where $\\mu$ and $\\sigma$ are the training-set mean and standard deviation, respectively."""))

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

cells.append(code("""# Imputation — fit ONLY on training data
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test), columns=feature_cols, index=X_test.index
)

print(f"Missing after imputation — Train: {X_train_imputed.isnull().sum().sum()}, "
      f"Test: {X_test_imputed.isnull().sum().sum()}")"""))

cells.append(code("""# Standardization — fit ONLY on training data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imputed), columns=feature_cols, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imputed), columns=feature_cols, index=X_test.index
)

print("Scaled training set statistics (should be ~0 mean, ~1 std):")
print(X_train_scaled.describe().loc[['mean', 'std']].round(3))"""))

# =====================================================================
# SECTION 4: MODEL BUILDING
# =====================================================================
cells.append(md("""---
## 4. Model Building — Multinomial Logistic Regression

### Multinomial Softmax Formulation

For $K$ classes, the model estimates:

$$P(Y = k \\mid \\mathbf{x}) = \\frac{\\exp(\\mathbf{w}_k^T \\mathbf{x} + b_k)}{\\sum_{j=1}^{K} \\exp(\\mathbf{w}_j^T \\mathbf{x} + b_j)}$$

where $\\mathbf{w}_k$ is the coefficient vector for class $k$.

### Regularization (L2)

L2 regularization adds a penalty $\\lambda \\sum_k \\|\\mathbf{w}_k\\|_2^2$ to the loss function, which:
- Prevents overfitting by shrinking coefficients toward zero
- Improves numerical stability
- Encourages distributed feature contributions rather than reliance on a single feature

### Configuration Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `solver` | `lbfgs` | Efficient quasi-Newton method for multinomial problems |
| `max_iter` | `1000` | Sufficient iterations for convergence |
| `class_weight` | `balanced` | Adjusts for class imbalance by up-weighting minority classes |
| `C` | `1.0` | Default inverse regularization strength |"""))

cells.append(code("""# Build the Multinomial Logistic Regression model
model = LogisticRegression(
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    C=1.0,
    random_state=RANDOM_STATE
)

print("Model Configuration:")
print(model)"""))

# =====================================================================
# SECTION 5: CROSS-VALIDATION
# =====================================================================
cells.append(md("""---
## 5. Stratified 5-Fold Cross-Validation

Cross-validation provides a robust estimate of model generalization by:
1. Training and evaluating on **different subsets** of the training data
2. Reporting **mean ± std** of metrics to assess stability

### Why Macro F1-Score Matters in Multi-Class Settings

- **Macro F1** computes F1 per class and averages equally, giving **equal importance to minority classes**
- This is critical when all subtypes are clinically important, regardless of prevalence
- A high accuracy with low Macro F1 indicates the model fails on rare classes

### Bias–Variance Interpretation

- **Low variance across folds** → stable model, unlikely to overfit
- **High variance** → sensitivity to data splits, suggesting insufficient data or model complexity"""))

cells.append(code("""# Stratified 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'f1_weighted': 'f1_weighted'
}

cv_results = cross_validate(
    model, X_train_scaled, y_train,
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
ax.set_title('Cross-Validation Performance Summary')
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
# SECTION 6: FINAL MODEL TRAINING
# =====================================================================
cells.append(md("""---
## 6. Final Model Training

### Why Retrain on the Full Training Set?

During cross-validation, the model was trained on only 80% of the training data in each fold. Now that we have validated performance, we retrain on the **entire training set** to:

1. **Maximize information** available for learning decision boundaries
2. **Produce the definitive model** for test-set evaluation
3. **Obtain stable coefficients** for interpretation"""))

cells.append(code("""# Train on full training set
model.fit(X_train_scaled, y_train)

print("Model trained on full training set.")
print(f"  Training samples: {X_train_scaled.shape[0]}")
print(f"  Features: {X_train_scaled.shape[1]}")
print(f"  Classes: {list(class_names)}")
print(f"  Converged: {model.n_iter_[0]} iterations")"""))

# =====================================================================
# SECTION 7: EVALUATION ON TEST SET
# =====================================================================
cells.append(md("""---
## 7. Evaluation on Test Set

We evaluate the trained model on the held-out test set using multiple metrics to capture different aspects of performance."""))

cells.append(code("""# Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

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

The confusion matrix reveals where the model confuses subtypes. Off-diagonal entries represent misclassifications.

**Clinical Significance:**
- **False Positives (FP):** A patient misclassified as having a more severe subtype (e.g., RRMS → SPMS) may receive unnecessarily aggressive treatment.
- **False Negatives (FN):** A progressive subtype misclassified as RRMS may delay escalation of therapy, risking irreversible disability."""))

cells.append(code("""fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix — Test Set')
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 7.2 ROC Curves (One-vs-Rest)

ROC curves plot the True Positive Rate against the False Positive Rate at varying classification thresholds. The Area Under the Curve (AUC) summarizes discriminative ability; AUC = 1.0 indicates perfect separation."""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 7))
for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc_i = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc_i:.3f})',
            color=CLASS_PALETTE.get(name, None))
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — One-vs-Rest')
ax.legend(loc='lower right')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 7.3 Precision-Recall Curves

Precision-Recall curves are especially informative for imbalanced classes, as they focus on the model's ability to identify positive instances without flooding predictions with false positives."""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 7))
for i, name in enumerate(class_names):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
    ax.plot(rec, prec, linewidth=2, label=f'{name} (AP = {ap:.3f})',
            color=CLASS_PALETTE.get(name, None))
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Per Class')
ax.legend(loc='lower left')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.05])
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 7.4 Misclassification Analysis

Examining which samples are misclassified helps identify patterns the model struggles with — potentially highlighting atypical presentations or boundary cases between subtypes."""))

cells.append(code("""# Misclassification analysis
misclassified_mask = y_pred != y_test
n_mis = misclassified_mask.sum()
print(f"Misclassified: {n_mis} / {len(y_test)} ({n_mis/len(y_test)*100:.1f}%)")

# Confusion pairs
mis_pairs = pd.DataFrame({'True': class_names[y_test[misclassified_mask]],
                           'Predicted': class_names[y_pred[misclassified_mask]]})
pair_counts = mis_pairs.groupby(['True', 'Predicted']).size().reset_index(name='Count')
pair_counts = pair_counts.sort_values('Count', ascending=False)
print("\\nMost common misclassification pairs:")
print(pair_counts.to_string(index=False))

# Plot
if len(pair_counts) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r['True']} → {r['Predicted']}" for _, r in pair_counts.iterrows()]
    ax.barh(labels[::-1], pair_counts['Count'].values[::-1], color='#E57373', edgecolor='white')
    ax.set_xlabel('Count')
    ax.set_title('Misclassification Pairs')
    plt.tight_layout()
    plt.show()"""))

# =====================================================================
# SECTION 8: COEFFICIENT AND FEATURE ANALYSIS
# =====================================================================
cells.append(md("""---
## 8. Coefficient and Feature Analysis

### Interpreting Logistic Regression Coefficients

For each class $k$, the coefficient $w_{k,j}$ for feature $j$ represents the change in **log-odds** of class $k$ (vs. the reference) per unit increase in $x_j$ (after standardization):

$$\\log\\frac{P(Y=k)}{P(Y=\\text{ref})} = \\mathbf{w}_k^T \\mathbf{x} + b_k$$

- **Positive coefficient** → feature increases the probability of that subtype
- **Negative coefficient** → feature decreases the probability
- **Magnitude** → strength of the association

### Limitations

Linear models assume a linear relationship in the log-odds space. Complex nonlinear biological interactions (e.g., gene-environment, threshold effects in neurodegeneration) may not be captured."""))

cells.append(code("""# Coefficient table
coef_df = pd.DataFrame(model.coef_, columns=feature_cols, index=class_names)
print("Model Coefficients (per class):")
coef_df.round(4)"""))

cells.append(code("""# Top features per class by absolute magnitude
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, name in enumerate(class_names):
    coefs = coef_df.loc[name].sort_values(key=abs, ascending=True)
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in coefs]
    axes[idx].barh(coefs.index, coefs.values, color=colors, edgecolor='white', height=0.7)
    axes[idx].set_title(f'{name} — Feature Coefficients', fontweight='bold')
    axes[idx].axvline(x=0, color='black', linewidth=0.8, linestyle='-')
    axes[idx].set_xlabel('Coefficient')
for j in range(idx+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Feature Contributions by MS Subtype', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(code("""# Heatmap of coefficients
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(coef_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Coefficient'})
ax.set_title('Coefficient Heatmap — All Subtypes × Features')
ax.set_ylabel('Subtype')
plt.tight_layout()
plt.show()"""))

# =====================================================================
# SECTION 9: MODEL DIAGNOSTICS
# =====================================================================
cells.append(md("""---
## 9. Model Diagnostics

### Assumptions of Logistic Regression

1. **Linearity in log-odds:** The log-odds of each class are a linear function of the features.
2. **Independence of observations:** Each patient is an independent sample.
3. **No severe multicollinearity:** Highly correlated features inflate coefficient variance and make interpretation unreliable.

### Variance Inflation Factor (VIF)

VIF quantifies how much the variance of a coefficient is inflated due to collinearity:

$$\\text{VIF}_j = \\frac{1}{1 - R_j^2}$$

- VIF > 5 → moderate collinearity (warrants investigation)
- VIF > 10 → severe collinearity (consider removing or combining features)"""))

cells.append(code("""# VIF calculation
vif_data = pd.DataFrame()
vif_data['Feature'] = feature_cols
vif_data['VIF'] = [variance_inflation_factor(X_train_scaled.values, i) for i in range(len(feature_cols))]
vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

print("Variance Inflation Factors:")
print(vif_data.to_string(index=False))
print(f"\\nFeatures with VIF > 5: {(vif_data['VIF'] > 5).sum()}")
print(f"Features with VIF > 10: {(vif_data['VIF'] > 10).sum()}")"""))

cells.append(code("""# VIF bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors_vif = ['#F44336' if v > 10 else '#FF9800' if v > 5 else '#4CAF50' for v in vif_data['VIF']]
ax.barh(vif_data['Feature'][::-1], vif_data['VIF'][::-1], color=colors_vif[::-1], edgecolor='white')
ax.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, label='Moderate (VIF=5)')
ax.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='Severe (VIF=10)')
ax.set_xlabel('Variance Inflation Factor')
ax.set_title('Multicollinearity Assessment (VIF)')
ax.legend()
plt.tight_layout()
plt.show()"""))

cells.append(md("""### 9.1 Residual Misclassification Analysis

We examine the predicted probability distributions for correctly and incorrectly classified samples to assess model confidence."""))

cells.append(code("""# Confidence analysis
max_proba = y_proba.max(axis=1)
correct_mask = y_pred == y_test

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(max_proba[correct_mask], bins=25, alpha=0.7, label='Correct', color='#4CAF50', edgecolor='white')
ax.hist(max_proba[~correct_mask], bins=25, alpha=0.7, label='Misclassified', color='#F44336', edgecolor='white')
ax.set_xlabel('Maximum Predicted Probability')
ax.set_ylabel('Count')
ax.set_title('Prediction Confidence Distribution')
ax.legend()
plt.tight_layout()
plt.show()

print(f"Mean confidence (correct):       {max_proba[correct_mask].mean():.4f}")
if (~correct_mask).sum() > 0:
    print(f"Mean confidence (misclassified): {max_proba[~correct_mask].mean():.4f}")"""))

# =====================================================================
# SECTION 10: PERFORMANCE VISUALIZATION SUMMARY
# =====================================================================
cells.append(md("""---
## 10. Model Performance Summary Visualization

A consolidated view of the model's performance across all key metrics and classes."""))

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

# (b) Overall metrics comparison: CV vs Test
cv_means = [cv_results['test_accuracy'].mean(), cv_results['test_f1_macro'].mean(), cv_results['test_f1_weighted'].mean()]
test_vals = [acc, f1_mac, f1_wt]
metric_labels = ['Accuracy', 'Macro F1', 'Weighted F1']
x2 = np.arange(len(metric_labels))
axes[0, 1].bar(x2 - 0.15, cv_means, 0.3, label='CV (mean)', color='#7E57C2', edgecolor='white')
axes[0, 1].bar(x2 + 0.15, test_vals, 0.3, label='Test', color='#26A69A', edgecolor='white')
axes[0, 1].set_xticks(x2)
axes[0, 1].set_xticklabels(metric_labels)
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('(b) CV vs Test Performance')
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1.15)

# (c) Per-class support
support = np.bincount(y_test, minlength=len(class_names))
axes[1, 0].bar(class_names, support, color=[CLASS_PALETTE[n] for n in class_names], edgecolor='white')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('(c) Test Set Class Distribution')
for i, v in enumerate(support):
    axes[1, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

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

fig.suptitle('Model Performance Dashboard — Multinomial Logistic Regression', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()"""))

cells.append(md("""---
## Summary

This notebook established a **Multinomial Logistic Regression baseline** for MS subtype classification. Key takeaways:

1. **Linear Model Baseline:** Logistic Regression provides an interpretable, well-calibrated baseline against which more complex models (Random Forest, XGBoost, etc.) can be compared.
2. **Coefficient Interpretability:** The model reveals which clinical and MRI features most strongly distinguish each subtype.
3. **Class Imbalance Handling:** Using `class_weight='balanced'` ensures minority subtypes (PPMS, CIS) are not ignored.
4. **Multicollinearity:** VIF analysis identified correlated features that may benefit from dimensionality reduction in future work.
5. **Limitations:** The linear decision boundary may not capture nonlinear biological interactions inherent in MS pathophysiology.

**Next Steps:** Compare against ensemble methods and explore nonlinear models for improved classification performance."""))

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

output_path = os.path.join(os.path.dirname(__file__), 'Logistic_Regression.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generated: {output_path}")
print(f"Total cells: {len(cells)}")
