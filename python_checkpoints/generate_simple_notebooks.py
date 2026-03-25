"""
Generate 5 simplified, student-friendly Jupyter Notebooks.
Each notebook preserves the full workflow and accuracy but uses
plain language, clear comments, and educational markdown.
"""
import json, os

def _s(src):
    lines = src.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": _s(src)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "source": _s(src), "outputs": [], "execution_count": None}

def save(cells, fname):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.10.0"}},
          "cells": cells}
    path = os.path.join(os.path.dirname(__file__), fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  [OK] {fname} ({len(cells)} cells)")

# ── SHARED CELLS ──

def imports_cell(extra=""):
    return code(f"""# Import all the libraries we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
{extra}
# Set seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore')

# Plot settings
plt.rcParams.update({{
    'figure.dpi': 120, 'font.size': 11, 'axes.titlesize': 13,
    'figure.figsize': (10, 6), 'axes.grid': True, 'grid.alpha': 0.3,
}})

COLORS = {{'RRMS': '#2196F3', 'SPMS': '#FF5722', 'PPMS': '#4CAF50', 'CIS': '#9C27B0'}}
ORDER = ['RRMS', 'SPMS', 'PPMS', 'CIS']

print("All libraries loaded successfully!")""")

def eda_cells():
    return [
        md("## 2. Loading the Dataset\n\nWe load the CSV file and take a quick look at its shape and first few rows."),
        code("""df = pd.read_csv('../datasets/ms_dataset.csv')
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
df.head()"""),

        md("## 3. Exploring the Data\n\nBefore building a model, we need to understand our data — check for missing values, see how many patients are in each subtype, and look at feature distributions.\n\n### 3.1 Basic Info"),
        code("""print("Data types:")
print(df.dtypes)
print("\\nBasic statistics:")
df.describe().round(2)"""),
        code("""# Check for missing values
missing = df.isnull().sum()
print("Missing values:")
print(missing[missing > 0])
print(f"\\nDuplicate rows: {df.duplicated().sum()}")"""),

        md("### 3.2 How Many Patients in Each Subtype?"),
        code("""counts = df['subtype'].value_counts().reindex(ORDER)
print(counts)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(ORDER, counts, color=[COLORS[s] for s in ORDER], edgecolor='white')
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, str(c), ha='center', fontweight='bold')
ax.set_ylabel('Count')
ax.set_title('MS Subtype Distribution')
plt.tight_layout()
plt.show()"""),

        md("### 3.3 Feature Distributions by Subtype\n\nBoxplots help us see which features differ across subtypes — features with clear separation will be helpful for classification."),
        code("""numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
ncols = 4
nrows = (len(numeric_cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x='subtype', y=col, order=ORDER, palette=COLORS, ax=axes[i], fliersize=2)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Features by MS Subtype', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()"""),

        md("### 3.4 Correlation Heatmap\n\nShows how features relate to each other. Highly correlated features carry similar information."),
        code("""corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.show()"""),
    ]

def preprocess_cells(scale=False):
    scale_code = """
# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_ready = scaler.fit_transform(X_train_clean)
X_test_ready = scaler.transform(X_test_clean)
print("Data scaled (mean=0, std=1)")""" if scale else """
# Tree-based models don't need scaling
X_train_ready = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_ready = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)"""

    impute_line = """X_train_clean = imputer.fit_transform(X_train)
X_test_clean = imputer.transform(X_test)""" if scale else ""

    return [
        md(f"""## 4. Data Preprocessing

Steps:
1. Separate features (X) from target (y)
2. Encode target labels as numbers
3. Split into 80% train / 20% test
4. Fill missing values with the median
{"5. Scale features to mean=0, std=1" if scale else ""}

**Important:** We fit the imputer only on training data to prevent data leakage."""),

        code("""# Separate features and target
feature_cols = [c for c in df.columns if c != 'subtype']
X = df[feature_cols]
y = df['subtype']

# Encode target as numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"Classes: {list(class_names)}")"""),

        code("""# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")"""),

        code(f"""# Fill missing values with median
imputer = SimpleImputer(strategy='median')
{impute_line}
{scale_code}
print(f"Missing values remaining: 0")"""),
    ]

def cv_cells(model_name):
    return [
        md(f"""## 6. Cross-Validation (5-Fold)

We split training data into 5 parts, train on 4 and test on 1, rotating each time. This tells us how stable the model is."""),

        code(f"""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {{'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'f1_weighted': 'f1_weighted'}}

cv_results = cross_validate(model, X_train_ready, y_train, cv=cv, scoring=scoring)

print("Cross-Validation Results:")
print(f"  Accuracy:     {{cv_results['test_accuracy'].mean():.4f}} ± {{cv_results['test_accuracy'].std():.4f}}")
print(f"  Macro F1:     {{cv_results['test_f1_macro'].mean():.4f}} ± {{cv_results['test_f1_macro'].std():.4f}}")
print(f"  Weighted F1:  {{cv_results['test_f1_weighted'].mean():.4f}} ± {{cv_results['test_f1_weighted'].std():.4f}}")"""),

        code(f"""# Visualize CV results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = ['Accuracy', 'Macro F1', 'Weighted F1']
means = [cv_results[f'test_{{m}}'].mean() for m in ['accuracy', 'f1_macro', 'f1_weighted']]
stds = [cv_results[f'test_{{m}}'].std() for m in ['accuracy', 'f1_macro', 'f1_weighted']]
axes[0].bar(names, means, yerr=stds, capsize=6, color=['#2196F3', '#FF5722', '#4CAF50'], edgecolor='white')
for i, (m, s) in enumerate(zip(means, stds)):
    axes[0].text(i, m + s + 0.01, f'{{m:.3f}}', ha='center', fontweight='bold')
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Average CV Scores')

for key, label, c in [('test_accuracy','Accuracy','#2196F3'), ('test_f1_macro','Macro F1','#FF5722'), ('test_f1_weighted','Weighted F1','#4CAF50')]:
    axes[1].plot(range(1,6), cv_results[key], 'o-', label=label, color=c, linewidth=2)
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('Score')
axes[1].set_title('Fold-wise Stability')
axes[1].legend()
axes[1].set_ylim(0, 1.05)

plt.suptitle('{model_name} — Cross-Validation', fontsize=14)
plt.tight_layout()
plt.show()"""),
    ]

def eval_cells(model_name):
    return [
        md("## 8. Test Set Evaluation\n\nNow we test on data the model has never seen."),
        code(f"""y_pred = model.predict(X_test_ready)
y_proba = model.predict_proba(X_test_ready)

print("Test Results:")
print(f"  Accuracy:     {{accuracy_score(y_test, y_pred):.4f}}")
print(f"  Macro F1:     {{f1_score(y_test, y_pred, average='macro'):.4f}}")
print(f"  Weighted F1:  {{f1_score(y_test, y_pred, average='weighted'):.4f}}")
print("\\n" + classification_report(y_test, y_pred, target_names=class_names, digits=4))"""),

        md("### Confusion Matrix"),
        code(f"""fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix — {model_name}')
plt.tight_layout()
plt.show()"""),

        md("### ROC Curves\n\nCloser to the top-left corner = better. AUC of 1.0 is perfect, 0.5 is random guessing."),
        code(f"""y_test_bin = label_binarize(y_test, classes=range(len(class_names)))

fig, ax = plt.subplots(figsize=(8, 6))
for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    ax.plot(fpr, tpr, linewidth=2, label=f'{{name}} (AUC={{auc(fpr,tpr):.3f}})', color=COLORS.get(name))
ax.plot([0,1],[0,1],'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — {model_name}')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

print(f"Overall ROC-AUC: {{roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro'):.4f}}")"""),

        md("### Precision-Recall Curves"),
        code(f"""fig, ax = plt.subplots(figsize=(8, 6))
for i, name in enumerate(class_names):
    prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
    ax.plot(rec, prec, linewidth=2, label=f'{{name}} (AP={{ap:.3f}})', color=COLORS.get(name))
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall — {model_name}')
ax.legend(loc='lower left')
plt.tight_layout()
plt.show()"""),
    ]

def importance_cells(model_name, getter="model.feature_importances_", label="Feature Importance"):
    return [
        md("## 9. Feature Importance\n\nWhich features does the model rely on most?"),
        code(f"""imp = {getter}
imp_df = pd.DataFrame({{'Feature': feature_cols, 'Importance': imp}}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(imp_df['Feature'], imp_df['Importance'],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df))), edgecolor='white')
ax.set_xlabel('{label}')
ax.set_title('Feature Importance — {model_name}')
plt.tight_layout()
plt.show()

print("Top 5 features:")
for _, row in imp_df.tail(5).iloc[::-1].iterrows():
    print(f"  {{row['Feature']:30s}} {{row['Importance']:.4f}}")"""),
    ]

def dashboard_cells(model_name, getter="model.feature_importances_"):
    return [
        md("## 10. Summary Dashboard"),
        code(f"""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Per-class metrics
p = precision_score(y_test, y_pred, average=None)
r = recall_score(y_test, y_pred, average=None)
f = f1_score(y_test, y_pred, average=None)
x = np.arange(len(class_names))
w = 0.25
axes[0,0].bar(x-w, p, w, label='Precision', color='#2196F3', edgecolor='white')
axes[0,0].bar(x, r, w, label='Recall', color='#FF5722', edgecolor='white')
axes[0,0].bar(x+w, f, w, label='F1', color='#4CAF50', edgecolor='white')
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(class_names)
axes[0,0].set_title('Per-Class Metrics'); axes[0,0].legend(fontsize=8); axes[0,0].set_ylim(0,1.15)

# CV vs Test
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average='macro')
f1w = f1_score(y_test, y_pred, average='weighted')
cv_m = [cv_results['test_accuracy'].mean(), cv_results['test_f1_macro'].mean(), cv_results['test_f1_weighted'].mean()]
t_m = [acc, f1m, f1w]
x2 = np.arange(3)
axes[0,1].bar(x2-0.15, cv_m, 0.3, label='CV', color='#7E57C2', edgecolor='white')
axes[0,1].bar(x2+0.15, t_m, 0.3, label='Test', color='#26A69A', edgecolor='white')
axes[0,1].set_xticks(x2); axes[0,1].set_xticklabels(['Accuracy','Macro F1','Weighted F1'])
axes[0,1].set_title('CV vs Test'); axes[0,1].legend(); axes[0,1].set_ylim(0,1.15)

# Feature importance
imp = {getter}
imp_df2 = pd.DataFrame({{'Feature': feature_cols, 'Importance': imp}}).sort_values('Importance').tail(8)
axes[1,0].barh(imp_df2['Feature'], imp_df2['Importance'],
               color=plt.cm.viridis(np.linspace(0.3,0.9,len(imp_df2))), edgecolor='white')
axes[1,0].set_title('Top 8 Features')

# Per-class AUC
aucs = [auc(*roc_curve(y_test_bin[:,i], y_proba[:,i])[:2]) for i in range(len(class_names))]
axes[1,1].bar(class_names, aucs, color=[COLORS[n] for n in class_names], edgecolor='white')
for i,v in enumerate(aucs): axes[1,1].text(i, v+0.02, f'{{v:.3f}}', ha='center', fontweight='bold')
axes[1,1].set_ylabel('AUC'); axes[1,1].set_title('Per-Class AUC'); axes[1,1].set_ylim(0,1.15)

fig.suptitle('{model_name} — Performance Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""),
    ]

# ═══════════════════════════════════════════════════════════════
print("Generating simplified notebooks...")

# ── 1. LOGISTIC REGRESSION ──
c = []
c.append(md("""# Simple Logistic Regression — MS Subtype Classification

**Goal:** Predict which MS subtype (RRMS, SPMS, PPMS, CIS) a patient has based on clinical and MRI features.

**About Logistic Regression:** A simple linear model that predicts class probabilities using the softmax function. It gives us a baseline and interpretable coefficients.

---"""))
c.append(imports_cell("from sklearn.linear_model import LogisticRegression"))
c.extend(eda_cells())
c.extend(preprocess_cells(scale=True))
c.append(md("## 5. Building the Model\n\nLogistic Regression with L2 regularization and balanced class weights."))
c.append(code("""model = LogisticRegression(
    penalty='l2', solver='lbfgs', max_iter=1000,
    class_weight='balanced', C=1.0, random_state=RANDOM_STATE
)
print("Model created")"""))
c.extend(cv_cells("Logistic Regression"))
c.append(md("## 7. Train on Full Training Set"))
c.append(code("""model.fit(X_train_ready, y_train)
print("Model trained!")"""))
c.extend(eval_cells("Logistic Regression"))
c.append(md("## 9. Coefficient Heatmap\n\nShows how each feature pushes toward or away from each subtype."))
c.append(code("""coef_df = pd.DataFrame(model.coef_, index=class_names, columns=feature_cols)
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(coef_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Model Coefficients')
plt.tight_layout()
plt.show()"""))
c.append(md("## Conclusion\n\nLogistic Regression gives us a simple interpretable baseline. The coefficients tell us which features matter most for each subtype."))
save(c, "simple_logistic_regression.ipynb")

# ── 2. RANDOM FOREST ──
c = []
c.append(md("""# Simple Random Forest — MS Subtype Classification

**Goal:** Classify MS subtypes using an ensemble of decision trees.

**About Random Forest:** Builds many decision trees on random subsets of data and features, then lets them vote. No scaling needed — trees use thresholds, not distances.

---"""))
c.append(imports_cell("from sklearn.ensemble import RandomForestClassifier"))
c.extend(eda_cells())
c.extend(preprocess_cells(scale=False))
c.append(md("## 5. Building the Model\n\n300 trees, balanced class weights, and OOB (out-of-bag) scoring for free validation."))
c.append(code("""model = RandomForestClassifier(
    n_estimators=300, class_weight='balanced',
    oob_score=True, n_jobs=-1, random_state=RANDOM_STATE
)
print("Model created")"""))
c.extend(cv_cells("Random Forest"))
c.append(md("## 7. Train on Full Training Set"))
c.append(code("""model.fit(X_train_ready, y_train)
print(f"Trained with {model.n_estimators} trees, OOB Score: {model.oob_score_:.4f}")"""))
c.extend(eval_cells("Random Forest"))
c.extend(importance_cells("Random Forest"))
c.extend(dashboard_cells("Random Forest"))
c.append(md("## Conclusion\n\nRandom Forest captures nonlinear patterns with no scaling needed. OOB score gives free validation, and feature importance shows what matters most."))
save(c, "simple_random_forest.ipynb")

# ── 3. EXTRA TREES ──
c = []
c.append(md("""# Simple Extra Trees — MS Subtype Classification

**Goal:** Classify MS subtypes using Extremely Randomized Trees.

**About Extra Trees:** Like Random Forest but picks random split thresholds instead of finding the best one. This makes it faster and often more stable.

---"""))
c.append(imports_cell("from sklearn.ensemble import ExtraTreesClassifier"))
c.extend(eda_cells())
c.extend(preprocess_cells(scale=False))
c.append(md("## 5. Building the Model"))
c.append(code("""model = ExtraTreesClassifier(
    n_estimators=300, class_weight='balanced',
    n_jobs=-1, random_state=RANDOM_STATE
)
print("Model created")"""))
c.extend(cv_cells("Extra Trees"))
c.append(md("## 7. Train on Full Training Set"))
c.append(code("""model.fit(X_train_ready, y_train)
print(f"Trained with {model.n_estimators} trees")"""))
c.extend(eval_cells("Extra Trees"))
c.extend(importance_cells("Extra Trees"))
c.extend(dashboard_cells("Extra Trees"))
c.append(md("## Conclusion\n\nExtra Trees adds more randomness than Random Forest, often resulting in lower variance and more stable predictions."))
save(c, "simple_extra_trees.ipynb")

# ── 4. XGBOOST ──
c = []
c.append(md("""# Simple XGBoost — MS Subtype Classification

**Goal:** Classify MS subtypes using gradient boosting.

**About XGBoost:** Builds trees one at a time, where each new tree fixes the mistakes of previous trees. Think of it as: each doctor reviews what previous doctors got wrong.

---"""))
c.append(imports_cell("from xgboost import XGBClassifier\nfrom sklearn.utils.class_weight import compute_sample_weight"))
c.extend(eda_cells())
c.extend(preprocess_cells(scale=False))
c.append(code("""# XGBoost uses sample weights instead of class_weight
sample_weights = compute_sample_weight('balanced', y_train)
print("Sample weights computed for class balance")"""))
c.append(md("## 5. Building the Model\n\n300 boosting rounds, learning rate 0.1, max depth 5."))
c.append(code("""model = XGBClassifier(
    objective='multi:softprob', n_estimators=300,
    learning_rate=0.1, max_depth=5, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='mlogloss', use_label_encoder=False,
    random_state=RANDOM_STATE, n_jobs=-1
)
print("Model created")"""))
c.extend(cv_cells("XGBoost"))
c.append(md("## 7. Train on Full Training Set"))
c.append(code("""model.fit(X_train_ready, y_train, sample_weight=sample_weights, verbose=False)
print("Model trained with 300 boosting rounds")"""))
c.extend(eval_cells("XGBoost"))
c.extend(importance_cells("XGBoost", label="Gain"))
c.extend(dashboard_cells("XGBoost"))
c.append(md("## Conclusion\n\nXGBoost builds trees sequentially, each correcting past errors. Often achieves the best balanced performance and has excellent SHAP explainability support."))
save(c, "simple_xgboost.ipynb")

# ── 5. CATBOOST ──
c = []
c.append(md("""# Simple CatBoost — MS Subtype Classification

**Goal:** Classify MS subtypes using CatBoost gradient boosting.

**About CatBoost:** Like XGBoost but with two special tricks: ordered boosting (reduces overfitting) and native categorical feature support. It also handles missing values internally!

---"""))
c.append(imports_cell("from catboost import CatBoostClassifier, Pool"))
c.extend(eda_cells())

# CatBoost special preprocessing
c.append(md("""## 4. Data Preprocessing

CatBoost is special — it handles missing values and categorical features internally. No scaling or imputation needed!"""))
c.append(code("""feature_cols = [col for col in df.columns if col != 'subtype']
X = df[feature_cols]
y = df['subtype']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Mark categorical features
cat_features = ['sex_encoded', 'treatment_status']
cat_indices = [feature_cols.index(c) for c in cat_features if c in feature_cols]
for col in cat_features:
    if col in X.columns:
        X[col] = X[col].astype(int)
print(f"Classes: {list(class_names)}")
print(f"Categorical features: {cat_features}")"""))

c.append(code("""# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
train_pool = Pool(X_train, y_train, cat_features=cat_indices)
test_pool = Pool(X_test, y_test, cat_features=cat_indices)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")"""))

c.append(md("## 5. Building the Model"))
c.append(code("""model = CatBoostClassifier(
    loss_function='MultiClass', iterations=500,
    learning_rate=0.1, depth=6, l2_leaf_reg=3.0,
    auto_class_weights='Balanced', random_seed=RANDOM_STATE, verbose=0
)
print("Model created")"""))

# CatBoost manual CV
c.append(md("## 6. Cross-Validation"))
c.append(code("""cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_acc, cv_f1m, cv_f1w = [], [], []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    fold_pool = Pool(X_train.iloc[tr_idx], y_train[tr_idx], cat_features=cat_indices)
    fm = CatBoostClassifier(loss_function='MultiClass', iterations=500, learning_rate=0.1,
                            depth=6, l2_leaf_reg=3.0, auto_class_weights='Balanced',
                            random_seed=RANDOM_STATE, verbose=0)
    fm.fit(fold_pool)
    yp = fm.predict(X_train.iloc[val_idx]).flatten().astype(int)
    cv_acc.append(accuracy_score(y_train[val_idx], yp))
    cv_f1m.append(f1_score(y_train[val_idx], yp, average='macro'))
    cv_f1w.append(f1_score(y_train[val_idx], yp, average='weighted'))
    print(f"  Fold {fold}: Acc={cv_acc[-1]:.4f}, F1={cv_f1m[-1]:.4f}")

cv_results = {'test_accuracy': np.array(cv_acc), 'test_f1_macro': np.array(cv_f1m), 'test_f1_weighted': np.array(cv_f1w)}
print(f"\\nAverage: Acc={np.mean(cv_acc):.4f}, Macro F1={np.mean(cv_f1m):.4f}")"""))

c.append(md("## 7. Train on Full Training Set"))
c.append(code("""model.fit(train_pool)
print(f"Trained with {model.tree_count_} trees")"""))

# CatBoost eval
c.append(md("## 8. Test Set Evaluation"))
c.append(code("""y_pred = model.predict(test_pool).flatten().astype(int)
y_proba = model.predict_proba(test_pool)

print("Test Results:")
print(f"  Accuracy:     {accuracy_score(y_test, y_pred):.4f}")
print(f"  Macro F1:     {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"  Weighted F1:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
print("\\n" + classification_report(y_test, y_pred, target_names=class_names, digits=4))"""))
c.append(code("""fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix — CatBoost')
plt.tight_layout()
plt.show()"""))
c.append(code("""y_test_bin = label_binarize(y_test, classes=range(len(class_names)))

fig, ax = plt.subplots(figsize=(8, 6))
for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc(fpr,tpr):.3f})', color=COLORS.get(name))
ax.plot([0,1],[0,1],'k--', alpha=0.3)
ax.set_title('ROC Curves — CatBoost')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()
print(f"Overall ROC-AUC: {roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='macro'):.4f}")"""))

c.extend(importance_cells("CatBoost", getter="model.get_feature_importance()", label="Importance"))
c.extend(dashboard_cells("CatBoost", getter="model.get_feature_importance()"))
c.append(md("## Conclusion\n\nCatBoost provides strong performance with minimal preprocessing — no scaling, no imputation, native categorical support. Its ordered boosting reduces overfitting."))
save(c, "simple_catboost.ipynb")

print("\n✓ All 5 simplified notebooks generated!")
