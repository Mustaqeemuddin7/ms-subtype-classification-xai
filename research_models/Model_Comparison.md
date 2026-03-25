# Model Comparison — MS Subtype Classification
## Augmented Dataset (565 samples, PPMS: 60 → 125)

---

## Test Set Performance Summary

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) |
|-------|----------|----------|-------------|----------------|
| Logistic Regression | 0.8407 | 0.8028 | 0.8480 | 0.9796 |
| Random Forest | 0.8938 | 0.8560 | 0.8935 | 0.9805 |
| **Extra Trees** | **0.9115** | **0.8712** | **0.9090** | 0.9800 |
| XGBoost | 0.9027 | 0.8705 | 0.9040 | **0.9833** |
| CatBoost | 0.8850 | 0.8527 | 0.8860 | 0.9815 |

> [!IMPORTANT]
> **Extra Trees leads on Accuracy (0.9115) and Macro F1 (0.8712)**, closely followed by XGBoost (0.8705). Augmenting PPMS improved all models significantly.

---

## Cross-Validation Performance (5-Fold Stratified)

| Model | CV Accuracy | CV Macro F1 | CV Weighted F1 |
|-------|-------------|-------------|----------------|
| Logistic Regression | 0.7965 ± 0.0113 | 0.7533 ± 0.0108 | 0.8062 ± 0.0093 |
| Random Forest | 0.8628 ± 0.0242 | 0.7725 ± 0.0599 | 0.8530 ± 0.0272 |
| Extra Trees | 0.8607 ± 0.0235 | 0.7837 ± 0.0272 | 0.8531 ± 0.0230 |
| XGBoost | **0.8760 ± 0.0181** | 0.8206 ± 0.0361 | **0.8728 ± 0.0200** |
| CatBoost | 0.8805 ± 0.0082 | **0.8372 ± 0.0263** | 0.8807 ± 0.0083 |

> [!NOTE]
> **CatBoost has the lowest CV variance** (σ=0.008), making it the most stable model. Logistic Regression also shows excellent stability (σ=0.011).

---

## Per-Class F1 Scores (Test Set)

| Model | CIS | PPMS | RRMS | SPMS |
|-------|-----|------|------|------|
| Logistic Regression | 0.6087 | **0.8889** | 0.8627 | 0.8511 |
| Random Forest | 0.7500 | 0.8571 | 0.9369 | 0.8800 |
| Extra Trees | **0.7692** | 0.8800 | **0.9558** | 0.8800 |
| XGBoost | 0.7778 | 0.8846 | 0.9444 | 0.8750 |
| CatBoost | 0.7778 | 0.8462 | 0.9358 | 0.8511 |

### Impact of PPMS Augmentation

| Model | PPMS F1 (Before) | PPMS F1 (After) | Improvement |
|-------|-------------------|-----------------|-------------|
| Logistic Regression | 0.6400 | **0.8889** | +0.2489 ✨ |
| Random Forest | 0.5556 | **0.8571** | +0.3015 ✨ |
| Extra Trees | 0.5000 | **0.8800** | +0.3800 ✨ |
| XGBoost | 0.7368 | **0.8846** | +0.1478 |
| CatBoost | 0.6000 | **0.8462** | +0.2462 ✨ |

> [!TIP]
> **PPMS F1 improved dramatically across all models** — from 0.50–0.74 to 0.85–0.89. Extra Trees saw the largest improvement (+0.38). Augmenting rare classes directly improves clinical classification.

---

## Ranking Summary

| Metric | 🥇 Best | 🥈 Second | 🥉 Third |
|--------|---------|-----------|----------|
| **Accuracy** | Extra Trees (0.9115) | XGBoost (0.9027) | RF (0.8938) |
| **Macro F1** | Extra Trees (0.8712) | XGBoost (0.8705) | RF (0.8560) |
| **ROC-AUC** | XGBoost (0.9833) | CatBoost (0.9815) | RF (0.9805) |
| **CV Stability** | CatBoost (σ=0.008) | LR (σ=0.011) | XGB (σ=0.018) |
| **PPMS F1** | LR (0.8889) | XGBoost (0.8846) | ET (0.8800) |
| **RRMS F1** | Extra Trees (0.9558) | XGBoost (0.9444) | RF (0.9369) |

---

## Recommendations

1. **Best overall: Extra Trees** — Highest accuracy (0.9115) and Macro F1 (0.8712) with balanced per-class performance.

2. **Best for explainability (SHAP/DiCE): XGBoost** — Near-best performance (Macro F1 = 0.8705) with superior SHAP TreeExplainer support.

3. **Most stable: CatBoost** — Lowest CV variance (σ=0.008), most consistent across data splits.

4. **PPMS no longer the bottleneck** — All models now achieve F1 > 0.84 on PPMS after augmentation.

5. **CIS remains the hardest class** — Smallest sample (n=40), F1 ranges 0.61–0.78.
