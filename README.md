# Model 06: Ensemble Model (RF + GradientBoosting + XGBoost + LightGBM)

## Table of Contents

1. [Background](#background)
2. [Overview](#overview)
3. [Motivation & Design Rationale](#motivation--design-rationale)
4. [Architecture](#architecture)
   - [Ensemble Composition](#ensemble-composition)
   - [Voting Strategy](#voting-strategy)
5. [Feature Engineering](#feature-engineering)
   - [Sequence Features](#sequence-features)
   - [Chemistry Features](#chemistry-features)
   - [Context Features](#context-features)
   - [Thermodynamic Features](#thermodynamic-features)
   - [Enhanced (Biological) Features](#enhanced-biological-features)
   - [Experimental Metadata](#experimental-metadata)
   - [Complete Feature Summary](#complete-feature-summary)
6. [Training Pipeline](#training-pipeline)
   - [Data Loading & Splitting](#data-loading--splitting)
   - [Feature Extraction](#feature-extraction)
   - [Feature Scaling](#feature-scaling)
   - [Individual Model Training](#individual-model-training)
   - [Ensemble Construction](#ensemble-construction)
7. [Evaluation Protocol](#evaluation-protocol)
   - [Data Split](#data-split-51)
   - [Spearman Rank Correlation](#spearman-rank-correlation--mean_spearman_corr-52)
   - [Enrichment Factor](#enrichment-factor--top_pred_target_ratio_median-53)
   - [Required Output Format](#required-output-format--metricsjson-7)
8. [Results & Analysis](#results--analysis)
   - [Comparison with OligoAI Baseline](#comparison-with-oligoai-baseline-73)
9. [Reproduction Instructions](#reproduction-instructions)
   - [Using run.sh / Docker](#using-runsh--docker-74)
10. [File Reference](#file-reference)
11. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Background

Antisense oligonucleotides (ASOs) are short, synthetic, single-stranded nucleic acids designed to bind complementary RNA sequences and modulate gene expression. In RNase H–mediated ASOs, hybridization forms a DNA–RNA heteroduplex that is cleaved by RNase H, resulting in degradation of the target RNA. Therapeutic ASOs commonly employ **gapmer designs** with:

- A central DNA gap required for RNase H activity
- Chemically modified flanking regions (e.g., 2'-MOE, cEt) for stability
- Backbone modifications (e.g., phosphorothioate/PS) for pharmacokinetics

Predicting ASO efficacy remains challenging due to interactions between sequence, target RNA context, chemical modifications, and experimental conditions. This project uses the same ~180K ASO dataset and evaluation methodology as **OligoAI** (Spearman ρ ≈ 0.42, enrichment ≈ 3×) to explore whether alternative architectures can achieve better or complementary performance.

---

## Overview

Model 06 is a **classical machine learning ensemble** that combines four gradient-based and tree-based regressors into a voting ensemble. Unlike the deep learning approaches (Models 03 and 05), this model relies entirely on **handcrafted features** extracted from ASO sequences, chemical modifications, target RNA context, and experimental metadata — no pretrained language model is involved.

| Property | Value |
|----------|-------|
| **Task** | Regression — predict ASO inhibition (%) |
| **Primary metric** | Spearman rank correlation (ρ) |
| **Secondary metric** | Enrichment factor (top-10% hit rate) |
| **Framework** | scikit-learn, XGBoost, LightGBM |
| **Ensemble type** | VotingRegressor (simple average) |
| **Number of features** | 119 |
| **Training script** | `models/06_ensemble/train.py` |

---

## Motivation & Design Rationale

The project requirements explicitly suggest exploring an **Ensemble Voting (RF + Gradient Boosting + XGBoost)** approach as an alternative to deep learning. The key motivations are:

1. **Complementary to deep learning**: Tree-based models capture different patterns than neural networks — they excel at tabular data with heterogeneous feature types and can model complex non-linear interactions without requiring GPU training.

2. **Interpretable feature importance**: Random Forests and gradient boosters provide built-in feature importance rankings, enabling biological insight into which features drive efficacy.

3. **Fast training**: The entire ensemble trains in minutes on CPU, compared to hours/days for the deep learning models. This enables rapid iteration and debugging.

4. **Robustness**: Ensemble methods reduce variance through averaging, and mixed model types (bagging via RF + boosting via GB/XGB/LGB) provide diversity in the ensemble.

5. **No pretrained model dependency**: No need for RiNALMo weights, GPU, or external tools like ViennaRNA — only the tabular features suffice.

---

## Architecture

### Ensemble Composition

```
┌──────────────────────────────────────────────────────────────┐
│                                                                │
│  Raw Data (180K ASO records)                                   │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────┐                                      │
│  │  Feature Engineering  │  → 119 numerical features            │
│  │  (handcrafted)        │                                      │
│  └──────────┬───────────┘                                      │
│             │                                                  │
│       ┌─────▼─────┐                                            │
│       │ Std Scaler │                                            │
│       └─────┬─────┘                                            │
│             │                                                  │
│    ┌────────┼────────┬────────────┐                             │
│    ▼        ▼        ▼            ▼                             │
│ ┌──────┐ ┌──────┐ ┌───────┐ ┌──────────┐                      │
│ │  RF  │ │  GB  │ │  XGB  │ │  LightGBM │                      │
│ │ 200  │ │ 200  │ │  200  │ │   200     │                      │
│ │trees │ │trees │ │ trees │ │  trees    │                      │
│ └──┬───┘ └──┬───┘ └──┬────┘ └──┬───────┘                      │
│    │        │        │          │                               │
│    └────────┼────────┼──────────┘                               │
│             ▼                                                  │
│    ┌─────────────────┐                                         │
│    │ VotingRegressor  │  (simple average of all predictions)    │
│    └────────┬────────┘                                         │
│             │                                                  │
│             ▼                                                  │
│    Predicted Inhibition (%)                                    │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### Individual Models

#### Random Forest (RF)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sufficient trees for convergence |
| `max_depth` | 15 | Moderate depth to prevent overfitting |
| `min_samples_split` | 10 | Regularization |
| `min_samples_leaf` | 5 | Regularization |
| `n_jobs` | -1 (all cores) | Maximize CPU utilization |

Random Forest is a **bagging** ensemble: each tree sees a random bootstrap sample of the data and a random subset of features at each split. This provides strong variance reduction.

#### Gradient Boosting (GB)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sequential boosting rounds |
| `max_depth` | 6 | Shallower trees for boosting |
| `learning_rate` | 0.1 | Standard shrinkage |
| `subsample` | 0.8 | Stochastic gradient boosting |

Gradient Boosting builds trees sequentially, each correcting errors from the previous ensemble. This is a **boosting** approach that reduces bias.

#### XGBoost (XGB)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Boosting rounds |
| `max_depth` | 6 | Matched to GB |
| `learning_rate` | 0.1 | Standard shrinkage |
| `subsample` | 0.8 | Row subsampling |
| `colsample_bytree` | 0.8 | Column subsampling |
| `tree_method` | `hist` | Histogram-based splitting (fast) |
| `n_jobs` | -1 | All CPU cores |

XGBoost adds level-2 regularization (L1/L2) and parallel tree construction on top of gradient boosting.

#### LightGBM (LGB)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Boosting rounds |
| `max_depth` | 6 | Matched to GB |
| `learning_rate` | 0.1 | Standard shrinkage |
| `subsample` | 0.8 | Row subsampling |
| `colsample_bytree` | 0.8 | Column subsampling |
| `n_jobs` | -1 | All CPU cores |

LightGBM uses leaf-wise growth (vs. level-wise) and GOSS (Gradient-based One-Side Sampling) for faster convergence.

### Voting Strategy

The final ensemble uses scikit-learn's `VotingRegressor` with **simple averaging** (equal weights):

$$
\hat{y}_{\text{ensemble}} = \frac{1}{4}(\hat{y}_{\text{RF}} + \hat{y}_{\text{GB}} + \hat{y}_{\text{XGB}} + \hat{y}_{\text{LGB}})
$$

---

## Feature Engineering

The model extracts **119 numerical features** organized into six categories. All feature extraction is performed in a single pass per sample via the `extract_all_features()` function.

### Sequence Features

Extracted from the ASO sequence (`aso_sequence_5_to_3`) by `extract_sequence_features()`:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| `length` | 1 | ASO sequence length |
| `frac_{A,T,G,C}` | 4 | Nucleotide composition fractions |
| `gc_content` | 1 | (G + C) / length |
| `mw` | 1 | Approximate molecular weight (sum of nucleotide MWs) |
| `di_{XX}` | 16 | All 16 dinucleotide frequencies (count / (length - 1)) |
| `bad_{motif}` | 5 | Individual bad motif counts (GGGG, AAAA, TAAA, CTAA, CCTA) |
| `good_{motif}` | 5 | Individual good motif counts (TTGT, GTAT, CGTA, GTCG, GCGT) |
| `total_bad_motifs` | 1 | Sum of all bad motif occurrences |
| `total_good_motifs` | 1 | Sum of all good motif occurrences |
| `pos_{i}_{nt}` | ~5 | One-hot indicators for first 5 nucleotide positions |
| `end_{i}_{nt}` | ~5 | One-hot indicators for last 5 nucleotide positions |

**Nucleotide molecular weights** (Da):
| Nucleotide | MW |
|------------|-----|
| A | 331.2 |
| T | 322.2 |
| G | 347.2 |
| C | 307.2 |
| U | 308.2 |

### Chemistry Features

Extracted from `sugar_mods` and `backbone_mods` columns by `extract_chemistry_features()`:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| **Sugar counts** | 4 | Count of DNA, MOE, cEt, LNA modifications |
| **Sugar fractions** | 4 | Fraction of each sugar type |
| **Backbone counts** | 2 | Count of PS and PO bonds |
| **Backbone fraction** | 1 | Fraction of PS bonds |
| **Gapmer detection** | 2 | `is_gapmer` (binary), `gap_length` |
| **Position sugar type** | 10 | Sugar type index at first 5 / last 5 positions |
| **Position backbone** | 10 | PS indicator at first 5 / last 5 positions |
| **Wing/gap transitions** | 2 | `wing5_to_gap_transition`, `wing3_modified` |

**Gapmer detection logic**: A sequence is classified as a gapmer if:
- Positions 0–2 (5' wing) contain non-DNA modifications (MOE, cEt, or LNA)
- Positions 3 to (n-3) (gap) contain DNA
- Last 3 positions (3' wing) contain non-DNA modifications

**Sugar type encoding**: DNA=0, MOE=1, cEt=2, LNA=3, Other=4

### Context Features

Extracted from the target RNA context by `extract_context_features()`:

| Feature | Description |
|---------|-------------|
| `ctx_length` | Length of RNA context |
| `ctx_gc` | GC content of context |
| `ctx_mirna_sites` | Count of miRNA target motifs found |
| `ctx_max_purine_run` | Longest consecutive purine run (AG) |
| `ctx_max_pyrimidine_run` | Longest consecutive pyrimidine run (TC) |

Purine/pyrimidine run lengths serve as a **proxy for secondary structure**: long runs of purines tend to be unstructured (accessible), while alternating purine/pyrimidine runs suggest base pairing.

### Thermodynamic Features

Computed using the nearest-neighbor model from `utils/thermo_features.py`:

| Feature | Description |
|---------|-------------|
| `delta_g` | Gibbs free energy of ASO-mRNA duplex (kcal/mol) |
| `delta_g_per_nt` | ΔG normalized by ASO length |

**Note**: These features are only available if the `utils.thermo_features` module can be imported. If not, they are silently omitted.

### Enhanced (Biological) Features

Extracted using helper functions from `utils/enhanced_dataset.py`:

| Feature | Count | Description |
|---------|-------|-------------|
| `region_{0..5}` | 6 | One-hot encoded genomic region (exon/intron/5'UTR/3'UTR/splice/unknown) |
| `mirna_overlap` | 1 | Binary: miRNA binding site found in context |
| `structure_accessibility` | 1 | Approximate: `1 - GC_content` of context |

### Experimental Metadata

| Feature | Description |
|---------|-------------|
| `log_dosage` | ln(1 + dosage_nM), log-transformed ASO dosage |
| `method_gymnosis` | Binary: free uptake / gymnosis |
| `method_lipo` | Binary: lipofection |
| `method_electro` | Binary: electroporation |

### Complete Feature Summary

| Category | Feature Count | Extraction Function |
|----------|--------------|---------------------|
| Sequence | ~45 | `extract_sequence_features()` |
| Chemistry | ~35 | `extract_chemistry_features()` |
| Context | ~5 | `extract_context_features()` |
| Thermodynamic | 2 | `compute_delta_g_nn()` (optional) |
| Biological | 8 | `parse_genomic_region()`, `check_mirna_overlap()` |
| Metadata | 4 | Direct column extraction |
| **Total** | **~119** | `extract_all_features()` |

Note: Exact count varies slightly because positional one-hot features (`pos_{i}_{nt}`, `end_{i}_{nt}`) are sparse and depend on what nucleotides appear at those positions across the dataset.

---

## Training Pipeline

### Data Loading & Splitting

1. **Data source**: `data/raw/aso_inhibitions_21_08_25_incl_context_w_flank_50_df.csv`
2. **Split**: Patent-level split via `create_patent_split(df, random_state=42)`
   - 80% train / 10% validation / 10% test
   - Patent ID extracted from `custom_id` (text before `_table_`)
   - Identical split to all other models in the pipeline

### Feature Extraction

Feature extraction is performed by `create_feature_matrix()` which:

1. Iterates over every row in the DataFrame
2. Calls `extract_all_features(row)` to compute all feature values
3. Converts the list of dictionaries to a NumPy matrix
4. Fills NaN values with 0
5. Returns the feature matrix `X` and feature name list

This step takes approximately 2-5 minutes for 180K samples on a modern CPU.

### Feature Scaling

All features are standardized using `sklearn.preprocessing.StandardScaler`:

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

- `fit_transform()` on training data
- `transform()` on validation and test data (no data leakage)

### Individual Model Training

Each model is trained independently on the full training set:

```python
# Training order:
1. Random Forest       → rf.fit(X_train, y_train)
2. Gradient Boosting   → gb.fit(X_train, y_train)
3. XGBoost             → xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
4. LightGBM            → lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

XGBoost and LightGBM use the validation set for internal early stopping / monitoring. Each individual model's validation Spearman is printed for comparison.

### Ensemble Construction

After all models are trained, a `VotingRegressor` is created from the list of fitted estimators and fit on the training data:

```python
ensemble = VotingRegressor(estimators=[
    ('rf', rf), ('gb', gb), ('xgb', xgb_model), ('lgb', lgb_model)
])
ensemble.fit(X_train, y_train)
```

**Graceful degradation**: If XGBoost or LightGBM are not installed, the ensemble is built with whatever models are available (minimum: RF + GB).

---

## Evaluation Protocol

Following the OligoAI methodology (see `docs/project_requirements.txt` §5):

### Data Split (§5.1)

The dataset is split at the **patent level** using `create_patent_split()` from `utils/dataset.py`:
- 80% train / 10% validation / 10% test
- All ASOs originating from the same patent belong to the same split
- Patent ID is extracted from `custom_id` (text before `_table_`)
- Deterministic with `random_state=42`

Random row-level or gene-level splits are not used.

### Spearman Rank Correlation — `mean_spearman_corr` (§5.2)

Per the OligoAI paper, Spearman correlation must be computed as:
1. Group predictions and ground-truth values by **experimental screen** (patent)
2. Compute Spearman ρ within each screen (groups with ≥3 samples)
3. Report the **mean** across all screens

$$
\rho_{\text{screen}} = 1 - \frac{6\sum d_i^2}{n(n^2-1)}
$$

$$
\text{mean-spearman-corr} = \frac{1}{|S|}\sum_{s \in S} \rho_s
$$

> **⚠️ Implementation note**: The `train.py` script computes **global Spearman** (single correlation over the entire test set) for speed. For the official per-screen metric matching the OligoAI definition, use the standalone evaluation script:
> ```bash
> python scripts/evaluate.py --model 06
> ```
> The `evaluate.py` script groups by patent ID and computes mean Spearman across screens, producing results in the required `metrics.json` format.

### Enrichment Factor — `top_pred_target_ratio_median` (§5.3)

Following the OligoAI definition:
1. Rank ASOs by predicted inhibition
2. Select the top 10% predicted ASOs
3. Compute the fraction of these that are also in the top 10% by true inhibition
4. Compute enrichment as:

$$
\text{Enrichment Factor} = \frac{\text{Hit rate in predicted top 10\%}}{0.10}
$$

An enrichment of 1.0× means random performance; higher is better.

### Required Output Format — `metrics.json` (§7)

All metrics are written to `results/06_ensemble/metrics.json` in the required format:

```json
{
  "train": {
    "r2": ..., "mae": ..., "rmse": ...,
    "mean_spearman_corr": ..., "top_pred_target_ratio_median": ...
  },
  "val": { ... },
  "test": { ... }
}
```

Key names match the project deliverable specification:
- `r2` — Coefficient of determination
- `mae` — Mean absolute error
- `rmse` — Root mean squared error
- `mean_spearman_corr` — Mean Spearman correlation (per-screen when using `evaluate.py`)
- `top_pred_target_ratio_median` — Enrichment factor

### Metrics Computed

Metrics are reported for train, validation, and test sets separately.

### Predictions

The model predicts **raw inhibition percentage** (not standardized), making predictions directly interpretable without inverse scaling.

---

## Results & Analysis

### Test Set Performance

| Metric | Model 06 (Ensemble) | Model 05 (Deep) | OligoAI Baseline |
|--------|---------------------|------------------|-----------------|
| **Spearman ρ** | **0.384** | 0.366 | 0.419 |
| **Enrichment** | **2.55×** | 2.61× | ~3× |
| **R²** | **0.158** | 0.157 | — |
| **MAE** | 20.38 | 0.731* | — |
| **RMSE** | 24.84 | 0.888* | — |

*Model 05 values are on standardized (z-score) scale; Model 06 predicts raw inhibition %.

Test set contains **23,372 samples** (patent-level hold-out).

### Per-Split Breakdown

| Split | Samples | R² | MAE | RMSE | Spearman ρ | Enrichment |
|-------|--------:|-----|-----|------|------------|------------|
| **Train** | ~144K | 0.446 | 17.00 | 20.86 | 0.679 | 4.55× |
| **Val** | ~18K | 0.095 | 22.15 | 26.32 | 0.370 | 2.35× |
| **Test** | 23,372 | 0.158 | 20.38 | 24.84 | 0.384 | 2.55× |

### Training Characteristics

Unlike the deep learning Model 05, the ensemble has **no iterative training log** — scikit-learn tree-based models are trained in a single `.fit()` call. Key observations from the metrics:

1. **Train vs. test gap**: Spearman drops from 0.679 (train) to 0.384 (test) — a 43% decrease. R² drops from 0.446 to 0.158 (65% decrease). This indicates **moderate overfitting**, typical for tree-based models on tabular data when some features encode patent-specific patterns.

2. **Val vs. test performance**: The slight test improvement over validation (Spearman 0.384 vs. 0.370, R² 0.158 vs. 0.095) suggests the validation set may contain harder patents. The model generalizes reasonably to unseen patents.

3. **Training speed**: The entire pipeline (feature extraction + model training + evaluation) completes in approximately **5–10 minutes** on a modern multi-core CPU, compared to several hours for Model 05 on GPU.

4. **No early stopping was triggered** for XGBoost or LightGBM within 200 boosting rounds — all four models trained to completion with their configured `n_estimators=200`.

### Saved Artifacts

| File | Size | Description |
|------|-----:|:------------|
| `ensemble.pkl` | 218 MB | Pickled VotingRegressor + StandardScaler + feature names |
| `predictions.csv` | ~1 MB | 23,373 rows (header + 23,372 test predictions) |
| `metrics.json` | <1 KB | Full train/val/test metrics + OligoAI comparison |

The `ensemble.pkl` file contains the complete trained ensemble (all four models), the fitted `StandardScaler`, and the ordered list of 119 feature names required for inference.

### Comparison with OligoAI Baseline (§7.3)

As required by the project deliverables, direct comparison with OligoAI:

| Metric | Model 06 (Ensemble) | Model 05 (Deep) | OligoAI Paper | Status |
|--------|---------------------|------------------|---------------|--------|
| **Spearman ρ** | **0.384** | 0.366 | ≈ 0.42 | Below baseline |
| **Enrichment** | 2.55× | **2.61×** | ≈ 3× | Below baseline |

The OligoAI baseline values come from the OligoAI research paper (see `docs/reference_paper.txt`).

> Per the project requirements (§6): "It is acceptable if no approach outperforms OligoAI, as long as the validation protocol is followed, results are reported clearly, and limitations are analyzed thoughtfully."

### Analysis

1. **Best Spearman in the pipeline**: At **0.384**, the ensemble achieves the highest test Spearman among all models in this project, outperforming the deep learning Model 05 (0.366). This is 8.3% below the OligoAI baseline.

2. **Train-test gap**: The significant R² gap (0.446 train → 0.158 test) and Spearman gap (0.679 → 0.384) indicate **moderate overfitting**, typical for tree-based models on tabular data with many features relative to the effective diversity of the data.

3. **Predictions on natural scale**: Unlike Model 05, predictions are in raw inhibition percentages (0–100), making them directly interpretable for practitioners.

4. **Fast iteration**: The entire training pipeline completes in under 10 minutes on CPU, enabling rapid experimentation with features and hyperparameters.

5. **Feature engineering matters**: Despite lacking pretrained language model representations, the ensemble with 119 handcrafted features achieves a competitive Spearman, validating the importance of domain-specific feature engineering in this problem.

---

## Reproduction Instructions

### Prerequisites

```bash
# Python 3.9+
pip install scikit-learn numpy pandas scipy tqdm

# Optional (recommended)
pip install xgboost lightgbm
```

If XGBoost or LightGBM are not installed, the ensemble will be built with only RF + GB (2 models instead of 4).

### Training

```bash
# From project root
cd chiral_pipeline
python models/06_ensemble/train.py
```

No CLI arguments are needed — all configurations are hardcoded in the script. Training takes approximately 5–10 minutes on a modern multi-core CPU.

### Using run.sh / Docker (§7.4)

The project provides shell scripts and a Docker setup for reproducibility:

```bash
# Via run.sh (recommended)
./scripts/run.sh 06

# Via Docker
./scripts/build.sh                    # Build image
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    aso-prediction bash scripts/run.sh 06
```

### Output

```
results/06_ensemble/
├── metrics.json          # Train/val/test metrics + comparison with OligoAI (<1 KB)
├── predictions.csv       # Test set y_true vs y_pred (23,373 rows, raw inhibition %)
└── weights/
    └── ensemble.pkl      # Pickled ensemble model + scaler + feature names (218 MB)
```

### Loading a Trained Model for Inference

```python
import pickle
import numpy as np

# Load saved model
with open('results/06_ensemble/weights/ensemble.pkl', 'rb') as f:
    saved = pickle.load(f)

ensemble = saved['ensemble']
scaler = saved['scaler']
feature_names = saved['feature_names']

# Prepare features (119 features in correct order)
# X_new should be a numpy array of shape (n_samples, 119)
X_new_scaled = scaler.transform(X_new)
predictions = ensemble.predict(X_new_scaled)  # Returns inhibition %
```

---

## File Reference

| File | Description |
|------|-------------|
| `models/06_ensemble/train.py` | Complete training pipeline (547 lines): feature engineering, model training, evaluation |
| `utils/dataset.py` | `create_patent_split()` — patent-level data splitting |
| `utils/enhanced_dataset.py` | `parse_genomic_region()`, `check_mirna_overlap()` — biological feature helpers |
| `utils/thermo_features.py` | `compute_delta_g_nn()` — thermodynamic feature computation (optional) |
| `results/06_ensemble/metrics.json` | Final evaluation results |
| `results/06_ensemble/predictions.csv` | Test set predictions (23,373 rows) |
| `results/06_ensemble/weights/ensemble.pkl` | Serialized model + scaler + feature names |

### Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| scikit-learn | RF, GB, VotingRegressor, StandardScaler | ✓ |
| numpy | Numerical operations | ✓ |
| pandas | Data loading & manipulation | ✓ |
| scipy | Spearman correlation | ✓ |
| tqdm | Progress bars | ✓ |
| xgboost | XGBoost regressor | Optional |
| lightgbm | LightGBM regressor | Optional |

---

## Known Limitations & Future Work

1. **Below OligoAI baseline**: The 0.384 Spearman is 8.3% below OligoAI's ≈0.42. The gap may be partially due to missing the rich contextual representations that a pretrained language model provides. Per requirements §6, negative results are still considered valuable outcomes.

2. **Spearman metric discrepancy**: The `train.py` script computes **global Spearman** during evaluation, not the per-screen mean Spearman required by the OligoAI protocol (§5.2). Use `scripts/evaluate.py` for the official per-screen metric.

3. **Moderate overfitting**: The train Spearman of 0.679 vs. test 0.384 (−43%) and R² of 0.446 vs. 0.158 (−65%) show the tree-based models capture some patent-specific patterns that don't generalize. Possible mitigations:
   - Stronger regularization (lower `max_depth`, higher `min_samples_leaf`)
   - Feature selection (remove noisy or correlated features)
   - Cross-validated hyperparameter tuning

4. **Equal weighting**: The VotingRegressor uses simple averaging. Performance could improve with:
   - Inverse-variance weighting based on validation performance
   - Stacking (meta-learner on base model predictions)
   - Bayesian model combination

4. **No sequence order information**: Handcrafted features lose sequential dependencies (e.g., motif positions, local structure around a specific nucleotide). k-mer or n-gram features could partially address this.

5. **Sparse positional one-hot features**: The `pos_{i}_{nt}` features are sparse (only one pos×nucleotide combination fires per position). These could be replaced with numerical position-specific property encodings.

6. **Potential improvements**:
   - Optuna hyperparameter search across all four models
   - Feature importance analysis to prune low-utility features
   - Stacking ensemble with Model 05 predictions as an additional feature
   - CatBoost as an additional ensemble member (handles categoricals natively)
   - SHAP analysis for biological interpretation of predictions
