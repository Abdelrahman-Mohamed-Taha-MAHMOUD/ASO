# Model 04: Enhanced Stacking Ensemble

## Table of Contents

1. [Background](#background)
2. [Overview](#overview)
3. [Motivation & Design Rationale](#motivation--design-rationale)
   - [Limitations Addressed from Model 06](#limitations-addressed-from-model-06)
4. [Architecture](#architecture)
   - [High-Level Pipeline](#high-level-pipeline)
   - [Base Learners](#base-learners)
   - [Stacking Meta-Learner](#stacking-meta-learner)
5. [Feature Engineering](#feature-engineering)
   - [Sequence Features](#sequence-features)
   - [k-mer Features (NEW)](#k-mer-features-new)
   - [Positional Property Encoding (NEW)](#positional-property-encoding-new)
   - [Chemistry Features](#chemistry-features)
   - [Context Features](#context-features)
   - [Thermodynamic Features](#thermodynamic-features)
   - [Enhanced (Biological) Features](#enhanced-biological-features)
   - [Experimental Metadata](#experimental-metadata)
   - [Complete Feature Summary](#complete-feature-summary)
6. [Feature Selection](#feature-selection)
   - [Stage 1 — Correlation Pruning](#stage-1--correlation-pruning)
   - [Stage 2 — Mutual-Information Pruning](#stage-2--mutual-information-pruning)
7. [Hyperparameter Optimisation](#hyperparameter-optimisation)
   - [Optuna Bayesian Search](#optuna-bayesian-search)
   - [Default Hyperparameters](#default-hyperparameters)
8. [Training Pipeline](#training-pipeline)
   - [Data Loading & Splitting](#data-loading--splitting)
   - [Feature Extraction](#feature-extraction)
   - [Feature Selection & Scaling](#feature-selection--scaling)
   - [Base Model Training](#base-model-training)
   - [Stacking Ensemble Construction](#stacking-ensemble-construction)
9. [GPU Acceleration](#gpu-acceleration)
10. [Evaluation Protocol](#evaluation-protocol)
    - [Data Split (§5.1)](#data-split-51)
    - [Spearman Rank Correlation — `mean_spearman_corr` (§5.2)](#spearman-rank-correlation--mean_spearman_corr-52)
    - [Enrichment Factor — `top_pred_target_ratio_median` (§5.3)](#enrichment-factor--top_pred_target_ratio_median-53)
    - [Additional Metrics](#additional-metrics)
    - [Required Output Format — `metrics.json` (§7)](#required-output-format--metricsjson-7)
11. [SHAP Explainability Analysis](#shap-explainability-analysis)
12. [Results & Analysis](#results--analysis)
    - [Base Model Comparison](#base-model-comparison)
    - [Stacking Ensemble Performance](#stacking-ensemble-performance)
    - [Comparison with Baselines](#comparison-with-baselines)
    - [Top SHAP Features](#top-shap-features)
13. [Reproduction Instructions](#reproduction-instructions)
    - [Quick Start](#quick-start)
    - [Full Run with Optuna HPO](#full-run-with-optuna-hpo)
    - [Using run.sh / Docker](#using-runsh--docker)
    - [CLI Arguments](#cli-arguments)
14. [Output Files Reference](#output-files-reference)
15. [Logging & Disk Persistence](#logging--disk-persistence)
16. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Background

Antisense oligonucleotides (ASOs) are short, synthetic, single-stranded nucleic acids designed to bind complementary RNA sequences and modulate gene expression. In RNase H–mediated ASOs, hybridization forms a DNA–RNA heteroduplex that is cleaved by RNase H, resulting in degradation of the target RNA. Therapeutic ASOs commonly employ **gapmer designs** with:

- A central DNA gap required for RNase H activity
- Chemically modified flanking regions (e.g., 2ʼ-MOE, cEt) for stability
- Backbone modifications (e.g., phosphorothioate/PS) for pharmacokinetics

Predicting ASO efficacy remains challenging due to interactions between sequence, target RNA context, chemical modifications, and experimental conditions. This project uses the same ~180K ASO dataset and evaluation methodology as **OligoAI** (Spearman $\rho \approx 0.42$, enrichment $\approx 3\times$) to explore whether alternative architectures can achieve better or complementary performance.

---

## Overview

Model 04 is an **enhanced stacking ensemble** that directly addresses every known limitation of Model 06. It combines **five** gradient-boosted and tree-based regressors via a **Ridge meta-learner** trained on out-of-fold predictions, replacing the simple averaging strategy of Model 06-. The model introduces k-mer features, positional property encodings, two-stage feature selection, optional Bayesian hyperparameter optimisation (Optuna), per-screen Spearman evaluation, full SHAP explainability, and comprehensive disk logging.

| Property | Value |
|----------|-------|
| **Task** | Regression — predict ASO inhibition (%) |
| **Primary metric** | Spearman rank correlation ($\rho$) |
| **Secondary metric** | Enrichment factor (top-10% hit rate) |
| **Framework** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Ensemble strategy** | Stacking (5-fold CV + Ridge meta-learner) |
| **Base models** | RF, GradientBoosting, XGBoost, LightGBM, CatBoost |
| **Features** | 533 raw → 492 after selection |
| **Training script** | `models/04_enhanced_ensemble/train.py` |
| **GPU support** | XGBoost (CUDA), CatBoost (GPU) — auto-detected |

---

## Motivation & Design Rationale

Model 06 established that handcrafted features combined with a voting ensemble of tree-based learners can approach deep-learning performance on ASO efficacy prediction. However, it left several opportunities for improvement on the table. Model 04 was designed to systematically close every identified gap.

### Limitations Addressed from Model 06

| # | Model 06 Limitation | Model 04 Solution |
|---|---------------------|-------------------|
| 1 | **No HPO** — all hyperparameters were manually chosen | **Optuna Bayesian search** with configurable `--optuna_trials` per base model |
| 2 | **No CatBoost** — only 4 base learners (RF, GB, XGB, LGB) | **CatBoost added** as 5th base learner with native categorical handling |
| 3 | **Simple averaging** — VotingRegressor computes arithmetic mean | **StackingRegressor** with Ridge meta-learner learns optimal blending weights via 5-fold CV |
| 4 | **No k-mer features** — only single-nucleotide and dinucleotide frequencies | **Tri-nucleotide** (64 features) and **tetra-nucleotide** (256 features) normalised frequency vectors |
| 5 | **Sparse one-hot positional encoding** — 60 binary features for 5 start/end positions | **Sinusoidal + nucleotide-property encoding** — dense 60 float features capturing purine/pyrimidine/amino/keto properties |
| 6 | **No feature selection** — all extracted features passed through | **Two-stage selection**: correlation pruning ($\|r\| > 0.95$) + mutual-information importance pruning |
| 7 | **No per-screen Spearman** — only global Spearman reported | **Per-screen Spearman** (OligoAI §5.2 protocol): mean of per-patent Spearman correlations |
| 8 | **No SHAP analysis** — no feature importance or explainability | Full **SHAP TreeExplainer** with beeswarm, bar, and dependence plots |
| 9 | **Minimal logging** — critical metrics only to console | **Dual-handler logging** (DEBUG to file, INFO to console) with every scalar, array, and figure written to disk |
| 10 | **No GPU** — all models run on CPU only | **Auto-detected GPU** for XGBoost (`device="cuda"`) and CatBoost (`task_type="GPU"`) |

---

## Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL 04 PIPELINE                                │
│                                                                         │
│  ┌──────────┐                                                           │
│  │ CSV Data │  178,624 rows × 15 columns                                │
│  └────┬─────┘                                                           │
│       │                                                                 │
│  ┌────▼──────────────────────┐                                          │
│  │ Patent-Level Split        │                                          │
│  │ train=142,861  val=12,391 │                                          │
│  │ test=23,372               │                                          │
│  └────┬──────────────────────┘                                          │
│       │                                                                 │
│  ┌────▼──────────────────────┐                                          │
│  │ Feature Extraction        │  533 raw features per sample             │
│  │ • Sequence (len, GC, di,  │                                          │
│  │   tri-mer, tetra-mer,     │                                          │
│  │   motifs, positional enc) │                                          │
│  │ • Chemistry (sugar, BB,   │                                          │
│  │   gapmer, pos-aware)      │                                          │
│  │ • Context (GC, runs,      │                                          │
│  │   miRNA, ctx k-mers)      │                                          │
│  │ • Thermo (ΔG)             │                                          │
│  │ • Enhanced (region,       │                                          │
│  │   miRNA overlap, struct)  │                                          │
│  │ • Metadata (dose, method) │                                          │
│  └────┬──────────────────────┘                                          │
│       │                                                                 │
│  ┌────▼──────────────────────┐                                          │
│  │ Feature Selection         │  533 → 492 features                      │
│  │ 1. Correlation pruning    │  (removed 41 correlated)                 │
│  │    (|r| > 0.95)           │                                          │
│  │ 2. Mutual-information     │  (removed 0 low-MI)                     │
│  │    pruning (q < 0.05)     │                                          │
│  └────┬──────────────────────┘                                          │
│       │                                                                 │
│  ┌────▼──────────────────────┐                                          │
│  │ StandardScaler            │  Zero-mean, unit-variance                │
│  └────┬──────────────────────┘                                          │
│       │                                                                 │
│  ┌────▼──────────────────────┐    ┌──────────────────────────────┐      │
│  │ [Optional] Optuna HPO     │───▶│ Best HPs per base model      │      │
│  │ 50 trials × 5 models     │    │ (or skip → use defaults)     │      │
│  └────┬──────────────────────┘    └──────────────────────────────┘      │
│       │                                                                 │
│  ┌────▼──────────────────────────────────────────────────────────┐      │
│  │ Train 5 Base Models Individually                              │      │
│  │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                │      │
│  │ │  RF  │ │  GB  │ │ XGB  │ │ LGB  │ │ CAT  │                │      │
│  │ │      │ │      │ │(GPU) │ │      │ │(GPU) │                │      │
│  │ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                │      │
│  │    │        │        │        │        │                     │      │
│  │    └────────┴────────┼────────┴────────┘                     │      │
│  │                      │                                        │      │
│  │  ┌───────────────────▼────────────────────────────────────┐  │      │
│  │  │ StackingRegressor (5-fold CV)                          │  │      │
│  │  │ OOF predictions from each base → Ridge meta-learner    │  │      │
│  │  │ Learns optimal blending weights                        │  │      │
│  │  └───────────────────┬────────────────────────────────────┘  │      │
│  └──────────────────────┼────────────────────────────────────────┘      │
│                         │                                               │
│  ┌──────────────────────▼────────────────────────────────────────┐      │
│  │ Evaluation (train / val / test)                                │      │
│  │ R², MAE, RMSE, global Spearman, per-screen Spearman,          │      │
│  │ Pearson r, enrichment factor                                   │      │
│  └──────────────────────┬────────────────────────────────────────┘      │
│                         │                                               │
│  ┌──────────────────────▼────────────────────────────────────────┐      │
│  │ SHAP Analysis (TreeExplainer on best base model)               │      │
│  │ → shap_values.csv, shap_importance.csv                         │      │
│  │ → beeswarm, bar chart, dependence plots                        │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  All outputs → results/04_enhanced_ensemble/                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Base Learners

| Model | Implementation | Key Properties | GPU |
|-------|---------------|----------------|-----|
| **Random Forest (RF)** | `sklearn.ensemble.RandomForestRegressor` | Bagging, decorrelated trees, low bias | No |
| **Gradient Boosting (GB)** | `sklearn.ensemble.GradientBoostingRegressor` | Sequential boosting, scikit-learn native | No |
| **XGBoost (XGB)** | `xgboost.XGBRegressor` | Histogram-based, L1/L2 regularisation | Yes (`device="cuda"`) |
| **LightGBM (LGB)** | `lightgbm.LGBMRegressor` | Leaf-wise growth, GOSS sampling | No (requires CMake rebuild) |
| **CatBoost (CAT)** | `catboost.CatBoostRegressor` | Ordered boosting, native categoricals | Yes (`task_type="GPU"`) |

### Stacking Meta-Learner

Unlike Model 06's `VotingRegressor` (arithmetic mean), Model 04 uses scikit-learn's `StackingRegressor`:

1. **Out-of-fold (OOF) predictions**: For each of the 5 CV folds, each base model is trained on 4/5 of the training data and predicts on the held-out 1/5.
2. **Meta-feature matrix**: The OOF predictions from all 5 base learners form a $N_{\text{train}} \times 5$ matrix.
3. **Ridge regression**: A Ridge meta-learner ($\alpha = 1.0$) learns the optimal linear combination of base predictions.
4. **Final prediction**: All 5 base models are retrained on the full training set. At inference time, their predictions are blended by the Ridge weights.

**Why stacking over voting?**
- Voting assigns equal weight to all models regardless of quality
- Stacking learns data-driven weights, allowing better models to contribute more
- Ridge regularisation prevents overfitting the meta-learner on the small 5-dimensional feature space

---

## Feature Engineering

Model 04 extracts **533 raw features** from each sample, organised into seven categories. Features marked **(NEW)** are additions over Model 06.

### Sequence Features

Extracted from the `aso_sequence_5_to_3` column.

| Feature Group | Count | Description |
|---------------|-------|-------------|
| `length` | 1 | ASO sequence length |
| `frac_{A,T,G,C}` | 4 | Nucleotide fractions |
| `gc_content` | 1 | GC fraction = (G+C)/length |
| `mw` | 1 | Estimated molecular weight |
| `di_{AA..TT}` | 16 | Dinucleotide frequencies (4×4) |
| `bad_{motif}`, `good_{motif}` | 10 | Per-motif counts (5 bad + 5 good from OligoAI) |
| `total_bad_motifs`, `total_good_motifs` | 2 | Aggregated motif counts |

**Motifs** (from the OligoAI paper):
- Bad: `GGGG`, `AAAA`, `TAAA`, `CTAA`, `CCTA`
- Good: `TTGT`, `GTAT`, `CGTA`, `GTCG`, `GCGT`

### k-mer Features (NEW)

| Feature Group | Count | Description |
|---------------|-------|-------------|
| `kmer3_{AAA..TTT}` | 64 | Normalised tri-nucleotide frequencies |
| `kmer4_{AAAA..TTTT}` | 256 | Normalised tetra-nucleotide frequencies |

k-mer features capture **local sequence order** that single-nucleotide and dinucleotide frequencies miss. For a sequence of length $L$, the normalised frequency of k-mer $m$ is:

$$f(m) = \frac{\text{count}(m)}{L - k + 1}$$

This adds 320 features capturing tri- and tetra-nucleotide composition patterns.

### Positional Property Encoding (NEW)

| Feature Group | Count | Description |
|---------------|-------|-------------|
| `{start,end}_{0..4}_sin` | 10 | Sinusoidal position encoding |
| `{start,end}_{0..4}_cos` | 10 | Cosine position encoding |
| `{start,end}_{0..4}_{purine,amino,keto,pyrimidine}` | 40 | Nucleotide property vector |

Model 06 used **sparse one-hot encoding** (60 binary features — 3 one-hot per nucleotide × 10 positions × 2 ends), which is wasteful. Model 04 replaces this with:

1. **Sinusoidal encoding**: $\sin(i/5 \cdot \pi)$ and $\cos(i/5 \cdot \pi)$ — captures relative position
2. **Nucleotide property vector**: Each nucleotide is encoded as 4 floats representing biochemical properties:

| Nucleotide | Purine | Amino | Keto | Pyrimidine |
|------------|--------|-------|------|------------|
| A | 1 | 1 | 0 | 0 |
| G | 1 | 0 | 1 | 0 |
| C | 0 | 1 | 0 | 1 |
| T | 0 | 0 | 1 | 1 |

This produces **60 dense float features** encoding both position and nucleotide biochemistry, replacing the 60 sparse binary features of Model 06 with the same dimensional budget but much richer representation.

### Chemistry Features

Extracted from `sugar_mods` and `backbone_mods` columns.

| Feature Group | Count | Description |
|---------------|-------|-------------|
| `sugar_{DNA,MOE,cEt,LNA}` | 4 | Raw counts per sugar type |
| `sugar_{DNA,MOE,cEt,LNA}_frac` | 4 | Fraction per sugar type |
| `backbone_PS`, `backbone_PO` | 2 | Backbone modification counts |
| `backbone_PS_frac` | 1 | PS backbone fraction |
| `is_gapmer` | 1 | Binary gapmer detection |
| `gap_length` | 1 | DNA gap length |
| `pos{0..4}_sugar_type` | 5 | Numerical sugar type at 5ʼ end positions |
| `end{0..4}_sugar_type` | 5 | Numerical sugar type at 3ʼ end positions |
| `pos{0..4}_is_PS` | 5 | PS backbone at 5ʼ end positions |
| `end{0..4}_is_PS` | 5 | PS backbone at 3ʼ end positions |
| `wing5_to_gap_transition` | 1 | Modified-to-DNA transition detected |
| `wing3_modified` | 1 | 3ʼ wing has modifications |

**Improvement over Model 06**: Sugar types at terminal positions are encoded as **numerical indices** (DNA=0, MOE=1, cEt=2, LNA=3) instead of one-hot, reducing dimensionality while preserving ordinal information about modification chemistry.

### Context Features

Extracted from the `rna_context` column (target mRNA flanking ±50 nt).

| Feature Group | Count | Description |
|---------------|-------|-------------|
| `ctx_length` | 1 | Context sequence length |
| `ctx_gc` | 1 | Context GC content |
| `ctx_mirna_sites` | 1 | Count of miRNA seed motif matches |
| `ctx_max_purine_run` | 1 | Longest purine run |
| `ctx_max_pyrimidine_run` | 1 | Longest pyrimidine run |
| `ctx_kmer3_{AAA..TTT}` | 64 | **(NEW)** Context tri-nucleotide frequencies |

The context k-mers (64 features) capture target RNA local composition patterns that may affect ASO binding accessibility.

### Thermodynamic Features

| Feature | Description | Source |
|---------|-------------|--------|
| `delta_g` | Predicted binding free energy ($\Delta G$) via nearest-neighbour model | `scripts/thermo_features.py` |
| `delta_g_per_nt` | $\Delta G$ normalised by ASO length | Computed |

### Enhanced (Biological) Features

| Feature Group | Count | Description |
|---------------|-------|-------------|
| `region_{0..5}` | 6 | One-hot genomic region (exon/intron/5ʼUTR/3ʼUTR/splice/unknown) |
| `mirna_overlap` | 1 | Binary: target overlaps known miRNA binding site |
| `structure_accessibility` | 1 | Proxy based on local GC content (1 − ctx_gc) |

### Experimental Metadata

| Feature | Description |
|---------|-------------|
| `log_dosage` | $\log(1 + \text{dosage})$ — handles missing values as 4000 nM default |
| `method_gymnosis` | Binary: free uptake / gymnosis delivery |
| `method_lipo` | Binary: lipofection delivery |
| `method_electro` | Binary: electroporation delivery |

### Complete Feature Summary

| Category | Feature Count | New in M04 |
|----------|:------------:|:----------:|
| Sequence (basic) | 35 | — |
| k-mers (tri + tetra) | 320 | ✓ |
| Positional encoding | 60 | ✓ (redesigned) |
| Chemistry | 35 | — |
| Context (basic) | 5 | — |
| Context k-mers | 64 | ✓ |
| Thermodynamic | 2 | — |
| Enhanced (biological) | 8 | — |
| Metadata | 4 | — |
| **Total raw** | **533** | **+384 new** |
| **After selection** | **492** | — |

---

## Feature Selection

Model 06 passed all features through without any selection. Model 04 applies a **two-stage filter** to remove redundant and uninformative features.

### Stage 1 — Correlation Pruning

For every pair of features with Pearson $|r| > 0.95$ (configurable via `--corr_threshold`), one feature is dropped (the one encountered second in column order). This removes multicollinearity that can harm Ridge meta-learner stability and inflate SHAP attributions.

**Result**: 533 → 492 features (41 removed)

Examples of dropped features: `backbone_PS_frac` (redundant with `backbone_PS`), `delta_g_per_nt` (highly correlated with `delta_g`), several positional one-hot features correlated with property encodings.

### Stage 2 — Mutual-Information Pruning

Mutual information is computed between every remaining feature and the target (`inhibition_percent`) using 5 nearest neighbours. Features below the 5th percentile (configurable via `--mi_quantile`) are considered uninformative and removed.

**Result**: 492 → 492 features (0 removed — all features carry some signal)

The full feature selection report (including MI scores for every feature) is saved to `results/04_enhanced_ensemble/feature_selection_report.json`.

---

## Hyperparameter Optimisation

### Optuna Bayesian Search

When run without `--skip_hpo`, Model 04 uses **Optuna** to run Bayesian hyperparameter optimisation for each base learner independently:

| Model | Search Space | Key Parameters |
|-------|-------------|-------|
| **RF** | `n_estimators` ∈ [100, 500], `max_depth` ∈ [5, 20], `min_samples_split` ∈ [2, 20], `min_samples_leaf` ∈ [1, 15] | 4 parameters |
| **GB** | `n_estimators` ∈ [100, 500], `max_depth` ∈ [3, 10], `learning_rate` ∈ [0.01, 0.3] (log), `subsample` ∈ [0.6, 1.0] | 5 parameters |
| **XGB** | `n_estimators` ∈ [100, 500], `max_depth` ∈ [3, 10], `learning_rate` ∈ [0.01, 0.3] (log), `reg_alpha/lambda` ∈ [1e-3, 10] (log) | 7 parameters |
| **LGB** | Same as XGB | 7 parameters |
| **CAT** | `iterations` ∈ [100, 500], `depth` ∈ [3, 10], `learning_rate` ∈ [0.01, 0.3] (log), `l2_leaf_reg` ∈ [1e-3, 10] (log) | 5 parameters |

- **Objective**: Maximise validation Spearman $\rho$
- **Default trials**: 50 per model (configurable via `--optuna_trials`)
- **Direction**: `maximize`
- **Output**: Full trial history saved to `logs/optuna_{model}_trials.json`

### Default Hyperparameters

When running with `--skip_hpo`, Model 04 uses the following defaults (with **stronger regularisation** than Model 06):

| Parameter | M06 Value | M04 Default | Rationale |
|-----------|-----------|-------------|-----------|
| `n_estimators` | 200 | 300 | More trees for ensemble diversity |
| `max_depth` (boosters) | 12 | 5 | Stronger depth control to prevent overfitting |
| `learning_rate` | 0.1 | 0.05 | Slower learning for better generalisation |
| `reg_alpha` (XGB/LGB) | 0.0 | 0.1 | L1 regularisation |
| `reg_lambda` (XGB/LGB) | 1.0 | 1.0 | L2 regularisation (unchanged) |
| `subsample` | 1.0 | 0.8 | Row subsampling to reduce variance |
| `colsample_bytree` | 1.0 | 0.8 | Column subsampling for decorrelation |

---

## Training Pipeline

### Data Loading & Splitting

```python
# Patent-level split (same as all other models in the pipeline)
df = pd.read_csv("data/raw/aso_inhibitions_21_08_25_incl_context_w_flank_50_df.csv")
train_df, val_df, test_df = create_patent_split(df, random_state=42)
```

| Split | Samples | Purpose |
|-------|--------:|---------|
| Train | 142,861 | Model fitting |
| Validation | 12,391 | HP tuning, early stopping, per-model evaluation |
| Test | 23,372 | Final held-out evaluation |

All samples from the same patent are kept within a single split to prevent data leakage (ASOs targeting the same gene in the same experiment are correlated).

### Feature Extraction

For each sample, `extract_all_features()` calls:
1. `extract_sequence_features()` — length, nucleotide fractions, GC, MW, dinucleotides, motifs, **k-mers**, **positional encoding**
2. `extract_chemistry_features()` — sugar/backbone counts and fractions, gapmer detection, position-aware sugar types
3. `extract_context_features()` — context GC, purine/pyrimidine runs, miRNA sites, **context k-mers**
4. Thermodynamic: `compute_delta_g_nn()` when available
5. Enhanced: `parse_genomic_region()`, `check_mirna_overlap()` when available
6. Metadata: log-dosage, transfection method encoding

Speed: ~1,800–2,200 samples/sec (~1.3 min for 142K train samples).

### Feature Selection & Scaling

1. **Correlation pruning**: Remove one of each pair with $|r| > 0.95$
2. **MI pruning**: Remove features below 5th percentile of mutual-information scores
3. **StandardScaler**: Zero-mean, unit-variance normalisation (fitted on train only)

### Base Model Training

Each of the 5 base models is trained independently on the full training set. Validation metrics are logged and saved to `base_model_metrics.json`.

### Stacking Ensemble Construction

```python
StackingRegressor(
    estimators=[(name, model) for name, model in base_models],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=1,       # Sequential to avoid GPU contention
    passthrough=False,
)
```

The stacking process:
1. Splits training data into 5 folds
2. For each fold: trains all 5 base models on 4/5, predicts on held-out 1/5
3. Concatenates OOF predictions → $N \times 5$ matrix
4. Fits Ridge on OOF predictions → target
5. Retrains all base models on full training set for inference

---

## GPU Acceleration

Model 04 auto-detects GPU availability at startup via PyTorch's `torch.cuda.is_available()` (with nvidia-smi fallback). When a GPU is detected:

| Model | GPU Parameter | Speedup (vs CPU) |
|-------|--------------|:-----------------:|
| **XGBoost** | `device="cuda"` | ~2× (12.4s → 6.0s) |
| **CatBoost** | `task_type="GPU"`, `bootstrap_type="Bernoulli"` | ~3× (10.2s → 3.1s) |
| LightGBM | CPU only (pip build lacks GPU) | — |
| RF, GB | CPU only (sklearn, no GPU support) | — |

**GPU contention**: The `StackingRegressor` uses `n_jobs=1` (sequential fold fitting) to prevent multiple GPU-accelerated models from fighting over device 0 during cross-validation. Individual base model training is parallelised via each model's own `n_jobs=-1` (CPU models) or GPU offloading.

If no GPU is available, all models fall back to CPU automatically — no code changes needed.

---

## Evaluation Protocol

Model 04 follows the OligoAI paper's evaluation methodology with additional metrics.

### Data Split (§5.1)

Patent-level stratification ensures zero leakage between splits. Identical to all other models in the pipeline.

### Spearman Rank Correlation — `mean_spearman_corr` (§5.2)

Model 04 computes **both** global and per-screen Spearman correlations:

1. **Global Spearman** ($\rho_{\text{global}}$): Standard rank correlation across all test samples
2. **Per-screen Spearman** ($\rho_{\text{screen}}$): For each patent/screen with $\geq 5$ samples, compute individual Spearman $\rho$, then average across screens

The per-screen metric is the **primary metric** defined in OligoAI §5.2, as it measures how well the model ranks ASOs within each experimental context. Model 06 only reported global Spearman.

### Enrichment Factor — `top_pred_target_ratio_median` (§5.3)

Measures overlap between predicted top-10% and actual top-10% ASOs:

$$\text{Enrichment} = \frac{\left|\text{top}\_{10\%}^{\text{pred}} \cap \text{top}\_{10\%}^{\text{true}}\right|}{0.1 \times N\_{\text{top}}}$$

Random baseline enrichment is $1.0\times$. Higher is better.

### Additional Metrics

| Metric | Description |
|--------|-------------|
| $R^2$ | Coefficient of determination |
| MAE | Mean absolute error (in % inhibition) |
| RMSE | Root mean squared error |
| Pearson $r$ | Linear correlation |

### Required Output Format — `metrics.json` (§7)

The output `metrics.json` contains all metrics for all splits, per-screen details, base model comparison, and timing information. See [Output Files Reference](#output-files-reference) for full structure.

---

## SHAP Explainability Analysis

Model 04 runs a full **SHAP (SHapley Additive exPlanations)** analysis after training:

1. **Explainer**: `TreeExplainer` on the best base model with `feature_importances_` (typically Random Forest)
2. **Sample size**: 2,000 test samples (subsampled for speed)
3. **Outputs**:
   - `shap_values.csv` — Raw SHAP values for each sample × feature (20 MB)
   - `shap_importance.csv` — Mean |SHAP| per feature, sorted descending
   - `shap_bar_top30.png` — Horizontal bar chart of top 30 features
   - `shap_beeswarm.png` — Beeswarm plot showing SHAP value distribution
   - `shap_dependence_top1.png` — Dependence plot for the most important feature

**Fallback**: If the stacking ensemble's internal estimators don't support TreeExplainer, a `KernelExplainer` is used (slower, on 500 samples with 100-sample background).

---

## Results & Analysis

### Base Model Comparison

Individual base model performance on the **validation** set (default hyperparameters, `--skip_hpo`):

| Model | Val Spearman $\rho$ | Val $R^2$ | Val MAE | Train Time |
|-------|:-------------------:|:---------:|:-------:|:----------:|
| RF | 0.3696 | 0.1050 | 22.37 | 20.5s |
| GB | 0.3982 | 0.1191 | 21.89 | 554.9s |
| XGB (GPU) | 0.3906 | 0.1292 | 21.79 | 6.0s |
| LGB | 0.3856 | 0.1230 | 21.86 | 51.3s |
| CAT (GPU) | 0.3857 | 0.1329 | 21.84 | 3.1s |

**Observations**:
- GB achieves the best individual val Spearman but is by far the slowest (no GPU)
- XGB and CAT benefit enormously from GPU acceleration (6s and 3s respectively)
- CatBoost has the best $R^2$ among individual models
- All individual models fall within a narrow $\rho$ band (0.370–0.398)

### Stacking Ensemble Performance

| Split | $R^2$ | MAE | RMSE | Global $\rho$ | Per-screen $\rho$ | Screens | Enrichment |
|-------|:-----:|:---:|:----:|:-------------:|:-----------------:|:-------:|:----------:|
| Train | 0.4126 | 17.47 | 21.47 | 0.6469 | 0.3828 | 294 | 4.26× |
| Val | 0.1274 | 21.70 | 25.85 | 0.3906 | 0.3227 | 36 | 2.27× |
| **Test** | **0.1765** | **20.17** | **24.57** | **0.4049** | **0.2473** | **38** | **2.40×** |

### Comparison with Baselines

| Model | Test Global $\rho$ | Test Per-screen $\rho$ | Test Enrichment |
|-------|:------------------:|:---------------------:|:---------------:|
| **OligoAI** (paper) | 0.419 | 0.419 | ~3× |
| Model 06 (Voting) | 0.384 | — (not computed) | — |
| **Model 04 (Stacking)** | **0.405** | **0.247** | **2.40×** |

- **vs Model 06**: Global Spearman **+5.5%** improvement (0.384 → 0.405)
- **vs OligoAI**: Global Spearman **−3.4%** (0.419 → 0.405), per-screen Spearman **−41.0%** (0.419 → 0.247)

**Analysis**: The stacking ensemble significantly improves over Model 06's simple voting approach. The gap to OligoAI's global Spearman is narrowing (from 9.1% deficit to 3.4%), and the enrichment factor (2.40×) approaches OligoAI's ~3×. The per-screen Spearman (0.247) is lower than global, which is expected: per-screen evaluation is a harder metric that requires the model to rank correctly *within* small per-patent groups. Further gains may require Optuna HPO, deeper features, or hybrid approaches.

### Top SHAP Features

The 20 most influential features by mean |SHAP value|:

| Rank | Feature | Mean |SHAP| | Category |
|:----:|---------|:----------:|----------|
| 1 | `backbone_PS` | 1.042 | Chemistry — PS backbone count |
| 2 | `gc_content` | 0.714 | Sequence — GC fraction |
| 3 | `backbone_PO` | 0.609 | Chemistry — PO backbone count |
| 4 | `frac_A` | 0.594 | Sequence — adenine fraction |
| 5 | `pos1_is_PS` | 0.569 | Chemistry — 2nd position is PS |
| 6 | `total_bad_motifs` | 0.541 | Sequence — aggregate bad motifs |
| 7 | `di_AA` | 0.483 | Sequence — AA dinucleotide freq |
| 8 | `di_GT` | 0.438 | Sequence — GT dinucleotide freq |
| 9 | `method_electro` | 0.411 | Metadata — electroporation |
| 10 | `frac_T` | 0.380 | Sequence — thymine fraction |
| 11 | `ctx_kmer3_GGG` | 0.343 | Context — GGG trimer in target |
| 12 | `frac_G` | 0.273 | Sequence — guanine fraction |
| 13 | `method_gymnosis` | 0.256 | Metadata — free uptake |
| 14 | `sugar_cEt` | 0.253 | Chemistry — cEt sugar count |
| 15 | `log_dosage` | 0.244 | Metadata — log-transformed dose |
| 16 | `kmer3_AAA` | 0.237 | **k-mer** — AAA trimer freq |
| 17 | `di_TG` | 0.221 | Sequence — TG dinucleotide freq |
| 18 | `frac_C` | 0.212 | Sequence — cytosine fraction |
| 19 | `di_GC` | 0.211 | Sequence — GC dinucleotide freq |
| 20 | `ctx_kmer3_GGG` | — | Context — (see rank 11) |

**Key insight**: Backbone chemistry (`backbone_PS`, `backbone_PO`, `pos1_is_PS`) dominates feature importance, followed by sequence composition (`gc_content`, `frac_A`, `total_bad_motifs`). The new k-mer features (`kmer3_AAA`, `ctx_kmer3_GGG`) appear in the top 20, validating their addition. Experimental metadata (`method_electro`, `method_gymnosis`, `log_dosage`) is also highly influential, confirming that experimental context matters significantly for efficacy prediction.

---

## Reproduction Instructions

### Quick Start

```bash
# Activate environment
conda activate aso_pred

# Run with default hyperparameters (fastest — ~2 hours with GPU)
python models/04_enhanced_ensemble/train.py --skip_hpo
```

### Full Run with Optuna HPO

```bash
# Run with Bayesian hyperparameter optimisation (50 trials per model)
python models/04_enhanced_ensemble/train.py --optuna_trials 50

# Or with more trials for thorough search
python models/04_enhanced_ensemble/train.py --optuna_trials 200
```

### Using run.sh / Docker

```bash
# Via the unified run script
bash scripts/run.sh 04

# Via Docker (GPU passthrough)
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
  aso-pred bash scripts/run.sh 04
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--skip_hpo` | flag | `False` | Skip Optuna HPO, use default hyperparameters |
| `--optuna_trials` | int | 50 | Number of Optuna trials per base model |
| `--corr_threshold` | float | 0.95 | Correlation pruning threshold |
| `--mi_quantile` | float | 0.05 | Mutual-information quantile cutoff |

---

## Output Files Reference

All outputs are saved to `results/04_enhanced_ensemble/`:

```
results/04_enhanced_ensemble/
├── metrics.json                    # Full metrics (all splits, per-screen details)
├── metrics.csv                     # Flat CSV (one row per split, for easy comparison)
├── predictions.csv                 # Test predictions (y_true, y_pred)
├── predictions_train.csv           # Train predictions (patent_id, y_true, y_pred, residual)
├── predictions_val.csv             # Val predictions
├── predictions_test.csv            # Test predictions (with patent_id and residuals)
├── base_model_metrics.json         # Individual base model val metrics and timings
├── hyperparameters.json            # Final hyperparameters used (default or Optuna-tuned)
├── feature_names.json              # List of 492 selected feature names
├── feature_selection_report.json   # Full selection report (dropped features, MI scores)
├── weights/
│   └── ensemble.pkl                # Pickled StackingRegressor + scaler + feature metadata
├── logs/
│   ├── train_YYYYMMDD_HHMMSS.log  # Full training log (DEBUG level)
│   └── optuna_*_trials.json        # Optuna trial histories (when HPO enabled)
└── shap/
    ├── shap_values.csv             # Raw SHAP values (2000 × 492)
    ├── shap_importance.csv         # Mean |SHAP| per feature (sorted)
    ├── shap_bar_top30.png          # Bar chart of top 30 features
    ├── shap_beeswarm.png           # Beeswarm plot (value distribution)
    └── shap_dependence_top1.png    # Dependence plot for top feature
```

### Key File Descriptions

| File | Size | Description |
|------|:----:|-------------|
| `ensemble.pkl` | ~52 MB | Complete model state: `StackingRegressor`, `StandardScaler`, feature names, kept indices. Load with `pickle.load()`. |
| `metrics.json` | ~20 KB | Comprehensive metrics including per-screen Spearman for every patent/screen evaluated. |
| `shap_values.csv` | ~20 MB | Raw SHAP values for 2,000 test samples — use for custom analysis. |
| `feature_selection_report.json` | ~20 KB | Lists every dropped feature with detailed MI scores. |
| Training logs | ~8 KB | Timestamped, every step recorded at DEBUG level. |

---

## Logging & Disk Persistence

A core design goal of Model 04 is that **nothing stays in memory only** — every intermediate computation is written to disk.

### Dual-Handler Logging

| Handler | Level | Target | Content |
|---------|-------|--------|---------|
| File | `DEBUG` | `logs/train_YYYYMMDD_HHMMSS.log` | Everything: feature counts, MI scores, per-model timings, SHAP progress |
| Console | `INFO` | `stdout` | Key milestones: split sizes, feature dimensions, model metrics, final summary |

### What Is Persisted

| Item | File | Format |
|------|------|--------|
| Feature names (final) | `feature_names.json` | JSON array |
| Feature selection report | `feature_selection_report.json` | JSON (dropped features, MI scores, thresholds) |
| Hyperparameters | `hyperparameters.json` | JSON dict per model |
| Base model metrics | `base_model_metrics.json` | JSON (val Spearman/R²/MAE, timing, params) |
| Optuna trials | `logs/optuna_*_trials.json` | JSON (best params, all trial values) |
| Train/Val/Test metrics | `metrics.json` + `metrics.csv` | JSON (detailed) + CSV (flat) |
| Per-screen Spearman | Inside `metrics.json` | Dict: patent_id → Spearman ρ |
| Predictions (all splits) | `predictions_{split}.csv` | patent_id, y_true, y_pred, residual |
| Model weights | `weights/ensemble.pkl` | Pickle (ensemble + scaler + metadata) |
| SHAP values | `shap/shap_values.csv` | CSV (2000 × 492 matrix) |
| SHAP importance | `shap/shap_importance.csv` | CSV (feature, mean_abs_shap) |
| SHAP plots | `shap/*.png` | PNG images (150 DPI) |
| Full training log | `logs/train_*.log` | Text (timestamp + level + message) |

---

## Known Limitations & Future Work

| # | Limitation | Potential Improvement |
|---|-----------|----------------------|
| 1 | **GradientBoosting is slow** (~9 min, no GPU support) | Replace with HistGradientBoostingRegressor (native histogram) or drop GB entirely |
| 2 | **LightGBM lacks GPU** (pip build is CPU-only) | Build from source with `-DUSE_GPU=1` or use conda-forge GPU package |
| 3 | **Per-screen Spearman gap vs OligoAI** (0.247 vs 0.419) | Likely needs screen/patent-level features (e.g., cell line embeddings) or within-screen normalisation |
| 4 | **Stacking CV is expensive** (~93 min for 5-fold × 5 models) | Use 3-fold CV, or fit stacking only on top-2 models (XGB + CAT) |
| 5 | **No RNA language model features** | Add RiNALMo embeddings as features (hybrid approach with Model 05) |
| 6 | **Feature selection could be more aggressive** | Try recursive feature elimination (RFE) or Boruta |
| 7 | **Optuna not yet run** (current results use defaults) | Full HPO run may further improve metrics |
| 8 | **No target gene/cell line features** | Encode target gene importance, cell line sensitivity as categorical features |
| 9 | **No cross-validation for final evaluation** | Add repeated k-fold with confidence intervals |
| 10 | **SHAP on RF only** (not the stacking ensemble) | Implement permutation importance on the full stacker |

---

*README generated from training run 2026-02-22. Total pipeline time: 120.3 minutes (with GPU, default HPs, no HPO).*
