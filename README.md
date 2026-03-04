# Model 03: acc_clean-Powered LightGBM Hybrid

## Overview

Model 03 combines the two most powerful approaches across the entire pipeline:

1. **acc_clean's ACC feature columns** — 700 position-specific nucleotide×chemistry co-occurrence features (`01Ao`…`26sT`) that encode the ASO at each position. This is the primary driver behind acc_clean's **0.48 mean Spearman** performance.

2. **Model 04's rich feature engineering** — tri/tetra-nucleotide k-mers, positional property encoding, chemistry (sugar/backbone mods), context features, motifs — all engineered to capture ASO–RNA binding.

The backbone is **LightGBM** (native API) optimised with **Optuna Bayesian HPO targeting the Spearman correlation metric**, exactly as in acc_clean.

---

## Architecture at a Glance

| Component | Detail |
|---|---|
| **Model type** | LightGBM (native API, `lgb.train`) |
| **HPO** | Optuna, 100 trials, minimises `-mean_Spearman_by_patent_screen` |
| **Split** | `create_patent_split` — patent-level stratification (80/10/10) |
| **ACC features** | 700 columns (`01Ao`…`26sT`) from acc-enriched CSV |
| **Sequence features** | GC, fractions, MW, motifs, tri/tetra k-mers, sinusoidal positional encoding |
| **Chemistry features** | Sugar (DNA/MOE/cEt/LNA), backbone (PS/PO), gapmer detection, wing transitions |
| **Context features** | GC frac, AU frac, homopolymer runs, miRNA motifs, trimer k-mers |
| **Dosage** | raw + log₁p |
| **Transfection method** | Single LightGBM categorical column (fitted on TRAIN only) |
| **Feature selection** | Correlation pruning (`r>0.97`) → mutual-information quantile (`q<0.03`) |
| **Evaluation** | Mean Spearman by `custom_id`, EF@10%, top-pred-ratio, R², MAE, RMSE |
| **SHAP** | TreeExplainer analysis on test set |

---

## Quick Start

```bash
cd /path/to/chiral_pipeline

# Fast test (no HPO, base CSV)
python models/03_acc_lgbm_hybrid/train.py --skip_hpo

# Full run pointing to the ACC-enriched CSV (recommended)
python models/03_acc_lgbm_hybrid/train.py \
    --data_path ../acc_clean/data/aso_inhibitions_21_08_25_incl_context_w_flank_50_acc_normalized_clean.csv \
    --n_trials 100 \
    --num_boost_round 5000 \
    --early_stopping_rounds 200 \
    --output_dir results/03_acc_lgbm_hybrid
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | auto-detect | Path to CSV. Auto-detects acc-enriched CSV first, falls back to base. |
| `--output_dir` | `results/03_acc_lgbm_hybrid` | Where to write all outputs |
| `--train_ratio` | 0.80 | Patent-level train fraction |
| `--val_ratio` | 0.10 | Patent-level val fraction |
| `--use_aso_length` | off | Include `aso_length` as a feature |
| `--use_dg_features` | off | Include thermodynamic ΔG features |
| `--use_gpu` | off | Enable LightGBM GPU device |
| `--skip_hpo` | off | Skip Optuna, use tuned default HPs |
| `--n_trials` | 100 | Number of Optuna trials |
| `--num_boost_round` | 5000 | Max LightGBM boosting rounds |
| `--early_stopping_rounds` | 200 | Early stopping patience |
| `--corr_threshold` | 0.97 | Correlation-pruning threshold |
| `--mi_quantile` | 0.03 | MI quantile cutoff (drops bottom 3%) |

---

## Output Files

All outputs are written to `results/03_acc_lgbm_hybrid/`:

| File | Description |
|---|---|
| `metrics.json` | Train/val/test metrics (Spearman, EF, R², MAE, RMSE) |
| `best_params.json` | Final LightGBM hyperparameters |
| `optuna_summary.json` | Optuna search summary |
| `feature_names.json` | Final selected feature list |
| `feature_importance.csv` | Gain + split importance for all features |
| `feature_selection_report.json` | Correlation + MI pruning report |
| `tm_categories.json` | Transfection method categories (fitted on train) |
| `lgbm_model.txt` | LightGBM native model file |
| `model_bundle.pkl` | Full inference bundle (model + scaler + metadata) |
| `predictions_{train,val,test}.csv` | Per-sample predictions |
| `shap/shap_values.csv` | Raw SHAP values |
| `shap/shap_importance.csv` | Mean |SHAP| per feature |
| `shap/shap_bar_top30.png` | Bar chart of top-30 SHAP features |
| `logs/train_YYYYMMDD_HHMMSS.log` | Full run log |

---

## How It Differs From acc_clean

| | acc_clean | Model 03 |
|---|---|---|
| **Data split** | `ASODataModule` (rinalmo) | `create_patent_split` |
| **Sequence features** | — | ✅ k-mers, GC, motifs, positional encoding |
| **Chemistry features** | — | ✅ sugar/backbone mods (position-aware) |
| **Context features** | gc_frac, au_frac, homopolymer | ✅ + miRNA, purine runs, context k-mers |
| **Feature selection** | — | ✅ corr + MI pruning |
| **SHAP analysis** | — | ✅ |
| **Optuna search space** | 9 params | **13 params** (adds `path_smooth`, wider `num_leaves`) |

## How It Differs From Model 04

| | Model 04 | Model 03 |
|---|---|---|
| **ACC features** | ❌ | ✅ 700 columns |
| **Model type** | StackingRegressor (5 base learners) | LightGBM (native) |
| **HPO objective** | per-model Spearman | **joint mean Spearman by custom_id** (acc_clean protocol) |
| **Categorical handling** | binary flags | **single LGB categorical column** (lighter, more accurate) |

---

## Expected Performance (on test set)

| Metric | acc_clean (val) | Model 04 | **Model 03 target** |
|---|---|---|---|
| Mean Spearman | **0.48** | ~0.35 | **>0.48** |
| EF@10% | 2.5 | — | >2.5 |
| R² | 0.36 | — | — |
