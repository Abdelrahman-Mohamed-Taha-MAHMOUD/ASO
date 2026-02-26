# Model 05: Enhanced OligoAI — Stage 3 Stabilized

## Table of Contents

1. [Background](#background)
2. [Overview](#overview)
3. [Motivation & Design Rationale](#motivation--design-rationale)
4. [Architecture](#architecture)
   - [High-Level Diagram](#high-level-diagram)
   - [Component Details](#component-details)
5. [Feature Engineering](#feature-engineering)
6. [Loss Functions](#loss-functions)
7. [Training Pipeline](#training-pipeline)
   - [Data Loading & Splitting](#data-loading--splitting)
   - [Gradual Unfreezing Strategy](#gradual-unfreezing-strategy)
   - [Hyperparameter Configurations](#hyperparameter-configurations)
   - [Training Loop & Callbacks](#training-loop--callbacks)
8. [Evaluation Protocol](#evaluation-protocol)
   - [Data Split](#data-split-51)
   - [Spearman Rank Correlation](#spearman-rank-correlation--mean_spearman_corr-52)
   - [Enrichment Factor](#enrichment-factor--top_pred_target_ratio_median-53)
   - [Required Output Format](#required-output-format--metricsjson-7)
9. [Results & Analysis](#results--analysis)
   - [Training History](#training-history)
   - [Comparison with OligoAI Baseline](#comparison-with-oligoai-baseline-73)
10. [Reproduction Instructions](#reproduction-instructions)
    - [Using run.sh / Docker](#using-runsh--docker-74)
11. [File Reference](#file-reference)
12. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Background

Antisense oligonucleotides (ASOs) are short, synthetic, single-stranded nucleic acids designed to bind complementary RNA sequences and modulate gene expression. In RNase H–mediated ASOs, hybridization forms a DNA–RNA heteroduplex that is cleaved by RNase H, resulting in degradation of the target RNA. Therapeutic ASOs commonly employ **gapmer designs** with:

- A central DNA gap required for RNase H activity
- Chemically modified flanking regions (e.g., 2'-MOE, cEt) for stability
- Backbone modifications (e.g., phosphorothioate/PS) for pharmacokinetics

Predicting ASO efficacy remains challenging due to interactions between sequence, target RNA context, chemical modifications, and experimental conditions. This project uses the same ~180K ASO dataset and evaluation methodology as **OligoAI** (Spearman ρ ≈ 0.42, enrichment ≈ 3×) to explore whether alternative architectures can achieve better or complementary performance.

---

## Overview

Model 05 is the most feature-rich deep learning pipeline in this project. It implements **all ten enhancements** described in or inspired by the OligoAI paper, built on top of the **RiNALMo** RNA language model backbone. The goal is to maximize Spearman rank correlation on the ASO knockdown efficacy prediction task by combining:

- A larger pretrained RNA language model (RiNALMo `mega` or `giga`)
- Rich auxiliary features (motifs, genomic regions, RNA structure, thermodynamics, miRNA overlap)
- Advanced training techniques (gradual unfreezing, mixed ranking losses, cross-attention)

| Property | Value |
|----------|-------|
| **Task** | Regression — predict ASO inhibition (%) |
| **Primary metric** | Spearman rank correlation (ρ) |
| **Secondary metric** | Enrichment factor (top-10% hit rate) |
| **Framework** | PyTorch Lightning |
| **Backbone** | RiNALMo (mega / giga / micro) |
| **Training script** | `models/05_oligoai_stage3_stabilized/train.py` |

---

## Motivation & Design Rationale

The OligoAI paper demonstrated that combining RNA language model embeddings with chemical modification features and an MLP head achieves a median Spearman ρ ≈ 0.42. Model 05 was designed to push beyond this baseline by systematically incorporating every available signal:

1. **Backbone scaling**: Moving from RiNALMo-micro (≈5M params) to mega (≈32M) or giga (≈650M) for richer contextual representations.
2. **Cross-attention**: Allowing ASO token representations to directly attend to target RNA context, modeling the binding interaction explicitly rather than through simple concatenation.
3. **Domain-specific features**: Encoding known biological priors (motif patterns, genomic region effects, RNA secondary structure accessibility) as learnable embeddings.
4. **Ranking-aware training**: Using ListNet or RankNet losses to directly optimize for rank correlation rather than pointwise MSE.
5. **Gradual unfreezing**: Preventing catastrophic forgetting of pretrained representations by progressively unfreezing transformer layers.

---

## Architecture

### High-Level Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT                                                            │
│  ┌──────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────────┐  │
│  │ ASO Seq  │  │ RNA Context  │  │Chemistry│  │ Metadata     │  │
│  │(tokens)  │  │  (tokens)    │  │(sugar,  │  │(method,dose, │  │
│  │          │  │              │  │backbone)│  │region,motif) │  │
│  └────┬─────┘  └──────┬───────┘  └────┬────┘  └──────┬───────┘  │
│       │               │               │               │          │
│  ┌────▼─────┐  ┌──────▼───────┐       │               │          │
│  │ RiNALMo  │  │  RiNALMo     │       │               │          │
│  │ Encoder  │  │  Encoder     │       │               │          │
│  │(shared)  │  │  (shared)    │       │               │          │
│  └────┬─────┘  └──────┬───────┘       │               │          │
│       │               │               │               │          │
│       │        ┌──────▼───────┐       │               │          │
│       │        │+ Structural  │       │               │          │
│       │        │  Features    │       │               │          │
│       │        └──────┬───────┘       │               │          │
│       │               │               │               │          │
│  ┌────▼───────────────▼────┐          │               │          │
│  │   Cross-Attention       │          │               │          │
│  │   (ASO → Context)      │          │               │          │
│  └────┬────────────────────┘          │               │          │
│       │                               │               │          │
│  ┌────▼───────────────────────┐       │               │          │
│  │ Position-Aware Chemistry   │◄──────┘               │          │
│  │ Embedding (pos + sugar +   │                       │          │
│  │ backbone → 24-dim)         │                       │          │
│  └────┬───────────────────────┘                       │          │
│       │                                               │          │
│  ┌────▼───────────────────┐                           │          │
│  │ Bottleneck Fusion      │                           │          │
│  │ (D+24 → 256 → D)      │                           │          │
│  │ LayerNorm + GELU       │                           │          │
│  └────┬───────────────────┘                           │          │
│       │                                               │          │
│  ┌────▼────────────┐  ┌──────────┐                    │          │
│  │ Masked Mean Pool│  │ctx Pool  │                    │          │
│  │   → 128-dim     │  │ → 128-dim│                    │          │
│  └────┬────────────┘  └────┬─────┘                    │          │
│       │                    │                          │          │
│  ┌────▼────────────────────▼──────────────────────────▼──────┐   │
│  │  CONCAT: ASO(128)+Ctx(128)+Method(8)+Motif(16)+           │   │
│  │          Region(8)+StructType(8)+ΔG(1)+miRNA(1) = 298-dim │   │
│  └────┬──────────────────────────────────────────────────────┘   │
│       │                                                          │
│  ┌────▼───────────────────────────────────────┐                  │
│  │  MLP Head                                   │                  │
│  │  298 → 256 (LN+GELU+Drop)                  │                  │
│  │  256 → 128 (LN+GELU+Drop)                  │                  │
│  │  128 → 64  (GELU)                           │                  │
│  │   64 → 1   (Linear)                         │                  │
│  └────┬───────────────────────────────────────┘                  │
│       │                                                          │
│  ┌────▼──────┐                                                   │
│  │ Predicted │                                                   │
│  │ Inhibition│                                                   │
│  └───────────┘                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. RiNALMo Backbone (Shared Encoder)

The **RiNALMo** (RNA Informative Language Model) transformer is used as a shared encoder for both the ASO sequence and the target RNA context. It is a pretrained RNA language model that produces contextual token-level embeddings.

| Config | Layers | Embedding Dim | Parameters |
|--------|--------|---------------|------------|
| `micro` | 4 | 128 | ~5M |
| `mega` | 12 | 640 | ~32M |
| `giga` | 33 | 1280 | ~650M |

Both the ASO and context sequences are passed through the **same** RiNALMo instance (weight sharing), producing `[B, L, D]` representations.

**Implementation**: `utils/enhanced_model.py` → `EnhancedOligoAI.__init__()` and `forward()`

#### 2. Cross-Attention Module

When enabled (`use_cross_attention=True`), a multi-head attention layer allows ASO token representations to attend to the context RNA representations. This models the physical binding interaction between the ASO and its target.

- **Query**: ASO embeddings `[B, L_aso, D]`
- **Key/Value**: Context embeddings `[B, L_ctx, D]`
- **Heads**: 8
- **Dropout**: 0.1
- **Residual**: LayerNorm(query + dropout(attention_output))

**Implementation**: `utils/enhanced_model.py` → `CrossAttention`

#### 3. Position-Aware Chemistry Embeddings

Rather than using simple per-position sugar/backbone lookup embeddings, this module learns **position-specific** chemistry representations capturing the known biological effect that modification type matters differently at different positions (5' wing vs. gap vs. 3' wing in gapmers).

```
Position embedding (8-dim) + Sugar embedding (16-dim) + Backbone embedding (8-dim)
    ↓ Linear fusion
    → 24-dim position-aware chemistry vector per position
```

**Implementation**: `utils/enhanced_model.py` → `PositionAwareChemistryEmbedding`

#### 4. Bottleneck Fusion Network

Fuses the sequence embeddings (from RiNALMo, potentially enriched by cross-attention) with the chemistry embeddings:

```
Input:  [B, L_aso, D + 24]  (sequence + chemistry)
    → Linear(D+24, 256) → LayerNorm → GELU → Dropout
    → Linear(256, D) → LayerNorm
Output: [B, L_aso, D]
```

**Implementation**: `utils/enhanced_model.py` → `BottleneckNetwork`

#### 5. Auxiliary Feature Embeddings

| Feature | Module | Input | Output Dim | Source |
|---------|--------|-------|------------|--------|
| **Motifs** | `MotifFeatureExtractor` | 12-dim count vector (5 bad + 5 good + 2 totals) | 16-dim | Sequence analysis |
| **Genomic Region** | `GenomicRegionEmbedding` | Index 0-5 (exon/intron/5'UTR/3'UTR/splice/unknown) | 8-dim | `custom_id` / `gene_region` column |
| **Structure Type** | `StructureTypeEmbedding` | Index 0-7 (unpaired/hairpin/multiloop/exterior/stem/bulge/internal/unknown) | 8-dim | Inferred from unpaired probabilities |
| **ΔG (Thermo)** | Raw scalar | Gibbs free energy (kcal/mol) | 1-dim | Nearest-neighbor model or ViennaRNA |
| **miRNA Overlap** | Binary indicator | Presence of miRNA seed motifs in context | 1-dim | Motif scan |
| **Method × Dosage** | `method_emb` × dosage | Transfection method (4 types) scaled by log1p(dosage) | 8-dim | Experimental metadata |

#### 6. MLP Prediction Head

A 4-layer MLP with LayerNorm, GELU activations, and dropout:

```
298 → 256 → 128 → 64 → 1
```

LayerNorm is applied after the first two linear layers for training stability.

---

## Feature Engineering

### Sequence Features (RiNALMo)

Raw nucleotide sequences (ASO and target RNA context) are tokenized using the RiNALMo `Alphabet` and encoded into dense representations by the transformer backbone. ASO sequences are truncated to 30 tokens; context sequences to 120 tokens.

### Motif Features

Based on the OligoAI paper's findings:

| Category | Motifs | Effect on Efficacy |
|----------|--------|--------------------|
| **Bad** | GGGG, AAAA, TAAA, CTAA, CCTA | Lower inhibition |
| **Good** | TTGT, GTAT, CGTA, GTCG, GCGT | Higher inhibition |

Extracted as a 12-dimensional vector: 5 bad counts + 5 good counts + total_bad + total_good.

### Genomic Region

Parsed from `custom_id` and `gene_region` fields using keyword matching:

| Region | Index | Typical Efficacy |
|--------|-------|-----------------|
| Exon/CDS | 0 | Higher |
| Intron | 1 | Variable |
| 5' UTR | 2 | Lower |
| 3' UTR | 3 | Higher |
| Splice junction | 4 | Variable |
| Unknown | 5 | — |

### RNA Secondary Structure

- **Unpaired probabilities**: Computed per-nucleotide using RNAplfold (window=40, max span=1). Stored in `data/processed/structural_features.pkl`.
- **Structure type**: Inferred from mean unpaired probability (mean > 0.7 → accessible, < 0.3 → hairpin/stem).
- Projected into the embedding space and added residually to context representations.

### Thermodynamic Features

Binding free energy (ΔG) between ASO and target RNA computed using:

1. **Primary**: Nearest-neighbor thermodynamic parameters for DNA/RNA hybrids (Sugimoto et al., 1995). Zero external dependencies.
2. **Optional**: ViennaRNA `RNAcofold` for more accurate predictions (requires ViennaRNA installation).

Stored in `data/processed/thermo_features.pkl`. The ΔG value is passed as a raw scalar to the MLP head.

### Chemistry Features

Position-specific encoding of sugar and backbone modifications:

| Modification Type | Vocabulary | Embedding Dim |
|-------------------|-----------|---------------|
| Sugar | DNA(0), MOE(1), cEt(2), LNA(3), PAD(4) | 16 |
| Backbone | PO(0), PS(1), PAD(2) | 8 |
| Position | 0..29 (max ASO length) | 8 |

These are fused into a 24-dim vector per position using a learned linear projection.

### miRNA Binding Site Overlap

A binary indicator (0/1) for whether the RNA context contains any of six known miRNA target seed sequences (mir-15/16, mir-27, let-7, mir-17, mir-19, mir-17 family).

---

## Loss Functions

Model 05 supports five loss configurations:

### 1. MSE Loss (`mse`)
Standard mean squared error:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
$$

### 2. ListNet Loss (`listnet`)
A listwise ranking loss that treats the batch as a ranked list:

$$
\mathcal{L}_{\text{ListNet}} = -\sum_{i} P_{\text{true}}(i) \cdot \log P_{\text{pred}}(i)
$$

where $P(i) = \text{softmax}(y_i / \tau)$ with temperature $\tau$.

### 3. RankNet Loss (`ranknet`)
A pairwise ranking loss (Burges et al., 2005):

$$
\mathcal{L}_{\text{RankNet}} = -\bar{P}_{ij}\log(\hat{P}_{ij}) - (1-\bar{P}_{ij})\log(1-\hat{P}_{ij})
$$

Efficient implementation samples up to 512 random pairs per batch.

### 4. Mixed Loss (`mixed`) — **CLI Default**

$$
\mathcal{L}_{\text{mixed}} = \alpha \cdot \mathcal{L}_{\text{MSE}} + (1-\alpha) \cdot \mathcal{L}_{\text{ListNet}}
$$

with $\alpha = 0.5$. Balances pointwise accuracy with ranking quality.

### 5. RankNet Mixed (`ranknet_mixed`) — **Used in Actual Training Run**

$$
\mathcal{L}_{\text{ranknet-mixed}} = \alpha \cdot \mathcal{L}_{\text{MSE}} + (1-\alpha) \cdot \mathcal{L}_{\text{RankNet}}
$$

> **Note**: The training run that produced the saved checkpoints used `ranknet_mixed` (confirmed by `hparams.yaml`), not the CLI default `mixed`. This combines MSE for pointwise accuracy with RankNet for pairwise ranking quality.

**Implementation**: `utils/enhanced_model.py` → `ListNetLoss`, `RankNetLoss`, `RankingMSELoss`

---

## Training Pipeline

### Data Loading & Splitting

1. **Data source**: `data/raw/aso_inhibitions_21_08_25_incl_context_w_flank_50_df.csv` (~180K records)
2. **Split method**: Patent-level split (all ASOs from the same patent in the same split)
   - 80% train / 10% validation / 10% test
   - Patent ID extracted from `custom_id` (text before `_table_`)
   - Deterministic with `random_state=42`
3. **Target scaling**: StandardScaler fitted on training set, applied to val/test
4. **Dataset class**: `EnhancedOligoAIDataset` (from `utils/enhanced_dataset.py`)
5. **Collation**: `enhanced_collate_fn` — pads variable-length sequences and aligns chemistry

**DataLoader configuration**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_workers` | min(16, cpu_count) | Maximize CPU utilization |
| `pin_memory` | True | Faster GPU transfer |
| `prefetch_factor` | 4 | Preload batches in RAM |
| `persistent_workers` | True | Avoid worker restart overhead |

### Gradual Unfreezing Strategy

The RiNALMo backbone is trained with a **gradual unfreezing** schedule to prevent catastrophic forgetting of pretrained representations:

| Epoch Range | Strategy |
|-------------|----------|
| 0–4 | RiNALMo fully frozen; only head parameters train |
| 5–7 | Top 3 transformer blocks unfrozen |
| 8–10 | Top 6 blocks unfrozen |
| 11–13 | Top 9 blocks unfrozen |
| 14+ | Top 12+ blocks unfrozen (depends on model size) |

Additionally, the LM receives a **reduced learning rate** (`lr × lm_lr_factor`) via separate optimizer parameter groups.

**Implementation**: `train.py` → `GradualUnfreezeCallback`

### Hyperparameter Configurations

Four presets are provided:

| Config | Backbone | LR | LM LR Factor | Batch Size | Cross-Attn | Pos-Chem | VRAM Target |
|--------|----------|-----|-------------|------------|------------|----------|-------------|
| `mega` | mega | 5e-5 | 0.05 | 32 | ✓ | ✓ | 24+ GB |
| `giga` | giga | 2e-5 | 0.02 | 8 | ✓ | ✓ | 12+ GB |
| `mega_light` | mega | 5e-5 | 0.05 | 16 | ✗ | ✓ | 6 GB |
| `micro_full` | micro | 1e-4 | 0.10 | 48 | ✓ | ✓ | 4 GB |

### Training Loop & Callbacks

| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Save top-3 checkpoints by `val_spearman` + last checkpoint |
| `EarlyStopping` | Stop after 15 epochs with no improvement (min_delta=0.001) |
| `LearningRateMonitor` | Log LR per epoch |
| `GradualUnfreezeCallback` | Progressive LM unfreezing |
| `MetricsCallback` | Track best Spearman / enrichment / epoch |

**Optimizer**: AdamW with separate parameter groups:
- Head parameters: `lr`
- RiNALMo parameters: `lr × lm_lr_factor`
- Weight decay: 0.01

**Scheduler**: CosineAnnealingWarmRestarts (`T_0=10`, `T_mult=2`, `eta_min=1e-6`)

**Precision**: FP16 mixed precision (`16-mixed`) for memory efficiency.

**Gradient clipping**: Max norm = 1.0

**Gradient accumulation**: Configurable (default: 2 steps), effective batch size = `batch_size × grad_accum`.

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

> **⚠️ Implementation note**: The `train.py` script computes **global Spearman** (single correlation over the entire test set) for simplicity during training. For the official per-screen metric matching the OligoAI definition, use the standalone evaluation script:
> ```bash
> python scripts/evaluate.py --model 05 --checkpoint results/05_oligoai_stage3_stabilized/weights/last.ckpt
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

All metrics are written to `results/05_oligoai_stage3_stabilized/metrics.json` in the required format:

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

---

## Results & Analysis

### Test Set Performance

| Metric | Model 05 | OligoAI Baseline |
|--------|----------|-----------------|
| **Spearman ρ** | 0.366 | 0.419 |
| **Enrichment** | 2.61× | ~3× |
| **R²** | 0.157 | — |
| **MAE** | 0.731 | — |
| **RMSE** | 0.888 | — |

Test set contains **23,372 samples** (patent-level hold-out).

### Actual Configuration (from `hparams.yaml`)

The training run that produced the saved checkpoints used the following hyperparameters, confirmed by `results/05_oligoai_stage3_stabilized/logs/version_0/hparams.yaml`:

| Parameter | Value |
|-----------|-------|
| Backbone (`lm_config_name`) | `mega` |
| Batch size | 12 |
| **Loss function** (`loss_type`) | **`ranknet_mixed`** |
| Alpha (MSE vs. RankNet weight) | 0.5 |
| Learning rate (head) | 5e-05 |
| LM LR factor | 0.05 (→ LM LR = 2.5e-06) |
| Weight decay | 0.01 |
| Cross-attention | Enabled |
| Position chemistry | Enabled |
| Motifs | Enabled |
| Genomic region | Enabled |
| Structure type | Enabled |
| Thermodynamic features | Enabled |
| Dropout (bottleneck) | 0.2 |
| Dropout (MLP) | 0.3 |
| Freeze LM epochs | 5 |

### Training History

Training ran for **20 epochs** (2,381 optimizer steps, ~119 steps/epoch) with gradual unfreezing of the RiNALMo backbone. The learning rate followed a **CosineAnnealingWarmRestarts** schedule with `T_0=10` (restart at epoch 10).

#### Epoch-by-Epoch Metrics

| Epoch | Phase | Mean Train Loss | Val Loss | Val Spearman | Val Enrichment | Note |
|------:|:------|----------------:|---------:|-------------:|:--------------:|:-----|
| 0 | Frozen LM | 0.7645 | 0.7955 | 0.2228 | 1.70× | Initial |
| 1 | Frozen LM | 0.7525 | 0.7699 | 0.3409 | 1.79× | +53% Spearman |
| 2 | Frozen LM | 0.7397 | 0.7817 | 0.3036 | 1.55× | |
| 3 | Frozen LM | 0.7251 | 0.7648 | 0.3358 | 1.79× | |
| 4 | Frozen LM | 0.7272 | 0.7624 | 0.3728 | 1.79× | End frozen phase |
| 5 | Top-3 unfrozen | 0.7360 | 0.7587 | 0.3614 | 1.97× | LM layers start training |
| 6 | Top-3 unfrozen | 0.7398 | 0.7608 | 0.3551 | 1.82× | |
| 7 | Top-3 unfrozen | 0.7118 | 0.7569 | 0.3732 | 1.99× | |
| 8 | Top-6 unfrozen | 0.7029 | 0.7613 | 0.3613 | 1.86× | |
| 9 | Top-6 unfrozen | 0.7034 | 0.7598 | 0.3697 | 1.86× | |
| 10 | Top-6 unfrozen | 0.7074 | 0.7631 | 0.3522 | 1.93× | LR restart |
| 11 | Top-9 unfrozen | 0.7143 | 0.7670 | 0.3753 | 1.95× | |
| 12 | Top-9 unfrozen | 0.7392 | 0.7634 | 0.3597 | 1.78× | |
| **13** | **Top-9 unfrozen** | **0.7075** | **0.7459** | **0.3856** | **2.02×** | **Best checkpoint** |
| 14 | Top-12 unfrozen | 0.7270 | 0.7653 | 0.3761 | 1.86× | |
| 15 | Top-12 unfrozen | 0.7111 | 0.7898 | 0.3377 | 1.80× | Val loss spike |
| 16 | Top-12 unfrozen | 0.6849 | 0.7616 | 0.3538 | 1.79× | |
| 17 | Top-12 unfrozen | 0.6765 | 0.7772 | 0.3381 | 1.93× | |
| 18 | Top-12 unfrozen | 0.7278 | 0.7884 | 0.3416 | 1.71× | |
| 19 | Top-12 unfrozen | 0.6687 | 0.7688 | 0.3777 | 1.86× | Final epoch |

#### Training Dynamics

1. **Frozen LM phase (epochs 0–4)**: The MLP head trains while the RiNALMo backbone is frozen. Validation Spearman rises rapidly from 0.223 to 0.373 (+67%), demonstrating that even frozen pretrained embeddings carry significant predictive signal.

2. **Gradual unfreezing (epochs 5–13)**: As transformer layers are progressively unfrozen (top-3 → top-6 → top-9), performance continues to climb. The best result occurs at **epoch 13** (end of the top-9 phase) with val_spearman = 0.386 and val_enrichment = 2.02×. Training loss decreases steadily (0.736 → 0.708).

3. **Full unfreezing (epochs 14–19)**: After all 12 mega layers are unfrozen, performance becomes more volatile. Validation loss occasionally spikes (0.790 at epoch 15, 0.788 at epoch 18), and val_spearman oscillates between 0.338–0.378 without surpassing the epoch-13 peak. Train loss continues to decrease (0.727 → 0.669), indicating emerging overfitting.

4. **Convergence assessment**: Training did not reach early stopping (patience=15 would trigger at epoch 28). The increasing val–train loss gap in late epochs (train: 0.669, val: 0.769 at epoch 19) confirms overfitting. The best checkpoint at epoch 13 was correctly selected.

#### Learning Rate Schedule

| Epoch | Head LR | LM LR | Event |
|------:|--------:|------:|:------|
| 0 | 5.00e-05 | 2.50e-06 | Initial (LM frozen) |
| 4 | 1.98e-05 | 9.91e-07 | End frozen phase, near LR minimum |
| 5 | 1.75e-05 | 8.75e-07 | LM starts training |
| 9 | 2.20e-06 | 1.10e-07 | LR approaching trough |
| 10 | 5.00e-05 | 2.50e-06 | **CosineAnnealing restart** (T_0=10) |
| 13 | 4.53e-05 | 2.27e-06 | Best epoch |
| 19 | 2.93e-05 | 1.87e-06 | Final epoch |

The LR schedule uses `CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)`. The warm restart at epoch 10 coincides with the transition from top-6 unfreezing, providing a fresh learning rate as more parameters become trainable.

#### Saved Checkpoints

| File | Size | Description |
|------|-----:|:------------|
| `enhanced-epoch=13-spearman=val_spearman=0.386.ckpt` | 590 MB | **Best** (top-1 by val_spearman) |
| `enhanced-epoch=14-spearman=val_spearman=0.376.ckpt` | 590 MB | Top-2 |
| `enhanced-epoch=19-spearman=val_spearman=0.378.ckpt` | 590 MB | Top-3 |
| `last.ckpt` | 590 MB | Final epoch (epoch 19) |

Total checkpoint storage: ~2.3 GB. Each checkpoint contains the full model state dict (RiNALMo mega backbone + all feature heads).

### Comparison with OligoAI Baseline (§7.3)

As required by the project deliverables, direct comparison with OligoAI:

| Metric | Model 05 | OligoAI Paper | Δ | Status |
|--------|----------|---------------|---|--------|
| **Spearman ρ** | 0.366 | ≈ 0.42 | −0.054 (−12.6%) | Below baseline |
| **Enrichment** | 2.61× | ≈ 3× | −0.39 | Below baseline |

The OligoAI baseline values come from the OligoAI research paper (see `docs/reference_paper.pdf`).

> Per the project requirements (§6): "It is acceptable if no approach outperforms OligoAI, as long as the validation protocol is followed, results are reported clearly, and limitations are analyzed thoughtfully."

### Analysis

- The model achieves a test Spearman of **0.366**, which is 12.6% below the OligoAI baseline of ≈0.42. As shown in the training history, the best validation Spearman of **0.386** (epoch 13) drops to 0.366 on the test set — a modest gap suggesting limited overfitting at the best checkpoint.
- The best performance coincides with the **top-9 unfreezing phase** (not full unfreezing), suggesting that exposing all 12 transformer layers at once (epochs 14+) introduces instability that degrades generalization.
- The enrichment factor of **2.61×** shows the model still substantially outperforms random selection (1.0×) for identifying high-efficacy ASOs.
- Predictions are on a standardized (z-score) scale since the target was scaled with `StandardScaler`.
- Training loss continued decreasing through epoch 19 (0.669) while validation loss rose (0.769), confirming that additional epochs would increase overfitting rather than improve performance.

---

## Reproduction Instructions

### Prerequisites

```bash
# Python 3.9+
conda create -n aso_pred python=3.9 -y
conda activate aso_pred

# Core dependencies
pip install -r docker/requirements.txt

# Optional: ViennaRNA for thermodynamic features
conda install -c bioconda viennarna
```

### Pre-compute Features (Optional)

```bash
# RNA structural features (requires ViennaRNA — RNAplfold)
python utils/rna_features.py

# Thermodynamic features (ΔG)
python utils/thermo_features.py
```

Output files:
- `data/processed/structural_features.pkl`
- `data/processed/thermo_features.pkl`

If these files do not exist, the model trains without them (zero-filled fallback).

### Training

```bash
# Default (mega_light config — fits 6GB VRAM)
python models/05_oligoai_stage3_stabilized/train.py

# Specific configuration
python models/05_oligoai_stage3_stabilized/train.py --config mega --batch_size 32

# Full options
python models/05_oligoai_stage3_stabilized/train.py \
    --config giga \
    --batch_size 8 \
    --epochs 50 \
    --grad_accum 4 \
    --loss mixed
```

### Using run.sh / Docker (§7.4)

The project provides shell scripts and a Docker setup for reproducibility:

```bash
# Via run.sh (recommended)
./scripts/run.sh 05

# Via run.sh with extra arguments
./scripts/run.sh 05 --config mega --batch_size 16

# Via Docker
./scripts/build.sh                    # Build image
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    aso-prediction bash scripts/run.sh 05
```

### Post-Training Evaluation (Official Metrics)

To compute the official per-screen Spearman metric matching OligoAI's definition:

```bash
python scripts/evaluate.py --model 05 \
    --checkpoint results/05_oligoai_stage3_stabilized/weights/last.ckpt
```

This overwrites `metrics.json` with per-screen `mean_spearman_corr` values.

### CLI Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--config` | str | `mega_light` | `mega`, `giga`, `mega_light`, `micro_full` | Preset configuration |
| `--batch_size` | int | None | — | Override batch size from config |
| `--epochs` | int | 50 | — | Maximum training epochs |
| `--grad_accum` | int | 2 | — | Gradient accumulation steps |
| `--loss` | str | `mixed` | `mse`, `listnet`, `ranknet`, `mixed`, `ranknet_mixed` | Loss function |

### Output

```
results/05_oligoai_stage3_stabilized/
├── metrics.json          # Train/val/test metrics + config + comparison
├── predictions.csv       # Test set y_true vs y_pred (23,373 rows incl. header)
├── weights/              # PyTorch Lightning checkpoints (~590 MB each)
│   ├── enhanced-epoch=13-spearman=val_spearman=0.386.ckpt  (best)
│   ├── enhanced-epoch=14-spearman=val_spearman=0.376.ckpt  (top-2)
│   ├── enhanced-epoch=19-spearman=val_spearman=0.378.ckpt  (top-3)
│   └── last.ckpt                                            (final epoch)
└── logs/
    └── version_0/
        ├── hparams.yaml  # Actual hyperparameters used
        └── metrics.csv   # Per-step training logs (2,421 rows)
```

---

## File Reference

| File | Description |
|------|-------------|
| `models/05_oligoai_stage3_stabilized/train.py` | Main training script with CLI, data loading, training loop, evaluation |
| `utils/enhanced_model.py` | `EnhancedOligoAI` model (671 lines): all neural network components |
| `utils/enhanced_dataset.py` | `EnhancedOligoAIDataset` + `enhanced_collate_fn`: feature extraction & batching |
| `utils/dataset.py` | `ChemistryTokenizer`, `create_patent_split`: tokenization & data splitting |
| `utils/model.py` | Base `OligoAI` model (used by Model 03, shares some components) |
| `utils/rna_features.py` | RNAplfold structural feature extraction (parallel processing) |
| `utils/thermo_features.py` | Nearest-neighbor ΔG computation (standalone, no external deps) |
| `utils/rinalmo/` | Vendored RiNALMo language model (config, pretrained weights, transformer) |
| `results/05_oligoai_stage3_stabilized/metrics.json` | Evaluation results |
| `results/05_oligoai_stage3_stabilized/predictions.csv` | Test predictions |
| `results/05_oligoai_stage3_stabilized/logs/version_0/hparams.yaml` | Actual hyperparameters used in training |
| `results/05_oligoai_stage3_stabilized/logs/version_0/metrics.csv` | Per-step training logs (2,421 rows) |

---

## Known Limitations & Future Work

1. **Below OligoAI baseline**: The 0.366 Spearman (vs. ≈0.42 baseline) suggests that adding more features without precise hyperparameter tuning can hurt rather than help. The training history shows the model plateaued at val_spearman ≈ 0.386 (epoch 13), and full unfreezing (epochs 14+) introduced volatility without further improvement. Per requirements §6, negative results are still considered valuable outcomes.

2. **Spearman metric discrepancy**: The `train.py` script computes **global Spearman** during training/validation, not the per-screen mean Spearman required by the OligoAI protocol (§5.2). Use `scripts/evaluate.py` for the official per-screen metric. This means the reported `val_spearman` values in the training history table are an approximation of the true per-screen metric.

3. **Loss function choice**: The actual training used `ranknet_mixed` (MSE + RankNet), not the CLI default `mixed` (MSE + ListNet). This is recorded in `hparams.yaml`. The pairwise RankNet loss optimizes for ranking quality, which is directly relevant to the Spearman metric.

4. **Standard Scaler on targets**: Predictions are on a z-score scale. The ensemble model (Model 06) avoids this by predicting raw inhibition percentages.

5. **Missing structural/thermo features**: If `structural_features.pkl` or `thermo_features.pkl` are absent, the model falls back to zero vectors, which means several feature branches contribute no signal.

6. **Overfitting in late epochs**: Training loss continued decreasing (0.765 → 0.669) while validation loss rose after epoch 13 (0.746 → 0.769 at epoch 19). The `GradualUnfreezeCallback` correctly exposed more layers over time, but the full unfreezing phase (epochs 14+) coincided with worsening generalization.

7. **Potential improvements**:
   - Screen-level Spearman evaluation during training (currently uses global Spearman)
   - Earlier stopping (the best epoch was 13, but training ran to 20)
   - More conservative unfreezing — the best result was in the top-9 phase, suggesting top-12 opens too many parameters
   - Hyperparameter search with Optuna across config presets
   - Data augmentation (reverse complement, noise injection)
   - Attention-weighted pooling instead of masked mean pooling
   - Pre-train the head on Model 06 ensemble predictions as pseudo-labels
