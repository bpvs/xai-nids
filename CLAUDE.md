# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase comparing 5 XAI (explainability) techniques across 3 model architectures (CNN, XGBoost, Random Forest) on the NSL-KDD network intrusion detection dataset. Part of PhD research on cross-domain XAI evaluation.

There are **two operational modes**:
- **Baseline pipeline** (`src/xai_pipeline.py`): Single-env, monolithic, full run
- **Staged pipeline** (`src/stages/`): Env-separated stages with FalkorDB as results store

## Environments

Two isolated conda environments are required because ydata-profiling and TensorFlow/SHAP/XGBoost have irreconcilable dependency conflicts.

| Env | Python | Use for |
|-----|--------|---------|
| `xai-nids` | 3.10 | Model training, XAI methods (TF, SHAP, LIME) |
| `dataprep` | 3.11 | Data cleaning, profiling, graph loading, analysis |

Stages hand off via files (`.parquet`, `.json`) — the two envs never share a process.

```bash
conda create -n xai-nids python=3.10 && conda activate xai-nids && pip install -r requirements-xai.txt
conda create -n dataprep python=3.11 && conda activate dataprep && pip install -r requirements-dataprep.txt
```

## Commands

### Baseline pipeline (xai-nids env)

```bash
python src/xai_pipeline.py            # full run (~10-30 min)
python src/xai_pipeline.py --fast     # dev run (~5 min, fewer samples + epochs)
```

Outputs: `results_cnn/`, `results_xgboost/`, `results_rf/`, `cross_model_comparison.png`, `report.json`

### Staged pipeline

Run each stage with `conda run -n <env>`:

```bash
# Stage 1 — profile raw data (dataprep)
conda run -n dataprep python src/stages/profile_before.py --data-dir data --out reports/profile_before.html

# Stage 2 — clean: drop zero-variance + ≥95%-zero columns, 41 → 26 features (dataprep)
conda run -n dataprep python src/stages/clean.py \
    --data-dir data --drop-zero-frac 0.95 \
    --out data/processed/clean.parquet \
    --dropped data/processed/dropped_columns.json

# Stage 3 — train 3 models + run 5 XAI methods on 100 deterministic samples (xai-nids)
conda run -n xai-nids python src/stages/train_explain.py \
    --clean data/processed/clean.parquet \
    --manifest data/processed/sample_manifest.json \
    --results data/processed/results.json \
    --attributions data/processed/attributions.json \
    --n-samples 100
# Add --fast for dev runs

# Stage 4 — load results into FalkorDB graph (dataprep); requires Docker
docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest
conda run -n dataprep python src/stages/load_graph.py \
    --results data/processed/results.json \
    --attributions data/processed/attributions.json \
    --graph-name xai_results

# Stage 5 — Jaccard@k + Spearman rank-correlation heatmaps (dataprep)
conda run -n dataprep python src/stages/analyze_xai.py \
    --attributions data/processed/attributions.json \
    --out-dir reports --slice all
```

Slice options: `all`, `attack`, `normal`.

### MLflow UI

```bash
conda run -n dataprep mlflow ui   # http://localhost:5000
```

Experiment name: `xai-nids-graph-augmentation`

## Architecture

### Core abstractions

**`src/models/base.py`** — `ModelWrapper` abstract base class. All models implement:
- `build(n_features)`, `fit(X_train, y_train, X_val, y_val, fast_mode)`, `predict_proba(X)`, `evaluate(X_test, y_test)`
- `supports_gradients` property (True only for CNN)

**`src/explainers.py`** — All 5 XAI methods with uniform signature `(model_wrapper, X_bg, X_explain, feat_names, fast_mode)`:
- SHAP: TreeExplainer for tree models (exact), KernelExplainer for CNN (model-agnostic)
- LIME: LimeTabularExplainer
- Counterfactuals: TF GradientTape for CNN, finite-diff numerical gradients for trees
- Saliency maps + Grad-CAM: CNN only (require `supports_gradients`)

All explainers invoke only `model_wrapper.predict_proba()` — fully model-agnostic.

**`src/data_utils.py`** — `load_data()` downloads NSL-KDD, encodes categoricals, MinMax-scales, and returns stratified train/val/test splits. Binary labels: 0=normal, 1=attack.

**`src/evaluation.py`** — Fidelity (top-5 feature masking), pairwise Jaccard@10 consistency, cross-model metric heatmaps.

### Staged pipeline data flow

```
raw NSL-KDD (41 features)
    → clean.py (dataprep)  → clean.parquet (26 features)
    → train_explain.py (xai-nids) → results.json + attributions.json
    → load_graph.py (dataprep) → FalkorDB graph (xai_results)
    → analyze_xai.py (dataprep) → Jaccard/Spearman heatmaps + analysis_summary.json
```

FalkorDB schema: `Model`, `Sample`, `Method`, `Feature` nodes; `ASSIGNS` edges with importance values. Browser UI at `http://localhost:3000`.

## Critical Dependency Pins

These pins are non-negotiable — changing them causes hard crashes:

- **`xgboost==1.7.6` + `shap==0.49.1`**: XGBoost 3.x serializes `base_score` as a bracketed array string that SHAP's TreeExplainer cannot parse.
- **`numpy<2`**: TensorFlow-Metal + shap + xgboost 1.7.6 all require numpy 1.x.
- **`setuptools<81`**: ydata-profiling imports `pkg_resources`, removed in setuptools 81.
- **`tensorflow-metal`**: Apple Silicon GPU; CPU fallback is automatic when absent.

## No Test Suite

This is a research codebase — there are no automated tests. Correctness is validated by comparing output plots and metrics across runs.
