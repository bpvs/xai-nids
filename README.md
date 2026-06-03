# XAI for Network Intrusion Detection — Multi-Model

Proof-of-concept comparing 5 XAI techniques across 3 model architectures on the NSL-KDD dataset. Part of PhD research on cross-domain XAI evaluation.

## What it does

Trains **three models** on NSL-KDD intrusion detection and applies XAI methods to each:

| Method | CNN (1D) | XGBoost | Random Forest |
|---|---|---|---|
| SHAP | ✓ (Kernel) | ✓ (Tree) | ✓ (Tree) |
| LIME | ✓ | ✓ | ✓ |
| Counterfactuals | ✓ (TF gradients) | ✓ (numerical) | ✓ (numerical) |
| Saliency | ✓ | — | — |
| Grad-CAM (1D) | ✓ | — | — |

Each model gets its own results folder so you can compare them directly. A cross-model comparison plot is generated at the end.

## Quick start

```bash
pip install -r requirements.txt
python xai_pipeline.py            # full run
python xai_pipeline.py --fast     # fast dev mode
```

## Project structure

```
xai-nids/
├── xai_pipeline.py           # main orchestrator
├── data_utils.py             # NSL-KDD loading + preprocessing
├── explainers.py             # all 5 XAI methods
├── evaluation.py             # fidelity, consistency, cross-model comparison
├── models/
│   ├── base.py               # ModelWrapper abstract base
│   ├── cnn.py                # 1D-CNN (Keras/TensorFlow)
│   ├── xgboost_model.py      # XGBoost classifier
│   └── rf.py                 # Random Forest
├── results_cnn/              # CNN outputs (auto-created)
├── results_xgboost/          # XGBoost outputs (auto-created)
├── results_rf/               # Random Forest outputs (auto-created)
├── cross_model_comparison.png
└── report.json
```

## Apple Silicon optimizations

- **CNN**: Metal GPU via `tensorflow-metal`, memory growth enabled, `@tf.function`-compiled prediction, GradientTape for counterfactuals (~40× faster than finite differences)
- **XGBoost**: `tree_method='hist'` + `n_jobs=-1` for all M1 cores
- **Random Forest**: `n_jobs=-1` for parallel tree building
- **SHAP**: uses `TreeExplainer` (exact, fast) for tree models and `KernelExplainer` for the CNN

## References

- SHAP: Lundberg & Lee (2017) — [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- LIME: Ribeiro et al. (2016) — [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)
- Grad-CAM: Selvaraju et al. (2017) — [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- NSL-KDD: Tavallaee et al. (2009)