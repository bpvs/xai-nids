# XAI for Network Intrusion Detection

Proof-of-concept applying Explainable AI techniques to a 1D-CNN trained on the NSL-KDD dataset. Part of PhD research on cross-domain XAI evaluation.

## What it does

1. Downloads & preprocesses NSL-KDD (normal vs. attack, binary classification)
2. Trains a 1D-CNN classifier
3. Generates **SHAP**, **LIME**, and **Counterfactual** explanations
4. Evaluates explanation quality (fidelity, SHAP–LIME consistency)
5. Saves all plots to `results/`

## Quick start

```bash
pip install -r requirements.txt
python xai_pipeline.py
```

Or use the notebook:

```bash
jupyter notebook walkthrough.ipynb
```

## Dataset

**NSL-KDD** — 41 network traffic features, ~125K train / ~22K test records. Auto-downloaded on first run.

## References

- SHAP: Lundberg & Lee (2017) — [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- LIME: Ribeiro et al. (2016) — [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)
