"""
XAI for Network Intrusion Detection — Multi-Model Pipeline
===========================================================
Trains three models (CNN, XGBoost, Random Forest) on NSL-KDD,
then applies XAI methods to each and compares them.

Universal methods (run on all 3 models):  SHAP, LIME, Counterfactuals
CNN-only methods:                          Saliency, 1D Grad-CAM

Usage:
    python xai_pipeline.py              # full run
    python xai_pipeline.py --fast       # fast mode (dev/debugging)
"""

import os, sys, json, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from data_utils import load_data
from models import CNNModel, XGBoostModel, RandomForestModel, configure_gpu
from explainers import (
    run_shap, run_lime, run_counterfactuals,
    run_saliency, run_gradcam_1d,
)
from evaluation import evaluate_explanations, cross_model_comparison


FAST_MODE = "--fast" in sys.argv
np.random.seed(42)


def run_pipeline_for_model(model, X_tr, y_tr, X_v, y_v, X_te, y_te,
                            X_explain, feat_names, fast_mode=False):
    """Train one model and run all applicable XAI methods on it."""
    print(f"\n{'═' * 60}")
    print(f"  Pipeline for {model.name.upper()}")
    print(f"{'═' * 60}")

    # Build + train
    model.build(X_tr.shape[1])
    model.fit(X_tr, y_tr, X_v, y_v, fast_mode=fast_mode)

    # Evaluate model on test set
    metrics = model.evaluate(X_te, y_te)

    # XAI methods — universal
    shap_vals = run_shap(model, X_tr, X_explain, feat_names, fast_mode=fast_mode)
    lime_weights = run_lime(model, X_tr, X_explain, feat_names, fast_mode=fast_mode)
    cf_results = run_counterfactuals(model, X_explain, feat_names, fast_mode=fast_mode)

    # XAI methods — CNN only
    saliency = run_saliency(model, X_explain, feat_names)
    gradcam = run_gradcam_1d(model, X_explain, feat_names)

    # Evaluate explanations
    eval_results = evaluate_explanations(
        model, X_explain, feat_names,
        shap_vals=shap_vals, lime_weights=lime_weights,
        saliency_grads=saliency, gradcam_maps=gradcam,
    )

    return {
        "metrics": {k: v for k, v in metrics.items() if k not in ["y_pred", "y_prob"]},
        "evaluation": eval_results,
        "counterfactuals": {
            "success_rate": sum(r["success"] for r in cf_results) / len(cf_results)
        },
    }


def main():
    print("=" * 60)
    print("XAI for Network Intrusion Detection — Multi-Model Pipeline")
    print("=" * 60)
    if FAST_MODE:
        print("⚡ FAST MODE enabled")

    # GPU/Metal setup (affects CNN only)
    configure_gpu()

    # Load data (shared across all models)
    print("\n[1/4] Loading data...")
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = load_data()

    # Pick samples to explain — SAME indices across all models for fair comparison
    print("\n[2/4] Selecting explanation samples...")
    # Train a quick reference predictor to find correctly-classified samples
    # We just pick by ground-truth here so the selection is model-agnostic.
    norm_idx = np.where(y_te == 0)[0]
    att_idx = np.where(y_te == 1)[0]
    sel = np.concatenate([
        np.random.choice(norm_idx, 2, replace=False),
        np.random.choice(att_idx, 3, replace=False),
    ])
    X_explain = X_te[sel]
    print(f"  {len(sel)} samples (2 normal, 3 attack) — same across all models")

    # Run pipeline for each model
    print("\n[3/4] Training and explaining all models...")
    all_results = {}
    all_metrics = {}
    all_evaluations = {}

    for ModelClass in [CNNModel, XGBoostModel, RandomForestModel]:
        model = ModelClass()
        result = run_pipeline_for_model(
            model, X_tr, y_tr, X_v, y_v, X_te, y_te,
            X_explain, feats, fast_mode=FAST_MODE,
        )
        all_results[model.name] = result
        all_metrics[model.name] = result["metrics"]
        all_evaluations[model.name] = result["evaluation"]

    # Cross-model comparison
    print("\n[4/4] Cross-model comparison...")
    cross_model_comparison(
        all_evaluations, all_metrics,
        output_path="cross_model_comparison.png",
    )

    # Save consolidated report
    report = {
        "fast_mode": FAST_MODE,
        "models": all_results,
        "summary": {
            "best_accuracy": max(all_metrics.items(), key=lambda x: x[1]["accuracy"])[0],
            "best_f1": max(all_metrics.items(), key=lambda x: x[1]["f1"])[0],
            "best_auc": max(all_metrics.items(), key=lambda x: x[1]["auc"])[0],
        },
    }
    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Results saved to:")
    print(f"  - results_cnn/")
    print(f"  - results_xgboost/")
    print(f"  - results_rf/")
    print(f"  - cross_model_comparison.png")
    print(f"  - report.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()