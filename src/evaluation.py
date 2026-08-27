"""
Evaluation metrics for XAI explanations.

Two main metrics:
- Fidelity: does masking the top-K features change the prediction?
- Consistency: do different methods agree on which features matter? (Jaccard@10)

Plus a cross-model comparison at the end.
"""

import os
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


METHOD_COLORS = {
    "SHAP": "#2ecc71",
    "LIME": "#e74c3c",
    "Saliency": "#8e44ad",
    "Grad-CAM": "#e67e22",
    "Counterfactual": "#3498db",
}


def _build_importance_dicts(method, idx, shap_vals, lime_weights,
                             saliency_grads, gradcam_maps, feat_names):
    """Build per-sample importance dict for the given method."""
    if method == "SHAP" and shap_vals is not None:
        return dict(zip(feat_names, shap_vals[idx]))
    if method == "LIME" and lime_weights is not None:
        return lime_weights[idx]
    if method == "Saliency" and saliency_grads is not None:
        return dict(zip(feat_names, np.abs(saliency_grads[idx])))
    if method == "Grad-CAM" and gradcam_maps is not None:
        return dict(zip(feat_names, gradcam_maps[idx]))
    return None


def evaluate_explanations(model_wrapper, X_explain, feat_names,
                          shap_vals=None, lime_weights=None,
                          saliency_grads=None, gradcam_maps=None):
    """Per-method fidelity + pairwise consistency for one model."""
    print(f"\n── Evaluation [{model_wrapper.name}] ──")
    results_dir = model_wrapper.results_dir

    methods = []
    if shap_vals is not None: methods.append("SHAP")
    if lime_weights is not None: methods.append("LIME")
    if saliency_grads is not None: methods.append("Saliency")
    if gradcam_maps is not None: methods.append("Grad-CAM")

    # Fidelity
    fidelity = {m: [] for m in methods}
    for i in range(len(X_explain)):
        x = X_explain[i].reshape(1, -1)
        orig = model_wrapper.predict_proba(x)[0]
        for m in methods:
            imp = _build_importance_dicts(m, i, shap_vals, lime_weights,
                                          saliency_grads, gradcam_maps, feat_names)
            if imp is None:
                continue
            top5 = sorted(imp, key=lambda f: abs(imp[f]), reverse=True)[:5]
            x_mask = x.copy()
            x_mask[0, [feat_names.index(f) for f in top5]] = 0
            fid = abs(orig - model_wrapper.predict_proba(x_mask)[0])
            fidelity[m].append(float(fid))

    for m in methods:
        print(f"  {m:10s} fidelity: {np.mean(fidelity[m]):.4f} ± {np.std(fidelity[m]):.4f}")

    # Pairwise consistency
    pair_consistency = {}
    for m1, m2 in combinations(methods, 2):
        scores = []
        for i in range(len(X_explain)):
            imp1 = _build_importance_dicts(m1, i, shap_vals, lime_weights,
                                           saliency_grads, gradcam_maps, feat_names)
            imp2 = _build_importance_dicts(m2, i, shap_vals, lime_weights,
                                           saliency_grads, gradcam_maps, feat_names)
            if imp1 is None or imp2 is None:
                continue
            top1 = set(sorted(imp1, key=lambda f: abs(imp1[f]), reverse=True)[:10])
            top2 = set(sorted(imp2, key=lambda f: abs(imp2[f]), reverse=True)[:10])
            scores.append(len(top1 & top2) / len(top1 | top2) if (top1 | top2) else 0)
        pair_consistency[f"{m1}-{m2}"] = scores
        print(f"  {m1}-{m2} consistency: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    means = [np.mean(fidelity[m]) for m in methods]
    stds = [np.std(fidelity[m]) for m in methods]
    a1.bar(methods, means, yerr=stds,
           color=[METHOD_COLORS.get(m, "#999") for m in methods], capsize=5, alpha=0.85)
    a1.set_title(f"Mean Fidelity — {model_wrapper.name.upper()}")
    a1.set_ylabel("Prediction change (higher = better)")
    a1.grid(alpha=0.3)

    n = len(methods)
    matrix = np.eye(n)
    for pair_name, scores in pair_consistency.items():
        m1, m2 = next((a, b) for a, b in combinations(methods, 2)
                      if f"{a}-{b}" == pair_name)
        i1, i2 = methods.index(m1), methods.index(m2)
        matrix[i1, i2] = matrix[i2, i1] = np.mean(scores)
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=methods, yticklabels=methods,
                cmap="YlGnBu", vmin=0, vmax=1, ax=a2)
    a2.set_title(f"Pairwise Consistency — {model_wrapper.name.upper()}")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/evaluation.png", dpi=150)
    plt.close()

    return {
        "fidelity": {m: float(np.mean(fidelity[m])) for m in methods},
        "consistency": {k: float(np.mean(v)) for k, v in pair_consistency.items()},
    }


def cross_model_comparison(all_evaluations, all_metrics, output_path="cross_model_comparison.png"):
    """
    Side-by-side comparison across all three models.
    `all_evaluations` is a dict of model_name → evaluation dict.
    `all_metrics` is a dict of model_name → {accuracy, f1, auc}.
    """
    print("\n── Cross-model comparison ──")

    models = list(all_evaluations.keys())
    universal_methods = ["SHAP", "LIME"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Model performance
    metrics_to_show = ["accuracy", "f1", "auc"]
    x = np.arange(len(models))
    width = 0.25
    for i, metric in enumerate(metrics_to_show):
        vals = [all_metrics[m][metric] for m in models]
        axes[0].bar(x + i * width, vals, width, label=metric.upper(), alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([m.upper() for m in models])
    axes[0].set_title("Model Performance")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Fidelity comparison for universal methods
    for i, method in enumerate(universal_methods):
        vals = [all_evaluations[m]["fidelity"].get(method, 0) for m in models]
        axes[1].bar(x + i * width, vals, width, label=method,
                    color=METHOD_COLORS.get(method, "#999"), alpha=0.85)
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels([m.upper() for m in models])
    axes[1].set_title("XAI Fidelity (universal methods)")
    axes[1].set_ylabel("Prediction change")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. SHAP-LIME consistency across models
    consist_vals = [all_evaluations[m]["consistency"].get("SHAP-LIME", 0) for m in models]
    axes[2].bar([m.upper() for m in models], consist_vals,
                color=["#3498db", "#27ae60", "#16a085"], alpha=0.85)
    axes[2].set_title("SHAP–LIME Consistency Across Models")
    axes[2].set_ylabel("Jaccard@10")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)

    plt.suptitle("Cross-Model XAI Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved to {output_path}")