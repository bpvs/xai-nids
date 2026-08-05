"""
XAI explanation methods.

Each method takes a ModelWrapper instance and works via its predict_proba()
interface, plus model-specific methods (get_input_gradients, get_conv_grad_model)
for CNN-only methods (saliency, Grad-CAM).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular
import tensorflow as tf
from tensorflow.keras import layers


# ═══════════════════════════════════════════════════════════════════════════
# SHAP
# ═══════════════════════════════════════════════════════════════════════════
def run_shap(model_wrapper, X_bg, X_explain, feat_names, fast_mode=False):
    """SHAP explanations. Uses TreeExplainer for tree models (exact + fast)
    and KernelExplainer for neural networks."""
    print(f"\n── SHAP [{model_wrapper.name}] ──")
    results_dir = model_wrapper.results_dir

    # Pick the right explainer per model type
    if hasattr(model_wrapper.model, "estimators_") or "xgb" in type(model_wrapper.model).__module__.lower():
        # Tree model — use exact TreeExplainer
        print("  Using TreeExplainer (exact, fast)")
        explainer = shap.TreeExplainer(model_wrapper.model)
        sv = explainer.shap_values(X_explain)
        # For sklearn binary classifiers, TreeExplainer returns list of length 2
        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]  # positive class (attack)
        # Newer SHAP versions return shape (n, features, 2) for binary
        elif sv.ndim == 3 and sv.shape[-1] == 2:
            sv = sv[..., 1]
    else:
        # Neural net — use KernelExplainer
        print("  Using KernelExplainer (model-agnostic, slower)")
        n_bg = 50 if fast_mode else 100
        n_samples = 200 if fast_mode else 500
        bg = X_bg[np.random.choice(len(X_bg), n_bg, replace=False)]
        explainer = shap.KernelExplainer(lambda x: model_wrapper.predict_proba(x), bg)
        sv = explainer.shap_values(X_explain, nsamples=n_samples)

    # Summary plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X_explain, feature_names=feat_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Per-sample plots
    for i in range(len(X_explain)):
        idx = np.argsort(np.abs(sv[i]))[::-1][:12]
        plt.figure(figsize=(7, 4))
        plt.barh(range(len(idx)), sv[i][idx],
                 color=["#e74c3c" if v > 0 else "#3498db" for v in sv[i][idx]])
        plt.yticks(range(len(idx)), [feat_names[j] for j in idx])
        plt.xlabel("SHAP value"); plt.title(f"SHAP — Sample {i} [{model_wrapper.name}]")
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig(f"{results_dir}/shap_sample_{i}.png", dpi=150)
        plt.close()

    print(f"  Saved SHAP plots for {len(X_explain)} samples")
    return sv


# ═══════════════════════════════════════════════════════════════════════════
# LIME
# ═══════════════════════════════════════════════════════════════════════════
def run_lime(model_wrapper, X_train, X_explain, feat_names, fast_mode=False):
    """LIME explanations — works with any model via predict_proba."""
    print(f"\n── LIME [{model_wrapper.name}] ──")
    results_dir = model_wrapper.results_dir
    n_samples = 500 if fast_mode else 1000

    def predict_proba_2d(X):
        p = model_wrapper.predict_proba(X)
        return np.column_stack([1 - p, p])

    exp = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feat_names,
        class_names=["Normal", "Attack"],
        mode="classification", random_state=42,
    )

    lime_weights = []
    for i in range(len(X_explain)):
        explanation = exp.explain_instance(
            X_explain[i], predict_proba_2d,
            num_features=15, num_samples=n_samples,
        )
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(7, 4)
        plt.title(f"LIME — Sample {i} [{model_wrapper.name}]")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/lime_sample_{i}.png", dpi=150, bbox_inches="tight")
        plt.close()

        weights = {feat_names[fi]: w for fi, w in explanation.local_exp[1]}
        lime_weights.append(weights)

    print(f"  Saved LIME plots for {len(X_explain)} samples")
    return lime_weights


# ═══════════════════════════════════════════════════════════════════════════
# COUNTERFACTUALS
# ═══════════════════════════════════════════════════════════════════════════
def run_counterfactuals(model_wrapper, X_explain, feat_names, fast_mode=False):
    """Counterfactual explanations.

    For CNN: uses TensorFlow GradientTape (GPU-accelerated, exact gradients).
    For tree models: uses finite-difference numerical gradients.
    """
    print(f"\n── Counterfactuals [{model_wrapper.name}] ──")
    results_dir = model_wrapper.results_dir
    max_iter = 100 if fast_mode else 500
    lr = 0.01

    results = []
    for i in range(len(X_explain)):
        x_orig = X_explain[i].copy().astype(np.float32)
        orig_pred = float(model_wrapper.predict_proba(x_orig.reshape(1, -1))[0])
        target = 0 if orig_pred > 0.5 else 1
        success = False

        if model_wrapper.supports_gradients:
            # CNN path: TF GradientTape
            x_cf = tf.Variable(x_orig.reshape(1, -1, 1), dtype=tf.float32)
            x_orig_tf = tf.constant(x_orig.reshape(1, -1, 1), dtype=tf.float32)

            for it in range(max_iter):
                with tf.GradientTape() as tape:
                    pred = model_wrapper.model(x_cf, training=False)
                    loss = (-pred if target == 1 else pred)
                    loss = loss + 0.05 * tf.reduce_sum(tf.abs(x_cf - x_orig_tf))
                grad = tape.gradient(loss, x_cf)
                x_cf.assign_sub(lr * grad)
                x_cf.assign(tf.clip_by_value(x_cf, 0.0, 1.0))

                current = float(model_wrapper.model(x_cf, training=False).numpy().flatten()[0])
                if (target == 1 and current > 0.5) or (target == 0 and current < 0.5):
                    success = True
                    break
            cf_np = x_cf.numpy().flatten()
        else:
            # Tree path: finite differences
            x = x_orig.copy().astype(np.float64)
            for it in range(max_iter):
                pred = model_wrapper.predict_proba(x.reshape(1, -1))[0]
                if (target == 1 and pred > 0.5) or (target == 0 and pred < 0.5):
                    success = True
                    break
                # Vectorized finite-difference gradient
                eps = 1e-3
                X_plus = np.tile(x, (len(x), 1))
                X_plus[np.arange(len(x)), np.arange(len(x))] += eps
                X_minus = np.tile(x, (len(x), 1))
                X_minus[np.arange(len(x)), np.arange(len(x))] -= eps
                p_plus = model_wrapper.predict_proba(X_plus)
                p_minus = model_wrapper.predict_proba(X_minus)
                grad = (p_plus - p_minus) / (2 * eps)
                x -= lr * ((-grad if target == 1 else grad) + 0.05 * np.sign(x - x_orig))
                x = np.clip(x, 0, 1)
            cf_np = x

        cf_pred = float(model_wrapper.predict_proba(cf_np.reshape(1, -1))[0])
        delta = cf_np - x_orig
        changes = {feat_names[j]: float(delta[j]) for j in range(len(delta)) if abs(delta[j]) > 1e-6}

        # Plot
        if changes:
            top = sorted(changes.items(), key=lambda t: abs(t[1]), reverse=True)[:12]
            plt.figure(figsize=(7, 4))
            plt.barh(range(len(top)), [v for _, v in top],
                     color=["#e74c3c" if v > 0 else "#3498db" for _, v in top])
            plt.yticks(range(len(top)), [k for k, _ in top])
            lbl = lambda p: "Attack" if p > 0.5 else "Normal"
            plt.xlabel("Feature change")
            plt.title(f"Counterfactual — Sample {i} [{model_wrapper.name}]\n"
                      f"({lbl(orig_pred)}→{lbl(cf_pred)})")
            plt.gca().invert_yaxis(); plt.tight_layout()
            plt.savefig(f"{results_dir}/cf_sample_{i}.png", dpi=150)
            plt.close()

        status = "flipped" if success else "failed"
        print(f"  Sample {i}: {status} in {it+1} iters")
        results.append({
            "success": success, "changes": changes,
            "orig": orig_pred, "cf": cf_pred,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SALIENCY (CNN only)
# ═══════════════════════════════════════════════════════════════════════════
def run_saliency(model_wrapper, X_explain, feat_names):
    """Vanilla gradient saliency — CNN only."""
    if not model_wrapper.supports_gradients:
        print(f"\n── Saliency [{model_wrapper.name}] ── SKIPPED (gradient-based, not applicable)")
        return None

    print(f"\n── Saliency [{model_wrapper.name}] ──")
    results_dir = model_wrapper.results_dir
    grads = model_wrapper.get_input_gradients(X_explain)  # (n_samples, n_features)

    for i in range(len(X_explain)):
        abs_g = np.abs(grads[i])
        idx = np.argsort(abs_g)[::-1][:12]
        plt.figure(figsize=(7, 4))
        plt.barh(range(len(idx)), abs_g[idx], color="#8e44ad")
        plt.yticks(range(len(idx)), [feat_names[j] for j in idx])
        plt.xlabel("|Gradient|"); plt.title(f"Saliency — Sample {i} [{model_wrapper.name}]")
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig(f"{results_dir}/saliency_sample_{i}.png", dpi=150)
        plt.close()

    print(f"  Saved saliency plots for {len(X_explain)} samples")
    return grads


# ═══════════════════════════════════════════════════════════════════════════
# GRAD-CAM (CNN only)
# ═══════════════════════════════════════════════════════════════════════════
def run_gradcam_1d(model_wrapper, X_explain, feat_names):
    """1D Grad-CAM — CNN only."""
    if not model_wrapper.supports_gradients:
        print(f"\n── Grad-CAM [{model_wrapper.name}] ── SKIPPED (CNN-only method)")
        return None

    print(f"\n── 1D Grad-CAM [{model_wrapper.name}] ──")
    results_dir = model_wrapper.results_dir

    result = model_wrapper.get_conv_grad_model()
    if result is None:
        print("  [!] No Conv1D layer found — skipping")
        return None
    grad_model, layer_name = result
    print(f"  Target layer: {layer_name}")

    all_heatmaps = []
    for i in range(len(X_explain)):
        x_tensor = tf.constant(X_explain[i].reshape(1, -1, 1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(x_tensor, training=False)
            target_score = pred[0, 0]
        grads = tape.gradient(target_score, conv_out)
        weights = tf.reduce_mean(grads, axis=1)
        conv_out_np = conv_out.numpy()[0]
        weights_np = weights.numpy()[0]
        heatmap = np.maximum(np.dot(conv_out_np, weights_np), 0)

        # Resize back to feature length
        if len(heatmap) != len(feat_names):
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(heatmap))
            x_new = np.linspace(0, 1, len(feat_names))
            heatmap = interp1d(x_old, heatmap, kind="linear")(x_new)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        all_heatmaps.append(heatmap)

        idx = np.argsort(heatmap)[::-1][:12]
        plt.figure(figsize=(7, 4))
        plt.barh(range(len(idx)), heatmap[idx], color="#e67e22")
        plt.yticks(range(len(idx)), [feat_names[j] for j in idx])
        plt.xlabel("Grad-CAM activation")
        plt.title(f"1D Grad-CAM — Sample {i} [{model_wrapper.name}]")
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig(f"{results_dir}/gradcam_sample_{i}.png", dpi=150)
        plt.close()

    print(f"  Saved Grad-CAM plots for {len(X_explain)} samples")
    return np.array(all_heatmaps)
