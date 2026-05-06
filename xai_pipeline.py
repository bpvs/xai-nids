"""
XAI for Network Intrusion Detection — Compact Pipeline
=======================================================
Train a 1D-CNN on NSL-KDD, then explain predictions with SHAP, LIME,
Saliency Maps, 1D Grad-CAM, and Counterfactuals. Evaluate explanation quality.

Usage:
    python xai_pipeline.py              # full run
    python xai_pipeline.py --fast       # fast mode (dev/debugging)
"""

import os, sys, json, warnings, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import shap
import lime.lime_tabular

# ── GPU / Metal configuration ───────────────────────────────────────────────
def configure_gpu():
    """Detect and configure GPU (Apple Metal, CUDA, or CPU fallback)."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU detected: {gpus[0].name}")
        # Set TF to use float32 globally (Metal works best with float32)
        tf.keras.backend.set_floatx("float32")
    else:
        print("  No GPU detected — running on CPU")
    # Enable XLA JIT compilation for faster TF ops
    #tf.config.optimizer.set_jit(True)
    return len(gpus) > 0

HAS_GPU = configure_gpu()

# ── Fast mode ───────────────────────────────────────────────────────────────
FAST_MODE = "--fast" in sys.argv
if FAST_MODE:
    print("  ⚡ FAST MODE enabled (reduced iterations for dev/debugging)")

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
np.random.seed(42)

# ── Column definitions ──────────────────────────────────────────────────────
COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label","difficulty_level",
]

ATTACKS = {
    "normal":"normal",
    "back":"DoS","land":"DoS","neptune":"DoS","pod":"DoS","smurf":"DoS",
    "teardrop":"DoS","mailbomb":"DoS","apache2":"DoS","processtable":"DoS","udpstorm":"DoS",
    "ipsweep":"Probe","nmap":"Probe","portsweep":"Probe","satan":"Probe","mscan":"Probe","saint":"Probe",
    "ftp_write":"R2L","guess_passwd":"R2L","imap":"R2L","multihop":"R2L","phf":"R2L",
    "spy":"R2L","warezclient":"R2L","warezmaster":"R2L","snmpgetattack":"R2L","named":"R2L",
    "xlock":"R2L","xsnoop":"R2L","sendmail":"R2L","httptunnel":"R2L","worm":"R2L","snmpguess":"R2L",
    "buffer_overflow":"U2R","loadmodule":"U2R","perl":"U2R","rootkit":"U2R",
    "xterm":"U2R","ps":"U2R","sqlattack":"U2R",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_data(data_dir="data"):
    """Download NSL-KDD if needed, return preprocessed X/y splits."""
    os.makedirs(data_dir, exist_ok=True)
    urls = {
        "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "KDDTest+.txt":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    }
    for fname, url in urls.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"  Downloading {fname}...")
            r = requests.get(url, timeout=60); r.raise_for_status()
            open(path, "w").write(r.text)

    train = pd.read_csv(f"{data_dir}/KDDTrain+.txt", names=COLS, header=None).drop("difficulty_level", axis=1)
    test  = pd.read_csv(f"{data_dir}/KDDTest+.txt",  names=COLS, header=None).drop("difficulty_level", axis=1)

    # Encode categoricals
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]).unique())
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])

    # Binary labels
    y_train = (train["label"].map(lambda x: ATTACKS.get(x,"unknown")) != "normal").astype(int).values
    y_test  = (test["label"].map(lambda x: ATTACKS.get(x,"unknown"))  != "normal").astype(int).values

    feat_cols = [c for c in train.columns if c != "label"]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train[feat_cols].values.astype(np.float32))
    X_test  = scaler.transform(test[feat_cols].values.astype(np.float32))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feat_cols


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL
# ═══════════════════════════════════════════════════════════════════════════
def build_cnn(n_features):
    """1D-CNN: 3 conv blocks → dense classifier."""
    model = keras.Sequential([
        layers.Conv1D(64,  3, padding="same", input_shape=(n_features, 1)),
        layers.BatchNormalization(), layers.ReLU(), layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same"),
        layers.BatchNormalization(), layers.ReLU(), layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same"),
        layers.BatchNormalization(), layers.ReLU(), layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"), layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(model, X_tr, y_tr, X_v, y_v, X_te, y_te):
    """Train model, plot history, print evaluation."""
    n_epochs = 10 if FAST_MODE else 30
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    hist = model.fit(
        X_tr.reshape(-1, X_tr.shape[1], 1), y_tr,
        validation_data=(X_v.reshape(-1, X_v.shape[1], 1), y_v),
        epochs=n_epochs, batch_size=256, callbacks=cbs, verbose=1,
    )
    # Training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(hist.history["loss"], label="Train"); a1.plot(hist.history["val_loss"], label="Val")
    a1.set_title("Loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(hist.history["accuracy"], label="Train"); a2.plot(hist.history["val_accuracy"], label="Val")
    a2.set_title("Accuracy"); a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS}/training_history.png", dpi=150); plt.close()

    # Evaluate
    y_prob = model.predict(X_te.reshape(-1, X_te.shape[1], 1), verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    acc, f1, auc = accuracy_score(y_te, y_pred), f1_score(y_te, y_pred, average="weighted"), roc_auc_score(y_te, y_prob)
    print(f"\n  Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Normal","Attack"]))

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(f"{RESULTS}/confusion_matrix.png", dpi=150); plt.close()

    return y_pred, y_prob, {"accuracy": acc, "f1": f1, "auc": auc}


# ═══════════════════════════════════════════════════════════════════════════
# 3. EXPLAINERS
# ═══════════════════════════════════════════════════════════════════════════
def predict_flat(model, X):
    """2D in → 1D probability out. Uses TF directly (GPU-accelerated)."""
    X_3d = tf.constant(X.reshape(-1, X.shape[-1], 1), dtype=tf.float32)
    return model(X_3d, training=False).numpy().flatten()


@tf.function(reduce_retracing=True)
def _predict_tf(model, X_3d):
    """Compiled TF prediction — avoids Python overhead on repeated calls."""
    return model(X_3d, training=False)


def predict_flat_batch(model, X_batch):
    """Batch prediction for multiple samples at once (much faster on GPU)."""
    X_3d = tf.constant(X_batch.reshape(-1, X_batch.shape[-1], 1), dtype=tf.float32)
    return _predict_tf(model, X_3d).numpy().flatten()


def run_shap(model, X_bg, X_explain, feat_names):
    """KernelSHAP explanations."""
    print("\n── SHAP ──")
    n_bg = 50 if FAST_MODE else 100
    n_samples = 200 if FAST_MODE else 500
    bg = X_bg[np.random.choice(len(X_bg), n_bg, replace=False)]
    explainer = shap.KernelExplainer(lambda x: predict_flat(model, x), bg)
    sv = explainer.shap_values(X_explain, nsamples=n_samples)

    # Summary plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X_explain, feature_names=feat_names, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(f"{RESULTS}/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    # Per-sample bar plots
    for i in range(len(X_explain)):
        idx = np.argsort(np.abs(sv[i]))[::-1][:12]
        plt.figure(figsize=(7, 4))
        plt.barh(range(len(idx)), sv[i][idx], color=["#e74c3c" if v>0 else "#3498db" for v in sv[i][idx]])
        plt.yticks(range(len(idx)), [feat_names[j] for j in idx])
        plt.xlabel("SHAP value"); plt.title(f"SHAP — Sample {i}"); plt.gca().invert_yaxis()
        plt.tight_layout(); plt.savefig(f"{RESULTS}/shap_sample_{i}.png", dpi=150); plt.close()

    print(f"  Saved SHAP plots for {len(X_explain)} samples")
    return sv


def run_lime(model, X_train, X_explain, feat_names):
    """LIME explanations."""
    print("\n── LIME ──")
    predict_proba = lambda X: np.column_stack([1 - predict_flat(model, X), predict_flat(model, X)])
    exp = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feat_names, class_names=["Normal","Attack"], mode="classification", random_state=42)

    lime_weights = []
    n_lime_samples = 500 if FAST_MODE else 1000
    for i in range(len(X_explain)):
        explanation = exp.explain_instance(X_explain[i], predict_proba, num_features=15, num_samples=n_lime_samples)
        fig = explanation.as_pyplot_figure(); fig.set_size_inches(7, 4)
        plt.title(f"LIME — Sample {i}"); plt.tight_layout()
        plt.savefig(f"{RESULTS}/lime_sample_{i}.png", dpi=150, bbox_inches="tight"); plt.close()

        weights = {feat_names[fi]: w for fi, w in explanation.local_exp[1]}
        lime_weights.append(weights)

    print(f"  Saved LIME plots for {len(X_explain)} samples")
    return lime_weights


def run_counterfactuals(model, X_explain, feat_names, max_iter=500, lr=0.01):
    """
    Counterfactual search using TensorFlow GradientTape (GPU-accelerated).
    Replaces slow finite-difference gradients with analytic TF gradients.
    """
    print("\n── Counterfactuals ──")
    iters = (100 if FAST_MODE else max_iter)
    results = []

    for i in range(len(X_explain)):
        x_orig = X_explain[i].copy().astype(np.float32)
        x_cf = tf.Variable(x_orig.reshape(1, -1, 1), dtype=tf.float32)
        orig_pred = model(tf.constant(x_orig.reshape(1, -1, 1)), training=False).numpy().flatten()[0]
        target = 0 if orig_pred > 0.5 else 1
        success = False

        for it in range(iters):
            with tf.GradientTape() as tape:
                pred = model(x_cf, training=False)
                # Loss: push prediction toward target
                if target == 1:
                    loss = -pred  # maximize
                else:
                    loss = pred   # minimize
                # L1 sparsity penalty
                l1 = 0.05 * tf.reduce_sum(tf.abs(x_cf - x_orig.reshape(1, -1, 1)))
                loss = loss + l1

            grad = tape.gradient(loss, x_cf)
            x_cf.assign_sub(lr * grad)
            x_cf.assign(tf.clip_by_value(x_cf, 0.0, 1.0))

            current_pred = model(x_cf, training=False).numpy().flatten()[0]
            if (target == 1 and current_pred > 0.5) or (target == 0 and current_pred < 0.5):
                success = True; break

        cf_pred = model(x_cf, training=False).numpy().flatten()[0]
        cf_np = x_cf.numpy().flatten()
        delta = cf_np - x_orig.flatten()
        changes = {feat_names[j]: float(delta[j]) for j in range(len(delta)) if abs(delta[j]) > 1e-6}

        # Plot
        if changes:
            top = sorted(changes.items(), key=lambda t: abs(t[1]), reverse=True)[:12]
            plt.figure(figsize=(7, 4))
            plt.barh(range(len(top)), [v for _,v in top], color=["#e74c3c" if v>0 else "#3498db" for _,v in top])
            plt.yticks(range(len(top)), [k for k,_ in top])
            lbl = lambda p: "Attack" if p > 0.5 else "Normal"
            plt.xlabel("Feature change"); plt.title(f"Counterfactual — Sample {i}  ({lbl(orig_pred)}→{lbl(cf_pred)})")
            plt.gca().invert_yaxis(); plt.tight_layout()
            plt.savefig(f"{RESULTS}/cf_sample_{i}.png", dpi=150); plt.close()

        status = "flipped" if success else "failed"
        print(f"  Sample {i}: {status} in {it+1} iters")
        results.append({"success": success, "changes": changes, "orig": float(orig_pred), "cf": float(cf_pred)})

    return results


def run_saliency(model, X_explain, feat_names):
    """
    Saliency maps (vanilla gradient).
    Computes ∂output/∂input — how much each input feature affects the
    prediction if nudged infinitesimally. Fast (one backward pass per sample).
    """
    print("\n── Saliency Maps ──")
    all_grads = []

    for i in range(len(X_explain)):
        x_tensor = tf.constant(X_explain[i].reshape(1, -1, 1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            pred = model(x_tensor, training=False)
        grads = tape.gradient(pred, x_tensor).numpy().flatten()  # shape: (n_features,)
        all_grads.append(grads)

        # Plot
        abs_g = np.abs(grads)
        idx = np.argsort(abs_g)[::-1][:12]
        plt.figure(figsize=(7, 4))
        plt.barh(range(len(idx)), abs_g[idx], color="#8e44ad")
        plt.yticks(range(len(idx)), [feat_names[j] for j in idx])
        plt.xlabel("|Gradient|"); plt.title(f"Saliency Map — Sample {i}"); plt.gca().invert_yaxis()
        plt.tight_layout(); plt.savefig(f"{RESULTS}/saliency_sample_{i}.png", dpi=150); plt.close()

    print(f"  Saved saliency plots for {len(X_explain)} samples")
    return np.array(all_grads)


def run_gradcam_1d(model, X_explain, feat_names):
    """
    1D Grad-CAM (Gradient-weighted Class Activation Mapping).
    Targets the last Conv1D layer: computes the gradient of the output w.r.t.
    that layer's activations, then weights each filter by its mean gradient
    and sums to get a per-feature importance heatmap.

    In 2D image models this produces a spatial heatmap over pixels.
    Here it produces a 1D heatmap over the feature dimension — which features
    the final conv layer "looked at" most for this prediction.
    """
    print("\n── 1D Grad-CAM ──")

    # Find the last Conv1D layer
    conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv1D):
            conv_layer = layer
            break
    if conv_layer is None:
        print("  [!] No Conv1D layer found — skipping Grad-CAM")
        return None

    print(f"  Target layer: {conv_layer.name} (filters={conv_layer.filters})")

    # Build a sub-model: input → [conv_layer_output, final_output]
    inp = keras.Input(shape=(len(feat_names), 1))
    x = inp
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer == conv_layer:
            conv_out = x
    grad_model = keras.Model(inputs=inp, outputs=[conv_out, x])

    all_heatmaps = []
    for i in range(len(X_explain)):
        x_tensor = tf.constant(X_explain[i].reshape(1, -1, 1), dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(x_tensor, training=False)
            # For binary: use the single sigmoid output directly
            target_score = pred[0, 0]

        # Gradient of output w.r.t. conv layer activations
        grads = tape.gradient(target_score, conv_out)  # (1, seq_len, filters)

        # Global-average-pool the gradients over the spatial dim → weight per filter
        weights = tf.reduce_mean(grads, axis=1)  # (1, filters)

        # Weighted combination of conv feature maps
        conv_out_np = conv_out.numpy()[0]   # (seq_len, filters)
        weights_np = weights.numpy()[0]     # (filters,)

        # Grad-CAM heatmap: weighted sum across filters, then ReLU
        heatmap = np.dot(conv_out_np, weights_np)  # (seq_len,)
        heatmap = np.maximum(heatmap, 0)            # ReLU — only positive contributions

        # The conv+pooling layers may have changed the sequence length,
        # so we interpolate back to the original feature count
        if len(heatmap) != len(feat_names):
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(heatmap))
            x_new = np.linspace(0, 1, len(feat_names))
            heatmap = interp1d(x_old, heatmap, kind="linear")(x_new)

        # Normalise to 0–1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        all_heatmaps.append(heatmap)

        # Plot
        idx = np.argsort(heatmap)[::-1][:12]
        plt.figure(figsize=(7, 4))
        plt.barh(range(len(idx)), heatmap[idx], color="#e67e22")
        plt.yticks(range(len(idx)), [feat_names[j] for j in idx])
        plt.xlabel("Grad-CAM activation"); plt.title(f"1D Grad-CAM — Sample {i}"); plt.gca().invert_yaxis()
        plt.tight_layout(); plt.savefig(f"{RESULTS}/gradcam_sample_{i}.png", dpi=150); plt.close()

    # Summary: average heatmap across all samples
    avg_heatmap = np.mean(all_heatmaps, axis=0)
    idx = np.argsort(avg_heatmap)[::-1][:15]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(idx)), avg_heatmap[idx], color="#e67e22", alpha=0.85)
    plt.yticks(range(len(idx)), [feat_names[j] for j in idx])
    plt.xlabel("Mean Grad-CAM activation"); plt.title("1D Grad-CAM — Average Feature Importance")
    plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig(f"{RESULTS}/gradcam_summary.png", dpi=150); plt.close()

    print(f"  Saved Grad-CAM plots for {len(X_explain)} samples")
    return np.array(all_heatmaps)


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_explanations(model, X_explain, shap_vals, lime_weights, feat_names,
                          saliency_grads=None, gradcam_maps=None):
    """Fidelity + consistency metrics across all methods."""
    print("\n── Evaluation ──")

    # Build importance dicts for each method per sample
    def _imp(method, i):
        if method == "SHAP":
            return dict(zip(feat_names, shap_vals[i]))
        elif method == "LIME":
            return lime_weights[i]
        elif method == "Saliency" and saliency_grads is not None:
            return dict(zip(feat_names, np.abs(saliency_grads[i])))
        elif method == "Grad-CAM" and gradcam_maps is not None:
            return dict(zip(feat_names, gradcam_maps[i]))
        return None

    methods = ["SHAP", "LIME"]
    if saliency_grads is not None:
        methods.append("Saliency")
    if gradcam_maps is not None:
        methods.append("Grad-CAM")

    # Fidelity per method
    fidelity = {m: [] for m in methods}
    for i in range(len(X_explain)):
        x = X_explain[i].reshape(1, -1)
        orig = predict_flat(model, x)[0]
        for m in methods:
            imp = _imp(m, i)
            if imp is None:
                continue
            top5 = sorted(imp, key=lambda f: abs(imp[f]), reverse=True)[:5]
            x_mask = x.copy()
            x_mask[0, [feat_names.index(f) for f in top5]] = 0
            fid = abs(orig - predict_flat(model, x_mask)[0])
            fidelity[m].append(fid)

    for m in methods:
        print(f"  {m:10s} fidelity: {np.mean(fidelity[m]):.4f} ± {np.std(fidelity[m]):.4f}")

    # Pairwise consistency (Jaccard@10) for all method pairs
    from itertools import combinations
    pair_consistency = {}
    for m1, m2 in combinations(methods, 2):
        scores = []
        for i in range(len(X_explain)):
            imp1, imp2 = _imp(m1, i), _imp(m2, i)
            if imp1 is None or imp2 is None:
                continue
            top1 = set(sorted(imp1, key=lambda f: abs(imp1[f]), reverse=True)[:10])
            top2 = set(sorted(imp2, key=lambda f: abs(imp2[f]), reverse=True)[:10])
            scores.append(len(top1 & top2) / len(top1 | top2) if (top1 | top2) else 0)
        pair_consistency[f"{m1}–{m2}"] = scores
        print(f"  {m1}–{m2} consistency: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Plot: fidelity comparison
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"SHAP": "#2ecc71", "LIME": "#e74c3c", "Saliency": "#8e44ad", "Grad-CAM": "#e67e22"}
    means = [np.mean(fidelity[m]) for m in methods]
    stds  = [np.std(fidelity[m]) for m in methods]
    bars = a1.bar(methods, means, yerr=stds, color=[colors[m] for m in methods], capsize=5, alpha=0.85)
    a1.set_title("Mean Fidelity by Method"); a1.set_ylabel("Prediction change (higher = better)"); a1.grid(alpha=0.3)

    # Plot: pairwise consistency heatmap
    n = len(methods)
    matrix = np.eye(n)
    for pair_name, scores in pair_consistency.items():
        m1, m2 = pair_name.split("–")
        i1, i2 = methods.index(m1.split("–")[0]), methods.index(m2.split("–")[0])
        matrix[i1, i2] = matrix[i2, i1] = np.mean(scores)
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=methods, yticklabels=methods,
                cmap="YlGnBu", vmin=0, vmax=1, ax=a2)
    a2.set_title("Pairwise Consistency (Jaccard@10)")

    plt.tight_layout(); plt.savefig(f"{RESULTS}/evaluation.png", dpi=150); plt.close()

    return {
        "fidelity": {m: float(np.mean(fidelity[m])) for m in methods},
        "consistency": {k: float(np.mean(v)) for k, v in pair_consistency.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("XAI for Network Intrusion Detection")
    print("=" * 50)

    # Data
    print("\n[1/8] Loading data...")
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = load_data()

    # Model
    print("\n[2/8] Training 1D-CNN...")
    model = build_cnn(X_tr.shape[1])
    y_pred, y_prob, metrics = train_and_evaluate(model, X_tr, y_tr, X_v, y_v, X_te, y_te)

    # Select samples to explain (mix of normal + attack)
    print("\n[3/8] Selecting samples...")
    norm_idx = np.where((y_te == 0) & (y_pred == 0))[0]
    att_idx  = np.where((y_te == 1) & (y_pred == 1))[0]
    sel = np.concatenate([np.random.choice(norm_idx, 2, replace=False), np.random.choice(att_idx, 3, replace=False)])
    X_exp = X_te[sel]
    print(f"  {len(sel)} samples selected")

    # Explain
    print("\n[4/8] SHAP explanations...")
    sv = run_shap(model, X_tr, X_exp, feats)

    print("\n[5/8] LIME explanations...")
    lw = run_lime(model, X_tr, X_exp, feats)

    print("\n[6/8] Saliency maps + Grad-CAM...")
    sal = run_saliency(model, X_exp, feats)
    gcam = run_gradcam_1d(model, X_exp, feats)

    print("\n[7/8] Counterfactual explanations...")
    cf = run_counterfactuals(model, X_exp, feats)

    # Evaluate
    print("\n[8/8] Evaluating all explanations...")
    ev = evaluate_explanations(model, X_exp, sv, lw, feats,
                               saliency_grads=sal, gradcam_maps=gcam)

    # Save report
    report = {"model": metrics, "evaluation": ev,
              "counterfactuals": {"success_rate": sum(r["success"] for r in cf) / len(cf)}}
    json.dump(report, open(f"{RESULTS}/report.json", "w"), indent=2)

    print(f"\n{'='*50}")
    print(f"Done! All results saved to {RESULTS}/")
    print(f"{'='*50}")