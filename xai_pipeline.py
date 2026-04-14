"""
XAI for Network Intrusion Detection — Compact Pipeline
=======================================================
Train a 1D-CNN on NSL-KDD dataset, then explain predictions with SHAP, LIME,
and counterfactuals. Evaluate explanation quality.

Usage:  python xai_pipeline.py
"""

import os, json, warnings, requests
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
# Downloads the NSL-KDD dataset (two text files) if not already on disk
# Encodes the three categorical columns (protocol type, service, and flag) 
# into numbers using the LabelEncoder. Creates mapping from attack labels 
# to binary (normal = 0, attack = 1), scales all 42 features to 0-1 range 
# with MinMaxScaler, splits off 10% of training data for validation
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
# defines the neural network. It's 3 convolutional blocks stacked - each one
# does ConvID --> Batch Norm --> ReLu --> Pooling. The features get reshaped
# from a flat row into a 1D "sequence" so the convolutions can detect local 
# patterns accross neighboring features (relationships, between src_bytes, 
# dst_bytes, and land). After the conv blocks, GlobalAveragePooling collapses 
# everything, then a Dense(64) + Dropout + Dense(1 sigmoid) gives a probability 
# of "attack."
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

# Trains with early stopping (stops if validation loss doesn't improve for 5 epochs) 
# and learning rate reduction. Then it evaluates on the test set and saves 
# the confusion matrix.
def train_and_evaluate(model, X_tr, y_tr, X_v, y_v, X_te, y_te):
    """Train model, plot history, print evaluation."""
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    hist = model.fit(
        X_tr.reshape(-1, X_tr.shape[1], 1), y_tr,
        validation_data=(X_v.reshape(-1, X_v.shape[1], 1), y_v),
        epochs=30, batch_size=256, callbacks=cbs, verbose=1,
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
    """2D in → 1D probability out (wrapper for XAI tools)."""
    return model.predict(X.reshape(-1, X.shape[1], 1), verbose=0).flatten()

# This takes 100 random training samples as a "background" baseline, then
# for each of yout 5 selected samples, it perturbs features ~500 times and
# measures how each perturbation changes the prediction. The result is a SHAP 
# value per feature sample - positive means "pushes toward attack", negative 
# means "pushes toward normal."
def run_shap(model, X_bg, X_explain, feat_names):
    """KernelSHAP explanations."""
    print("\n── SHAP ──")
    bg = X_bg[np.random.choice(len(X_bg), 100, replace=False)]
    explainer = shap.KernelExplainer(lambda x: predict_flat(model, x), bg)
    sv = explainer.shap_values(X_explain, nsamples=500)

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

# works sample-by-sample. For each sample, it generates 1000 perturbed neighbors,
# gets the model's prediction on all of them, then fits a simple linear regression.
# The coefficients of that linear model becomes the "explanation" - they tell you 
# which  features the model relied on locally.
def run_lime(model, X_train, X_explain, feat_names):
    """LIME explanations."""
    print("\n── LIME ──")
    predict_proba = lambda X: np.column_stack([1 - predict_flat(model, X), predict_flat(model, X)])
    exp = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feat_names, class_names=["Normal","Attack"], mode="classification", random_state=42)

    lime_weights = []
    for i in range(len(X_explain)):
        explanation = exp.explain_instance(X_explain[i], predict_proba, num_features=15)
        fig = explanation.as_pyplot_figure(); fig.set_size_inches(7, 4)
        plt.title(f"LIME — Sample {i}"); plt.tight_layout()
        plt.savefig(f"{RESULTS}/lime_sample_{i}.png", dpi=150, bbox_inches="tight"); plt.close()

        weights = {feat_names[fi]: w for fi, w in explanation.local_exp[1]}
        lime_weights.append(weights)

    print(f"  Saved LIME plots for {len(X_explain)} samples")
    return lime_weights

# is intuitive. For each sample, it asks: "what's the smallest change I can make
# to flip the prediction?" It does this by nudging features in the direction of 
# the gradient (computed via finite differences) while penalizing large changes 
# (L1 regularization). If it flips the prediction within 500 iterations, it reports 
# which feature changed and by how much.
def run_counterfactuals(model, X_explain, feat_names, max_iter=500, lr=0.01):
    """Gradient-free counterfactual search."""
    print("\n── Counterfactuals ──")
    results = []
    for i in range(len(X_explain)):
        x = X_explain[i].copy().astype(np.float64)
        x_orig = x.copy()
        orig_pred = predict_flat(model, x.reshape(1, -1))[0]
        target = 0 if orig_pred > 0.5 else 1
        success = False

        for it in range(max_iter):
            pred = predict_flat(model, x.reshape(1, -1))[0]
            if (target == 1 and pred > 0.5) or (target == 0 and pred < 0.5):
                success = True; break
            # Finite-difference gradient
            grad = np.zeros_like(x)
            for j in range(len(x)):
                xp, xm = x.copy(), x.copy()
                xp[j] += 1e-4; xm[j] -= 1e-4
                grad[j] = (predict_flat(model, xp.reshape(1,-1))[0] - predict_flat(model, xm.reshape(1,-1))[0]) / 2e-4
            x -= lr * ((-grad if target == 1 else grad) + 0.05 * np.sign(x - x_orig))
            x = np.clip(x, 0, 1)

        cf_pred = predict_flat(model, x.reshape(1, -1))[0]
        delta = x - x_orig
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


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# compares SHAP and LIME by two measures: fidelity(if you zero the top-5 features
# each method says are important, how much does the predicition actually change?)
# and consistency (do both methods agree on which top-10 features matter, measured 
# by Jaccard overlap?)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_explanations(model, X_explain, shap_vals, lime_weights, feat_names):
    """Fidelity + consistency metrics."""
    print("\n── Evaluation ──")
    shap_fid, lime_fid, consistency = [], [], []

    for i in range(len(X_explain)):
        x = X_explain[i].reshape(1, -1)
        orig = predict_flat(model, x)[0]

        # Fidelity: mask top-5 features, measure prediction change
        for tag, imp in [("SHAP", dict(zip(feat_names, shap_vals[i]))), ("LIME", lime_weights[i])]:
            top5 = sorted(imp, key=lambda f: abs(imp[f]), reverse=True)[:5]
            x_mask = x.copy()
            x_mask[0, [feat_names.index(f) for f in top5]] = 0
            fid = abs(orig - predict_flat(model, x_mask)[0])
            (shap_fid if tag == "SHAP" else lime_fid).append(fid)

        # Consistency: Jaccard overlap of top-10 features
        shap_top = set(sorted(feat_names, key=lambda f: abs(dict(zip(feat_names, shap_vals[i]))[f]), reverse=True)[:10])
        lime_top = set(sorted(lime_weights[i], key=lambda f: abs(lime_weights[i][f]), reverse=True)[:10])
        consistency.append(len(shap_top & lime_top) / len(shap_top | lime_top) if (shap_top | lime_top) else 0)

    print(f"  SHAP fidelity:  {np.mean(shap_fid):.4f} ± {np.std(shap_fid):.4f}")
    print(f"  LIME fidelity:  {np.mean(lime_fid):.4f} ± {np.std(lime_fid):.4f}")
    print(f"  Consistency:    {np.mean(consistency):.4f} ± {np.std(consistency):.4f}")

    # Plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(["SHAP","LIME"], [np.mean(shap_fid), np.mean(lime_fid)],
           yerr=[np.std(shap_fid), np.std(lime_fid)], color=["#2ecc71","#e74c3c"], capsize=5)
    a1.set_title("Mean Fidelity"); a1.set_ylabel("Prediction change"); a1.grid(alpha=0.3)
    a2.bar(range(len(consistency)), consistency, color="#3498db")
    a2.set_title("SHAP–LIME Consistency (Jaccard@10)"); a2.set_xlabel("Sample"); a2.set_ylim(0,1); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{RESULTS}/evaluation.png", dpi=150); plt.close()

    return {"shap_fidelity": float(np.mean(shap_fid)), "lime_fidelity": float(np.mean(lime_fid)), "consistency": float(np.mean(consistency))}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("XAI for Network Intrusion Detection")
    print("=" * 50)

    # Data
    print("\n[1/6] Loading data...")
    X_tr, X_v, X_te, y_tr, y_v, y_te, feats = load_data()

    # Model
    print("\n[2/6] Training 1D-CNN...")
    model = build_cnn(X_tr.shape[1])
    y_pred, y_prob, metrics = train_and_evaluate(model, X_tr, y_tr, X_v, y_v, X_te, y_te)

    # Select samples to explain (mix of normal + attack)
    print("\n[3/6] Selecting samples...")
    norm_idx = np.where((y_te == 0) & (y_pred == 0))[0]
    att_idx  = np.where((y_te == 1) & (y_pred == 1))[0]
    sel = np.concatenate([np.random.choice(norm_idx, 2, replace=False), np.random.choice(att_idx, 3, replace=False)])
    X_exp = X_te[sel]
    print(f"  {len(sel)} samples selected")

    # Explain
    print("\n[4/6] SHAP explanations...")
    sv = run_shap(model, X_tr, X_exp, feats)

    print("\n[5/6] LIME + Counterfactual explanations...")
    lw = run_lime(model, X_tr, X_exp, feats)
    cf = run_counterfactuals(model, X_exp, feats)

    # Evaluate
    print("\n[6/6] Evaluating explanations...")
    ev = evaluate_explanations(model, X_exp, sv, lw, feats)

    # Save report
    report = {"model": metrics, "evaluation": ev,
              "counterfactuals": {"success_rate": sum(r["success"] for r in cf) / len(cf)}}
    json.dump(report, open(f"{RESULTS}/report.json", "w"), indent=2)

    print(f"\n{'='*50}")
    print(f"Done! All results saved to {RESULTS}/")
    print(f"{'='*50}")
