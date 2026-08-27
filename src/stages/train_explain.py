"""
Stage: train_explain
====================
Trains the three models (CNN, XGBoost, Random Forest) on the CLEANED NSL-KDD
table and runs all five XAI methods, over a deterministic 100-sample set.

Run (from repo root, xai env -- needs TensorFlow/SHAP/LIME):
    conda run -n xai python src/stages/train_explain.py \
        --clean data/processed/clean.parquet \
        --manifest data/processed/sample_manifest.json \
        --results data/processed/results.json \
        --attributions data/processed/attributions.json \
        --n-samples 100

    Add --fast for a quick dev run.

Requires (xai env): tensorflow, shap, lime, xgboost, scikit-learn, pandas,
    pyarrow, mlflow
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


SEED = 42


def _add_repo_root_to_path():
    here = os.path.abspath(os.path.dirname(__file__))
    d = here
    for _ in range(5):
        if os.path.exists(os.path.join(d, "data_utils.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
        d = os.path.dirname(d)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    return cwd


REPO_ROOT = _add_repo_root_to_path()

# Reuse the existing pipeline components verbatim.
from data_utils import ATTACKS  # noqa: E402
from models import CNNModel, XGBoostModel, RandomForestModel, configure_gpu  # noqa: E402
from explainers import (  # noqa: E402
    run_shap, run_lime, run_counterfactuals, run_saliency, run_gradcam_1d,
)
from evaluation import evaluate_explanations  # noqa: E402


def load_clean(clean_path):
    """Load clean.parquet, encode categoricals, MinMax-scale, split by `split`.

    The cleaned table holds RAW values (protocol_type='tcp', label='neptune'),
    so encoding + scaling happen here on whatever feature set survived cleaning.
    Returns X_tr, X_val, X_te, y_tr, y_val, y_te, feat_cols, test_labels_raw.
    """
    if not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"{clean_path} not found. Run the clean stage first:\n"
            f"  conda run -n dataprep python src/stages/clean.py "
            f"--drop-zero-frac 0.95 --out {clean_path} ..."
        )
    df = pd.read_parquet(clean_path)

    feat_cols = [c for c in df.columns if c not in ("label", "split")]

    # Encode the categorical features that survived cleaning (if present).
    for col in ["protocol_type", "service", "flag"]:
        if col in feat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Binary label via the same ATTACKS mapping as the baseline pipeline.
    y_all = (df["label"].map(lambda x: ATTACKS.get(x, "unknown")) != "normal").astype(int).values

    # Split by the existing column (train stays train, test stays test).
    is_train = (df["split"] == "train").values
    is_test = (df["split"] == "test").values

    scaler = MinMaxScaler()
    X_train_full = scaler.fit_transform(df.loc[is_train, feat_cols].values.astype(np.float32))
    X_test = scaler.transform(df.loc[is_test, feat_cols].values.astype(np.float32))

    y_train_full = y_all[is_train]
    y_test = y_all[is_test]
    test_labels_raw = df.loc[is_test, "label"].values  # keep raw attack names

    # Carve a validation slice out of train (same 0.1 as the baseline).
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=SEED,
        stratify=y_train_full,
    )
    print(f"  Train: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"  Features: {len(feat_cols)} -> {feat_cols}")
    return X_tr, X_val, X_test, y_tr, y_val, y_test, feat_cols, test_labels_raw


def select_samples(y_test, n_samples, seed=SEED):
    """Deterministic stratified selection of n_samples test rows, preserving the
    attack fraction. Returns sorted indices into the test arrays."""
    rng = np.random.default_rng(seed)
    norm_idx = np.where(y_test == 0)[0]
    att_idx = np.where(y_test == 1)[0]
    frac = len(att_idx) / len(y_test)
    n_att = int(round(n_samples * frac))
    n_norm = n_samples - n_att
    n_att = min(n_att, len(att_idx))
    n_norm = min(n_norm, len(norm_idx))
    sel = np.concatenate([
        rng.choice(norm_idx, n_norm, replace=False),
        rng.choice(att_idx, n_att, replace=False),
    ])
    return np.sort(sel)


def per_class_metrics(model, X_te, y_te):
    """Return the per-class precision/recall/f1 dict (Normal/Attack) so we can
    watch ATTACK recall specifically."""
    y_prob = model.predict_proba(X_te)
    y_pred = (y_prob > 0.5).astype(int)
    rep = classification_report(y_te, y_pred, target_names=["Normal", "Attack"],
                                output_dict=True, zero_division=0)
    return {
        "normal": {k: float(rep["Normal"][k]) for k in ["precision", "recall", "f1-score"]},
        "attack": {k: float(rep["Attack"][k]) for k in ["precision", "recall", "f1-score"]},
    }


def extract_attributions(raw, feat_names, y_labels_for_samples):
    """Convert the raw explainer outputs for one model into a uniform,
    graph-loadable structure: per-sample, per-method, per-feature importances.

    `raw` is the dict returned per model with keys shap/lime/cf/saliency/gradcam.
    Returns a list (one entry per sample) of:
        {sample_index, label, methods: {METHOD: {feature: importance, ...}}}
    Raw scores are stored (NOT top-k or Jaccard) so downstream analysis can
    compute Jaccard-at-any-k and rank correlation without re-running anything.
    """
    n = len(next(v for v in raw.values() if v is not None))
    out = []
    for i in range(n):
        methods = {}

        # SHAP: array (n_samples, n_features)
        if raw.get("shap") is not None:
            methods["SHAP"] = {feat_names[j]: float(raw["shap"][i][j])
                               for j in range(len(feat_names))}
        # LIME: list of {feature_name: weight}
        if raw.get("lime") is not None:
            methods["LIME"] = {k: float(v) for k, v in raw["lime"][i].items()}
        # Saliency: array (n_samples, n_features) or None
        if raw.get("saliency") is not None:
            methods["Saliency"] = {feat_names[j]: float(abs(raw["saliency"][i][j]))
                                   for j in range(len(feat_names))}
        # Grad-CAM: array (n_samples, n_features) or None
        if raw.get("gradcam") is not None:
            methods["Grad-CAM"] = {feat_names[j]: float(raw["gradcam"][i][j])
                                   for j in range(len(feat_names))}
        # Counterfactual: list of {"changes": {feat: delta}, "success":..., ...}
        if raw.get("cf") is not None:
            cf = raw["cf"][i]
            methods["Counterfactual"] = {k: float(v) for k, v in cf["changes"].items()}

        out.append({
            "sample_index": i,
            "label": int(y_labels_for_samples[i]),
            "methods": methods,
        })
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Train + explain on cleaned NSL-KDD; emit results + attributions."
    )
    parser.add_argument("--clean", default="data/processed/clean.parquet")
    parser.add_argument("--manifest", default="data/processed/sample_manifest.json")
    parser.add_argument("--results", default="data/processed/results.json")
    parser.add_argument("--attributions", default="data/processed/attributions.json")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    np.random.seed(SEED)
    print(f"[train_explain] repo root: {REPO_ROOT}")
    configure_gpu()

    print(f"[train_explain] loading cleaned data: {args.clean}")
    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_cols, test_labels_raw = load_clean(args.clean)

    # Deterministic 100-sample selection.
    sel = select_samples(y_te, args.n_samples, seed=SEED)
    X_explain = X_te[sel]
    y_explain = y_te[sel]
    print(f"[train_explain] selected {len(sel)} samples "
          f"(attack fraction {y_explain.mean():.3f})")

    # Manifest -- the shared record every downstream stage reads.
    manifest = {
        "seed": SEED,
        "clean_source": args.clean,
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "n_samples": int(len(sel)),
        "sample_test_indices": sel.tolist(),
        "sample_labels": y_explain.tolist(),
        "sample_labels_raw": [str(test_labels_raw[i]) for i in sel],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[train_explain] wrote manifest -> {args.manifest}")

    results = {"fast_mode": args.fast, "n_features": len(feat_cols),
               "clean_source": args.clean, "models": {}}
    all_attributions = {}

    for ModelClass in [CNNModel, XGBoostModel, RandomForestModel]:
        model = ModelClass()
        name = model.name
        print(f"\n{'='*60}\n  {name.upper()}\n{'='*60}")

        model.build(X_tr.shape[1])
        model.fit(X_tr, y_tr, X_val, y_val, fast_mode=args.fast)
        metrics = model.evaluate(X_te, y_te)          # prints acc/f1/auc + report
        pc = per_class_metrics(model, X_te, y_te)     # captured per-class

        # Five explainers -- unchanged, on the shared sample set.
        shap_vals = run_shap(model, X_tr, X_explain, feat_cols, fast_mode=args.fast)
        lime_weights = run_lime(model, X_tr, X_explain, feat_cols, fast_mode=args.fast)
        cf_results = run_counterfactuals(model, X_explain, feat_cols, fast_mode=args.fast)
        saliency = run_saliency(model, X_explain, feat_cols)
        gradcam = run_gradcam_1d(model, X_explain, feat_cols)

        eval_results = evaluate_explanations(
            model, X_explain, feat_cols,
            shap_vals=shap_vals, lime_weights=lime_weights,
            saliency_grads=saliency, gradcam_maps=gradcam,
        )

        raw = {"shap": shap_vals, "lime": lime_weights, "cf": cf_results,
               "saliency": saliency, "gradcam": gradcam}
        all_attributions[name] = extract_attributions(raw, feat_cols, y_explain)

        results["models"][name] = {
            "metrics": {k: v for k, v in metrics.items() if k not in ("y_pred", "y_prob")},
            "per_class": pc,
            "evaluation": eval_results,
            "counterfactual_success_rate": (
                sum(r["success"] for r in cf_results) / len(cf_results)),
        }

        # Baseline sanity print (option 1).
        print(f"\n  >> {name.upper()} accuracy={metrics['accuracy']:.4f} "
              f"| ATTACK recall={pc['attack']['recall']:.4f} "
              f"| NORMAL recall={pc['normal']['recall']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    with open(args.attributions, "w") as f:
        json.dump({"feature_columns": feat_cols, "attributions": all_attributions}, f, indent=2)
    print(f"\n[train_explain] wrote results -> {args.results}")
    print(f"[train_explain] wrote attributions -> {args.attributions}")

    # Compact comparison table for the baseline eyeball.
    print(f"\n{'='*60}\n  BASELINE CHECK (option 1)\n{'='*60}")
    print(f"  {'model':10s} {'acc':>7s} {'f1':>7s} {'auc':>7s} "
          f"{'atk_rec':>8s} {'nrm_rec':>8s}")
    for name, m in results["models"].items():
        met, pc = m["metrics"], m["per_class"]
        print(f"  {name:10s} {met['accuracy']:7.4f} {met['f1']:7.4f} {met['auc']:7.4f} "
              f"{pc['attack']['recall']:8.4f} {pc['normal']['recall']:8.4f}")
    print("  Expected NSL-KDD test range ~0.75-0.82. Watch CNN vs trees, and")
    print("  ATTACK recall vs NORMAL recall (rare-attack sensitivity).")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("xai-nids-graph-augmentation")
            with mlflow.start_run(run_name="train_explain"):
                mlflow.log_param("stage", "train_explain")
                mlflow.log_param("n_features", len(feat_cols))
                mlflow.log_param("clean_source", os.path.basename(args.clean))
                mlflow.log_param("n_samples", len(sel))
                for name, m in results["models"].items():
                    mlflow.log_metric(f"{name}_accuracy", m["metrics"]["accuracy"])
                    mlflow.log_metric(f"{name}_f1", m["metrics"]["f1"])
                    mlflow.log_metric(f"{name}_auc", m["metrics"]["auc"])
                    mlflow.log_metric(f"{name}_attack_recall", m["per_class"]["attack"]["recall"])
                for p in (args.results, args.attributions, args.manifest):
                    if os.path.exists(p):
                        mlflow.log_artifact(p, artifact_path="train_explain")
            print("[train_explain] logged to MLflow.")
        except ImportError:
            print("  (mlflow not installed -- skipping logging)")

    print("[train_explain] done.")


if __name__ == "__main__":
    main()