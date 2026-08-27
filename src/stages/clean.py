"""
Stage: clean
============
Cleans the RAW NSL-KDD data by removing:
  (1) zero-variance (constant) columns — always removed; they carry no information.
  (2) OPTIONALLY, columns whose zero-fraction exceeds a threshold
      (--drop-zero-frac, e.g. 0.95)

Run -- safe comparison version (zero-variance only), kept side by side:
    conda run -n dataprep python src/stages/clean.py \
        --data-dir data \
        --out data/processed/clean_full.parquet \
        --dropped data/processed/dropped_columns_full.json \
        --no-profile
        
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


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

from data_utils import COLS, ATTACKS  # noqa: E402


def load_raw(data_dir):
    """Load train+test into one labelled dataframe (raw values, no scaling)."""
    paths = {
        "train": os.path.join(data_dir, "KDDTrain+.txt"),
        "test": os.path.join(data_dir, "KDDTest+.txt"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find {p}. Run the main pipeline once so "
                f"data_utils.load_data() downloads NSL-KDD into '{data_dir}/'."
            )
    train = pd.read_csv(paths["train"], names=COLS, header=None)
    test = pd.read_csv(paths["test"], names=COLS, header=None)
    train["split"] = "train"
    test["split"] = "test"
    df = pd.concat([train, test], ignore_index=True)
    return df.drop(columns=["difficulty_level"])


def classify_columns(df, feature_cols, zero_frac_threshold):
    """Split feature columns into: constants (zero-variance), high-zero-fraction
    (>= threshold, only if a threshold is given), and kept.

    A constant column is reported ONLY as a constant, even if it also exceeds the
    zero-fraction threshold, so the categories don't double-count.
    Returns (constants: dict, high_zero: dict, kept: list).
    """
    constants, high_zero, kept = {}, {}, []
    for c in feature_cols:
        nun = df[c].nunique(dropna=False)
        zf = float((df[c] == 0).mean())
        if nun == 1:
            val = df[c].iloc[0]
            constants[c] = {
                "reason": "zero_variance_constant",
                "constant_value": (float(val) if np.isscalar(val)
                                   and not isinstance(val, str) else str(val)),
                "zero_fraction": zf,
            }
        elif zero_frac_threshold is not None and zf >= zero_frac_threshold:
            high_zero[c] = {
                "reason": f"zero_fraction_ge_{zero_frac_threshold}",
                "zero_fraction": zf,
                "n_unique": int(nun),
            }
        else:
            kept.append(c)
    return constants, high_zero, kept


def build_profile(df, title, minimal=True):
    from ydata_profiling import ProfileReport
    return ProfileReport(df, title=title, minimal=False)


def maybe_log_mlflow(out_path, dropped_path, record):
    try:
        import mlflow
    except ImportError:
        print("  (mlflow not installed -- skipping experiment logging)")
        return
    mlflow.set_experiment("xai-nids-graph-augmentation")
    with mlflow.start_run(run_name="clean"):
        mlflow.log_param("stage", "clean")
        mlflow.log_param("zero_frac_threshold", str(record["zero_frac_threshold"]))
        mlflow.log_metric("n_features_before", record["n_features_before"])
        mlflow.log_metric("n_features_after", record["n_features_after"])
        mlflow.log_metric("n_dropped_constant", len(record["dropped_constant"]))
        mlflow.log_metric("n_dropped_high_zero", len(record["dropped_high_zero"]))
        for p in (out_path, dropped_path):
            if os.path.exists(p):
                mlflow.log_artifact(p, artifact_path="clean")
        print("  Logged clean stage to MLflow.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean NSL-KDD: drop zero-variance columns (always) and "
                    "optionally columns above a zero-fraction threshold."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/processed/clean.parquet")
    parser.add_argument("--dropped", default="data/processed/dropped_columns.json")
    parser.add_argument("--drop-zero-frac", type=float, default=None,
                        help="Also drop columns whose zero-fraction is >= this "
                             "value (e.g. 0.95). Default: only zero-variance "
                             "constants are dropped.")
    parser.add_argument("--profile-out", default="reports/profile_clean.html")
    parser.add_argument("--full-profile", action="store_true",
                        help="Generate the FULL profile (with correlations). "
                             "Slower/heavier than the default minimal profile.")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    print(f"[clean] repo root: {REPO_ROOT}")
    print(f"[clean] loading raw NSL-KDD from {args.data_dir}...")
    df = load_raw(args.data_dir)
    feature_cols = [c for c in df.columns if c not in ("label", "split")]
    print(f"  {len(df):,} rows | {len(feature_cols)} feature columns.")

    thr = args.drop_zero_frac
    if thr is not None:
        print(f"[clean] drop rule: zero-variance constants + columns with "
              f">= {thr*100:.0f}% zeros (advisor-directed).")
    else:
        print("[clean] drop rule: zero-variance constants ONLY (safe mode).")

    constants, high_zero, kept = classify_columns(df, feature_cols, thr)

    if constants:
        print(f"  Dropping {len(constants)} CONSTANT column(s):")
        for c, m in constants.items():
            print(f"    - {c}  (constant {m['constant_value']}, "
                  f"{m['zero_fraction']*100:.1f}% zeros)")
    else:
        print("  No constant columns found.")

    if thr is not None:
        if high_zero:
            print(f"  Dropping {len(high_zero)} HIGH-ZERO (>= {thr*100:.0f}%) "
                  f"column(s) -- note: these include rare-attack detectors:")
            for c, m in sorted(high_zero.items(), key=lambda t: -t[1]["zero_fraction"]):
                print(f"    - {c}  ({m['zero_fraction']*100:.1f}% zeros)")
        else:
            print(f"  No non-constant columns exceed the {thr*100:.0f}% threshold.")

    dropped_all = set(constants) | set(high_zero)
    keep_cols = [c for c in df.columns if c not in dropped_all]
    clean_df = df[keep_cols].copy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    clean_df.to_parquet(args.out, index=False)
    print(f"[clean] wrote cleaned table -> {args.out} "
          f"({clean_df.shape[0]} rows x {clean_df.shape[1]} cols)")

    n_after = len([c for c in keep_cols if c not in ("label", "split")])
    record = {
        "zero_frac_threshold": thr,
        "n_features_before": len(feature_cols),
        "n_features_after": n_after,
        "dropped_constant": constants,
        "dropped_high_zero": high_zero,
        "kept_features": [c for c in kept],
        "note": ("Constants are always dropped. High-zero-fraction columns are "
                 "dropped only when --drop-zero-frac is set; on NSL-KDD these are "
                 "largely U2R/R2L rare-attack detectors."),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.dropped)), exist_ok=True)
    with open(args.dropped, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[clean] wrote drop record -> {args.dropped}")
    print(f"[clean] features: {len(feature_cols)} -> {n_after} "
          f"(dropped {len(dropped_all)}: {len(constants)} constant, "
          f"{len(high_zero)} high-zero)")

    if not args.no_profile:
        print("[clean] profiling cleaned data...")
        prof = build_profile(clean_df, "NSL-KDD -- cleaned",
                             minimal=not args.full_profile)
        os.makedirs(os.path.dirname(os.path.abspath(args.profile_out)), exist_ok=True)
        prof.to_file(args.profile_out)
        print(f"  wrote {args.profile_out}")

    if not args.no_mlflow:
        maybe_log_mlflow(args.out, args.dropped, record)

    print("[clean] done.")


if __name__ == "__main__":
    main()