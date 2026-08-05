"""
Stage: profile_before
======================
Profiles the RAW NSL-KDD data (before any graph transformation) and writes an
HTML profiling report. This is the "before" half of the YData before/after
comparison in the graph-augmentation experiment.

Why it profiles the raw dataframe (not load_data()'s output):
    data_utils.load_data() returns MinMax-scaled, split numpy arrays with no
    column context — useless for a human-readable profile. This stage instead
    loads the raw KDDTrain+/KDDTest+ files into a labelled dataframe, reusing
    the repo's own COLS and ATTACKS so there is a single source of truth and no
    drift from the main pipeline.

Run (from repo root, dataprep env):
    conda run -n dataprep python src/stages/profile_before.py \
        --data-dir data \
        --out reports/profile_before.html

Requires (in the dataprep env):
    pip install ydata-profiling pandas mlflow
"""

import argparse
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Make the repo's own modules importable regardless of where this file lives.
# We walk up from this file to find the directory containing data_utils.py.
# ---------------------------------------------------------------------------
def _add_repo_root_to_path():
    here = os.path.abspath(os.path.dirname(__file__))
    d = here
    for _ in range(5):  # search up to 5 levels up
        if os.path.exists(os.path.join(d, "data_utils.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            return d
        d = os.path.dirname(d)
    # Fallback: assume current working directory is the repo root
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    return cwd


REPO_ROOT = _add_repo_root_to_path()

# Reuse the repo's canonical column list + attack mapping — single source of truth.
from data_utils import COLS, ATTACKS  # noqa: E402


def load_raw_dataframe(data_dir: str) -> pd.DataFrame:
    """Load KDDTrain+ and KDDTest+ into one labelled, human-readable dataframe.

    Adds two helper columns for profiling only (not model features):
        - attack_category : normal / DoS / Probe / R2L / U2R
        - binary_label    : 'normal' vs 'attack'
        - split           : which file the row came from (train/test)
    """
    train_path = os.path.join(data_dir, "KDDTrain+.txt")
    test_path = os.path.join(data_dir, "KDDTest+.txt")

    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find {p}. Run the ingest stage first (or the main "
                f"pipeline once) so data_utils.load_data() downloads the files "
                f"into '{data_dir}/'."
            )

    train = pd.read_csv(train_path, names=COLS, header=None)
    test = pd.read_csv(test_path, names=COLS, header=None)
    train["split"] = "train"
    test["split"] = "test"

    df = pd.concat([train, test], ignore_index=True)
    df = df.drop(columns=["difficulty_level"])

    # Human-readable label columns derived from the raw 'label' field.
    df["attack_category"] = df["label"].map(lambda x: ATTACKS.get(x, "unknown"))
    df["binary_label"] = (df["attack_category"] != "normal").map(
        {True: "attack", False: "normal"}
    )
    return df


def build_profile(df: pd.DataFrame, title: str):
    """Create a ydata-profiling ProfileReport. Imported lazily so the module
    can be inspected/imported without the (heavy) profiling dependency."""
    from ydata_profiling import ProfileReport

    # minimal=True keeps it fast on ~148K rows; flip to False for the full
    # (slower) report with all correlations/interactions once you're happy.
    return ProfileReport(df, title=title, minimal=True)


def maybe_log_mlflow(df: pd.DataFrame, out_path: str, title: str):
    """Log basic stats + the HTML artifact to MLflow if it's installed.

    Silently no-ops if mlflow isn't available, so the stage still runs in a
    bare environment. Uses a local file store unless MLFLOW_TRACKING_URI is set.
    """
    try:
        import mlflow
    except ImportError:
        print("  (mlflow not installed — skipping experiment logging)")
        return

    mlflow.set_experiment("xai-nids-graph-augmentation")
    with mlflow.start_run(run_name="profile_before"):
        mlflow.log_param("stage", "profile_before")
        mlflow.log_param("n_rows", len(df))
        mlflow.log_param("n_columns", df.shape[1])
        mlflow.log_metric("attack_fraction",
                          float((df["binary_label"] == "attack").mean()))
        mlflow.log_metric("n_unique_services", int(df["service"].nunique()))
        if os.path.exists(out_path):
            mlflow.log_artifact(out_path, artifact_path="profiling")
        print("  Logged params + report artifact to MLflow "
              f"(experiment='xai-nids-graph-augmentation').")


def main():
    parser = argparse.ArgumentParser(
        description="Profile raw NSL-KDD (before graph transform)."
    )
    parser.add_argument("--data-dir", default="data",
                        help="Directory holding KDDTrain+.txt / KDDTest+.txt "
                             "(default: data)")
    parser.add_argument("--out", default="reports/profile_before.html",
                        help="Output HTML path (default: reports/profile_before.html)")
    parser.add_argument("--title", default="NSL-KDD — before graph transform",
                        help="Report title.")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging even if it's installed.")
    args = parser.parse_args()

    print(f"[profile_before] repo root: {REPO_ROOT}")
    print(f"[profile_before] loading raw data from: {args.data_dir}")
    df = load_raw_dataframe(args.data_dir)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns "
          f"(train+test combined).")
    print(f"  Attack fraction: {(df['binary_label'] == 'attack').mean():.3f}")

    print("[profile_before] building profile report "
          "(minimal mode; this takes ~1-2 min)...")
    profile = build_profile(df, args.title)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    profile.to_file(args.out)
    print(f"[profile_before] wrote report → {args.out}")

    if not args.no_mlflow:
        maybe_log_mlflow(df, args.out, args.title)

    print("[profile_before] done.")


if __name__ == "__main__":
    main()