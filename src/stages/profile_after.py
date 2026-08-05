"""
Stage: profile_after
====================
Profiles the graph-AUGMENTED table (after.parquet) and produces the
before/after comparison report — the core "YData before and after graphDB"
deliverable.

This stage emits THREE artifacts:
    1. profile_after.html      — standalone profile of the augmented table
    2. profile_comparison.html — ydata-profiling .compare() of before vs after,
                                 on the SHARED original features (apples-to-apples)
    3. profile_graph_feats.html— dedicated profile of the two NEW graph features
                                 (graph_degree, graph_pagerank), which don't exist
                                 in the "before" table and so can't be compared.

────────────────────────────────────────────────────────────────────────────
IMPORTANT METHODOLOGICAL NOTES (see README "Graph-augmentation notes"):

NOTE 1 — Test-set-only, self-consistent scaling.
    graph_transform.py preprocesses the TEST set on its OWN statistics (MinMax
    fit on test), NOT reusing the train-fitted scaler from data_utils.load_data().
    This is deliberate: the graph stage needs the sample scaled self-consistently
    while preserving row identity. Consequence: the whole before/after comparison
    is a TEST-set-vs-TEST-set-graph comparison. The test set's attack fraction
    (~0.569) differs from the train+test combined figure (~0.481) because NSL-KDD's
    test set was intentionally built with a harder, shifted distribution. This is
    internally consistent — the "shift" we measure is attributable to the graph
    step, not to a train/test scaling mismatch.

NOTE 2 — Compare like-with-like (scaled vs scaled), not raw vs scaled.
    profile_before.html (the human-readable reference) profiles the RAW-valued
    data. But after.parquet holds MinMax-SCALED features. Comparing raw magnitudes
    against scaled ones would show a "shift" that is purely an artifact of scaling,
    not of the graph transformation. So for the .compare() report, this stage
    rebuilds a SCALED version of the before-features (same preprocessing as the
    graph stage) and compares scaled-before vs scaled-after on the shared columns.
    That way any distribution shift is genuinely attributable to the graph step.
────────────────────────────────────────────────────────────────────────────

Run (from repo root, dataprep env):
    conda run -n dataprep python src/stages/profile_after.py \
        --data-dir data \
        --after data/processed/after.parquet \
        --out-dir reports

Requires (dataprep env):
    pip install ydata-profiling pandas pyarrow "setuptools<81" mlflow
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Repo-root import shim (same pattern as the other stages).
# ---------------------------------------------------------------------------
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # noqa: E402


def build_scaled_before(data_dir: str):
    """Rebuild the SCALED 'before' feature table from the raw TEST set, using
    the SAME preprocessing as graph_transform.py.

    See NOTE 2 above: we compare scaled-before vs scaled-after so the measured
    shift is attributable to the graph step, not to scaling. Returns a DataFrame
    of scaled features with the original feature-column names.
    """
    test_path = os.path.join(data_dir, "KDDTest+.txt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Could not find {test_path}. Run graph_transform (or the main "
            f"pipeline) once so NSL-KDD is present in '{data_dir}/'."
        )

    df = pd.read_csv(test_path, names=COLS, header=None).drop("difficulty_level", axis=1)
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    feat_cols = [c for c in df.columns if c != "label"]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feat_cols].values.astype(np.float32))
    return pd.DataFrame(X, columns=feat_cols)


def make_report(df, title, minimal=True):
    from ydata_profiling import ProfileReport
    return ProfileReport(df, title=title, minimal=minimal)


def maybe_log_mlflow(out_dir, artifacts):
    try:
        import mlflow
    except ImportError:
        print("  (mlflow not installed — skipping experiment logging)")
        return
    mlflow.set_experiment("xai-nids-graph-augmentation")
    with mlflow.start_run(run_name="profile_after"):
        mlflow.log_param("stage", "profile_after")
        for path in artifacts:
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="profiling")
        print("  Logged comparison + after reports to MLflow.")


def main():
    parser = argparse.ArgumentParser(
        description="Profile the graph-augmented table and compare before/after."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--after", default="data/processed/after.parquet")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[profile_after] repo root: {REPO_ROOT}")
    print(f"[profile_after] loading augmented table: {args.after}")
    after = pd.read_parquet(args.after)

    # Split the augmented table into: shared original features, and the new
    # graph features. Bookkeeping columns are excluded from profiling.
    graph_feats = ["graph_degree", "graph_pagerank"]
    bookkeeping = ["label_binary", "is_explained", "orig_test_idx"]
    orig_feat_cols = [c for c in after.columns
                      if c not in graph_feats + bookkeeping]
    print(f"  {len(orig_feat_cols)} original features, "
          f"{len(graph_feats)} graph features, {after.shape[0]} rows.")

    # --- Artifact 1: standalone profile of the augmented table (feats + graph) ---
    after_profile_df = after[orig_feat_cols + graph_feats]
    print("[profile_after] building standalone 'after' profile...")
    after_report = make_report(after_profile_df, "NSL-KDD — after graph transform")
    after_html = os.path.join(args.out_dir, "profile_after.html")
    after_report.to_file(after_html)
    print(f"  wrote {after_html}")

    # --- Artifact 2: before/after comparison on SHARED, SCALED features ---
    # NOTE 2: rebuild scaled 'before' so the comparison is scaled-vs-scaled.
    print("[profile_after] rebuilding scaled 'before' features for fair comparison...")
    before_scaled = build_scaled_before(args.data_dir)
    # Align to exactly the shared original feature columns (order-safe).
    shared = [c for c in orig_feat_cols if c in before_scaled.columns]
    before_shared = before_scaled[shared]
    after_shared = after[shared]
    print(f"  comparing on {len(shared)} shared scaled features "
          f"(before: {before_shared.shape[0]} rows, after: {after_shared.shape[0]} rows).")

    before_rep = make_report(before_shared, "before (scaled, full test)")
    after_rep = make_report(after_shared, "after (scaled, 10K graph sample)")
    print("[profile_after] generating .compare() report...")
    comparison = before_rep.compare(after_rep)
    comp_html = os.path.join(args.out_dir, "profile_comparison.html")
    comparison.to_file(comp_html)
    print(f"  wrote {comp_html}")

    # --- Artifact 3: dedicated profile of the NEW graph features ---
    # These have no 'before' counterpart, so they get their own report rather
    # than being forced into the comparison.
    print("[profile_after] profiling the new graph features on their own...")
    gf_report = make_report(after[graph_feats],
                            "New graph features (degree, pagerank)", minimal=False)
    gf_html = os.path.join(args.out_dir, "profile_graph_feats.html")
    gf_report.to_file(gf_html)
    print(f"  wrote {gf_html}")

    if not args.no_mlflow:
        maybe_log_mlflow(args.out_dir, [comp_html, after_html, gf_html])

    print("[profile_after] done.")
    print(f"\n  Deliverable: {comp_html}")
    print(f"  (open it in a browser — this is the before/after graphDB comparison)")


if __name__ == "__main__":
    main()  