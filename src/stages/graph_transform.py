"""
Stage: graph_transform
=======================
Builds a kNN feature-similarity graph over a 10K stratified sample of the
NSL-KDD TEST set in FalkorDB, extracts graph-derived node features
(degree + PageRank), and writes an AUGMENTED feature table (original features
+ graph features) plus a manifest recording exactly which rows were sampled
and which are the "explained" records.

This produces the "after" data for the YData before/after comparison, and the
augmented table is what the train/explain/evaluate stages consume.

Pipeline logic:
    1. Load + preprocess the TEST set exactly as data_utils.load_data() does
       (MinMax-scaled, categoricals encoded) so features match the main pipeline.
    2. Draw a deterministic stratified 10K sample (preserves attack fraction).
    3. Deterministically mark the 5 "explained" records (2 normal, 3 attack)
       INSIDE that sample, so downstream stages agree on which rows they are.
    4. Compute kNN edges (k=10) via scikit-learn over the scaled features.
    5. Load nodes + bidirectional edges into FalkorDB.
    6. Compute degree (Cypher) and PageRank (CALL algo.pageRank) per node.
    7. Write augmented parquet (features + degree + pagerank + flags) and a
       manifest JSON.

Why sklearn does the kNN and FalkorDB does PageRank:
    Nearest-neighbour search over 10K vectors is a numpy/sklearn job; asking
    Cypher to do O(n^2) similarity would be slow and awkward. FalkorDB is the
    graph store + algorithm engine (PageRank, degree), which is what it's good at.

Run (from repo root, dataprep env, with FalkorDB container running):
    conda run -n dataprep python src/stages/graph_transform.py \
        --data-dir data \
        --out data/processed/after.parquet \
        --manifest data/processed/sample_manifest.json

Requires (dataprep env):
    pip install falkordb scikit-learn pandas pyarrow mlflow
FalkorDB must be running:
    docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Make the repo's own modules importable regardless of where this file lives.
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
from sklearn.neighbors import NearestNeighbors  # noqa: E402


# Fixed seed so the sample and explained-record selection are reproducible
# and shared across stages.
SEED = 42


def load_and_preprocess_test(data_dir: str):
    """Load the TEST set and preprocess it the same way data_utils.load_data does.

    We re-implement the TEST-only path here (rather than call load_data(), which
    fits the scaler on TRAIN and returns splits) so we can scale the test set
    self-consistently and keep row identity. Returns (X_scaled, y, feat_cols, raw_df).
    """
    test_path = os.path.join(data_dir, "KDDTest+.txt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Could not find {test_path}. Run the main pipeline once so "
            f"data_utils.load_data() downloads NSL-KDD into '{data_dir}/'."
        )

    df = pd.read_csv(test_path, names=COLS, header=None).drop("difficulty_level", axis=1)

    # Encode categoricals (same columns as the main pipeline).
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    y = (df["label"].map(lambda x: ATTACKS.get(x, "unknown")) != "normal").astype(int).values

    feat_cols = [c for c in df.columns if c != "label"]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feat_cols].values.astype(np.float32))
    return X, y, feat_cols, df


def stratified_sample(X, y, n_sample, seed=SEED):
    """Draw a stratified sample of size n_sample preserving the class ratio.
    Returns the selected indices (into the original test arrays)."""
    rng = np.random.default_rng(seed)
    norm_idx = np.where(y == 0)[0]
    att_idx = np.where(y == 1)[0]

    frac = len(att_idx) / len(y)
    n_att = int(round(n_sample * frac))
    n_norm = n_sample - n_att
    n_att = min(n_att, len(att_idx))
    n_norm = min(n_norm, len(norm_idx))

    sel = np.concatenate([
        rng.choice(norm_idx, n_norm, replace=False),
        rng.choice(att_idx, n_att, replace=False),
    ])
    rng.shuffle(sel)
    return np.sort(sel)


def pick_explained(y_sample, seed=SEED):
    """Deterministically pick 2 normal + 3 attack rows (positions within the
    sample) to serve as the shared 'explained' records for downstream stages."""
    rng = np.random.default_rng(seed + 1)  # different stream from the sample
    norm_pos = np.where(y_sample == 0)[0]
    att_pos = np.where(y_sample == 1)[0]
    chosen = np.concatenate([
        rng.choice(norm_pos, 2, replace=False),
        rng.choice(att_pos, 3, replace=False),
    ])
    return np.sort(chosen)


def build_knn_edges(X_sample, k=10):
    """Return an array of directed (src, dst) index pairs for each node's k
    nearest neighbours (excluding self). We add both directions when we load
    into the graph, so the graph is effectively undirected/bidirectional."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")  # +1 to skip self
    nn.fit(X_sample)
    _, idx = nn.kneighbors(X_sample)
    edges = []
    for src in range(idx.shape[0]):
        for dst in idx[src, 1:]:  # skip column 0 (self)
            edges.append((src, int(dst)))
    return np.array(edges, dtype=int)


def build_graph_and_features(X_sample, edges, k, graph_name="nslkdd_knn"):
    """Load nodes + bidirectional edges into FalkorDB, compute degree + PageRank.
    Returns (degree_array, pagerank_array), aligned to sample row order."""
    from falkordb import FalkorDB

    n = X_sample.shape[0]
    db = FalkorDB(host="localhost", port=6379)
    g = db.select_graph(graph_name)

    # Start clean in case the stage is re-run.
    try:
        g.delete()
    except Exception:
        pass
    g = db.select_graph(graph_name)

    print(f"  Creating {n} nodes in FalkorDB...")
    # Batch node creation with UNWIND for speed.
    ids = list(range(n))
    for start in range(0, n, 1000):
        chunk = ids[start:start + 1000]
        g.query(
            "UNWIND $ids AS i CREATE (:Flow {id: i})",
            {"ids": chunk},
        )

    # Index on id so edge MATCHes are fast.
    g.query("CREATE INDEX FOR (f:Flow) ON (f.id)")

    print(f"  Creating {len(edges) * 2} directed edges (bidirectional kNN)...")
    # Bidirectional: add (a)->(b) and (b)->(a) so PageRank has incoming links
    # for every node. Deduplicate to avoid double-counting reciprocal pairs.
    edge_pairs = set()
    for s, d in edges:
        edge_pairs.add((int(s), int(d)))
        edge_pairs.add((int(d), int(s)))
    edge_list = [[s, d] for s, d in edge_pairs]

    for start in range(0, len(edge_list), 2000):
        chunk = edge_list[start:start + 2000]
        g.query(
            "UNWIND $pairs AS p "
            "MATCH (a:Flow {id: p[0]}), (b:Flow {id: p[1]}) "
            "CREATE (a)-[:SIMILAR]->(b)",
            {"pairs": chunk},
        )

    # Degree per node (count of SIMILAR relationships).
    print("  Computing degree...")
    degree = np.zeros(n, dtype=float)
    res = g.query(
        "MATCH (f:Flow)-[r:SIMILAR]-() RETURN f.id AS id, count(r) AS deg"
    )
    for row in res.result_set:
        degree[int(row[0])] = float(row[1])

    # PageRank via FalkorDB's native algorithm.
    print("  Computing PageRank (CALL algo.pageRank)...")
    pagerank = np.zeros(n, dtype=float)
    res = g.query(
        "CALL algo.pageRank('Flow', 'SIMILAR') YIELD node, score "
        "RETURN node.id AS id, score"
    )
    for row in res.result_set:
        pagerank[int(row[0])] = float(row[1])

    return degree, pagerank


def maybe_log_mlflow(out_path, manifest, n_edges):
    try:
        import mlflow
    except ImportError:
        print("  (mlflow not installed — skipping experiment logging)")
        return
    mlflow.set_experiment("xai-nids-graph-augmentation")
    with mlflow.start_run(run_name="graph_transform"):
        mlflow.log_param("stage", "graph_transform")
        mlflow.log_param("n_sample", manifest["n_sample"])
        mlflow.log_param("k", manifest["k"])
        mlflow.log_param("population", "test_only")
        mlflow.log_metric("n_edges_directed", n_edges)
        mlflow.log_metric("attack_fraction_sample", manifest["attack_fraction_sample"])
        if os.path.exists(out_path):
            mlflow.log_artifact(out_path, artifact_path="processed")
        print("  Logged params + augmented table to MLflow.")


def main():
    parser = argparse.ArgumentParser(
        description="Build kNN similarity graph in FalkorDB and extract features."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/processed/after.parquet")
    parser.add_argument("--manifest", default="data/processed/sample_manifest.json")
    parser.add_argument("--n-sample", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--graph-name", default="nslkdd_knn")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    print(f"[graph_transform] repo root: {REPO_ROOT}")
    print(f"[graph_transform] loading + preprocessing TEST set from {args.data_dir}...")
    X, y, feat_cols, raw_df = load_and_preprocess_test(args.data_dir)
    print(f"  Test set: {X.shape[0]:,} rows × {X.shape[1]} features.")

    print(f"[graph_transform] drawing stratified {args.n_sample} sample (seed={SEED})...")
    sel = stratified_sample(X, y, args.n_sample, seed=SEED)
    X_s, y_s = X[sel], y[sel]
    print(f"  Sample: {len(sel):,} rows | attack fraction {y_s.mean():.3f}")

    explained_pos = pick_explained(y_s, seed=SEED)
    print(f"  Explained records (positions in sample): {explained_pos.tolist()} "
          f"labels={y_s[explained_pos].tolist()}")

    print(f"[graph_transform] building kNN edges (k={args.k}) via sklearn...")
    edges = build_knn_edges(X_s, k=args.k)
    print(f"  {len(edges):,} directed kNN edges (pre-bidirectional).")

    print("[graph_transform] loading graph into FalkorDB + extracting features...")
    try:
        degree, pagerank = build_graph_and_features(
            X_s, edges, args.k, graph_name=args.graph_name
        )
    except Exception as e:
        print("\n[graph_transform] ERROR talking to FalkorDB.")
        print("  Is the container running?  docker ps  should show 'falkordb'.")
        print("  Start it with:")
        print("    docker run -d --name falkordb -p 6379:6379 -p 3000:3000 "
              "falkordb/falkordb:latest")
        raise

    # Assemble the augmented table: original scaled features + graph features + flags.
    aug = pd.DataFrame(X_s, columns=feat_cols)
    aug["graph_degree"] = degree
    aug["graph_pagerank"] = pagerank
    aug["label_binary"] = y_s
    aug["is_explained"] = False
    aug.loc[explained_pos, "is_explained"] = True
    aug["orig_test_idx"] = sel  # traceability back to the raw test set

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    aug.to_parquet(args.out, index=False)
    print(f"[graph_transform] wrote augmented table → {args.out}  "
          f"({aug.shape[0]} rows × {aug.shape[1]} cols)")

    manifest = {
        "seed": SEED,
        "population": "test_only",
        "n_sample": int(len(sel)),
        "k": int(args.k),
        "attack_fraction_sample": float(y_s.mean()),
        "feature_columns": feat_cols,
        "graph_features": ["graph_degree", "graph_pagerank"],
        "explained_positions": explained_pos.tolist(),
        "explained_orig_test_idx": sel[explained_pos].tolist(),
        "explained_labels": y_s[explained_pos].tolist(),
        "graph_name": args.graph_name,
        "output_parquet": args.out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[graph_transform] wrote manifest → {args.manifest}")

    if not args.no_mlflow:
        maybe_log_mlflow(args.out, manifest, n_edges=len(edges) * 2)

    print("[graph_transform] done.")


if __name__ == "__main__":
    main()