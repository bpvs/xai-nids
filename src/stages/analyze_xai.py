"""
Stage: analyze_xai
==================
Computes UNBIASED cross-method agreement from the results graph (or directly from
attributions.json), and renders heatmaps.

Two agreement metrics, both from the RAW stored importances:
  1. Jaccard across MULTIPLE k (default 5,10,15,20) -- top-k set overlap between
     method pairs. Multiple k avoids the arbitrary-threshold bias of a single k.
  2. Rank correlation (Spearman) over the FULL feature ordering -- captures
     agreement Jaccard's hard top-k cliff misses (e.g. rank 10 vs 11).

Data source: reads attributions.json directly by default (simple, exact). Pass
--from-graph to instead pull importances back out of FalkorDB (proves the graph
round-trips), which is slower but demonstrates the results-store queries.

Outputs (reports/):
  jaccard_<model>_k<k>.png            consistency heatmap per model per k
  rankcorr_<model>.png                Spearman heatmap per model
  jaccard_curve.png                   Jaccard vs k, per method-pair (pooled models)
  analysis_summary.json               all numbers, machine-readable

Run (from repo root, dataprep env):
    conda run -n dataprep python src/stages/analyze_xai.py \
        --attributions data/processed/attributions.json \
        --out-dir reports \
        --ks 5 10 15 20 \
        --slice all

Requires (dataprep env): pip install numpy scipy matplotlib seaborn falkordb
"""

import argparse
import json
import os
from itertools import combinations

import numpy as np


def load_from_json(path):
    with open(path) as f:
        doc = json.load(f)
    return doc["feature_columns"], doc["attributions"]


def load_from_graph(graph_name, host, port):
    """Pull importances back out of FalkorDB, reconstructing the same structure
    as attributions.json. Demonstrates the results-store round-trip."""
    from falkordb import FalkorDB
    db = FalkorDB(host=host, port=port)
    g = db.select_graph(graph_name)

    feats = [r[0] for r in g.query("MATCH (f:Feature) RETURN f.name").result_set]

    rows_q = g.query(
        "MATCH (e:Explanation)-[a:ASSIGNS]->(ft:Feature) "
        "RETURN e.model, e.method, e.sample_index, e.label, ft.name, a.importance"
    ).result_set

    # Reassemble {model: {sample_index: {label, methods:{method:{feat:imp}}}}}
    tmp = {}
    for model, method, sidx, label, fname, imp in rows_q:
        m = tmp.setdefault(model, {})
        s = m.setdefault(int(sidx), {"sample_index": int(sidx),
                                     "label": int(label), "methods": {}})
        s["methods"].setdefault(method, {})[fname] = float(imp)
    attributions = {model: [s for _, s in sorted(sm.items())]
                    for model, sm in tmp.items()}
    return feats, attributions


def imp_vector(feat_imps, feature_cols):
    """Dense importance vector aligned to feature_cols (missing = 0)."""
    return np.array([feat_imps.get(f, 0.0) for f in feature_cols], dtype=float)


def topk_set(vec, k):
    """Indices of the top-k features by |importance|."""
    return set(np.argsort(np.abs(vec))[::-1][:k].tolist())


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def filter_rows(rows, slice_mode):
    if slice_mode == "attack":
        return [r for r in rows if int(r["label"]) == 1]
    if slice_mode == "normal":
        return [r for r in rows if int(r["label"]) == 0]
    return rows


def analyze_model(rows, feature_cols, ks, slice_mode):
    """Return per-pair Jaccard@k (dict k->pair->mean) and Spearman (pair->mean),
    for the given class slice."""
    from scipy.stats import spearmanr

    rows = filter_rows(rows, slice_mode)
    # methods present in this model
    methods = sorted({m for r in rows for m in r["methods"].keys()})

    jac = {k: {} for k in ks}
    rho = {}
    for m1, m2 in combinations(methods, 2):
        pair = f"{m1}|{m2}"
        per_k = {k: [] for k in ks}
        rhos = []
        for r in rows:
            if m1 not in r["methods"] or m2 not in r["methods"]:
                continue
            v1 = imp_vector(r["methods"][m1], feature_cols)
            v2 = imp_vector(r["methods"][m2], feature_cols)
            for k in ks:
                per_k[k].append(jaccard(topk_set(v1, k), topk_set(v2, k)))
            # Spearman on absolute importances (rank by magnitude).
            if np.any(v1) and np.any(v2):
                rr = spearmanr(np.abs(v1), np.abs(v2)).correlation
                if not np.isnan(rr):
                    rhos.append(rr)
        for k in ks:
            if per_k[k]:
                jac[k][pair] = float(np.mean(per_k[k]))
        if rhos:
            rho[pair] = float(np.mean(rhos))
    return methods, jac, rho


def heatmap(matrix, labels, title, path, vmin=0, vmax=1, cmap="YlGnBu"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    n = len(labels)
    plt.figure(figsize=(1.4 * n + 2, 1.2 * n + 1.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=labels,
                yticklabels=labels, cmap=cmap, vmin=vmin, vmax=vmax, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def pair_matrix(methods, pair_scores):
    """Build a symmetric matrix from {'"m1|m2"': score} over `methods`."""
    n = len(methods)
    mat = np.eye(n)
    idx = {m: i for i, m in enumerate(methods)}
    for pair, val in pair_scores.items():
        m1, m2 = pair.split("|")
        i, j = idx[m1], idx[m2]
        mat[i, j] = mat[j, i] = val
    return mat


def main():
    ap = argparse.ArgumentParser(description="Unbiased XAI agreement analysis.")
    ap.add_argument("--attributions", default="data/processed/attributions.json")
    ap.add_argument("--from-graph", action="store_true",
                    help="Pull importances from FalkorDB instead of the JSON.")
    ap.add_argument("--graph-name", default="xai_results")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6379)
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--slice", choices=["all", "attack", "normal"], default="all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.from_graph:
        print(f"[analyze] reading importances from FalkorDB graph "
              f"'{args.graph_name}'...")
        feature_cols, attributions = load_from_graph(
            args.graph_name, args.host, args.port)
    else:
        print(f"[analyze] reading {args.attributions}...")
        feature_cols, attributions = load_from_json(args.attributions)

    print(f"[analyze] {len(feature_cols)} features, "
          f"{len(attributions)} models, slice='{args.slice}', ks={args.ks}")

    summary = {"slice": args.slice, "ks": args.ks, "models": {}}
    # For the pooled Jaccard-vs-k curve.
    pooled_curve = {}  # pair -> {k: [values across models]}

    for model_name, rows in attributions.items():
        methods, jac, rho = analyze_model(rows, feature_cols, args.ks, args.slice)
        summary["models"][model_name] = {
            "methods": methods,
            "jaccard": {str(k): jac[k] for k in args.ks},
            "spearman": rho,
        }
        print(f"\n  [{model_name}] methods: {methods}")

        # Jaccard heatmaps, one per k.
        for k in args.ks:
            if not jac[k]:
                continue
            mat = pair_matrix(methods, jac[k])
            heatmap(mat, methods,
                    f"Jaccard@{k} — {model_name} ({args.slice})",
                    os.path.join(args.out_dir,
                                 f"jaccard_{model_name}_k{k}_{args.slice}.png"))
            for pair, val in jac[k].items():
                pooled_curve.setdefault(pair, {}).setdefault(k, []).append(val)
            print(f"    Jaccard@{k}: " +
                  ", ".join(f"{p.replace('|','-')}={v:.2f}" for p, v in jac[k].items()))

        # Spearman heatmap.
        if rho:
            mat = pair_matrix(methods, rho)
            heatmap(mat, methods,
                    f"Spearman rank corr — {model_name} ({args.slice})",
                    os.path.join(args.out_dir,
                                 f"rankcorr_{model_name}_{args.slice}.png"),
                    vmin=-1, vmax=1, cmap="RdBu_r")
            print("    Spearman: " +
                  ", ".join(f"{p.replace('|','-')}={v:.2f}" for p, v in rho.items()))

    # Pooled Jaccard-vs-k curve (averaged across models).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for pair, kvals in sorted(pooled_curve.items()):
            xs = sorted(kvals.keys())
            ys = [float(np.mean(kvals[k])) for k in xs]
            plt.plot(xs, ys, marker="o", label=pair.replace("|", "-"))
        plt.xlabel("k (top-k features)")
        plt.ylabel("Mean Jaccard (pooled across models)")
        plt.title(f"Jaccard vs k — {args.slice}")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        curve_path = os.path.join(args.out_dir, f"jaccard_curve_{args.slice}.png")
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"\n[analyze] wrote Jaccard-vs-k curve -> {curve_path}")
    except Exception as e:
        print(f"  (curve plot skipped: {e})")

    out_json = os.path.join(args.out_dir, f"analysis_summary_{args.slice}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] wrote {out_json}")
    print("[analyze] done.")


if __name__ == "__main__":
    main()