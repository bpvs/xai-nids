"""
Stage: analyze_xai
==================
Computes UNBIASED cross-method agreement from the results graph (or directly from
attributions.json), and renders heatmaps.

Agreement metric from the RAW stored importances:
  - Jaccard at k=5 (default) -- top-k set overlap between method pairs.
    --ks overrides the k values if needed.

Spearman rank correlation is OPT-IN (off by default) via --spearman. It is kept
in the code (recoverable) but not computed or plotted unless requested, per
current analysis feedback. When enabled it captures agreement Jaccard's hard
top-k cliff misses (e.g. rank 10 vs 11).

Everything is sliceable by class: all / attack-only / normal-only.

Why this reads RAW importances rather than precomputed scores: it keeps the
analysis unbiased -- any k, any slice, either metric, computed on demand.

Data source: reads attributions.json directly by default (simple, exact). Pass
--from-graph to instead pull importances back out of FalkorDB (proves the graph
round-trips), which is slower but demonstrates the results-store queries.

Outputs (reports/):
  jaccard_<model>_k<k>.png            consistency heatmap per model per k
  jaccard_curve.png                   Jaccard vs k, per method-pair (pooled models)
  analysis_summary.json               all numbers, machine-readable
  rankcorr_<model>.png                Spearman heatmap (ONLY if --spearman given)

Run (from repo root, dataprep env):
    conda run -n dataprep python src/stages/analyze_xai.py \
        --attributions data/processed/attributions.json \
        --out-dir reports \
        --ks 5 \
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


def analyze_model(rows, feature_cols, ks, slice_mode, do_spearman=False):
    """Return per-pair Jaccard@k (dict k->pair->mean) and, if do_spearman,
    Spearman (pair->mean) for the given class slice. Spearman is off by default."""
    if do_spearman:
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
            # Spearman on absolute importances (rank by magnitude) -- opt-in.
            if do_spearman and np.any(v1) and np.any(v2):
                rr = spearmanr(np.abs(v1), np.abs(v2)).correlation
                if not np.isnan(rr):
                    rhos.append(rr)
        for k in ks:
            if per_k[k]:
                jac[k][pair] = float(np.mean(per_k[k]))
        if rhos:
            rho[pair] = float(np.mean(rhos))
    return methods, jac, rho



def aggregate_pairwise_agreement(rows, feature_cols, k, slice_mode, top_n=10):
    """Per method-pair: how often each feature is in BOTH methods' top-k across
    the slice. Returns {pair: [{feature, both_count, fraction}, ...]} ranked by
    frequency, capped at top_n. This is the 'what do these two agree on' view.
    """
    rows = filter_rows(rows, slice_mode)
    methods = sorted({m for r in rows for m in r["methods"].keys()})
    out = {}
    for m1, m2 in combinations(methods, 2):
        counts = {}
        n_pair = 0
        for r in rows:
            if m1 not in r["methods"] or m2 not in r["methods"]:
                continue
            n_pair += 1
            t1 = topk_set(imp_vector(r["methods"][m1], feature_cols), k)
            t2 = topk_set(imp_vector(r["methods"][m2], feature_cols), k)
            for i in (t1 & t2):
                counts[i] = counts.get(i, 0) + 1
        if n_pair == 0:
            continue
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        out[f"{m1}|{m2}"] = [
            {"feature": feature_cols[i], "both_count": c,
             "fraction": round(c / n_pair, 4)} for i, c in ranked
        ]
    return out, methods


def aggregate_full_consensus(rows, feature_cols, k, slice_mode, top_n=10):
    """Per MODEL: how often each feature is in the top-k of ALL applicable methods
    at once, aggregated across the slice. For the CNN this is a 5-way intersection
    (SHAP,LIME,Counterfactual,Saliency,Grad-CAM); for the trees a 3-way
    (SHAP,LIME,Counterfactual). Returns (list ranked by fraction, methods, n_used).

    NOTE: expect this to be SPARSE, especially on the CNN -- requiring a feature in
    ALL methods' top-k is a strict filter that passes through the weakly-agreeing
    gradient methods. An empty/short result is itself a finding (methods diverge),
    not a bug.
    """
    rows = filter_rows(rows, slice_mode)
    methods = sorted({m for r in rows for m in r["methods"].keys()})
    counts = {}
    n_used = 0
    for r in rows:
        present = [m for m in methods if m in r["methods"]]
        if len(present) < 2:
            continue
        n_used += 1
        # intersection of every present method's top-k for this sample
        inter = None
        for m in present:
            tk = topk_set(imp_vector(r["methods"][m], feature_cols), k)
            inter = tk if inter is None else (inter & tk)
        for i in (inter or set()):
            counts[i] = counts.get(i, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    consensus = [
        {"feature": feature_cols[i], "all_methods_count": c,
         "fraction": round(c / n_used, 4) if n_used else 0.0}
        for i, c in ranked
    ]
    return consensus, methods, n_used


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
    ap.add_argument("--ks", type=int, nargs="+", default=[5])
    ap.add_argument("--spearman", action="store_true",
                    help="Also compute Spearman rank correlation + rankcorr "
                         "heatmaps (off by default).")
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
        methods, jac, rho = analyze_model(rows, feature_cols, args.ks, args.slice,
                                          do_spearman=args.spearman)
        summary["models"][model_name] = {
            "methods": methods,
            "jaccard": {str(k): jac[k] for k in args.ks},
        }
        if args.spearman:
            summary["models"][model_name]["spearman"] = rho

        # Aggregate agreement features (pairwise) at the primary k (first --ks).
        kprime = args.ks[0]
        pair_agree, _ = aggregate_pairwise_agreement(
            rows, feature_cols, kprime, args.slice, top_n=10)
        summary["models"][model_name]["agreement_features"] = pair_agree
        print(f"    -- pairwise agreement features (k={kprime}) --")
        for pair, feats in pair_agree.items():
            top = ", ".join(f"{d['feature']}({int(d['fraction']*100)}%)" for d in feats[:5])
            print(f"      {pair.replace('|','-')}: {top}")

        # Full consensus: features in ALL methods' top-k (5-way CNN / 3-way trees).
        consensus, cons_methods, n_used = aggregate_full_consensus(
            rows, feature_cols, kprime, args.slice, top_n=10)
        summary["models"][model_name]["full_consensus"] = {
            "n_methods": len(cons_methods),
            "methods": cons_methods,
            "n_samples_used": n_used,
            "features": consensus,
        }
        nway = len(cons_methods)
        if consensus:
            top = ", ".join(f"{d['feature']}({int(d['fraction']*100)}%)" for d in consensus[:5])
            print(f"    -- {nway}-way consensus (feature in ALL {nway} methods' top-{kprime}): {top}")
        else:
            print(f"    -- {nway}-way consensus (k={kprime}): EMPTY "
                  f"(no feature in all {nway} methods' top-{kprime} -- methods diverge)")
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

        # Spearman heatmap (only when --spearman was requested).
        if args.spearman and rho:
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