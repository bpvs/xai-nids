"""
Stage: load_graph
=================
Loads the ML + XAI RESULTS into FalkorDB as a queryable results graph.

This is the OUTPUT-side use of FalkorDB (results store), NOT a feature transform.

Reads:
    data/processed/results.json        (per-model metrics + per-class)
    data/processed/attributions.json   (raw per-sample per-feature importances)

Graph schema (graph name: xai_results):
    (:Model    {name, accuracy, f1, auc, attack_recall, normal_recall})
    (:Sample   {index, label, label_raw})
    (:Method   {name})
    (:Feature  {name})
    (:Explanation {model, method, sample_index, label})
        (:Model)-[:PRODUCED]->(:Explanation)
        (:Explanation)-[:USING]->(:Method)
        (:Explanation)-[:FOR_SAMPLE]->(:Sample)
        (:Explanation)-[:ASSIGNS {importance}]->(:Feature)

Run (from repo root, dataprep env, FalkorDB running):
    conda run -n dataprep python src/stages/load_graph.py \
        --results data/processed/results.json \
        --attributions data/processed/attributions.json \
        --graph-name xai_results

Requires (dataprep env): pip install falkordb
FalkorDB running:
    docker start falkordb   (or docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest)
"""

import argparse
import json
import os


def connect(graph_name, host, port):
    from falkordb import FalkorDB
    db = FalkorDB(host=host, port=port)
    return db, db.select_graph(graph_name)


def main():
    ap = argparse.ArgumentParser(description="Load XAI results into FalkorDB.")
    ap.add_argument("--results", default="data/processed/results.json")
    ap.add_argument("--attributions", default="data/processed/attributions.json")
    ap.add_argument("--graph-name", default="xai_results")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6379)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    with open(args.results) as f:
        results = json.load(f)
    with open(args.attributions) as f:
        attr_doc = json.load(f)

    feature_cols = attr_doc["feature_columns"]
    attributions = attr_doc["attributions"]   # {model: [ {sample_index,label,methods}, ...]}

    print(f"[load_graph] connecting to FalkorDB {args.host}:{args.port} "
          f"graph='{args.graph_name}'")
    try:
        db, g = connect(args.graph_name, args.host, args.port)
    except Exception:
        print("\n[load_graph] ERROR connecting to FalkorDB. Is it running?")
        print("  docker start falkordb   (or the full docker run command)")
        raise

    # Fresh start so re-runs are clean.
    try:
        g.delete()
    except Exception:
        pass
    db, g = connect(args.graph_name, args.host, args.port)

    # --- Feature nodes -------------------------------------------------------
    print(f"[load_graph] creating {len(feature_cols)} Feature nodes...")
    g.query("UNWIND $names AS n CREATE (:Feature {name: n})", {"names": feature_cols})
    g.query("CREATE INDEX FOR (f:Feature) ON (f.name)")

    # --- Method nodes --------------------------------------------------------
    methods_present = set()
    for model_rows in attributions.values():
        for row in model_rows:
            methods_present.update(row["methods"].keys())
    methods_present = sorted(methods_present)
    print(f"[load_graph] creating {len(methods_present)} Method nodes: {methods_present}")
    g.query("UNWIND $m AS n CREATE (:Method {name: n})", {"m": methods_present})
    g.query("CREATE INDEX FOR (m:Method) ON (m.name)")

    # --- Sample nodes (from the first model's rows; identical across models) --
    any_model = next(iter(attributions))
    sample_rows = attributions[any_model]
    samples = [{"index": r["sample_index"], "label": int(r["label"])} for r in sample_rows]
    # attach raw labels from results manifest if present
    print(f"[load_graph] creating {len(samples)} Sample nodes...")
    g.query(
        "UNWIND $s AS row CREATE (:Sample {index: row.index, label: row.label})",
        {"s": samples},
    )
    g.query("CREATE INDEX FOR (s:Sample) ON (s.index)")

    # --- Model nodes (with metrics) ------------------------------------------
    print(f"[load_graph] creating {len(results['models'])} Model nodes...")
    for name, m in results["models"].items():
        met = m["metrics"]
        pc = m.get("per_class", {})
        g.query(
            "CREATE (:Model {name:$name, accuracy:$acc, f1:$f1, auc:$auc, "
            "attack_recall:$ar, normal_recall:$nr})",
            {
                "name": name,
                "acc": float(met.get("accuracy", 0)),
                "f1": float(met.get("f1", 0)),
                "auc": float(met.get("auc", 0)),
                "ar": float(pc.get("attack", {}).get("recall", 0)),
                "nr": float(pc.get("normal", {}).get("recall", 0)),
            },
        )
    g.query("CREATE INDEX FOR (m:Model) ON (m.name)")

    # --- Explanations + ASSIGNS edges ---------------------------------------
    # One Explanation node per (model, method, sample); ASSIGNS edges carry the
    # raw importance to each feature that method scored for that sample.
    print("[load_graph] creating Explanation nodes + ASSIGNS edges "
          "(this is the bulk load)...")
    n_expl = 0
    n_edges = 0
    for model_name, rows in attributions.items():
        for row in rows:
            s_idx = row["sample_index"]
            label = int(row["label"])
            for method, feat_imps in row["methods"].items():
                # Create the Explanation and wire it to model/method/sample.
                g.query(
                    "MATCH (mo:Model {name:$model}), (me:Method {name:$method}), "
                    "(sa:Sample {index:$sidx}) "
                    "CREATE (e:Explanation {model:$model, method:$method, "
                    "sample_index:$sidx, label:$label}) "
                    "CREATE (mo)-[:PRODUCED]->(e) "
                    "CREATE (e)-[:USING]->(me) "
                    "CREATE (e)-[:FOR_SAMPLE]->(sa)",
                    {"model": model_name, "method": method, "sidx": s_idx, "label": label},
                )
                n_expl += 1
                # ASSIGNS edges to features (batch via UNWIND).
                pairs = [{"f": f, "imp": float(v)} for f, v in feat_imps.items()]
                if pairs:
                    g.query(
                        "MATCH (e:Explanation {model:$model, method:$method, "
                        "sample_index:$sidx}) "
                        "UNWIND $pairs AS p "
                        "MATCH (ft:Feature {name: p.f}) "
                        "CREATE (e)-[:ASSIGNS {importance: p.imp}]->(ft)",
                        {"model": model_name, "method": method, "sidx": s_idx,
                         "pairs": pairs},
                    )
                    n_edges += len(pairs)

    print(f"[load_graph] created {n_expl} Explanation nodes, {n_edges} ASSIGNS edges.")

    # Quick verification counts.
    for label, q in [
        ("Model", "MATCH (n:Model) RETURN count(n)"),
        ("Method", "MATCH (n:Method) RETURN count(n)"),
        ("Sample", "MATCH (n:Sample) RETURN count(n)"),
        ("Feature", "MATCH (n:Feature) RETURN count(n)"),
        ("Explanation", "MATCH (n:Explanation) RETURN count(n)"),
        ("ASSIGNS", "MATCH ()-[r:ASSIGNS]->() RETURN count(r)"),
    ]:
        c = g.query(q).result_set[0][0]
        print(f"    {label:12s}: {c}")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("xai-nids-graph-augmentation")
            with mlflow.start_run(run_name="load_graph"):
                mlflow.log_param("stage", "load_graph")
                mlflow.log_param("graph_name", args.graph_name)
                mlflow.log_metric("n_explanations", n_expl)
                mlflow.log_metric("n_assigns_edges", n_edges)
            print("[load_graph] logged to MLflow.")
        except ImportError:
            pass

    print(f"[load_graph] done. Explore at http://localhost:3000 "
          f"(graph '{args.graph_name}').")


if __name__ == "__main__":
    main()