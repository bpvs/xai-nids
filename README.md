# XAI for Network Intrusion Detection — Multi-Model

Proof-of-concept comparing 5 XAI techniques across 3 model architectures on the NSL-KDD dataset. Part of PhD research on cross-domain XAI evaluation.

## What it does

Trains **three models** on NSL-KDD intrusion detection and applies XAI methods to each:

| Method | CNN (1D) | XGBoost | Random Forest |
|---|---|---|---|
| SHAP | ✓ (Kernel) | ✓ (Tree) | ✓ (Tree) |
| LIME | ✓ | ✓ | ✓ |
| Counterfactuals | ✓ (TF gradients) | ✓ (numerical) | ✓ (numerical) |
| Saliency | ✓ | — | — |
| Grad-CAM (1D) | ✓ | — | — |

Each model gets its own results folder so you can compare them directly. A cross-model comparison plot is generated at the end.

## Quick start

```bash
pip install -r requirements.txt
python xai_pipeline.py            # full run
python xai_pipeline.py --fast     # fast dev mode
```

## Project structure

```
xai-nids/
├── src/
│   ├── xai_pipeline.py           # main orchestrator (baseline experiment)
│   ├── data_utils.py             # NSL-KDD loading + preprocessing
│   ├── explainers.py             # all 5 XAI methods
│   ├── evaluation.py             # fidelity, consistency, cross-model comparison
│   ├── preprocess.py             # (legacy) earlier preprocessing helpers
│   ├── models/
│   │   ├── base.py               # ModelWrapper abstract base
│   │   ├── cnn.py                # 1D-CNN (Keras/TensorFlow)
│   │   ├── xgboost_model.py      # XGBoost classifier
│   │   └── rf.py                 # Random Forest
│   └── stages/                   # pipeline stages (graph-augmentation extension)
│       ├── profile_before.py     # YData profile of raw NSL-KDD
│       ├── graph_transform.py    # FalkorDB kNN graph + graph features
│       └── profile_after.py      # YData after-profile + before/after comparison
├── data/
│   ├── raw/  (KDDTrain+.txt, KDDTest+.txt)
│   └── processed/                # after.parquet + sample_manifest.json
├── reports/                      # profiling HTML reports
├── results_cnn/                  # CNN outputs (auto-created)
├── results_xgboost/              # XGBoost outputs (auto-created)
├── results_rf/                   # Random Forest outputs (auto-created)
├── cross_model_comparison.png
└── report.json
```

*Note: source files now live under `src/`. Internal imports remain flat
(`from data_utils import ...`); the stage scripts add the repo's `src/` to the
path automatically, so they run from the repo root regardless.*

## Apple Silicon optimizations

- **CNN**: Metal GPU via `tensorflow-metal`, memory growth enabled, `@tf.function`-compiled prediction, GradientTape for counterfactuals (~40× faster than finite differences)
- **XGBoost**: `tree_method='hist'` + `n_jobs=-1` for all M1 cores
- **Random Forest**: `n_jobs=-1` for parallel tree building
- **SHAP**: uses `TreeExplainer` (exact, fast) for tree models and `KernelExplainer` for the CNN

---

## Graph-Augmentation Extension (FalkorDB + YData)

An experimental extension that tests whether **graph-structure features** change
the behaviour of the five explainers on the NIDS domain. The raw tabular
NSL-KDD data is turned into a **feature-similarity graph**, graph-derived
features are extracted and appended to the tabular features, and the data is
profiled **before and after** the graph step for comparison.

> **Status:** proof-of-concept extension on the NIDS reference implementation.
> It is **not yet part of the dissertation Chapter 3 methodology** — kept as
> engineering / reproducibility work until results justify writing it up.

### Pipeline (dataprep env)

```
raw NSL-KDD
   │
   ├─▶ profile_before.py   →  YData profile of the RAW data          (reports/profile_before.html)
   │
   ├─▶ graph_transform.py  →  FalkorDB kNN similarity graph          (data/processed/after.parquet
   │                          + degree & PageRank features            + data/processed/sample_manifest.json)
   │
   └─▶ profile_after.py    →  YData after-profile + .compare()        (reports/profile_after.html,
                              + dedicated graph-feature profile        reports/profile_comparison.html,
                                                                       reports/profile_graph_feats.html)
```

- **Profiling tool:** open-source `ydata-profiling` (no license/token required).
  Used for the "before/after graphDB" comparison via `ProfileReport.compare()`.
- **Graph DB:** FalkorDB, run as a Docker service (OpenCypher + GraphBLAS).
- **Graph construction:** each sampled record is a node; **kNN (k=10)** edges by
  **cosine similarity** over the scaled features (standard NSL-KDD has no IPs, so
  edges are similarity-based, not host/flow-based). Edges are loaded
  **bidirectionally** so PageRank is well-defined for every node.
- **Graph features extracted:** `graph_degree` and `graph_pagerank` (kept minimal
  and interpretable). Computed in FalkorDB via `CALL algo.pageRank(...)` and a
  degree Cypher query; kNN edges are computed in scikit-learn and loaded into the
  graph (the right tool for each job).

### Running it

Requires Docker (for FalkorDB) and a separate `dataprep` conda env (keeps
`ydata-profiling`/`falkordb` away from the TensorFlow-Metal pins in the model env).

```bash
# 0. FalkorDB service
docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest

# 1. dataprep env
conda create -n dataprep python=3.11
conda activate dataprep
pip install ydata-profiling falkordb scikit-learn node2vec networkx pandas pyarrow mlflow "setuptools<81"

# 2. run the stages (from repo root)
conda run -n dataprep python src/stages/profile_before.py  --data-dir data --out reports/profile_before.html
conda run -n dataprep python src/stages/graph_transform.py --data-dir data --out data/processed/after.parquet --manifest data/processed/sample_manifest.json
conda run -n dataprep python src/stages/profile_after.py   --data-dir data --after data/processed/after.parquet --out-dir reports
```

Runs are tracked in **MLflow** (experiment `xai-nids-graph-augmentation`):

```bash
conda run -n dataprep mlflow ui   # http://localhost:5000
```

### Key parameters and decisions

- **Population:** TEST set only (matches where the explained records are drawn from).
- **Sample:** stratified **10,000 rows** (seed = 42), preserving the test-set attack
  fraction (~0.569).
- **Reproducibility:** `sample_manifest.json` records the seed, the sampled test
  indices, and the shared **"explained" record positions**, so downstream
  train/explain/evaluate stages use the *same* records instead of re-rolling
  `np.random.choice`.

### Methodological notes (read before interpreting the comparison)

1. **Test-set-only, self-consistent scaling.** `graph_transform.py` preprocesses the
   TEST set on its own MinMax statistics (not the train-fitted scaler in
   `data_utils.load_data()`), so the sample is scaled self-consistently while
   preserving row identity. The whole before/after comparison is therefore a
   **test-vs-test-graph** comparison. The test set's attack fraction (~0.569)
   differs from the train+test figure (~0.481) because NSL-KDD's test set was
   built with a harder, shifted distribution — this is expected and internally
   consistent.

2. **Comparison is scaled-vs-scaled, not raw-vs-scaled.** `profile_before.html`
   profiles the *raw* data (human-readable reference), but `after.parquet` is
   *scaled*. To avoid a false "shift" that is purely a scaling artifact,
   `profile_after.py` rebuilds a **scaled** version of the before-features and
   compares scaled-before vs scaled-after on the shared columns. Any shift in
   `profile_comparison.html` is therefore attributable to the graph step.

3. **Package migration (`ydata-profiling` → `fg-data-profiling`).** `ydata-profiling`
   is deprecated in favour of `fg-data-profiling` (import becomes
   `import data_profiling`). Do **not** migrate mid-experiment — regenerate all
   profiles together after switching. A `SettingWithCopyWarning` raised from inside
   ydata-profiling during report generation is harmless (library-internal).

### Observations so far

- Sampling preserved the feature distributions (before/after profiles of the 41
  shared features are near-identical) — the meaningful change is the two added
  graph features.
- `graph_degree` and `graph_pagerank` correlate strongly (**r ≈ 0.94**) — expected
  for a homogeneous kNN graph, and relevant when interpreting XAI attributions
  over these two near-redundant features.

## References

- SHAP: Lundberg & Lee (2017) — [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- LIME: Ribeiro et al. (2016) — [arXiv:1602.04938](https://arxiv.org/abs/1602.04938)
- Grad-CAM: Selvaraju et al. (2017) — [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- NSL-KDD: Tavallaee et al. (2009)
- FalkorDB: <https://docs.falkordb.com/>
- YData Profiling: <https://docs.profiling.ydata.ai/>
