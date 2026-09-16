# XAI Method Agreement Analysis — NSL-KDD (26 features, 100 samples)

**Sources:** `analysis_summary_{all,attack,normal}.json`, `jaccard_*_k5_*.png`
**Metric:** Jaccard@5 = mean top-5 overlap between a method pair, per sample; `full_consensus` = fraction of samples where *all* of a model's methods share a feature in their top-5.
**Setup:** NSL-KDD reduced to 26 features by a 95%-zero cut (which removed rare-attack U2R/R2L detectors); 100 stratified samples (57 attack, 43 normal); **no significance test.**

> All numeric differences below of order ≤ 0.1 Jaccard, and all consensus fractions built on single-digit counts, are **directional only** at n=100. Read this document as pattern description, not hypothesis confirmation.

---

## 1. Jaccard@5 per model

### CNN — 5 methods (Counterfactual, Grad-CAM, LIME, SHAP, Saliency)

| Pair | all | attack | normal |
|------|-----|--------|--------|
| **LIME–SHAP** | **0.33** | **0.32** | **0.35** |
| Counterfactual–Saliency | 0.19 | 0.18 | 0.20 |
| LIME–Saliency | 0.16 | 0.14 | 0.19 |
| Counterfactual–Grad-CAM | 0.14 | 0.13 | 0.15 |
| Grad-CAM–LIME | 0.13 | 0.13 | 0.14 |
| SHAP–Saliency | 0.12 | 0.10 | 0.13 |
| Counterfactual–SHAP | 0.11 | 0.10 | 0.11 |
| Grad-CAM–SHAP | 0.10 | 0.13 | 0.07 |
| Counterfactual–LIME | 0.10 | 0.07 | 0.13 |
| Grad-CAM–Saliency | 0.08 | 0.09 | 0.08 |

**LIME–SHAP dominates** (0.33) — roughly double the next pair. Everything else sits in a dim 0.08–0.19 band; the heatmap is one bright cell surrounded by near-uniform low agreement. Weakest pair is **Grad-CAM–Saliency (0.08)**, which is notable because both are gradient methods (see §2).

### XGBoost — 3 methods (Counterfactual, LIME, SHAP)

| Pair | all | attack | normal |
|------|-----|--------|--------|
| **Counterfactual–LIME** | **0.33** | **0.33** | **0.33** |
| LIME–SHAP | 0.23 | 0.20 | 0.26 |
| Counterfactual–SHAP | 0.19 | 0.18 | 0.20 |

**Note the exception:** here the strongest pair is **Counterfactual–LIME, not LIME–SHAP.** Both anchor on `protocol_type` + `src_bytes` (see §3). This breaks the "perturbation pair wins" pattern seen in CNN and RF, and is the most non-obvious finding in the set.

### RF — 3 methods (Counterfactual, LIME, SHAP)

| Pair | all | attack | normal |
|------|-----|--------|--------|
| **LIME–SHAP** | **0.44** | **0.38** | **0.51** |
| Counterfactual–LIME | 0.27 | 0.29 | 0.25 |
| Counterfactual–SHAP | 0.23 | 0.26 | 0.20 |

**RF LIME–SHAP (0.44 overall, 0.51 normal) is the strongest agreement anywhere in the study.** But this is partly a degeneracy artifact: `dst_bytes` and `src_bytes` occupy the top-5 in almost every sample for both methods (§3), which mechanically inflates overlap. High Jaccard here reflects a *concentrated* explanation space as much as method concordance.

---

## 2. Which pairs agree, and why

**Perturbation pair (LIME–SHAP) is the consistent front-runner for CNN and RF.** Both estimate feature effect by perturbing / marginalizing inputs, so they share an underlying question — "how does output move when this feature changes?" — and land on similar top-5 sets.

**XGBoost is the exception:** Counterfactual–LIME edges out LIME–SHAP. On this model, the counterfactual boundary and LIME's local fit both key on the same two dominant features (`protocol_type`, `src_bytes`), so they coincide more than SHAP does. This is model-specific and should not be generalized from three methods on one dataset.

**Gradient methods disagree with each other.** Grad-CAM–Saliency = 0.08, near the floor, despite both being gradient-based. They compute different objects: Saliency is ∂output/∂input (raw per-feature sensitivity); Grad-CAM weights the 1D-conv activation maps by pooled gradients (a smoothed, position-level summary). On tabular input these diverge.

**Counterfactual is an outlier in CNN but an anchor in XGBoost.** This is partly *definitional*: a counterfactual reports the minimum perturbation to flip the prediction (a contrastive question), not a feature-importance ranking. Comparing its top-5 against attribution top-5 mixes two explanation types, so low Counterfactual agreement in CNN is expected, not a method defect.

---

## 3. What the pairs actually converge on (`agreement_features`)

**CNN — connection-error / state features.** The strongest pair, LIME–SHAP, co-selects `logged_in` (63/100), `srv_serror_rate` (58/100), `dst_host_serror_rate` (47/100), `dst_host_srv_serror_rate` (30/100). These are classic DoS/SYN-flood signatures (elevated connection-error rates, login-state shifts) — the dominant attack signal surviving the 95% cut. On the normal slice the same features rise (`logged_in` 72%, `srv_serror_rate` 65%, `dst_host_serror_rate` 60%), consistent with normal traffic being more homogeneous so the same features top-rank across samples.

**XGBoost — volume + protocol.** `src_bytes` is central to every pair (LIME–SHAP 78/100, Counterfactual–SHAP 73/100, Counterfactual–LIME 61/100). The strong Counterfactual–LIME pair adds `protocol_type` (86/100 overall, and a striking **97.7%** on the normal slice) and `duration` (38/100). The near-universal `protocol_type` agreement on normal traffic likely reflects normal samples clustering in a narrow protocol range, making it trivially top-ranked.

**RF — two byte-volume features dominate.** LIME–SHAP co-selects `dst_bytes` (95/100) and `src_bytes` (75/100), plus `flag` (66/100). On normal these intensify (`dst_bytes` 95%, `src_bytes` 81%, `flag` 74%). This concentration is what drives RF's headline Jaccard.

**Cross-model contrast (directional):** CNN explanations key on *error-rate / connection-state* features; both tree models key on *byte-volume* features (`src_bytes`, `dst_bytes`) plus `protocol_type`/`flag`. That divergence is a property of what each **architecture learned**, and is orthogonal to how well the XAI methods agree *within* a model — the two should not be conflated.

---

## 4. Full consensus: CNN 5-way vs tree 3-way

### Tree 3-way — robust
- **RF (all):** `dst_bytes` 77%, `src_bytes` 62%, then a sparse tail (`count` 6%, `dst_host_rerror_rate` 2%, `flag` 1%).
- **XGBoost (all):** `src_bytes` 61%, `dst_bytes` 15%, sparse tail.

Two conditions make tree consensus strong: only **3** sets must intersect, and the attributions are concentrated in 2–3 features that top-rank for most samples. On the attack slice RF tightens further (`dst_bytes` 82%, `src_bytes` 63%); XGBoost stays `src_bytes`-led (67%). The RF-vs-XGBoost lead swap (dst_bytes vs src_bytes) is a plausible model-weighting difference but untested at n=100.

### CNN 5-way — near-empty, and mechanically so
- **all:** `srv_serror_rate` 3/100, `dst_host_serror_rate` 1/100.
- **attack (n=57):** `srv_serror_rate` 3 (5%), `dst_host_serror_rate` 1 (2%).
- **normal (n=43): empty.**

This is **expected from the pairwise numbers, not a pathology.** A feature enters the 5-way consensus only if it sits in the top-5 of *all five* methods for the same sample. The pairwise Jaccards are the ceiling: LIME–SHAP already caps at 0.33, and every Counterfactual/gradient pairing runs 0.08–0.19. Requiring simultaneous agreement across five sets — including the weak Counterfactual and Grad-CAM–Saliency links — forces the joint overlap toward zero. The empty normal result is the same effect amplified by the smaller slice (43 samples). The near-empty CNN 5-way says the five methods **ask different questions**; it is *not* evidence that the CNN's explanations are unreliable.

---

## 5. Attack vs normal

For the strong pairs, agreement is generally **lower on attack than normal** (directional):

| Model | Pair | attack | normal |
|-------|------|--------|--------|
| RF | LIME–SHAP | 0.38 | 0.51 |
| XGBoost | LIME–SHAP | 0.20 | 0.26 |
| CNN | LIME–SHAP | 0.32 | 0.35 |

Two non-exclusive explanations:
1. **Attack heterogeneity.** The 57 attack samples mix DoS/Probe (plus residual R2L/U2R), which engage different features, so top-5 sets vary more sample-to-sample and depress mean Jaccard. Normal traffic is more uniform.
2. **Feature-cut bias.** The 95% cut stripped the U2R/R2L-specific detectors, leaving a DoS/Probe-leaning feature set that never forms a clean signature for the full range of attacks — raising within-class explanation variance for attacks.

Counter-note: XGBoost's Counterfactual–LIME is flat (~0.33) across slices, so the attack<normal gap is pair-dependent, not universal. The attack slice is also *larger* (57 vs 43), so its estimates are marginally more stable — the gap is not merely a sample-size artifact.

---

## 6. Limitations (load-bearing)

- **n=100, no significance test.** Any Jaccard gap ≲ 0.1, and any consensus feature resting on single-digit counts, is indistinguishable from noise. Ordering is directional.
- **The 95%-zero cut shapes the conclusions.** Removing 15 features (including rare-attack detectors) concentrates the space onto DoS/Probe-relevant features. The consensus features (`src_bytes`, `dst_bytes`, error-rate family) are as much a product of the cut as of model learning; the full 41-feature set would likely differ.
- **Counterfactual ≠ attribution.** Its top-5 answers "what flips the prediction," not "what mattered." Low Counterfactual agreement (and its odd role as XGBoost's *strongest* pair) partly reflects this type mismatch.
- **Grad-CAM & Saliency are CNN-only.** No cross-model comparison of gradient methods is possible; the gradient-vs-perturbation contrast is observable only within the CNN.
- **RF's high Jaccard is inflated by feature degeneracy.** Two features saturating the top-5 make overlap easy; it is not a clean signal of superior method concordance.
