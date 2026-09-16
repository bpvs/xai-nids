# Attack vs Normal: Is the Minority Class Harder to *Explain* Consistently?

**Sources:** `analysis_summary_attack.json`, `analysis_summary_normal.json`
**Slices:** 57 attack samples, 43 normal samples (of 100 stratified).
**Metric:** Jaccard@5 (mean top-5 overlap per method pair) and `agreement_features` (features the pair co-selects, with per-slice frequency).
**No significance test.** Everything below is a **directional trend**, and §4 argues most single gaps are within plausible noise.

---

## Bottom line first

There is a **directionally consistent** signal that the **SHAP–LIME pair agrees less when explaining attacks than when explaining normal traffic**, in all three models. The effect is small for CNN, moderate for XGBoost, and largest for RF. This is consistent with — but does **not prove** — the idea that the attack class is harder to explain *consistently*, not just harder to detect.

Two things stop this from being a clean "attacks are harder to explain" claim:
1. **Counterfactual-based pairs do not follow the pattern** (and RF's reverse it), so the trend is specific to attribution–attribution agreement, not explanations in general.
2. The **95%-zero feature cut** and the **heterogeneity of the attack class by construction** are confounds that could produce the same signal without any intrinsic "difficulty." At n≈50 per slice, the smaller gaps are not distinguishable from sampling noise.

---

## 1. Does agreement differ between attack and normal? (Jaccard@5)

Signed gap = normal − attack (positive ⇒ methods agree *more* on normal).

### SHAP–LIME specifically — consistent across all three models

| Model | attack | normal | gap (normal − attack) |
|-------|--------|--------|------------------------|
| CNN | 0.321 | 0.346 | **+0.025** |
| XGBoost | 0.205 | 0.258 | **+0.053** |
| RF | 0.384 | 0.506 | **+0.122** |

All three point the same way: LIME and SHAP land on more of the same top-5 features for normal connections than for attacks. Magnitude ordering CNN < XGBoost < RF.

### All pairs, per model

**CNN (5 methods):**

| Pair | attack | normal | gap |
|------|--------|--------|-----|
| LIME–SHAP | 0.321 | 0.346 | +0.025 |
| Counterfactual–LIME | 0.073 | 0.132 | +0.059 |
| LIME–Saliency | 0.140 | 0.187 | +0.047 |
| SHAP–Saliency | 0.105 | 0.132 | +0.027 |
| Counterfactual–Saliency | 0.177 | 0.203 | +0.026 |
| Counterfactual–Grad-CAM | 0.129 | 0.151 | +0.022 |
| Grad-CAM–LIME | 0.126 | 0.139 | +0.013 |
| Counterfactual–SHAP | 0.101 | 0.112 | +0.011 |
| Grad-CAM–Saliency | 0.086 | 0.081 | −0.005 |
| **Grad-CAM–SHAP** | 0.129 | 0.069 | **−0.060** |

Mostly normal > attack, but weakly. Grad-CAM–SHAP reverses noticeably.

**XGBoost (3 methods):**

| Pair | attack | normal | gap |
|------|--------|--------|-----|
| LIME–SHAP | 0.205 | 0.258 | +0.053 |
| Counterfactual–SHAP | 0.185 | 0.204 | +0.020 |
| Counterfactual–LIME | 0.326 | 0.328 | +0.002 (flat) |

**RF (3 methods):**

| Pair | attack | normal | gap |
|------|--------|--------|-----|
| LIME–SHAP | 0.384 | 0.506 | +0.122 |
| Counterfactual–LIME | 0.286 | 0.254 | −0.032 |
| **Counterfactual–SHAP** | 0.258 | 0.202 | **−0.056** |

**Reading:** the "normal > attack" trend is carried by the **attribution–attribution (SHAP–LIME)** comparison. The moment a **counterfactual** enters the pair, the sign becomes mixed and in RF flips outright (both RF Counterfactual pairs agree *more* on attacks). So the honest statement is narrow: *SHAP and LIME are less mutually consistent on attacks*, not *"attacks are harder to explain."*

---

## 2. Do the anchor features change between slices?

Yes — and the shift is informative. Across models the **core anchors are shared**, but (a) normal concentrates more tightly on them, and (b) attack uniquely pulls in the **rerror (rejected-connection error) family**. Comparing the SHAP–LIME `agreement_features` per slice:

**CNN (SHAP–LIME co-selection frequency):**

| Feature | attack | normal |
|---------|--------|--------|
| logged_in | 0.56 | 0.72 |
| srv_serror_rate | 0.53 | 0.65 |
| dst_host_serror_rate | 0.37 | 0.60 |
| dst_host_srv_serror_rate | 0.26 | 0.35 |
| **dst_host_rerror_rate** | **0.35** | 0.09 |
| **rerror_rate** | **0.14** | (absent) |

Same serror/`logged_in` backbone in both, but the **rerror family is a top-4 anchor for attacks and nearly vanishes for normal**, and every normal frequency is higher (tighter agreement).

**XGBoost (SHAP–LIME):**

| Feature | attack | normal |
|---------|--------|--------|
| src_bytes | 0.77 | 0.79 |
| dst_bytes | 0.19 | 0.30 |
| **dst_host_same_srv_rate** | 0.14 | **0.51** |
| dst_host_same_src_port_rate | 0.14 | 0.16 |
| **dst_host_rerror_rate** | **0.16** | (absent) |

`src_bytes` anchors both; normal adds a strong `dst_host_same_srv_rate` signal (0.51), attack again surfaces `dst_host_rerror_rate`.

**RF (SHAP–LIME):**

| Feature | attack | normal |
|---------|--------|--------|
| dst_bytes | 0.95 | 0.95 |
| src_bytes | 0.70 | 0.81 |
| flag | 0.60 | 0.74 |
| **dst_host_srv_count** | 0.07 | **0.37** |
| **logged_in** | 0.04 | **0.33** |
| **dst_host_rerror_rate** | **0.12** | (absent) |

Byte-volume + `flag` backbone shared; normal recruits connection-state features (`dst_host_srv_count`, `logged_in`) far more strongly; attack again brings in `dst_host_rerror_rate`.

**Pattern (directional):** normal explanations converge tightly on a small set of "benign-state" features (serror rates / `logged_in` / `same_srv_rate` / `flag`), while attack explanations keep those anchors *but* spread onto the rerror family and additional scattered features — lower per-feature frequencies overall. More spread ⇒ lower Jaccard. This is a coherent mechanism for the §1 gap, but see the confounds in §4.

---

## 3. Connection to the detection gap (recall 0.61–0.68 vs ~0.97)

The known result is that all three models detect attacks far worse than normal (attack recall 0.61–0.68 vs normal recall ~0.97). The explanation-consistency gap here **runs in the same direction**: the class that is harder to *detect* also shows lower *explanation agreement* (for SHAP–LIME).

A plausible **common cause** ties them together: the attack class is heterogeneous (mixed DoS/Probe, plus residual R2L/U2R) and — after the 95% cut removed the rare-attack detectors — under-served by the surviving 26 features. A class that lacks clean, canonical features would be **both** harder to classify **and** harder to explain with mutual consistency, because there is no single stable feature story for the model to expose.

**But this link is suggestive, not demonstrated**, for a concrete reason: the Jaccard here is computed over **all** attack samples, **not** filtered to the misclassified ones. So we are *not* measuring "explanations of the errors." To actually connect the two, we would need per-sample (correctness × pairwise-agreement) — i.e. do the mis-detected attacks also have the least method agreement? That correlation is not in this data. The current evidence shows two gaps pointing the same way with a plausible shared driver, which is weaker than a causal link.

---

## 4. How much of this is noise? (rigor at n ≈ 57/43)

Reasons for caution, roughly in order of importance:

1. **Sample size.** Each Jaccard is a mean over ~43–57 per-sample values. The summary JSON reports only means (no per-sample variance), so a precise SE isn't computable, but a back-of-envelope with an assumed per-sample sd ≈ 0.25 gives SE ≈ 0.03–0.04 per slice and ≈ 0.05 for a normal−attack **difference**. Against that yardstick:
   - CNN SHAP–LIME **+0.025 → within noise.**
   - XGBoost SHAP–LIME **+0.053 → ~1 SE, suggestive only.**
   - RF SHAP–LIME **+0.122 → ~2–2.5 SE, most likely real but untested.**
   Only the RF gap would plausibly survive a test; the CNN gap should not be leaned on.

2. **Consistency-across-models is itself weak evidence.** All three models showing normal > SHAP–LIME−attack is directionally reassuring, but the three runs share the same samples and dataset, so they are **not independent** — do not treat "3 for 3" as three confirmations.

3. **Feature-cut confound (large).** The 95%-zero cut deleted the U2R/R2L-specific detectors. Attack samples that were those types now have *no natural explanation features left*, forcing every method onto proxy features and inflating dispersion. The lower attack agreement could be substantially an **artifact of the cut**, not intrinsic difficulty. The recurring appearance of the rerror family on attacks (a Probe/scan signal) is consistent with methods reaching for whatever survived.

4. **Heterogeneity by construction.** "Attack" pools multiple attack types; "normal" is one class. A multi-type class will show more per-sample explanation variance — and thus lower mean Jaccard — **regardless** of any difficulty, purely because different attack types legitimately have different top features. Part of the gap is definitional, not a model property.

5. **Counterfactual pairs contradict the trend** (§1), which alone rules out a blanket "attacks are harder to explain" reading.

---

## Verdict

- **Directional yes, for SHAP–LIME:** attribution methods agree less on attacks than on normal traffic in all three models (CNN +0.025, XGBoost +0.053, RF +0.122). Only the RF gap is large enough to likely exceed noise.
- **The anchor features shift, not just the numbers:** normal explanations concentrate on benign-state features; attack explanations retain those anchors but add the rerror family and spread out.
- **It rhymes with the recall gap** and a shared cause (heterogeneous, post-cut under-featured attack class) is plausible — but the data here cannot link inconsistent *explanations* to incorrect *detections* at the sample level, so treat the connection as a hypothesis, not a result.
- **Do not over-claim:** counterfactual pairs break the pattern, the feature cut is a live confound, attack heterogeneity is baked in, and at n≈50 per slice only the RF effect is robust to eyeball-level noise.
