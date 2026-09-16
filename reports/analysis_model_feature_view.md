# Model & Feature View — What Each Model Anchors On, and What the 95% Cut Made It Blind To

**Sources:** `results.json` (metrics, per-class recall, fidelity, consistency, counterfactual success), `attributions.json` (feature list + raw importances), `analysis_summary_*.json` (`full_consensus`, `agreement_features`).
**Scope:** proof-of-concept, NSL-KDD only, 100 samples, **26 features after an advisor-directed 95%-zero cut**. All feature→attack-type mappings below are standard NSL-KDD domain knowledge; all agreement numbers are directional at n=100.

---

## 0. Model scoreboard (from `results.json`)

| Model | Acc | F1 | AUC | Attack recall | Normal recall | SHAP fidelity | SHAP–LIME consist. | CF success |
|-------|-----|----|----|---------------|---------------|---------------|--------------------|------------|
| CNN | 0.767 | 0.764 | **0.832** | 0.615 | 0.968 | 0.406 | 0.363 | 0.40 |
| XGBoost | **0.805** | **0.804** | 0.969 | **0.679** | 0.971 | **0.575** | 0.343 | **0.85** |
| RF | 0.785 | 0.783 | 0.970 | 0.644 | 0.971 | 0.236 | **0.484** | 0.44 |

Common to all three: **normal recall ~0.97, attack recall 0.61–0.68.** The models are near-perfect on normal and miss roughly a third of attacks. §3 argues part of that miss is *structural* — a direct consequence of the feature cut.

---

## 1. The surviving 26 vs the dropped 15 (NSL-KDD taxonomy)

The full NSL-KDD feature set is 41, grouped into four families. The 95%-zero cut kept 26 and dropped 15. Mapped to the standard taxonomy:

**Kept (26):**
- **Basic (6/9):** `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`
- **Content (1/13):** `logged_in` — *only one content feature survived*
- **Time-traffic (9/9):** `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate`
- **Host-traffic (10/10):** all `dst_host_*` features

**Dropped (15):**
- **Basic (3):** `land`, `wrong_fragment`, `urgent`
- **Content (12):** `hot`, `num_failed_logins`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`, `num_outbound_cmds`, `is_host_login`, `is_guest_login`

**The cut fell almost entirely on the content family.** It kept every traffic/statistical feature and gutted the payload/host-state features. That is the single most important fact for interpreting everything below: these features are near-zero for most connections (hence removed by a 95%-zero rule), but they are *not* uninformative — they are precisely the sparse, high-signal detectors for the rare attack classes.

---

## 2. Each model anchors on a *different* feature family

Reading the `full_consensus` (tree 3-way) and strongest-pair `agreement_features` (all-slice):

**CNN → connection-error / state family.** Its LIME–SHAP anchors are `logged_in` (63/100), `srv_serror_rate` (58), `dst_host_serror_rate` (47), `dst_host_srv_serror_rate` (30), `dst_host_rerror_rate` (24). The CNN reads the **serror/rerror rate signature** — the classic footprint of SYN-flood DoS (elevated SYN-error rates) and scanning (REJ-error rates).

**XGBoost → byte-volume + protocol, `src_bytes`-led.** 3-way consensus: `src_bytes` (61/100), `dst_bytes` (15), `protocol_type` (5), `dst_host_same_src_port_rate` (4). XGBoost reads **"how much data was sent and over what protocol."**

**RF → byte-volume, `dst_bytes`-led, plus `flag`.** 3-way consensus: `dst_bytes` (77/100), `src_bytes` (62), `count` (6). Note `flag` is a strong LIME–SHAP anchor (66/100) but drops out of the 3-way consensus because the counterfactual method doesn't select it — so `flag` is an *attribution* anchor for RF, not a boundary-flipping one.

**Takeaway:** two models (XGBoost, RF) key on **byte counts** and differ mainly in direction (`src_bytes` vs `dst_bytes`) and whether `flag` matters; the CNN keys on an entirely different family — **traffic error-rate statistics**. No model's explanations lean on the surviving basic/protocol features and the traffic families simultaneously; each has picked one lane.

---

## 3. What the models can and cannot see (attack-type blind spots)

Mapping surviving features to NSL-KDD's four attack categories:

| Attack class | Natural detectors | Status after cut | Model visibility |
|--------------|-------------------|------------------|------------------|
| **DoS** (neptune/smurf/back…) | serror rates, `src_bytes`/`dst_bytes`, `count` | **Kept** | **Visible** — SYN/ICMP floods leave clear traffic-stat footprints |
| **Probe** (satan/ipsweep/portsweep…) | rerror rates, `diff_srv_rate`, `dst_host_*` counts, `service` | **Kept** | **Visible** — scan patterns survive |
| **R2L** (guess_passwd/ftp_write/imap…) | `hot`, `num_failed_logins`, `is_guest_login`, `num_file_creations` | **Dropped** (only `logged_in` left) | **Largely blind** — looks like normal traffic at flow level |
| **U2R** (buffer_overflow/rootkit/perl…) | `root_shell`, `su_attempted`, `num_root`, `num_shells`, `num_access_files` | **All dropped** | **Effectively blind** |

Two sub-notes:
- Even within **DoS**, the *malformed-packet* subtypes (land, teardrop, pod) lost their specific detectors (`land`, `wrong_fragment`, `urgent`); only the *flooding* subtypes (neptune → serror, smurf → counts) remain cleanly detectable.
- **R2L and U2R are the classes the models cannot resolve.** These attacks look like ordinary connections at the network-flow level; their signal lives entirely in the content features that were removed. With those gone, such samples are almost forced into the "normal" prediction.

**This directly feeds the attack-recall gap.** Attack recall sits at 0.61–0.68 not only because attacks are intrinsically hard, but because any R2L/U2R samples in the test set are, by construction, missing the features needed to flag them — they get classified as normal and drag attack recall down. The models are effectively **DoS/Probe detectors** with a thin `logged_in`-based proxy for everything else. The comparatively small size of R2L/U2R in NSL-KDD is the main reason recall didn't collapse further.

---

## 4. Tension between performance and what the explanations rely on

**Tension A — the higher-AUC models rely on the *cruder* features.** The two top-AUC models (XGBoost 0.969, RF 0.970) anchor on raw **byte counts**; the model that reads the semantically richer **attack-signature features** (CNN, serror/rerror rates) has the **lowest AUC (0.832)**. The interpretable-sounding feature family did not buy better discrimination here — byte volume alone separates DoS/normal well enough that the trees win on the metric while resting on a shallower story.

**Tension B — RF: high agreement, low fidelity.** RF has the **highest method agreement** (SHAP–LIME consistency 0.484) yet the **lowest fidelity** (SHAP 0.236) — masking its top features barely dents the prediction. Read together, this signals **redundant feature usage**: `dst_bytes`/`src_bytes` dominate the top-5 that the methods agree on, but the model can fall back on correlated features when they're removed. High agreement here reflects a *concentrated* explanation, not a *load-bearing* one. Its low counterfactual success (0.44) fits: a redundant, distributed boundary is hard to flip.

**Tension C — XGBoost is the best-aligned.** Highest accuracy (0.805), highest fidelity (SHAP 0.575), and highest counterfactual success (0.85). The 0.85 CF rate implies a **sharp, low-dimensional boundary**: small perturbations to a couple of byte/protocol features flip the label most of the time. Here the explanation is both concentrated *and* faithful — the model genuinely leans on the features its explanations name.

**Tension D — CNN's gradient explanations are barely faithful.** Saliency fidelity 0.074 and Grad-CAM 0.110 are near-floor, while its SHAP fidelity (0.406) is respectable. The CNN's *perturbation* explanations point at load-bearing features, but its *gradient* explanations (Saliency/Grad-CAM) do not — a caution against trusting the gradient methods on this 1D-CNN. Combined with the lowest CF success (0.40), the CNN has the least legible decision surface of the three.

---

## 5. How the cleaning decision bounds every conclusion above

- **The 95% cut defines the feature vocabulary of the whole study.** Because it removed almost the entire content family, the explanations can only ever be about DoS/Probe traffic signatures. "Which features does the model use to detect intrusions?" really means "…to detect the DoS/Probe subset it can still see." Findings should not be read as general NIDS explanations.
- **The consensus features are partly artifacts of the cut.** `src_bytes`/`dst_bytes`/serror-rates dominate in part because the features that would compete with them (content detectors) are gone. On the full 41-feature set, R2L/U2R samples would likely pull explanations toward `hot`, `num_failed_logins`, `root_shell`, etc., and the model-by-model feature-family split could look different.
- **The attack-recall ceiling is co-determined by the cut, not just the models.** Any claim comparing model "attack detection ability" is confounded by the fact that all three were denied the R2L/U2R detectors equally.
- **NSL-KDD's own limits still apply.** It is a decades-old, synthetic benchmark with known redundancy and non-representativeness for modern traffic. This is a proof-of-concept pipeline; none of the feature-importance rankings should be lifted into operational claims.

---

## Bottom line

- **Different anchors:** CNN → error-rate/connection-state features; XGBoost → `src_bytes`-led byte volume + protocol; RF → `dst_bytes`-led byte volume + `flag`. Each model committed to one feature family.
- **Structural blindness:** the cut removed 12 of 13 content features, i.e. the R2L and U2R detectors — the models are effectively DoS/Probe detectors, and that blindness is a direct contributor to the 0.61–0.68 attack recall.
- **Performance ≠ explanation depth:** the two high-AUC tree models rest on crude byte counts (XGBoost faithfully/sharply, RF redundantly/low-fidelity), while the CNN's richer error-rate features yield the lowest AUC and the least faithful gradient explanations.
- **Bounded interpretation:** every ranking here is a property of *this* 26-feature cleaned NSL-KDD slice at n=100, not of network intrusion detection in general.
