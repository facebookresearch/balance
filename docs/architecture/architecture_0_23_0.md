# Architecture — balance 0.23.0: Outcome Modelling (`outcomes_hat`) and Doubly-Robust Estimation

> **Status:** DESIGN / implementation plan (feature target: 0.23.0, currently unreleased; 0.22.0 has shipped).
> Unlike [`architecture_0_19_0.md`](architecture_0_19_0.md) — a *retrospective* of a shipped
> refactor — this document is a *forward-looking* design that is refined as implementation
> questions are resolved. Resolved decisions are logged in [§0](#0-decisions-log); [§13](#13-design-decisions-status--deferred-phases)
> records the decision status and explicitly deferred later phases. As of the round-6 review, no
> items remain **(OPEN)**.
>
> For the evergreen overview see [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md); for the review
> checklist see [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md).

---

## 0. Decisions log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-15 | **`outcomes_hat` is the canonical spelling everywhere — public and internal — as a clean rename (no alias, no `FutureWarning`, no migration).** Public: `outcomes_hat()`, `df_outcomes_hat`, `outcomes_hat_columns`. Internal: `_column_roles` key `"predicted"` → `"outcomes_hat"`, the `_create()`/`from_frame()` parameter `predicted_outcome_columns` → `outcomes_hat_columns`, internal locals (`predicted_list` → `outcomes_hat_list`, etc.), the overlap-validation dict key, and the protocol member `_outcomes_hat_columns`. The old `predicted_outcome_columns` param/property and the `"predicted"` role key are **removed outright**. | One vocabulary end-to-end; removes the overload with predicted-*weights* (`predict_weights`); distinct from `.outcomes()` (observed Y). No deprecation needed — the `predicted` role was reserved scaffolding **unused by any code or user** (only storage/validation/tests referenced it) and was never populated, so there is nothing to keep back-compatible and nothing to migrate. |
| 2026-07-15 | **Scope: outcome model now, AIPW later.** Ship IPW/Hájek (exists) + outcome-model / g-computation estimate (`μ̂_OM`) in 0.23.0; defer explicit AIPW/DR to a follow-up. Phase 1 delivers **no general DR** — only the linear-WLS special case (§1). | Smaller, reviewable diff stack that delivers the core ask (average `outcomes_hat` on the target). AIPW variance/CI + the normalization contract get their own design pass. |
| 2026-07-15 | **Estimate lives on the `outcomes_hat()` view:** `μ̂_OM = bf.outcomes_hat().mean()` (target row). Not under `.outcomes()`, no separate `outcome_estimate(method=)` dispatcher for phase 1. | Keeps the estimate where the predicted outcomes live; reuses the existing weighted-mean/CI machinery for free. |
| 2026-07-16 | **Fit is UNWEIGHTED by default** (`weighted=False`); pass `weighted=True` to fit with the active weight. Weighting the fit is a **modelling choice, not the DR mechanism**: the model is fit **independently of the IPW weights**, and double robustness comes from the **AIPW combination** (estimator #3) — combine the stored `outcomes_hat` with the IPW weights at *estimation* time. A weighted fit whose estimator's `fit` lacks `sample_weight` raises `TypeError`. | The outcome model estimates `E[Y|X]` and is usually best left unbiased by the design weights, so the default flipped from the earlier `weighted=True` plan to avoid silently weight-biasing `μ̂_OM`. The intended DR workflow is still fit → `adjust()`/IPW → combine `outcomes_hat` with the IPW weights (not through model fitting); survey-weighting the fit can reduce bias under misspecification. |
| 2026-07-15 | **Model + `outcomes_hat` deliberately PERSIST across weight-changing ops** (`adjust`/`trim`/re-adjust); invalidated only on an explicit re-fit. | `outcomes_hat = ĝ(X)` is a **deterministic function of covariates, invariant to weights** — so it is *not* stale after `adjust()`; the estimate correctly re-weights it with the new/target weights. Persisting the model is exactly what enables "fit model → adjust → combine for DR." |
| 2026-07-15 | **Binary calibration: lightweight default (uncalibrated) + easy opt-in.** Default stores raw `predict_proba`; `fit_outcome_model(calibrate=True)` wraps the classifier in `CalibratedClassifierCV`. A diagnostic compares weighted-mean(ŷ_R) vs weighted-mean(y_R) to flag miscalibration. | Calibration × bootstrap is expensive, so a light default is better; but calibration is one flag away, and the diagnostic surfaces prevalence bias without forcing the cost. |
| 2026-07-15 | **Estimate raises if the target `outcomes_hat` is missing.** `outcomes_hat().mean()`/estimate raises an actionable error when a model exists but the target isn't populated (call `predict_outcomes(on="target")` first) — never silently returns the responder in-sample mean. | Prevents a silent wrong answer (responder mean masquerading as the population estimate). |
| 2026-07-16 | **`summary()` reports the fit-weights and *scopes* any DR statement to them — never a blanket "doubly robust".** For a linear WLS fit with non-uniform weights it states e.g. "g-computation; doubly robust w.r.t. weights `<col>`"; uniform-weight or non-linear fits are reported as plain g-computation. | The DR guarantee is relative to the specific weight set used in the fit; reporting it (not a bare flag) is honest and avoids false-DR claims. DR *for the target estimand* still requires those weights to be the target-reweighting/IPW weights. |
| 2026-07-16 | **`outcomes_hat` columns follow a `<outcome>_hat` naming convention** (e.g. `happiness_hat`) — self-describing in `.df`, discernible from covariates/outcomes; `from_frame` **warns** if an unclaimed `_hat`-suffixed column would be inferred as a covariate. | Makes predicted columns recognizable end-to-end and prevents a reloaded target's `_hat` column from silently corrupting the next `adjust()`'s propensity model (the round-trip covariate leak). |
| 2026-07-16 | **`outcomes_hat().mean_with_ci()` defaults to `ci_method="bootstrap"`** on a BalanceFrame+target-backed view; a lone/target-less view **raises** (never silently returns the analytic CI, which under-covers `μ̂_OM` by treating `ŷ_T` as fixed). `mean()` stays the O(1) point estimate. | The analytic `ci_of_weighted_mean` ignores model-fitting uncertainty; bootstrap is the honest default. |
| 2026-07-16 | **AIPW normalization contract: deferred to phase 2 but DOCUMENTED now.** The augmentation is valid only when `w_R` and `w_T` are on the same population scale (both from `adjust()`); Hájek self-normalization gives *asymptotic* DR (ratio bias `O(1/n)`). Phase-2 `aipw()` should require `adjust()`-calibrated weights + assert `|Σw_R−Σw_T|/Σw_T < tol`. | Records the correctness precondition so the deferred AIPW isn't built on an unstated assumption. |
| 2026-07-15 | **Default estimator = sklearn `HistGradientBoosting{Regressor,Classifier}`** (dispatched by outcome type via `_is_discrete_series`); pluggable via **`model=`** — a single sklearn estimator (cloned per outcome; one type only for mixed outcomes), a `{"_discrete": clf, "_continuous": reg}` **type map** (a missing type falls back to `"auto"`), or a `{outcome_column: estimator}` **column map**. | One robust default for continuous **and** binary outcomes, mixed/categorical features, and non-linearity, with **no new dependency** (IPW already uses `HistGradientBoostingClassifier` for custom models). True XGBoost/LightGBM stay optional plug-ins (balance keeps deps light — copilot §7). **Caveat:** a non-linear default makes phase-1 `μ̂_OM` a pure g-computation estimate (not auto-DR); a *linear* estimator **fit by WLS with the adjusted IPW weights (after `adjust()`)** recovers the exact DR identity only as a special case, and the AIPW phase restores DR for any estimator. |
| 2026-07-16 | **`fit_outcome_model` API finalized:** the estimator argument is **`model=`** (renamed from `learner=`), carrying the `"auto"` / estimator / type-map / column-map dispatch above; a new **`variables=`** restricts the model inputs `X` to a validated subset of the covariate columns (default `None` = all covars); the per-outcome estimator choice is logged at **INFO**. | sklearn-like naming (`model=`) with a single argument covering every dispatch mode; `variables=` gives the outcome axis the same covariate-subset control the weighting axis has, without touching the frame's covariate role. |
| 2026-07-15 | **New `outcome_models/` package** (parallel to `weighting_methods/`). | Distinct axis from `BALANCE_WEIGHTING_METHODS`; room for multiple learners + AIPW helpers. |
| 2026-07-15 | **sklearn-style trio: `fit_outcome_model()` (fit + store, **no** populate), `predict_outcomes()` (predict + persist Ŷ), `fit_predict_outcomes()` (both).** `fit_*` never auto-writes Ŷ columns. | Mirrors sklearn `fit`/`predict`/`fit_predict`; `predict_outcomes` stays distinct from the weighting axis's `predict_weights`; explicit persistence avoids surprise column writes. |
| 2026-07-15 | **Phase-1 CI = bootstrap** the fit→predict→average loop (resample responders, refit `ĝ`, predict on the fixed target, re-average; percentile CI). | Honest inference that captures outcome-model estimation uncertainty; the only tractable option for the non-linear default (no closed form). Point estimate `outcomes_hat().mean()` stays O(1); bootstrap is opt-in for the CI. |
| 2026-07-15 | **Preprocessing is learner-dependent:** tree/boosting learners use `use_model_matrix=False` (native categoricals, no scaler); linear learners use `use_model_matrix=True` (patsy one-hot + `StandardScaler`). Param `use_model_matrix="auto"` resolves this. | Best per-learner fit; avoids one-hot blow-up for boosting; mirrors IPW's dual paths. **Caveat:** native categoricals (`categorical_features="from_dtype"`) need **sklearn ≥ 1.4**; on sklearn < 1.4 (pss2) the boosting default **falls back to one-hot** so the default works everywhere. Native-categorical *transfer* must persist fit-time category levels (see [§7](#7-design-matrix-reuse-fit-vs-transfer)). |
| 2026-07-15 | **Binary outcomes store the predicted probability** `P̂(Y=1\|X)` (`predict_proba[:,1]`); `μ̂_OM` = weighted mean of probabilities = estimated target prevalence. | Correct g-computation estimand for `E[Y]=P(Y=1)`; preserves calibration (hard labels would bias the prevalence). |
| 2026-07-15 | **Bootstrap = nonparametric, target fixed:** resample responders with replacement (keep weights), refit `ĝ*`, predict on the **fixed** target, re-average; percentile CI. Defaults `n_bootstrap=200`, `random_seed=2020`. The reusable loop lives in `outcome_models/` and stores only the **B scalar outputs, not B models**; `BalanceFrame` orchestrates it (sample→refit, target→predict). `BalanceDFOutcomesHat` **overrides** `mean_with_ci(ci_method="bootstrap")` to require a BalanceFrame-backed view and delegate to that bespoke routine — it **bypasses the linked-view machinery** (which can't reach the learner); a lone-`SampleFrame` view **raises**. | Captures the dominant uncertainty (responder sample + model fit); target is a known population so only responders resample — memory-light (keep B outputs). Weighted/Bayesian bootstrap is a later upgrade. |
| 2026-07-15 | **Standalone fit + stored fitted preprocessing (sklearn-Pipeline flow):** `fit_outcome_model` fits on the responders **alone** (no target needed) and stores the *fitted* preprocessing (design-matrix `X_matrix_columns` + `fit_scaler` + `categorical_levels`) alongside the learner; applying to the target **replays that stored transformer deterministically**. Default **`transformations=None`** (vs IPW's `"default"`). | Honors "the fit lives in the SampleFrame, applied to the target later" (standard sklearn fit/transform). Data-dependent balance transforms (`quantize`/`fct_lump`) aren't freezable for replay yet → rejected on transfer (a future "freeze transformer" can lift this); the boosting default doesn't need them, and linear learners get one-hot + scaler which *are* stored and replayed. |
| 2026-07-15 | **Multiple outcomes mirror the `outcomes()` view:** `outcomes_hat().mean()` and `outcomes_hat().mean_with_ci()` return the same source-indexed / one-column-per-outcome structure as `BalanceDFOutcomes`. | Consistency with the existing outcome API; free via inherited `BalanceDF` machinery; easy IPW-vs-outcome-model comparison. |
| 2026-07-15 | **`outcomes_hat().summary()` flags the estimator type** — states g-computation vs doubly-robust and warns when a non-linear learner is used (not DR). | Prevents misreading a boosting-based `μ̂_OM` as doubly robust. |
| 2026-07-15 | **sklearn < 1.4: graceful fallback to one-hot** for the boosting default (native categoricals on ≥ 1.4). | Feature works on every supported sklearn / pss version; no loss of pss2 support. |

---

## 1. Motivation — three estimators of the target-population outcome

`balance` today answers one question: *reweight a biased responder sample to a target
population, then estimate the population mean of an outcome*. That estimate is currently a
single estimator — the **Hájek (self-normalized, inverse-propensity-weighted) mean** of the
observed outcome (`weighted_stats.weighted_mean`, `stats_and_plots/weighted_stats.py:158-165` —
it divides by `Σw`, so balance produces Hájek, not raw Horvitz–Thompson, estimators throughout).

This feature adds the two missing estimators that use the *same covariates* differently, and
(in a later phase) combines them:

| # | Estimator | What it uses | Consistent when… | Phase |
|---|-----------|--------------|------------------|-------|
| 1 | **IPW / Hájek** `μ̂_IPW = Σ_R wᵢ yᵢ / Σ_R wᵢ` | responders' outcomes + adjusted weights | the **weights** (propensity model) are correct | exists today |
| 2 | **Outcome model / g-computation** `μ̂_OM = Σ_T w_Tⱼ ĝ(x_Tⱼ) / Σ_T w_Tⱼ` | a learner `ĝ(X)≈E[Y\|X]` fit on responders, averaged over the **target** covariates | the **outcome model** is correct | **0.23.0 (this doc)** |
| 3 | **Doubly-robust / AIPW** `μ̂_DR = μ̂_OM + Σ_R wᵢ(yᵢ − ĝ(xᵢ)) / Σ_R wᵢ` | both weights **and** the outcome model | **either** the weights **or** the outcome model is correct | later phase |

Where `R` = responders (the sample / `_sf_sample`), `T` = target (`_sf_target`),
`y` = observed outcome, `ĝ` = the fitted outcome learner, `ŷ = ĝ(x)` = the **`outcomes_hat`**
column this feature introduces. **Two weight sets are kept distinct:** `w_R^fit` = the weights used
to *fit* `ĝ` (the responders' active weight at fit time — often the **design** weights, since the
model is fit independently of IPW); `w_R` / `w_T` = the **adjusted (IPW) responder weights** and
**target weights** used at *estimation* time (in `μ̂_OM`, the AIPW augmentation, and the special
case). `w_R^fit` and `w_R` coincide only when the model is fit *after* `adjust()`.

The **outcome-model estimate** the user describes — *"the average of the `outcomes_hat` on the
target sampleframe"* — is estimator #2, surfaced as `bf.outcomes_hat().mean()` (the target row).
The outcome model `ĝ` is fit **independently of the IPW weights** (e.g. on the responders' design
weights); **double robustness is delivered by estimator #3's AIPW combination** — pairing the
model's `outcomes_hat` with the IPW weights *at estimation time* — not by fitting the model with
the IPW weights. So the model and its `outcomes_hat` deliberately persist across `adjust()`
(`outcomes_hat = ĝ(X)` is weight-invariant), enabling the workflow *fit model → adjust → combine*.

> **Two ways to obtain double robustness — don't conflate them:**
> **Path 1 — explicit AIPW (general; phase 2):** fit `ĝ` any way (e.g. on design weights), persist
> it, and combine `outcomes_hat` with the IPW weights via `aipw()`. Works for *any* learner — the
> primary DR mechanism.
> **Path 2 — implicit weighted-linear (special case):** fit a *linear* `ĝ` by WLS with the
> **adjusted IPW weights** (`w_R^fit == w_R`, i.e. fit after `adjust()`); then the augmentation is
> algebraically 0 and `μ̂_OM == μ̂_DR` with no explicit combination. Linear-only, *asymptotic*, and
> it **breaks if you re-`adjust()` afterwards** (the orthogonality held for the old weights).
> Phase 1 makes **no** blanket DR claim; `summary()` reports the fit-weights and scopes any DR
> statement to them.

> **Special case — a weighted-linear fit gives DR with no explicit augmentation.**
> Beyond the general AIPW combination (estimator #3), there is a shortcut: if `ĝ` is a **linear**
> model fit by **weighted least squares using the adjusted (IPW) weights `w_R`** — i.e. fit
> **after `adjust()`** — and it includes an intercept, then the AIPW augmentation term is
> algebraically **zero**:
> `Σ_R wᵢ (yᵢ − ĝ(xᵢ)) / Σ_R wᵢ = 0` (the weighted normal equations make the weighted residuals
> orthogonal to the design, including the constant). So `μ̂_OM == μ̂_DR` (this is Path 2 above).
> Two caveats: it is **asymptotic** DR — Hájek self-normalization leaves an `O(1/n)` ratio bias —
> and it holds **only for the exact weights used at fit time**, so a later `adjust()`/`trim()`
> breaks the identity. It is **not** the phase-1 default (the default learner is non-linear and is
> fit independently of the IPW weights); general DR arrives via the explicit AIPW phase (#3).

> **The default estimator is non-linear.** `model="auto"` resolves to sklearn's
> `HistGradientBoosting{Regressor,Classifier}` (see [§0](#0-decisions-log)), so **by default
> `μ̂_OM` is a pure g-computation estimate** — consistent only if the outcome model is right.
> Plug in a weighted linear learner to recover the exact DR identity above; the later AIPW phase
> restores double robustness for *any* learner.

This also brings doubly-robust estimation **in-house**. Today DR only exists via delegation to
the external `diff-diff` library for staggered DiD (`interop/diff_diff.py`, 0.21.0 CHANGELOG);
estimators #2–#3 make balance itself produce regression and doubly-robust population estimates.

---

## 2. Key insight — the `"predicted"` (Ŷ) role is already reserved

The data model was *designed* for this. `sample_frame.py`'s module docstring already lists the
five roles as **"covariates (X), weights (W), outcomes (Y), predicted_outcomes (Y_hat),
ignored"**, and `_column_roles` already carries a `"predicted"` key
(`sample_frame.py:151-157`):

```python
instance._column_roles = {
    "covars":    list(covar_columns),
    "weights":   list(weight_columns),
    "outcomes":  list(outcome_columns or []),
    "predicted": list(predicted_outcome_columns or []),   # ← Ŷ / outcomes_hat, reserved
    "ignored":   list(ignored_columns or []),
}
```

What **already works** (construction + name-read + role bookkeeping only):

- `SampleFrame.from_frame(..., predicted_outcome_columns=...)` accepts, normalizes (`str`→list),
  and validates the columns exist (`sample_frame.py:369-377`).
- Predicted columns are **excluded from inferred covariates** (`sample_frame.py:406`) and
  participate in role-overlap validation (`sample_frame.py:412-428`).
- `SampleFrame.predicted_outcome_columns -> list[str]` returns a copy (`sample_frame.py:527-546`).
- `_column_roles` (incl. `"predicted"`) is deep-copied (`sample_frame.py:126`) and pruned on
  column drops (`balance_frame.py:3436-3440`).
- Tests lock the storage contract (`tests/test_sample_frame.py:296-325, 522-527, 684-688`);
  the fixture convention names a predicted column for outcome `y` as **`p_y`**.

What is **absent** and must be built (verified by package-wide search — no matches):

- **No** `df_outcomes_hat` DataFrame accessor (contrast `df_outcomes`, `sample_frame.py:644-667`).
- **No** `_outcomes_hat_columns` protocol accessor (contrast `_outcome_columns`, `:790-811`).
- **No** `outcomes_hat()` view factory and **no** `BalanceDFOutcomesHat` view class.
- **No** fitted-model store, **no** learner-fitting method, **no** way to *produce* Ŷ.
- **Not preserved** on `Sample → SampleFrame` round-trips (`from_sample`, `:1304-1306`).

**Naming resolution ([§0](#0-decisions-log)):** `outcomes_hat` is the canonical spelling
**everywhere, internal and public** (bare `"hat"` is rejected as cryptic vs the full-word keys
`covars`/`weights`/`outcomes`/`ignored`). Renamed internals: the `_column_roles` key
`"predicted"` → `"outcomes_hat"`; the `_create()`/`from_frame()` parameter and internal locals
`predicted_outcome_columns`/`predicted_list` → `outcomes_hat_columns`/`outcomes_hat_list`; the
overlap-validation dict key; the protocol member `_outcomes_hat_columns`. Public surface:
`outcomes_hat()` (view), `df_outcomes_hat` (data), `outcomes_hat_columns` (names) — lexically
distinct from `.outcomes()` (observed Y) and the weighting axis's `predict_weights()`. The old
`predicted_outcome_columns` param/property and the `"predicted"` role key are **removed outright —
no alias, no `FutureWarning`, no migration** — because the role was reserved scaffolding unused by
any code or user (only storage/validation/tests referenced it, and it was never populated).

> **Naming collision to avoid.** "predict" is heavily overloaded in the *weighting* code:
> `predict_weights()` / `_predict_weights_from_model` mean **predicted weights** transferred to
> new data. The `outcomes_hat` surface stays lexically distinct so the two axes don't blur.

---

## 3. Design overview — mirror the fit-artifact machinery, on a new axis

The outcome model is **structurally the IPW fit-artifact workflow, redirected**:

```
                       IPW (weighting axis, exists)      OUTCOME MODEL (new axis)
                       ─────────────────────────────    ─────────────────────────────
   learner             LogisticRegression on            regressor/classifier on
                       y = 1{sample} vs 0{target}       y = the observed OUTCOME column
   design matrix       build_design_matrix(X_R, X_T)    build_design_matrix(X_R, X_T)   ← SAME util
   fit weights         balance_classes offset           responders' balancing weights (survey-wtd)
   prediction          predict_proba[:,1] → link →      learner.predict(X)  → ŷ
                       weights_from_link → weights      (no link, no trimming, no balance_classes)
   applied to          responders (to get weights)      the TARGET (to get the estimate)
   stored artifact     BalanceFrame._adjustment_model   SampleFrame._outcome_model
                       {"method":"ipw","fit":est,...}   {"method":"outcome_model",
                                                          "fit":{y→est},"X_matrix_columns",
                                                          "fit_scaler","formula",...}
   transfer to new     set_fitted_model / predict_      set_fitted_outcome_model /
   data                weights(data=)                   predict_outcomes(on=|data=)
   replay engine       _compute_ipw_matrices            _compute_design_matrices (generalized)
   estimate            outcomes().mean()  (μ̂_IPW)       outcomes_hat().mean()  (μ̂_OM)
```

**Default estimator.** `model="auto"` resolves to sklearn `HistGradientBoostingRegressor`
(continuous outcome) or `HistGradientBoostingClassifier` (binary), dispatched by
`input_validation._is_discrete_series`. One robust default for both outcome types with no new
dependency (IPW already uses `HistGradientBoostingClassifier` for custom models); XGBoost/LightGBM
remain optional plug-ins via `model=`. Preprocessing is **learner-dependent** (see
[§0](#0-decisions-log)): boosting/tree learners use the `use_model_matrix=False` native-categorical
path (no scaler); linear learners use patsy one-hot + `StandardScaler`. Native categoricals require
**sklearn ≥ 1.4**, so on sklearn < 1.4 the boosting default **falls back to one-hot**.

The reusable engine is `utils/model_matrix.py::build_design_matrix` (`:778`), which already
supports **train mode** (fit a `StandardScaler`, return the fit-time `columns`) and **replay
mode** (`project_to_columns=<stored columns>` + `fit_scaler=<stored scaler>` → transform only,
zero-filling unseen columns). IPW's `_compute_ipw_matrices` (`balance_frame.py:1490-1551`) is a
thin wrapper we generalize (see [§7](#7-design-matrix-reuse-fit-vs-transfer)).

**Ownership choice (matches the user's mental model).** The IPW model lives on `BalanceFrame`
because it *needs* both sample and target (the sample-vs-target indicator). The **outcome model
only needs the responders' own X and Y**, so it can be fit on a lone `SampleFrame` — therefore
it is **stored on the `SampleFrame`** ("stored as part of the sampleframe"). `BalanceFrame`
orchestrates *transfer to the target* and *estimation*, exactly as it orchestrates `adjust()`
and `set_fitted_model()` for weights.

---

## 4. Where each responsibility lives

```
┌────────────────────────────────────────┬───────────────────────────────────────────────┐
│              Responsibility             │                    Class / member             │
├────────────────────────────────────────┼───────────────────────────────────────────────┤
│ outcomes_hat (Ŷ) column storage         │ SampleFrame._column_roles["outcomes_hat"]      │
│ Fitted outcome-model store              │ SampleFrame._outcome_model : dict | None  (NEW) │
│ Per-Ŷ-column provenance metadata        │ SampleFrame._prediction_metadata : dict  (NEW)  │
│ Ŷ name/data accessors                   │ outcomes_hat_columns / df_outcomes_hat /        │
│                                         │   _outcomes_hat_columns  (all NEW; replace the  │
│                                         │   removed predicted_outcome_columns)            │
│ Add a Ŷ column post-construction        │ SampleFrame.add_outcomes_hat_column(...)  (NEW) │
│ Fit ĝ(X)≈E[Y|X] on responders (weighted)│ SampleFrame.fit_outcome_model(...)        (NEW) │
│   (store learner + preprocessing)       │   → writes _outcome_model                        │
│ Populate self's Ŷ from stored model     │ SampleFrame.predict_outcomes(...)         (NEW) │
│ Deep-copy of new state                  │ SampleFrame.__deepcopy__  (extend, :102-132)    │
│ Transfer ĝ to the TARGET, produce Ŷ_T   │ BalanceFrame.predict_outcomes(on="target") (NEW)│
│ Apply a foreign fitted model (holdout)  │ BalanceFrame.set_fitted_outcome_model(...) (NEW)│
│ Ŷ view + linked (self/target/unadj)     │ BalanceDFOutcomesHat(BalanceDF)           (NEW) │
│ Ŷ view factory                          │ BalanceFrame.outcomes_hat() / SampleFrame.…()   │
│ Outcome-model estimate  μ̂_OM            │ outcomes_hat().mean()  (target row) (inherited) │
│ IPW/Hájek estimate  μ̂_IPW (exists)      │ outcomes().mean()                               │
│ Doubly-robust estimate  μ̂_DR (later)    │ outcomes_hat().aipw()  (later phase)            │
│ Design-matrix build + replay            │ utils/model_matrix.build_design_matrix (reuse)  │
│ Learner/scaler storage in model dict    │ mirror ipw.py model dict (reuse conventions)    │
└────────────────────────────────────────┴───────────────────────────────────────────────┘
```

---

## 5. Diagram — column roles with `outcomes_hat` (Ŷ)

```
SampleFrame._df  (one DataFrame; every column has exactly one role)

  ┌───────────┬──────────────────────────────────────────────────────────────┐
  │  Column   │  Role                                                          │
  ├───────────┼──────────────────────────────────────────────────────────────┤
  │  id       │  id_column          (str name; data via id_series)            │
  │  weight   │  weights            (active + history cols; weight_series)     │
  │  age,os…  │  covars             (X)  ← the residual after all other roles  │
  │  happiness│  outcomes           (Y)  observed; df_outcomes / outcomes()    │
  │  happy_hat│  outcomes_hat       (Ŷ)  NEW live role: df_outcomes_hat /      │
  │           │                          outcomes_hat() ; produced by ĝ(X)     │
  │  notes    │  ignored                                                       │
  └───────────┴──────────────────────────────────────────────────────────────┘

  covars = all columns − id − weight − outcomes − ignored − outcomes_hat   (unchanged rule)

  Accessor families (follow the existing table in ARCHITECTURE.md §"Accessor naming"):
    *_columns → list[str]      outcomes_hat_columns  (NEW; predicted_outcome_columns removed)
    df_*      → DataFrame|None  df_outcomes_hat            (NEW, mirrors df_outcomes→None)
    _* (proto)→ DataFrame|None  _outcomes_hat_columns      (NEW, mirrors _outcome_columns)
    <role>()  → BalanceDF|None  outcomes_hat()             (NEW, mirrors outcomes())
```

Asymmetry that falls out naturally: a **target usually has no observed `outcomes`** but **will
carry `outcomes_hat`** (that's the whole point — Ŷ_T is what we average). `_call_on_linked`
already skips `None` children (`balancedf_class.py:469-480`), so `outcomes().mean()` and
`outcomes_hat().mean()` each expand across only the sources that actually have that role.

---

## 6. Diagram — the outcome-model lifecycle

```
 STEP A — fit on responders (within a lone SampleFrame or a BalanceFrame's _sf_sample)
 ─────────────────────────────────────────────────────────────────────────────────────
   # fit only — does NOT persist Ŷ. Model is fit INDEPENDENTLY of IPW (design weights are fine);
   # DR comes from the AIPW combination later (STEP D), not from fitting with the IPW weights.
   sf.fit_outcome_model(model="auto", outcome_columns=None, variables=None, formula=None,
                        transformations=None, na_action="add_indicator", weighted=False)
     │
     ├─ choose_variables(sf.df_covars) ................. covariate set (X)
     ├─ Y = sf.df_outcomes[outcome_columns] ............ observed outcome(s)
     ├─ build_design_matrix(X_R, X_R,  scaler_weights=w_R,     ← TRAIN mode
     │      use_model_matrix=<False for boosting / True for linear>,   # learner-dependent
     │      formula=, one_hot_encoding=, na_action="add_indicator")
     │      → combined_matrix, columns, fit_scaler, resolved_formula (+ categorical_levels)
     ├─ for each outcome col y:
     │      learner_y = _resolve_learner(learner, Y[y])   # "auto" -> HistGradientBoosting{Reg,Clf}
     │      learner_y.fit(matrix_R, Y[y],
     │                    sample_weight=w_R if weighted else None)   # survey-weighted regression
     ├─ sf._outcome_model = {
     │      "method": "outcome_model",
     │      "fit": {y: learner_y, ...},               # sklearn estimators
     │      "X_matrix_columns": columns,
     │      "fit_scaler": fit_scaler, "formula": resolved_formula,
     │      "na_action": "add_indicator", "one_hot_encoding": ...,
     │      "transformations": ..., "use_model_matrix": ..., "weighted": False,
     │      "outcome_columns": [...], "perf": {y: {"r2"/"deviance"...}},
     │      "training_sample_index": X_R.index.copy(),
     │    }
     └─ (fit does NOT persist Ŷ — the stored preprocessing + learner IS the sklearn "pipeline".
         Use predict_outcomes()/fit_predict_outcomes() to write the outcomes_hat columns.)

 STEP B — combine with a target → BalanceFrame, transfer the fit to the target
 ─────────────────────────────────────────────────────────────────────────────────────
   bf = sf.set_target(target)              # or BalanceFrame(sample=sf, target=target)
   bf.predict_outcomes(on="target")        # dispatch on _sf_sample._outcome_model["method"]
     │
     ├─ matrix_T = _compute_design_matrices(model, source=bf, side="target")   ← REPLAY mode
     │      build_design_matrix(X_R, X_T, project_to_columns=model["X_matrix_columns"],
     │          fit_scaler=model["fit_scaler"], na_action="add_indicator", ...)[target rows]
     ├─ ŷ_T = { y: _predict(model["fit"][y], matrix_T) for y in outcome_columns }
     │        # regressor → .predict ;  classifier → .predict_proba[:,1]  (P̂(Y=1))
     └─ writes target._df outcomes_hat columns (add_outcomes_hat_column on _sf_target)

 STEP C — the estimate = weighted mean of Ŷ on the target
 ─────────────────────────────────────────────────────────────────────────────────────
   # point estimate — RAISES if a model exists but target ŷ_T is unpopulated (predict first):
   μ̂_OM = bf.outcomes_hat().mean()         # linked: target (and self/unadj) rows; O(1)
        = weighted_mean(ŷ_T, w_T)          # Σ w_T ŷ_T / Σ w_T   (weighted_stats.py:158)
   # bootstrap CI — engine in outcome_models/, orchestrated by BalanceFrame, surfaced on the view:
   bf.outcomes_hat().mean_with_ci(ci_method="bootstrap", n_bootstrap=200, random_seed=2020)
     → for b in 1..B: resample responders, refit ĝ*, predict on the FIXED target,
                      μ̂_OM*(b) = weighted_mean(ĝ*(X_T), w_T);   percentile CI over {μ̂_OM*(b)}
     (keeps only the B scalar μ̂_OM*(b), never B models; recomputes internally, so it does not
      depend on a prior predict_outcomes())

 STEP D — (LATER PHASE) doubly-robust / AIPW
 ─────────────────────────────────────────────────────────────────────────────────────
   μ̂_DR = μ̂_OM  +  weighted_mean(y_R − ŷ_R, w_R)      # target g-comp + IPW-weighted residual mean
        = bf.outcomes_hat().aipw()
   # THE general DR mechanism: ĝ fit INDEPENDENTLY (any weights); w_R = the IPW/adjusted weights.
   # needs outcomes() + outcomes_hat() on responders (residual y−ŷ) and outcomes_hat() on target.
   # NB special case: a weighted-linear ĝ fit with w_R makes the residual term 0 ⇒ μ̂_DR == μ̂_OM (§1).
```

---

## 7. Design-matrix reuse: fit vs transfer

The whole transfer story rests on `build_design_matrix` producing **identical features** on
train and score data. This already exists for IPW — we reuse it verbatim.

```
TRAIN (fit on responders)                     REPLAY (score on target/holdout)
──────────────────────────                    ────────────────────────────────
build_design_matrix(                          build_design_matrix(
   X_R, X_R,                                      X_R, X_T,
   scaler_weights = w_R,   ← fits scaler          project_to_columns = model["X_matrix_columns"],
   formula=, one_hot_encoding=,                    fit_scaler       = model["fit_scaler"],  ← transform only
   na_action="add_indicator")                      formula          = model["formula"],
      │                                             na_action        = "add_indicator",
      ├→ columns   (persist)                        matrix_type      = model["fit_matrix_type"])
      ├→ fit_scaler(persist)                           │
      └→ resolved_formula (persist)                    └→ reindexed to fit-time columns
                                                          (unseen → zero, novel → dropped)
```

**Hard constraints inherited from the replay path** (must be documented in the API):

- `na_action="drop"` is **rejected** on the projection/holdout path (`model_matrix.py:634-640`)
  because dropping rows breaks the sample/target boundary → outcome-model transfer requires
  **`na_action="add_indicator"`** (default). `fit_outcome_model()` should therefore reject
  `na_action="drop"` when the model will be transferred, mirroring how `fit(method="ipw")`
  rejects `na_action="drop"` with stored artifacts (`balance_frame.py:1164-1170`).
- Data-dependent default transformations (`quantize`, `fct_lump` in
  `utils/data_transformation.py`) are **not replay-safe**; the same guards that protect
  rake/poststratify transfer scoring (`functools.partial` rejection, 0.22 CHANGELOG) apply.
  In-place populate on the *same* frame is always safe; transfer to a new frame requires
  deterministic transformations.
- Variable names containing `[` / `]` are rejected by patsy (`model_matrix.py:313-318`).
- **Native-categorical learners** (`use_model_matrix=False`, e.g. the boosting default on
  sklearn ≥ 1.4) skip patsy/scaler; their transfer requires **persisting the fit-time category
  levels** and re-applying them to the target columns (`pd.Categorical` with the same categories)
  so the learner sees identical integer codes. **This is a correctness fix, not just metadata:**
  today `build_design_matrix`'s raw path (`_build_raw_covariates`) re-derives categories from
  `concat(sample, target)` on every call, so a novel or missing target category silently *shifts
  the integer codes*, and `reindex(columns=..., fill_value=0)` (column alignment) will NOT catch
  it → wrong predictions. The replay builder must accept stored `_outcome_model["categorical_levels"]`.
- **`na_action="drop"` is NOT guarded on the raw path.** The existing drop-rejection lives only on
  the `use_model_matrix=True` projection branch (`model_matrix.py:634-640`); the raw/native path
  just concatenates. So `fit_outcome_model` must reject `na_action="drop"` itself (mirroring
  `BalanceFrame.fit`'s IPW guard, `balance_frame.py:1164-1170`) for the boosting default.
- **Matrix type per learner.** `HistGradientBoosting*` rejects sparse input (IPW densifies via
  `_convert_to_dense_array` before its custom-model fit); linear learners accept sparse. The fit
  routine must set `matrix_type` per learner and **store `fit_matrix_type`**, re-passing it on
  replay exactly as IPW does — else fit- and score-time matrix types diverge.

---

## 8. Diagram — `BalanceDFOutcomesHat` view + protocol extension

`BalanceDF` gives us the weighted-mean / CI / plot / summary machinery for free — a new view
just needs three wiring points. `name` is **load-bearing**: `_balancedf_child_from_linked_samples`
calls `getattr(linked_source, self.__name)()` (`balancedf_class.py:214-380`), so the view's
`name` must equal the accessor method name — here `"outcomes_hat"`.

```
                          ┌──────────────────────────────┐
                          │   BalanceDFSource (Protocol)  │  balancedf_class.py:40-98
                          │   + _outcomes_hat_columns (NEW)│  → pd.DataFrame | None
                          └──────────────┬───────────────┘
      implemented by ─────────────────────┼─────────────────────
        SampleFrame._outcomes_hat_columns │   BalanceFrame._outcomes_hat_columns
        = df_outcomes_hat  (NEW)          │   = delegate to _sf_sample  (NEW, cf. :297-307)
                                          ▼
                          ┌──────────────────────────────┐
                          │ class BalanceDFOutcomesHat(   │
                          │       BalanceDF):             │
                          │     __init__(sample, links):  │
                          │       yhat = sample._outcomes_hat_columns
                          │       if yhat is None: raise  │
                          │       super().__init__(yhat,  │
                          │         sample, name="outcomes_hat",
                          │         links=links)          │
                          └──────────────┬───────────────┘
                                         │  inherits: mean() std() var_of_mean()
                                         │    ci_of_mean() plot()
                                         │  OVERRIDES: mean_with_ci() (ci_method=),
                                         │    summary() (scoped DR statement)
                                         ▼
        μ̂_OM = target.outcomes_hat().mean(on_linked_samples=False)
             = weighted_mean(ŷ_T, w_T)             weighted_stats.py:164-165
        CI  = mean_with_ci(ci_method='bootstrap')  override → BalanceFrame bootstrap; NOT the
              analytic ci_of_weighted_mean (it treats ŷ_T as fixed and under-covers μ̂_OM)
```

Factory methods (mirror `outcomes()` at `balance_frame.py:2916-2945` and
`sample_frame.py:1064-1088`):

```python
def outcomes_hat(self) -> "BalanceDFOutcomesHat | None":
    if not self._<...>_column_roles["outcomes_hat"]:
        return None
    return BalanceDFOutcomesHat(cast(BalanceDFSource, self), links=self._build_links_dict())
```

`BalanceDFSource` protocol note: `Sample` and `SampleFrame` must both satisfy the protocol, so
the new `_outcomes_hat_columns` member is added to both; `BalanceFrame` continues to pass
`cast(BalanceDFSource, self)` (it already fails structural conformance on `weight_column`).
The bootstrap-CI guard keys off the presence of a target (`_sf_target is not None`), **not**
`isinstance(_sample, BalanceFrame)` — a `Sample` *is* a `BalanceFrame` via MRO, so a target-less
`Sample.outcomes_hat().mean_with_ci(ci_method="bootstrap")` must still raise.

---

## 9. Diagram — estimator taxonomy & data flow

```
                 responders R (sample)                 target T
             X_R, Y_R, adjusted weights w_R        X_T, weights w_T
                        │                                │
      ┌─────────────────┼────────────────┐               │
      ▼                 ▼                 ▼               ▼
  (1) IPW/Hájek    (2) OUTCOME MODEL              (2) apply ĝ to target
  weighted mean    fit ĝ(X)≈E[Y|X] on R          ŷ_T = ĝ(X_T)
  of Y over R      (survey-weighted)                     │
      │            store on _outcome_model               ▼
      │                 │  ŷ_R = ĝ(X_R)          μ̂_OM = Σ w_T ŷ_T / Σ w_T
      │                 │                                │   = outcomes_hat().mean()[target]
      ▼                 └──────────┐                     │
  μ̂_IPW =                          ▼                     │
  outcomes().mean()      residual r_R = Y_R − ŷ_R        │
      │                            │                     │
      │              (LATER)  Σ w_R r_R / Σ w_R ────────►│  (3) AIPW  outcomes_hat().aipw()
      │                                                   ▼
      └──────────────────────────►  μ̂_DR = μ̂_OM + Σ w_R (Y_R − ŷ_R) / Σ w_R
                                     (linear learner fit by WLS with w_R ⇒ residual = 0 ⇒ μ̂_DR = μ̂_OM)
```

---

## 10. Proposed API surface (provisional signatures)

All new public entry points are **keyword-only** past the first argument, return `Self` where
chainable, and honor `inplace` semantics (`True` mutates+returns self; `False` returns a copy) —
matching `fit()` / `set_fitted_model()` (`balance_frame.py:1036, 1220`;
`tests/test_balance_frame.py:2709-2748`).

```python
# --- SampleFrame (owns the model; can fit standalone) ---------------------------------------
def fit_outcome_model(                                # fit + store the pipeline; does NOT persist Ŷ
    self, *,
    model: str | Any | dict[str, Any] = "auto",     # "auto"=HistGradientBoosting{Reg,Clf} by outcome type; a single sklearn est; a {"_discrete": clf, "_continuous": reg} type map; or a {col: est} column map
    outcome_columns: list[str] | str | None = None, # default: all outcome_columns
    variables: list[str] | str | None = None,       # default None = all covars; else a validated covariate subset for X
    formula: str | list[str] | None = None,
    transformations: str | dict | None = None,      # default None (deterministic → replay-safe); vs IPW "default"
    na_action: str = "add_indicator",
    use_model_matrix: bool | str = "auto",           # "auto": raw-categorical for trees, one-hot for linear
    weighted: bool = False,                           # default unweighted; True = WLS with the active weight (a modelling choice; DR is via the AIPW combination). TypeError if the estimator's fit lacks sample_weight
    calibrate: bool = False,                          # opt-in: wrap classifiers in CalibratedClassifierCV (binary only)
    inplace: bool = True,
) -> Self: ...

def predict_outcomes(self, *, data: SampleFrame | None = None,
                     populate: bool = True) -> pd.DataFrame: ...   # Ŷ from stored model; persists
                     #   columns named "<outcome>_hat" (self-describing in .df; guards the round-trip leak)

def fit_predict_outcomes(self, *, populate: bool = True, **fit_kwargs) -> pd.DataFrame: ...  # fit then predict

def add_outcomes_hat_column(self, name: str, values: pd.Series,
                            metadata: dict[str, Any] | None = None) -> None: ...  # cf add_weight_column

@property
def df_outcomes_hat(self) -> pd.DataFrame | None: ...
@property
def outcomes_hat_columns(self) -> list[str]: ...                 # NEW primary names accessor
# (the old predicted_outcome_columns property + from_frame()/Sample.from_frame()/_create() kwarg
#  are REMOVED outright — no alias, no FutureWarning; the "predicted" role was unused scaffolding.)
@property
def _outcomes_hat_columns(self) -> pd.DataFrame | None: ...      # BalanceDFSource protocol
@property
def outcome_model(self) -> dict[str, Any] | None: ...           # cf. BalanceFrame.model
def outcomes_hat(self) -> "BalanceDFOutcomesHat | None": ...

# --- BalanceFrame (orchestrates target transfer + estimation) -------------------------------
def fit_outcome_model(self, *, target=None, model="auto", variables=None,
                      weighted=False, inplace=True, **kw) -> Self: ...
def set_fitted_outcome_model(self, fitted: "BalanceFrame", *, inplace: bool = True) -> Self: ...
def predict_outcomes(self, *, on: Literal["sample","target","both"] = "target",
                     data: "BalanceFrame | None" = None, populate: bool = True
                     ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: ...

# --- BalanceDFOutcomesHat (view) ------------------------------------------------------------
# inherits mean/std/var_of_mean/ci_of_mean/plot from BalanceDF unchanged; OVERRIDES
#   mean_with_ci() (adds ci_method=) and summary() (flags g-computation vs doubly-robust).
#   μ̂_OM = outcomes_hat().mean()          (target row; O(1))
#         → RAISES if a model exists but the target outcomes_hat is unpopulated
#           ("... call predict_outcomes(on='target') first") — never returns the responder mean.
#   CI:   outcomes_hat().mean_with_ci(ci_method="bootstrap", n_bootstrap=200, random_seed=2020)
#         → the override delegates to the backing BalanceFrame's bootstrap routine (resample
#           responders → refit → predict on the fixed target), keeping only B outputs. It
#           BYPASSES _call_on_linked (which can't reach the learner). A lone-SampleFrame-backed
#           view RAISES ("bootstrap CI requires a BalanceFrame with a target").
#   (later) outcomes_hat().aipw()          doubly-robust estimate (combine outcomes_hat + IPW weights)
```

**Actionable-error convention** (stable substrings, matched by tests — cf.
`tests/test_balance_frame.py`): `"no outcome model has been fit"`, `"na_action='drop' ...
incompatible with outcome-model transfer"`, `"matching sample covariate column names"`,
`"no outcome columns"`. Transfer shares the fitted estimator object by identity (like
`set_fitted_model`, `test_fit_on_subset_and_apply_to_holdout:2929-2947`), not a deep copy.

---

## 11. The stored `_outcome_model` dict (mirror the IPW model dict)

Follows `ipw.py:1122-1172` so the same `build_design_matrix` replay works. Distinct key names
avoid clobbering the propensity `model` dict.

```python
_outcome_model = {
    "method": "outcome_model",
    "fit": {outcome_col: <fitted sklearn estimator>, ...},   # regressor or classifier per Y
    "X_matrix_columns": list[str],           # fit-time design-matrix columns (projection target)
    "fit_scaler": StandardScaler | None,     # reused transform-only at score time
    "formula": str | list[str],              # resolved patsy formula
    "na_action": "add_indicator",
    "one_hot_encoding": bool,
    "use_model_matrix": bool,
    "transformations": None | dict | "default",  # default None (§0); "default" only if opted in + replay-safe
    "fit_matrix_type": "sparse" | "dense" | "dataframe",
    "fit_weight": {"column": str, "uniform": bool},  # for scoped-DR summary + bootstrap refit-weighting
    "weighted": bool,                        # whether fit used responder weights
    "calibrated": bool,                      # whether classifiers were wrapped in CalibratedClassifierCV
    "prediction_kind": {outcome_col: "regression" | "proba"},  # classifiers store P̂(Y=1)
    "categorical_levels": dict | None,       # fit-time category levels for native-categorical transfer
    "outcome_columns": list[str],
    "learner": str | repr,                   # for provenance / summary
    "perf": {outcome_col: {"r2"/"deviance_explained"/..., "n": int}},  # cf ipw "perf"
    "training_sample_index": pd.Index,       # staleness / alignment
}
```

Storage caveats (from the deep-copy audit): the new `_outcome_model` and `_prediction_metadata`
attributes **must** be initialized in `SampleFrame._create()` (`:151-163`) **and** copied in
`SampleFrame.__deepcopy__()` (`:102-132`) — otherwise deep copies silently lose them. sklearn
estimators are deep-copyable, so `deepcopy(_outcome_model)` works once wired. Populating an Ŷ
column must **mutate `_sf_sample._df` in place** (via `add_outcomes_hat_column`) so it does
**not** swap `_sf_sample` and therefore does **not** flip `is_adjusted` (which is defined
structurally as `_sf_sample is not _sf_sample_pre_adjust`, `balance_frame.py:697`).

**Single source of truth:** `BalanceFrame.outcome_model` / `_outcomes_hat_columns` **delegate** to
`_sf_sample` (mirroring `_outcome_columns`, `balance_frame.py:297-307`) rather than being copied by
`_sync_sampleframe_state_from_responder` — so a `Sample` (which is both classes) can't hold a stale
inherited copy. `perf` needs a **new weighted-R² / weighted-deviance helper** in
`stats_and_plots/weighted_stats.py` (none exists; `ipw.model_coefs` covers only linear
coefficients). The DR flag in `summary()` must verify a linear learner actually has
`fit_intercept=True` before claiming double robustness (the design matrix drops the patsy intercept
via `-1`, so the constant comes from the learner).

**Deepcopy & lifecycle (from the adversarial pass).** `SampleFrame.__deepcopy__` must copy
`_outcome_model`/`_prediction_metadata` (with a `getattr(..., None)` default for old pickles) but
**reference-share the fitted estimator** (immutable post-fit) instead of deep-cloning it on every
`adjust()`/`trim()` — mirroring `_copy_adjustment_history_from(deep=False)`, which deliberately
avoids duplicating large fitted artifacts. Store the fit-weight identity (`fit_weight`) so
`summary()` can name the DR weights and the bootstrap can **refit with the same fit-weighting** as
the stored model, not the currently-active weights. A **re-fit drops the `outcomes_hat` columns it
supersedes** (and fingerprints the rest via `training_sample_index`), so `outcomes_hat().mean()`
never averages stale Ŷ against a new model. `set_target()` **preserves** `_outcome_model` across its
reset (copying it onto the restored baseline) so a model fit after `adjust()` isn't silently lost.
`keep_only_some_rows_columns` that drops rows **invalidates** the model (stale `training_sample_index`).

---

## 12. Implementation plan (file-by-file)

| File | Change |
|------|--------|
| `sample_frame.py` | **Rename the internal `predicted*` structure → `outcomes_hat*`:** `_column_roles` key `"predicted"` → `"outcomes_hat"` (`_create` `:151-157`, `from_frame` overlap dict `:412-418`, covar-exclusion `:406`, module docstring); rename the `_create()`/`from_frame()` param `predicted_outcome_columns` → `outcomes_hat_columns` and internal locals (`predicted_list` → `outcomes_hat_list`, `missing_predicted` → …); make `outcomes_hat_columns` the primary names property and **remove** the old `predicted_outcome_columns` property + `from_frame`/`_create` kwarg outright (`:527-546`) — no alias/`FutureWarning`, no `"predicted"`-key migration (role never populated; an optional defensive `.get("outcomes_hat", [])` is enough). Add `_outcome_model`/`_prediction_metadata` attrs (init in `_create`, copy in `__deepcopy__`); `df_outcomes_hat`, `_outcomes_hat_columns`, `outcome_model` accessors; `add_outcomes_hat_column()`; `outcomes_hat()` factory; `fit_outcome_model()` (standalone fit + store pipeline; **no** populate), `predict_outcomes()` (persist Ŷ, in-place or on `data=`), and `fit_predict_outcomes()`. |
| `balance_frame.py` | update role-prune consumer (`:3436-3440`) to the renamed key; delegate `_outcomes_hat_columns`/`outcomes_hat()`; `fit_outcome_model()` (delegates to responder); `predict_outcomes(on=...)` (target transfer via generalized `_compute_design_matrices`); `fit_predict_outcomes()`; `set_fitted_outcome_model()`; orchestrate the `outcome_models` bootstrap loop (B refits on the sample, predictions on the fixed target) surfaced via `outcomes_hat().mean_with_ci`. Generalize `_compute_ipw_matrices` → shared `_compute_design_matrices(model, source, side)`. |
| `balancedf_class.py` | `class BalanceDFOutcomesHat(BalanceDF)` (`name="outcomes_hat"`); override `summary()` (estimator type / not-DR warning) and `mean_with_ci()` (add `ci_method="bootstrap"`, delegating the refit loop to the backing BalanceFrame); `mean()` inherited (source-indexed, one column per outcome, mirroring `BalanceDFOutcomes`); add `_outcomes_hat_columns` to the `BalanceDFSource` protocol. |
| new `outcome_models/` package | `outcome_model.py`: `_resolve_learner()` (default `HistGradientBoosting{Regressor,Classifier}` by outcome type via `input_validation._is_discrete_series`; accepts any sklearn estimator / `{col: estimator}`); the weighted **standalone** fit routine + `predict`/`fit_predict` helpers that store & replay the fitted preprocessing; learner-dependent preprocessing (`use_model_matrix="auto"`: native categoricals for trees on sklearn ≥ 1.4 with **one-hot fallback on < 1.4**, one-hot + scaler for linear); classifiers store `predict_proba[:,1]`; `perf` (`r2`/`deviance_explained`, with `ipw.model_coefs` only for linear learners); the bootstrap-CI loop (`n_bootstrap=200`, `random_seed=2020`, percentile, target fixed) that keeps only the **B scalar outputs, not B models** (orchestrated from `BalanceFrame`, which supplies sample-for-refits + fixed-target-for-predictions). Parallel to `weighting_methods/`, a *separate axis* (not a `BALANCE_WEIGHTING_METHODS` entry). |
| `utils/` | reuse `build_design_matrix`, `choose_variables`, `_check_weighting_methods_input`, `_extract_series_and_weights`, `_apply_na_action_to_frame_pair`, `_assert_type` (no new utils expected). |
| `sample_class.py` | expose `fit_outcome_model`/`predict_outcomes`/`outcomes_hat`/`outcome_model` on `Sample` (inherited via MRO; verify no facade gaps); rename the `Sample.from_frame` kwarg `predicted_outcome_columns` → `outcomes_hat_columns` (remove the old name outright). |
| `__init__.py` | export `BalanceDFOutcomesHat`. |
| `tests/` | new `test_outcome_model.py` (fit/populate/transfer/estimate; weighted-linear == DR sanity check); extend `test_sample_frame.py` (df_outcomes_hat, add_outcomes_hat_column, deepcopy), `test_balancedf.py` (BalanceDFOutcomesHat mean/CI/summary), `test_balance_frame.py` (transfer, error paths). Any learner needing sklearn≥1.4 uses **both** guards (`@pytest.mark.requires_sklearn_1_4` + `@unittest.skipUnless(_SKLEARN_1_4_AVAILABLE, ...)`, import from `balance.testutil`). Test the **sklearn < 1.4 one-hot fallback** for the boosting default, the binary `predict_proba` path, and bootstrap-CI reproducibility (`random_seed`). |
| `CHANGELOG.md` | add to a **`# 0.23.0 (Unreleased - TBD)`** section — **create it at the top of `CHANGELOG.md`, above the now-released `# 0.22.0`** — under `## New Features` + `## Tests` (+ a `## Documentation` bullet linking this doc). The `predicted_outcome_columns` → `outcomes_hat_columns` rename needs **no `## Deprecations` entry** (the old name was inert, unused scaffolding); mention it in passing in the `outcomes_hat` feature bullet. |
| `ARCHITECTURE.md` | add outcome-model rows to "Where each responsibility lives"; add an "Outcome-model workflow" section next to "Fit-artifact workflow". |

**Review-driven additions (round 5):**
- `stats_and_plots/weighted_stats.py` — a **weighted-R² / weighted-deviance** helper for `perf` (none exists; balance's first regression metric).
- `outcome_models/` — densify for `HistGradientBoosting` + set/store `fit_matrix_type` per learner; reject `na_action="drop"`; `calibrate=True` → `CalibratedClassifierCV` + the calibration diagnostic; `categorical_levels` capture/replay (correctness); guard the DR flag on `fit_intercept=True`.
- `balance_frame.py` — `outcome_model`/`_outcomes_hat_columns` **delegate** to `_sf_sample` (single source of truth); `outcomes_hat().mean()` **raises** when the target isn't populated; the bespoke bootstrap routine + lone-view guard.
- `utils/model_matrix.py` — raw path accepts stored `categorical_levels` (or do the restoration in `outcome_models/`).
- `typing.py` — `Union`-style `Literal` aliases for `ci_method` / learner-string (py3.9 floor forbids `|` in aliases).
- `cli.py` — **out of scope for 0.23.0** (weights-only CLI unchanged; documented so it's a decision, not an omission).
- `tutorials/` + `README.md` — an outcome-model tutorial on `happiness` (regression, validated vs held-out target truth) **plus a binarized `happiness > median`** for the `predict_proba` path (no bundled binary outcome exists); a README subsection distinguishing μ̂_IPW (`outcomes()`) vs μ̂_OM (`outcomes_hat()`) vs `outcomes().weights_impact_on_outcome_ss()`.
- **Expanded tests:** seeded-bootstrap equality (same seed identical / different differ, tolerances); weighted-linear `μ̂_OM == μ̂_DR` numeric identity (+ non-zero augmentation for boosting); standalone-fit→transfer alignment + covariate-mismatch substring; binary `predict_proba` prevalence; missing-Y drop + weight renormalization; native-cat vs one-hot-fallback equivalence (dual sklearn guards); degenerate cases (empty / single-row / single-class Y) raising actionable errors; deepcopy preserves `_outcome_model` + `_prediction_metadata`; lone-view `mean_with_ci(ci_method="bootstrap")` raises.
- **Actionable-error substrings to add:** transfer with non-deterministic `transformations`; single-class binary outcome; `learner` dict key not an outcome column; bootstrap on a lone-SampleFrame view.

**Review-driven additions (round 6 — persistence/DR adversarial pass):**
- `sample_frame.py` — `SampleFrame.__deepcopy__` copies `_outcome_model`/`_prediction_metadata` (getattr-default; **reference-share** the estimator, deep-copy only metadata); `predict_outcomes` names columns `<outcome>_hat`; `from_frame` **warns** when an unclaimed `_hat`-suffixed column would be inferred as a covariate; `keep_only_some_rows_columns` invalidates the model on row-drop; the role-prune consumer is at `balance_frame.py:3437-3444` (use `.get`).
- `balance_frame.py` — `set_target` (both paths) **preserves** `_outcome_model` across the reset; `fit_outcome_model` re-fit **drops superseded `outcomes_hat` columns**; the bootstrap routine **refits with the stored `fit_weight`** (not active weights) and guards on `_sf_target is not None` (not `isinstance(BalanceFrame)`); `summary()` reports `fit_weight` and scopes the DR statement to it.
- `to_sample`/`from_sample` — carry the `outcomes_hat` role through round-trips (complements the `_hat` convention).
- Docs/comments — record the **AIPW normalization contract** (same-scale `w_R`/`w_T`; asymptotic Hájek DR, `O(1/n)`) for phase 2; keep the **two-ways-to-DR** distinction (Path 1 AIPW / Path 2 weighted-linear).
- **Tests:** model survives `trim()`/`adjust()` (deepcopy) AND the estimator is reference-shared (identity); re-fit drops stale Ŷ; `set_target` preserves the model (both fit orders); a round-tripped `_hat` column does **not** leak into covariates; lone/target-less-`Sample` bootstrap raises; bootstrap refit uses the stored fit-weighting; the `summary()` DR statement is scoped to the fit-weights.
- **Phase-2 note:** the AIPW residual should be **cross-fitted** (out-of-fold `ŷ_R`) to avoid own-observation optimism — don't lock the persistence API into in-sample-only residuals.

### Diff stack (self-contained, reviewable)

Ship as an ordered stack; **each diff compiles, passes tests, and is independently reviewable**
(copilot-instructions §0 "one idea per PR"). Every title starts with `[balance]`. Phase 1 (diffs
1–9) delivers the g-computation estimate; phase 2 (AIPW) is a separate later stack. The stack is
strictly bottom-up: pure data-model/rename first, then the standalone learner package, then
SampleFrame wiring, then BalanceFrame orchestration, then inference. **There is no separate docs
diff** — every diff carries its own docstrings (with usage examples), a `CHANGELOG.md` entry, and
whatever `ARCHITECTURE.md`/`README.md`/tutorial updates its own functionality warrants (its `*Docs:*`
line below).

> **Note (as executed):** the stack deviated slightly from this plan — the design doc (this base diff) and the end-to-end tutorial (the top-of-stack diff) each ship as their own standalone diff, rather than being folded into the functional diffs.

**Every diff ships a runnable example in three lockstep places** (its `*Example:*` below): (1) a
code snippet with expected outputs in the **Phabricator summary**, (2) a **test** in the same diff
that runs those exact calls and asserts those outputs, and (3) the same example in the **docstring**
of the new public API (copilot §4). The three must agree. If a diff can't produce a user-visible
snippet, that's a signal it is mis-scoped.

1. **`[balance] Rename the predicted-outcome role to outcomes_hat`** — pure rename, **no behavior change**.
   - *Example* (→ summary; asserted by a test; mirrored in the docstring):
     ```python
     sf.outcomes_hat_columns                    # -> ["p_y"]  (renamed from predicted_outcome_columns)
     hasattr(sf, "predicted_outcome_columns")   # -> False    (old name removed outright)
     ```
   - *Scope:* `_column_roles` key `"predicted"`→`"outcomes_hat"`; `_create`/`from_frame`/`Sample.from_frame` kwarg + internal locals + overlap-dict key `predicted_outcome_columns`→`outcomes_hat_columns`; property renamed; **remove the old names outright** (no alias/`FutureWarning`); update the role-prune consumer (`balance_frame.py:3437-3444`, use `.get`); module docstring.
   - *Files:* `sample_frame.py`, `sample_class.py`, `balance_frame.py`, existing role tests, `CHANGELOG.md`.
   - *Tests:* existing role tests updated green; `outcomes_hat_columns` works and the old name is gone.
   - *Docs:* `CHANGELOG.md` rename note (no `## Deprecations` — the role was inert) + a `## Documentation` bullet linking this design doc; `ARCHITECTURE.md` — update any `predicted`-role references in the column-role/accessor tables; docstrings on the renamed accessors.
   - *Deps:* none.

2. **`[balance] outcomes_hat data-model accessors + add_outcomes_hat_column`** — attach/read predicted columns; **no learner yet**.
   - *Example* (→ summary; asserted by a test; mirrored in the docstring):
     ```python
     sf.add_outcomes_hat_column("happiness_hat", pd.Series([52., 58., 68., 79.]))
     sf.df_outcomes_hat["happiness_hat"].tolist()   # -> [52.0, 58.0, 68.0, 79.0]
     list(sf.df_covars.columns)                     # -> ["age"]   (Ŷ is not a covariate)
     ```
   - *Scope:* `df_outcomes_hat` + `_outcomes_hat_columns` (SampleFrame) and BalanceFrame delegates; add `_outcomes_hat_columns` to the `BalanceDFSource` protocol (docstring member count); `add_outcomes_hat_column(name, values, metadata)` (mirrors `add_weight_column`) enforcing the `<outcome>_hat` naming convention; `_prediction_metadata` attr (init in `_create`, class annotation, `__deepcopy__`); `from_frame` **warns** on an unclaimed `_hat`-suffixed column; `to_sample`/`from_sample` carry the `outcomes_hat` role.
   - *Files:* `sample_frame.py`, `balance_frame.py`, `balancedf_class.py` (protocol), tests.
   - *Tests:* add/read/`df_outcomes_hat`; `_hat` warning; deepcopy of `_prediction_metadata`; round-trip role preserved; covariate exclusion still holds after an add.
   - *Docs:* `CHANGELOG.md` (new accessors + `add_outcomes_hat_column` + `_hat` convention + `from_frame` warning); `ARCHITECTURE.md` accessor-naming table (`outcomes_hat_columns`/`df_outcomes_hat`/`_outcomes_hat_columns`) + `BalanceDFSource` member count; usage-example docstrings on every new public accessor/method.
   - *Deps:* 1.

3. **`[balance] BalanceDFOutcomesHat view + outcomes_hat() factory`** — weighted mean/CI over attached Ŷ columns.
   - *Example* (→ summary; asserted by a test; mirrored in the docstring):
     ```python
     sf.add_outcomes_hat_column("happiness_hat", pd.Series([52., 58., 68., 79.]))
     sf.outcomes_hat().mean()                    # -> weighted mean of Ŷ (one column per Ŷ)
     SampleFrame.from_frame(df).outcomes_hat()   # -> None   (no outcomes_hat columns)
     ```
   - *Scope:* `class BalanceDFOutcomesHat(BalanceDF)` (`name="outcomes_hat"`), inherits mean/std/var_of_mean/ci_of_mean/plot; `outcomes_hat()` factory on SampleFrame + BalanceFrame (None if empty); export in `__init__.py`.
   - *Files:* `balancedf_class.py`, `sample_frame.py`, `balance_frame.py`, `__init__.py`, tests.
   - *Tests:* `outcomes_hat().mean()`/`mean_with_ci()` on manually-attached columns; None when empty; linked self/target expansion.
   - *Docs:* `CHANGELOG.md` (`BalanceDFOutcomesHat` + `outcomes_hat()`); `ARCHITECTURE.md` role-specific-views list + `__init__` export note; docstrings with examples.
   - *Deps:* 2.

4. **`[balance] Add weighted R²/deviance to weighted_stats`** — reusable stats primitive.
   - *Example* (→ summary; asserted by a test; mirrored in the docstring):
     ```python
     from balance.stats_and_plots.weighted_stats import weighted_r2
     weighted_r2(pd.Series([1., 2., 3.]), pd.Series([1.1, 1.9, 3.2]), w=pd.Series([1., 1., 2.]))
     # -> 0.9636363636363636   (unweighted call equals sklearn.metrics.r2_score)
     ```
   - *Scope:* `weighted_r2` (+ Gaussian `weighted_deviance_explained`) in `stats_and_plots/weighted_stats.py`, from existing `weighted_mean`/`weighted_var`.
   - *Files:* `stats_and_plots/weighted_stats.py`, tests.
   - *Tests:* weighted R² vs known values; unweighted reduces to `sklearn.metrics.r2_score`.
   - *Docs:* `CHANGELOG.md` (`weighted_r2`/`weighted_deviance_explained`); docstrings with examples; `ARCHITECTURE.md` supporting-modules note if applicable.
   - *Deps:* none (land any time before 5).

5. **`[balance] outcome_models package: learner + fit/predict + preprocessing`** — **pure functions on DataFrames**, no frame wiring.
   - *Example* (→ summary; asserted by a test; mirrored in the docstring; names provisional):
     ```python
     from balance.outcome_models import fit_outcome_model, predict_outcome
     model = fit_outcome_model(covars_R, outcomes_R, sample_weight=w_R)  # -> {"fit", "X_matrix_columns", ...}
     predict_outcome(model, covars_T)["happiness"][:3]                   # -> ŷ on new (target) data
     ```
   - *Scope:* new `outcome_models/` package: `_resolve_learner` (`HistGradientBoosting{Reg,Clf}` default via `_is_discrete_series`; sklearn<1.4 one-hot fallback; `calibrate` opt-in via `CalibratedClassifierCV`); the weighted fit routine (`build_design_matrix` train mode; store estimator + `X_matrix_columns` + `fit_scaler` + `categorical_levels` + `fit_matrix_type` + `fit_weight` + `perf` + `prediction_kind`); predict/replay (`project_to_columns`; `.predict` / `.predict_proba[:,1]`); learner-dependent `use_model_matrix` + densify for boosting; `na_action="drop"` rejection; a learner-string `Literal` in `typing.py`; extend `utils/model_matrix.py` to accept stored `categorical_levels`.
   - *Files:* `outcome_models/` (new), `typing.py`, `utils/model_matrix.py`, tests.
   - *Tests:* fit/predict on plain frames; native-cat vs one-hot-fallback equivalence (dual sklearn guards); `sample_weight`; densify; `categorical_levels` replay with novel/missing categories; binary `predict_proba`; weighted `perf`; `na_action="drop"` raises.
   - *Docs:* `CHANGELOG.md` (new internal `outcome_models/` package — likely under *Code Quality & Refactoring*); `ARCHITECTURE.md` file-layout (add `outcome_models/`) + a stub *Outcome-model workflow* section; module + function docstrings.
   - *Deps:* 4.

6. **`[balance] SampleFrame.fit_outcome_model / predict_outcomes / fit_predict_outcomes + storage`** — **standalone** (no target).
   - *Example* (→ summary; asserted by a test; mirrored in the docstring):
     ```python
     sf.fit_outcome_model()                         # fit + store (model="auto"); does NOT persist Ŷ
     sf.outcome_model["method"]                     # -> "outcome_model"
     sf.predict_outcomes()                          # persists "<outcome>_hat" columns
     sf.df_outcomes_hat["happiness_hat"].tolist()   # -> [...]  (in-sample ŷ on the responders)
     ```
   - *Scope:* wire `outcome_models` into SampleFrame; `_outcome_model` attr (init in `_create`, class annotation, `__deepcopy__` **reference-sharing the estimator** + `getattr` default); `outcome_model` property; the sklearn trio (`fit_outcome_model` no-populate, `predict_outcomes` persists `<outcome>_hat`, `fit_predict_outcomes`); **re-fit drops superseded Ŷ**; the `model=`/`variables=` args + `weighted=False`/`transformations=None` defaults; keyword-only signatures.
   - *Files:* `sample_frame.py`, tests.
   - *Tests:* standalone fit→predict; deepcopy preserves model with **estimator identity** shared; re-fit drops stale Ŷ; missing-Y dropped + weights renormalized; positional args rejected.
   - *Docs:* `CHANGELOG.md` (`fit_outcome_model`/`predict_outcomes`/`fit_predict_outcomes` + `outcome_model` on `SampleFrame`); `ARCHITECTURE.md` *Outcome-model workflow* (standalone fit/store, the fit-artifact analog) + responsibility rows; usage-example docstrings.
   - *Deps:* 3, 5.

7. **`[balance] BalanceFrame outcome-model transfer + point estimate`** — apply to the target.
   - *Example* (→ summary; asserted by a test; mirrored in the docstring):
     ```python
     bf = sample.set_target(target)
     bf.fit_outcome_model(); bf.predict_outcomes(on="target")
     bf.outcomes_hat().mean()          # -> μ̂_OM: weighted mean of Ŷ on the target (the estimate)
     ```
   - *Scope:* generalize `_compute_ipw_matrices`→`_compute_design_matrices(model, source, side)`; `BalanceFrame.fit_outcome_model`/`fit_predict_outcomes` (delegate to responder); `predict_outcomes(on="sample"|"target"|"both")` (replay to target, **deep-copy `_sf_target` before writing**); `outcome_model`/`_outcomes_hat_columns` **delegate** to `_sf_sample`; `outcomes_hat().mean()` **raises** if the target is unpopulated; `set_target` **preserves** the model; `keep_only_some_rows_columns` **invalidates** on row-drop; verify `Sample` MRO facade.
   - *Files:* `balance_frame.py`, `balancedf_class.py`, `sample_class.py`, tests.
   - *Tests:* fit on sample → transfer → `outcomes_hat().mean()`[target] = μ̂_OM; raises when target unpopulated; `set_target`/`adjust` preserve the model (both fit orders); row-filter invalidates; no covariate leak after populate + re-adjust.
   - *Docs:* `CHANGELOG.md` (target transfer + `outcomes_hat().mean()` estimate); **README** subsection (μ̂_IPW vs μ̂_OM vs `outcomes().weights_impact_on_outcome_ss()`) + a **new tutorial notebook** (`happiness` regression validated vs held-out target truth **and** a binarized `happiness > median` `predict_proba` path); `ARCHITECTURE.md` *Where each responsibility lives* outcome rows + workflow completion; docstrings.
   - *Deps:* 6.

8. **`[balance] Bootstrap CI + scoped-DR summary for outcomes_hat`** — honest inference.
   - *Example* (→ summary; asserted by a test; mirrored in the docstring):
     ```python
     bf.outcomes_hat().mean_with_ci(ci_method="bootstrap", n_bootstrap=200, random_seed=2020)
     # -> per outcome: estimate + (ci_low, ci_high)
     bf.outcomes_hat().summary()       # -> "Estimator:" line: "g-computation (model=...); doubly robust w.r.t. weights weight"
     ```
   - *Scope:* bootstrap routine in `outcome_models` (resample responders→refit→predict on the **fixed** target→average; keep **B outputs**; `random_seed`; refit uses the stored `fit_weight`); `BalanceDFOutcomesHat.mean_with_ci(ci_method="bootstrap", ...)` **override** delegating to BalanceFrame, **lone/target-less guard on `_sf_target`**; `summary()` override reporting `fit_weight` + **scoped DR statement**; `ci_method` `Literal` in `typing.py`.
   - *Files:* `outcome_models/`, `balancedf_class.py`, `balance_frame.py`, `typing.py`, tests.
   - *Tests:* seeded bootstrap equality (same seed identical / different differ); lone/target-less-`Sample` raises; bootstrap refit uses stored fit-weighting; `summary()` DR statement scoped to fit-weights; analytic-vs-bootstrap CIs differ.
   - *Docs:* `CHANGELOG.md` (`mean_with_ci(ci_method="bootstrap")` + scoped-DR `summary()`); extend the tutorial + README with the bootstrap CI and the DR-scoping note; `ARCHITECTURE.md` note the bootstrap CI in the workflow; docstrings.
   - *Deps:* 7.

9. **`[balance] set_fitted_outcome_model (train/holdout transfer)`** — foreign-frame transfer.
   - *Example* (→ summary; asserted by a test; mirrored in the docstring):
     ```python
     scored = holdout_bf.set_fitted_outcome_model(train_bf, inplace=False)
     scored.predict_outcomes(on="target")
     scored.outcomes_hat().mean()      # -> μ̂_OM on the holdout target via train_bf's fitted model
     ```
   - *Scope:* `BalanceFrame.set_fitted_outcome_model(fitted, *, inplace)` mirroring `set_fitted_model`; covariate-name validation (reuse the `"matching sample covariate column names"` substring); reject non-deterministic `transformations` + `na_action="drop"` on transfer; **share the fitted estimator by identity**.
   - *Files:* `balance_frame.py`, tests.
   - *Tests:* train/holdout transfer alignment; covariate-mismatch raises; non-deterministic transformations rejected on transfer; estimator identity shared.
   - *Docs:* `CHANGELOG.md` (`set_fitted_outcome_model`); extend the tutorial/README with a train/holdout-transfer example (mirroring `set_fitted_model`); `ARCHITECTURE.md` fit-artifact-workflow section; docstrings.
   - *Deps:* 7.

Cross-cutting doc reminders (applied within whichever diff introduces the change): the
`transformations=None` §5.5-A divergence justification and the CHANGELOG framing live in §14;
the "Test website build" job runs `make_docs.sh -n`, so any README/CHANGELOG links must be
absolute GitHub URLs; the tutorial notebook (created in diff 7, extended in 8–9) must execute
end-to-end in CI.

**Phase 2 (separate later stack):**
- **`[balance] outcomes_hat().aipw() doubly-robust estimator`** — combine `outcomes_hat` + IPW weights; **assert** the same-scale `w_R`/`w_T` normalization contract; **cross-fitted** (out-of-fold) responder residuals; AIPW variance/CI.
- Optional follow-ons: weighted/Bayesian bootstrap; BCa intervals.

---

## 13. Design decisions status & deferred phases

All headline design questions are now resolved (see the [§0 decisions log](#0-decisions-log)):
naming (`outcomes_hat`, incl. the internal role key), scope (OM now / AIPW later), estimate home
(`outcomes_hat().mean()` / `mean_with_ci()`, mirroring `outcomes()`), weighted fitting, default
learner (`HistGradientBoosting`), module (`outcome_models/`), method names, bootstrap CI,
learner-dependent preprocessing with sklearn < 1.4 one-hot fallback, binary → predicted
probability, and `summary()` flagging the estimator type. Round-4 refinements: standalone fit
**stores the fitted preprocessing** and replays it (default `transformations=None`); the bootstrap
engine lives in `outcome_models/` (keeps B outputs, not B models), is **orchestrated by
`BalanceFrame`**, and is surfaced via an **override** of `outcomes_hat().mean_with_ci`; the sklearn
**`fit`/`predict`/`fit_predict` trio** (no auto-populate).

**Round-5 (adversarial review) resolutions:** DR is delivered by the **AIPW combination** — the
model is fit *independently* of the IPW weights, and `outcomes_hat` (`= ĝ(X)`, weight-invariant)
+ the model **persist across `adjust()`** (fitting-after-adjust becomes a linear-only special case,
not the rule); binary calibration is **lightweight-default + `calibrate=True` opt-in** with a
diagnostic; the estimate **raises** on a missing target; the bootstrap `mean_with_ci` override
**delegates to a bespoke BalanceFrame routine** (lone-view raises). Plus fixes: Hájek (not HT)
labelling; `categorical_levels` correctness; a new weighted-R² `perf` helper; per-learner
matrix-type/densify; raw-path `na_action` guard; delegate-not-copy accessors; `fit_intercept` DR
guard; CLI out-of-scope; expanded tests/errors/docs (§12).

**Round-6 (persistence/DR adversarial pass) resolutions:** implicit-DR is **not** claimed by
default — `summary()` reports the fit-weights and *scopes* any DR statement to them; `outcomes_hat`
columns use the `<outcome>_hat` convention + a `from_frame` leak-warning; `mean_with_ci` defaults to
bootstrap and **raises on a lone/target-less view**; the AIPW normalization contract (same-scale
weights; asymptotic Hájek DR) is documented for phase 2; and lifecycle fixes — deepcopy preserves
the model (**reference-sharing** the estimator), `set_target` preserves it, re-fit drops stale Ŷ,
and the bootstrap refits with the stored fit-weighting. Plus coherence fixes: Hájek sweep, demoted
the special case, marked `mean_with_ci` overridden, disambiguated `w_R^fit` vs `w_R`, and added a
canonical two-ways-to-DR callout (§1). **With these folded in, the design is ready to implement.**

Deferred to implementation / later phases (not blocking this design):

- **AIPW / doubly-robust** (`outcomes_hat().aipw()`) — the phase-2 estimator combining weights +
  outcome predictions; needs its own variance/CI design.
- **AIPW normalization contract** — require `adjust()`-calibrated `w_R`/`w_T` on the same
  population scale + assert `|Σw_R−Σw_T|/Σw_T < tol`; treat DR as asymptotic (Hájek `O(1/n)`).
- **Weighted / Bayesian bootstrap** — a robustness upgrade over the nonparametric bootstrap.
- **BCa intervals** — an accuracy upgrade over percentile CIs if warranted.
- **Cross-fitting** — to reduce own-observation bias when the same responders fit `ĝ` and enter
  the (future) AIPW augmentation term.

---

## 14. Changelog & doc placement

- Insert into a `# 0.23.0 (Unreleased - TBD)` section — **create it at the top of `CHANGELOG.md`,
  above the now-released `# 0.22.0`** (do not reuse the shipped 0.22.0 section); use the
  bold-lead-sentence bullet style; flag any breaking change with `**Breaking:**`.
- CHANGELOG/README markdown links must be **absolute GitHub URLs** (Docusaurus builds with
  `onBrokenLinks:'throw'`).
- Document the **`transformations=None` default** (vs IPW's `"default"`) as a deliberate §5.5-A
  divergence — justified by replay-safety (data-dependent `quantize`/`fct_lump` can't be frozen for
  transfer) and the boosting default's non-need for binning.
- Cite the methods paper for estimator theory: Sarig, Galili & Eilat (2023),
  arXiv:2307.06024.
