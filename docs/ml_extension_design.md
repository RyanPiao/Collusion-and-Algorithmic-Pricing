# ML-Econometrics Extension Design: Latent Adoption Propensity Proxy

## Objective
This extension augments the multicity fuzzy-RDD pipeline with an **unsupervised, pre-policy latent adoption propensity proxy**. The proxy is designed to summarize host/listing characteristics associated with likely Smart Pricing uptake intensity, while explicitly avoiding post-cutoff information leakage.

The estimand remains unchanged from the broader design: local pricing responses around rollout timing. The ML layer supplies an additional empirical regressor/proxy; it does **not** identify true treatment status.

## Econometric Positioning in the Existing Design
The baseline Step 4 framework uses listing-step availability as the main uptake-related proxy near policy timing. The ML extension adds a listing-level latent proxy constructed from pre-cutoff observables only:

- Baseline proxy: `available` (listing-step, time varying).
- ML proxy: `latent_adoption_propensity_proxy` (listing-level, time invariant within the ±3 month window).

This allows a structured comparison of reduced-form model behavior under two different proxy constructions.

## Identification Logic (What the Extension Can and Cannot Do)
1. **Core source of quasi-experimental timing remains rollout cutoff logic** from the fuzzy-RDD setup.
2. The ML extension does **not** introduce a new instrument; it introduces a new proxy regressor.
3. The latent proxy is interpreted as a **pre-policy heterogeneity index** (host/listing sophistication and platform-readiness dimensions), not observed adoption.
4. Any coefficient on the latent proxy is associational in this implementation unless additional exclusion/first-stage structure is imposed.

## Leakage Controls (Critical)
To reduce post-treatment leakage risk, the feature-construction protocol is strictly pre-cutoff:

- Input panel: `fact_listing_day_multicity_bw_3m.csv.gz`.
- Feature extraction subset: only rows with `post_cutoff == 0`.
- Listing-level aggregation: means/counts over pre-cutoff rows only.
- No post-cutoff outcomes or post-cutoff behavior are used in clustering.

This ensures the latent proxy is fixed before cutoff in the analysis sample.

## Feature Construction
### Numeric pre-cutoff aggregates (listing-level means)
- `log_price_pre`, `available_pre`
- `minimum_nights_pre`, `maximum_nights_pre`
- `accommodates_pre`, `bathrooms_pre`, `bedrooms_pre`
- `host_tenure_days_pre`
- `host_is_superhost_pre`, `host_identity_verified_pre`
- `host_response_rate_pre`, `host_acceptance_rate_pre`
- `pre_obs` (number of pre-cutoff listing-step observations)

### Categorical features
- `city_slug`
- `room_type`
- `property_type`

## Unsupervised Model Choices
Two unsupervised models are estimated on the same preprocessed feature matrix:

1. **KMeans**
   - Role: hard partition benchmark.
   - Hyperparameters: `n_clusters=4`, `n_init=20`, `random_state=42`.

2. **Gaussian Mixture Model (GMM)**
   - Role: soft assignment and posterior probabilities.
   - Hyperparameters: `n_components=4`, `covariance_type='full'`, `n_init=3`, `max_iter=300`, `reg_covar=1e-6`, `random_state=42`.

### Preprocessing
- Numeric columns: standardized via `StandardScaler`.
- Categorical columns: one-hot encoded via `OneHotEncoder(handle_unknown='ignore')`.

## Constructing the Latent Adoption Propensity Proxy
The extension constructs a continuous proxy using GMM posterior probabilities:

1. Compute a pre-cutoff sophistication/readiness index at listing level:
   - Positive loadings on response/acceptance rates, superhost, identity verification, and larger accommodates.
   - Negative loading on minimum nights.
2. Compute cluster-level means of this index (by GMM cluster).
3. Normalize cluster-level scores to `[0,1]`.
4. Form listing-level latent proxy as probability-weighted cluster score:
   \[
   \text{latent\_proxy}_i = \sum_k \Pr(k \mid X_i) \cdot w_k
   \]
5. Normalize final listing-level proxy to `[0,1]`.

Interpretation: relative latent propensity ranking, not true adoption probability.

## Econometric Comparison Layer
Using the full ±3 month listing-step panel, the script estimates four OLS models with city fixed effects and local running-variable controls (`days_from_cutoff`, `post_cutoff × days_from_cutoff`). HC1 robust standard errors are reported.

### First-stage comparison
- Baseline first stage: `available ~ post_cutoff + controls`
- ML first stage: `latent_adoption_propensity_proxy ~ post_cutoff + controls`

### Second-stage comparison
- Baseline second stage: `log_price ~ available + controls`
- ML second stage: `log_price ~ latent_adoption_propensity_proxy + controls`

These are comparison regressions, not definitive structural treatment-effect estimates.

## Validation and Diagnostics
The implementation includes:

- Reproducible random seeds for clustering models.
- Export of cluster memberships and GMM posterior probabilities per listing.
- Export of model coefficients, HC1 SEs, confidence intervals, and R².
- Run summary with proxy distribution moments and output file map.

Suggested follow-up validation (future work):
- Cluster stability across different `k` and initialization seeds.
- Split-sample stability checks by city and listing type.
- Sensitivity to alternative proxy-composition weights.
- Placebo timing tests using pre-cutoff pseudo cutoffs.

## Limitations
1. **No ground-truth adoption label:** the latent proxy is not verified against true Smart Pricing uptake.
2. **Associational interpretation:** second-stage comparisons with latent proxy are not causal estimates of algorithm adoption effects.
3. **Model dependence:** proxy values depend on clustering specification and feature engineering choices.
4. **Potential omitted dimensions:** unobserved host behavior quality can still confound proxy interpretation.
5. **Time-invariant proxy within window:** this extension emphasizes cross-listing heterogeneity, not within-listing dynamic adoption timing.

## Reproducibility
Run from repository root:

```bash
source .venv/bin/activate
python scripts/ml_unsupervised_extension.py --repo-root .
```

Primary outputs:
- `data/processed/ml_extension/listing_latent_proxy.csv`
- `data/processed/ml_extension/listing_cluster_membership.csv`
- `data/processed/ml_extension/first_stage_comparison.csv`
- `data/processed/ml_extension/second_stage_comparison.csv`
- `data/processed/ml_extension/run_summary.json`
