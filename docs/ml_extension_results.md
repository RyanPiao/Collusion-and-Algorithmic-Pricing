# ML-Econometrics Extension Results (Computed)

## Run Scope
- Script: `scripts/ml_unsupervised_extension.py`
- Input panel: `data/processed/step2/fact_listing_day_multicity_bw_3m.csv.gz`
- Estimation sample: 24,228,568 listing-step observations (±3 month window)
- Listing-level latent proxy count: 131,677 listings
- Clustering models: KMeans (`k=4`) and GMM (`k=4`)

## Output Artifacts
- `data/processed/ml_extension/listing_latent_proxy.csv`
- `data/processed/ml_extension/listing_cluster_membership.csv`
- `data/processed/ml_extension/first_stage_comparison.csv`
- `data/processed/ml_extension/second_stage_comparison.csv`
- `data/processed/ml_extension/run_summary.json`

## Proxy Distribution
From `run_summary.json`:
- Min: 0.0000
- Mean: 0.2913
- Max: 1.0000

Interpretation: this is a relative latent adoption propensity proxy index (normalized), not observed or true Smart Pricing adoption.

## First-Stage Comparison (HC1 Robust SE)
Source: `first_stage_comparison.csv`

1. **Baseline first stage** (`available ~ post_cutoff + controls`)
   - Coef on `post_cutoff`: **-0.04450**
   - SE: 0.00038
   - t: -118.02
   - 95% CI: [-0.04524, -0.04377]
   - R²: 0.1357

2. **ML first stage** (`latent_adoption_propensity_proxy ~ post_cutoff + controls`)
   - Coef on `post_cutoff`: **0.00003**
   - SE: 0.00011
   - t: 0.24
   - p: 0.8109
   - 95% CI: [-0.00019, 0.00024]
   - R²: 0.1360

Readout: the baseline proxy shifts materially at cutoff in this specification, while the pre-built latent proxy is effectively unchanged around cutoff timing (as expected for a pre-cutoff, listing-level index).

## Second-Stage Comparison (HC1 Robust SE)
Source: `second_stage_comparison.csv`

1. **Baseline second stage** (`log_price ~ available + controls`)
   - Coef on `available`: **0.10145**
   - SE: 0.00034
   - t: 298.87
   - 95% CI: [0.10079, 0.10212]
   - R²: 0.3080

2. **ML second stage** (`log_price ~ latent_adoption_propensity_proxy + controls`)
   - Coef on latent proxy: **1.43645**
   - SE: 0.00114
   - t: 1260.63
   - 95% CI: [1.43422, 1.43869]
   - R²: 0.3474

Readout: listings with higher latent adoption propensity proxy values are strongly associated with higher log prices in pooled comparisons.

## Interpretation Constraints (Important)
1. **No true adoption label**: the latent index is unsupervised and should not be interpreted as verified Smart Pricing participation.
2. **Associational second stage**: these regressions do not alone identify a causal treatment effect of algorithm adoption.
3. **Specification dependence**: magnitude depends on feature engineering, clustering choices, and normalization.
4. **Time-invariant proxy behavior**: the weak first-stage shift for the latent proxy is mechanically consistent with pre-cutoff construction.
5. **Policy inference caution**: use this extension as a heterogeneity/proxy layer, not a replacement for validated treatment measurement or full IV identification.

## Refined Dynamic Proxy and Heterogeneous Effects

### Dynamic pre-cutoff feature redesign (leakage-safe)
The refined pipeline in `scripts/ml_unsupervised_extension.py` now computes behavioral features using **only** rows where `post_cutoff == 0`:

- `price_variance_pre`: listing-level variance of daily `log_price` before cutoff.
- `weekend_premium_pre`: mean(`price_usd` on Fri/Sat) minus mean(`price_usd` on weekdays) before cutoff.
- `price_change_frequency_pre`: share of pre-cutoff step-to-step transitions with a non-zero price change.

Implementation notes:
- Missing variance for listings with <2 pre-cutoff observations is median-imputed (post-filter, pre-clustering).
- Weekend premium / change frequency missingness is also median-imputed to retain listings while preventing leakage.
- Dynamic features are added to the standardized clustering feature matrix along with the existing static/pre-period features.

### Updated latent proxy (GMM k=4)
The unsupervised stage reruns KMeans/GMM with the expanded feature set and exports:

- `data/processed/ml_extension/listing_features_refined.csv`
- `data/processed/ml_extension/listing_cluster_membership.csv`
- `data/processed/ml_extension/listing_latent_proxy.csv`
- `data/processed/ml_extension/run_summary.json`

`run_summary.json` now reports the refined latent proxy distribution (`min`, `mean`, `max`) for the updated `latent_adoption_propensity_proxy`.

### Heterogeneous fuzzy-RDD / IV interaction model
The refined evaluation includes an interaction IV design:

- **First stage**: `available ~ post_cutoff + latent_proxy + post_cutoff*latent_proxy + controls`
- **Second stage**: `log_price ~ available_hat + latent_proxy + available_hat*latent_proxy + controls`

Export:
- `data/processed/ml_extension/heterogeneous_iv_interaction_results.csv`

This file contains HC1 robust coefficient tables for both stages (including interaction terms):
- First-stage interaction: `post_x_latent`
- Second-stage heterogeneous treatment interaction: `available_hat_x_latent`

### PSM-style High-vs-Low propensity DiD + Event Study
New script: `scripts/ml_extension_psm_did.py`

Design:
- High propensity (treatment): top quartile of latent proxy.
- Low propensity (control): bottom quartile of latent proxy.
- TWFE DiD around rollout/event time 0 on this restricted sample.
- Event-study dynamics over a symmetric window (default `[-30, 30]`, reference period `t=-1`).

Exports:
- `data/processed/ml_extension/psm_did_twfe_results.csv`
- `data/processed/ml_extension/psm_did_event_study.csv`
- `data/processed/ml_extension/psm_did_event_study_plot.png`
- `data/processed/ml_extension/psm_did_summary.json`

Interpretation focus:
- `high_propensity_x_post` in TWFE DiD captures differential post-rollout price response for high-vs-low propensity listings.
- Event-study coefficients trace pre-trend validity and dynamic post effects by propensity group.
