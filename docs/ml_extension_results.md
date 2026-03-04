# ML-Econometrics Extension Results (Computed)

## Run Scope
- Script: `scripts/ml_unsupervised_extension.py`
- Input panel: `data/processed/day2/fact_listing_day_multicity_bw_3m.csv.gz`
- Estimation sample: 24,228,568 listing-day observations (±3 month window)
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
