# Algorithmic Pricing, Market Conduct, and Antitrust Risk in Short-Term Rentals: Evidence from Airbnb Smart Pricing

## Abstract
This project studies whether platform-assisted algorithmic pricing is associated with discontinuous changes in Airbnb listing prices or fosters tacit collusion. Using an eight-city daily panel (Austin, Boston, Chicago, Los Angeles, New York City, San Francisco, Seattle, Washington, DC), we implement a multicity fuzzy-RDD/IV design, an unsupervised machine learning heterogeneity extension, and longitudinal panel methods (TWFE and Propensity-Stratified DiD).

Our baseline pooled estimates indicate that the instrumented local price-level effects around rollout are economically small and statistically imprecise across all tested bandwidths (e.g., **0.0027**, p > 0.05 at ±3 months). However, using a refined latent adoption proxy built from pre-treatment dynamic pricing behavior, we find significant heterogeneity. Furthermore, longitudinal Two-Way Fixed Effects (TWFE) models relying on structural volatility breaks reveal the true causal footprint of the algorithm: **algorithmic adoption does not raise baseline average prices, but it significantly increases forward-looking price volatility (p < 0.001).** We find no evidence of neighborhood spatial spillovers (price umbrellas). The algorithm acts as a productivity multiplier for sophisticated hosts, driving complex price discrimination rather than market-wide rent inflation.

## Motivation
Algorithmic pricing systems can reduce search and adjustment costs, but they can also create synchronized responses to market signals across many sellers. That dual effect is central to modern antitrust debates in digital platforms. Airbnb offers a useful setting because Smart Pricing was introduced with clear timing and heterogeneous host adoption, enabling an empirical test of whether observed price changes are consistent with unilateral optimization or with behavior that could elevate coordination concerns.

## Research Question and Hypothesis
### Research Question
How does increased exposure to Airbnb Smart Pricing affect host nightly pricing behavior (both price levels and price volatility) around the policy introduction window?

### Hypothesis
- **H1 (Level vs. Volatility):** Algorithmic pricing adoption does not uniformly raise baseline price levels, but rather increases the frequency and variance of price adjustments (price discrimination).
- **H2 (Heterogeneity):** The behavioral impact of the algorithm is concentrated among already-sophisticated hosts who face cognitive constraints in manual dynamic pricing.
- **H3 (Coordination/Spillovers):** If the algorithm facilitates tacit collusion, we should observe "price umbrellas" where non-adopting hosts raise prices when surrounded by high algorithmic penetration.

## Identification Strategy (Summary)
The core design uses **announcement-date policy timing** with a **fuzzy RDD** implementation, supplemented by **Machine Learning** and **Longitudinal Panel** methods.

1. **Baseline Fuzzy RDD / IV:** Uses Smart Pricing rollout timing (`post_cutoff`) as an instrument for listing-step availability (`available`) to test for immediate, market-wide level shifts.
2. **Unsupervised Latent Adoption Propensity:** Uses KMeans/GMM clustering on *strictly pre-cutoff* static and dynamic features (e.g., pre-cutoff price variance, weekend premiums) to identify host sophistication and proxy algorithmic uptake without post-treatment leakage.
3. **Heterogeneous Treatment Effects (Interacted IV) & PSM-DiD:** Interacts the RDD instrument with the latent proxy, and runs a Propensity-Stratified Difference-in-Differences comparing the top quartile (High Propensity) to the bottom quartile (Low Propensity).
4. **Structural Breaks & TWFE Panel:** Uses `ruptures` to detect behavioral shifts in rolling price variance to proxy exact adoption timing, then applies Two-Way Fixed Effects models to isolate within-listing changes in price levels and volatility.

## Repository Structure
```text
.
├── README.md
├── Step_1_Data_Cleaning.ipynb
├── calendar_data_cleaning.ipynb
├── Step_2_Data_Clustering.ipynb
├── fuzzy_rdd_boston.ipynb
├── scripts/
│   ├── day2_build_multicity_panels.py
│   ├── day3_multicity_eda.py
│   ├── day4_multicity_fuzzy_rdd.py
│   ├── ml_unsupervised_extension.py
│   ├── ml_extension_psm_did.py
│   ├── panel_extension_1_structural_breaks.py
│   ├── panel_extension_2_twfe.py
│   ├── panel_extension_3_event_study.py
│   ├── panel_extension_4_spillovers.py
│   └── panel_extension_run_all.py
├── data/
│   ├── processed/step2/ ... (generated Step 2 outputs)
│   ├── processed/step3/ ... (generated Step 3 EDA outputs)
│   ├── processed/step4/ ... (generated Step 4 baseline IV outputs)
│   ├── processed/ml_extension/ ... (generated ML interaction & PSM outputs)
│   └── processed/panel_extension/ ... (generated TWFE & structural break outputs)
└── docs/
    ├── working_paper_us_major_cities.md
    └── ... (design and status documentation)
