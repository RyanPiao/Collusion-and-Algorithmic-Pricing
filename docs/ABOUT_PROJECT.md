# About This Project

## Title
Algorithmic Pricing, Market Conduct, and Antitrust Risk in U.S. Major Cities

## Purpose
This project evaluates whether Airbnb Smart Pricing rollout timing is associated with discontinuous changes in listing-level prices, and whether effects are heterogeneous across latent host adoption propensity segments.

## Empirical Design
- Multicity fuzzy RDD / IV around policy cutoff timing.
- Robustness checks (bandwidth/placebo/continuity style diagnostics).
- ML-econometrics extension using strictly pre-cutoff dynamic pricing features.
- Propensity-stratified DiD/event-study follow-up.

## Current Bottom Line
The baseline pooled discontinuity estimates are near zero and statistically imprecise; heterogeneity layers reveal structured differences in responsiveness but do not overturn the average near-null cutoff result.

## Important Data Construction Caveat
Inside Airbnb data comes from periodic forward-calendar scrapes. The resulting daily panel is a snapshot of scheduled future prices at scrape time, not continuous between-scrape real-time edits. Volatility results should therefore be interpreted as forward schedule complexity/differentiation rather than definitive proof of within-interval dynamic repricing.

## Where to Read More
- Working paper: `docs/working_paper_us_major_cities.md`
- ML extension results: `docs/ml_extension_results.md`
