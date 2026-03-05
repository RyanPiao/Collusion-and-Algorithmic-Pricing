# Day 3 EDA Note: Multi-city descriptives around cutoff

## 1) City-level trend behavior around the cutoff
- Across city-day means, average nightly price is 380.48 USD pre-cutoff versus 380.44 USD post-cutoff.
- Figure `docs/figures/day3/city_trend_mean_price_cutoff.png` shows within-city trajectories in relative event time with a cutoff marker at day 0.
- The visual pattern is best read as descriptive evidence for local continuity/discontinuity checks; it is not a causal estimate by itself.

## 2) Pre/Post distribution shifts
- In the ±1 month window, the average post-minus-pre shift in mean price is -0.04 USD across cities.
- In the ±1 month window, the average post-minus-pre shift in median price is 0.00 USD across cities.
- Largest positive mean shift (±1m): boston (+0.00 USD).
- Largest negative mean shift (±1m): los-angeles (-0.31 USD).
- Figure `docs/figures/day3/prepost_logprice_distribution_bw1m.png` overlays sampled log-price histograms (pre vs post) by city for shape comparison.

## 3) Treatment-support diagnostics by city/window
- Thin-support flags: 0 of 24 city-window cells are flagged thin.
- Composite support-ok flag (balanced pre/post observations and day coverage): 24 of 24 cells.
- Support diagnostics table is in `data/processed/day3/treatment_support_diagnostics_city_window.csv`.

## 4) Missingness and data quality
- Dataset-level QA checks remain clean: duplicates=0, window nesting violations=0, cutoff misalignment=0.
- Highest city-level field missingness: new-york-city at 43.13% (top field: host_response_rate).
- Full missingness summary is in `data/processed/day3/missingness_summary_city.csv`.

## Policy-facing interpretation
- The Day 3 outputs indicate where identifying support is strong enough for local fuzzy-RDD estimation and where city-window cells may require caution due to thinner support.
- Pre/post distribution movement appears heterogeneous across markets; this supports reporting city-specific descriptives before pooled structural interpretation.
- Structural QA checks are clean, but several host-side covariates have non-trivial missingness in some cities; inference specs should report covariate completeness and sensitivity to missing-data handling.
