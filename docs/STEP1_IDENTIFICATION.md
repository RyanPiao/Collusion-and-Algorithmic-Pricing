# Step 1 Identification Notes

## Empirical objective
Estimate the local causal effect of Smart Pricing exposure on nightly listing prices around policy rollout timing.

## Design summary
A fuzzy regression discontinuity design (RDD) is used rather than a sharp design because adoption is incomplete and heterogeneous near the cutoff.

## Components
1. **Policy timing cutoff**
   - Primary rollout threshold centered on Smart Pricing introduction timing.
   - Secondary marker retained for public-availability diagnostics.

2. **Running variable**
   - Calendar time indexed as days relative to cutoff.

3. **Eligibility and assignment**
   - Listing-night eligibility proxied by `available == 1`.
   - Post-cutoff indicator captures assignment pressure at policy timing.

4. **Fuzzy treatment intensity**
   - Uptake propensity modeled from host/listing characteristics (tenure, response/acceptance behavior, hosting scale, booking settings, and related controls).
   - Predicted uptake used as treatment intensity in the second stage.

5. **Outcome and controls**
   - Outcome: nightly price.
   - Controls: host traits, listing attributes, and time structure variables.

## Inference and diagnostics (current)
- HC-robust estimation in second-stage regressions.
- Bandwidth sensitivity checks.
- Collinearity, heteroskedasticity, RESET, and residual diagnostics.

## Planned upgrades
- Report first-stage strength metrics in main output tables.
- Add placebo cutoffs and pre-trend stress tests.
- Harmonize specification reporting for cross-city extension.
