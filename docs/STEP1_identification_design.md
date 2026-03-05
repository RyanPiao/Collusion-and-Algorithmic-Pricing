# Day 1 Identification Design

## 1) Causal estimand
Target parameter: the **local average treatment effect (LATE)** of Smart Pricing uptake on nightly price for listings near the feature announcement cutoff.

- Outcome: \(Y_{it}\) = nightly price (primary: \(\log(price_{it})\), secondary: price level)
- Treatment: \(D_{it}\) = Smart Pricing adoption/use intensity (latent or proxy-measured)
- Running variable: \(r_t = date_t - c\), where \(c\) is announcement/rollout date
- Cutoff indicator: \(Post_t = 1[r_t \ge 0]\)

## 2) Why fuzzy RDD here
Adoption does not switch from 0 to 1 deterministically at the cutoff. Instead, rollout timing shifts the **probability** of uptake. That is a classic fuzzy setting: treatment propensity jumps at the cutoff, not treatment status for everyone.

## 3) Operational design (aligned to existing notebooks)
### Assignment/eligibility channel
Use listing-day availability as eligibility proxy:
- \(E_{it} = 1[available_{it} = 1]\)

Intuition: a listing that is not available cannot effectively deploy dynamic pricing for that date; available listings are “at risk” of using Smart Pricing.

### First stage (uptake probability)
Estimate probability of uptake with host/listing characteristics and timing:
\[
\Pr(D_{it}=1) = \Lambda\left(\alpha_0 + \alpha_1 Post_t + \alpha_2 E_{it} + \alpha_3(Post_t \times E_{it}) + \alpha_4 X_{it} + f(r_t)\right)
\]
where \(X_{it}\) includes host and listing controls used in notebooks (e.g., host tenure, acceptance rate, superhost, instant-bookable, availability metrics, review scores).

**Notebook-compatible proxy:** current code uses predicted `Prob_Smart_Pricing_Tool` from a logit-like first stage.

### Second stage (local outcome equation)
Within bandwidth \(|r_t| \le h\):
\[
Y_{it} = \beta \widehat{D}_{it} + g(r_t) + \theta'X_{it} + \delta_{month} + \delta_{week} + \delta_{neighborhood} + \varepsilon_{it}
\]
- \(\widehat{D}_{it}\): predicted adoption probability from first stage
- Inference: heteroskedasticity-robust SE at minimum; preferred clustering by listing (and date, if feasible)

## 4) Identification assumptions (must be explicit)
1. **Continuity:** absent the rollout shock, potential outcomes evolve smoothly through cutoff.
2. **First-stage relevance:** treatment propensity changes at cutoff (non-zero jump).
3. **Local exclusion (interpreted carefully):** cutoff affects prices primarily through Smart Pricing uptake locally, conditional on controls and time effects.
4. **No precise manipulation of running variable:** dates are calendar-driven; hosts cannot sort observations around cutoff in the RDD sense.

## 5) Required diagnostics and falsification
1. **Cutoff validity audit**
   - Verify exact announcement/rollout date from platform communications.
2. **McCrary-style density/distribution checks**
   - Confirm no discontinuity in observation mass at cutoff.
3. **Covariate continuity tests**
   - Pre-determined covariates should not jump at cutoff.
4. **Placebo cutoffs**
   - Re-estimate at fake dates; true effect should be strongest at real cutoff.
5. **Bandwidth sensitivity**
   - Confirm coefficient stability across principled bandwidths.
6. **Functional form checks**
   - Local linear + triangular kernel preferred; compare with polynomial specs.

## 6) Relationship to current notebook results
`fuzzy_rdd_boston.ipynb` already implements core elements:
- cutoff-based design,
- first-stage propensity model,
- second-stage price regression,
- bandwidth sweep showing positive treatment coefficients.

Day-1 interpretation: **promising but not yet publication-grade** until treatment measurement and cutoff-exclusion assumptions are strengthened with explicit validation checks.

## 7) Immediate specification upgrades (priority)
- Make `log(price)` the primary outcome.
- Explicitly include \(Post_t \times E_{it}\) as first-stage relevance term.
- Use local-linear RDD estimation around cutoff as baseline.
- Report first-stage jump statistics and weak-IV diagnostics.
- Separate “announcement” and “public rollout” windows (e.g., 2023-03-25 vs 2023-05-25) as distinct designs.
