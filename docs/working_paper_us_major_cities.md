# Algorithmic Pricing, Market Conduct, and Antitrust Risk in U.S. Major Cities: Evidence from Airbnb Smart Pricing

## Abstract
This paper evaluates whether Airbnb Smart Pricing rollout timing is associated with discontinuous local shifts in listing price levels across major U.S. markets. We assemble a multicity daily listing panel and estimate a fuzzy regression discontinuity / instrumental-variables design using policy timing (`post_cutoff`) as an instrument for listing-day availability (`available`) within symmetric bandwidths around the cutoff. The analysis covers Austin, Boston, Chicago, Los Angeles, New York City, San Francisco, Seattle, and Washington, DC.

Using pooled Day 4 estimates, the instrumented local price effects are economically small and statistically imprecise in all windows: 0.0034 (SE 0.0265) in ±1 month, 0.0033 (SE 0.0174) in ±2 months, and 0.0027 (SE 0.0145) in ±3 months. In semi-elasticity terms, these correspond to approximately 0.27%–0.34% point estimates with confidence intervals that include substantively negative and positive values. First-stage relevance is strong in pooled samples (F-statistics from 4,158 to 13,906), and bandwidth/placebo diagnostics do not show stable alternative breakpoints. To motivate heterogeneity follow-up without overturning this baseline null framing, we add an ML-econometrics extension that builds an unsupervised listing-level latent adoption propensity proxy and compares first- and second-stage behavior in the ±3 month panel. In that extension, `post_cutoff` is near-zero for the latent proxy first stage (0.00003, p = 0.8109), while higher latent propensity is strongly associated with higher pooled log prices; we interpret this as a proxy-based heterogeneity signal rather than causal proof of Smart Pricing adoption effects.

## 1. Introduction
Algorithmic pricing tools are increasingly central to digital marketplace operations, but their competition-policy implications remain contested. A core concern is whether shared algorithmic adjustment rules can generate synchronized pricing behavior consistent with tacit coordination, even absent explicit communication. Airbnb Smart Pricing provides a useful empirical setting because rollout timing offers a quasi-experimental anchor and host responses are heterogeneous.

This study asks a focused question: **is there a local discontinuous change in listing price levels around Smart Pricing rollout timing in a multicity panel?** We prioritize transparent baseline identification, explicit first-stage diagnostics, and conservative interpretation. Our objective at this stage is screening for large immediate level effects, not making final structural or legal claims.

## 2. Data and Setting
### 2.1 Data construction
The empirical panel is generated from the project’s multicity pipeline and stored in processed outputs under `data/processed/day2/` through `data/processed/day4/`. The baseline estimation windows use symmetric cutoffs at ±1, ±2, and ±3 months around assigned city policy timing.

### 2.2 Geographic scope
The analysis includes eight U.S. cities:
- Austin
- Boston
- Chicago
- Los Angeles
- New York City
- San Francisco
- Seattle
- Washington, DC

### 2.3 Outcome and treatment proxy
- **Outcome:** listing-day `log_price`
- **Endogenous proxy treatment:** `available` (listing-day availability)
- **Instrument:** `post_cutoff`

This is a proxy implementation of Smart Pricing exposure and should not be interpreted as direct adoption telemetry.

## 3. Empirical Method
We estimate a local fuzzy-RDD-style IV specification around the cutoff:

1. **First stage:**
   \[
   available_{it} = \alpha + \pi\,post\_cutoff_{it} + f(days\_from\_cutoff_{it}) + X_{it}'\gamma + u_{it}
   \]

2. **Second stage:**
   \[
   log\_price_{it} = \beta\,\widehat{available}_{it} + f(days\_from\_cutoff_{it}) + X_{it}'\delta + \varepsilon_{it}
   \]

where \(f(\cdot)\) contains local linear running-variable terms and interactions consistent with the Day 4 baseline scripts. Estimation is repeated for ±1m, ±2m, and ±3m windows.

## 4. Baseline Results (Day 4)

### 4.1 First-stage relevance (pooled)
From `data/processed/day4/first_stage_strength_city_window.csv`, pooled first-stage F-statistics are:
- ±1 month: **4,158.50**
- ±2 months: **9,715.53**
- ±3 months: **13,905.54**

These values indicate strong pooled instrument relevance for the proxy treatment channel.

### 4.2 Pooled second-stage estimates
From `data/processed/day4/second_stage_pooled_window_estimates.csv`:

| Window | N | \(\hat\beta\) on `available_hat` | SE | 95% CI |
|---|---:|---:|---:|---:|
| ±1 month | 8,163,974 | 0.0034 | 0.0265 | [-0.0486, 0.0554] |
| ±2 months | 16,327,948 | 0.0033 | 0.0174 | [-0.0307, 0.0374] |
| ±3 months | 24,228,568 | 0.0027 | 0.0145 | [-0.0256, 0.0311] |

Interpretation: point estimates are close to zero and imprecise across all windows. In approximate percentage terms, these correspond to 0.34%, 0.34%, and 0.27% semi-elasticities, respectively.

### 4.3 City-window heterogeneity
From `data/processed/day4/second_stage_city_window_estimates.csv`, city-window estimates are generally imprecise, and none are conventionally significant in this baseline specification. Several city-window cells also show weaker first-stage strength, reinforcing caution against over-interpreting city-level point estimates.

## 5. Robustness and Diagnostic Evidence

### 5.1 Bandwidth sensitivity
`data/processed/day4/diagnostic_bandwidth_sensitivity_pooled.csv` shows stable near-zero pooled coefficients as bandwidth expands from ±1m to ±3m, with confidence intervals crossing zero in every window.

### 5.2 Placebo cutoffs
`data/processed/day4/diagnostic_placebo_cutoff_pooled_bw3m.csv` reports:
- True cutoff (0d): coefficient = 0.0027, p = 0.8501
- Placebo −30d: coefficient = −0.0447, p = 0.9426
- Placebo +30d: coefficient = 0.0163, p = 0.9090

No placebo specification yields a stable significant discontinuity.

### 5.3 Week 2 inference and validity checks
Additional Week 2 outputs are directionally consistent with the Day 4 baseline:
- Clustered-inference table (`data/processed/week2/inference_clustered_iv.csv`) keeps coefficients near zero across windows.
- Density checks (`data/processed/week2/density_tests.csv`) show balanced left/right mass around cutoff in the current panel construction.
- Covariate continuity (`data/processed/week2/covariate_continuity_by_bw.csv`) shows zero measured jumps for predetermined listing/host attributes in tested local windows.
- Mechanism endpoint files (`data/processed/week2/dispersion_endpoint_estimates.csv`, `data/processed/week2/comovement_endpoint_estimates.csv`) suggest some city-specific synchrony/dispersion movements but no uniform cross-city mechanism pattern.

## 6. ML-Econometrics Extension (Unsupervised Latent Proxy)
To complement the baseline proxy-IV screening result, we estimate a descriptive ML extension using outputs documented in `docs/ml_extension_results.md`.

### 6.1 Method summary
- Estimation sample: 24,228,568 listing-day observations in the ±3 month window.
- Listing-level latent proxy count: 131,677 listings.
- Construction: unsupervised clustering workflow (KMeans `k=4`, GMM `k=4`) to create a normalized latent adoption propensity proxy (`min=0.0000`, `mean=0.2913`, `max=1.0000`).
- Comparison design: baseline and ML first/second-stage specifications from `data/processed/ml_extension/first_stage_comparison.csv` and `second_stage_comparison.csv`.

### 6.2 Measured findings
- **Baseline first stage (`available ~ post_cutoff + controls`)**: coefficient on `post_cutoff` = **-0.04450** (SE 0.00038).
- **ML first stage (`latent_adoption_propensity_proxy ~ post_cutoff + controls`)**: coefficient on `post_cutoff` = **0.00003** (SE 0.00011, p = 0.8109; 95% CI [-0.00019, 0.00024]).
- **Baseline second stage (`log_price ~ available + controls`)**: coefficient on `available` = **0.10145** (SE 0.00034).
- **ML second stage (`log_price ~ latent_adoption_propensity_proxy + controls`)**: coefficient on latent proxy = **1.43645** (SE 0.00114).

Read jointly, these extension estimates do not overturn the baseline cutoff-null result: the latent proxy itself is not discontinuously shifted at policy timing, while cross-sectional latent-propensity differences are strongly associated with price levels.

### 6.3 Interpretation caveats
1. The latent propensity index is unsupervised and is **not** verified Smart Pricing adoption telemetry.
2. The extension second stage is associational and does not by itself identify a causal treatment effect.
3. Magnitudes are specification-dependent (feature set, clustering choice, normalization).
4. The weak latent-proxy first-stage shift is mechanically consistent with a largely time-invariant listing-level construction.
5. Policy inference should therefore treat this ML layer as heterogeneity mapping, not as a replacement for validated treatment measurement or full IV identification.

## 7. Limitations
1. **Proxy treatment channel:** `available` is not direct Smart Pricing adoption telemetry.
2. **Identification assumptions:** exclusion and monotonicity are stronger than in a design with observed adoption status.
3. **Inference sensitivity:** clustered versus homoskedastic uncertainty can differ in very large panels.
4. **Mechanism scope:** level effects alone are not sufficient to evaluate coordination-risk channels such as dispersion compression or dynamic co-movement.
5. **City-level precision heterogeneity:** some city-window first stages are weak, limiting local interpretation.

## 8. Conclusion
Under the current multicity fuzzy-RDD/IV proxy implementation, we do **not** detect a robust large immediate discontinuity in Airbnb listing price levels around Smart Pricing rollout timing. The pooled estimates are consistently near zero, and core diagnostics (bandwidth and placebo) do not overturn that result.

The ML extension adds a descriptive heterogeneity layer: a latent adoption propensity proxy is strongly associated with pooled price levels but shows no discontinuous first-stage jump at rollout timing. This reinforces (rather than reverses) the baseline null framing on immediate cutoff-level effects.

This should be read as a baseline screening conclusion rather than a final statement on algorithmic coordination risk. The most policy-relevant next steps are richer mechanism-focused designs (dynamic co-movement, dispersion structure, and host-segment heterogeneity) built on stronger treatment observability.

## Reproducibility Pointers
- Baseline script: `scripts/day4_multicity_fuzzy_rdd.py`
- Baseline outputs: `data/processed/day4/`
- Week 2 diagnostics: `data/processed/week2/`
- Interpretation notes: `docs/DAY4_interpretation_notes.md`, `docs/week2_*.md`
