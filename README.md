# Algorithmic Pricing, Market Conduct, and Antitrust Risk in Short-Term Rentals: A Fuzzy RDD Study of Airbnb Smart Pricing

## Abstract
This project studies whether platform-assisted algorithmic pricing changes seller behavior in ways that are relevant for competition policy. Using listing-level Airbnb data, the empirical design evaluates how pricing dynamics shift around the rollout of Airbnb Smart Pricing and whether those shifts are consistent with greater price coordination risk or with benign efficiency improvements.

The analysis is built around a fuzzy regression discontinuity design (RDD) that exploits policy timing as a quasi-experimental cutoff while allowing incomplete compliance in tool adoption. Rather than assuming deterministic treatment at the cutoff, the design models treatment intensity through listing-night eligibility (`available == 1`) and host-level uptake propensity. Under the identifying assumptions of the design, this approach is intended to recover a local causal effect of increased Smart Pricing exposure on nightly prices while controlling for listing attributes, host characteristics, and time patterns. Current estimates should be treated as provisional until full placebo and continuity stress tests are completed.

## Motivation
Algorithmic pricing systems can reduce search and adjustment costs, but they can also create synchronized responses to market signals across many sellers. That dual effect is central to modern antitrust debates in digital platforms. Airbnb offers a useful setting because Smart Pricing was introduced with clear timing and heterogeneous host adoption, enabling an empirical test of whether observed price changes are consistent with unilateral optimization or with behavior that could elevate coordination concerns.

## Research Question and Hypothesis
### Research Question
How does increased exposure to Airbnb Smart Pricing affect host nightly pricing behavior around the policy introduction window?

### Hypothesis
- **H1 (primary):** Listings with higher post-cutoff Smart Pricing uptake propensity experience a statistically significant shift in nightly prices relative to comparable listings near the cutoff.
- **H2 (competition-policy interpretation):** If the estimated effect indicates tighter price movement and reduced independent variation among exposed listings, this pattern is more consistent with elevated coordination risk than with purely idiosyncratic host-level pricing.

## Identification Strategy (Summary)
The core design uses **announcement-date policy timing** with a **fuzzy RDD** implementation.

1. **Cutoff and running variable**
   - Main cutoff anchored to Smart Pricing rollout timing (early-access marker: `2023-03-25`; public availability marker tracked in diagnostics: `2023-05-25`).
   - Running variable: days relative to the cutoff (`days_from_cutoff` / date index).

2. **Eligibility and fuzzy assignment**
   - Eligibility proxy at listing-night level: `available == 1`.
   - Post-cutoff assignment indicator captures policy exposure timing.
   - Treatment is fuzzy (not all eligible hosts adopt immediately; some non-eligible observations have low exposure), so first-stage uptake is modeled rather than imposed.

3. **First stage (uptake propensity)**
   - Smart Pricing uptake propensity is estimated from host-side observables (e.g., tenure, acceptance/response patterns, superhost status, listing portfolio, booking settings) and selected controls.
   - This generates a continuous adoption-intensity measure used in the second stage.

4. **Second stage (local price effect)**
   - Nightly price is regressed on predicted Smart Pricing exposure within bandwidth around the cutoff.
   - Robust inference and specification diagnostics are included (bandwidth sensitivity, multicollinearity checks, heteroskedasticity test, RESET, residual normality checks).

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
│   └── day3_multicity_eda.py
├── day3_multicity_eda.ipynb
├── data/
│   ├── processed/day2/ ... (generated Day 2 outputs)
│   └── processed/day3/ ... (generated Day 3 EDA outputs)
└── docs/
    ├── DAY1_IDENTIFICATION.md
    ├── DAY1_ROADMAP.md
    ├── DAY1_STATUS.md
    ├── DAY1_identification_design.md
    ├── DAY1_next_steps.md
    ├── DAY1_problem_framing.md
    ├── DAY2_multicity_data_design.md
    └── DAY2_STATUS.md
```

## Notebook Map
- **`Step_1_Data_Cleaning.ipynb`**  
  Ingestion and standardization of InsideAirbnb source files, city-level cleaning routines, and merged panel exports used downstream.

- **`calendar_data_cleaning.ipynb`**  
  Calendar-focused preprocessing pipeline, including weekly filtering/merging and preparation of analysis-ready calendar-listing joins.

- **`Step_2_Data_Clustering.ipynb`**  
  Intermediate aggregation/clustering workflow used to organize listings and construct grouped analysis inputs.

- **`fuzzy_rdd_boston.ipynb`**  
  Main empirical notebook: merge, feature engineering, cutoff construction, fuzzy RDD estimation, and robustness/diagnostic tests.

- **`day3_multicity_eda.ipynb`**  
  Day 3 descriptive notebook for multicity cutoff-window EDA: city trends, pre/post distribution shifts, support diagnostics, and missingness/data-quality views.

## Day 1 Status
Completed foundation work includes:
- Data ingestion and cleaning notebooks consolidated in the repository.
- Listing-calendar merge and core variable preparation implemented.
- Policy timing markers and fuzzy RDD skeleton operationalized in the Boston analysis notebook.
- First-pass diagnostics and robustness scaffolding added (bandwidth sensitivity and specification tests).

Detailed notes:
- [`docs/DAY1_STATUS.md`](docs/DAY1_STATUS.md)
- [`docs/DAY1_IDENTIFICATION.md`](docs/DAY1_IDENTIFICATION.md)
- [`docs/DAY1_problem_framing.md`](docs/DAY1_problem_framing.md)
- [`docs/DAY1_identification_design.md`](docs/DAY1_identification_design.md)

## Day 2 Multi-City Status
- Day 2 pipeline executed via `scripts/day2_build_multicity_panels.py`.
- Produced the primary daily panel (`fact_listing_day_multicity`) and secondary monthly panel (`agg_city_month_multicity`).
- Exported ±1m/±2m/±3m windowed extracts and QA outputs under `data/processed/day2/qa/`.
- Requested Miami market was unavailable in current source endpoints; Washington DC was used as the approved alternate (logged in `city_selection_audit.csv`).
- City-specific Smart Pricing rollout dates were not credibly observed from source files; pipeline falls back to auditable pooled cutoffs in `city_cutoff_map.csv`.
- Full execution summary and QA highlights: [`docs/DAY2_STATUS.md`](docs/DAY2_STATUS.md).

## Planned Day 2+ Roadmap
1. Finalize treatment coding decisions and document any alternative cutoff/event-window definitions.
2. Tighten first-stage feature set and report first-stage strength/fit statistics transparently.
3. Produce baseline and robustness tables (multiple bandwidths, placebo cutoffs, heterogeneity splits).
4. Add inference-ready outputs (tables/figures) and reproducibility notes for full reruns.
5. Extend beyond single-city validation to multi-city comparisons where data consistency is adequate.

Roadmap details:
- [`docs/DAY1_ROADMAP.md`](docs/DAY1_ROADMAP.md)
- [`docs/DAY1_next_steps.md`](docs/DAY1_next_steps.md)
- [`docs/DAY2_multicity_data_design.md`](docs/DAY2_multicity_data_design.md)

## Day 2 Multi-City Build

Day 2 extends the Boston-focused setup into an 8-city multicity panel with explicit cutoff mapping, daily identification windows, and monthly robustness aggregates.

### What was added
- **Primary dataset:** `fact_listing_day_multicity` (daily listing-level panel).
- **Secondary dataset:** `agg_city_month_multicity` (city-month robustness panel).
- **Window extracts:** ±1, ±2, and ±3 month datasets around the assigned city cutoff.
- **QA outputs:** city-date coverage, missingness reports, support near cutoff, first-stage prep tables, and consistency checks.

### Cutoff policy for Day 2
- Pipeline supports city-specific cutoff overrides when credible dates are available.
- Where city-specific rollout dates are not available, Day 2 uses a pooled fallback cutoff and records this decision in `city_cutoff_map` with source metadata.

### Data-frequency rationale
- **Daily (primary):** needed for local identification around cutoff timing in fuzzy-RDD style designs.
- **Monthly (secondary):** used as a robustness and communication layer to validate that directional patterns are not artifacts of high-frequency noise.

## Day 3 Multi-City EDA Around Cutoff

Day 3 adds policy-oriented descriptive outputs on top of Day 2 multicity panels.

### Day 3 outputs
- **EDA script:** `scripts/day3_multicity_eda.py`
- **EDA notebook:** `day3_multicity_eda.ipynb`
- **Trend aggregates:** `data/processed/day3/city_day_trends_cutoff.csv`
- **Distribution shifts:**
  - `data/processed/day3/prepost_distribution_stats_city_window.csv`
  - `data/processed/day3/prepost_distribution_shift_summary.csv`
- **Treatment-support diagnostics:** `data/processed/day3/treatment_support_diagnostics_city_window.csv`
- **Missingness/data quality:**
  - `data/processed/day3/missingness_summary_city.csv`
  - `data/processed/day3/data_quality_checks_summary.csv`
- **Figures:**
  - `docs/figures/day3/city_trend_mean_price_cutoff.png`
  - `docs/figures/day3/prepost_logprice_distribution_bw1m.png`
  - `docs/figures/day3/treatment_support_diagnostics.png`
- **Interpretation note:** `docs/DAY3_eda_note.md`
