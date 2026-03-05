# Step 2 Execution Status

## Scope completed
- Built multi-city daily listing-level panel (primary) with 8-city coverage.
- Built secondary city-month aggregates for robustness and communication.
- Exported ±1m/±2m/±3m windowed datasets around cutoff.
- Exported QA outputs: coverage, missingness, support near cutoff, first-stage prep, and consistency checks.

## City scope used
Boston, New York City, Los Angeles, San Francisco, Austin, Chicago, Seattle, Washington DC

> Requested Miami coverage was unavailable in the current InsideAirbnb endpoint; Washington DC was used as the approved alternate replacement (see `data/processed/step2/city_selection_audit.csv`).

## Cutoff mapping
- Primary pooled fallback cutoff: **2025-09-01**
- Secondary pooled sensitivity cutoff: **2025-10-01**
- Reason for pooled fallback: city-specific Smart Pricing rollout dates were not credibly observed in source files; fallback kept auditable in `city_cutoff_map.csv`.

## Panel size summary
- Daily panel rows: **24,228,568**
- Daily panel unique city-listings: **131,677**
- Monthly aggregate rows: **56**

### Unique listings by city

| city_slug | unique_listings |
|---|---:|
| los-angeles | 45,008 |
| new-york-city | 37,428 |
| austin | 15,418 |
| chicago | 8,741 |
| san-francisco | 7,817 |
| seattle | 6,762 |
| washington-dc | 6,077 |
| boston | 4,426 |

## QA check highlights
- Duplicate keys (`city_slug`,`listing_id`,`date`): **0**
- Window nesting violations: **0**
- Cutoff alignment violations (`days_from_cutoff == 0` but not post): **0**
- Max daily-vs-monthly mean price diff: **0.0000000000**
- Max daily-vs-monthly listing-step count diff: **0**

## Economist rationale (data-frequency choice)
- **Daily panel (primary)** preserves within-city, within-window timing variation needed for local identification around the cutoff.
- **Monthly aggregates (secondary)** provide lower-noise trend robustness, communication clarity, and a useful check against daily compositional volatility.

## Output locations
- `data/processed/step2/fact_listing_day_multicity.csv.gz`
- `data/processed/step2/fact_listing_day_multicity_bw_1m.csv.gz`
- `data/processed/step2/fact_listing_day_multicity_bw_2m.csv.gz`
- `data/processed/step2/fact_listing_day_multicity_bw_3m.csv.gz`
- `data/processed/step2/agg_city_month_multicity.csv`
- `data/processed/step2/city_cutoff_map.csv`
- `data/processed/step2/city_selection_audit.csv`
- `data/processed/step2/qa/*`