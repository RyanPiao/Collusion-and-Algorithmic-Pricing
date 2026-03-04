# Day 2 Multi-City Data Design (Implementation Brief)

## Goal
Build a multi-city analysis panel for fuzzy RDD around Airbnb Smart Pricing rollout timing, with:
- **Primary dataset:** daily listing-level panel
- **Secondary dataset:** monthly aggregates (robustness + communication)
- **Event windows:** **±1, ±2, ±3 months** around each city’s cutoff
- **Cutoff logic:** city-specific rollout dates when available, otherwise global fallback

---

## 1) City scope (8 major markets)

Use these cities in Day 2:
1. New York City (`new-york-city`)
2. Los Angeles (`los-angeles`)
3. San Francisco (`san-francisco`)
4. Chicago (`chicago`)
5. Boston (`boston`)
6. Washington, DC (`washington-dc`)
7. Seattle (`seattle`)
8. Austin (`austin`)

> Keep both a display name and canonical slug so joins are stable across files.

---

## 2) Data products

## A. Primary: `fact_listing_day_multicity`
- **Unit:** `city × listing_id × date`
- **Use:** baseline estimation and all windowed fuzzy-RDD specs
- **Storage target:** parquet/csv in processed data folder

### Required fields

**Keys / indexing**
- `city_slug`
- `city_name`
- `listing_id`
- `date` (listing-day)

**Outcome + treatment proxy inputs**
- `price_usd` (clean numeric)
- `log_price`
- `available` (0/1)
- `minimum_nights`
- `maximum_nights`

**Running variable + assignment**
- `cutoff_date_city` (final assigned cutoff for that city)
- `cutoff_source` (`city_specific` | `global_fallback`)
- `days_from_cutoff` (= `date - cutoff_date_city`)
- `post_cutoff` (1 if `date >= cutoff_date_city`)

**Window flags (month-based)**
- `in_bw_1m` (within ±1 month)
- `in_bw_2m` (within ±2 months)
- `in_bw_3m` (within ±3 months)

**Core controls (already used in Day 1, harmonized cross-city)**
- Listing: room/property type, accommodates, bedrooms/bathrooms (if available)
- Host: superhost, response/acceptance indicators, host tenure proxy
- Time/location: day-of-week, month FE, neighborhood identifier (or city-neighborhood FE)

## B. Secondary: `agg_city_month_multicity`
- **Unit:** `city × year_month`
- **Use:** robustness summaries, communication visuals, and sanity checks

### Required fields
- `city_slug`, `year_month`
- `n_listings_active`
- `n_listing_days`
- `mean_price_usd`, `median_price_usd`
- `mean_log_price`
- `availability_rate`
- `share_post_cutoff`
- `mean_days_from_cutoff`

---

## 3) Cutoff mapping strategy

Create and version a lookup table: **`city_cutoff_map`** with:
- `city_slug`
- `cutoff_date_primary`
- `cutoff_date_secondary` (optional)
- `source_type` (`city_specific` / `global_fallback`)
- `source_note` (URL/doc reference)
- `confidence` (`high` / `medium` / `low`)
- `last_verified_at`

### Assignment hierarchy
1. **If credible city-specific rollout date exists**, use it as `cutoff_date_primary`.
2. Else use global default primary cutoff: **2023-03-25**.
3. Keep **2023-05-25** as secondary/common sensitivity cutoff when city-specific public rollout is unavailable.
4. Persist source metadata so date choices are auditable.

### Implementation notes
- Join `city_cutoff_map` into daily panel before creating running-variable fields.
- Recompute `days_from_cutoff`, `post_cutoff`, and `in_bw_*` after any cutoff update.
- Keep a switch to run models on:
  - city-specific primary cutoffs (preferred), and
  - global-only cutoffs (robustness).

---

## 4) Event-window definitions (required)

Use **calendar-month arithmetic** (not fixed 30-day approximation):
- `in_bw_1m = 1` if `date ∈ [cutoff_date_city - 1 month, cutoff_date_city + 1 month]`
- `in_bw_2m = 1` if `date ∈ [cutoff_date_city - 2 months, cutoff_date_city + 2 months]`
- `in_bw_3m = 1` if `date ∈ [cutoff_date_city - 3 months, cutoff_date_city + 3 months]`

`in_bw_1m ⊂ in_bw_2m ⊂ in_bw_3m` must always hold.

---

## 5) QA checks (must-pass before estimation)

1. **Coverage by city/window**
   - Each city has observations on both sides of cutoff for ±1m, ±2m, ±3m.
   - Flag city-window cells with thin support.

2. **Key uniqueness**
   - No duplicates on (`city_slug`, `listing_id`, `date`).

3. **Price cleaning integrity**
   - Currency symbols/commas removed correctly.
   - Nonpositive or implausible outliers handled/documented.

4. **Cutoff-map validity**
   - Every city has exactly one assigned `cutoff_date_city` in primary run.
   - `cutoff_source` populated and auditable.

5. **Window logic consistency**
   - Nested window rule holds (`1m` inside `2m`, `2m` inside `3m`).
   - `days_from_cutoff == 0` aligns with `date == cutoff_date_city`.

6. **Missingness thresholds**
   - Report missing rates for key model fields (price, availability, FE identifiers, controls).
   - Pre-specify drop/imputation rules and apply consistently across cities.

7. **Daily vs monthly reconciliation**
   - Monthly aggregates reproduce daily means/counts within tolerance.

8. **Pre-model balance diagnostics (quick pass)**
   - Covariate means around cutoff by city and window.
   - Check no obvious discontinuity in observation mass at cutoff.

---

## Day 2 deliverable outputs
- `fact_listing_day_multicity` (primary analysis panel)
- `agg_city_month_multicity` (robustness/communication panel)
- `city_cutoff_map` (with source metadata and confidence tags)
- QA summary table/log documenting pass/fail and flagged issues by city
