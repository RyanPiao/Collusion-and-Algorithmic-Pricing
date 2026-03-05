# Step 4 Interpretation Notes: Multicity Fuzzy RDD Baseline

## What was estimated (baseline)
Step 4 implements a multicity fuzzy-RDD-style IV baseline over Step 2 windows (±1m/±2m/±3m) using the daily listing panel.

- **Outcome:** `log_price`
- **Endogenous treatment proxy:** `available` (listing-step availability)
- **Excluded instrument:** `post_cutoff`
- **RDD controls:** local linear running-variable terms (`days_from_cutoff`, `post_cutoff × days_from_cutoff`) plus listing/host controls
- **Estimands reported:** pooled and city-level local effects by bandwidth window

This should be interpreted as a **baseline reduced-form/IV proxy design**, not a final structural estimate of Smart Pricing adoption effects.

## Core Step 4 readout

### 1) First stage (relevance)
Pooled first-stage strength is strong across windows:

- ±1m: F = **4158.50**, post coefficient = **-0.0419**
- ±2m: F = **9715.53**, post coefficient = **-0.0452**
- ±3m: F = **13905.54**, post coefficient = **-0.0445**

So, in pooled samples, the cutoff indicator is highly predictive of the treatment proxy.

At the city-window level, first-stage strength is heterogeneous; a small subset of cells is weak (F < 10), so those city-specific second-stage estimates should be treated as low-information.

### 2) Second stage (local effect on log price)
Pooled second-stage estimates are near zero and imprecise across all windows:

- ±1m: 0.0034 (SE 0.0265), 95% CI [-0.0486, 0.0554]
- ±2m: 0.0033 (SE 0.0174), 95% CI [-0.0307, 0.0374]
- ±3m: 0.0027 (SE 0.0145), 95% CI [-0.0256, 0.0311]

City-level second-stage estimates are also generally imprecise in this baseline setup (no city-window estimate is conventionally significant).

### 3) Diagnostics
- **Bandwidth sensitivity:** pooled estimates are directionally stable and remain close to zero from ±1m to ±3m.
- **Placebo cutoffs (±30 days in ±3m sample):** placebo specifications do not produce stable statistically significant effects; one placebo has a weak first stage, reinforcing caution in over-interpreting placebo magnitudes.

## Causal-language guardrails
These Step 4 estimates support a cautious statement:

> Under this baseline proxy implementation, we do not detect a robust local discontinuous shift in listing price levels attributable to the instrumented treatment proxy around the assigned cutoff.

Important limits:
1. `available` is a proxy channel, not a direct Smart Pricing adoption flag.
2. Exclusion and monotonicity assumptions are stronger here than in a design with observed adoption telemetry.
3. Inference is baseline-grade; stronger robustness layers (cluster-robust SEs, richer fixed effects, pre-trend falsifications, covariate continuity tables) are still recommended.

## Antitrust relevance framing (careful)
Policy-relevant interpretation should remain restrained:

- These baseline results **do not provide evidence of a large immediate level shift** in prices at the cutoff through this proxy channel.
- That is **not equivalent** to evidence of no algorithmic coordination risk.
- Antitrust-relevant concerns often operate through **dispersion compression, synchronized adjustment, or dynamic co-movement**, which can be weakly reflected in level effects alone.

So, the Step 4 output is best viewed as a screening baseline: useful for ruling out very large local level effects, but insufficient by itself to adjudicate collusion-risk mechanisms.

## Files produced
- `data/processed/step4/first_stage_strength_city_window.csv`
- `data/processed/step4/second_stage_pooled_window_estimates.csv`
- `data/processed/step4/second_stage_city_window_estimates.csv`
- `data/processed/step4/diagnostic_bandwidth_sensitivity_pooled.csv`
- `data/processed/step4/diagnostic_placebo_cutoff_pooled_bw3m.csv`
- `data/processed/step4/day4_run_summary.json`
