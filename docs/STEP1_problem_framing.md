# Step 1 Problem Framing

## Updated research title (recommended)
**Who Sets the Price? Estimating the Causal Impact of Airbnb Smart Pricing from a Feature-Announcement Natural Experiment**

## Polished abstract
Airbnb’s Smart Pricing tool may change how hosts set nightly prices, but estimating its causal effect is difficult because adoption is endogenous: hosts who expect stronger demand are also more likely to use pricing automation. This project studies whether Smart Pricing increases or decreases listing-level prices by exploiting a plausibly exogenous timing shock around the platform’s feature announcement/rollout date. Using high-frequency listing-calendar panel data (Boston, 2022–2023) built from InsideAirbnb snapshots, we implement a **fuzzy regression discontinuity design (RDD)** centered on the announcement date. Eligibility for effective tool use is proxied by listing-step availability (`available == 1`), and host/listing characteristics are used to model the probability of Smart Pricing uptake in the first stage. The second stage estimates the local causal effect of predicted uptake on nightly prices near the cutoff, with neighborhood and calendar fixed effects and robust inference. Preliminary notebook results suggest a positive local price effect (roughly +$40 to +$50 across bandwidths), but this should be treated as provisional pending stronger treatment validation, placebo checks, and continuity diagnostics. The Step-1 contribution is to formalize the estimand, sharpen identification assumptions, and define a replication-first roadmap for credible causal inference.

## Core research question
What is the **local causal effect** of Airbnb Smart Pricing adoption on listing nightly prices around the product announcement/rollout date?

## Why this matters
- **Platform governance:** Algorithmic pricing tools can influence market-level pricing power and potential coordination.
- **Policy relevance:** Results speak to antitrust concerns around algorithmic pricing and platform-mediated price setting.
- **Method value:** A transparent quasi-experimental design is more credible than simple before/after comparisons.

## Unit of analysis and scope
- **Unit:** listing-step
- **Geography:** currently Boston notebook pipeline (extendable to multi-city panel)
- **Period in current workflow:** primarily 2023 around cutoff, with historical preprocessing from 2022 snapshots
- **Outcome:** nightly listing price (recommended primary spec on `log(price)`; level-price retained for interpretability)

## Step-1 takeaways from existing notebooks
- Data engineering pipeline already exists (cleaning, feature engineering, host-level clustering).
- A fuzzy-RDD-style implementation exists in `fuzzy_rdd_boston.ipynb` with:
  - cutoff around **2023-03-25** (early access/announcement marker),
  - first-stage uptake probability model,
  - second-stage price model with time and location controls,
  - robustness over bandwidths.
- Existing outputs indicate a positive coefficient for predicted Smart Pricing uptake, but design still needs stricter treatment validation and placebo/continuity tests.

## Key Step-1 risks to address
1. **Treatment observability gap:** direct Smart Pricing adoption is not explicitly observed; current proxies may mix adoption with listing strategy.
2. **Cutoff validation:** announcement/rollout timing must be externally documented and fixed ex ante.
3. **Eligibility definition:** `available == 1` is plausible for eligibility but may also reflect demand or host behavior; needs careful interpretation.
4. **Functional-form sensitivity:** level-price OLS around cutoff is useful but should be complemented by local-linear RDD and log-price outcomes.

## Step-1 output of this framing
- A tighter causal question,
- A formal identification blueprint (see `DAY1_identification_design.md`),
- A prioritized execution plan for replication and credibility checks (see `DAY1_next_steps.md`).
