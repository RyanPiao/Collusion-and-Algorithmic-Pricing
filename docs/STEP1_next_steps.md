# Step 1 Next Steps (Execution Plan)

## Objective for the next 3–5 working days
Move from promising prototype estimates to a defensible causal design with transparent assumptions, reproducible code, and clear robustness evidence.

## Priority 0: lock design metadata (today)
- [ ] Confirm and document the exact Smart Pricing feature announcement/rollout timeline used as cutoff(s).
- [ ] Freeze the primary cutoff date and define any secondary cutoff (e.g., early access vs public release).
- [ ] Write a short `design_log.md` with estimand, sample window, and fixed-effects choices.

## Priority 1: treatment measurement upgrade
- [ ] Reconcile treatment proxy across notebooks:
  - host-level `tool_use` from clustering (Step 2),
  - listing-step eligibility via `available == 1`,
  - combined uptake proxy (e.g., `actual_tool_use = tool_use * available`) vs propensity score approach.
- [ ] Produce a one-page “treatment validation note” describing what each proxy captures and misses.
- [ ] Report first-stage strength for each candidate proxy.

## Priority 2: baseline causal model (replication-first)
- [ ] Re-run baseline fuzzy RDD with a clean script/notebook section:
  - running variable centered at cutoff,
  - local window around cutoff,
  - first stage + second stage clearly separated,
  - primary outcome `log(price)` plus level-price secondary spec.
- [ ] Keep control set parsimonious and pre-registered (host + listing + time FE).
- [ ] Cluster SE by listing_id (and evaluate two-way clustering).

## Priority 3: credibility/robustness package
- [ ] Bandwidth sweep with confidence intervals (not coefficients only).
- [ ] Covariate continuity tests at cutoff.
- [ ] Placebo cutoffs (multiple fake dates).
- [ ] Donut RDD (exclude days immediately around cutoff).
- [ ] Pre-trend/event-study visualization around cutoff.

## Priority 4: data QA and reproducibility
- [ ] Verify no duplicate listing-date rows after merge/clean steps.
- [ ] Audit outlier handling (avoid index misalignment when filtering price outliers).
- [ ] Export a compact analysis dataset + data dictionary for the final model.
- [ ] Save all model outputs to machine-readable tables (CSV/JSON) for reporting.

## Draft result language (if current pattern persists)
If effects remain stable after diagnostics, expected conclusion framing:
> Around the Smart Pricing rollout window, listings with higher predicted uptake probability exhibit a local price increase relative to comparable listings near the cutoff.

(Keep this as tentative until placebo and continuity tests pass.)

## Deliverables checklist
- [ ] `docs/DAY1_problem_framing.md` ✅
- [ ] `docs/DAY1_identification_design.md` ✅
- [ ] `docs/DAY1_next_steps.md` ✅
- [ ] Cutoff validation note (next)
- [ ] Reproducible baseline estimation script (next)
- [ ] Robustness appendix tables/figures (next)
