# Step 6 Technical Note: Robustness Upgrade and Credibility Hardening

## Objective
Move from Stage-1 baseline synthesis to a **credibility-hardened empirical package** by implementing the highest-value technical upgrades:
1. cluster-robust inference,
2. continuity/pre-trend diagnostics,
3. dispersion and co-movement metrics (beyond level effects).

---

## 1) Why Step 6 matters
Step 5 established that baseline local level effects are near zero under the current proxy design. Step 6 is about ensuring that conclusion is not an artifact of inference choices or insufficient diagnostics.

This step focuses on improving **identification credibility**, not adding narrative.

---

## 2) Priority technical modules

### Module A — Cluster-robust inference (highest priority)
**Problem:** Current SEs may understate uncertainty if errors are correlated within hosts/neighborhoods.

**Actions:**
- Re-estimate Step 4 pooled and city-window models with:
  - host-level clustered SE,
  - neighborhood-level clustered SE,
  - (optional) two-way cluster if computationally feasible.
- Export side-by-side table: conventional vs clustered SE.

**Success criterion:** Main sign/magnitude conclusions remain stable; confidence intervals reported with clustering.

---

### Module B — Continuity and manipulation diagnostics
**Problem:** RDD credibility requires no strategic sorting/manipulation around cutoff and smooth covariates.

**Actions:**
- McCrary-style density test around cutoff (or practical histogram/bin test if full test unavailable).
- Covariate continuity checks at cutoff for key controls:
  - listing capacity/room-type proxies,
  - host tenure and superhost status,
  - calendar seasonality controls.
- Document pass/fail with caveats.

**Success criterion:** No major discontinuities in pre-determined covariates; density irregularity not severe enough to invalidate design.

---

### Module C — Dispersion and co-movement diagnostics (antitrust-relevant)
**Problem:** Collusion/coordinated pricing risk can emerge through reduced dispersion and synchronized movement, not level shifts.

**Actions:**
- Compute pre/post price-dispersion metrics by city-window:
  - standard deviation of log price,
  - interquartile range (IQR),
  - coefficient of variation.
- Compute co-movement proxy:
  - within-neighborhood pairwise correlation in daily log-price changes.
- Run reduced-form discontinuity checks on these metrics.

**Success criterion:** Clear evidence on whether algorithmic exposure aligns with tighter dispersion or stronger co-movement.

---

## 3) Deliverables (Step 6)

### Tables (target paths)
- `data/processed/step6/inference_clustered_comparison.csv`
- `data/processed/step6/covariate_continuity_tests.csv`
- `data/processed/step6/dispersion_metrics_pre_post.csv`
- `data/processed/step6/comovement_metrics_pre_post.csv`

### Plots
- `data/processed/step6/cutoff_density_check.png`
- `data/processed/step6/dispersion_by_window.png`
- `data/processed/step6/comovement_by_window.png`

### Documentation
- `docs/DAY6_technical_note.md` (this file)
- `docs/DAY6_STATUS.md`

---

## 4) Interpretation guardrails for Step 6 output
1. If clustered SEs widen intervals materially, report this prominently.
2. If continuity checks fail in specific cities/windows, downgrade causal confidence for those cells.
3. If dispersion/co-movement shifts are detected without level shifts, treat as **behavioral-structure signal**, not direct proof of collusion.
4. Avoid legal conclusions; keep language empirical and mechanism-oriented.

---

## 5) Step 7 handoff preview
If Step 6 modules run cleanly, Step 7 should produce a **policy-ready technical brief** with:
- confidence grading by diagnostic pillar,
- clear “what we can say / cannot say,”
- Stage-2 implementation backlog ordered by impact and effort.

---

## Summary
Step 6 is the bridge between baseline findings and decision-grade evidence. The key goal is to test whether the Stage-1 near-zero level effect conclusion survives stronger inference and to directly examine antitrust-relevant pricing structure metrics.