# Algorithmic Pricing, Market Conduct, and Antitrust Risk in U.S. Major Cities

## About This Project
This repository studies whether Airbnb Smart Pricing rollout timing is associated with local discontinuities in listing-level prices, and whether any effects are concentrated in latent high-propensity host segments.

The project combines:
- a multicity fuzzy RDD / IV baseline,
- robustness and diagnostic checks,
- an ML-econometrics heterogeneity extension,
- panel/event-study follow-on analyses,
- and a working-paper synthesis.

Primary paper draft:
- `docs/working_paper_us_major_cities.md`

---

## Final Empirical Readout (Current)
Based on the working paper and latest repository outputs:

1. **Average market effect (baseline Step 4):**
   pooled second-stage coefficients are near zero and statistically imprecise across ±1m / ±2m / ±3m windows.
2. **First stage:**
   pooled instrument relevance is strong in baseline proxy specifications.
3. **Diagnostics:**
   bandwidth and placebo checks do not reveal a stable, robust alternative discontinuity.
4. **Heterogeneity extension:**
   refined pre-cutoff latent propensity modeling identifies strong cross-sectional heterogeneity signals, but does not overturn the baseline near-null average discontinuity result.
5. **Interpretation discipline:**
   findings are consistent with heterogeneity in levels/responsiveness rather than a large immediate pooled price-level break at cutoff.

---

## Critical Data Limitation (Explicit)
Inside Airbnb is built from **periodic forward-calendar scrapes** (e.g., monthly/quarterly). Therefore, the constructed “daily panel” reflects scheduled prices visible at scrape time, not continuous real-time step-to-day edits between scrapes.

Accordingly, volatility findings should be interpreted as evidence of **differentiated forward price scheduling** (calendar complexity), not definitive proof of continuous within-interval dynamic repricing.

(See Section 7 in `docs/working_paper_us_major_cities.md`.)

---

## Repository Structure
```text
.
├── README.md
├── docs/
│   ├── working_paper_us_major_cities.md
│   ├── ml_extension_results.md
│   ├── DAY*_STATUS.md, WEEK2_*.md, and interpretation notes
├── scripts/
│   ├── step2_build_multicity_panels.py
│   ├── step3_multicity_eda.py
│   ├── step4_multicity_fuzzy_rdd.py
│   ├── ml_unsupervised_extension.py
│   ├── ml_extension_psm_did.py
│   └── panel_extension_*.py
└── data/processed/
    ├── step2/
    ├── step3/
    ├── step4/
    ├── ml_extension/
    └── panel_extension/
```

---

## Reproducibility (Core)
Create environment and run core pipeline:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn scipy statsmodels linearmodels matplotlib

python scripts/step2_build_multicity_panels.py
python scripts/step3_multicity_eda.py
python scripts/step4_multicity_fuzzy_rdd.py
```

Run refined ML heterogeneity extension:

```bash
python scripts/ml_unsupervised_extension.py --repo-root .
python scripts/ml_extension_psm_did.py --repo-root .
```

---

## Scope and Claims
- This repo is designed for transparent empirical workflow and policy-relevant diagnostics.
- Latent proxy measures are **not** direct Smart Pricing adoption telemetry.
- Results should be interpreted as quasi-experimental evidence under stated assumptions and known data-construction limits.
