# Algorithmic Pricing, Market Conduct, and Antitrust Risk in U.S. Major Cities: Evidence from Airbnb Smart Pricing

**1. Abstract**
This project investigates whether the rollout of Airbnb's Smart Pricing algorithm caused a discontinuous jump in local market prices or fostered tacit collusion. Using a multicity panel and a combination of quasi-experimental and machine learning methods, we find no evidence of an average market-wide price level increase or localized spatial collusion. Instead, the algorithm acts as a productivity multiplier for sophisticated hosts, significantly increasing forward-looking price volatility and complex price discrimination without raising the baseline rent.

**2. Why the Question Matters**
Algorithmic pricing tools are increasingly dominant in digital marketplaces, sparking intense antitrust and competition-policy debates. Regulators are concerned that shared pricing algorithms might facilitate tacit collusion (price-fixing without explicit communication). Understanding exactly *how* these algorithms change host behavior—whether they universally hike prices or simply optimize volatility—is critical for modern antitrust enforcement and platform regulation.

**3. Data Used**
* **Type:** Public Real. 
* **Provenance:** Inside Airbnb multicity daily listing panel (Austin, Boston, Chicago, Los Angeles, New York City, San Francisco, Seattle, and Washington, DC).
* **Scope:** 24.2+ million listing-day observations spanning symmetric ±1 to ±3 month bandwidths around city-specific policy rollouts. 

**4. Method in Plain English**
We employed a three-tiered empirical pipeline to isolate the algorithm's true effect from cross-sectional noise:
* **Baseline (Fuzzy RDD / IV):** Tested for a sudden, market-wide price jump exactly when the algorithm was introduced, using policy rollout timing as an instrument. 
* **Heterogeneity (Unsupervised ML):** Built a "latent adoption propensity proxy" using KMeans/GMM clustering on *strictly pre-rollout* dynamic pricing behavior to identify the types of hosts most likely to use the algorithm. 
* **Longitudinal Causal (TWFE & Structural Breaks):** Used `ruptures` to detect behavioral volatility breaks, then ran Two-Way Fixed Effects and propensity-stratified Difference-in-Differences models to isolate the algorithm's causal footprint while stripping out omitted variable bias. 
* **Estimand & Assumptions:** We estimate the Local Average Treatment Effect (LATE) of algorithmic adoption on price levels and variance. This assumes parallel pre-trends (for DiD) and that the rollout cutoff only affects prices through the algorithmic adoption channel (exclusion restriction).

**5. Key Findings**
* **The "Big Null" on Average Prices:** The rollout of the algorithm did *not* cause a sudden, market-wide jump in average price levels (Fuzzy RDD coefficient: 0.0027, p = 0.85). 
* **Omitted Variable Bias is Deceptive:** While ML shows algorithm adopters charge massively higher prices generally, our fixed-effects models prove this is purely associational; expensive, professionalized hosts are simply more likely to adopt algorithms.
* **The True Footprint is Volatility:** Adopting the algorithm significantly increases rolling 7-day price variance (p < 0.001) but leaves average price levels unchanged (p = 0.10). 
* **No Price Umbrellas:** There is zero statistically significant evidence of localized spatial collusion; a neighborhood heavily saturated with algorithmic pricing does not allow non-adopting neighbors to artificially hike their own rates.

**6. Robustness Summary**
The baseline null effect holds across multiple symmetric bandwidths (±1, ±2, and ±3 months) and passes all placebo cutoff tests (-30d, +30d). The structural break panel extension successfully detrends pre-existing pricing trajectories and achieves a stable 13.35% treated sample retention rate, confirming that sparsity did not drive the final volatility findings. 

**7. Limitations (What We Cannot Claim)**
* **Temporal Resolution:** Because Inside Airbnb data relies on periodic calendar scrapes, our "daily panel" measures the variance of *forward-looking scheduled prices*, not real-time daily adjustments. We can prove the algorithm creates complex scheduled price discrimination, but not that it reacts dynamically between scrape intervals. 
* **Proxy Telemetry:** We observe algorithmic *behavioral footprints* (availability and volatility breaks) rather than direct internal Airbnb telemetry of exactly who clicked "turn on Smart Pricing."

**8. Practical Implications**
These findings support the "Cognitive Constraint" hypothesis: pricing algorithms act as productivity multipliers for already-sophisticated hosts rather than magic wands that arbitrarily hike market rents. For antitrust regulators, this means surveillance shouldn't rely on aggregate market-wide price averages. Algorithm-linked responses generate structurally concentrated, highly dynamic price discrimination among top-tier hosts, requiring segment-aware regulatory frameworks.

**9. Reproducibility Steps**
1. Activate virtual environment: `source .venv/bin/activate`
2. Run baseline Fuzzy RDD: `python scripts/day4_multicity_fuzzy_rdd.py`
3. Run ML proxy generation: `python scripts/ml_unsupervised_extension.py --repo-root .`
4. Run longitudinal panel extensions: `python scripts/panel_extension_run_all.py`
*Outputs are mapped to `data/processed/day4/`, `ml_extension/`, and `panel_extension/`.*

**10. Evidence Links & Next Steps**
* **Repo:** `tech-econ-airbnb-algorithmic-pricing`
* **Citations:** Callaway & Sant'Anna (2021) for staggered DiD framework; Bai & Perron (1998) for structural break logic. 
* **Next-Week Plan:** Switch to Synthetic Data (Week B). Build a theoretical matching market simulation to test how platform fee structures interact with algorithmic price ceilings.

### 🎓 Teaching Note (For ECON 5200 / ECON 3916 / ECON 1116)

**Module:** Omitted Variable Bias & Unsupervised Proxies in Applied Data
**Discussion Prompt:** In this project, a naive Machine Learning regression found a massive positive correlation between algorithmic pricing and rent levels. However, a Two-Way Fixed Effects (TWFE) panel model proved the true causal effect on price *levels* was zero. 

1. **For ECON 1116 (Principles of Microeconomics):** How does the "Cognitive Constraint" hypothesis explain why only sophisticated hosts see behavioral changes from the algorithm? Discuss the microeconomic mechanism of price discrimination observed here (increased volatility vs. flat levels).
2. **For ECON 5200 (Applied Data Analytics in Econ) / ECON 3916 (Statistical & Machine Learning):** How did the use of KMeans/GMM clustering on *strictly pre-cutoff* data prevent data leakage while solving the missing telemetry problem? 
3. **Data Literacy:** The dataset is built on periodic scrapes of a forward-looking calendar. Why does this structural limitation fundamentally change how we must interpret the "daily price variance" coefficient?
