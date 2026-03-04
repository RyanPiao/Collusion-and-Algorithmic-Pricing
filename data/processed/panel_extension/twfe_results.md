# Panel Extension TWFE Results

- Estimator: **linearmodels.PanelOLS**
- Fallback used: **False**
- N obs: **247,664**
- Listings (entities): **1,346**
- Dates: **184**
- Controls: available, minimum_nights, maximum_nights, post_cutoff, price_volatility_7d, price_volatility_14d
- Sampling: kept all treated listings and 1/100 of non-adopters (247,664/24,228,568 rows).
- `dynamic_algo_adopted`: coef = -0.111849, SE = 0.039048, p = 0.004178, 95% CI [-0.188382, -0.035316]

Specification: `log_price ~ dynamic_algo_adopted + controls + listing FE + date FE`, clustered SE by listing.
