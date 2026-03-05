# Spillover TWFE Results

- Estimator: **linearmodels.PanelOLS**
- N obs: **4,236,416**
- Listings (entities): **18,721**
- Dates: **184**
- Controls: available, minimum_nights, maximum_nights, price_volatility_7d, price_volatility_14d
- Local spillover radius: **1.0 km** (BallTree(haversine))
- Sampling: kept all treated listings and 1/100 of non-adopters (4,236,416/27,452,432 rows).

- `dynamic_algo_adopted`: coef = 0.000077, SE = 0.000052, p = 0.1421, 95% CI [-0.000026, 0.000179]
- `algo_penetration_1km`: coef = -0.000005, SE = 0.000032, p = 0.8775, 95% CI [-0.000068, 0.000058]
- `dynamic_x_penetration`: coef = -0.000063, SE = 0.000076, p = 0.4063, 95% CI [-0.000211, 0.000086]

Specification: `log_price ~ dynamic_algo_adopted + algo_penetration_1km + dynamic_x_penetration + controls + listing FE + date FE`, clustered by listing.
