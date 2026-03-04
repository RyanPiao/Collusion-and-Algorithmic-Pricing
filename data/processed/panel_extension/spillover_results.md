# Spillover TWFE Results

- Estimator: **linearmodels.PanelOLS**
- N obs: **247,664**
- Listings (entities): **1,346**
- Dates: **184**
- Controls: available, minimum_nights, maximum_nights, post_cutoff, price_volatility_7d, price_volatility_14d
- Sampling: kept all treated listings and 1/100 of non-adopters (247,664/24,228,568 rows).

- `dynamic_algo_adopted`: coef = -0.100228, SE = 0.045549, p = 0.02778, 95% CI [-0.189503, -0.010954]
- `neighborhood_algo_penetration`: coef = 0.092740, SE = 0.106442, p = 0.3836, 95% CI [-0.115884, 0.301364]
- `adoption_x_penetration`: coef = -6.733238, SE = 8.039153, p = 0.4023, 95% CI [-22.489766, 9.023290]

Specification: `log_price ~ own adoption + neighborhood penetration + interaction + controls + listing FE + date FE`, clustered by listing.
