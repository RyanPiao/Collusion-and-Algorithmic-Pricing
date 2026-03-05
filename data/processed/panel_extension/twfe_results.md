# Panel Extension TWFE Results

## Levels model
- `log_price`: coef = 0.000045, SE = 0.000028, p = 0.1054, 95% CI [-0.000009, 0.000099]

## Volatility models
- `abs_price_change`: coef = -0.003031, SE = 0.009455, p = 0.7485, 95% CI [-0.021562, 0.015500]
- `rolling_7d_variance`: coef = 0.004697, SE = 0.000120, p = 0, 95% CI [0.004461, 0.004933]

## Sampling and specification
- Sampling: kept all treated listings and 1/100 of non-adopters (4,236,416/27,452,432 rows).
- FE: listing and date fixed effects.
- SE: clustered by listing when feasible.
