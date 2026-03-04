#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/panel_extension"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT / "dynamic_proxy_panel.csv"
BREAKS_PATH = OUT / "listing_breaks.csv"
RESULTS_CSV = OUT / "twfe_results.csv"
RESULTS_MD = OUT / "twfe_results.md"
SUMMARY_JSON = OUT / "twfe_run_summary.json"

CONTROL_SAMPLE_MOD = 100  # keep ~1% of non-adopters by listing_id hash; retain all adopters.


def load_panel() -> tuple[pd.DataFrame, dict]:
    usecols = [
        "listing_id",
        "date",
        "log_price",
        "dynamic_algo_adopted",
        "available",
        "minimum_nights",
        "maximum_nights",
        "post_cutoff",
        "price_volatility_7d",
        "price_volatility_14d",
    ]
    usecols = [c for c in usecols if c in pd.read_csv(PANEL_PATH, nrows=0).columns]

    dtype_map = {
        "listing_id": "int64",
        "log_price": "float32",
        "dynamic_algo_adopted": "float32",
        "available": "float32",
        "minimum_nights": "float32",
        "maximum_nights": "float32",
        "post_cutoff": "float32",
        "price_volatility_7d": "float32",
        "price_volatility_14d": "float32",
    }

    breaks = pd.read_csv(BREAKS_PATH, usecols=["listing_id", "adopted"])
    treated_ids = set(breaks.loc[breaks["adopted"] == 1, "listing_id"].astype("int64").tolist())

    dtype_use = {k: v for k, v in dtype_map.items() if k in usecols}
    kept_chunks: list[pd.DataFrame] = []
    n_rows_in = 0
    n_rows_kept = 0

    for chunk in pd.read_csv(PANEL_PATH, usecols=usecols, dtype=dtype_use, parse_dates=["date"], chunksize=500_000):
        n_rows_in += len(chunk)
        keep = chunk["listing_id"].isin(treated_ids) | ((chunk["listing_id"] % CONTROL_SAMPLE_MOD) == 0)
        kept = chunk.loc[keep].copy()
        if kept.empty:
            continue
        kept_chunks.append(kept)
        n_rows_kept += len(kept)

    if not kept_chunks:
        raise RuntimeError("No rows retained for TWFE estimation after sampling.")

    df = pd.concat(kept_chunks, ignore_index=True)
    df = df.dropna(subset=["listing_id", "date", "log_price", "dynamic_algo_adopted"]).copy()

    for col in [c for c in usecols if c not in {"listing_id", "date"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sample_meta = {
        "control_sample_mod": CONTROL_SAMPLE_MOD,
        "treated_listings_retained_all": True,
        "rows_input_total": int(n_rows_in),
        "rows_kept_after_sampling": int(n_rows_kept),
        "sampling_share_rows": float(n_rows_kept / n_rows_in) if n_rows_in else np.nan,
        "n_treated_listings": int(len(treated_ids)),
    }
    return df, sample_meta


def choose_controls(df: pd.DataFrame) -> list[str]:
    candidates = ["available", "minimum_nights", "maximum_nights", "post_cutoff", "price_volatility_7d", "price_volatility_14d"]
    controls: list[str] = []
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            controls.append(c)
    return controls


def impute_controls(df: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    for c in controls:
        med_listing = df.groupby("listing_id")[c].transform("median")
        df[c] = df[c].fillna(med_listing)
        global_med = df[c].median()
        if pd.isna(global_med):
            global_med = 0.0
        df[c] = df[c].fillna(global_med)
    return df


def fit_linearmodels(df: pd.DataFrame, controls: list[str]) -> tuple[pd.DataFrame, dict]:
    from linearmodels.panel import PanelOLS

    use_cols = ["dynamic_algo_adopted"] + controls
    panel = df.set_index(["listing_id", "date"]).sort_index()
    y = panel["log_price"].astype(float)
    X = panel[use_cols].astype(float)

    model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = model.fit(cov_type="clustered", cluster_entity=True)

    ci = res.conf_int()
    out = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "std_error": res.std_errors.values,
            "t_stat": res.tstats.values,
            "p_value": res.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
        }
    )
    out["nobs"] = float(res.nobs)
    out["n_entities"] = int(panel.index.get_level_values(0).nunique())
    out["n_dates"] = int(panel.index.get_level_values(1).nunique())
    out["rsquared_within"] = float(getattr(res, "rsquared_within", np.nan))
    out["rsquared_overall"] = float(getattr(res, "rsquared_overall", np.nan))
    out["estimator"] = "linearmodels.PanelOLS"
    out["covariance"] = "clustered_by_listing"
    out["fallback_used"] = False

    summary = {
        "estimator": "linearmodels.PanelOLS",
        "fallback_used": False,
        "controls_used": controls,
        "nobs": float(res.nobs),
        "n_entities": int(panel.index.get_level_values(0).nunique()),
        "n_dates": int(panel.index.get_level_values(1).nunique()),
        "rsquared_within": float(getattr(res, "rsquared_within", np.nan)),
        "rsquared_overall": float(getattr(res, "rsquared_overall", np.nan)),
    }
    return out, summary


def fit_statsmodels_fallback(df: pd.DataFrame, controls: list[str]) -> tuple[pd.DataFrame, dict]:
    import statsmodels.formula.api as smf

    rhs = ["dynamic_algo_adopted"] + controls + ["C(listing_id)", "C(date)"]
    formula = "log_price ~ " + " + ".join(rhs)
    model = smf.ols(formula=formula, data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["listing_id"]})

    ci = res.conf_int()
    out = pd.DataFrame(
        {
            "term": res.params.index,
            "coef": res.params.values,
            "std_error": res.bse.values,
            "t_stat": res.tvalues.values,
            "p_value": res.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
        }
    )
    out["nobs"] = float(res.nobs)
    out["n_entities"] = int(df["listing_id"].nunique())
    out["n_dates"] = int(df["date"].nunique())
    out["rsquared_within"] = np.nan
    out["rsquared_overall"] = float(res.rsquared)
    out["estimator"] = "statsmodels.OLS_with_dummies"
    out["covariance"] = "clustered_by_listing"
    out["fallback_used"] = True

    summary = {
        "estimator": "statsmodels.OLS_with_dummies",
        "fallback_used": True,
        "controls_used": controls,
        "nobs": float(res.nobs),
        "n_entities": int(df["listing_id"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "rsquared_within": np.nan,
        "rsquared_overall": float(res.rsquared),
    }
    return out, summary


def write_markdown(results: pd.DataFrame, summary: dict) -> None:
    dyn = results.loc[results["term"] == "dynamic_algo_adopted"]
    if dyn.empty:
        dyn_line = "- `dynamic_algo_adopted` dropped/absorbed in this specification."
    else:
        r = dyn.iloc[0]
        dyn_line = (
            f"- `dynamic_algo_adopted`: coef = {r['coef']:.6f}, SE = {r['std_error']:.6f}, "
            f"p = {r['p_value']:.4g}, 95% CI [{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
        )

    sampling = summary.get("sampling", {})
    md = [
        "# Panel Extension TWFE Results",
        "",
        f"- Estimator: **{summary['estimator']}**",
        f"- Fallback used: **{summary['fallback_used']}**",
        f"- N obs: **{int(summary['nobs']):,}**",
        f"- Listings (entities): **{summary['n_entities']:,}**",
        f"- Dates: **{summary['n_dates']:,}**",
        f"- Controls: {', '.join(summary['controls_used']) if summary['controls_used'] else '(none)'}",
        (
            "- Sampling: kept all treated listings and 1/"
            + str(sampling.get("control_sample_mod", "NA"))
            + " of non-adopters "
            + f"({int(sampling.get('rows_kept_after_sampling', 0)):,}/{int(sampling.get('rows_input_total', 0)):,} rows)."
            if sampling
            else "- Sampling: none"
        ),
        dyn_line,
        "",
        "Specification: `log_price ~ dynamic_algo_adopted + controls + listing FE + date FE`, clustered SE by listing.",
    ]
    RESULTS_MD.write_text("\n".join(md) + "\n")


def main() -> None:
    df, sample_meta = load_panel()
    controls = choose_controls(df)
    df = impute_controls(df, controls)

    try:
        results, summary = fit_linearmodels(df, controls)
    except Exception as e:
        print(f"linearmodels unavailable or failed, using fallback: {e}")
        results, summary = fit_statsmodels_fallback(df, controls)

    summary["sampling"] = sample_meta
    results.to_csv(RESULTS_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    write_markdown(results, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
