#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/panel_extension"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT / "dynamic_proxy_panel.csv"
BREAKS_PATH = OUT / "listing_breaks.csv"
RESULTS_LEVELS_CSV = OUT / "twfe_results_levels.csv"
RESULTS_VOL_CSV = OUT / "twfe_results_volatility.csv"
RESULTS_MD = OUT / "twfe_results.md"
SUMMARY_JSON = OUT / "twfe_run_summary.json"

CONTROL_SAMPLE_MOD = 100  # keep ~1% of non-adopters by listing_id hash; retain all adopters.
REQUIRED_TWFE_TERMS = ["dynamic_algo_adopted"]

LOGGER = logging.getLogger("panel_extension_2")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(PANEL_PATH, nrows=0)

    usecols = [
        "listing_id",
        "date",
        "log_price",
        "price_usd",
        "abs_price_change",
        "price_volatility_7d",
        "rolling_7d_variance",
        "dynamic_algo_adopted",
        "available",
        "minimum_nights",
        "maximum_nights",
        "post_cutoff",
    ]
    usecols = [c for c in usecols if c in header.columns]

    dtype_map = {
        "listing_id": "int64",
        "log_price": "float32",
        "price_usd": "float32",
        "abs_price_change": "float32",
        "price_volatility_7d": "float32",
        "rolling_7d_variance": "float32",
        "dynamic_algo_adopted": "float32",
        "available": "float32",
        "minimum_nights": "float32",
        "maximum_nights": "float32",
        "post_cutoff": "float32",
    }
    dtype_use = {k: v for k, v in dtype_map.items() if k in usecols}

    breaks = pd.read_csv(BREAKS_PATH, usecols=["listing_id", "adopted"])
    treated_ids = set(breaks.loc[breaks["adopted"] == 1, "listing_id"].astype("int64"))

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
    df = df.dropna(subset=["listing_id", "date", "dynamic_algo_adopted"]).copy()

    # Ensure explicit volatility DVs exist.
    if "abs_price_change" not in df.columns and "price_usd" in df.columns:
        df = df.sort_values(["listing_id", "date"]).copy()
        df["abs_price_change"] = df.groupby("listing_id", observed=True)["price_usd"].diff().abs().astype("float32")
    if "rolling_7d_variance" not in df.columns and "price_volatility_7d" in df.columns:
        df["rolling_7d_variance"] = (pd.to_numeric(df["price_volatility_7d"], errors="coerce") ** 2).astype("float32")

    for c in [c for c in df.columns if c not in {"listing_id", "date"}]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    sample_meta = {
        "control_sample_mod": CONTROL_SAMPLE_MOD,
        "treated_listings_retained_all": True,
        "rows_input_total": int(n_rows_in),
        "rows_kept_after_sampling": int(n_rows_kept),
        "sampling_share_rows": float(n_rows_kept / n_rows_in) if n_rows_in else np.nan,
        "n_treated_listings": int(len(treated_ids)),
    }
    return df, sample_meta


def choose_controls(df: pd.DataFrame, dependent_var: str) -> list[str]:
    # NOTE: post_cutoff is excluded on purpose: with listing/date FE plus dynamic adoption,
    # it is frequently fully absorbed and can also absorb the treatment variation itself.
    candidates = ["available", "minimum_nights", "maximum_nights", "price_volatility_7d"]
    controls: list[str] = []
    for c in candidates:
        if c == dependent_var:
            continue
        if c in df.columns and df[c].notna().any():
            controls.append(c)
    return controls


def impute_controls(df: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    for c in controls:
        med_listing = df.groupby("listing_id", observed=True)[c].transform("median")
        df[c] = df[c].fillna(med_listing)
        global_med = df[c].median()
        if pd.isna(global_med):
            global_med = 0.0
        df[c] = df[c].fillna(global_med)
    return df


def ensure_required_terms(
    out: pd.DataFrame,
    dependent_var: str,
    required_terms: list[str],
    *,
    nobs: float,
    n_entities: int,
    n_dates: int,
    estimator: str,
    covariance: str,
    fallback_used: bool,
    rsquared_within: float,
    rsquared_overall: float,
) -> pd.DataFrame:
    out = out.copy()
    if "term_status" not in out.columns:
        out["term_status"] = "estimated"

    existing = set(out["term"].astype(str))
    missing = [t for t in required_terms if t not in existing]
    if not missing:
        return out

    fill_rows = []
    for term in missing:
        fill_rows.append(
            {
                "term": term,
                "coef": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "dependent_var": dependent_var,
                "nobs": float(nobs),
                "n_entities": int(n_entities),
                "n_dates": int(n_dates),
                "rsquared_within": float(rsquared_within),
                "rsquared_overall": float(rsquared_overall),
                "estimator": estimator,
                "covariance": covariance,
                "fallback_used": bool(fallback_used),
                "term_status": "absorbed_or_dropped",
            }
        )

    LOGGER.warning("Required TWFE terms missing for %s: %s", dependent_var, ", ".join(missing))
    out = pd.concat([out, pd.DataFrame(fill_rows)], ignore_index=True)
    return out


def fit_linearmodels(df: pd.DataFrame, dependent_var: str, controls: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from linearmodels.panel import PanelOLS

    panel = df.dropna(subset=[dependent_var]).set_index(["listing_id", "date"]).sort_index()
    y = panel[dependent_var].astype(float)
    exog_cols = ["dynamic_algo_adopted"] + controls
    X = panel[exog_cols].astype(float)

    model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
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
    out["dependent_var"] = dependent_var
    out["nobs"] = float(res.nobs)
    out["n_entities"] = int(panel.index.get_level_values(0).nunique())
    out["n_dates"] = int(panel.index.get_level_values(1).nunique())
    out["rsquared_within"] = float(getattr(res, "rsquared_within", np.nan))
    out["rsquared_overall"] = float(getattr(res, "rsquared_overall", np.nan))
    out["estimator"] = "linearmodels.PanelOLS"
    out["covariance"] = "clustered_by_listing"
    out["fallback_used"] = False
    out["term_status"] = "estimated"

    out = ensure_required_terms(
        out,
        dependent_var,
        REQUIRED_TWFE_TERMS,
        nobs=float(res.nobs),
        n_entities=int(panel.index.get_level_values(0).nunique()),
        n_dates=int(panel.index.get_level_values(1).nunique()),
        estimator="linearmodels.PanelOLS",
        covariance="clustered_by_listing",
        fallback_used=False,
        rsquared_within=float(getattr(res, "rsquared_within", np.nan)),
        rsquared_overall=float(getattr(res, "rsquared_overall", np.nan)),
    )

    summary = {
        "dependent_var": dependent_var,
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


def fit_statsmodels_fallback(df: pd.DataFrame, dependent_var: str, controls: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    import statsmodels.formula.api as smf

    rhs = ["dynamic_algo_adopted"] + controls + ["C(listing_id)", "C(date)"]
    formula = f"{dependent_var} ~ " + " + ".join(rhs)

    data = df.dropna(subset=[dependent_var]).copy()
    model = smf.ols(formula=formula, data=data)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": data["listing_id"]})

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
    out["dependent_var"] = dependent_var
    out["nobs"] = float(res.nobs)
    out["n_entities"] = int(data["listing_id"].nunique())
    out["n_dates"] = int(data["date"].nunique())
    out["rsquared_within"] = np.nan
    out["rsquared_overall"] = float(res.rsquared)
    out["estimator"] = "statsmodels.OLS_with_dummies"
    out["covariance"] = "clustered_by_listing"
    out["fallback_used"] = True
    out["term_status"] = "estimated"

    out = ensure_required_terms(
        out,
        dependent_var,
        REQUIRED_TWFE_TERMS,
        nobs=float(res.nobs),
        n_entities=int(data["listing_id"].nunique()),
        n_dates=int(data["date"].nunique()),
        estimator="statsmodels.OLS_with_dummies",
        covariance="clustered_by_listing",
        fallback_used=True,
        rsquared_within=np.nan,
        rsquared_overall=float(res.rsquared),
    )

    summary = {
        "dependent_var": dependent_var,
        "estimator": "statsmodels.OLS_with_dummies",
        "fallback_used": True,
        "controls_used": controls,
        "nobs": float(res.nobs),
        "n_entities": int(data["listing_id"].nunique()),
        "n_dates": int(data["date"].nunique()),
        "rsquared_within": np.nan,
        "rsquared_overall": float(res.rsquared),
    }
    return out, summary


def fit_model(df: pd.DataFrame, dependent_var: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    controls = choose_controls(df, dependent_var=dependent_var)
    model_df = df[["listing_id", "date", dependent_var, "dynamic_algo_adopted", *controls]].copy()
    model_df = impute_controls(model_df, controls)

    try:
        return fit_linearmodels(model_df, dependent_var=dependent_var, controls=controls)
    except Exception as exc:
        LOGGER.warning("PanelOLS failed for %s, falling back to statsmodels: %s", dependent_var, exc)
        return fit_statsmodels_fallback(model_df, dependent_var=dependent_var, controls=controls)


def write_markdown(levels: pd.DataFrame, volatility: pd.DataFrame, summary: dict[str, Any]) -> None:
    def fmt_effect(df: pd.DataFrame, dep_var: str) -> str:
        row = df[(df["dependent_var"] == dep_var) & (df["term"] == "dynamic_algo_adopted")]
        if row.empty:
            return f"- `{dep_var}`: `dynamic_algo_adopted` dropped/absorbed."
        r = row.iloc[0]
        if pd.isna(r.get("coef")):
            return f"- `{dep_var}`: `dynamic_algo_adopted` dropped/absorbed."
        return (
            f"- `{dep_var}`: coef = {r['coef']:.6f}, SE = {r['std_error']:.6f}, "
            f"p = {r['p_value']:.4g}, 95% CI [{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
        )

    sampling = summary.get("sampling", {})

    md = [
        "# Panel Extension TWFE Results",
        "",
        "## Levels model",
        fmt_effect(levels, "log_price"),
        "",
        "## Volatility models",
        fmt_effect(volatility, "abs_price_change"),
        fmt_effect(volatility, "rolling_7d_variance"),
        "",
        "## Sampling and specification",
        (
            "- Sampling: kept all treated listings and 1/"
            + str(sampling.get("control_sample_mod", "NA"))
            + " of non-adopters "
            + f"({int(sampling.get('rows_kept_after_sampling', 0)):,}/{int(sampling.get('rows_input_total', 0)):,} rows)."
            if sampling
            else "- Sampling: none"
        ),
        "- FE: listing and date fixed effects.",
        "- SE: clustered by listing when feasible.",
    ]

    RESULTS_MD.write_text("\n".join(md) + "\n")


def main() -> None:
    configure_logging()

    df, sample_meta = load_panel()

    level_results, level_summary = fit_model(df, dependent_var="log_price")

    vol_frames: list[pd.DataFrame] = []
    vol_summaries: list[dict[str, Any]] = []

    for dep_var in ["abs_price_change", "rolling_7d_variance"]:
        if dep_var not in df.columns or df[dep_var].notna().sum() == 0:
            LOGGER.warning("Skipping volatility DV %s: column missing or fully NA.", dep_var)
            continue
        try:
            out_df, out_summary = fit_model(df, dependent_var=dep_var)
            vol_frames.append(out_df)
            vol_summaries.append(out_summary)
        except Exception as exc:
            LOGGER.warning("Volatility model failed for %s: %s", dep_var, exc)

    if not vol_frames:
        raise RuntimeError("No volatility TWFE model could be estimated.")

    vol_results = pd.concat(vol_frames, ignore_index=True)

    level_results.to_csv(RESULTS_LEVELS_CSV, index=False)
    vol_results.to_csv(RESULTS_VOL_CSV, index=False)

    run_summary: dict[str, Any] = {
        "levels": level_summary,
        "volatility_models": vol_summaries,
        "sampling": sample_meta,
        "files": {
            "levels_csv": str(RESULTS_LEVELS_CSV),
            "volatility_csv": str(RESULTS_VOL_CSV),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(run_summary, indent=2))
    write_markdown(level_results, vol_results, run_summary)

    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
