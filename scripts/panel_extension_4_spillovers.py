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
RESULTS_CSV = OUT / "spillover_results.csv"
RESULTS_MD = OUT / "spillover_results.md"
PEN_SUMMARY_CSV = OUT / "neighborhood_penetration_summary.csv"
SUMMARY_JSON = OUT / "spillover_run_summary.json"

CONTROL_SAMPLE_MOD = 100  # keep ~1% non-adopters + all adopters for TWFE estimation.


def header_and_cols() -> tuple[str, list[str], dict[str, str]]:
    header = pd.read_csv(PANEL_PATH, nrows=0)
    neighbourhood_col = "neighbourhood" if "neighbourhood" in header.columns else "neighbourhood_cleansed"

    usecols = [
        "city_slug",
        neighbourhood_col,
        "listing_id",
        "date",
        "log_price",
        "available",
        "dynamic_algo_adopted",
        "minimum_nights",
        "maximum_nights",
        "post_cutoff",
        "price_volatility_7d",
        "price_volatility_14d",
    ]
    usecols = [c for c in usecols if c in header.columns]

    dtype_map = {
        "city_slug": "string",
        neighbourhood_col: "string",
        "listing_id": "int64",
        "log_price": "float32",
        "available": "float32",
        "dynamic_algo_adopted": "float32",
        "minimum_nights": "float32",
        "maximum_nights": "float32",
        "post_cutoff": "float32",
        "price_volatility_7d": "float32",
        "price_volatility_14d": "float32",
    }
    dtype_use = {k: v for k, v in dtype_map.items() if k in usecols}
    return neighbourhood_col, usecols, dtype_use


def get_treated_ids() -> set[int]:
    breaks = pd.read_csv(BREAKS_PATH, usecols=["listing_id", "adopted"])
    return set(breaks.loc[breaks["adopted"] == 1, "listing_id"].astype("int64").tolist())


def build_group_aggregates(neighbourhood_col: str, usecols: list[str], dtype_use: dict[str, str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    keys = ["city_slug", neighbourhood_col, "date"]

    for chunk in pd.read_csv(PANEL_PATH, usecols=usecols, dtype=dtype_use, parse_dates=["date"], chunksize=500_000):
        chunk["city_slug"] = chunk["city_slug"].fillna("UNKNOWN")
        chunk[neighbourhood_col] = chunk[neighbourhood_col].fillna("UNKNOWN")
        chunk["available"] = pd.to_numeric(chunk["available"], errors="coerce").fillna(0.0)
        chunk["dynamic_algo_adopted"] = pd.to_numeric(chunk["dynamic_algo_adopted"], errors="coerce").fillna(0.0)
        chunk["active"] = (chunk["available"] > 0).astype("int8")
        chunk["adopt_active"] = (chunk["dynamic_algo_adopted"] * chunk["active"]).astype("float32")

        grp = (
            chunk.groupby(keys, observed=True)
            .agg(group_adopt_active_sum=("adopt_active", "sum"), group_active_count=("active", "sum"))
            .reset_index()
        )
        parts.append(grp)

    agg = (
        pd.concat(parts, ignore_index=True)
        .groupby(keys, observed=True, as_index=False)[["group_adopt_active_sum", "group_active_count"]]
        .sum()
    )
    return agg


def build_sample_with_penetration(
    neighbourhood_col: str,
    usecols: list[str],
    dtype_use: dict[str, str],
    agg: pd.DataFrame,
    treated_ids: set[int],
) -> tuple[pd.DataFrame, dict]:
    keys = ["city_slug", neighbourhood_col, "date"]
    sampled_chunks: list[pd.DataFrame] = []

    n_rows_in = 0
    n_rows_kept = 0

    for chunk in pd.read_csv(PANEL_PATH, usecols=usecols, dtype=dtype_use, parse_dates=["date"], chunksize=500_000):
        n_rows_in += len(chunk)
        chunk["city_slug"] = chunk["city_slug"].fillna("UNKNOWN")
        chunk[neighbourhood_col] = chunk[neighbourhood_col].fillna("UNKNOWN")
        chunk["available"] = pd.to_numeric(chunk["available"], errors="coerce").fillna(0.0)
        chunk["dynamic_algo_adopted"] = pd.to_numeric(chunk["dynamic_algo_adopted"], errors="coerce").fillna(0.0)
        chunk["active"] = (chunk["available"] > 0).astype("int8")
        chunk["adopt_active"] = (chunk["dynamic_algo_adopted"] * chunk["active"]).astype("float32")

        chunk = chunk.merge(agg, on=keys, how="left")
        chunk["group_adopt_active_sum"] = chunk["group_adopt_active_sum"].fillna(0.0)
        chunk["group_active_count"] = chunk["group_active_count"].fillna(0.0)

        peer_num = chunk["group_adopt_active_sum"] - chunk["adopt_active"]
        peer_den = chunk["group_active_count"] - chunk["active"]
        chunk["neighborhood_algo_penetration"] = np.where(peer_den > 0, peer_num / peer_den, np.nan)
        chunk["neighborhood_algo_penetration"] = chunk["neighborhood_algo_penetration"].astype("float32").fillna(0.0)
        chunk["adoption_x_penetration"] = (chunk["dynamic_algo_adopted"] * chunk["neighborhood_algo_penetration"]).astype("float32")

        keep = chunk["listing_id"].isin(treated_ids) | ((chunk["listing_id"] % CONTROL_SAMPLE_MOD) == 0)
        kept = chunk.loc[keep].copy()
        if kept.empty:
            continue
        sampled_chunks.append(kept)
        n_rows_kept += len(kept)

    if not sampled_chunks:
        raise RuntimeError("No sampled rows for spillover TWFE.")

    df = pd.concat(sampled_chunks, ignore_index=True)
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
    return [c for c in candidates if c in df.columns and df[c].notna().any()]


def fit_model(df: pd.DataFrame, controls: list[str]) -> tuple[pd.DataFrame, dict]:
    from linearmodels.panel import PanelOLS

    df = df.dropna(subset=["listing_id", "date", "log_price", "dynamic_algo_adopted"]).copy()

    for c in controls:
        med_listing = df.groupby("listing_id")[c].transform("median")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med_listing)
        med = df[c].median()
        if pd.isna(med):
            med = 0.0
        df[c] = df[c].fillna(med)

    exog_cols = ["dynamic_algo_adopted", "neighborhood_algo_penetration", "adoption_x_penetration"] + controls

    panel = df.set_index(["listing_id", "date"]).sort_index()
    y = panel["log_price"].astype(float)
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
    out["nobs"] = float(res.nobs)
    out["n_entities"] = int(panel.index.get_level_values(0).nunique())
    out["n_dates"] = int(panel.index.get_level_values(1).nunique())
    out["estimator"] = "linearmodels.PanelOLS"
    out["covariance"] = "clustered_by_listing"

    summary = {
        "estimator": "linearmodels.PanelOLS",
        "nobs": float(res.nobs),
        "n_entities": int(panel.index.get_level_values(0).nunique()),
        "n_dates": int(panel.index.get_level_values(1).nunique()),
        "controls_used": controls,
        "rsquared_within": float(getattr(res, "rsquared_within", np.nan)),
    }
    return out, summary


def write_penetration_summary(df: pd.DataFrame, neighbourhood_col: str) -> None:
    summary = (
        df.groupby(["city_slug", neighbourhood_col], observed=True)["neighborhood_algo_penetration"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "penetration_mean", "median": "penetration_median", "std": "penetration_std", "count": "n_obs"})
    )
    summary.to_csv(PEN_SUMMARY_CSV, index=False)


def write_markdown(results: pd.DataFrame, summary: dict) -> None:
    def fmt_term(term: str) -> str:
        row = results.loc[results["term"] == term]
        if row.empty:
            return f"- `{term}` dropped/absorbed."
        r = row.iloc[0]
        return (
            f"- `{term}`: coef = {r['coef']:.6f}, SE = {r['std_error']:.6f}, "
            f"p = {r['p_value']:.4g}, 95% CI [{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
        )

    sampling = summary.get("sampling", {})
    md = [
        "# Spillover TWFE Results",
        "",
        f"- Estimator: **{summary['estimator']}**",
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
        "",
        fmt_term("dynamic_algo_adopted"),
        fmt_term("neighborhood_algo_penetration"),
        fmt_term("adoption_x_penetration"),
        "",
        "Specification: `log_price ~ own adoption + neighborhood penetration + interaction + controls + listing FE + date FE`, clustered by listing.",
    ]
    RESULTS_MD.write_text("\n".join(md) + "\n")


def main() -> None:
    neighbourhood_col, usecols, dtype_use = header_and_cols()
    treated_ids = get_treated_ids()
    agg = build_group_aggregates(neighbourhood_col, usecols, dtype_use)

    df, sample_meta = build_sample_with_penetration(neighbourhood_col, usecols, dtype_use, agg, treated_ids)
    write_penetration_summary(df, neighbourhood_col)

    controls = choose_controls(df)
    results, summary = fit_model(df, controls)
    summary["sampling"] = sample_meta

    results.to_csv(RESULTS_CSV, index=False)
    write_markdown(results, summary)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
