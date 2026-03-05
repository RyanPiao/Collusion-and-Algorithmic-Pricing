#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/panel_extension"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT / "dynamic_proxy_panel.csv"
BREAKS_PATH = OUT / "listing_breaks.csv"
COEF_PATH = OUT / "event_study_coefficients.csv"
PLOT_PATH = OUT / "event_study_plot.png"
SUMMARY_JSON = OUT / "event_study_summary.json"

CONTROL_SAMPLE_MOD = 100  # retain ~1% controls and all treated listings.
EVENT_MIN, EVENT_MAX = -30, 30
REFERENCE_PERIOD = -1

LOGGER = logging.getLogger("panel_extension_3")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(PANEL_PATH, nrows=0)
    base_cols = [
        "listing_id",
        "date",
        "log_price",
        "available",
        "minimum_nights",
        "maximum_nights",
        "post_cutoff",
        "price_volatility_7d",
        "price_volatility_14d",
    ]
    base_cols = [c for c in base_cols if c in header.columns]

    dtype_map = {
        "listing_id": "int64",
        "log_price": "float32",
        "available": "float32",
        "minimum_nights": "float32",
        "maximum_nights": "float32",
        "post_cutoff": "float32",
        "price_volatility_7d": "float32",
        "price_volatility_14d": "float32",
    }
    dtype_use = {k: v for k, v in dtype_map.items() if k in base_cols}

    breaks = pd.read_csv(BREAKS_PATH, usecols=["listing_id", "break_date", "adopted"], low_memory=False)
    breaks = breaks[breaks["adopted"] == 1].copy()
    breaks["break_date"] = pd.to_datetime(breaks["break_date"], errors="coerce")
    breaks = breaks.dropna(subset=["break_date"]).copy()
    treated_ids = set(breaks["listing_id"].astype("int64").tolist())
    break_map = breaks.set_index("listing_id")["break_date"]

    kept_chunks: list[pd.DataFrame] = []
    n_rows_in = 0
    n_rows_kept = 0

    for chunk in pd.read_csv(PANEL_PATH, usecols=base_cols, dtype=dtype_use, parse_dates=["date"], chunksize=500_000):
        n_rows_in += len(chunk)
        keep = chunk["listing_id"].isin(treated_ids) | ((chunk["listing_id"] % CONTROL_SAMPLE_MOD) == 0)
        chunk = chunk.loc[keep].copy()
        if chunk.empty:
            continue

        chunk["break_date"] = pd.to_datetime(chunk["listing_id"].map(break_map), errors="coerce")
        chunk["is_treated_listing"] = chunk["break_date"].notna()
        chunk["event_time"] = np.where(
            chunk["is_treated_listing"],
            (chunk["date"] - chunk["break_date"]).dt.days,
            np.nan,
        )

        treated_keep = (
            chunk["is_treated_listing"]
            & chunk["event_time"].between(EVENT_MIN, EVENT_MAX, inclusive="both")
            & (chunk["event_time"] != REFERENCE_PERIOD)
        )
        control_keep = ~chunk["is_treated_listing"]
        chunk = chunk.loc[treated_keep | control_keep].copy()

        if chunk.empty:
            continue

        kept_chunks.append(chunk)
        n_rows_kept += len(chunk)

    if not kept_chunks:
        raise RuntimeError("No rows retained for event-study estimation.")

    df = pd.concat(kept_chunks, ignore_index=True)
    df = df.dropna(subset=["listing_id", "date", "log_price"]).copy()
    df["event_time"] = pd.to_numeric(df["event_time"], errors="coerce")

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
    # post_cutoff omitted to avoid absorbing event-time variation under listing/date FE.
    candidates = ["available", "minimum_nights", "maximum_nights", "price_volatility_7d", "price_volatility_14d"]
    return [c for c in candidates if c in df.columns and df[c].notna().any()]


def add_event_dummies(df: pd.DataFrame, controls: list[str]) -> tuple[pd.DataFrame, list[str]]:
    event_terms: list[str] = []
    for k in range(EVENT_MIN, EVENT_MAX + 1):
        if k == REFERENCE_PERIOD:
            continue
        col = f"event_{k}"
        df[col] = (df["event_time"] == k).astype("float32")
        event_terms.append(col)

    for c in controls:
        med_listing = df.groupby("listing_id", observed=True)[c].transform("median")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med_listing)
        med = df[c].median()
        if pd.isna(med):
            med = 0.0
        df[c] = df[c].fillna(med)

    return df, event_terms


def residualize_listing_specific_trend(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    FWL residualization by listing-specific intercept + linear time trend:
    y_it <- y_it - (a_i + b_i * t_it), same for each regressor.
    Equivalent to including listing FE + listing-specific linear trends.
    """
    out = df.copy()
    out = out.sort_values(["listing_id", "date"]).copy()

    out["trend_global"] = (out["date"] - out["date"].min()).dt.days.astype(float)

    n = out.groupby("listing_id", observed=True)["trend_global"].transform("size").astype(float)
    sum_t = out.groupby("listing_id", observed=True)["trend_global"].transform("sum")
    t2 = out["trend_global"] ** 2
    sum_t2 = t2.groupby(out["listing_id"], observed=True).transform("sum")
    denom = (n * sum_t2 - sum_t * sum_t).astype(float)

    for col in cols:
        v = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        sum_v = v.groupby(out["listing_id"], observed=True).transform("sum")
        sum_tv = (v * out["trend_global"]).groupby(out["listing_id"], observed=True).transform("sum")

        slope = np.where(np.abs(denom) > 1e-12, (n * sum_tv - sum_t * sum_v) / denom, 0.0)
        intercept = np.where(n > 0, (sum_v - slope * sum_t) / n, 0.0)

        out[f"{col}_detr"] = (v - intercept - slope * out["trend_global"]).astype("float32")

    return out


def fit_event_study(df: pd.DataFrame, controls: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from linearmodels.panel import PanelOLS

    df, event_terms = add_event_dummies(df, controls)

    model_cols = ["log_price"] + event_terms + controls
    df = residualize_listing_specific_trend(df, model_cols)

    y_col = "log_price_detr"
    x_cols = [f"{c}_detr" for c in event_terms + controls]

    panel = df.set_index(["listing_id", "date"]).sort_index()
    y = panel[y_col].astype(float)
    X = panel[x_cols].astype(float)

    # Keep PanelOLS FE structure, with listing-specific trends absorbed via detrending.
    model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    res = model.fit(cov_type="clustered", cluster_entity=True)

    ci = res.conf_int()
    rows = []
    for term in res.params.index:
        if not term.startswith("event_"):
            continue

        k = int(term.replace("event_", "").replace("_detr", ""))
        rows.append(
            {
                "event_time": k,
                "term": term,
                "coef": float(res.params[term]),
                "std_error": float(res.std_errors[term]),
                "t_stat": float(res.tstats[term]),
                "p_value": float(res.pvalues[term]),
                "ci_low": float(ci.loc[term].iloc[0]),
                "ci_high": float(ci.loc[term].iloc[1]),
            }
        )

    coef = pd.DataFrame(rows)

    full_event_times = [k for k in range(EVENT_MIN, EVENT_MAX + 1) if k != REFERENCE_PERIOD]
    full_window = pd.DataFrame({"event_time": full_event_times})
    coef = full_window.merge(coef, on="event_time", how="left")
    coef["term"] = coef["term"].fillna(coef["event_time"].map(lambda k: f"event_{k}_detr"))
    coef["term_status"] = np.where(coef["coef"].isna(), "absorbed_or_dropped", "estimated")

    reference_row = pd.DataFrame(
        [
            {
                "event_time": REFERENCE_PERIOD,
                "term": "reference_period",
                "coef": 0.0,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "term_status": "reference_omitted",
            }
        ]
    )
    coef = pd.concat([coef, reference_row], ignore_index=True).sort_values("event_time").reset_index(drop=True)

    estimated = coef[coef["term_status"] == "estimated"]
    pre = estimated[(estimated["event_time"] >= EVENT_MIN) & (estimated["event_time"] <= -2)]
    post = estimated[(estimated["event_time"] >= 0) & (estimated["event_time"] <= EVENT_MAX)]

    summary = {
        "estimator": "linearmodels.PanelOLS",
        "trend_adjustment": "listing-specific linear trends (FWL detrending before PanelOLS)",
        "nobs": float(res.nobs),
        "n_treated_listings": int(df.loc[df["event_time"].notna(), "listing_id"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "controls_used": controls,
        "event_window": [EVENT_MIN, EVENT_MAX],
        "reference_period": REFERENCE_PERIOD,
        "n_event_coefficients_full_window": int(len(coef)),
        "n_event_coefficients_estimated": int(len(estimated)),
        "n_event_coefficients_absorbed_or_dropped": int((coef["term_status"] == "absorbed_or_dropped").sum()),
        "rsquared_within": float(getattr(res, "rsquared_within", np.nan)),
        "mean_pre_period_coef": float(pre["coef"].mean()) if not pre.empty else np.nan,
        "mean_post_period_coef": float(post["coef"].mean()) if not post.empty else np.nan,
        "share_pre_period_p_lt_0_05": float((pre["p_value"] < 0.05).mean()) if not pre.empty else np.nan,
    }
    return coef, summary


def make_plot(coef: pd.DataFrame) -> None:
    if coef.empty:
        raise RuntimeError("Event-study coefficient frame is empty; cannot plot.")

    if "term_status" in coef.columns:
        plot_df = coef.loc[coef["term_status"] == "estimated"].copy()
    else:
        plot_df = coef.copy()
    if plot_df.empty:
        raise RuntimeError("No estimated event-study coefficients available for plotting.")

    plt.figure(figsize=(9, 5))
    x = plot_df["event_time"].to_numpy()
    y = plot_df["coef"].to_numpy()
    yerr = 1.96 * plot_df["std_error"].to_numpy()

    plt.errorbar(x, y, yerr=yerr, fmt="o", color="#1d4ed8", ecolor="#93c5fd", alpha=0.9, capsize=2)
    plt.plot(x, y, color="#1d4ed8", linewidth=1)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="#dc2626", linestyle="--", linewidth=1)
    plt.title("Event-Study Dynamic Effects (Reference: t = -1)")
    plt.xlabel("Event time (days relative to break date)")
    plt.ylabel("Coefficient (detrended by listing-specific linear trends)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    configure_logging()

    try:
        df, sample_meta = load_data()
        controls = choose_controls(df)
        coef, summary = fit_event_study(df, controls)
        summary["sampling"] = sample_meta

        coef.to_csv(COEF_PATH, index=False)

        try:
            make_plot(coef)
        except Exception as exc:
            LOGGER.warning("Could not generate event-study plot: %s", exc)

        SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))

    except Exception as exc:
        LOGGER.exception("Event-study script failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
