#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/panel_extension"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT / "dynamic_proxy_panel.csv"
BREAKS_PATH = OUT / "listing_breaks.csv"
COEF_PATH = OUT / "event_study_coefficients.csv"
PLOT_PATH = OUT / "event_study_plot.png"
SUMMARY_JSON = OUT / "event_study_summary.json"

CONTROL_SAMPLE_MOD = 100  # retain ~1% controls and all treated listings.


def load_data() -> tuple[pd.DataFrame, dict]:
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
            & chunk["event_time"].between(-30, 30, inclusive="both")
            & (chunk["event_time"] != -1)
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
    candidates = ["available", "minimum_nights", "maximum_nights", "post_cutoff", "price_volatility_7d", "price_volatility_14d"]
    controls = [c for c in candidates if c in df.columns and df[c].notna().any()]
    return controls


def fit_event_study(df: pd.DataFrame, controls: list[str]) -> tuple[pd.DataFrame, dict]:
    from linearmodels.panel import PanelOLS

    for c in controls:
        med_listing = df.groupby("listing_id")[c].transform("median")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med_listing)
        med = df[c].median()
        if pd.isna(med):
            med = 0.0
        df[c] = df[c].fillna(med)

    event_terms: list[str] = []
    for k in range(-30, 31):
        if k == -1:
            continue
        col = f"event_{k}"
        df[col] = (df["event_time"] == k).astype("float32")
        event_terms.append(col)

    exog_cols = event_terms + controls

    panel = df.set_index(["listing_id", "date"]).sort_index()
    y = panel["log_price"].astype(float)
    X = panel[exog_cols].astype(float)

    model = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    res = model.fit(cov_type="clustered", cluster_entity=True)

    ci = res.conf_int()
    rows = []
    for term in res.params.index:
        if not term.startswith("event_"):
            continue
        k = int(term.replace("event_", ""))
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

    coef = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)

    summary = {
        "estimator": "linearmodels.PanelOLS",
        "nobs": float(res.nobs),
        "n_treated_listings": int(df.loc[df["event_time"].notna(), "listing_id"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "controls_used": controls,
        "event_window": [-30, 30],
        "reference_period": -1,
        "n_event_coefficients": int(len(coef)),
        "rsquared_within": float(getattr(res, "rsquared_within", np.nan)),
    }
    return coef, summary


def make_plot(coef: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    x = coef["event_time"].to_numpy()
    y = coef["coef"].to_numpy()
    yerr = 1.96 * coef["std_error"].to_numpy()

    plt.errorbar(x, y, yerr=yerr, fmt="o", color="#1d4ed8", ecolor="#93c5fd", alpha=0.9, capsize=2)
    plt.plot(x, y, color="#1d4ed8", linewidth=1)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="#dc2626", linestyle="--", linewidth=1)
    plt.title("Event-Study Dynamic Effects (Reference: t = -1)")
    plt.xlabel("Event time (days relative to break date)")
    plt.ylabel("Coefficient on event-time indicator")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


def main() -> None:
    df, sample_meta = load_data()
    controls = choose_controls(df)
    coef, summary = fit_event_study(df, controls)
    summary["sampling"] = sample_meta

    coef.to_csv(COEF_PATH, index=False)
    make_plot(coef)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
