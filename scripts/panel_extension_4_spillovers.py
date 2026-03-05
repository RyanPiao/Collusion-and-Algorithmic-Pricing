#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/panel_extension"
RAW_DAY2 = ROOT / "data/raw/step2"
OUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT / "dynamic_proxy_panel.csv"
BREAKS_PATH = OUT / "listing_breaks.csv"
RESULTS_CSV = OUT / "spillover_results.csv"
RESULTS_MD = OUT / "spillover_results.md"
PEN_SUMMARY_CSV = OUT / "neighborhood_penetration_summary.csv"
SUMMARY_JSON = OUT / "spillover_run_summary.json"

CONTROL_SAMPLE_MOD = 100  # keep ~1% non-adopters + all adopters for TWFE estimation.
RADIUS_KM = 1.0
EARTH_RADIUS_KM = 6371.0088
SPILLOVER_OWN_TERM = "dynamic_algo_adopted"
SPILLOVER_PEN_TERM = "algo_penetration_1km"
SPILLOVER_INTERACTION_TERM = "dynamic_x_penetration"

LOGGER = logging.getLogger("panel_extension_4")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def get_treated_ids() -> set[int]:
    breaks = pd.read_csv(BREAKS_PATH, usecols=["listing_id", "adopted"])
    return set(breaks.loc[breaks["adopted"] == 1, "listing_id"].astype("int64").tolist())


def load_listing_coordinates() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for listings_path in RAW_DAY2.glob("*/listings.csv.gz"):
        city_slug = listings_path.parent.name
        try:
            df = pd.read_csv(
                listings_path,
                compression="gzip",
                usecols=["id", "latitude", "longitude"],
                low_memory=False,
            )
            df = df.rename(columns={"id": "listing_id"})
            df["listing_id"] = pd.to_numeric(df["listing_id"], errors="coerce").astype("Int64")
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df = df.dropna(subset=["listing_id", "latitude", "longitude"]).copy()
            df["listing_id"] = df["listing_id"].astype("int64")
            df["city_slug"] = city_slug
            rows.append(df[["city_slug", "listing_id", "latitude", "longitude"]])
        except Exception as exc:
            LOGGER.warning("Could not load coordinates from %s: %s", listings_path, exc)

    if not rows:
        return pd.DataFrame(columns=["city_slug", "listing_id", "latitude", "longitude"])

    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["city_slug", "listing_id"])


def load_sample(treated_ids: set[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    header = pd.read_csv(PANEL_PATH, nrows=0)

    has_coords_in_panel = {"latitude", "longitude"}.issubset(set(header.columns))

    usecols = [
        "city_slug",
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
        "latitude",
        "longitude",
    ]
    usecols = [c for c in usecols if c in header.columns]

    dtype_map = {
        "city_slug": "string",
        "listing_id": "int64",
        "log_price": "float32",
        "available": "float32",
        "dynamic_algo_adopted": "float32",
        "minimum_nights": "float32",
        "maximum_nights": "float32",
        "post_cutoff": "float32",
        "price_volatility_7d": "float32",
        "price_volatility_14d": "float32",
        "latitude": "float64",
        "longitude": "float64",
    }
    dtype_use = {k: v for k, v in dtype_map.items() if k in usecols}

    coords_map = None
    if not has_coords_in_panel:
        LOGGER.info("Coordinates not found in dynamic panel; loading from raw listings snapshots.")
        coords_map = load_listing_coordinates()

    chunks: list[pd.DataFrame] = []
    n_rows_in = 0
    n_rows_kept = 0

    for chunk in pd.read_csv(PANEL_PATH, usecols=usecols, dtype=dtype_use, parse_dates=["date"], chunksize=450_000):
        n_rows_in += len(chunk)
        keep = chunk["listing_id"].isin(treated_ids) | ((chunk["listing_id"] % CONTROL_SAMPLE_MOD) == 0)
        chunk = chunk.loc[keep].copy()
        if chunk.empty:
            continue

        if coords_map is not None and not coords_map.empty:
            chunk = chunk.merge(coords_map, on=["city_slug", "listing_id"], how="left", suffixes=("", "_raw"))
            if "latitude_raw" in chunk.columns:
                chunk["latitude"] = pd.to_numeric(chunk.get("latitude"), errors="coerce")
                chunk["latitude"] = chunk["latitude"].fillna(pd.to_numeric(chunk["latitude_raw"], errors="coerce"))
                chunk = chunk.drop(columns=["latitude_raw"])
            if "longitude_raw" in chunk.columns:
                chunk["longitude"] = pd.to_numeric(chunk.get("longitude"), errors="coerce")
                chunk["longitude"] = chunk["longitude"].fillna(pd.to_numeric(chunk["longitude_raw"], errors="coerce"))
                chunk = chunk.drop(columns=["longitude_raw"])

        chunks.append(chunk)
        n_rows_kept += len(chunk)

    if not chunks:
        raise RuntimeError("No sampled rows for spillover TWFE.")

    df = pd.concat(chunks, ignore_index=True)

    for col in [
        "log_price",
        "available",
        "dynamic_algo_adopted",
        "minimum_nights",
        "maximum_nights",
        "post_cutoff",
        "price_volatility_7d",
        "price_volatility_14d",
        "latitude",
        "longitude",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sample_meta = {
        "control_sample_mod": CONTROL_SAMPLE_MOD,
        "treated_listings_retained_all": True,
        "rows_input_total": int(n_rows_in),
        "rows_kept_after_sampling": int(n_rows_kept),
        "sampling_share_rows": float(n_rows_kept / n_rows_in) if n_rows_in else np.nan,
        "n_treated_listings": int(len(treated_ids)),
        "radius_km": RADIUS_KM,
        "distance_method": "BallTree(haversine)",
        "coords_available_share": float(df[["latitude", "longitude"]].notna().all(axis=1).mean())
        if {"latitude", "longitude"}.issubset(df.columns)
        else 0.0,
    }
    return df, sample_meta


def compute_local_penetration(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    if g.empty:
        g["local_penetration_1km"] = np.nan
        return g

    if "latitude" not in g.columns or "longitude" not in g.columns:
        g["local_penetration_1km"] = np.nan
        return g

    g["available"] = pd.to_numeric(g["available"], errors="coerce").fillna(0.0)
    g["dynamic_algo_adopted"] = pd.to_numeric(g["dynamic_algo_adopted"], errors="coerce").fillna(0.0)

    valid_coords = g[["latitude", "longitude"]].notna().all(axis=1)
    active = (g["available"] > 0) & valid_coords

    penetration = np.full(len(g), np.nan, dtype=float)

    if active.sum() == 0 or valid_coords.sum() == 0:
        g["local_penetration_1km"] = penetration
        return g

    active_df = g.loc[active, ["listing_id", "latitude", "longitude", "dynamic_algo_adopted"]].copy()
    all_df = g.loc[valid_coords, ["listing_id", "latitude", "longitude"]].copy()

    active_coords = np.radians(active_df[["latitude", "longitude"]].to_numpy(dtype=float))
    all_coords = np.radians(all_df[["latitude", "longitude"]].to_numpy(dtype=float))

    tree = BallTree(active_coords, metric="haversine")
    radius = RADIUS_KM / EARTH_RADIUS_KM
    neighbors = tree.query_radius(all_coords, r=radius, return_distance=False)

    active_ids = active_df["listing_id"].to_numpy()
    active_adopt = active_df["dynamic_algo_adopted"].to_numpy(dtype=float)
    query_ids = all_df["listing_id"].to_numpy()

    local_vals = np.full(len(all_df), np.nan, dtype=float)
    for i, neigh_idx in enumerate(neighbors):
        if neigh_idx.size == 0:
            continue

        # Exclude self by listing_id.
        keep_idx = neigh_idx[active_ids[neigh_idx] != query_ids[i]]
        if keep_idx.size == 0:
            continue

        local_vals[i] = float(np.nanmean(active_adopt[keep_idx]))

    valid_pos = np.flatnonzero(valid_coords.to_numpy())
    penetration[valid_pos] = local_vals

    g["local_penetration_1km"] = penetration
    return g


def add_localized_penetration(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Computing 1km localized penetration with BallTree(haversine).")

    out_parts: list[pd.DataFrame] = []
    grouped = df.groupby(["city_slug", "date"], observed=True, sort=False)

    for idx, (_, gp) in enumerate(grouped, start=1):
        try:
            out_gp = compute_local_penetration(gp)
            out_parts.append(out_gp)
        except Exception as exc:
            LOGGER.warning("Penetration failed for one city-date group; filling NaN. Error: %s", exc)
            gp = gp.copy()
            gp["local_penetration_1km"] = np.nan
            out_parts.append(gp)

        if idx % 150 == 0:
            LOGGER.info("Penetration progress: %s city-date groups processed.", f"{idx:,}")

    out = pd.concat(out_parts, ignore_index=True)
    out[SPILLOVER_PEN_TERM] = pd.to_numeric(out["local_penetration_1km"], errors="coerce").fillna(0.0).astype("float32")
    out[SPILLOVER_INTERACTION_TERM] = (
        pd.to_numeric(out[SPILLOVER_OWN_TERM], errors="coerce").fillna(0.0) * out[SPILLOVER_PEN_TERM]
    ).astype("float32")
    return out


def choose_controls(df: pd.DataFrame) -> list[str]:
    # NOTE: post_cutoff is excluded to avoid absorbing own-adoption/spillover terms
    # once listing/date FE are included.
    candidates = ["available", "minimum_nights", "maximum_nights", "price_volatility_7d", "price_volatility_14d"]
    return [c for c in candidates if c in df.columns and df[c].notna().any()]


def ensure_required_terms(
    out: pd.DataFrame,
    required_terms: list[str],
    *,
    nobs: float,
    n_entities: int,
    n_dates: int,
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
                "nobs": float(nobs),
                "n_entities": int(n_entities),
                "n_dates": int(n_dates),
                "estimator": "linearmodels.PanelOLS",
                "covariance": "clustered_by_listing",
                "term_status": "absorbed_or_dropped",
            }
        )

    LOGGER.warning("Spillover required terms missing: %s", ", ".join(missing))
    return pd.concat([out, pd.DataFrame(fill_rows)], ignore_index=True)


def fit_model(df: pd.DataFrame, controls: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from linearmodels.panel import PanelOLS

    model_df = df.dropna(subset=["listing_id", "date", "log_price", "dynamic_algo_adopted"]).copy()

    for c in controls:
        med_listing = model_df.groupby("listing_id", observed=True)[c].transform("median")
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce").fillna(med_listing)
        med = model_df[c].median()
        if pd.isna(med):
            med = 0.0
        model_df[c] = model_df[c].fillna(med)

    exog_cols = [SPILLOVER_OWN_TERM, SPILLOVER_PEN_TERM, SPILLOVER_INTERACTION_TERM] + controls

    panel = model_df.set_index(["listing_id", "date"]).sort_index()
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
    nobs = float(res.nobs)
    n_entities = int(panel.index.get_level_values(0).nunique())
    n_dates = int(panel.index.get_level_values(1).nunique())

    out["nobs"] = nobs
    out["n_entities"] = n_entities
    out["n_dates"] = n_dates
    out["estimator"] = "linearmodels.PanelOLS"
    out["covariance"] = "clustered_by_listing"
    out["term_status"] = "estimated"

    out = ensure_required_terms(
        out,
        [SPILLOVER_OWN_TERM, SPILLOVER_PEN_TERM, SPILLOVER_INTERACTION_TERM],
        nobs=nobs,
        n_entities=n_entities,
        n_dates=n_dates,
    )

    summary = {
        "estimator": "linearmodels.PanelOLS",
        "nobs": float(res.nobs),
        "n_entities": int(panel.index.get_level_values(0).nunique()),
        "n_dates": int(panel.index.get_level_values(1).nunique()),
        "controls_used": controls,
        "rsquared_within": float(getattr(res, "rsquared_within", np.nan)),
        "radius_km": RADIUS_KM,
        "distance_method": "BallTree(haversine)",
    }
    return out, summary


def write_penetration_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["city_slug"], observed=True)[SPILLOVER_PEN_TERM]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{SPILLOVER_PEN_TERM}_mean",
                "median": f"{SPILLOVER_PEN_TERM}_median",
                "std": f"{SPILLOVER_PEN_TERM}_std",
                "count": "n_obs",
            }
        )
    )
    summary.to_csv(PEN_SUMMARY_CSV, index=False)


def write_markdown(results: pd.DataFrame, summary: dict[str, Any]) -> None:
    def fmt_term(term: str) -> str:
        row = results.loc[results["term"] == term]
        if row.empty:
            return f"- `{term}` dropped/absorbed."
        r = row.iloc[0]
        if pd.isna(r.get("coef")):
            return f"- `{term}` dropped/absorbed."
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
        f"- Local spillover radius: **{summary.get('radius_km', RADIUS_KM)} km** ({summary.get('distance_method', 'BallTree(haversine)')})",
        (
            "- Sampling: kept all treated listings and 1/"
            + str(sampling.get("control_sample_mod", "NA"))
            + " of non-adopters "
            + f"({int(sampling.get('rows_kept_after_sampling', 0)):,}/{int(sampling.get('rows_input_total', 0)):,} rows)."
            if sampling
            else "- Sampling: none"
        ),
        "",
        fmt_term(SPILLOVER_OWN_TERM),
        fmt_term(SPILLOVER_PEN_TERM),
        fmt_term(SPILLOVER_INTERACTION_TERM),
        "",
        f"Specification: `log_price ~ {SPILLOVER_OWN_TERM} + {SPILLOVER_PEN_TERM} + {SPILLOVER_INTERACTION_TERM} + controls + listing FE + date FE`, clustered by listing.",
    ]
    RESULTS_MD.write_text("\n".join(md) + "\n")


def main() -> None:
    configure_logging()

    treated_ids = get_treated_ids()
    df, sample_meta = load_sample(treated_ids=treated_ids)

    try:
        df = add_localized_penetration(df)
    except Exception as exc:
        LOGGER.exception("Localized penetration computation failed: %s", exc)
        raise

    write_penetration_summary(df)

    controls = choose_controls(df)

    try:
        results, summary = fit_model(df, controls)
    except Exception as exc:
        LOGGER.exception("Spillover PanelOLS failed: %s", exc)
        raise

    summary["sampling"] = sample_meta

    results.to_csv(RESULTS_CSV, index=False)
    write_markdown(results, summary)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
