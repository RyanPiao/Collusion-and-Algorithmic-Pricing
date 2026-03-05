#!/usr/bin/env python3
"""Refined ML-econometrics extension with dynamic pre-cutoff proxy + heterogeneous IV.

Improvements:
1) Dynamic pre-cutoff feature engineering (strictly post_cutoff == 0)
2) Updated GMM latent proxy (k=4 default)
3) Heterogeneous fuzzy-RDD/IV interaction model
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CITY_ORDER = [
    "boston",
    "new-york-city",
    "los-angeles",
    "san-francisco",
    "austin",
    "chicago",
    "seattle",
    "washington-dc",
]

RAW_NUMERIC_COLS = [
    "price_usd",
    "log_price",
    "available",
    "minimum_nights",
    "maximum_nights",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "host_tenure_days",
    "host_is_superhost",
    "host_identity_verified",
    "host_response_rate",
    "host_acceptance_rate",
]
RAW_CAT_COLS = ["city_slug", "room_type", "property_type"]
MODEL_USECOLS = [
    "listing_id",
    "city_slug",
    "log_price",
    "available",
    "post_cutoff",
    "days_from_cutoff",
    "minimum_nights",
    "maximum_nights",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "host_tenure_days",
    "host_is_superhost",
    "host_identity_verified",
]


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _clean_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return np.nan_to_num(arr, nan=default, posinf=default, neginf=default)


def _safe_inv(x: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.nan_to_num(x, nan=0.0), rcond=1e-10)


def _ols_hc1(x: np.ndarray, y: np.ndarray, names: List[str]) -> pd.DataFrame:
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[valid]
    y = y[valid]
    n, k = x.shape
    if n == 0:
        raise ValueError("No valid observations for OLS.")

    inv = _safe_inv(x.T @ x)
    beta = inv @ (x.T @ y)
    resid = y - x @ beta
    meat = x.T @ (x * (resid[:, None] ** 2))
    hc1 = (n / max(n - k, 1)) * (inv @ meat @ inv)
    se = np.sqrt(np.clip(np.diag(hc1), 0.0, np.inf))

    out = pd.DataFrame({"term": names, "coef": beta, "se_hc1": se})
    out["t"] = out["coef"] / out["se_hc1"]
    out["ci95_low"] = out["coef"] - 1.96 * out["se_hc1"]
    out["ci95_high"] = out["coef"] + 1.96 * out["se_hc1"]
    out["n_obs"] = n
    return out


def build_pre_cutoff_listing_features(input_file: Path, chunk_size: int = 1_000_000) -> pd.DataFrame:
    """Build listing-level features using only pre-cutoff rows.

    Dynamic additions:
    - price_variance_pre: variance of daily log price
    - weekend_premium_pre: mean(weekend price_usd) - mean(weekday price_usd)
    - price_change_frequency_pre: share of step-to-step pre-cutoff transitions with price change
    """

    usecols = ["listing_id", "date", "post_cutoff", *RAW_NUMERIC_COLS, *RAW_CAT_COLS]

    sum_parts: List[pd.DataFrame] = []
    count_parts: List[pd.DataFrame] = []
    cat_parts: List[pd.DataFrame] = []

    # Dynamic-stat running aggregators
    var_count: Dict[int, int] = {}
    var_sum: Dict[int, float] = {}
    var_sumsq: Dict[int, float] = {}

    wknd_sum: Dict[int, float] = {}
    wknd_cnt: Dict[int, int] = {}
    wkdy_sum: Dict[int, float] = {}
    wkdy_cnt: Dict[int, int] = {}

    last_price: Dict[int, float] = {}
    change_cnt: Dict[int, int] = {}
    trans_cnt: Dict[int, int] = {}

    reader = pd.read_csv(input_file, compression="gzip", usecols=usecols, chunksize=chunk_size, low_memory=False)

    for chunk in reader:
        pre = chunk.loc[chunk["post_cutoff"] == 0, ["listing_id", "date", *RAW_NUMERIC_COLS, *RAW_CAT_COLS]].copy()
        if pre.empty:
            continue

        for c in RAW_NUMERIC_COLS:
            pre[c] = pd.to_numeric(pre[c], errors="coerce")
        pre["listing_id"] = pd.to_numeric(pre["listing_id"], errors="coerce")
        pre["date"] = pd.to_datetime(pre["date"], errors="coerce")
        pre = pre.dropna(subset=["listing_id"]).copy()
        pre["listing_id"] = pre["listing_id"].astype("int64")

        # Existing static-ish means
        g_sum = pre.groupby("listing_id", sort=False)[RAW_NUMERIC_COLS].sum(min_count=1)
        g_count = pre.groupby("listing_id", sort=False)[RAW_NUMERIC_COLS].count()
        g_cat = pre.groupby("listing_id", sort=False)[RAW_CAT_COLS].first()

        sum_parts.append(g_sum)
        count_parts.append(g_count)
        cat_parts.append(g_cat)

        # Dynamic aggregates
        pre = pre.sort_values(["listing_id", "date"], kind="stable")
        for lid, g in pre.groupby("listing_id", sort=False):
            lp = pd.to_numeric(g["log_price"], errors="coerce")
            lp = lp[np.isfinite(lp)]
            if not lp.empty:
                var_count[lid] = var_count.get(lid, 0) + int(lp.size)
                var_sum[lid] = var_sum.get(lid, 0.0) + float(lp.sum())
                var_sumsq[lid] = var_sumsq.get(lid, 0.0) + float((lp * lp).sum())

            prices = pd.to_numeric(g["price_usd"], errors="coerce")
            dts = pd.to_datetime(g["date"], errors="coerce")
            is_wknd = dts.dt.weekday.isin([4, 5])

            wknd_vals = prices[is_wknd & prices.notna()]
            wkdy_vals = prices[(~is_wknd) & prices.notna()]
            if not wknd_vals.empty:
                wknd_sum[lid] = wknd_sum.get(lid, 0.0) + float(wknd_vals.sum())
                wknd_cnt[lid] = wknd_cnt.get(lid, 0) + int(wknd_vals.size)
            if not wkdy_vals.empty:
                wkdy_sum[lid] = wkdy_sum.get(lid, 0.0) + float(wkdy_vals.sum())
                wkdy_cnt[lid] = wkdy_cnt.get(lid, 0) + int(wkdy_vals.size)

            pvals = prices.to_numpy(dtype=np.float64)
            finite = np.isfinite(pvals)
            prev = last_price.get(lid, np.nan)
            for v, ok in zip(pvals, finite):
                if not ok:
                    continue
                if np.isfinite(prev):
                    trans_cnt[lid] = trans_cnt.get(lid, 0) + 1
                    if abs(v - prev) > 1e-12:
                        change_cnt[lid] = change_cnt.get(lid, 0) + 1
                prev = v
            if np.isfinite(prev):
                last_price[lid] = float(prev)

    if not sum_parts:
        raise ValueError("No pre-cutoff observations found for feature construction.")

    sum_df = pd.concat(sum_parts, axis=0).groupby(level=0).sum()
    count_df = pd.concat(count_parts, axis=0).groupby(level=0).sum()
    cat_df = pd.concat(cat_parts, axis=0).groupby(level=0).first()

    mean_df = sum_df.div(count_df.where(count_df > 0)).rename(columns={c: f"{c}_pre" for c in RAW_NUMERIC_COLS})
    out = mean_df.join(cat_df, how="left")
    out["pre_obs"] = count_df["available"]

    idx = out.index.astype("int64")
    vc = pd.Series({k: v for k, v in var_count.items()}, dtype="float64")
    vs = pd.Series({k: v for k, v in var_sum.items()}, dtype="float64")
    vss = pd.Series({k: v for k, v in var_sumsq.items()}, dtype="float64")

    n = vc.reindex(idx)
    s = vs.reindex(idx)
    ss = vss.reindex(idx)
    var = (ss - (s * s / n)) / (n - 1)
    var = var.where(n >= 2)

    wknd_mean = (pd.Series(wknd_sum).reindex(idx) / pd.Series(wknd_cnt).reindex(idx)).astype("float64")
    wkdy_mean = (pd.Series(wkdy_sum).reindex(idx) / pd.Series(wkdy_cnt).reindex(idx)).astype("float64")
    weekend_premium = wknd_mean - wkdy_mean

    pcf = (pd.Series(change_cnt).reindex(idx).fillna(0.0) / pd.Series(trans_cnt).reindex(idx)).astype("float64")

    out["price_variance_pre"] = var.to_numpy()
    out["weekend_premium_pre"] = weekend_premium.to_numpy()
    out["price_change_frequency_pre"] = pcf.to_numpy()

    # leak-safe imputation
    for c in ["price_variance_pre", "weekend_premium_pre", "price_change_frequency_pre"]:
        med = float(out[c].median()) if out[c].notna().any() else 0.0
        out[c] = out[c].fillna(med)

    out = out.reset_index()
    ordered_cols = [
        "listing_id",
        *RAW_CAT_COLS,
        "pre_obs",
        *[f"{c}_pre" for c in RAW_NUMERIC_COLS],
        "price_variance_pre",
        "weekend_premium_pre",
        "price_change_frequency_pre",
    ]
    return out[ordered_cols]


def fit_unsupervised_models(features: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    feature_df = features.copy()
    numeric_cols = [
        "pre_obs",
        "log_price_pre",
        "available_pre",
        "minimum_nights_pre",
        "maximum_nights_pre",
        "accommodates_pre",
        "bathrooms_pre",
        "bedrooms_pre",
        "host_tenure_days_pre",
        "host_is_superhost_pre",
        "host_identity_verified_pre",
        "host_response_rate_pre",
        "host_acceptance_rate_pre",
        "price_variance_pre",
        "weekend_premium_pre",
        "price_change_frequency_pre",
    ]
    cat_cols = ["city_slug", "room_type", "property_type"]

    for c in numeric_cols:
        feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce")
        med = float(feature_df[c].median()) if feature_df[c].notna().any() else 0.0
        feature_df[c] = feature_df[c].fillna(med)
    for c in cat_cols:
        feature_df[c] = feature_df[c].astype("string").fillna("missing")

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    x = pre.fit_transform(feature_df)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    kmeans_labels = kmeans.fit_predict(x)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", max_iter=300, n_init=3, reg_covar=1e-6, random_state=42)
    gmm.fit(x)
    gmm_labels = gmm.predict(x)
    gmm_probs = gmm.predict_proba(x)

    score_raw = (
        feature_df["host_response_rate_pre"].to_numpy(dtype=np.float64)
        + feature_df["host_acceptance_rate_pre"].to_numpy(dtype=np.float64)
        + feature_df["host_is_superhost_pre"].to_numpy(dtype=np.float64)
        + feature_df["host_identity_verified_pre"].to_numpy(dtype=np.float64)
        + np.log1p(np.clip(feature_df["accommodates_pre"].to_numpy(dtype=np.float64), 0.0, None))
        - np.log1p(np.clip(feature_df["minimum_nights_pre"].to_numpy(dtype=np.float64), 0.0, None))
        + np.log1p(np.clip(feature_df["price_variance_pre"].to_numpy(dtype=np.float64), 0.0, None))
        + np.clip(feature_df["price_change_frequency_pre"].to_numpy(dtype=np.float64), 0.0, 1.0)
    )

    cs = pd.DataFrame({"gmm_cluster": gmm_labels, "score_raw": score_raw}).groupby("gmm_cluster")["score_raw"].mean().sort_index().to_numpy()
    w = np.full_like(cs, 0.5) if np.allclose(np.nanmax(cs), np.nanmin(cs)) else (cs - np.nanmin(cs)) / (np.nanmax(cs) - np.nanmin(cs))

    latent = gmm_probs @ w
    latent = np.full_like(latent, 0.5) if np.allclose(np.nanmax(latent), np.nanmin(latent)) else (latent - np.nanmin(latent)) / (np.nanmax(latent) - np.nanmin(latent))

    out = feature_df[["listing_id", "city_slug", "room_type", "property_type"]].copy()
    out["kmeans_cluster"] = kmeans_labels.astype(int)
    out["gmm_cluster"] = gmm_labels.astype(int)
    for k in range(n_clusters):
        out[f"gmm_prob_cluster_{k}"] = gmm_probs[:, k]
    out["latent_adoption_propensity_proxy"] = latent
    out["gmm_bic"] = float(gmm.bic(x))
    out["kmeans_inertia"] = float(kmeans.inertia_)
    return out


def _prepare_common_arrays(chunk: pd.DataFrame, proxy_map: pd.Series, city_dummies: List[str]) -> Dict[str, np.ndarray]:
    listing_ids = pd.to_numeric(chunk["listing_id"], errors="coerce")
    latent_proxy = listing_ids.map(proxy_map)

    y_log_price = _clean_numeric(chunk["log_price"], default=np.nan)
    y_available = _clean_numeric(chunk["available"], default=np.nan)
    y_latent = _clean_numeric(latent_proxy, default=np.nan)

    post = np.clip(_clean_numeric(chunk["post_cutoff"], default=np.nan), 0.0, 1.0)
    running = _clean_numeric(chunk["days_from_cutoff"], default=np.nan) / 30.0
    post_running = post * running

    min_n = np.log1p(np.clip(_clean_numeric(chunk["minimum_nights"], default=0.0), 0.0, None))
    max_n = np.log1p(np.clip(_clean_numeric(chunk["maximum_nights"], default=0.0), 0.0, None))
    accom = np.clip(_clean_numeric(chunk["accommodates"], default=0.0), 0.0, 20.0)
    bath = np.clip(_clean_numeric(chunk["bathrooms"], default=0.0), 0.0, 15.0)
    bed = np.clip(_clean_numeric(chunk["bedrooms"], default=0.0), 0.0, 20.0)
    tenure = np.clip(_clean_numeric(chunk["host_tenure_days"], default=0.0) / 365.0, 0.0, 30.0)
    superhost = np.clip(_clean_numeric(chunk["host_is_superhost"], default=0.0), 0.0, 1.0)
    verified = np.clip(_clean_numeric(chunk["host_identity_verified"], default=0.0), 0.0, 1.0)

    city_vals = chunk["city_slug"].astype(str)
    city_cols = [(city_vals == c).to_numpy(dtype=np.float64) for c in city_dummies]
    controls = np.column_stack([running, post_running, min_n, max_n, accom, bath, bed, tenure, superhost, verified, *city_cols])

    return {
        "intercept": np.ones(len(chunk), dtype=np.float64),
        "controls": controls,
        "post": post,
        "available": y_available,
        "latent": y_latent,
        "log_price": y_log_price,
    }


def run_fuzzy_rdd_interaction_iv(input_file: Path, listing_proxy: pd.DataFrame, chunk_size: int = 1_000_000) -> pd.DataFrame:
    proxy_map = listing_proxy.set_index("listing_id")["latent_adoption_propensity_proxy"]
    proxy_fill = float(listing_proxy["latent_adoption_propensity_proxy"].median())
    city_dummies = [c for c in CITY_ORDER if c != CITY_ORDER[0]]

    fs_X_parts, fs_y_parts = [], []
    reader = pd.read_csv(input_file, compression="gzip", usecols=MODEL_USECOLS, chunksize=chunk_size, low_memory=False)
    for chunk in reader:
        a = _prepare_common_arrays(chunk, proxy_map, city_dummies)
        latent = np.nan_to_num(a["latent"], nan=proxy_fill, posinf=proxy_fill, neginf=proxy_fill)
        post_latent = a["post"] * latent
        x = np.column_stack([a["intercept"], a["post"], latent, post_latent, a["controls"]])
        y = a["available"]
        m = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if m.any():
            fs_X_parts.append(x[m])
            fs_y_parts.append(y[m])

    fs_X = np.vstack(fs_X_parts)
    fs_y = np.concatenate(fs_y_parts)
    control_names = ["running", "post_running", "log1p_min_nights", "log1p_max_nights", "accommodates", "bathrooms", "bedrooms", "tenure_years", "superhost", "verified", *[f"city_{c}" for c in city_dummies]]
    fs_names = ["const", "post_cutoff", "latent_proxy", "post_x_latent", *control_names]
    fs_res = _ols_hc1(fs_X, fs_y, fs_names)
    beta_fs = fs_res.set_index("term")["coef"].to_dict()

    ss_X_parts, ss_y_parts = [], []
    reader = pd.read_csv(input_file, compression="gzip", usecols=MODEL_USECOLS, chunksize=chunk_size, low_memory=False)
    for chunk in reader:
        a = _prepare_common_arrays(chunk, proxy_map, city_dummies)
        latent = np.nan_to_num(a["latent"], nan=proxy_fill, posinf=proxy_fill, neginf=proxy_fill)
        post_latent = a["post"] * latent
        fs_x = np.column_stack([a["intercept"], a["post"], latent, post_latent, a["controls"]])
        avail_hat = fs_x @ np.array([beta_fs[t] for t in fs_names], dtype=np.float64)

        x2 = np.column_stack([a["intercept"], avail_hat, latent, avail_hat * latent, a["controls"]])
        y2 = a["log_price"]
        m2 = np.isfinite(y2) & np.isfinite(x2).all(axis=1)
        if m2.any():
            ss_X_parts.append(x2[m2])
            ss_y_parts.append(y2[m2])

    ss_X = np.vstack(ss_X_parts)
    ss_y = np.concatenate(ss_y_parts)
    ss_names = ["const", "available_hat", "latent_proxy", "available_hat_x_latent", *control_names]
    ss_res = _ols_hc1(ss_X, ss_y, ss_names)

    fs_out = fs_res.assign(stage="first_stage")
    ss_out = ss_res.assign(stage="second_stage")
    return pd.concat([fs_out, ss_out], ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refined ML-econometrics extension")
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument("--input-file", type=str, default="data/processed/step2/fact_listing_day_multicity_bw_3m.csv.gz")
    p.add_argument("--chunk-size", type=int, default=1_000_000)
    p.add_argument("--clusters", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    input_file = (repo / args.input_file).resolve()
    output_dir = repo / "data" / "processed" / "ml_extension"
    _ensure_dirs([output_dir])

    if not input_file.exists():
        raise FileNotFoundError(f"Input panel not found: {input_file}")

    print("[ML-EXT] Building refined pre-cutoff listing features...")
    listing_features = build_pre_cutoff_listing_features(input_file=input_file, chunk_size=args.chunk_size)

    print("[ML-EXT] Running KMeans + GMM with dynamic features...")
    cluster_df = fit_unsupervised_models(listing_features, n_clusters=args.clusters)
    listing_proxy = cluster_df[["listing_id", "city_slug", "latent_adoption_propensity_proxy"]].copy()

    print("[ML-EXT] Running heterogeneous fuzzy-RDD/IV interaction model...")
    iv_interaction = run_fuzzy_rdd_interaction_iv(input_file=input_file, listing_proxy=listing_proxy, chunk_size=args.chunk_size)

    listing_features.to_csv(output_dir / "listing_features_refined.csv", index=False)
    cluster_df.sort_values("listing_id").to_csv(output_dir / "listing_cluster_membership.csv", index=False)
    listing_proxy.sort_values("listing_id").to_csv(output_dir / "listing_latent_proxy.csv", index=False)
    iv_interaction.to_csv(output_dir / "heterogeneous_iv_interaction_results.csv", index=False)

    summary = {
        "input_file": str(input_file.relative_to(repo)),
        "clusters": int(args.clusters),
        "n_listings_proxy": int(len(listing_proxy)),
        "proxy_distribution": {
            "min": float(listing_proxy["latent_adoption_propensity_proxy"].min()),
            "mean": float(listing_proxy["latent_adoption_propensity_proxy"].mean()),
            "max": float(listing_proxy["latent_adoption_propensity_proxy"].max()),
        },
        "dynamic_features": ["price_variance_pre", "weekend_premium_pre", "price_change_frequency_pre"],
        "outputs": {
            "listing_features_refined": "data/processed/ml_extension/listing_features_refined.csv",
            "listing_cluster_membership": "data/processed/ml_extension/listing_cluster_membership.csv",
            "listing_latent_proxy": "data/processed/ml_extension/listing_latent_proxy.csv",
            "heterogeneous_iv_interaction_results": "data/processed/ml_extension/heterogeneous_iv_interaction_results.csv",
        },
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
