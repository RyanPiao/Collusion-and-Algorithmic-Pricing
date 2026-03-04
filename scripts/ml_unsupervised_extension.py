#!/usr/bin/env python3
"""ML-econometrics extension: latent adoption propensity proxy via unsupervised learning.

This script builds listing-level pre-cutoff features, fits unsupervised models
(KMeans + Gaussian Mixture), constructs a latent adoption propensity proxy, and
compares simple econometric specifications against the baseline availability
proxy in the Day 2 multicity panel.

Outputs are written to data/processed/ml_extension/.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:  # pragma: no cover - explicit runtime guard
    raise SystemExit(
        "Missing required dependencies. Install in a virtual environment, e.g.\n"
        "python3 -m venv .venv && source .venv/bin/activate && "
        "pip install numpy pandas scikit-learn scipy statsmodels"
    ) from exc


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


@dataclass
class OLSResult:
    model_name: str
    outcome: str
    treatment: str
    n_obs: int
    k_params: int
    coef_treatment: float
    se_hc1_treatment: float
    t_hc1_treatment: float
    p_hc1_treatment: float
    ci95_low: float
    ci95_high: float
    r2: float


def _normal_two_sided_p(t: float) -> float:
    z = abs(float(t))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, 2.0 * (1.0 - cdf))))


def _safe_inv(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.linalg.pinv(x, rcond=1e-10)


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _clean_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    arr = np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
    return arr


class OnePassOLS:
    """Two-pass OLS with HC1 robust variance, chunk-friendly."""

    def __init__(self, model_name: str, outcome: str, treatment: str):
        self.model_name = model_name
        self.outcome = outcome
        self.treatment = treatment

        self.n = 0
        self.k = None

        self.xtx = None
        self.xty = None

        self.beta = None
        self.inv_xtx = None

        self.meat = None
        self.sse = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0

    def pass1_update(self, x: np.ndarray, y: np.ndarray) -> None:
        if y.size == 0:
            return

        if self.k is None:
            self.k = x.shape[1]
            self.xtx = np.zeros((self.k, self.k), dtype=np.float64)
            self.xty = np.zeros(self.k, dtype=np.float64)

        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.n += int(y.size)
        self.sum_y += float(np.sum(y))
        self.sum_y2 += float(np.sum(y * y))

    def finalize_pass1(self) -> None:
        if self.n == 0 or self.k is None:
            raise ValueError(f"No observations for model {self.model_name}")
        self.inv_xtx = _safe_inv(self.xtx)
        self.beta = self.inv_xtx @ self.xty
        self.meat = np.zeros((self.k, self.k), dtype=np.float64)

    def pass2_update(self, x: np.ndarray, y: np.ndarray) -> None:
        if y.size == 0:
            return
        resid = y - (x @ self.beta)
        self.sse += float(np.sum(resid * resid))
        self.meat += x.T @ (x * (resid[:, None] ** 2))

    def result(self) -> OLSResult:
        if self.n == 0:
            raise ValueError(f"No observations for model {self.model_name}")

        hc1_scale = self.n / max(self.n - self.k, 1)
        cov_hc1 = hc1_scale * (self.inv_xtx @ self.meat @ self.inv_xtx)
        se = np.sqrt(np.clip(np.diag(cov_hc1), 0.0, np.inf))

        # treatment is second column after intercept by construction
        coef_treat = float(self.beta[1])
        se_treat = float(se[1]) if np.isfinite(se[1]) else np.nan
        t_treat = coef_treat / se_treat if se_treat and np.isfinite(se_treat) else np.nan
        p_treat = _normal_two_sided_p(t_treat) if np.isfinite(t_treat) else np.nan
        ci_low = coef_treat - 1.96 * se_treat if np.isfinite(se_treat) else np.nan
        ci_high = coef_treat + 1.96 * se_treat if np.isfinite(se_treat) else np.nan

        y_mean = self.sum_y / self.n
        tss = self.sum_y2 - self.n * y_mean * y_mean
        r2 = 1.0 - (self.sse / tss) if tss > 1e-12 else np.nan

        return OLSResult(
            model_name=self.model_name,
            outcome=self.outcome,
            treatment=self.treatment,
            n_obs=self.n,
            k_params=self.k,
            coef_treatment=coef_treat,
            se_hc1_treatment=se_treat,
            t_hc1_treatment=t_treat,
            p_hc1_treatment=p_treat,
            ci95_low=ci_low,
            ci95_high=ci_high,
            r2=float(r2) if np.isfinite(r2) else np.nan,
        )


def build_pre_cutoff_listing_features(input_file: Path, chunk_size: int = 1_000_000) -> pd.DataFrame:
    usecols = ["listing_id", "post_cutoff", *RAW_NUMERIC_COLS, *RAW_CAT_COLS]

    sum_parts: List[pd.DataFrame] = []
    count_parts: List[pd.DataFrame] = []
    cat_parts: List[pd.DataFrame] = []

    reader = pd.read_csv(
        input_file,
        compression="gzip",
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        pre = chunk.loc[chunk["post_cutoff"] == 0, ["listing_id", *RAW_NUMERIC_COLS, *RAW_CAT_COLS]].copy()
        if pre.empty:
            continue

        for c in RAW_NUMERIC_COLS:
            pre[c] = pd.to_numeric(pre[c], errors="coerce")

        g_sum = pre.groupby("listing_id", sort=False)[RAW_NUMERIC_COLS].sum(min_count=1)
        g_count = pre.groupby("listing_id", sort=False)[RAW_NUMERIC_COLS].count()
        g_cat = pre.groupby("listing_id", sort=False)[RAW_CAT_COLS].first()

        sum_parts.append(g_sum)
        count_parts.append(g_count)
        cat_parts.append(g_cat)

    if not sum_parts:
        raise ValueError("No pre-cutoff observations found for feature construction.")

    sum_df = pd.concat(sum_parts, axis=0).groupby(level=0).sum()
    count_df = pd.concat(count_parts, axis=0).groupby(level=0).sum()
    cat_df = pd.concat(cat_parts, axis=0).groupby(level=0).first()

    mean_df = sum_df.div(count_df.where(count_df > 0))
    mean_df = mean_df.rename(columns={c: f"{c}_pre" for c in RAW_NUMERIC_COLS})

    out = mean_df.join(cat_df, how="left")
    out["pre_obs"] = count_df["available"]
    out = out.reset_index()

    # deterministic column order
    ordered_cols = ["listing_id", *RAW_CAT_COLS, "pre_obs", *[f"{c}_pre" for c in RAW_NUMERIC_COLS]]
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
    ]
    categorical_cols = ["city_slug", "room_type", "property_type"]

    for c in numeric_cols:
        feature_df[c] = pd.to_numeric(feature_df[c], errors="coerce")
        med = float(feature_df[c].median()) if feature_df[c].notna().any() else 0.0
        feature_df[c] = feature_df[c].fillna(med)

    for c in categorical_cols:
        feature_df[c] = feature_df[c].astype("string").fillna("missing")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    x = preprocessor.fit_transform(feature_df)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    kmeans_labels = kmeans.fit_predict(x)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        max_iter=300,
        n_init=3,
        reg_covar=1e-6,
        random_state=42,
    )
    gmm.fit(x)
    gmm_labels = gmm.predict(x)
    gmm_probs = gmm.predict_proba(x)

    # Construct latent adoption propensity proxy from posterior probabilities,
    # weighting clusters by a pre-cutoff host/listing sophistication index.
    score_raw = (
        feature_df["host_response_rate_pre"].to_numpy(dtype=np.float64)
        + feature_df["host_acceptance_rate_pre"].to_numpy(dtype=np.float64)
        + feature_df["host_is_superhost_pre"].to_numpy(dtype=np.float64)
        + feature_df["host_identity_verified_pre"].to_numpy(dtype=np.float64)
        + np.log1p(np.clip(feature_df["accommodates_pre"].to_numpy(dtype=np.float64), 0.0, None))
        - np.log1p(np.clip(feature_df["minimum_nights_pre"].to_numpy(dtype=np.float64), 0.0, None))
    )

    score_df = pd.DataFrame({"gmm_cluster": gmm_labels, "score_raw": score_raw})
    cluster_scores = score_df.groupby("gmm_cluster", as_index=True)["score_raw"].mean().sort_index()

    cs = cluster_scores.to_numpy(dtype=np.float64)
    if np.allclose(np.nanmax(cs), np.nanmin(cs)):
        cluster_weights = np.full_like(cs, 0.5, dtype=np.float64)
    else:
        cluster_weights = (cs - np.nanmin(cs)) / (np.nanmax(cs) - np.nanmin(cs))

    latent_proxy = gmm_probs @ cluster_weights
    if np.nanmax(latent_proxy) > np.nanmin(latent_proxy):
        latent_proxy = (latent_proxy - np.nanmin(latent_proxy)) / (
            np.nanmax(latent_proxy) - np.nanmin(latent_proxy)
        )
    else:
        latent_proxy = np.full_like(latent_proxy, 0.5)

    out = feature_df[["listing_id", "city_slug", "room_type", "property_type"]].copy()
    out["kmeans_cluster"] = kmeans_labels.astype(int)
    out["gmm_cluster"] = gmm_labels.astype(int)

    for k in range(n_clusters):
        out[f"gmm_prob_cluster_{k}"] = gmm_probs[:, k]

    out["latent_adoption_propensity_proxy"] = latent_proxy
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

    controls = np.column_stack(
        [running, post_running, min_n, max_n, accom, bath, bed, tenure, superhost, verified, *city_cols]
    )
    intercept = np.ones(len(chunk), dtype=np.float64)

    return {
        "intercept": intercept,
        "controls": controls,
        "post": post,
        "available": y_available,
        "latent": y_latent,
        "log_price": y_log_price,
    }


def _stack_x(intercept: np.ndarray, treatment: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return np.column_stack([intercept, treatment, controls])


def run_ols_comparisons(
    input_file: Path,
    listing_proxy: pd.DataFrame,
    chunk_size: int = 1_000_000,
) -> Dict[str, OLSResult]:
    proxy_map = listing_proxy.set_index("listing_id")["latent_adoption_propensity_proxy"]
    proxy_fill = float(listing_proxy["latent_adoption_propensity_proxy"].median())

    city_dummies = [c for c in CITY_ORDER if c != CITY_ORDER[0]]

    models = {
        "first_stage_baseline": OnePassOLS(
            model_name="first_stage_baseline",
            outcome="available",
            treatment="post_cutoff",
        ),
        "first_stage_ml": OnePassOLS(
            model_name="first_stage_ml",
            outcome="latent_adoption_propensity_proxy",
            treatment="post_cutoff",
        ),
        "second_stage_baseline": OnePassOLS(
            model_name="second_stage_baseline",
            outcome="log_price",
            treatment="available",
        ),
        "second_stage_ml": OnePassOLS(
            model_name="second_stage_ml",
            outcome="log_price",
            treatment="latent_adoption_propensity_proxy",
        ),
    }

    # PASS 1
    reader = pd.read_csv(
        input_file,
        compression="gzip",
        usecols=MODEL_USECOLS,
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk in reader:
        arrays = _prepare_common_arrays(chunk, proxy_map=proxy_map, city_dummies=city_dummies)
        arrays["latent"] = np.nan_to_num(arrays["latent"], nan=proxy_fill, posinf=proxy_fill, neginf=proxy_fill)

        specs = [
            ("first_stage_baseline", arrays["available"], arrays["post"]),
            ("first_stage_ml", arrays["latent"], arrays["post"]),
            ("second_stage_baseline", arrays["log_price"], arrays["available"]),
            ("second_stage_ml", arrays["log_price"], arrays["latent"]),
        ]

        for model_key, y, t in specs:
            x = _stack_x(arrays["intercept"], t, arrays["controls"])
            valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
            models[model_key].pass1_update(x[valid], y[valid])

    for m in models.values():
        m.finalize_pass1()

    # PASS 2
    reader = pd.read_csv(
        input_file,
        compression="gzip",
        usecols=MODEL_USECOLS,
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk in reader:
        arrays = _prepare_common_arrays(chunk, proxy_map=proxy_map, city_dummies=city_dummies)
        arrays["latent"] = np.nan_to_num(arrays["latent"], nan=proxy_fill, posinf=proxy_fill, neginf=proxy_fill)

        specs = [
            ("first_stage_baseline", arrays["available"], arrays["post"]),
            ("first_stage_ml", arrays["latent"], arrays["post"]),
            ("second_stage_baseline", arrays["log_price"], arrays["available"]),
            ("second_stage_ml", arrays["log_price"], arrays["latent"]),
        ]

        for model_key, y, t in specs:
            x = _stack_x(arrays["intercept"], t, arrays["controls"])
            valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
            models[model_key].pass2_update(x[valid], y[valid])

    return {k: v.result() for k, v in models.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML-econometrics unsupervised extension")
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument(
        "--input-file",
        type=str,
        default="data/processed/day2/fact_listing_day_multicity_bw_3m.csv.gz",
    )
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

    print("[ML-EXT] Building pre-cutoff listing-level feature matrix...")
    listing_features = build_pre_cutoff_listing_features(input_file=input_file, chunk_size=args.chunk_size)

    print("[ML-EXT] Running unsupervised models (KMeans + GMM)...")
    cluster_df = fit_unsupervised_models(listing_features, n_clusters=args.clusters)

    listing_proxy = cluster_df[["listing_id", "city_slug", "latent_adoption_propensity_proxy"]].copy()

    print("[ML-EXT] Running first-stage/second-stage comparison regressions...")
    results = run_ols_comparisons(input_file=input_file, listing_proxy=listing_proxy, chunk_size=args.chunk_size)

    first_stage_rows = []
    for key in ["first_stage_baseline", "first_stage_ml"]:
        r = results[key]
        first_stage_rows.append(
            {
                "model_name": r.model_name,
                "outcome": r.outcome,
                "treatment": r.treatment,
                "n_obs": r.n_obs,
                "k_params": r.k_params,
                "coef_treatment": r.coef_treatment,
                "se_hc1_treatment": r.se_hc1_treatment,
                "t_hc1_treatment": r.t_hc1_treatment,
                "p_hc1_treatment": r.p_hc1_treatment,
                "ci95_low": r.ci95_low,
                "ci95_high": r.ci95_high,
                "r2": r.r2,
            }
        )

    second_stage_rows = []
    for key in ["second_stage_baseline", "second_stage_ml"]:
        r = results[key]
        second_stage_rows.append(
            {
                "model_name": r.model_name,
                "outcome": r.outcome,
                "treatment": r.treatment,
                "n_obs": r.n_obs,
                "k_params": r.k_params,
                "coef_treatment": r.coef_treatment,
                "se_hc1_treatment": r.se_hc1_treatment,
                "t_hc1_treatment": r.t_hc1_treatment,
                "p_hc1_treatment": r.p_hc1_treatment,
                "ci95_low": r.ci95_low,
                "ci95_high": r.ci95_high,
                "r2": r.r2,
            }
        )

    listing_proxy_path = output_dir / "listing_latent_proxy.csv"
    cluster_path = output_dir / "listing_cluster_membership.csv"
    first_stage_path = output_dir / "first_stage_comparison.csv"
    second_stage_path = output_dir / "second_stage_comparison.csv"
    summary_path = output_dir / "run_summary.json"

    listing_proxy.sort_values("listing_id").to_csv(listing_proxy_path, index=False)
    cluster_df.sort_values("listing_id").to_csv(cluster_path, index=False)
    pd.DataFrame(first_stage_rows).to_csv(first_stage_path, index=False)
    pd.DataFrame(second_stage_rows).to_csv(second_stage_path, index=False)

    summary = {
        "input_file": str(input_file.relative_to(repo)),
        "n_listings_proxy": int(len(listing_proxy)),
        "proxy_min": float(listing_proxy["latent_adoption_propensity_proxy"].min()),
        "proxy_mean": float(listing_proxy["latent_adoption_propensity_proxy"].mean()),
        "proxy_max": float(listing_proxy["latent_adoption_propensity_proxy"].max()),
        "clusters": int(args.clusters),
        "outputs": {
            "listing_latent_proxy": str(listing_proxy_path.relative_to(repo)),
            "listing_cluster_membership": str(cluster_path.relative_to(repo)),
            "first_stage_comparison": str(first_stage_path.relative_to(repo)),
            "second_stage_comparison": str(second_stage_path.relative_to(repo)),
        },
        "first_stage_models": first_stage_rows,
        "second_stage_models": second_stage_rows,
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
