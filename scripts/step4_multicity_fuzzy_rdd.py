#!/usr/bin/env python3
"""Step 4 multicity baseline fuzzy-RDD modeling.

Baseline implementation (scalable one-pass IV per sample):
- Endogenous treatment proxy: listing-step availability (available)
- Excluded instrument: post-cutoff indicator
- Running-variable controls: local linear terms in days-from-cutoff
- Additional listing/host controls and pooled city fixed effects

Produces Step 4 required outputs under data/processed/step4/.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


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

BASE_CONTROL_COLS = [
    "minimum_nights",
    "maximum_nights",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "host_tenure_days",
    "host_is_superhost",
    "host_identity_verified",
]

USECOLS = [
    "city_slug",
    "log_price",
    "available",
    "post_cutoff",
    "days_from_cutoff",
    *BASE_CONTROL_COLS,
]


@dataclass
class ModelResult:
    n_obs: int
    first_stage_coef_post: float
    first_stage_se_post: float
    first_stage_t_post: float
    first_stage_f_post: float
    first_stage_r2: float
    first_stage_available_pre: float
    first_stage_available_post: float
    second_stage_coef_available_hat: float
    second_stage_se_available_hat: float
    second_stage_t_available_hat: float
    second_stage_p_available_hat: float
    second_stage_ci95_low: float
    second_stage_ci95_high: float


def _normal_two_sided_p(t_value: float) -> float:
    z = abs(float(t_value))
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, 2.0 * (1.0 - cdf))))


def _safe_inv(m: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    return np.linalg.pinv(m, rcond=1e-10)


def _safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = a @ b
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class OnePassIVEstimator:
    """One-pass first-stage OLS and 2SLS with homoskedastic SEs."""

    def __init__(self, k_w: int):
        self.k_w = k_w
        self.k_z = k_w + 1  # [W, excluded Z=post]
        self.k_x = k_w + 1  # [W, endogenous D]

        self.n = 0

        self.S_ZZ = np.zeros((self.k_z, self.k_z), dtype=np.float64)
        self.S_ZD = np.zeros(self.k_z, dtype=np.float64)
        self.S_ZY = np.zeros(self.k_z, dtype=np.float64)
        self.S_ZX = np.zeros((self.k_z, self.k_x), dtype=np.float64)

        self.S_XX = np.zeros((self.k_x, self.k_x), dtype=np.float64)
        self.S_XY = np.zeros(self.k_x, dtype=np.float64)

        self.sum_d = 0.0
        self.sum_d2 = 0.0
        self.sum_y2 = 0.0

        self.sum_d_pre = 0.0
        self.sum_d_post = 0.0
        self.n_pre = 0
        self.n_post = 0

    def update(self, y: np.ndarray, d: np.ndarray, w: np.ndarray, z_excl: np.ndarray) -> None:
        if y.size == 0:
            return

        z = np.column_stack([w, z_excl])
        x = np.column_stack([w, d])

        self.S_ZZ += _safe_matmul(z.T, z)
        self.S_ZD += _safe_matmul(z.T, d)
        self.S_ZY += _safe_matmul(z.T, y)
        self.S_ZX += _safe_matmul(z.T, x)

        self.S_XX += _safe_matmul(x.T, x)
        self.S_XY += _safe_matmul(x.T, y)

        self.n += int(y.size)
        self.sum_d += float(np.sum(d))
        self.sum_d2 += float(np.sum(d * d))
        self.sum_y2 += float(np.sum(y * y))

        pre_mask = z_excl < 0.5
        post_mask = ~pre_mask
        if pre_mask.any():
            self.sum_d_pre += float(np.sum(d[pre_mask]))
            self.n_pre += int(np.sum(pre_mask))
        if post_mask.any():
            self.sum_d_post += float(np.sum(d[post_mask]))
            self.n_post += int(np.sum(post_mask))

    def result(self) -> ModelResult:
        if self.n == 0:
            raise ValueError("No observations in estimator")

        inv_ZZ = _safe_inv(self.S_ZZ)

        # First stage: D ~ [W, Z_excluded]
        fs_beta = _safe_matmul(inv_ZZ, self.S_ZD)
        fs_sse = self.sum_d2 - float(np.dot(fs_beta, self.S_ZD))
        fs_df = max(self.n - self.k_z, 1)
        fs_sigma2 = max(fs_sse / fs_df, 0.0)
        fs_cov = fs_sigma2 * inv_ZZ

        fs_se_all = np.sqrt(np.clip(np.diag(fs_cov), 0.0, np.inf))
        fs_idx = self.k_z - 1
        fs_coef_post = float(fs_beta[fs_idx])
        fs_se_post = float(fs_se_all[fs_idx]) if np.isfinite(fs_se_all[fs_idx]) else np.nan
        fs_t_post = fs_coef_post / fs_se_post if fs_se_post and np.isfinite(fs_se_post) else np.nan
        fs_f_post = fs_t_post * fs_t_post if np.isfinite(fs_t_post) else np.nan

        d_mean = self.sum_d / self.n
        fs_sst = self.sum_d2 - self.n * d_mean * d_mean
        fs_r2 = 1.0 - (fs_sse / fs_sst) if fs_sst > 1e-12 else np.nan

        # 2SLS: Y ~ [W, D], D instrumented by [W, Z_excluded]
        S_XZ = self.S_ZX.T
        A = _safe_matmul(_safe_matmul(S_XZ, inv_ZZ), self.S_ZX)
        B = _safe_inv(A)
        b = _safe_matmul(_safe_matmul(S_XZ, inv_ZZ), self.S_ZY)
        iv_beta = _safe_matmul(B, b)

        # Homoskedastic IV variance: sigma^2 * (X'PzX)^-1
        iv_sse = (
            self.sum_y2
            - 2.0 * float(np.dot(iv_beta, self.S_XY))
            + float(np.dot(iv_beta, _safe_matmul(self.S_XX, iv_beta)))
        )
        iv_df = max(self.n - self.k_x, 1)
        iv_sigma2 = max(iv_sse / iv_df, 0.0)
        iv_cov = iv_sigma2 * B

        iv_se_all = np.sqrt(np.clip(np.diag(iv_cov), 0.0, np.inf))
        iv_idx = self.k_x - 1

        coef = float(iv_beta[iv_idx])
        se = float(iv_se_all[iv_idx]) if np.isfinite(iv_se_all[iv_idx]) else np.nan
        t = coef / se if se and np.isfinite(se) else np.nan
        p = _normal_two_sided_p(t) if np.isfinite(t) else np.nan
        ci_low = coef - 1.96 * se if np.isfinite(se) else np.nan
        ci_high = coef + 1.96 * se if np.isfinite(se) else np.nan

        mean_pre = self.sum_d_pre / self.n_pre if self.n_pre else np.nan
        mean_post = self.sum_d_post / self.n_post if self.n_post else np.nan

        return ModelResult(
            n_obs=self.n,
            first_stage_coef_post=fs_coef_post,
            first_stage_se_post=fs_se_post,
            first_stage_t_post=fs_t_post,
            first_stage_f_post=fs_f_post,
            first_stage_r2=float(fs_r2) if np.isfinite(fs_r2) else np.nan,
            first_stage_available_pre=float(mean_pre) if np.isfinite(mean_pre) else np.nan,
            first_stage_available_post=float(mean_post) if np.isfinite(mean_post) else np.nan,
            second_stage_coef_available_hat=coef,
            second_stage_se_available_hat=se,
            second_stage_t_available_hat=t,
            second_stage_p_available_hat=p,
            second_stage_ci95_low=ci_low,
            second_stage_ci95_high=ci_high,
        )


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _clean_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    arr = np.nan_to_num(arr, nan=default, posinf=default, neginf=default)
    return arr


def build_w_matrix(df: pd.DataFrame, pooled: bool, city_dummies: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = _clean_numeric(df["log_price"], default=np.nan)
    d = _clean_numeric(df["available"], default=np.nan)
    post = _clean_numeric(df["post_cutoff"], default=np.nan)
    r = _clean_numeric(df["days_from_cutoff"], default=np.nan) / 30.0

    min_n = np.log1p(np.clip(_clean_numeric(df["minimum_nights"], default=0.0), 0, None))
    max_n = np.log1p(np.clip(_clean_numeric(df["maximum_nights"], default=0.0), 0, None))
    # Cap at practical range to prevent undue leverage from extreme outliers.
    min_n = np.clip(min_n, 0.0, 8.0)
    max_n = np.clip(max_n, 0.0, 8.0)

    accom = np.clip(_clean_numeric(df["accommodates"], default=0.0), 0.0, 20.0)
    bath = np.clip(_clean_numeric(df["bathrooms"], default=0.0), 0.0, 15.0)
    bed = np.clip(_clean_numeric(df["bedrooms"], default=0.0), 0.0, 20.0)
    tenure_yrs = np.clip(_clean_numeric(df["host_tenure_days"], default=0.0) / 365.0, 0.0, 30.0)
    superhost = np.clip(_clean_numeric(df["host_is_superhost"], default=0.0), 0.0, 1.0)
    verified = np.clip(_clean_numeric(df["host_identity_verified"], default=0.0), 0.0, 1.0)

    pieces = [
        np.ones_like(y),
        r,
        post * r,
        min_n,
        max_n,
        accom,
        bath,
        bed,
        tenure_yrs,
        superhost,
        verified,
    ]

    if pooled:
        city_vals = df["city_slug"].astype(str)
        for c in city_dummies:
            pieces.append((city_vals == c).to_numpy(dtype=np.float64))

    w = np.column_stack(pieces)

    valid = np.isfinite(y) & np.isfinite(d) & np.isfinite(post)
    if not valid.all():
        y = y[valid]
        d = d[valid]
        post = post[valid]
        w = w[valid, :]

    return y, d, post, w


def model_result_to_row(city_slug: str, window_months: int, r: ModelResult) -> Dict:
    return {
        "city_slug": city_slug,
        "window_months": window_months,
        "n_obs": r.n_obs,
        "first_stage_coef_post": r.first_stage_coef_post,
        "first_stage_se_post": r.first_stage_se_post,
        "first_stage_t_post": r.first_stage_t_post,
        "first_stage_f_post": r.first_stage_f_post,
        "first_stage_r2": r.first_stage_r2,
        "first_stage_available_pre": r.first_stage_available_pre,
        "first_stage_available_post": r.first_stage_available_post,
        "second_stage_coef_available_hat": r.second_stage_coef_available_hat,
        "second_stage_se_available_hat": r.second_stage_se_available_hat,
        "second_stage_t_available_hat": r.second_stage_t_available_hat,
        "second_stage_p_available_hat": r.second_stage_p_available_hat,
        "second_stage_ci95_low": r.second_stage_ci95_low,
        "second_stage_ci95_high": r.second_stage_ci95_high,
    }


def run_window_models(window_file: Path, window_months: int, chunk_size: int = 500_000) -> Tuple[List[Dict], List[Dict]]:
    city_dummies = [c for c in CITY_ORDER if c != CITY_ORDER[0]]

    pooled_est = OnePassIVEstimator(k_w=11 + len(city_dummies))
    city_est = {c: OnePassIVEstimator(k_w=11) for c in CITY_ORDER}

    reader = pd.read_csv(
        window_file,
        compression="gzip",
        usecols=USECOLS,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        y, d, post, w = build_w_matrix(chunk, pooled=True, city_dummies=city_dummies)
        pooled_est.update(y=y, d=d, w=w, z_excl=post)

        for city, g in chunk.groupby("city_slug", sort=False):
            if city not in city_est:
                continue
            y_c, d_c, post_c, w_c = build_w_matrix(g, pooled=False, city_dummies=[])
            city_est[city].update(y=y_c, d=d_c, w=w_c, z_excl=post_c)

    pooled_result = pooled_est.result()
    pooled_rows = [model_result_to_row("pooled", window_months, pooled_result)]

    city_rows = []
    for c in CITY_ORDER:
        city_rows.append(model_result_to_row(c, window_months, city_est[c].result()))

    return pooled_rows, city_rows


def run_placebo_pooled(window_file: Path, offsets_days: List[int], chunk_size: int = 500_000) -> pd.DataFrame:
    city_dummies = [c for c in CITY_ORDER if c != CITY_ORDER[0]]
    rows = []

    for offset in offsets_days:
        est = OnePassIVEstimator(k_w=11 + len(city_dummies))
        reader = pd.read_csv(
            window_file,
            compression="gzip",
            usecols=USECOLS,
            chunksize=chunk_size,
            low_memory=False,
        )

        for chunk in reader:
            days = _clean_numeric(chunk["days_from_cutoff"], default=np.nan)
            chunk = chunk.copy()
            chunk["days_from_cutoff"] = days - offset
            chunk["post_cutoff"] = (chunk["days_from_cutoff"] >= 0).astype(int)

            y, d, post, w = build_w_matrix(chunk, pooled=True, city_dummies=city_dummies)
            est.update(y=y, d=d, w=w, z_excl=post)

        r = est.result()
        rows.append(
            {
                "placebo_label": f"placebo_{offset:+d}d",
                "placebo_offset_days": offset,
                "n_obs": r.n_obs,
                "first_stage_f_post": r.first_stage_f_post,
                "second_stage_coef_available_hat": r.second_stage_coef_available_hat,
                "second_stage_se_available_hat": r.second_stage_se_available_hat,
                "second_stage_t_available_hat": r.second_stage_t_available_hat,
                "second_stage_p_available_hat": r.second_stage_p_available_hat,
                "second_stage_ci95_low": r.second_stage_ci95_low,
                "second_stage_ci95_high": r.second_stage_ci95_high,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4 multicity fuzzy-RDD baseline")
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument("--chunk-size", type=int, default=500_000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).resolve()

    day2_proc = repo / "data" / "processed" / "step2"
    day4_proc = repo / "data" / "processed" / "step4"
    _ensure_dirs([day4_proc])

    pooled_rows_all: List[Dict] = []
    city_rows_all: List[Dict] = []

    for m in (1, 2, 3):
        window_file = day2_proc / f"fact_listing_day_multicity_bw_{m}m.csv.gz"
        print(f"[Step4] Running baseline fuzzy-RDD for ±{m}m window...")
        pooled_rows, city_rows = run_window_models(window_file=window_file, window_months=m, chunk_size=args.chunk_size)
        pooled_rows_all.extend(pooled_rows)
        city_rows_all.extend(city_rows)

    pooled_df = pd.DataFrame(pooled_rows_all).sort_values("window_months").reset_index(drop=True)
    city_df = pd.DataFrame(city_rows_all).sort_values(["city_slug", "window_months"]).reset_index(drop=True)

    first_stage_cols = [
        "city_slug",
        "window_months",
        "n_obs",
        "first_stage_available_pre",
        "first_stage_available_post",
        "first_stage_coef_post",
        "first_stage_se_post",
        "first_stage_t_post",
        "first_stage_f_post",
        "first_stage_r2",
    ]
    first_stage_df = pd.concat([pooled_df, city_df], ignore_index=True)
    first_stage_df[first_stage_cols].to_csv(day4_proc / "first_stage_strength_city_window.csv", index=False)

    pooled_second_cols = [
        "window_months",
        "n_obs",
        "second_stage_coef_available_hat",
        "second_stage_se_available_hat",
        "second_stage_t_available_hat",
        "second_stage_p_available_hat",
        "second_stage_ci95_low",
        "second_stage_ci95_high",
    ]
    pooled_df[pooled_second_cols].to_csv(day4_proc / "second_stage_pooled_window_estimates.csv", index=False)

    city_second_cols = [
        "city_slug",
        "window_months",
        "n_obs",
        "second_stage_coef_available_hat",
        "second_stage_se_available_hat",
        "second_stage_t_available_hat",
        "second_stage_p_available_hat",
        "second_stage_ci95_low",
        "second_stage_ci95_high",
    ]
    city_df[city_second_cols].to_csv(day4_proc / "second_stage_city_window_estimates.csv", index=False)

    bandwidth_diag = pooled_df[
        [
            "window_months",
            "n_obs",
            "first_stage_f_post",
            "second_stage_coef_available_hat",
            "second_stage_se_available_hat",
            "second_stage_ci95_low",
            "second_stage_ci95_high",
        ]
    ].copy()
    bandwidth_diag = bandwidth_diag.rename(columns={"first_stage_f_post": "first_stage_f_stat"})
    bandwidth_diag.to_csv(day4_proc / "diagnostic_bandwidth_sensitivity_pooled.csv", index=False)

    placebo_df = run_placebo_pooled(
        day2_proc / "fact_listing_day_multicity_bw_3m.csv.gz",
        offsets_days=[-30, 30],
        chunk_size=args.chunk_size,
    )
    # Add true-cutoff row from pooled ±3m baseline for side-by-side comparison.
    true_row = pooled_df[pooled_df["window_months"] == 3].iloc[0]
    true_df = pd.DataFrame(
        [
            {
                "placebo_label": "true_cutoff_0d",
                "placebo_offset_days": 0,
                "n_obs": int(true_row["n_obs"]),
                "first_stage_f_post": true_row["first_stage_f_post"],
                "second_stage_coef_available_hat": true_row["second_stage_coef_available_hat"],
                "second_stage_se_available_hat": true_row["second_stage_se_available_hat"],
                "second_stage_t_available_hat": true_row["second_stage_t_available_hat"],
                "second_stage_p_available_hat": true_row["second_stage_p_available_hat"],
                "second_stage_ci95_low": true_row["second_stage_ci95_low"],
                "second_stage_ci95_high": true_row["second_stage_ci95_high"],
            }
        ]
    )
    placebo_out = pd.concat([true_df, placebo_df], ignore_index=True)
    placebo_out.to_csv(day4_proc / "diagnostic_placebo_cutoff_pooled_bw3m.csv", index=False)

    summary = {
        "first_stage_table": "data/processed/step4/first_stage_strength_city_window.csv",
        "second_stage_pooled": "data/processed/step4/second_stage_pooled_window_estimates.csv",
        "second_stage_city": "data/processed/step4/second_stage_city_window_estimates.csv",
        "diagnostic_bandwidth": "data/processed/step4/diagnostic_bandwidth_sensitivity_pooled.csv",
        "diagnostic_placebo": "data/processed/step4/diagnostic_placebo_cutoff_pooled_bw3m.csv",
        "pooled_window_rows": int(len(pooled_df)),
        "city_window_rows": int(len(city_df)),
    }
    (day4_proc / "day4_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
