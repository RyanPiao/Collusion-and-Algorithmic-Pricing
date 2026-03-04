#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import ruptures as rpt  # type: ignore

    HAS_RUPTURES = True
except Exception:
    HAS_RUPTURES = False

ROOT = Path(__file__).resolve().parents[1]
DAY2 = ROOT / "data/processed/day2"
OUT = ROOT / "data/processed/panel_extension"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_PANEL = DAY2 / "fact_listing_day_multicity_bw_3m.csv.gz"
OUTPUT_PANEL = OUT / "dynamic_proxy_panel.csv"
OUTPUT_BREAKS = OUT / "listing_breaks.csv"
OUTPUT_BREAK_DIST = OUT / "break_date_distribution.csv"
OUTPUT_META = OUT / "structural_break_metadata.json"


@dataclass
class BreakResult:
    idx: int | None
    method: str
    pre_mean_abs_change: float
    post_mean_abs_change: float
    z_score: float
    mean_ratio: float


def detect_break_fallback(abs_changes: np.ndarray, min_segment: int = 14, z_threshold: float = 2.0, ratio_threshold: float = 1.25) -> BreakResult:
    y = np.asarray(abs_changes, dtype=float)
    n = y.size
    if n < (2 * min_segment + 1):
        return BreakResult(None, "fallback_cusum", np.nan, np.nan, np.nan, np.nan)

    best_idx: int | None = None
    best_score = -np.inf
    best_pre = np.nan
    best_post = np.nan
    best_z = np.nan
    best_ratio = np.nan

    for t in range(min_segment, n - min_segment + 1):
        pre = y[:t]
        post = y[t:]
        pre_mean = float(np.mean(pre))
        post_mean = float(np.mean(post))

        if not np.isfinite(pre_mean) or not np.isfinite(post_mean):
            continue
        if post_mean <= pre_mean:
            continue

        pre_var = float(np.var(pre, ddof=1)) if pre.size > 1 else 0.0
        post_var = float(np.var(post, ddof=1)) if post.size > 1 else 0.0
        se = np.sqrt(max(pre_var, 0.0) / max(pre.size, 1) + max(post_var, 0.0) / max(post.size, 1))
        z = (post_mean - pre_mean) / (se + 1e-8)
        ratio = (post_mean + 1e-6) / (pre_mean + 1e-6)
        score = z * np.log1p(max(ratio - 1.0, 0.0))

        if score > best_score:
            best_score = score
            best_idx = t
            best_pre = pre_mean
            best_post = post_mean
            best_z = float(z)
            best_ratio = float(ratio)

    if best_idx is None:
        return BreakResult(None, "fallback_cusum", np.nan, np.nan, np.nan, np.nan)

    if (not np.isfinite(best_z)) or (best_z < z_threshold) or (best_ratio < ratio_threshold):
        return BreakResult(None, "fallback_cusum", best_pre, best_post, best_z, best_ratio)

    return BreakResult(best_idx, "fallback_cusum", best_pre, best_post, best_z, best_ratio)


def detect_break(abs_changes: np.ndarray, min_segment: int = 14) -> BreakResult:
    y = np.asarray(abs_changes, dtype=float)
    n = y.size
    if n < (2 * min_segment + 1):
        return BreakResult(None, "insufficient_length", np.nan, np.nan, np.nan, np.nan)

    if HAS_RUPTURES:
        try:
            algo = rpt.Binseg(model="l2").fit(y.reshape(-1, 1))
            pred = algo.predict(n_bkps=1)
            idx = int(pred[0]) if pred else None
            if idx is not None:
                idx = max(min_segment, min(idx, n - min_segment))
                pre = y[:idx]
                post = y[idx:]
                pre_mean = float(np.mean(pre))
                post_mean = float(np.mean(post))
                pre_var = float(np.var(pre, ddof=1)) if pre.size > 1 else 0.0
                post_var = float(np.var(post, ddof=1)) if post.size > 1 else 0.0
                se = np.sqrt(max(pre_var, 0.0) / max(pre.size, 1) + max(post_var, 0.0) / max(post.size, 1))
                z = (post_mean - pre_mean) / (se + 1e-8)
                ratio = (post_mean + 1e-6) / (pre_mean + 1e-6)
                if post_mean > pre_mean and z >= 2.0 and ratio >= 1.25:
                    return BreakResult(idx, "ruptures_binseg", pre_mean, post_mean, float(z), float(ratio))
                return BreakResult(None, "ruptures_binseg_rejected", pre_mean, post_mean, float(z), float(ratio))
        except Exception:
            pass

    return detect_break_fallback(y, min_segment=min_segment)


def process_listing(group: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sub = group.sort_values("date").copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["price_usd"] = pd.to_numeric(sub["price_usd"], errors="coerce")
    sub["log_price"] = pd.to_numeric(sub["log_price"], errors="coerce")

    sub = sub.dropna(subset=["date", "price_usd", "log_price"])  # keep clean time series per listing
    listing_id = int(group["listing_id"].iloc[0])

    if sub.empty:
        info = {
            "listing_id": listing_id,
            "break_date": None,
            "adopted": 0,
            "n_obs": 0,
            "method": "empty",
            "pre_mean_abs_change": np.nan,
            "post_mean_abs_change": np.nan,
            "z_score": np.nan,
            "mean_ratio": np.nan,
        }
        return sub, info

    sub["abs_price_change"] = sub["price_usd"].diff().abs().fillna(0.0)
    sub["price_volatility_7d"] = sub["abs_price_change"].rolling(7, min_periods=3).std().fillna(0.0)
    sub["price_volatility_14d"] = sub["abs_price_change"].rolling(14, min_periods=5).std().fillna(0.0)

    br = detect_break(sub["abs_price_change"].to_numpy(), min_segment=14)

    if br.idx is not None and br.idx < len(sub):
        break_date = pd.Timestamp(sub.iloc[br.idx]["date"])
        sub["dynamic_algo_adopted"] = (sub["date"] >= break_date).astype("int8")
        sub["break_date"] = break_date
        sub["event_time"] = (sub["date"] - break_date).dt.days.astype("int16")
        adopted = 1
        break_date_value = break_date.date().isoformat()
    else:
        sub["dynamic_algo_adopted"] = 0
        sub["break_date"] = pd.NaT
        sub["event_time"] = np.nan
        adopted = 0
        break_date_value = None

    sub["price"] = sub["price_usd"]
    sub["neighbourhood"] = sub.get("neighbourhood_cleansed", pd.Series(index=sub.index, dtype="object"))

    out_cols = [
        "city_slug",
        "listing_id",
        "date",
        "price",
        "price_usd",
        "log_price",
        "available",
        "neighbourhood",
        "neighbourhood_cleansed",
        "minimum_nights",
        "maximum_nights",
        "post_cutoff",
        "abs_price_change",
        "price_volatility_7d",
        "price_volatility_14d",
        "break_date",
        "event_time",
        "dynamic_algo_adopted",
    ]
    out_cols = [c for c in out_cols if c in sub.columns]
    sub = sub[out_cols]

    info = {
        "listing_id": listing_id,
        "break_date": break_date_value,
        "adopted": adopted,
        "n_obs": int(len(sub)),
        "method": br.method,
        "pre_mean_abs_change": br.pre_mean_abs_change,
        "post_mean_abs_change": br.post_mean_abs_change,
        "z_score": br.z_score,
        "mean_ratio": br.mean_ratio,
    }
    return sub, info


def flush_buffer(buffer: list[pd.DataFrame], write_header: bool) -> bool:
    if not buffer:
        return write_header
    out_df = pd.concat(buffer, ignore_index=True)
    out_df.to_csv(OUTPUT_PANEL, mode="w" if write_header else "a", header=write_header, index=False)
    return False


def main() -> None:
    if OUTPUT_PANEL.exists():
        OUTPUT_PANEL.unlink()

    usecols = [
        "city_slug",
        "listing_id",
        "date",
        "price_usd",
        "log_price",
        "available",
        "neighbourhood_cleansed",
        "minimum_nights",
        "maximum_nights",
        "post_cutoff",
    ]

    dtype_map = {
        "city_slug": "string",
        "listing_id": "int64",
        "price_usd": "float32",
        "log_price": "float32",
        "available": "float32",
        "neighbourhood_cleansed": "string",
        "minimum_nights": "float32",
        "maximum_nights": "float32",
        "post_cutoff": "float32",
    }

    listing_infos: list[dict[str, Any]] = []
    row_count = 0
    write_header = True
    carry = pd.DataFrame()
    buffer: list[pd.DataFrame] = []
    buffer_rows = 0

    reader = pd.read_csv(INPUT_PANEL, usecols=usecols, dtype=dtype_map, chunksize=400_000)

    for chunk_idx, chunk in enumerate(reader, start=1):
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        if chunk.empty:
            continue

        last_listing = int(chunk["listing_id"].iloc[-1])
        complete_mask = chunk["listing_id"] != last_listing
        complete = chunk.loc[complete_mask].copy()
        carry = chunk.loc[~complete_mask].copy()

        if complete.empty:
            continue

        for _, grp in complete.groupby("listing_id", sort=False):
            out_sub, info = process_listing(grp)
            if not out_sub.empty:
                buffer.append(out_sub)
                buffer_rows += len(out_sub)
                row_count += len(out_sub)
            listing_infos.append(info)

            if buffer_rows >= 250_000:
                write_header = flush_buffer(buffer, write_header)
                buffer = []
                buffer_rows = 0

        if chunk_idx % 10 == 0:
            print(f"Processed chunks: {chunk_idx}, listings so far: {len(listing_infos):,}, rows written: {row_count:,}")

    if not carry.empty:
        for _, grp in carry.groupby("listing_id", sort=False):
            out_sub, info = process_listing(grp)
            if not out_sub.empty:
                buffer.append(out_sub)
                buffer_rows += len(out_sub)
                row_count += len(out_sub)
            listing_infos.append(info)

    write_header = flush_buffer(buffer, write_header)

    listing_breaks = pd.DataFrame(listing_infos)
    listing_breaks.to_csv(OUTPUT_BREAKS, index=False)

    adopted = listing_breaks["adopted"].fillna(0).astype(int)
    break_dates = listing_breaks.loc[listing_breaks["break_date"].notna(), "break_date"].astype(str)

    break_dist = (
        break_dates.value_counts()
        .rename_axis("break_date")
        .reset_index(name="n_listings")
        .sort_values("break_date")
        .reset_index(drop=True)
    )
    break_dist.to_csv(OUTPUT_BREAK_DIST, index=False)

    method_counts = listing_breaks["method"].value_counts(dropna=False).to_dict()

    meta = {
        "input_panel": str(INPUT_PANEL),
        "output_panel": str(OUTPUT_PANEL),
        "n_rows_output": int(row_count),
        "n_listings_total": int(len(listing_breaks)),
        "n_listings_adopted": int(adopted.sum()),
        "share_listings_adopted": float(adopted.mean()) if len(adopted) else np.nan,
        "n_unique_break_dates": int(break_dist.shape[0]),
        "break_detection": {
            "ruptures_available": bool(HAS_RUPTURES),
            "method_when_unavailable": "fallback_cusum",
            "criteria": {
                "min_segment_days": 14,
                "z_threshold": 2.0,
                "ratio_threshold": 1.25,
                "target_shift": "sticky_to_volatile (post mean absolute price change > pre)",
            },
            "method_counts": method_counts,
        },
        "files": {
            "dynamic_proxy_panel": str(OUTPUT_PANEL),
            "listing_breaks": str(OUTPUT_BREAKS),
            "break_date_distribution": str(OUTPUT_BREAK_DIST),
        },
    }

    OUTPUT_META.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
