#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parents[1]
STEP2 = ROOT / "data/processed/step2"
RAW_DAY2 = ROOT / "data/raw/step2"
OUT = ROOT / "data/processed/panel_extension"
OUT.mkdir(parents=True, exist_ok=True)

INPUT_PANEL = STEP2 / "fact_listing_day_multicity_bw_3m.csv.gz"
OUTPUT_PANEL = OUT / "dynamic_proxy_panel.csv"
OUTPUT_BREAKS = OUT / "listing_breaks.csv"
OUTPUT_BREAK_DIST = OUT / "break_date_distribution.csv"
OUTPUT_META = OUT / "structural_break_metadata.json"

MIN_SEGMENT_DAYS = 14
TARGET_ADOPTION_RANGE = (0.05, 0.20)
TARGET_ADOPTION_MIDPOINT = 0.125
TUNING_SAMPLE_MOD = 101

LOGGER = logging.getLogger("panel_extension_1")


@dataclass(frozen=True)
class BreakConfig:
    model: str
    penalty_scale: float


@dataclass
class BreakResult:
    idx: int | None
    method: str
    signal: str | None
    pre_mean: float
    post_mean: float
    z_score: float
    mean_ratio: float
    score: float


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_listing_coordinates() -> pd.DataFrame:
    """Load listing-level lat/lon from raw listings snapshots (city, listing_id keyed)."""
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
            df = df[["city_slug", "listing_id", "latitude", "longitude"]].drop_duplicates(
                subset=["city_slug", "listing_id"],
                keep="first",
            )
            rows.append(df)
        except Exception as exc:
            LOGGER.warning("Could not load coordinates from %s: %s", listings_path, exc)

    if not rows:
        LOGGER.warning("No raw coordinates found under %s; spillover script may fallback.", RAW_DAY2)
        return pd.DataFrame(columns=["city_slug", "listing_id", "latitude", "longitude"])

    coords = pd.concat(rows, ignore_index=True)
    LOGGER.info("Loaded coordinates for %s city-listing pairs.", f"{len(coords):,}")
    return coords


def iter_listing_groups(
    usecols: list[str],
    dtype_map: dict[str, str],
    chunksize: int = 450_000,
    coords: pd.DataFrame | None = None,
) -> Iterable[pd.DataFrame]:
    """Yield full listing groups while handling chunk boundaries safely."""
    carry = pd.DataFrame()

    reader = pd.read_csv(INPUT_PANEL, usecols=usecols, dtype=dtype_map, chunksize=chunksize)

    for chunk_idx, chunk in enumerate(reader, start=1):
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        if chunk.empty:
            continue

        if coords is not None and not coords.empty:
            for col in ["latitude", "longitude"]:
                if col in chunk.columns:
                    chunk = chunk.drop(columns=[col])
            chunk = chunk.merge(coords, on=["city_slug", "listing_id"], how="left")

        last_listing = int(chunk["listing_id"].iloc[-1])
        complete_mask = chunk["listing_id"] != last_listing
        complete = chunk.loc[complete_mask].copy()
        carry = chunk.loc[~complete_mask].copy()

        if chunk_idx % 12 == 0:
            LOGGER.info("Chunk %s processed for group iteration.", chunk_idx)

        if complete.empty:
            continue

        for _, grp in complete.groupby("listing_id", sort=False):
            yield grp

    if not carry.empty:
        if coords is not None and not coords.empty and "latitude" not in carry.columns:
            carry = carry.merge(coords, on=["city_slug", "listing_id"], how="left")
        for _, grp in carry.groupby("listing_id", sort=False):
            yield grp


def _safe_stats(pre: np.ndarray, post: np.ndarray) -> tuple[float, float, float, float, float]:
    pre_mean = float(np.nanmean(pre)) if pre.size else np.nan
    post_mean = float(np.nanmean(post)) if post.size else np.nan

    pre_var = float(np.nanvar(pre, ddof=1)) if pre.size > 1 else 0.0
    post_var = float(np.nanvar(post, ddof=1)) if post.size > 1 else 0.0
    se = np.sqrt(max(pre_var, 0.0) / max(pre.size, 1) + max(post_var, 0.0) / max(post.size, 1))
    z = (post_mean - pre_mean) / (se + 1e-8)

    ratio_raw = (post_mean + 1e-8) / (pre_mean + 1e-8)
    ratio_sym = max(ratio_raw, 1.0 / max(ratio_raw, 1e-8))
    score = float(abs(z) * abs(np.log(max(ratio_raw, 1e-8))))
    return pre_mean, post_mean, float(z), float(ratio_sym), score


def detect_pelt_single_series(
    series: np.ndarray,
    config: BreakConfig,
    min_segment: int = MIN_SEGMENT_DAYS,
) -> BreakResult:
    y = np.asarray(series, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < (2 * min_segment + 1):
        return BreakResult(None, "insufficient_length", None, np.nan, np.nan, np.nan, np.nan, np.nan)

    y = y[mask]
    n = y.size
    if n < (2 * min_segment + 1):
        return BreakResult(None, "insufficient_length", None, np.nan, np.nan, np.nan, np.nan, np.nan)

    var_y = float(np.var(y))
    base_penalty = max(var_y, 1e-6) * np.log(n + 1.0)
    penalty = float(config.penalty_scale * base_penalty)

    try:
        algo = rpt.Pelt(model=config.model, min_size=min_segment, jump=1).fit(y.reshape(-1, 1))
        bkps = algo.predict(pen=penalty)
    except Exception as exc:
        return BreakResult(None, f"ruptures_error:{type(exc).__name__}", None, np.nan, np.nan, np.nan, np.nan, np.nan)

    if not bkps:
        return BreakResult(None, "no_break_found", None, np.nan, np.nan, np.nan, np.nan, np.nan)

    best: BreakResult | None = None
    for bkp in bkps:
        if bkp >= n:
            continue
        if bkp < min_segment or bkp > (n - min_segment):
            continue

        idx = int(bkp)
        pre = y[:idx]
        post = y[idx:]
        pre_mean, post_mean, z, ratio, score = _safe_stats(pre, post)

        candidate = BreakResult(
            idx=idx,
            method=f"ruptures_pelt_{config.model}",
            signal=None,
            pre_mean=pre_mean,
            post_mean=post_mean,
            z_score=z,
            mean_ratio=ratio,
            score=score,
        )

        if best is None or (np.isfinite(candidate.score) and candidate.score > best.score):
            best = candidate

    if best is None:
        return BreakResult(None, "no_valid_break", None, np.nan, np.nan, np.nan, np.nan, np.nan)

    return best


def detect_break_from_volatility(
    vol7: np.ndarray,
    vol14: np.ndarray,
    config: BreakConfig,
    min_segment: int = MIN_SEGMENT_DAYS,
) -> BreakResult:
    c7 = detect_pelt_single_series(vol7, config=config, min_segment=min_segment)
    c7.signal = "availability_volatility_7d"

    c14 = detect_pelt_single_series(vol14, config=config, min_segment=min_segment)
    c14.signal = "availability_volatility_14d"

    candidates = [c for c in [c7, c14] if c.idx is not None and np.isfinite(c.score)]
    if not candidates:
        return BreakResult(None, "no_valid_signal_break", None, np.nan, np.nan, np.nan, np.nan, np.nan)

    best = max(candidates, key=lambda x: x.score)
    return best


def compute_listing_series(group: pd.DataFrame) -> pd.DataFrame:
    sub = group.sort_values("date").copy()
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["price_usd"] = pd.to_numeric(sub["price_usd"], errors="coerce")
    sub["log_price"] = pd.to_numeric(sub["log_price"], errors="coerce")
    sub["available"] = pd.to_numeric(sub.get("available"), errors="coerce")
    sub = sub.dropna(subset=["date", "price_usd", "log_price", "available"])

    if sub.empty:
        return sub

    sub["abs_price_change"] = sub["price_usd"].diff().abs().fillna(0.0)

    # Price is near-constant for most listings in this panel. We therefore use
    # booking-availability churn as the volatility signal for structural breaks.
    sub["availability_change_abs"] = sub["available"].diff().abs().fillna(0.0)
    sub["availability_volatility_7d"] = (
        sub["availability_change_abs"].rolling(7, min_periods=3).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    sub["availability_volatility_14d"] = (
        sub["availability_change_abs"].rolling(14, min_periods=5).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # Keep legacy column names for downstream scripts.
    sub["price_volatility_7d"] = sub["availability_volatility_7d"].astype(float)
    sub["price_volatility_14d"] = sub["availability_volatility_14d"].astype(float)
    sub["rolling_7d_variance"] = (sub["availability_volatility_7d"] ** 2).astype(float)
    return sub


def tune_parameters() -> tuple[BreakConfig, float, float, pd.DataFrame, dict[str, Any]]:
    LOGGER.info("Starting tuning pass for structural-break proxy.")

    usecols = ["listing_id", "date", "price_usd", "log_price", "available"]
    dtype_map = {
        "listing_id": "int64",
        "price_usd": "float32",
        "log_price": "float32",
        "available": "float32",
    }

    config_grid = [
        BreakConfig(model=m, penalty_scale=s)
        for m in ["rbf", "l2"]
        for s in [0.35, 0.55, 0.80, 1.10]
    ]

    threshold_grid = [(z, r) for z in [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] for r in [1.01, 1.03, 1.05, 1.08, 1.10]]

    tuning_rows: list[dict[str, Any]] = []
    n_sampled_listings = 0
    n_total_seen = 0

    for grp in iter_listing_groups(usecols=usecols, dtype_map=dtype_map, chunksize=500_000, coords=None):
        n_total_seen += 1
        listing_id = int(grp["listing_id"].iloc[0])
        if listing_id % TUNING_SAMPLE_MOD != 0:
            continue

        try:
            sub = compute_listing_series(grp)
            if sub.empty or len(sub) < (2 * MIN_SEGMENT_DAYS + 1):
                continue

            v7 = sub["price_volatility_7d"].to_numpy(dtype=float)
            v14 = sub["price_volatility_14d"].to_numpy(dtype=float)

            n_sampled_listings += 1
            for cfg in config_grid:
                br = detect_break_from_volatility(v7, v14, config=cfg, min_segment=MIN_SEGMENT_DAYS)
                tuning_rows.append(
                    {
                        "listing_id": listing_id,
                        "model": cfg.model,
                        "penalty_scale": cfg.penalty_scale,
                        "has_break": int(br.idx is not None),
                        "z_score": float(br.z_score) if np.isfinite(br.z_score) else np.nan,
                        "mean_ratio": float(br.mean_ratio) if np.isfinite(br.mean_ratio) else np.nan,
                    }
                )

            if n_sampled_listings % 250 == 0:
                LOGGER.info("Tuning progress: sampled %s listings.", f"{n_sampled_listings:,}")
        except Exception as exc:
            LOGGER.warning("Tuning skip for listing %s due to error: %s", listing_id, exc)
            continue

    if not tuning_rows or n_sampled_listings == 0:
        raise RuntimeError("Tuning failed: no listing-level tuning rows were generated.")

    tuning_df = pd.DataFrame(tuning_rows)
    summary_rows: list[dict[str, Any]] = []

    for cfg in config_grid:
        cfg_mask = (tuning_df["model"] == cfg.model) & (tuning_df["penalty_scale"] == cfg.penalty_scale)
        cfg_df = tuning_df.loc[cfg_mask].copy()
        if cfg_df.empty:
            continue

        for z_thr, ratio_thr in threshold_grid:
            adopted = (
                (cfg_df["has_break"] == 1)
                & (cfg_df["z_score"].abs() >= z_thr)
                & (cfg_df["mean_ratio"] >= ratio_thr)
            )
            share = float(adopted.mean()) if len(adopted) else np.nan
            in_target = TARGET_ADOPTION_RANGE[0] <= share <= TARGET_ADOPTION_RANGE[1]
            dist_mid = abs(share - TARGET_ADOPTION_MIDPOINT)
            summary_rows.append(
                {
                    "model": cfg.model,
                    "penalty_scale": float(cfg.penalty_scale),
                    "z_threshold": float(z_thr),
                    "ratio_threshold": float(ratio_thr),
                    "adoption_share_sample": share,
                    "in_target_range": bool(in_target),
                    "distance_to_midpoint": dist_mid,
                    "sample_listings": int(len(cfg_df)),
                }
            )

    score_df = pd.DataFrame(summary_rows)
    if score_df.empty:
        raise RuntimeError("Tuning failed: no grid summary generated.")

    in_target = score_df[score_df["in_target_range"]].copy()
    if not in_target.empty:
        chosen_row = in_target.sort_values("distance_to_midpoint").iloc[0]
    else:
        chosen_row = score_df.sort_values("distance_to_midpoint").iloc[0]

    chosen_cfg = BreakConfig(model=str(chosen_row["model"]), penalty_scale=float(chosen_row["penalty_scale"]))
    z_thr = float(chosen_row["z_threshold"])
    ratio_thr = float(chosen_row["ratio_threshold"])

    tuning_meta = {
        "tuning_sample_mod": TUNING_SAMPLE_MOD,
        "n_listings_seen_in_pass": int(n_total_seen),
        "n_listings_in_tuning_sample": int(n_sampled_listings),
        "target_adoption_range": [TARGET_ADOPTION_RANGE[0], TARGET_ADOPTION_RANGE[1]],
        "selected_sample_share": float(chosen_row["adoption_share_sample"]),
    }

    LOGGER.info(
        "Selected tuning config model=%s penalty_scale=%.3f abs(z)>=%.3f symmetric_ratio>=%.3f sample_share=%.3f",
        chosen_cfg.model,
        chosen_cfg.penalty_scale,
        z_thr,
        ratio_thr,
        float(chosen_row["adoption_share_sample"]),
    )

    return chosen_cfg, z_thr, ratio_thr, score_df, tuning_meta


def process_listing(
    group: pd.DataFrame,
    cfg: BreakConfig,
    z_threshold: float,
    ratio_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    listing_id = int(group["listing_id"].iloc[0])

    try:
        sub = compute_listing_series(group)
        if sub.empty:
            info = {
                "listing_id": listing_id,
                "break_date": None,
                "adopted": 0,
                "n_obs": 0,
                "method": "empty",
                "signal": None,
                "pre_mean_volatility": np.nan,
                "post_mean_volatility": np.nan,
                "z_score": np.nan,
                "mean_ratio": np.nan,
            }
            return sub, info

        br = detect_break_from_volatility(
            sub["price_volatility_7d"].to_numpy(dtype=float),
            sub["price_volatility_14d"].to_numpy(dtype=float),
            config=cfg,
            min_segment=MIN_SEGMENT_DAYS,
        )

        accepted = (
            br.idx is not None
            and np.isfinite(br.z_score)
            and np.isfinite(br.mean_ratio)
            and abs(br.z_score) >= z_threshold
            and br.mean_ratio >= ratio_threshold
        )

        if accepted and br.idx is not None and br.idx < len(sub):
            break_date = pd.Timestamp(sub.iloc[br.idx]["date"])
            sub["dynamic_algo_adopted"] = (sub["date"] >= break_date).astype("int8")
            sub["break_date"] = break_date
            sub["event_time"] = (sub["date"] - break_date).dt.days.astype("int16")
            adopted = 1
            break_date_value = break_date.date().isoformat()
            method = br.method
        else:
            sub["dynamic_algo_adopted"] = 0
            sub["break_date"] = pd.NaT
            sub["event_time"] = np.nan
            adopted = 0
            break_date_value = None
            method = f"{br.method}_rejected"

        sub["price"] = sub["price_usd"]
        if "neighbourhood_cleansed" in sub.columns:
            sub["neighbourhood"] = sub["neighbourhood_cleansed"]
        else:
            sub["neighbourhood"] = "UNKNOWN"

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
            "latitude",
            "longitude",
            "minimum_nights",
            "maximum_nights",
            "post_cutoff",
            "abs_price_change",
            "availability_change_abs",
            "availability_volatility_7d",
            "availability_volatility_14d",
            "price_volatility_7d",
            "price_volatility_14d",
            "rolling_7d_variance",
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
            "method": method,
            "signal": br.signal,
            "pre_mean_volatility": br.pre_mean,
            "post_mean_volatility": br.post_mean,
            "z_score": br.z_score,
            "mean_ratio": br.mean_ratio,
        }
        return sub, info

    except Exception as exc:
        LOGGER.warning("Error processing listing %s: %s", listing_id, exc)
        safe = group[[c for c in ["city_slug", "listing_id", "date"] if c in group.columns]].copy()
        safe["dynamic_algo_adopted"] = 0
        safe["break_date"] = pd.NaT
        safe["event_time"] = np.nan
        info = {
            "listing_id": listing_id,
            "break_date": None,
            "adopted": 0,
            "n_obs": int(len(safe)),
            "method": f"error:{type(exc).__name__}",
            "signal": None,
            "pre_mean_volatility": np.nan,
            "post_mean_volatility": np.nan,
            "z_score": np.nan,
            "mean_ratio": np.nan,
        }
        return safe, info


def flush_buffer(buffer: list[pd.DataFrame], write_header: bool) -> bool:
    if not buffer:
        return write_header
    out_df = pd.concat(buffer, ignore_index=True)
    out_df.to_csv(OUTPUT_PANEL, mode="w" if write_header else "a", header=write_header, index=False)
    return False


def main() -> None:
    configure_logging()

    if OUTPUT_PANEL.exists():
        OUTPUT_PANEL.unlink()

    cfg, z_thr, ratio_thr, tuning_grid_df, tuning_meta = tune_parameters()
    coords = load_listing_coordinates()

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
    buffer: list[pd.DataFrame] = []
    buffer_rows = 0

    for idx, grp in enumerate(iter_listing_groups(usecols=usecols, dtype_map=dtype_map, coords=coords), start=1):
        out_sub, info = process_listing(grp, cfg=cfg, z_threshold=z_thr, ratio_threshold=ratio_thr)

        if not out_sub.empty:
            buffer.append(out_sub)
            buffer_rows += len(out_sub)
            row_count += len(out_sub)

        listing_infos.append(info)

        if buffer_rows >= 250_000:
            write_header = flush_buffer(buffer, write_header)
            buffer = []
            buffer_rows = 0

        if idx % 10_000 == 0:
            LOGGER.info("Processed %s listings (%s rows written).", f"{idx:,}", f"{row_count:,}")

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

    tuning_grid_top = (
        tuning_grid_df.assign(
            target_gap=lambda d: np.where(
                d["in_target_range"],
                d["distance_to_midpoint"],
                d["distance_to_midpoint"] + 1.0,
            )
        )
        .sort_values(["target_gap", "distance_to_midpoint"])
        .head(15)
        .to_dict(orient="records")
    )

    meta = {
        "input_panel": str(INPUT_PANEL),
        "output_panel": str(OUTPUT_PANEL),
        "n_rows_output": int(row_count),
        "n_listings_total": int(len(listing_breaks)),
        "n_listings_adopted": int(adopted.sum()),
        "share_listings_adopted": float(adopted.mean()) if len(adopted) else np.nan,
        "n_unique_break_dates": int(break_dist.shape[0]),
        "break_detection": {
            "algorithm": "ruptures.Pelt",
            "volatility_signals": ["availability_volatility_7d", "availability_volatility_14d"],
            "selected_model": cfg.model,
            "selected_penalty_scale": cfg.penalty_scale,
            "selected_thresholds": {
                "abs_z_threshold": z_thr,
                "symmetric_ratio_threshold": ratio_thr,
                "min_segment_days": MIN_SEGMENT_DAYS,
            },
            "target_adoption_share_range": [TARGET_ADOPTION_RANGE[0], TARGET_ADOPTION_RANGE[1]],
            "method_counts": method_counts,
            "tuning": {
                **tuning_meta,
                "top_grid_candidates": tuning_grid_top,
            },
        },
        "files": {
            "dynamic_proxy_panel": str(OUTPUT_PANEL),
            "listing_breaks": str(OUTPUT_BREAKS),
            "break_date_distribution": str(OUTPUT_BREAK_DIST),
        },
    }

    share = float(meta["share_listings_adopted"])
    lo, hi = TARGET_ADOPTION_RANGE
    if not (lo <= share <= hi):
        raise RuntimeError(
            f"Adoption share {share:.4f} is outside required range [{lo:.2f}, {hi:.2f}] for recalibrated proxy."
        )
    if meta["break_detection"].get("algorithm") != "ruptures.Pelt":
        raise RuntimeError("Break detection algorithm must be ruptures.Pelt for recalibrated panel extension.")

    OUTPUT_META.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Structural-break run complete. Adopted share: %.4f", meta["share_listings_adopted"])
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
