#!/usr/bin/env python3
"""Day 2 multicity panel builder.

Builds:
1) Primary daily listing-level panel around cutoff (±3 months)
2) Windowed daily extracts (±1m, ±2m, ±3m)
3) Secondary monthly aggregates
4) QA outputs (coverage, missingness, support, first-stage prep, checks)

Notes:
- Miami is requested but not available on current InsideAirbnb US market endpoints.
  The script substitutes Washington DC as the approved alternate market.
- City-specific Smart Pricing rollout dates are not discoverable from the source files,
  so the script supports optional city-specific overrides and otherwise falls back
  to a pooled cutoff date with auditable metadata.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

REQUIRED_CITY_ORDER = [
    "boston",
    "new-york-city",
    "los-angeles",
    "san-francisco",
    "miami",
    "austin",
    "chicago",
    "seattle",
]

CITY_CATALOG = {
    "boston": {"city_name": "Boston", "market_path": "united-states/ma/boston", "snapshot_date": "2025-01-24"},
    "new-york-city": {
        "city_name": "New York City",
        "market_path": "united-states/ny/new-york-city",
        "snapshot_date": "2025-03-01",
    },
    "los-angeles": {
        "city_name": "Los Angeles",
        "market_path": "united-states/ca/los-angeles",
        "snapshot_date": "2025-03-01",
    },
    "san-francisco": {
        "city_name": "San Francisco",
        "market_path": "united-states/ca/san-francisco",
        "snapshot_date": "2025-03-01",
    },
    # Not available in current endpoint (kept as requested target for audit trail)
    "miami": {"city_name": "Miami", "market_path": "united-states/fl/miami", "snapshot_date": None},
    "austin": {"city_name": "Austin", "market_path": "united-states/tx/austin", "snapshot_date": "2025-01-16"},
    "chicago": {"city_name": "Chicago", "market_path": "united-states/il/chicago", "snapshot_date": "2025-02-16"},
    "seattle": {"city_name": "Seattle", "market_path": "united-states/wa/seattle", "snapshot_date": "2025-02-18"},
    # Approved alternate if one requested city is missing
    "washington-dc": {
        "city_name": "Washington DC",
        "market_path": "united-states/dc/washington-dc",
        "snapshot_date": "2025-01-24",
    },
}

LISTINGS_USECOLS = [
    "id",
    "host_id",
    "host_since",
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_identity_verified",
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
]

CALENDAR_USECOLS = ["listing_id", "date", "available", "price", "minimum_nights", "maximum_nights"]

KEY_MODEL_FIELDS = [
    "price_usd",
    "log_price",
    "available",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "host_is_superhost",
    "host_response_rate",
    "host_acceptance_rate",
    "host_tenure_days",
    "neighbourhood_cleansed",
]


@dataclass
class SelectedCity:
    city_slug: str
    city_name: str
    market_path: str
    snapshot_date: str
    requested: bool
    replacement_for: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"https://data.insideairbnb.com/{self.market_path}/{self.snapshot_date}/data"


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def head_ok(url: str, timeout: int = 15) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(getattr(r, "status", 0)) == 200
    except Exception:
        return False


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())


def parse_pct(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def parse_bool_tf(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"t", "true", "1", "yes"}:
        return 1.0
    if s in {"f", "false", "0", "no"}:
        return 0.0
    return np.nan


def parse_price(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", ".", "-", "-."}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def load_city_specific_cutoff_overrides(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.exists():
        return {}
    df = pd.read_csv(path)
    if "city_slug" not in df.columns or "cutoff_date_primary" not in df.columns:
        raise ValueError(f"Cutoff override file missing required columns: {path}")
    out = {}
    for _, r in df.iterrows():
        if pd.isna(r["city_slug"]) or pd.isna(r["cutoff_date_primary"]):
            continue
        out[str(r["city_slug"]).strip()] = str(r["cutoff_date_primary"]).strip()
    return out


def select_cities() -> List[SelectedCity]:
    selected: List[SelectedCity] = []
    needs_replacement = []

    for slug in REQUIRED_CITY_ORDER:
        meta = CITY_CATALOG[slug]
        snap = meta.get("snapshot_date")
        if snap:
            selected.append(
                SelectedCity(
                    city_slug=slug,
                    city_name=meta["city_name"],
                    market_path=meta["market_path"],
                    snapshot_date=snap,
                    requested=True,
                )
            )
        else:
            needs_replacement.append(slug)

    # Approved alternate only if requested city missing
    if needs_replacement:
        alt = CITY_CATALOG["washington-dc"]
        selected.append(
            SelectedCity(
                city_slug="washington-dc",
                city_name=alt["city_name"],
                market_path=alt["market_path"],
                snapshot_date=alt["snapshot_date"],
                requested=False,
                replacement_for=",".join(needs_replacement),
            )
        )

    if len(selected) != 8:
        raise RuntimeError(f"Expected 8 cities in final Day 2 panel, got {len(selected)}")

    return selected


def write_city_selection_audit(selected: List[SelectedCity], out_csv: Path) -> None:
    rows = []
    selected_slugs = {s.city_slug for s in selected}
    for slug in REQUIRED_CITY_ORDER:
        meta = CITY_CATALOG[slug]
        rows.append(
            {
                "city_slug": slug,
                "city_name": meta["city_name"],
                "requested": 1,
                "included_in_panel": int(slug in selected_slugs),
                "replacement_city_slug": "washington-dc" if slug == "miami" and slug not in selected_slugs else "",
                "reason": "requested_city_not_available_in_current_insideairbnb_endpoint"
                if slug == "miami" and slug not in selected_slugs
                else "requested_city_available",
            }
        )
    for s in selected:
        if not s.requested:
            rows.append(
                {
                    "city_slug": s.city_slug,
                    "city_name": s.city_name,
                    "requested": 0,
                    "included_in_panel": 1,
                    "replacement_city_slug": "",
                    "reason": f"approved_alternate_for:{s.replacement_for}",
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def build_cutoff_map(
    selected: List[SelectedCity],
    pooled_cutoff: str,
    secondary_cutoff: str,
    cutoff_overrides: Dict[str, str],
    out_csv: Path,
) -> pd.DataFrame:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = []
    for s in selected:
        if s.city_slug in cutoff_overrides:
            source_type = "city_specific"
            primary = cutoff_overrides[s.city_slug]
            note = "city-specific cutoff provided via override file"
            confidence = "medium"
        else:
            source_type = "global_fallback"
            primary = pooled_cutoff
            note = (
                "No credible city-specific Smart Pricing rollout date found in source files; "
                "using pooled fallback cutoff for Day 2 panel construction."
            )
            confidence = "low"

        rows.append(
            {
                "city_slug": s.city_slug,
                "city_name": s.city_name,
                "cutoff_date_primary": primary,
                "cutoff_date_secondary": secondary_cutoff,
                "source_type": source_type,
                "source_note": note,
                "confidence": confidence,
                "last_verified_at": now,
                "replacement_for": s.replacement_for or "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def load_listings_features(listings_gz_path: Path, snapshot_date: str) -> pd.DataFrame:
    listings = pd.read_csv(listings_gz_path, compression="gzip", usecols=LISTINGS_USECOLS, low_memory=False)
    listings = listings.rename(columns={"id": "listing_id"})
    listings["listing_id"] = listings["listing_id"].astype(str)

    listings["host_response_rate"] = listings["host_response_rate"].apply(parse_pct)
    listings["host_acceptance_rate"] = listings["host_acceptance_rate"].apply(parse_pct)
    listings["host_is_superhost"] = listings["host_is_superhost"].apply(parse_bool_tf)
    listings["host_identity_verified"] = listings["host_identity_verified"].apply(parse_bool_tf)

    snap = pd.Timestamp(snapshot_date)
    listings["host_since"] = pd.to_datetime(listings["host_since"], errors="coerce")
    listings["host_tenure_days"] = (snap - listings["host_since"]).dt.days

    listings = listings[
        [
            "listing_id",
            "host_id",
            "host_tenure_days",
            "host_response_rate",
            "host_acceptance_rate",
            "host_is_superhost",
            "host_identity_verified",
            "neighbourhood_cleansed",
            "property_type",
            "room_type",
            "accommodates",
            "bathrooms",
            "bedrooms",
        ]
    ]

    return listings.set_index("listing_id")


def process_city_calendar(
    city: SelectedCity,
    calendar_gz_path: Path,
    listings_features: pd.DataFrame,
    city_cutoff: str,
    cutoff_source: str,
    primary_panel_out: Path,
    chunk_size: int = 500_000,
) -> Dict[str, int]:
    cutoff = pd.Timestamp(city_cutoff)
    pre_1 = cutoff - pd.DateOffset(months=1)
    post_1 = cutoff + pd.DateOffset(months=1)
    pre_2 = cutoff - pd.DateOffset(months=2)
    post_2 = cutoff + pd.DateOffset(months=2)
    pre_3 = cutoff - pd.DateOffset(months=3)
    post_3 = cutoff + pd.DateOffset(months=3)
    start_date = pre_3
    end_date = post_3

    first_write = not primary_panel_out.exists()

    stats = {"rows_in": 0, "rows_kept": 0}

    reader = pd.read_csv(
        calendar_gz_path,
        compression="gzip",
        usecols=CALENDAR_USECOLS,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk in reader:
        stats["rows_in"] += len(chunk)

        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk = chunk[(chunk["date"] >= start_date) & (chunk["date"] <= end_date)]
        if chunk.empty:
            continue

        chunk["listing_id"] = chunk["listing_id"].astype(str)
        chunk["available"] = chunk["available"].apply(parse_bool_tf)
        chunk["price_usd"] = chunk["price"].apply(parse_price)

        # Keep observations with valid positive prices and non-missing availability.
        chunk = chunk[(chunk["price_usd"] > 0) & (chunk["available"].isin([0.0, 1.0]))]
        if chunk.empty:
            continue

        chunk = chunk.drop(columns=["price"]).join(listings_features, on="listing_id", how="left")

        chunk["city_slug"] = city.city_slug
        chunk["city_name"] = city.city_name
        chunk["cutoff_date_city"] = cutoff.date().isoformat()
        chunk["cutoff_source"] = cutoff_source
        chunk["days_from_cutoff"] = (chunk["date"] - cutoff).dt.days.astype(int)
        chunk["post_cutoff"] = (chunk["date"] >= cutoff).astype(int)
        chunk["in_bw_1m"] = ((chunk["date"] >= pre_1) & (chunk["date"] <= post_1)).astype(int)
        chunk["in_bw_2m"] = ((chunk["date"] >= pre_2) & (chunk["date"] <= post_2)).astype(int)
        chunk["in_bw_3m"] = ((chunk["date"] >= pre_3) & (chunk["date"] <= post_3)).astype(int)
        chunk["log_price"] = np.log(chunk["price_usd"])

        # Keep canonical column order
        cols = [
            "city_slug",
            "city_name",
            "listing_id",
            "date",
            "price_usd",
            "log_price",
            "available",
            "minimum_nights",
            "maximum_nights",
            "cutoff_date_city",
            "cutoff_source",
            "days_from_cutoff",
            "post_cutoff",
            "in_bw_1m",
            "in_bw_2m",
            "in_bw_3m",
            "neighbourhood_cleansed",
            "property_type",
            "room_type",
            "accommodates",
            "bathrooms",
            "bedrooms",
            "host_id",
            "host_tenure_days",
            "host_is_superhost",
            "host_identity_verified",
            "host_response_rate",
            "host_acceptance_rate",
        ]

        chunk = chunk[cols]

        chunk.to_csv(
            primary_panel_out,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
            compression="gzip",
        )
        first_write = False
        stats["rows_kept"] += len(chunk)

    return stats


def write_window_extracts(df: pd.DataFrame, out_dir: Path) -> Dict[str, int]:
    counts = {}
    for m in (1, 2, 3):
        sub = df[df[f"in_bw_{m}m"] == 1].copy()
        out_path = out_dir / f"fact_listing_day_multicity_bw_{m}m.csv.gz"
        sub.to_csv(out_path, index=False, compression="gzip")
        counts[f"bw_{m}m_rows"] = len(sub)
    return counts


def build_monthly_aggregates(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    d = df.copy()
    d["year_month"] = d["date"].dt.to_period("M").astype(str)

    agg = (
        d.groupby(["city_slug", "city_name", "year_month"], as_index=False)
        .agg(
            n_listings_active=("listing_id", "nunique"),
            n_listing_days=("listing_id", "size"),
            mean_price_usd=("price_usd", "mean"),
            median_price_usd=("price_usd", "median"),
            mean_log_price=("log_price", "mean"),
            availability_rate=("available", "mean"),
            share_post_cutoff=("post_cutoff", "mean"),
            mean_days_from_cutoff=("days_from_cutoff", "mean"),
        )
        .sort_values(["city_slug", "year_month"])
    )

    agg.to_csv(out_csv, index=False)
    return agg


def qa_checks(df: pd.DataFrame, monthly_df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    qa = {}

    # Coverage by city/date
    coverage = (
        df.groupby(["city_slug", "date"], as_index=False)
        .agg(n_listing_days=("listing_id", "size"), n_unique_listings=("listing_id", "nunique"), availability_rate=("available", "mean"))
        .sort_values(["city_slug", "date"])
    )
    coverage.to_csv(out_dir / "qa_coverage_city_date.csv", index=False)

    # Missingness by city
    miss_rows = []
    for city, g in df.groupby("city_slug"):
        for col in KEY_MODEL_FIELDS:
            miss_rows.append(
                {
                    "city_slug": city,
                    "field": col,
                    "missing_rate": float(g[col].isna().mean()),
                    "n_obs": int(len(g)),
                }
            )
    missingness = pd.DataFrame(miss_rows)
    missingness.to_csv(out_dir / "qa_missingness_city_field.csv", index=False)

    # Support near cutoff
    support_rows = []
    for city, gcity in df.groupby("city_slug"):
        for m in (1, 2, 3):
            g = gcity[gcity[f"in_bw_{m}m"] == 1]
            pre = g[g["post_cutoff"] == 0]
            post = g[g["post_cutoff"] == 1]
            support_rows.append(
                {
                    "city_slug": city,
                    "window_months": m,
                    "pre_listing_days": int(len(pre)),
                    "post_listing_days": int(len(post)),
                    "pre_unique_listings": int(pre["listing_id"].nunique()),
                    "post_unique_listings": int(post["listing_id"].nunique()),
                    "thin_support_flag": int((len(pre) < 5000) or (len(post) < 5000)),
                }
            )
    support = pd.DataFrame(support_rows)
    support.to_csv(out_dir / "qa_support_near_cutoff.csv", index=False)

    # First-stage prep tables (simple, transparent prep summaries)
    fs_rows = []
    for city, gcity in df.groupby("city_slug"):
        for m in (1, 2, 3):
            g = gcity[gcity[f"in_bw_{m}m"] == 1]
            for post, gp in g.groupby("post_cutoff"):
                fs_rows.append(
                    {
                        "city_slug": city,
                        "window_months": m,
                        "post_cutoff": int(post),
                        "n_obs": int(len(gp)),
                        "n_unique_listings": int(gp["listing_id"].nunique()),
                        "mean_available": float(gp["available"].mean()),
                        "mean_price_usd": float(gp["price_usd"].mean()),
                        "mean_log_price": float(gp["log_price"].mean()),
                        "mean_host_response_rate": float(gp["host_response_rate"].mean(skipna=True)),
                        "mean_host_acceptance_rate": float(gp["host_acceptance_rate"].mean(skipna=True)),
                    }
                )
    fs = pd.DataFrame(fs_rows)
    fs.to_csv(out_dir / "qa_first_stage_prep_city_window_post.csv", index=False)

    # Key uniqueness
    dup_count = int(df.duplicated(subset=["city_slug", "listing_id", "date"]).sum())
    qa["duplicate_city_listing_date"] = str(dup_count)

    # Window nesting consistency
    bad_nesting = int(((df["in_bw_1m"] > df["in_bw_2m"]) | (df["in_bw_2m"] > df["in_bw_3m"])).sum())
    qa["bad_window_nesting_rows"] = str(bad_nesting)

    # Cutoff alignment check
    misaligned = int((df.loc[df["days_from_cutoff"] == 0, "post_cutoff"] != 1).sum())
    qa["misaligned_cutoff_rows"] = str(misaligned)

    # Daily vs monthly reconciliation (city-month means)
    d = df.copy()
    d["year_month"] = d["date"].dt.to_period("M").astype(str)
    rec = (
        d.groupby(["city_slug", "year_month"], as_index=False)
        .agg(recalc_mean_price_usd=("price_usd", "mean"), recalc_n_listing_days=("listing_id", "size"))
        .merge(monthly_df[["city_slug", "year_month", "mean_price_usd", "n_listing_days"]], on=["city_slug", "year_month"], how="left")
    )
    rec["abs_mean_price_diff"] = (rec["recalc_mean_price_usd"] - rec["mean_price_usd"]).abs()
    rec["n_listing_days_diff"] = rec["recalc_n_listing_days"] - rec["n_listing_days"]
    rec.to_csv(out_dir / "qa_daily_monthly_reconciliation.csv", index=False)
    qa["max_abs_mean_price_diff_city_month"] = f"{rec['abs_mean_price_diff'].max():.10f}"
    qa["max_n_listing_days_diff_city_month"] = str(int(rec["n_listing_days_diff"].abs().max()))

    pd.DataFrame([qa]).to_csv(out_dir / "qa_checks_summary.csv", index=False)
    return qa


def write_day2_report(
    out_path: Path,
    selected: List[SelectedCity],
    panel_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    qa_summary: Dict[str, str],
    pooled_cutoff: str,
    secondary_cutoff: str,
) -> None:
    selected_names = ", ".join([s.city_name for s in selected])
    city_counts = panel_df.groupby("city_slug")["listing_id"].nunique().sort_values(ascending=False)

    lines = [
        "# Day 2 Execution Status",
        "",
        "## Scope completed",
        "- Built multi-city daily listing-level panel (primary) with 8-city coverage.",
        "- Built secondary city-month aggregates for robustness and communication.",
        "- Exported ±1m/±2m/±3m windowed datasets around cutoff.",
        "- Exported QA outputs: coverage, missingness, support near cutoff, first-stage prep, and consistency checks.",
        "",
        "## City scope used",
        f"{selected_names}",
        "",
        "## Cutoff mapping",
        f"- Primary pooled fallback cutoff: **{pooled_cutoff}**",
        f"- Secondary pooled sensitivity cutoff: **{secondary_cutoff}**",
        "- Reason for pooled fallback: city-specific Smart Pricing rollout dates were not credibly observed in source files; fallback kept auditable in `city_cutoff_map.csv`.",
        "",
        "## Panel size summary",
        f"- Daily panel rows: **{len(panel_df):,}**",
        f"- Daily panel unique city-listings: **{panel_df[['city_slug', 'listing_id']].drop_duplicates().shape[0]:,}**",
        f"- Monthly aggregate rows: **{len(monthly_df):,}**",
        "",
        "### Unique listings by city",
        "",
        "| city_slug | unique_listings |",
        "|---|---:|",
    ]

    for city_slug, n in city_counts.items():
        lines.append(f"| {city_slug} | {int(n):,} |")

    lines += [
        "",
        "## QA check highlights",
        f"- Duplicate keys (`city_slug`,`listing_id`,`date`): **{qa_summary['duplicate_city_listing_date']}**",
        f"- Window nesting violations: **{qa_summary['bad_window_nesting_rows']}**",
        f"- Cutoff alignment violations (`days_from_cutoff == 0` but not post): **{qa_summary['misaligned_cutoff_rows']}**",
        f"- Max daily-vs-monthly mean price diff: **{qa_summary['max_abs_mean_price_diff_city_month']}**",
        f"- Max daily-vs-monthly listing-day count diff: **{qa_summary['max_n_listing_days_diff_city_month']}**",
        "",
        "## Economist rationale (data-frequency choice)",
        "- **Daily panel (primary)** preserves within-city, within-window timing variation needed for local identification around the cutoff.",
        "- **Monthly aggregates (secondary)** provide lower-noise trend robustness, communication clarity, and a useful check against daily compositional volatility.",
        "",
        "## Output locations",
        "- `data/processed/day2/fact_listing_day_multicity.csv.gz`",
        "- `data/processed/day2/fact_listing_day_multicity_bw_1m.csv.gz`",
        "- `data/processed/day2/fact_listing_day_multicity_bw_2m.csv.gz`",
        "- `data/processed/day2/fact_listing_day_multicity_bw_3m.csv.gz`",
        "- `data/processed/day2/agg_city_month_multicity.csv`",
        "- `data/processed/day2/city_cutoff_map.csv`",
        "- `data/processed/day2/city_selection_audit.csv`",
        "- `data/processed/day2/qa/*`",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def update_readme_day2(readme_path: Path) -> None:
    text = readme_path.read_text(encoding="utf-8")

    section_header = "## Day 2 Multi-City Build"
    if section_header in text:
        return

    section = """
## Day 2 Multi-City Build

Day 2 extends the Boston-focused setup into an 8-city multicity panel with explicit cutoff mapping, daily identification windows, and monthly robustness aggregates.

### What was added
- **Primary dataset:** `fact_listing_day_multicity` (daily listing-level panel).
- **Secondary dataset:** `agg_city_month_multicity` (city-month robustness panel).
- **Window extracts:** ±1, ±2, and ±3 month datasets around the assigned city cutoff.
- **QA outputs:** city-date coverage, missingness reports, support near cutoff, first-stage prep tables, and consistency checks.

### Cutoff policy for Day 2
- Pipeline supports city-specific cutoff overrides when credible dates are available.
- Where city-specific rollout dates are not available, Day 2 uses a pooled fallback cutoff and records this decision in `city_cutoff_map` with source metadata.

### Data-frequency rationale
- **Daily (primary):** needed for local identification around cutoff timing in fuzzy-RDD style designs.
- **Monthly (secondary):** used as a robustness and communication layer to validate that directional patterns are not artifacts of high-frequency noise.
""".strip()

    text = text.rstrip() + "\n\n" + section + "\n"
    readme_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Day 2 multicity daily/monthly panels")
    p.add_argument("--repo-root", type=str, default=".")
    p.add_argument("--pooled-cutoff", type=str, default="2025-09-01")
    p.add_argument("--secondary-cutoff", type=str, default="2025-10-01")
    p.add_argument("--cutoff-overrides", type=str, default="")
    p.add_argument("--chunk-size", type=int, default=500000)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    repo = Path(args.repo_root).resolve()
    data_raw = repo / "data" / "raw" / "day2"
    data_proc = repo / "data" / "processed" / "day2"
    qa_dir = data_proc / "qa"
    docs_dir = repo / "docs"

    ensure_dirs([data_raw, data_proc, qa_dir, docs_dir, repo / "scripts"])

    selected = select_cities()

    # Verify URLs and download
    print("[1/7] Validating and downloading selected city snapshots...")
    for c in selected:
        cal_url = f"{c.base_url}/calendar.csv.gz"
        lis_url = f"{c.base_url}/listings.csv.gz"

        if not head_ok(cal_url) or not head_ok(lis_url):
            raise RuntimeError(f"Missing source files for {c.city_slug} at {c.base_url}")

        city_raw = data_raw / c.city_slug
        ensure_dirs([city_raw])
        download_file(cal_url, city_raw / "calendar.csv.gz")
        download_file(lis_url, city_raw / "listings.csv.gz")

    write_city_selection_audit(selected, data_proc / "city_selection_audit.csv")

    # Cutoff map
    print("[2/7] Building cutoff map...")
    cutoff_override_path = Path(args.cutoff_overrides).resolve() if args.cutoff_overrides else None
    cutoff_overrides = load_city_specific_cutoff_overrides(cutoff_override_path)

    cutoff_map_df = build_cutoff_map(
        selected=selected,
        pooled_cutoff=args.pooled_cutoff,
        secondary_cutoff=args.secondary_cutoff,
        cutoff_overrides=cutoff_overrides,
        out_csv=data_proc / "city_cutoff_map.csv",
    )

    cutoff_lookup = dict(zip(cutoff_map_df["city_slug"], cutoff_map_df["cutoff_date_primary"]))
    cutoff_source_lookup = dict(zip(cutoff_map_df["city_slug"], cutoff_map_df["source_type"]))

    # Primary panel write
    print("[3/7] Building primary daily panel...")
    primary_panel_out = data_proc / "fact_listing_day_multicity.csv.gz"
    if primary_panel_out.exists():
        primary_panel_out.unlink()

    city_stats_rows = []
    for c in selected:
        print(f"  - processing {c.city_slug} ({c.snapshot_date})")
        city_raw = data_raw / c.city_slug
        listings_features = load_listings_features(city_raw / "listings.csv.gz", c.snapshot_date)
        stats = process_city_calendar(
            city=c,
            calendar_gz_path=city_raw / "calendar.csv.gz",
            listings_features=listings_features,
            city_cutoff=cutoff_lookup[c.city_slug],
            cutoff_source=cutoff_source_lookup[c.city_slug],
            primary_panel_out=primary_panel_out,
            chunk_size=args.chunk_size,
        )
        city_stats_rows.append(
            {
                "city_slug": c.city_slug,
                "snapshot_date": c.snapshot_date,
                "rows_in_calendar": stats["rows_in"],
                "rows_kept_in_panel": stats["rows_kept"],
            }
        )

    pd.DataFrame(city_stats_rows).to_csv(data_proc / "build_city_stats.csv", index=False)

    print("[4/7] Loading primary panel and building windows/monthly aggregates...")
    panel_df = pd.read_csv(primary_panel_out, compression="gzip", low_memory=False, parse_dates=["date"])

    # Ensure dtypes
    for c in ["available", "post_cutoff", "in_bw_1m", "in_bw_2m", "in_bw_3m"]:
        panel_df[c] = panel_df[c].astype(int)

    window_counts = write_window_extracts(panel_df, data_proc)

    monthly_df = build_monthly_aggregates(panel_df, data_proc / "agg_city_month_multicity.csv")

    print("[5/7] Running QA checks...")
    qa_summary = qa_checks(panel_df, monthly_df, qa_dir)

    print("[6/7] Writing Day 2 report + README updates...")
    write_day2_report(
        out_path=docs_dir / "DAY2_STATUS.md",
        selected=selected,
        panel_df=panel_df,
        monthly_df=monthly_df,
        qa_summary=qa_summary,
        pooled_cutoff=args.pooled_cutoff,
        secondary_cutoff=args.secondary_cutoff,
    )

    update_readme_day2(repo / "README.md")

    summary = {
        "selected_cities": [c.city_slug for c in selected],
        "pooled_cutoff": args.pooled_cutoff,
        "secondary_cutoff": args.secondary_cutoff,
        "panel_rows": int(len(panel_df)),
        "monthly_rows": int(len(monthly_df)),
        "window_counts": window_counts,
        "qa_summary": qa_summary,
    }
    (data_proc / "day2_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[7/7] Done.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
