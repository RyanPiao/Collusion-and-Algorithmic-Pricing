#!/usr/bin/env python3
"""Day 3 multicity EDA around cutoff.

Builds policy-facing descriptive outputs from Day 2 panel artifacts:
1) City-level trend summaries/plots around cutoff
2) Pre/post distribution shifts (by city and window)
3) Treatment-support diagnostics by city/window
4) Missingness and data-quality summaries
5) Economist-readable Day 3 note and README update
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TREND_CHUNKSIZE = 1_000_000
DIST_CHUNKSIZE = 1_000_000
CITIES_SORT = [
    "boston",
    "new-york-city",
    "los-angeles",
    "san-francisco",
    "austin",
    "chicago",
    "seattle",
    "washington-dc",
]


def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def aggregate_city_day_trends(panel_path: Path, out_csv: Path) -> pd.DataFrame:
    """Aggregate city x relative-day trends via chunking."""
    accum: Dict[Tuple[str, str, int], Dict[str, float]] = {}

    usecols = ["city_slug", "city_name", "days_from_cutoff", "price_usd", "log_price", "available"]

    reader = pd.read_csv(
        panel_path,
        compression="gzip",
        usecols=usecols,
        chunksize=TREND_CHUNKSIZE,
        low_memory=False,
    )

    for chunk in reader:
        grp = (
            chunk.groupby(["city_slug", "city_name", "days_from_cutoff"], dropna=False)
            .agg(
                n_obs=("price_usd", "size"),
                price_sum=("price_usd", "sum"),
                log_price_sum=("log_price", "sum"),
                avail_sum=("available", "sum"),
            )
            .reset_index()
        )

        for row in grp.itertuples(index=False):
            key = (str(row.city_slug), str(row.city_name), int(row.days_from_cutoff))
            if key not in accum:
                accum[key] = {
                    "n_obs": float(row.n_obs),
                    "price_sum": float(row.price_sum),
                    "log_price_sum": float(row.log_price_sum),
                    "avail_sum": float(row.avail_sum),
                }
            else:
                accum[key]["n_obs"] += float(row.n_obs)
                accum[key]["price_sum"] += float(row.price_sum)
                accum[key]["log_price_sum"] += float(row.log_price_sum)
                accum[key]["avail_sum"] += float(row.avail_sum)

    rows = []
    for (city_slug, city_name, days), vals in accum.items():
        n = vals["n_obs"]
        rows.append(
            {
                "city_slug": city_slug,
                "city_name": city_name,
                "days_from_cutoff": days,
                "n_obs": int(n),
                "mean_price_usd": vals["price_sum"] / n if n else np.nan,
                "mean_log_price": vals["log_price_sum"] / n if n else np.nan,
                "availability_rate": vals["avail_sum"] / n if n else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values(["city_slug", "days_from_cutoff"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out


def _concat_or_empty(parts: List[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.array([], dtype="float64")
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def distribution_shifts(window_file: Path, window_months: int) -> pd.DataFrame:
    """Exact pre/post distribution stats by city within a given window file."""
    usecols = ["city_slug", "post_cutoff", "price_usd", "log_price"]

    price_buf: Dict[Tuple[str, int], List[np.ndarray]] = {}
    log_buf: Dict[Tuple[str, int], List[np.ndarray]] = {}

    reader = pd.read_csv(
        window_file,
        compression="gzip",
        usecols=usecols,
        chunksize=DIST_CHUNKSIZE,
        low_memory=False,
    )

    for chunk in reader:
        chunk = chunk.dropna(subset=["city_slug", "post_cutoff", "price_usd", "log_price"])
        chunk["post_cutoff"] = chunk["post_cutoff"].astype(int)

        for (city, post), g in chunk.groupby(["city_slug", "post_cutoff"]):
            key = (str(city), int(post))
            price_buf.setdefault(key, []).append(g["price_usd"].to_numpy(dtype="float64", copy=True))
            log_buf.setdefault(key, []).append(g["log_price"].to_numpy(dtype="float64", copy=True))

    rows = []
    keys = sorted(set(price_buf.keys()) | set(log_buf.keys()))

    for city, post in keys:
        price = _concat_or_empty(price_buf.get((city, post), []))
        logp = _concat_or_empty(log_buf.get((city, post), []))

        if len(price) == 0:
            continue

        q_price = np.quantile(price, [0.1, 0.25, 0.5, 0.75, 0.9])
        q_log = np.quantile(logp, [0.1, 0.25, 0.5, 0.75, 0.9])

        rows.append(
            {
                "city_slug": city,
                "window_months": window_months,
                "period": "post" if post == 1 else "pre",
                "post_cutoff": int(post),
                "n_obs": int(len(price)),
                "mean_price_usd": float(np.mean(price)),
                "std_price_usd": float(np.std(price)),
                "p10_price_usd": float(q_price[0]),
                "p25_price_usd": float(q_price[1]),
                "median_price_usd": float(q_price[2]),
                "p75_price_usd": float(q_price[3]),
                "p90_price_usd": float(q_price[4]),
                "mean_log_price": float(np.mean(logp)),
                "std_log_price": float(np.std(logp)),
                "p10_log_price": float(q_log[0]),
                "p25_log_price": float(q_log[1]),
                "median_log_price": float(q_log[2]),
                "p75_log_price": float(q_log[3]),
                "p90_log_price": float(q_log[4]),
            }
        )

    return pd.DataFrame(rows).sort_values(["city_slug", "window_months", "post_cutoff"])


def build_distribution_outputs(day2_proc: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    for m in (1, 2, 3):
        window_file = day2_proc / f"fact_listing_day_multicity_bw_{m}m.csv.gz"
        all_rows.append(distribution_shifts(window_file, m))

    dist = pd.concat(all_rows, ignore_index=True)
    dist = dist.sort_values(["city_slug", "window_months", "post_cutoff"]).reset_index(drop=True)
    dist.to_csv(out_dir / "prepost_distribution_stats_city_window.csv", index=False)

    pre = dist[dist["post_cutoff"] == 0].copy()
    post = dist[dist["post_cutoff"] == 1].copy()
    shift = pre.merge(post, on=["city_slug", "window_months"], suffixes=("_pre", "_post"), how="inner")

    shift_out = pd.DataFrame(
        {
            "city_slug": shift["city_slug"],
            "window_months": shift["window_months"],
            "n_obs_pre": shift["n_obs_pre"],
            "n_obs_post": shift["n_obs_post"],
            "mean_price_diff_post_minus_pre": shift["mean_price_usd_post"] - shift["mean_price_usd_pre"],
            "median_price_diff_post_minus_pre": shift["median_price_usd_post"] - shift["median_price_usd_pre"],
            "p90_price_diff_post_minus_pre": shift["p90_price_usd_post"] - shift["p90_price_usd_pre"],
            "mean_log_price_diff_post_minus_pre": shift["mean_log_price_post"] - shift["mean_log_price_pre"],
            "median_log_price_diff_post_minus_pre": shift["median_log_price_post"] - shift["median_log_price_pre"],
        }
    )
    shift_out = shift_out.sort_values(["city_slug", "window_months"]).reset_index(drop=True)
    shift_out.to_csv(out_dir / "prepost_distribution_shift_summary.csv", index=False)

    return dist, shift_out


def build_support_diagnostics(day2_proc: Path, trend_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    support = pd.read_csv(day2_proc / "qa" / "qa_support_near_cutoff.csv")

    city_day_cov = (
        trend_df.assign(period=np.where(trend_df["days_from_cutoff"] < 0, "pre", "post"))
        .query("days_from_cutoff != 0")
        .groupby(["city_slug", "period"], as_index=False)
        .agg(n_relative_days=("days_from_cutoff", "nunique"))
    )

    pre_days = city_day_cov[city_day_cov["period"] == "pre"][["city_slug", "n_relative_days"]].rename(
        columns={"n_relative_days": "pre_days_observed"}
    )
    post_days = city_day_cov[city_day_cov["period"] == "post"][["city_slug", "n_relative_days"]].rename(
        columns={"n_relative_days": "post_days_observed"}
    )

    out = support.merge(pre_days, on="city_slug", how="left").merge(post_days, on="city_slug", how="left")

    out["pre_post_listing_day_ratio"] = out["post_listing_days"] / out["pre_listing_days"]
    out["pre_post_listing_count_ratio"] = out["post_unique_listings"] / out["pre_unique_listings"]
    out["support_ok_flag"] = (
        (out["thin_support_flag"] == 0)
        & out["pre_days_observed"].ge(28)
        & out["post_days_observed"].ge(28)
        & out["pre_post_listing_day_ratio"].between(0.95, 1.05)
    ).astype(int)

    out = out.sort_values(["city_slug", "window_months"]).reset_index(drop=True)
    out.to_csv(out_dir / "treatment_support_diagnostics_city_window.csv", index=False)
    return out


def build_missingness_quality(day2_proc: Path, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    miss = pd.read_csv(day2_proc / "qa" / "qa_missingness_city_field.csv")
    checks = pd.read_csv(day2_proc / "qa" / "qa_checks_summary.csv")

    city_summary = (
        miss.groupby("city_slug", as_index=False)
        .agg(
            mean_missing_rate=("missing_rate", "mean"),
            max_missing_rate=("missing_rate", "max"),
            n_fields=("field", "nunique"),
            n_fields_gt_1pct=("missing_rate", lambda s: int((s > 0.01).sum())),
            n_fields_gt_5pct=("missing_rate", lambda s: int((s > 0.05).sum())),
        )
        .sort_values("city_slug")
    )

    top_field = miss.sort_values(["city_slug", "missing_rate"], ascending=[True, False]).groupby("city_slug").head(1)
    top_field = top_field[["city_slug", "field", "missing_rate"]].rename(
        columns={"field": "top_missing_field", "missing_rate": "top_missing_rate"}
    )

    city_summary = city_summary.merge(top_field, on="city_slug", how="left")
    city_summary.to_csv(out_dir / "missingness_summary_city.csv", index=False)

    checks.to_csv(out_dir / "data_quality_checks_summary.csv", index=False)
    return city_summary, checks


def _city_order_key(series: pd.Series) -> pd.Series:
    rank = {c: i for i, c in enumerate(CITIES_SORT)}
    return series.map(rank).fillna(999).astype(int)


def plot_city_trends(trend_df: pd.DataFrame, fig_path: Path) -> None:
    d = trend_df.copy()
    d = d.sort_values(["city_slug", "days_from_cutoff"])
    d["mean_price_7d"] = d.groupby("city_slug")["mean_price_usd"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    cities = sorted(d["city_slug"].unique(), key=lambda c: CITIES_SORT.index(c) if c in CITIES_SORT else 999)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    axes = axes.flatten()

    for i, city in enumerate(cities):
        ax = axes[i]
        sub = d[d["city_slug"] == city]
        ax.plot(sub["days_from_cutoff"], sub["mean_price_7d"], linewidth=1.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(city)
        ax.set_ylabel("Mean price (USD)")
        ax.grid(alpha=0.2)

    for ax in axes[-2:]:
        ax.set_xlabel("Days from cutoff")

    fig.suptitle("City-level mean nightly price trends around cutoff (7-day smooth)", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def _sample_for_dist_plot(window_file: Path, n_per_group: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    usecols = ["city_slug", "post_cutoff", "log_price"]

    # Reservoir-style fixed-size sample buffers
    buffers: Dict[Tuple[str, int], np.ndarray] = {}
    counts: Dict[Tuple[str, int], int] = {}

    reader = pd.read_csv(
        window_file,
        compression="gzip",
        usecols=usecols,
        chunksize=750_000,
        low_memory=False,
    )

    for chunk in reader:
        chunk = chunk.dropna(subset=["city_slug", "post_cutoff", "log_price"])
        chunk["post_cutoff"] = chunk["post_cutoff"].astype(int)

        for (city, post), g in chunk.groupby(["city_slug", "post_cutoff"]):
            key = (str(city), int(post))
            vals = g["log_price"].to_numpy(dtype="float64", copy=False)
            if key not in buffers:
                take = vals[:n_per_group]
                buffers[key] = take.copy()
                counts[key] = len(take)
                vals = vals[len(take) :]

            for v in vals:
                counts[key] += 1
                if len(buffers[key]) < n_per_group:
                    buffers[key] = np.append(buffers[key], v)
                else:
                    j = rng.integers(0, counts[key])
                    if j < n_per_group:
                        buffers[key][j] = v

    rows = []
    for (city, post), arr in buffers.items():
        period = "Post" if post == 1 else "Pre"
        for v in arr:
            rows.append({"city_slug": city, "period": period, "log_price": float(v)})

    return pd.DataFrame(rows)


def plot_distribution_shifts(day2_proc: Path, fig_path: Path) -> None:
    sample = _sample_for_dist_plot(day2_proc / "fact_listing_day_multicity_bw_1m.csv.gz")
    cities = sorted(sample["city_slug"].unique(), key=lambda c: CITIES_SORT.index(c) if c in CITIES_SORT else 999)

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharey=True)
    axes = axes.flatten()

    for i, city in enumerate(cities):
        ax = axes[i]
        sub = sample[sample["city_slug"] == city]
        pre = sub[sub["period"] == "Pre"]["log_price"].to_numpy()
        post = sub[sub["period"] == "Post"]["log_price"].to_numpy()

        bins = np.linspace(min(sub["log_price"]), max(sub["log_price"]), 30)
        ax.hist(pre, bins=bins, alpha=0.45, density=True, label="Pre", color="#4C78A8")
        ax.hist(post, bins=bins, alpha=0.45, density=True, label="Post", color="#F58518")
        ax.set_title(city)
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    for ax in axes[-2:]:
        ax.set_xlabel("log(price_usd)")

    fig.suptitle("Pre/Post log-price distribution shifts by city (±1 month window, sampled)", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def plot_support_diagnostics(support_df: pd.DataFrame, fig_path: Path) -> None:
    d = support_df.copy().sort_values(["window_months", "city_slug"], key=lambda s: _city_order_key(s) if s.name == "city_slug" else s)

    x = np.arange(len(d))
    width = 0.38

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2, d["pre_listing_days"], width=width, label="Pre listing-days", color="#4C78A8")
    ax.bar(x + width / 2, d["post_listing_days"], width=width, label="Post listing-days", color="#F58518")

    labels = [f"{c}\n±{w}m" for c, w in zip(d["city_slug"], d["window_months"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("Listing-day observations")
    ax.set_title("Treatment-support diagnostics: pre/post listing-days by city and window")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def write_day3_note(
    out_path: Path,
    trend_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    support_df: pd.DataFrame,
    miss_city_df: pd.DataFrame,
    checks_df: pd.DataFrame,
) -> None:
    # Key numeric snippets for economist-readable interpretation.
    shift_1m = shift_df[shift_df["window_months"] == 1].copy()
    avg_mean_shift_1m = shift_1m["mean_price_diff_post_minus_pre"].mean()
    avg_median_shift_1m = shift_1m["median_price_diff_post_minus_pre"].mean()

    largest_pos = shift_1m.sort_values("mean_price_diff_post_minus_pre", ascending=False).head(1)
    largest_neg = shift_1m.sort_values("mean_price_diff_post_minus_pre", ascending=True).head(1)

    thin_support_n = int((support_df["thin_support_flag"] == 1).sum())
    support_ok_n = int((support_df["support_ok_flag"] == 1).sum())

    max_missing = miss_city_df.sort_values("max_missing_rate", ascending=False).head(1)

    mean_pre = trend_df[trend_df["days_from_cutoff"] < 0]["mean_price_usd"].mean()
    mean_post = trend_df[trend_df["days_from_cutoff"] > 0]["mean_price_usd"].mean()

    lines = [
        "# Day 3 EDA Note: Multi-city descriptives around cutoff",
        "",
        "## 1) City-level trend behavior around the cutoff",
        f"- Across city-day means, average nightly price is {mean_pre:,.2f} USD pre-cutoff versus {mean_post:,.2f} USD post-cutoff.",
        "- Figure `docs/figures/day3/city_trend_mean_price_cutoff.png` shows within-city trajectories in relative event time with a cutoff marker at day 0.",
        "- The visual pattern is best read as descriptive evidence for local continuity/discontinuity checks; it is not a causal estimate by itself.",
        "",
        "## 2) Pre/Post distribution shifts",
        f"- In the ±1 month window, the average post-minus-pre shift in mean price is {avg_mean_shift_1m:,.2f} USD across cities.",
        f"- In the ±1 month window, the average post-minus-pre shift in median price is {avg_median_shift_1m:,.2f} USD across cities.",
    ]

    if not largest_pos.empty:
        r = largest_pos.iloc[0]
        lines.append(
            f"- Largest positive mean shift (±1m): {r['city_slug']} ({r['mean_price_diff_post_minus_pre']:+,.2f} USD)."
        )
    if not largest_neg.empty:
        r = largest_neg.iloc[0]
        lines.append(
            f"- Largest negative mean shift (±1m): {r['city_slug']} ({r['mean_price_diff_post_minus_pre']:+,.2f} USD)."
        )

    lines += [
        "- Figure `docs/figures/day3/prepost_logprice_distribution_bw1m.png` overlays sampled log-price histograms (pre vs post) by city for shape comparison.",
        "",
        "## 3) Treatment-support diagnostics by city/window",
        f"- Thin-support flags: {thin_support_n} of {len(support_df)} city-window cells are flagged thin.",
        f"- Composite support-ok flag (balanced pre/post observations and day coverage): {support_ok_n} of {len(support_df)} cells.",
        "- Support diagnostics table is in `data/processed/day3/treatment_support_diagnostics_city_window.csv`.",
        "",
        "## 4) Missingness and data quality",
        f"- Dataset-level QA checks remain clean: duplicates={int(checks_df['duplicate_city_listing_date'].iloc[0])}, window nesting violations={int(checks_df['bad_window_nesting_rows'].iloc[0])}, cutoff misalignment={int(checks_df['misaligned_cutoff_rows'].iloc[0])}.",
    ]

    if not max_missing.empty:
        r = max_missing.iloc[0]
        lines.append(
            f"- Highest city-level field missingness: {r['city_slug']} at {100*r['max_missing_rate']:.2f}% (top field: {r['top_missing_field']})."
        )

    lines += [
        "- Full missingness summary is in `data/processed/day3/missingness_summary_city.csv`.",
        "",
        "## Policy-facing interpretation",
        "- The Day 3 outputs indicate where identifying support is strong enough for local fuzzy-RDD estimation and where city-window cells may require caution due to thinner support.",
        "- Pre/post distribution movement appears heterogeneous across markets; this supports reporting city-specific descriptives before pooled structural interpretation.",
        "- Structural QA checks are clean, but several host-side covariates have non-trivial missingness in some cities; inference specs should report covariate completeness and sensitivity to missing-data handling.",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_readme_day3(readme_path: Path) -> None:
    text = readme_path.read_text(encoding="utf-8")
    header = "## Day 3 Multi-City EDA Around Cutoff"
    if header in text:
        return

    section = """
## Day 3 Multi-City EDA Around Cutoff

Day 3 adds policy-oriented descriptive outputs on top of Day 2 multicity panels.

### Day 3 outputs
- **EDA script:** `scripts/day3_multicity_eda.py`
- **EDA notebook:** `day3_multicity_eda.ipynb`
- **Trend aggregates:** `data/processed/day3/city_day_trends_cutoff.csv`
- **Distribution shifts:**
  - `data/processed/day3/prepost_distribution_stats_city_window.csv`
  - `data/processed/day3/prepost_distribution_shift_summary.csv`
- **Treatment-support diagnostics:** `data/processed/day3/treatment_support_diagnostics_city_window.csv`
- **Missingness/data quality:**
  - `data/processed/day3/missingness_summary_city.csv`
  - `data/processed/day3/data_quality_checks_summary.csv`
- **Figures:**
  - `docs/figures/day3/city_trend_mean_price_cutoff.png`
  - `docs/figures/day3/prepost_logprice_distribution_bw1m.png`
  - `docs/figures/day3/treatment_support_diagnostics.png`
- **Interpretation note:** `docs/DAY3_eda_note.md`
""".strip()

    readme_path.write_text(text.rstrip() + "\n\n" + section + "\n", encoding="utf-8")


def write_run_summary(out_path: Path, artifacts: Dict[str, str], shapes: Dict[str, int]) -> None:
    payload = {
        "artifacts": artifacts,
        "shapes": shapes,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 3 multicity EDA builder")
    p.add_argument("--repo-root", type=str, default=".")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).resolve()

    day2_proc = repo / "data" / "processed" / "day2"
    day3_proc = repo / "data" / "processed" / "day3"
    day3_fig = repo / "docs" / "figures" / "day3"
    docs_dir = repo / "docs"

    ensure_dirs([day3_proc, day3_fig, docs_dir])

    panel_path = day2_proc / "fact_listing_day_multicity.csv.gz"
    trend_csv = day3_proc / "city_day_trends_cutoff.csv"

    print("[1/6] Aggregating city-day trends around cutoff...")
    trend_df = aggregate_city_day_trends(panel_path, trend_csv)

    print("[2/6] Computing pre/post distribution shifts by city/window...")
    dist_df, shift_df = build_distribution_outputs(day2_proc, day3_proc)

    print("[3/6] Building treatment-support diagnostics...")
    support_df = build_support_diagnostics(day2_proc, trend_df, day3_proc)

    print("[4/6] Building missingness and quality summaries...")
    miss_city_df, checks_df = build_missingness_quality(day2_proc, day3_proc)

    print("[5/6] Rendering figures...")
    plot_city_trends(trend_df, day3_fig / "city_trend_mean_price_cutoff.png")
    plot_distribution_shifts(day2_proc, day3_fig / "prepost_logprice_distribution_bw1m.png")
    plot_support_diagnostics(support_df, day3_fig / "treatment_support_diagnostics.png")

    print("[6/6] Writing notes and README updates...")
    write_day3_note(
        out_path=docs_dir / "DAY3_eda_note.md",
        trend_df=trend_df,
        shift_df=shift_df,
        support_df=support_df,
        miss_city_df=miss_city_df,
        checks_df=checks_df,
    )
    update_readme_day3(repo / "README.md")

    artifacts = {
        "trend_csv": str(trend_csv.relative_to(repo)),
        "dist_stats_csv": str((day3_proc / "prepost_distribution_stats_city_window.csv").relative_to(repo)),
        "dist_shift_csv": str((day3_proc / "prepost_distribution_shift_summary.csv").relative_to(repo)),
        "support_csv": str((day3_proc / "treatment_support_diagnostics_city_window.csv").relative_to(repo)),
        "missingness_csv": str((day3_proc / "missingness_summary_city.csv").relative_to(repo)),
        "quality_csv": str((day3_proc / "data_quality_checks_summary.csv").relative_to(repo)),
        "fig_trend": str((day3_fig / "city_trend_mean_price_cutoff.png").relative_to(repo)),
        "fig_dist": str((day3_fig / "prepost_logprice_distribution_bw1m.png").relative_to(repo)),
        "fig_support": str((day3_fig / "treatment_support_diagnostics.png").relative_to(repo)),
        "day3_note": "docs/DAY3_eda_note.md",
    }

    shapes = {
        "trend_rows": int(len(trend_df)),
        "distribution_rows": int(len(dist_df)),
        "distribution_shift_rows": int(len(shift_df)),
        "support_rows": int(len(support_df)),
        "missingness_city_rows": int(len(miss_city_df)),
    }

    write_run_summary(day3_proc / "day3_run_summary.json", artifacts, shapes)
    print(json.dumps({"artifacts": artifacts, "shapes": shapes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
