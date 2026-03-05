#!/usr/bin/env python3
"""PSM-style High-vs-Low propensity DiD + event study around rollout.

Uses latent proxy from ml extension:
- Treatment: top quartile (high propensity)
- Control: bottom quartile (low propensity)
- Estimation: TWFE DiD and event-study (entity/date FE)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PSM + DiD around rollout")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--panel-file", default="data/processed/step2/fact_listing_day_multicity_bw_3m.csv.gz")
    p.add_argument("--proxy-file", default="data/processed/ml_extension/listing_latent_proxy.csv")
    p.add_argument("--event-min", type=int, default=-30)
    p.add_argument("--event-max", type=int, default=30)
    return p.parse_args()


def run_twfe_did(df: pd.DataFrame) -> pd.DataFrame:
    from linearmodels.panel import PanelOLS

    panel = df.set_index(["listing_id", "date"]).sort_index()
    y = panel["log_price"].astype(float)
    x = panel[["did"]].astype(float)

    mod = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    ci = res.conf_int()

    return pd.DataFrame(
        [
            {
                "term": "high_propensity_x_post",
                "coef": float(res.params.get("did", np.nan)),
                "se": float(res.std_errors.get("did", np.nan)),
                "t": float(res.tstats.get("did", np.nan)),
                "p": float(res.pvalues.get("did", np.nan)),
                "ci95_low": float(ci.loc["did"].iloc[0]) if "did" in ci.index else np.nan,
                "ci95_high": float(ci.loc["did"].iloc[1]) if "did" in ci.index else np.nan,
                "n_obs": int(res.nobs),
                "n_listings": int(df["listing_id"].nunique()),
            }
        ]
    )


def run_event_study(df: pd.DataFrame, event_min: int, event_max: int) -> pd.DataFrame:
    from linearmodels.panel import PanelOLS

    work = df.copy()
    terms = []
    for k in range(event_min, event_max + 1):
        if k == -1:
            continue
        c = f"event_{k}"
        work[c] = ((work["days_from_cutoff"] == k).astype(float) * work["high_propensity"].astype(float))
        terms.append(c)

    panel = work.set_index(["listing_id", "date"]).sort_index()
    y = panel["log_price"].astype(float)
    X = panel[terms].astype(float)

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    ci = res.conf_int()

    rows = []
    for t in terms:
        k = int(t.replace("event_", ""))
        if t in res.params.index:
            rows.append(
                {
                    "event_time": k,
                    "term": t,
                    "coef": float(res.params[t]),
                    "se": float(res.std_errors[t]),
                    "t": float(res.tstats[t]),
                    "p": float(res.pvalues[t]),
                    "ci95_low": float(ci.loc[t].iloc[0]),
                    "ci95_high": float(ci.loc[t].iloc[1]),
                    "status": "estimated",
                }
            )
        else:
            rows.append({"event_time": k, "term": t, "coef": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "status": "absorbed_or_dropped"})

    rows.append({"event_time": -1, "term": "reference", "coef": 0.0, "se": np.nan, "t": np.nan, "p": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "status": "reference_omitted"})
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


def plot_event_study(coef: pd.DataFrame, path: Path) -> None:
    plot_df = coef.loc[coef["status"] == "estimated"].copy()
    plt.figure(figsize=(9, 5))
    plt.errorbar(plot_df["event_time"], plot_df["coef"], yerr=1.96 * plot_df["se"], fmt="o", capsize=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Event Study: High vs Low Propensity (Reference t=-1)")
    plt.xlabel("Days from rollout cutoff")
    plt.ylabel("High propensity differential effect on log_price")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    panel_file = (repo / args.panel_file).resolve()
    proxy_file = (repo / args.proxy_file).resolve()

    out_dir = repo / "data" / "processed" / "ml_extension"
    out_dir.mkdir(parents=True, exist_ok=True)

    proxy = pd.read_csv(proxy_file, usecols=["listing_id", "latent_adoption_propensity_proxy"])
    q25 = float(proxy["latent_adoption_propensity_proxy"].quantile(0.25))
    q75 = float(proxy["latent_adoption_propensity_proxy"].quantile(0.75))

    proxy["high_propensity"] = (proxy["latent_adoption_propensity_proxy"] >= q75).astype(int)
    proxy["low_propensity"] = (proxy["latent_adoption_propensity_proxy"] <= q25).astype(int)
    proxy = proxy.loc[(proxy["high_propensity"] == 1) | (proxy["low_propensity"] == 1), ["listing_id", "high_propensity"]].copy()

    usecols = ["listing_id", "date", "log_price", "post_cutoff", "days_from_cutoff"]
    panel = pd.read_csv(panel_file, compression="gzip", usecols=usecols, low_memory=False)
    panel["listing_id"] = pd.to_numeric(panel["listing_id"], errors="coerce")
    panel["log_price"] = pd.to_numeric(panel["log_price"], errors="coerce")
    panel["post_cutoff"] = pd.to_numeric(panel["post_cutoff"], errors="coerce")
    panel["days_from_cutoff"] = pd.to_numeric(panel["days_from_cutoff"], errors="coerce")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    df = panel.merge(proxy, on="listing_id", how="inner")
    df = df.dropna(subset=["listing_id", "date", "log_price", "post_cutoff", "days_from_cutoff"]).copy()
    df["did"] = df["high_propensity"] * df["post_cutoff"]

    # Keep symmetric window around rollout for event-study comparability.
    df = df.loc[df["days_from_cutoff"].between(args.event_min, args.event_max)].copy()

    did = run_twfe_did(df)
    event = run_event_study(df, event_min=args.event_min, event_max=args.event_max)

    did_path = out_dir / "psm_did_twfe_results.csv"
    event_path = out_dir / "psm_did_event_study.csv"
    plot_path = out_dir / "psm_did_event_study_plot.png"
    summary_path = out_dir / "psm_did_summary.json"

    did.to_csv(did_path, index=False)
    event.to_csv(event_path, index=False)
    plot_event_study(event, plot_path)

    summary = {
        "panel_file": str(panel_file.relative_to(repo)),
        "proxy_file": str(proxy_file.relative_to(repo)),
        "event_window": [args.event_min, args.event_max],
        "quartiles": {"q25": q25, "q75": q75},
        "n_rows": int(len(df)),
        "n_listings": int(df["listing_id"].nunique()),
        "n_high": int(df.loc[df["high_propensity"] == 1, "listing_id"].nunique()),
        "n_low": int(df.loc[df["high_propensity"] == 0, "listing_id"].nunique()),
        "outputs": {
            "did": "data/processed/ml_extension/psm_did_twfe_results.csv",
            "event": "data/processed/ml_extension/psm_did_event_study.csv",
            "plot": "data/processed/ml_extension/psm_did_event_study_plot.png",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
