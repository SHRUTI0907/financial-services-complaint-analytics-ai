from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.complaints import add_month


def detect_spikes(df: pd.DataFrame, group_col: str = "product", min_history: int = 6) -> pd.DataFrame:
    dated = add_month(df).dropna(subset=["month"])
    counts = dated.groupby([group_col, "month"], dropna=False).size().reset_index(name="complaints")
    rows = []
    for group, hist in counts.groupby(group_col):
        hist = hist.sort_values("month").copy()
        hist["baseline_mean"] = hist["complaints"].shift(1).rolling(12, min_periods=min_history).mean()
        hist["baseline_std"] = hist["complaints"].shift(1).rolling(12, min_periods=min_history).std(ddof=0)
        hist["z_score"] = (hist["complaints"] - hist["baseline_mean"]) / hist["baseline_std"].replace(0, np.nan)
        hist["spike_flag"] = hist["z_score"] >= 2.0
        rows.append(hist)
    return pd.concat(rows, ignore_index=True) if rows else counts.assign(baseline_mean=np.nan, baseline_std=np.nan, z_score=np.nan, spike_flag=False)


def emerging_issue_score(df: pd.DataFrame, group_col: str = "issue") -> pd.DataFrame:
    spikes = detect_spikes(df, group_col=group_col, min_history=4)
    if spikes.empty:
        return spikes
    latest_month = spikes["month"].max()
    latest = spikes[spikes["month"] == latest_month].copy()
    latest["growth_component"] = latest["z_score"].fillna(0).clip(lower=0)
    max_volume = latest["complaints"].max() or 1
    latest["volume_component"] = latest["complaints"] / max_volume
    latest["emerging_issue_score"] = (0.65 * latest["growth_component"] + 0.35 * latest["volume_component"]).round(3)
    return latest.sort_values("emerging_issue_score", ascending=False)
