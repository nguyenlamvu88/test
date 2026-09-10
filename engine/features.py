from __future__ import annotations

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


FEATURE_VERSION = "daily_v1"
LABEL_VERSION = "forward_5d_v1"


def build_daily_features(bars: pd.DataFrame, mentions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create features using information available by each session close only."""
    if bars.empty:
        return pd.DataFrame()
    frames = []
    mentions = mentions if mentions is not None else pd.DataFrame()
    for ticker, group in bars.groupby("ticker", sort=True):
        daily = group.copy().sort_values("session_date").reset_index(drop=True)
        daily["asof_date"] = pd.to_datetime(daily["session_date"]).dt.date
        close = pd.to_numeric(daily["close"], errors="coerce")
        volume = pd.to_numeric(daily["volume"], errors="coerce")
        daily["return_1d"] = close.pct_change()
        daily["return_5d_lag"] = close.pct_change(5)
        baseline_volume = volume.shift(1).rolling(20, min_periods=5).median()
        daily["rvol_20d"] = volume / baseline_volume
        daily["dollar_volume"] = close * volume
        daily["realized_vol_20d"] = daily["return_1d"].shift(1).rolling(20, min_periods=5).std()

        social = _daily_social(ticker, daily["asof_date"], mentions)
        daily = daily.merge(social, on="asof_date", how="left")
        for column in ["mentions_1d", "mentions_3d", "unique_authors_3d", "communities_3d", "attention_accel"]:
            daily[column] = daily[column].fillna(0)
        daily["feature_version"] = FEATURE_VERSION
        frames.append(daily[[
            "ticker", "asof_date", "close", "return_1d", "return_5d_lag",
            "rvol_20d", "dollar_volume", "realized_vol_20d", "mentions_1d",
            "mentions_3d", "unique_authors_3d", "communities_3d",
            "attention_accel", "feature_version",
        ]])
    return pd.concat(frames, ignore_index=True)


def _daily_social(ticker: str, dates: pd.Series, mentions: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.DataFrame({"asof_date": list(dates)})
    if mentions.empty:
        return calendar.assign(
            mentions_1d=0, mentions_3d=0, unique_authors_3d=0,
            communities_3d=0, attention_accel=0.0,
        )
    social = mentions[mentions["ticker"] == ticker].copy()
    if social.empty:
        return calendar.assign(
            mentions_1d=0, mentions_3d=0, unique_authors_3d=0,
            communities_3d=0, attention_accel=0.0,
        )
    social["created_at"] = pd.to_datetime(social["created_at"], utc=True)
    social = social.sort_values("created_at").reset_index(drop=True)
    created = social["created_at"].to_numpy(dtype="datetime64[ns]")
    authors = social["author"].to_numpy()
    communities = social["community"].to_numpy()
    market_tz = ZoneInfo("America/New_York")
    rows = []
    for asof in calendar["asof_date"]:
        # The state is stamped at the regular-session close, including DST.
        # Later posts on the same UTC date are future information and excluded.
        close_local = pd.Timestamp(asof).tz_localize(market_tz) + pd.Timedelta(hours=16)
        cutoff = close_local.tz_convert("UTC")
        cutoff_ns = cutoff.tz_localize(None).to_datetime64()
        start_ns = (cutoff - pd.Timedelta(hours=72)).tz_localize(None).to_datetime64()
        split_ns = (cutoff - pd.Timedelta(hours=24)).tz_localize(None).to_datetime64()
        left = int(np.searchsorted(created, start_ns, side="right"))
        split = int(np.searchsorted(created, split_ns, side="right"))
        right = int(np.searchsorted(created, cutoff_ns, side="right"))
        recent_count = right - left
        today_count = right - split
        prior_daily = (split - left) / 2.0
        rows.append({
            "asof_date": asof,
            "mentions_1d": today_count,
            "mentions_3d": recent_count,
            "unique_authors_3d": int(pd.Series(authors[left:right]).nunique()),
            "communities_3d": int(pd.Series(communities[left:right]).nunique()),
            "attention_accel": float((today_count + 1) / (prior_daily + 1)),
        })
    return pd.DataFrame(rows)


def build_outcome_labels(
    bars: pd.DataFrame,
    horizon: int = 5,
    target: float = 0.50,
    adverse: float = 0.20,
) -> pd.DataFrame:
    """Label forward outcomes; same-session target/adverse hits remain ambiguous."""
    frames = []
    for ticker, group in bars.groupby("ticker", sort=True):
        daily = group.copy().sort_values("session_date").reset_index(drop=True)
        row_count = max(0, len(daily) - horizon)
        if row_count == 0:
            continue
        entry = pd.to_numeric(daily["close"], errors="coerce").to_numpy(dtype=float)[:row_count]
        valid = np.isfinite(entry) & (entry > 0)
        future_high = np.lib.stride_tricks.sliding_window_view(
            pd.to_numeric(daily["high"], errors="coerce").to_numpy(dtype=float)[1:], horizon
        )
        future_low = np.lib.stride_tricks.sliding_window_view(
            pd.to_numeric(daily["low"], errors="coerce").to_numpy(dtype=float)[1:], horizon
        )
        future_close = np.lib.stride_tricks.sliding_window_view(
            pd.to_numeric(daily["close"], errors="coerce").to_numpy(dtype=float)[1:], horizon
        )
        high_returns = future_high / entry[:, None] - 1
        low_returns = future_low / entry[:, None] - 1
        close_returns = future_close / entry[:, None] - 1
        target_mask = high_returns >= target
        adverse_mask = low_returns <= -adverse
        target_any = target_mask.any(axis=1)
        adverse_any = adverse_mask.any(axis=1)
        target_session = np.where(target_any, target_mask.argmax(axis=1) + 1, np.nan)
        adverse_session = np.where(adverse_any, adverse_mask.argmax(axis=1) + 1, np.nan)
        ambiguous = target_any & adverse_any & (target_session == adverse_session)
        outcome = np.full(row_count, "non_runner", dtype=object)
        clean = target_any & (~adverse_any | (target_session < adverse_session))
        outcome[target_any & ~clean] = "wild_50"
        outcome[clean] = "clean_50"
        outcome[ambiguous] = "ambiguous_50"
        frame = pd.DataFrame({
            "ticker": ticker,
            "asof_date": pd.to_datetime(daily["session_date"].iloc[:row_count]).dt.date,
            "horizon_sessions": horizon,
            "target_return": target,
            "forward_return_1d": close_returns[:, 0],
            "forward_return_3d": close_returns[:, min(2, horizon - 1)],
            "forward_return_5d": close_returns[:, -1],
            "forward_mfe": np.nanmax(high_returns, axis=1),
            "forward_mae": np.nanmin(low_returns, axis=1),
            "target_hit_session": target_session,
            "adverse_hit_session": adverse_session,
            "ambiguous_same_session": ambiguous,
            "outcome_class": outcome,
            "label_version": LABEL_VERSION,
        })
        frames.append(frame.loc[valid])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
