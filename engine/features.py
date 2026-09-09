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
    market_tz = ZoneInfo("America/New_York")
    rows = []
    for asof in calendar["asof_date"]:
        # The state is stamped at the regular-session close, including DST.
        # Later posts on the same UTC date are future information and excluded.
        close_local = pd.Timestamp(asof).tz_localize(market_tz) + pd.Timedelta(hours=16)
        cutoff = close_local.tz_convert("UTC")
        recent = social[
            (social["created_at"] <= cutoff)
            & (social["created_at"] > cutoff - pd.Timedelta(hours=72))
        ]
        today = recent[recent["created_at"] > cutoff - pd.Timedelta(hours=24)]
        previous = recent[recent["created_at"] <= cutoff - pd.Timedelta(hours=24)]
        prior_daily = len(previous) / 2.0
        rows.append({
            "asof_date": asof,
            "mentions_1d": int(len(today)),
            "mentions_3d": int(len(recent)),
            "unique_authors_3d": int(recent["author"].nunique()),
            "communities_3d": int(recent["community"].nunique()),
            "attention_accel": float((len(today) + 1) / (prior_daily + 1)),
        })
    return pd.DataFrame(rows)


def build_outcome_labels(
    bars: pd.DataFrame,
    horizon: int = 5,
    target: float = 0.50,
    adverse: float = 0.20,
) -> pd.DataFrame:
    """Label forward outcomes; same-session target/adverse hits remain ambiguous."""
    rows = []
    for ticker, group in bars.groupby("ticker", sort=True):
        daily = group.copy().sort_values("session_date").reset_index(drop=True)
        for index in range(max(0, len(daily) - horizon)):
            entry = float(daily.loc[index, "close"])
            if not np.isfinite(entry) or entry <= 0:
                continue
            future = daily.iloc[index + 1:index + horizon + 1]
            high_returns = pd.to_numeric(future["high"], errors="coerce") / entry - 1
            low_returns = pd.to_numeric(future["low"], errors="coerce") / entry - 1
            target_hits = np.flatnonzero(high_returns.to_numpy() >= target)
            adverse_hits = np.flatnonzero(low_returns.to_numpy() <= -adverse)
            target_session = int(target_hits[0] + 1) if len(target_hits) else None
            adverse_session = int(adverse_hits[0] + 1) if len(adverse_hits) else None
            ambiguous = bool(
                target_session is not None
                and adverse_session is not None
                and target_session == adverse_session
            )
            if target_session is None:
                outcome = "non_runner"
            elif adverse_session is None or target_session < adverse_session:
                outcome = "clean_50"
            else:
                outcome = "wild_50"
            if ambiguous:
                outcome = "ambiguous_50"
            rows.append({
                "ticker": ticker,
                "asof_date": pd.Timestamp(daily.loc[index, "session_date"]).date(),
                "horizon_sessions": horizon,
                "target_return": target,
                "forward_mfe": float(high_returns.max()),
                "forward_mae": float(low_returns.min()),
                "target_hit_session": target_session,
                "adverse_hit_session": adverse_session,
                "ambiguous_same_session": ambiguous,
                "outcome_class": outcome,
                "label_version": LABEL_VERSION,
            })
    return pd.DataFrame(rows)
