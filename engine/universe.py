from __future__ import annotations

from datetime import date

import pandas as pd


UNIVERSE_VERSION = "reddit_low_price_v1"


def evaluate_candidate(
    frame: pd.DataFrame,
    reference_date: date,
    max_price: float = 10.0,
    min_dollar_volume: float = 250_000.0,
    min_sessions: int = 252,
) -> dict:
    """Apply the frozen v1 price/liquidity rules to one candidate."""
    if frame.empty or len(frame) < min_sessions:
        return {"eligible": False, "reason": "insufficient_market_history"}

    history = frame.copy()
    history["date"] = pd.to_datetime(history["date"]).dt.date
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history["volume"] = pd.to_numeric(history["volume"], errors="coerce")
    before = history[history["date"] <= reference_date].dropna(subset=["close"])
    if before.empty:
        return {"eligible": False, "reason": "no_price_at_first_mention"}

    reference = before.iloc[-1]
    reference_price = float(reference["close"])
    liquid_window = before.tail(20)
    dollar_volume = float((liquid_window["close"] * liquid_window["volume"]).median())
    if reference_price <= 0 or reference_price > max_price:
        return {
            "eligible": False,
            "reason": "reference_price_above_limit",
            "reference_date": reference["date"],
            "reference_price": reference_price,
            "reference_dollar_volume": dollar_volume,
        }
    if pd.isna(dollar_volume) or dollar_volume < min_dollar_volume:
        return {
            "eligible": False,
            "reason": "insufficient_dollar_volume",
            "reference_date": reference["date"],
            "reference_price": reference_price,
            "reference_dollar_volume": dollar_volume,
        }
    return {
        "eligible": True,
        "reason": "top_archive_mentions; price<=10; median_20d_dollar_volume>=250k",
        "reference_date": reference["date"],
        "reference_price": reference_price,
        "reference_dollar_volume": dollar_volume,
    }
