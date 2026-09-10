from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from threading import Lock
import time
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from .config import settings
from .text import extract_tickers


_ARCHIVE_LOCK = Lock()
_LAST_ARCHIVE_REQUEST = 0.0


def _archive_get(url: str, params: dict[str, Any]) -> requests.Response:
    """Respect the free archive's request-rate guidance and retry one 429."""
    global _LAST_ARCHIVE_REQUEST
    for attempt in range(2):
        with _ARCHIVE_LOCK:
            delay = 0.65 - (time.monotonic() - _LAST_ARCHIVE_REQUEST)
            if delay > 0:
                time.sleep(delay)
            response = requests.get(url, params=params, timeout=30)
            _LAST_ARCHIVE_REQUEST = time.monotonic()
        if response.status_code != 429 or attempt == 1:
            return response
        retry_after = response.headers.get("Retry-After") or response.headers.get("X-RateLimit-Reset") or "2"
        try:
            time.sleep(min(15.0, max(1.0, float(retry_after))))
        except ValueError:
            time.sleep(2.0)
    return response


def fetch_market_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily bars with an exclusive end date, normalized for storage."""
    frame = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
    expected = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for column in expected:
        if column not in frame:
            frame[column] = None
    return frame[expected].dropna(subset=["date", "close"]).sort_values("date")


def fetch_reddit_history(
    subreddit: str,
    start: date,
    end: date,
    chunk_days: int = 1,
    allowed_tickers: set[str] | None = None,
    query_tickers: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch historical posts in small UTC chunks to reduce 100-row truncation."""
    cursor = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    stop = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    posts: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    while cursor < stop:
        chunk_end = min(cursor + timedelta(days=chunk_days), stop)
        page_after = cursor
        page = 0
        while page_after < chunk_end and page < 20:
            params = {
                "subreddit": subreddit,
                # Arctic Shift documents epoch seconds and a limited subset of
                # ISO-8601. Python's ``+00:00`` form is rejected with HTTP 400,
                # so use the unambiguous epoch representation for both the
                # initial daily window and subsequent pagination cursors.
                "after": int(page_after.timestamp()),
                "before": int(chunk_end.timestamp()),
                "sort": "asc",
                "limit": 100,
                "fields": "id,author,created_utc,subreddit,title,selftext,url,score",
            }
            if query_tickers:
                params["query"] = " OR ".join(sorted(query_tickers))
            try:
                response = _archive_get(
                    f"{settings.arctic_base_url}/api/posts/search",
                    params,
                )
                response.raise_for_status()
                payload = response.json()
                batch = payload.get("data", payload)
                if not isinstance(batch, list):
                    batch = []
            except Exception as exc:
                warnings.append(f"r/{subreddit} {cursor.date()}: {exc}")
                break
            latest_created = page_after
            for raw in batch:
                post_id = str(raw.get("id") or "")
                if not post_id:
                    continue
                created_utc = raw.get("created_utc")
                if isinstance(created_utc, str):
                    try:
                        created = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                elif created_utc:
                    created = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
                else:
                    continue
                latest_created = max(latest_created, created)
                title, body = raw.get("title") or "", raw.get("selftext") or ""
                posts[post_id] = {
                    "post_id": post_id,
                    "community": raw.get("subreddit") or subreddit,
                    "author": raw.get("author"),
                    "created_at": created,
                    "title": title,
                    "body": body,
                    "url": raw.get("url"),
                    "score_observed": raw.get("score"),
                    "tickers": extract_tickers(title, body, allowed_tickers),
                }
            page += 1
            if len(batch) < 100 or latest_created <= page_after:
                break
            page_after = latest_created + timedelta(microseconds=1)
        if page == 20:
            warnings.append(
                f"r/{subreddit} hit the 2,000-post safety cap for {cursor.date()}; "
                "that day may be incomplete."
            )
        cursor = chunk_end
    return list(posts.values()), warnings
