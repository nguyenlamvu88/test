from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Iterable

import pandas as pd

from .config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    rows_written INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS securities (
    ticker TEXT PRIMARY KEY,
    exchange TEXT,
    company_name TEXT,
    active BOOLEAN,
    first_seen DATE,
    last_seen DATE,
    source TEXT NOT NULL DEFAULT 'pipeline',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS market_bars (
    ticker TEXT NOT NULL REFERENCES securities(ticker),
    session_date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    adj_close NUMERIC,
    volume BIGINT,
    source TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, session_date)
);
CREATE INDEX IF NOT EXISTS market_bars_date_idx ON market_bars(session_date);

CREATE TABLE IF NOT EXISTS social_posts (
    source TEXT NOT NULL,
    post_id TEXT NOT NULL,
    community TEXT NOT NULL,
    author TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    title TEXT,
    body TEXT,
    url TEXT,
    score_observed INTEGER,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, post_id)
);
CREATE INDEX IF NOT EXISTS social_posts_created_idx ON social_posts(created_at);

CREATE TABLE IF NOT EXISTS social_mentions (
    source TEXT NOT NULL,
    post_id TEXT NOT NULL,
    ticker TEXT NOT NULL REFERENCES securities(ticker),
    created_at TIMESTAMPTZ NOT NULL,
    community TEXT NOT NULL,
    author TEXT,
    PRIMARY KEY (source, post_id, ticker),
    FOREIGN KEY (source, post_id) REFERENCES social_posts(source, post_id)
);
CREATE INDEX IF NOT EXISTS social_mentions_ticker_time_idx
    ON social_mentions(ticker, created_at);

CREATE TABLE IF NOT EXISTS daily_features (
    ticker TEXT NOT NULL REFERENCES securities(ticker),
    asof_date DATE NOT NULL,
    close NUMERIC,
    return_1d DOUBLE PRECISION,
    return_5d_lag DOUBLE PRECISION,
    rvol_20d DOUBLE PRECISION,
    dollar_volume DOUBLE PRECISION,
    realized_vol_20d DOUBLE PRECISION,
    mentions_1d INTEGER NOT NULL DEFAULT 0,
    mentions_3d INTEGER NOT NULL DEFAULT 0,
    unique_authors_3d INTEGER NOT NULL DEFAULT 0,
    communities_3d INTEGER NOT NULL DEFAULT 0,
    attention_accel DOUBLE PRECISION NOT NULL DEFAULT 0,
    feature_version TEXT NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, asof_date, feature_version)
);

CREATE TABLE IF NOT EXISTS outcome_labels (
    ticker TEXT NOT NULL REFERENCES securities(ticker),
    asof_date DATE NOT NULL,
    horizon_sessions INTEGER NOT NULL,
    target_return DOUBLE PRECISION NOT NULL,
    forward_mfe DOUBLE PRECISION,
    forward_mae DOUBLE PRECISION,
    target_hit_session INTEGER,
    adverse_hit_session INTEGER,
    ambiguous_same_session BOOLEAN NOT NULL DEFAULT false,
    outcome_class TEXT NOT NULL,
    label_version TEXT NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, asof_date, label_version)
);

CREATE TABLE IF NOT EXISTS validation_runs (
    id BIGSERIAL PRIMARY KEY,
    status TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    label_version TEXT NOT NULL,
    holdout_year INTEGER,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS validation_metrics (
    validation_run_id BIGINT NOT NULL REFERENCES validation_runs(id) ON DELETE CASCADE,
    test_year INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    train_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    positives INTEGER NOT NULL,
    roc_auc DOUBLE PRECISION,
    average_precision DOUBLE PRECISION,
    precision_at_k DOUBLE PRECISION,
    lift_at_k DOUBLE PRECISION,
    brier_score DOUBLE PRECISION,
    PRIMARY KEY (validation_run_id, test_year, model_name)
);
"""


def _database_url() -> str:
    url = settings.database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    if "sslmode=" not in url and ".render.com" in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url


@contextmanager
def connect():
    import psycopg

    with psycopg.connect(_database_url(), connect_timeout=12) as conn:
        yield conn


def init_schema() -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def healthcheck() -> dict[str, Any]:
    if not settings.database_url:
        return {"configured": False, "connected": False, "detail": "DATABASE_URL is not configured"}
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT current_database(), now()")
            database, server_time = cur.fetchone()
        return {"configured": True, "connected": True, "database": database, "server_time": server_time}
    except Exception as exc:
        return {"configured": True, "connected": False, "detail": str(exc)}


def start_run(run_type: str, parameters: dict[str, Any]) -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline_runs (run_type, status, parameters) VALUES (%s, 'running', %s::jsonb) RETURNING id",
            (run_type, json.dumps(parameters, default=str)),
        )
        return int(cur.fetchone()[0])


def finish_run(run_id: int, status: str, rows_written: int = 0, error: str | None = None) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE pipeline_runs SET status=%s, rows_written=%s, error_message=%s, finished_at=now() WHERE id=%s",
            (status, rows_written, error, run_id),
        )


def ensure_securities(tickers: Iterable[str], source: str = "pipeline") -> None:
    values = [(ticker, source) for ticker in sorted(set(tickers))]
    if not values:
        return
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO securities (ticker, source, first_seen, last_seen)
            VALUES (%s, %s, CURRENT_DATE, CURRENT_DATE)
            ON CONFLICT (ticker) DO UPDATE SET last_seen=CURRENT_DATE, updated_at=now()
            """,
            values,
        )


def upsert_market_bars(ticker: str, frame: pd.DataFrame, source: str = "yahoo") -> int:
    if frame.empty:
        return 0
    ensure_securities([ticker], source)
    rows = []
    for row in frame.itertuples(index=False):
        rows.append((
            ticker,
            pd.Timestamp(row.date).date(),
            _number(row.open), _number(row.high), _number(row.low), _number(row.close),
            _number(getattr(row, "adj_close", None)), _integer(row.volume), source,
        ))
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO market_bars
              (ticker, session_date, open, high, low, close, adj_close, volume, source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ticker, session_date) DO UPDATE SET
              open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
              close=EXCLUDED.close, adj_close=EXCLUDED.adj_close,
              volume=EXCLUDED.volume, source=EXCLUDED.source, ingested_at=now()
            """,
            rows,
        )
        first_date = min(row[1] for row in rows)
        last_date = max(row[1] for row in rows)
        cur.execute(
            """
            UPDATE securities
            SET first_seen=LEAST(COALESCE(first_seen, %s), %s),
                last_seen=GREATEST(COALESCE(last_seen, %s), %s),
                updated_at=now()
            WHERE ticker=%s
            """,
            (first_date, first_date, last_date, last_date, ticker),
        )
    return len(rows)


def upsert_social_posts(posts: list[dict[str, Any]]) -> tuple[int, int]:
    tickers = {ticker for post in posts for ticker in post.get("tickers", [])}
    ensure_securities(tickers, "reddit")
    post_rows, mention_rows = [], []
    for post in posts:
        created = post["created_at"]
        post_rows.append((
            "reddit", post["post_id"], post["community"], post.get("author"), created,
            post.get("title"), post.get("body"), post.get("url"), post.get("score_observed"),
        ))
        mention_rows.extend(
            ("reddit", post["post_id"], ticker, created, post["community"], post.get("author"))
            for ticker in post.get("tickers", [])
        )
    if not post_rows:
        return 0, 0
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO social_posts
              (source, post_id, community, author, created_at, title, body, url, score_observed)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (source, post_id) DO UPDATE SET
              community=EXCLUDED.community, author=EXCLUDED.author,
              title=EXCLUDED.title, body=EXCLUDED.body, url=EXCLUDED.url
            """,
            post_rows,
        )
        if mention_rows:
            cur.executemany(
                """
                INSERT INTO social_mentions
                  (source, post_id, ticker, created_at, community, author)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
                """,
                mention_rows,
            )
    return len(post_rows), len(mention_rows)


def read_frame(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [item.name for item in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=columns)


def execute_many(sql: str, rows: list[tuple[Any, ...]]) -> int:
    if not rows:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def _number(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _integer(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
