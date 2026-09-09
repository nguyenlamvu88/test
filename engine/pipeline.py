from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd
import numpy as np

from . import db
from .config import DEFAULT_SUBREDDITS, settings
from .features import FEATURE_VERSION, LABEL_VERSION, build_daily_features, build_outcome_labels
from .sources import fetch_market_history, fetch_reddit_history
from .text import normalize_ticker
from .validation import validation_decision, walk_forward_validate


def backfill_market(tickers: list[str], start: date, end: date) -> dict:
    run_id = db.start_run("market_backfill", {"tickers": tickers, "start": start, "end": end})
    rows = 0
    try:
        for ticker in tickers:
            rows += db.upsert_market_bars(ticker, fetch_market_history(ticker, start, end))
        db.finish_run(run_id, "completed", rows)
        return {"run_id": run_id, "rows_written": rows}
    except Exception as exc:
        db.finish_run(run_id, "failed", rows, str(exc))
        raise


def backfill_reddit(subreddits: list[str], start: date, end: date) -> dict:
    run_id = db.start_run("reddit_backfill", {"subreddits": subreddits, "start": start, "end": end})
    rows, mentions, warnings = 0, 0, []
    try:
        for subreddit in subreddits:
            posts, source_warnings = fetch_reddit_history(subreddit, start, end)
            post_count, mention_count = db.upsert_social_posts(posts)
            rows += post_count
            mentions += mention_count
            warnings.extend(source_warnings)
        db.finish_run(run_id, "completed_with_warnings" if warnings else "completed", rows)
        return {"run_id": run_id, "posts_written": rows, "mentions_written": mentions, "warnings": warnings}
    except Exception as exc:
        db.finish_run(run_id, "failed", rows, str(exc))
        raise


def compute_research_tables(start: date | None = None, end: date | None = None) -> dict:
    parameters = {"start": start, "end": end, "feature_version": FEATURE_VERSION, "label_version": LABEL_VERSION}
    run_id = db.start_run("feature_label_build", parameters)
    try:
        where, params = [], []
        if start:
            where.append("session_date >= %s")
            params.append(start)
        if end:
            where.append("session_date <= %s")
            params.append(end)
        predicate = f"WHERE {' AND '.join(where)}" if where else ""
        bars = db.read_frame(f"SELECT * FROM market_bars {predicate} ORDER BY ticker, session_date", tuple(params))
        mentions = db.read_frame("SELECT ticker, created_at, community, author FROM social_mentions ORDER BY created_at")
        features = build_daily_features(bars, mentions)
        labels = build_outcome_labels(
            bars,
            horizon=settings.outcome_horizon_sessions,
            target=settings.target_return,
            adverse=settings.max_adverse_excursion,
        )
        feature_rows = [tuple(_sql_value(v) for v in row) for row in features.itertuples(index=False, name=None)]
        label_rows = [tuple(_sql_value(v) for v in row) for row in labels.itertuples(index=False, name=None)]
        db.execute_many(
            """
            INSERT INTO daily_features
              (ticker,asof_date,close,return_1d,return_5d_lag,rvol_20d,dollar_volume,
               realized_vol_20d,mentions_1d,mentions_3d,unique_authors_3d,communities_3d,
               attention_accel,feature_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ticker,asof_date,feature_version) DO UPDATE SET
              close=EXCLUDED.close,return_1d=EXCLUDED.return_1d,return_5d_lag=EXCLUDED.return_5d_lag,
              rvol_20d=EXCLUDED.rvol_20d,dollar_volume=EXCLUDED.dollar_volume,
              realized_vol_20d=EXCLUDED.realized_vol_20d,mentions_1d=EXCLUDED.mentions_1d,
              mentions_3d=EXCLUDED.mentions_3d,unique_authors_3d=EXCLUDED.unique_authors_3d,
              communities_3d=EXCLUDED.communities_3d,attention_accel=EXCLUDED.attention_accel,
              computed_at=now()
            """,
            feature_rows,
        )
        db.execute_many(
            """
            INSERT INTO outcome_labels
              (ticker,asof_date,horizon_sessions,target_return,forward_mfe,forward_mae,
               target_hit_session,adverse_hit_session,ambiguous_same_session,outcome_class,label_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ticker,asof_date,label_version) DO UPDATE SET
              forward_mfe=EXCLUDED.forward_mfe,forward_mae=EXCLUDED.forward_mae,
              target_hit_session=EXCLUDED.target_hit_session,adverse_hit_session=EXCLUDED.adverse_hit_session,
              ambiguous_same_session=EXCLUDED.ambiguous_same_session,outcome_class=EXCLUDED.outcome_class,
              computed_at=now()
            """,
            label_rows,
        )
        total = len(feature_rows) + len(label_rows)
        db.finish_run(run_id, "completed", total)
        return {"run_id": run_id, "feature_rows": len(feature_rows), "label_rows": len(label_rows)}
    except Exception as exc:
        db.finish_run(run_id, "failed", 0, str(exc))
        raise


def run_validation(holdout_year: int | None = None) -> dict:
    holdout = holdout_year or settings.untouched_holdout_year
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO validation_runs (status, feature_version, label_version, holdout_year, parameters)
            VALUES ('running', %s, %s, %s, %s::jsonb) RETURNING id
            """,
            (FEATURE_VERSION, LABEL_VERSION, holdout, json.dumps({"top_fraction": 0.05})),
        )
        validation_run_id = int(cur.fetchone()[0])
    try:
        dataset = db.read_frame(
            """
            SELECT f.*, l.outcome_class, l.forward_mfe, l.forward_mae
            FROM daily_features f JOIN outcome_labels l USING (ticker, asof_date)
            WHERE f.feature_version=%s AND l.label_version=%s
            ORDER BY f.asof_date, f.ticker
            """,
            (FEATURE_VERSION, LABEL_VERSION),
        )
        metrics = walk_forward_validate(dataset, holdout_year=holdout)
        metric_rows = [
            (validation_run_id,) + tuple(_sql_value(v) for v in row)
            for row in metrics[["test_year","model_name","train_rows","test_rows","positives","roc_auc","average_precision","precision_at_k","lift_at_k","brier_score"]].itertuples(index=False, name=None)
        ] if not metrics.empty else []
        db.execute_many(
            """
            INSERT INTO validation_metrics
              (validation_run_id,test_year,model_name,train_rows,test_rows,positives,
               roc_auc,average_precision,precision_at_k,lift_at_k,brier_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            metric_rows,
        )
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE validation_runs SET status='completed', finished_at=now() WHERE id=%s", (validation_run_id,))
        return {"validation_run_id": validation_run_id, "metrics": metrics, "decision": validation_decision(metrics)}
    except Exception as exc:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE validation_runs SET status='failed', error_message=%s, finished_at=now() WHERE id=%s", (str(exc), validation_run_id))
        raise


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _sql_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Explosive repricing historical pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db")
    market = commands.add_parser("market")
    market.add_argument("--tickers", required=True, help="Comma-separated symbols")
    market.add_argument("--start", required=True, type=_parse_date)
    market.add_argument("--end", required=True, type=_parse_date)
    reddit = commands.add_parser("reddit")
    reddit.add_argument("--subreddits", default=",".join(DEFAULT_SUBREDDITS))
    reddit.add_argument("--start", required=True, type=_parse_date)
    reddit.add_argument("--end", required=True, type=_parse_date)
    compute = commands.add_parser("compute")
    compute.add_argument("--start", type=_parse_date)
    compute.add_argument("--end", type=_parse_date)
    validate = commands.add_parser("validate")
    validate.add_argument("--holdout-year", type=int, default=settings.untouched_holdout_year)
    args = parser.parse_args()
    if args.command == "init-db":
        db.init_schema(); result = {"status": "schema_ready"}
    elif args.command == "market":
        tickers = [ticker for value in args.tickers.split(",") if (ticker := normalize_ticker(value))]
        result = backfill_market(tickers, args.start, args.end)
    elif args.command == "reddit":
        result = backfill_reddit([s.strip().replace("r/", "") for s in args.subreddits.split(",") if s.strip()], args.start, args.end)
    elif args.command == "compute":
        result = compute_research_tables(args.start, args.end)
    else:
        result = run_validation(args.holdout_year)
    print(json.dumps(result, default=lambda value: value.to_dict("records") if isinstance(value, pd.DataFrame) else str(value), indent=2))


if __name__ == "__main__":
    main()
