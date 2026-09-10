# Explosive Repricing Research Engine

A hosted, point-in-time research system for testing whether early social-attention patterns add predictive value beyond market data before extreme short-horizon price moves. It is a research and validation engine, not a ticker recommender or trading bot.

## Architecture

- Dashboard: Streamlit on Render
- Persistence: Render Postgres via `DATABASE_URL`
- Prototype market adapter: Yahoo Finance (`yfinance`)
- Prototype Reddit archive adapter: Arctic Shift
- Validation: expanding-window logistic-regression ablation tests

Raw sources, features, forward labels, pipeline audit records, and validation results stay in separate tables.

## Commands

Set `DATABASE_URL`, then run:

```bash
python -m engine.pipeline init-db
python -m engine.pipeline market --tickers GME,AMC,KOSS --start 2019-01-01 --end 2025-12-31
python -m engine.pipeline reddit --focused --subreddits pennystocks,Shortsqueeze --start 2021-01-01 --end 2021-12-31
python -m engine.pipeline universe --target-size 30 --candidate-limit 100
python -m engine.pipeline compute
python -m engine.pipeline validate --holdout-year 2026
streamlit run app.py
```

Writes are idempotent. Focused Reddit backfills use the stored market universe to query only
potentially matching posts, verify the ticker text locally, and support one-year dashboard
windows. Unfiltered archive runs remain capped at 31 days in the dashboard.

## Research contract

- Unit: ticker at daily close
- Target: +50% high within five future sessions
- Clean outcome: target precedes a −20% adverse threshold
- Same-bar target/adverse ambiguity: excluded from clean-win training
- Evaluation: expanding-window out-of-time folds
- Holdout: 2026 stays sealed until features and the decision rule are frozen
- Gate: combined mean out-of-time average precision must be at least 1.20× market-only

## Starter universe v1

The first broader cohort is frozen from the stored Reddit archive before validation. Candidates
are ranked by archived mention count, then must have a price of $10 or less at first mention,
at least $250,000 in median 20-session dollar volume, and at least 252 historical daily bars.
The rule, inclusion rank, reference price, and reference liquidity are stored in Postgres.

The Signals & Outcomes page joins each mention state to its available session close and reports
the subsequent 1-, 3-, and 5-session closing returns plus the best and worst five-session path.
Daily closes are research proxies, not executable intraday quotes.

## Limitations

The current adapters are for prototyping, not final institutional validation. The universe is user-supplied, delisted coverage is incomplete, historical social archives may be truncated, daily OHLC cannot order intraday threshold hits, and SEC/catalyst features remain a later stage.
