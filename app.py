from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import db
from engine.config import DEFAULT_SUBREDDITS, settings
from engine.features import FEATURE_VERSION, LABEL_VERSION
from engine.pipeline import (
    backfill_market,
    backfill_reddit,
    build_starter_universe,
    compute_research_tables,
    run_validation,
)
from engine.sources import fetch_market_history, fetch_reddit_history
from engine.text import normalize_ticker
from engine.universe import UNIVERSE_VERSION
from engine.validation import validation_decision


st.set_page_config(page_title="Explosive Repricing Research Engine", page_icon="⚡", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top:1.15rem;padding-bottom:3rem;max-width:1480px}
      .hero {padding:1.35rem 1.5rem;border:1px solid rgba(120,120,120,.23);border-radius:20px;
             margin-bottom:1rem;background:linear-gradient(135deg,rgba(255,174,0,.10),rgba(80,100,255,.06))}
      .hero h1 {margin:0;font-size:2rem;letter-spacing:-.025em}.hero p{margin:.45rem 0 0;opacity:.76;max-width:900px}
      .pill {display:inline-block;padding:.2rem .58rem;border-radius:999px;border:1px solid rgba(120,120,120,.30);
             margin:.75rem .3rem 0 0;font-size:.78rem}.eyebrow{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;opacity:.62}
      div[data-testid="stMetric"]{border:1px solid rgba(120,120,120,.18);padding:.72rem .85rem;border-radius:14px}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero"><div class="eyebrow">REDDIT PENNY STOCK · RESEARCH SYSTEM</div>
    <h1>Explosive Repricing Research Engine</h1>
    <p>Reconstruct what was knowable before major moves, test whether social attention adds out-of-time
    value beyond market data, and reject signals that do not survive validation.</p>
    <span class="pill">Historical reconstruction</span><span class="pill">Point-in-time features</span>
    <span class="pill">Walk-forward validation</span><span class="pill">No buy recommendations</span></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def database_state() -> dict:
    state = db.healthcheck()
    if state.get("connected"):
        try:
            db.init_schema()
            state["schema_ready"] = True
        except Exception as exc:
            state.update(schema_ready=False, detail=str(exc))
    return state


@st.cache_data(ttl=60, show_spinner=False)
def read(sql: str, params: tuple = ()) -> pd.DataFrame:
    return db.read_frame(sql, params)


def refresh_data() -> None:
    read.clear()


def db_required() -> bool:
    state = database_state()
    if state.get("connected") and state.get("schema_ready"):
        return True
    st.warning("Postgres is not ready. Attach the existing explosive-repricing-db internal connection string to this web service as DATABASE_URL, then redeploy.")
    if state.get("detail"):
        st.caption(f"Connection detail: {state['detail']}")
    return False


def fmt_int(value) -> str:
    return f"{int(value):,}" if value is not None and not pd.isna(value) else "—"


def candlestick(frame: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(go.Candlestick(x=frame["session_date"], open=frame["open"], high=frame["high"], low=frame["low"], close=frame["close"], name=ticker))
    fig.update_layout(height=430, margin=dict(l=8, r=8, t=32, b=8), xaxis_rangeslider_visible=False, title=f"{ticker} · persisted daily bars")
    return fig


tabs = st.tabs(["Control Center", "Historical Data", "Signals & Outcomes", "Validation Lab", "Live Discovery", "Methodology"])

with tabs[0]:
    st.subheader("Research control center")
    if db_required():
        overview = read("""SELECT (SELECT count(DISTINCT ticker) FROM market_bars) market_securities,(SELECT count(*) FROM market_bars) market_bars,(SELECT count(*) FROM social_posts) social_posts,(SELECT count(*) FROM social_mentions) social_mentions,(SELECT count(*) FROM outcome_labels WHERE label_version=%s) label_rows,(SELECT max(session_date) FROM market_bars) latest_market_date,(SELECT max(created_at) FROM social_posts) latest_social_time""", (LABEL_VERSION,)).iloc[0]
        cols = st.columns(5)
        for col, label, key in zip(cols, ["Market securities", "Market bars", "Reddit posts", "Ticker mentions", "Labeled states"], ["market_securities", "market_bars", "social_posts", "social_mentions", "label_rows"]):
            col.metric(label, fmt_int(overview[key]))
        st.caption(f"Market through {overview['latest_market_date'] or '—'} · Social through {overview['latest_social_time'] or '—'} · Feature {FEATURE_VERSION} · Label {LABEL_VERSION}")
        left, right = st.columns([1.4, 1])
        with left:
            st.markdown("#### Pipeline activity")
            runs = read("SELECT id,run_type,status,rows_written,started_at,finished_at,error_message FROM pipeline_runs ORDER BY started_at DESC LIMIT 12")
            if runs.empty:
                st.info("No pipeline run recorded yet.")
            else:
                st.dataframe(runs, width="stretch", hide_index=True)
        with right:
            st.markdown("#### Validation gate")
            latest = read("SELECT id,status,holdout_year,started_at,finished_at FROM validation_runs ORDER BY started_at DESC LIMIT 1")
            if latest.empty:
                st.info("Not tested. Build features and labels, then run validation.")
            else:
                metrics = read("SELECT * FROM validation_metrics WHERE validation_run_id=%s", (int(latest.iloc[0]["id"]),))
                decision = validation_decision(metrics)
                st.metric("Current decision", decision["decision"]); st.write(decision["reason"])
                if decision.get("ap_lift") is not None: st.metric("Combined / market AP", f"{decision['ap_lift']:.2f}×")
        if int(overview["market_bars"]) == 0: st.info("Next: backfill a defined market universe in Historical Data.")
        elif int(overview["social_mentions"]) == 0: st.info("Next: reconstruct Reddit history for the same period, then compute research tables.")
        elif int(overview["label_rows"]) == 0: st.info("Next: build point-in-time features and forward outcome labels.")
        else: st.success("The dataset is ready for validation. Keep 2026 sealed until feature choices are frozen.")

with tabs[1]:
    st.subheader("Historical data pipeline")
    st.caption("Idempotent backfills write to Postgres; rerunning a range updates rows instead of duplicating them.")
    if db_required():
        universe_tab, market_tab, reddit_tab, build_tab = st.tabs(
            ["Starter universe", "Market bars", "Reddit archive", "Build research tables"]
        )
        with universe_tab:
            st.write(
                "Freeze a reproducible starter cohort from the most-mentioned symbols in the stored "
                "Reddit archive. A symbol qualifies only when its price at first mention was $10 or "
                "less, its prior 20-session median dollar volume was at least $250,000, and at least "
                "252 daily bars are available. The first successful build is frozen as version v1."
            )
            memberships = read(
                """SELECT inclusion_rank,ticker,reference_date,reference_price,
                          reference_dollar_volume,social_mentions
                   FROM research_universe_memberships
                   WHERE universe_version=%s ORDER BY inclusion_rank""",
                (UNIVERSE_VERSION,),
            )
            if memberships.empty:
                if st.button("Build and freeze 30-stock starter universe", type="primary", width="stretch"):
                    with st.spinner("Screening archived mentions and backfilling qualifying market history..."):
                        result = build_starter_universe()
                    refresh_data()
                    st.success(
                        f"Frozen {result['members']} members after reviewing {result['reviewed']} candidates; "
                        f"wrote {result['market_rows']:,} market bars."
                    )
                    if result["rejections"]:
                        with st.expander(f"{len(result['rejections'])} rejected candidates"):
                            st.dataframe(pd.DataFrame(result["rejections"]), hide_index=True, width="stretch")
            else:
                st.success(f"{UNIVERSE_VERSION} is frozen with {len(memberships)} members.")
                st.dataframe(
                    memberships.style.format({
                        "reference_price": "${:.2f}",
                        "reference_dollar_volume": "${:,.0f}",
                    }),
                    hide_index=True,
                    width="stretch",
                )
        with market_tab:
            with st.form("market_backfill"):
                tickers_text = st.text_input("Tickers", value="GME, AMC, KOSS", help="Comma-separated; maximum 10 per dashboard run.")
                c1, c2 = st.columns(2); start = c1.date_input("Start date", date(2019, 1, 1), key="market_start"); end = c2.date_input("End date", date(2025, 12, 31), key="market_end")
                submitted = st.form_submit_button("Backfill market history", type="primary", width="stretch")
            if submitted:
                tickers = [ticker for value in tickers_text.split(",") if (ticker := normalize_ticker(value))][:10]
                if not tickers or start >= end: st.error("Provide valid tickers and an end date after the start date.")
                else:
                    with st.spinner("Fetching and persisting daily bars..."): result = backfill_market(tickers, start, end)
                    refresh_data(); st.success(f"Run {result['run_id']} wrote {result['rows_written']:,} rows.")
            coverage = read("SELECT ticker,min(session_date) first_date,max(session_date) last_date,count(*) sessions FROM market_bars GROUP BY ticker ORDER BY ticker")
            if not coverage.empty: st.dataframe(coverage, width="stretch", hide_index=True)
        with reddit_tab:
            st.warning("The archive may be incomplete. Any 2,000-post daily safety cap or source failure is recorded as a warning.")
            with st.form("reddit_backfill"):
                subs = st.text_input("Subreddits", value=", ".join(DEFAULT_SUBREDDITS)); c1, c2 = st.columns(2)
                rstart = c1.date_input("Start date", date.today() - timedelta(days=7), key="reddit_start"); rend = c2.date_input("End date", date.today(), key="reddit_end")
                focused = st.checkbox(
                    "Only posts matching the stored market universe",
                    value=True,
                    help="Uses the archive's keyword query, then verifies ticker mentions locally. This supports ranges up to one year and avoids downloading unrelated posts.",
                )
                reddit_submit = st.form_submit_button("Backfill Reddit history", type="primary", width="stretch")
            if reddit_submit:
                communities = [v.strip().replace("r/", "") for v in subs.split(",") if v.strip()]
                max_days = 366 if focused else 31
                if rstart > rend or (rend - rstart).days > max_days: st.error(f"Use a valid range of {max_days} days or less in the dashboard; use the CLI for larger runs.")
                else:
                    with st.spinner("Reconstructing timestamped Reddit posts..."): result = backfill_reddit(communities, rstart, rend, focused=focused)
                    refresh_data(); st.success(f"Run {result['run_id']} persisted {result['posts_written']:,} posts and {result['mentions_written']:,} mentions.")
                    if result["warnings"]:
                        with st.expander(f"{len(result['warnings'])} source warning(s)"):
                            for warning in result["warnings"][:100]: st.write(warning)
        with build_tab:
            st.write("Join persisted sources into close-of-session feature states and separate five-session outcome labels. Future Reddit data never enters a feature row.")
            if st.button("Compute features and outcome labels", type="primary", width="stretch"):
                with st.spinner("Building features and labels..."): result = compute_research_tables()
                refresh_data(); st.success(f"Run {result['run_id']} wrote {result['feature_rows']:,} features and {result['label_rows']:,} labels.")

with tabs[2]:
    st.subheader("Signals and outcomes")
    if db_required():
        st.markdown("#### Reddit mentions matched to actual market outcomes")
        st.caption(
            "Each row uses the market close available at that session—not an intraday execution price—"
            "then shows the subsequent closing returns and the best/worst five-session path."
        )
        comparison = read(
            """
            SELECT f.ticker,f.asof_date,f.close AS price_at_session_close,
                   f.mentions_1d,f.mentions_3d,f.rvol_20d,
                   l.forward_return_1d,l.forward_return_3d,l.forward_return_5d,
                   l.forward_mfe AS best_5d_excursion,l.forward_mae AS worst_5d_excursion,
                   l.outcome_class
            FROM daily_features f
            JOIN outcome_labels l USING (ticker,asof_date)
            WHERE f.feature_version=%s AND l.label_version=%s AND f.mentions_1d>0
            ORDER BY f.asof_date DESC,f.mentions_1d DESC
            LIMIT 1000
            """,
            (FEATURE_VERSION, LABEL_VERSION),
        )
        if comparison.empty:
            st.info("No price-matched mention states yet. Build the universe and research tables first.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mention states", f"{len(comparison):,}")
            c2.metric("Matched tickers", f"{comparison['ticker'].nunique():,}")
            c3.metric("Average 5-day close return", f"{comparison['forward_return_5d'].mean():.1%}")
            c4.metric("Clean +50% rate", f"{(comparison['outcome_class']=='clean_50').mean():.1%}")
            st.dataframe(
                comparison.style.format({
                    "price_at_session_close": "${:.3f}",
                    "rvol_20d": "{:.2f}×",
                    "forward_return_1d": "{:.1%}",
                    "forward_return_3d": "{:.1%}",
                    "forward_return_5d": "{:.1%}",
                    "best_5d_excursion": "{:.1%}",
                    "worst_5d_excursion": "{:.1%}",
                }, na_rep="—"),
                hide_index=True,
                width="stretch",
            )
        st.markdown("#### Per-security outcome history")
        tickers = read(
            "SELECT DISTINCT ticker FROM outcome_labels WHERE label_version=%s ORDER BY ticker",
            (LABEL_VERSION,),
        )["ticker"].tolist()
        if not tickers: st.info("Backfill at least one ticker first.")
        else:
            ticker = st.selectbox("Security", tickers); bars = read("SELECT * FROM market_bars WHERE ticker=%s ORDER BY session_date", (ticker,))
            if not bars.empty: st.plotly_chart(candlestick(bars, ticker), width="stretch")
            labels = read("SELECT asof_date,outcome_class,forward_mfe,forward_mae,target_hit_session,adverse_hit_session,ambiguous_same_session FROM outcome_labels WHERE ticker=%s AND label_version=%s ORDER BY asof_date DESC", (ticker, LABEL_VERSION))
            if labels.empty: st.info("No labels yet. Build research tables first.")
            else:
                summary = labels.groupby("outcome_class").size().rename("states").reset_index(); c1, c2 = st.columns([1, 2]); c1.dataframe(summary, width="stretch", hide_index=True)
                fig = px.bar(summary, x="outcome_class", y="states", color="outcome_class"); fig.update_layout(height=280, showlegend=False, margin=dict(l=8,r=8,t=10,b=8)); c2.plotly_chart(fig, width="stretch")
                episodes = labels[labels["outcome_class"].isin(["clean_50","wild_50","ambiguous_50"])]
                st.markdown("#### +50% outcome episodes")
                if episodes.empty: st.info("No +50% outcomes in the stored range.")
                else: st.dataframe(episodes.style.format({"forward_mfe":"{:.1%}","forward_mae":"{:.1%}"}), width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Walk-forward validation lab")
    st.caption("Every test year is scored by models trained only on earlier years. 2026 is excluded until the feature set and gate are frozen.")
    if db_required():
        c1, c2, c3 = st.columns([1, 1, 2]); holdout = c1.number_input("Untouched holdout", 2020, 2035, settings.untouched_holdout_year); c2.metric("Advance gate", "≥ 1.20× AP"); c3.write("Combined mean out-of-time average precision must improve at least 20% over market-only.")
        if st.button("Run walk-forward ablation test", type="primary", width="stretch"):
            with st.spinner("Training market-only, social-only, and combined folds..."): result = run_validation(int(holdout))
            refresh_data(); st.success(f"Run {result['validation_run_id']} completed: {result['decision']['decision']}")
        latest = read("SELECT * FROM validation_runs ORDER BY started_at DESC LIMIT 1")
        if latest.empty: st.info("No validation run yet.")
        else:
            metrics = read("SELECT * FROM validation_metrics WHERE validation_run_id=%s ORDER BY test_year,model_name", (int(latest.iloc[0]["id"]),)); decision = validation_decision(metrics)
            c1, c2, c3 = st.columns(3); c1.metric("Decision", decision["decision"]); c2.metric("Combined / market AP", f"{decision['ap_lift']:.2f}×" if decision.get("ap_lift") else "—"); c3.metric("Sealed holdout", str(int(latest.iloc[0]["holdout_year"]))); st.write(decision["reason"])
            if not metrics.empty:
                chart = px.line(metrics, x="test_year", y="average_precision", color="model_name", markers=True, title="Out-of-time average precision"); chart.update_layout(height=390, margin=dict(l=8,r=8,t=45,b=8)); st.plotly_chart(chart, width="stretch")
                st.dataframe(metrics.style.format({"roc_auc":"{:.3f}","average_precision":"{:.3f}","precision_at_k":"{:.3f}","lift_at_k":"{:.2f}×","brier_score":"{:.3f}"}), width="stretch", hide_index=True)

with tabs[4]:
    st.subheader("Live discovery sandbox")
    st.caption("A bounded current snapshot for hypothesis generation, deliberately separate from validated historical results.")
    c1, c2, c3 = st.columns([2,1,1]); subs_text = c1.text_input("Communities", value=", ".join(DEFAULT_SUBREDDITS), key="live_subs"); lookback = c2.selectbox("Lookback days", [1,3,7], index=2); top_n = c3.selectbox("Market check", [10,15,20], index=0)
    if st.button("Scan current attention", width="stretch"):
        end, posts, warnings = date.today(), [], []
        communities = [v.strip().replace("r/", "") for v in subs_text.split(",") if v.strip()]
        market_universe = set()
        if db_required():
            market_universe = set(read("SELECT DISTINCT ticker FROM market_bars")["ticker"].tolist())
        with st.spinner("Scanning communities and checking the latest market closes..."):
            with ThreadPoolExecutor(max_workers=min(8, len(communities) or 1)) as executor:
                futures = {
                    executor.submit(
                        fetch_reddit_history,
                        subreddit,
                        end-timedelta(days=int(lookback)),
                        end,
                        1,
                        market_universe,
                    ): subreddit
                    for subreddit in communities
                }
                for future in as_completed(futures):
                    try:
                        batch, source_warnings = future.result()
                        posts.extend(batch); warnings.extend(source_warnings)
                    except Exception as exc:
                        warnings.append(f"r/{futures[future]}: {exc}")
        mentions = [{"ticker":t,"post_id":p["post_id"],"author":p.get("author"),"community":p["community"]} for p in posts for t in p["tickers"]]
        if not mentions:
            st.info(
                "No usable ticker mentions were returned for this window. The archive can lag live "
                "Reddit; try seven days, or use Historical Data for a completed archive window."
            )
        else:
            attention = pd.DataFrame(mentions).groupby("ticker").agg(mentions=("post_id","nunique"),authors=("author","nunique"),communities=("community","nunique")).reset_index().sort_values(["mentions","authors"],ascending=False).head(int(top_n)); market = []
            def market_snapshot(ticker):
                history = fetch_market_history(ticker, end-timedelta(days=35), end)
                if len(history) < 2:
                    return {"ticker":ticker,"latest_market_close":np.nan,"return_5d_before_scan":np.nan,"relative_volume":np.nan}
                last = history.iloc[-1]; baseline = pd.to_numeric(history["volume"].iloc[-21:-1],errors="coerce").median()
                return {"ticker":ticker,"latest_market_close":float(last["close"]),"return_5d_before_scan":float(last["close"]/history.iloc[-6]["close"]-1) if len(history)>=6 else np.nan,"relative_volume":float(last["volume"]/baseline) if baseline and baseline>0 else np.nan}
            with ThreadPoolExecutor(max_workers=min(8, len(attention))) as executor:
                market = list(executor.map(market_snapshot, attention["ticker"].tolist()))
            snapshot = attention.merge(pd.DataFrame(market), on="ticker", how="left")
            snapshot["stage"] = np.select([snapshot["return_5d_before_scan"].fillna(0)>=.50,snapshot["return_5d_before_scan"].fillna(0)>=.20],["Late / extended","Acceleration"],default="Early / unconfirmed")
            st.caption("Market values are the latest available Yahoo daily close, not a real-time executable quote.")
            st.dataframe(snapshot.style.format({"latest_market_close":"${:.3f}","return_5d_before_scan":"{:.1%}","relative_volume":"{:.2f}×"}), width="stretch", hide_index=True)
        if warnings:
            with st.expander(f"{len(warnings)} source warning(s)"):
                for warning in warnings[:100]: st.write(warning)

with tabs[5]:
    st.subheader("Research contract")
    st.markdown("""The engine tests one claim: **does point-in-time social information improve identification of near-term clean +50% moves beyond market information alone?**

**Unit:** ticker at daily close.

**Target:** +50% high within five future sessions.

**Clean:** target occurs before −20% adverse excursion.

**Ambiguity:** both thresholds in one daily bar is excluded from clean-win training.

**Validation:** expanding-window out-of-time folds; 2026 remains sealed.

**Gate:** combined average precision must be at least 1.20× market-only and stable across years.""")
    st.markdown("#### Known limitations")
    st.write("Yahoo Finance and Arctic Shift are prototype adapters. The universe is user-supplied and does not yet remove survivorship bias; delisted coverage is incomplete; historical Reddit can be truncated; final post scores are not used as timestamped evidence; and SEC/catalyst features remain a later stage.")
    st.markdown("#### Build order")
    st.write("Persist sources → build leakage-safe states → label outcomes → compare ablations → freeze features → open holdout → consider intraday data only if the edge survives.")
    st.warning("Low-priced securities can be volatile, illiquid, diluted, halted, or manipulated. This is a research instrument, not a recommendation or automated trading system.")
