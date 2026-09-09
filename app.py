import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Explosive Repricing Research Engine", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px;}
      .hero {padding: 1.25rem 1.35rem; border: 1px solid rgba(120,120,120,.24); border-radius: 18px; margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2rem; letter-spacing: -.02em;}
      .hero p {margin: .45rem 0 0 0; opacity: .75;}
      .pill {display:inline-block; padding:.18rem .55rem; border-radius:999px; border:1px solid rgba(120,120,120,.28); margin-right:.35rem; font-size:.78rem;}
      div[data-testid="stMetric"] {border:1px solid rgba(120,120,120,.18); padding:.65rem .8rem; border-radius:14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Explosive Repricing Research Engine</h1>
      <p>Find and study early precursor states associated with tradeable +50% moves in overlooked securities.</p>
      <div style="margin-top:.75rem">
        <span class="pill">Research prototype</span>
        <span class="pill">Point-in-time discipline</span>
        <span class="pill">Social + market + catalyst</span>
        <span class="pill">Not a trading recommendation</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com"
DEFAULT_SUBREDDITS = [
    "pennystocks", "10xPennyStocks", "100xPennyStocks", "Shortsqueeze",
    "RobinHoodPennyStocks", "wallstreetbets", "biotech_stocks", "SPACs",
]
CASHTAG_RE = re.compile(r"(?<![A-Z0-9])\$([A-Z]{1,5})(?![A-Z0-9])")
STOP_TICKERS = {"A","I","DD","CEO","CFO","FDA","SEC","IPO","ETF","USA","USD","EPS","AI","EV","YOLO","IMO","ATH","EOD","FOMO","OTC","RS","PR","PM","AH"}


def utc_now():
    return datetime.now(timezone.utc)


def normalize_ticker(t):
    if not t:
        return None
    t = t.upper().strip().replace("$", "")
    if not re.fullmatch(r"[A-Z]{1,5}", t) or t in STOP_TICKERS:
        return None
    return t


@st.cache_data(ttl=600, show_spinner=False)
def fetch_reddit_posts(subreddit, after_iso, before_iso, limit=100):
    params = {
        "subreddit": subreddit, "after": after_iso, "before": before_iso,
        "sort": "asc", "limit": min(int(limit), 100),
        "fields": "id,author,created_utc,subreddit,title,selftext,url",
    }
    try:
        r = requests.get(f"{ARCTIC_BASE}/api/posts/search", params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", payload)
        return data if isinstance(data, list) else []
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def get_market_daily(ticker, period="1mo"):
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=False, actions=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        hist.columns = [str(c).lower().replace(" ", "_") for c in hist.columns]
        return hist
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_range(ticker, start, end):
    try:
        hist = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if hist is None or hist.empty:
            return pd.DataFrame()
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist = hist.reset_index()
        hist.columns = [str(c).lower().replace(" ", "_") for c in hist.columns]
        return hist
    except Exception:
        return pd.DataFrame()


def post_tickers(post):
    text = f"{post.get('title') or ''}\n{post.get('selftext') or ''}".upper()
    return sorted({t for t in CASHTAG_RE.findall(text) if normalize_ticker(t)})


def market_features(ticker):
    hist = get_market_daily(ticker, "1mo")
    if hist.empty or len(hist) < 3:
        return None
    last, prev = hist.iloc[-1], hist.iloc[-2]
    close = float(last.get("close", np.nan))
    ret1 = close / float(prev.get("close", close)) - 1 if float(prev.get("close", 0) or 0) > 0 else np.nan
    ret5 = close / float(hist.iloc[-6]["close"]) - 1 if len(hist) >= 6 and float(hist.iloc[-6].get("close", 0) or 0) > 0 else np.nan
    vols = pd.to_numeric(hist.get("volume"), errors="coerce")
    baseline = vols.iloc[-21:-1].median() if len(vols) >= 21 else vols.iloc[:-1].median()
    rvol = float(last.get("volume", np.nan)) / baseline if baseline and baseline > 0 else np.nan
    dollar_vol = close * float(last.get("volume", np.nan)) if pd.notna(close) else np.nan
    return {"price": close, "ret_1d": ret1, "ret_5d": ret5, "rvol": rvol, "dollar_volume": dollar_vol}


def robust_z(s):
    s = pd.Series(s, dtype="float64")
    med = s.median(); mad = (s - med).abs().median()
    if mad == 0 or pd.isna(mad):
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std and std > 0 else pd.Series(np.zeros(len(s)), index=s.index)
    return 0.6745 * (s - med) / mad


def build_attention_snapshot(subreddits, hours=24, max_posts_per_sub=100):
    before = utc_now(); after = before - timedelta(hours=hours)
    records, errors = [], []
    for sub in subreddits:
        data = fetch_reddit_posts(sub, after.isoformat(), before.isoformat(), max_posts_per_sub)
        if isinstance(data, dict) and data.get("error"):
            errors.append(f"r/{sub}: {data['error']}")
            continue
        for p in data:
            for t in post_tickers(p):
                records.append({"ticker":t,"subreddit":p.get("subreddit") or sub,"author":p.get("author"),"post_id":p.get("id"),"created_utc":p.get("created_utc"),"title":p.get("title") or ""})
    if not records:
        return pd.DataFrame(), errors
    mentions = pd.DataFrame(records)
    agg = mentions.groupby("ticker").agg(mentions=("post_id","nunique"), unique_authors=("author","nunique"), communities=("subreddit","nunique"), first_seen=("created_utc","min"), last_seen=("created_utc","max")).reset_index()
    return agg, errors


def add_market_and_score(attn, top_n=20):
    if attn.empty:
        return attn
    work = attn.sort_values(["mentions","unique_authors"], ascending=False).head(top_n).copy()
    feats = []
    for t in work["ticker"]:
        f = market_features(t)
        feats.append(f or {"price":np.nan,"ret_1d":np.nan,"ret_5d":np.nan,"rvol":np.nan,"dollar_volume":np.nan})
    work = pd.concat([work.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    z_mentions = robust_z(work["mentions"].fillna(0)); z_authors = robust_z(work["unique_authors"].fillna(0)); z_comms = robust_z(work["communities"].fillna(0)); z_rvol = robust_z(work["rvol"].replace([np.inf,-np.inf],np.nan).fillna(1))
    late_penalty = np.clip(work["ret_5d"].fillna(0), 0, None) * 2.0
    raw = .35*z_mentions + .30*z_authors + .15*z_comms + .20*z_rvol - late_penalty
    work["prototype_score"] = (50 + 12*raw).clip(0,100)
    work["crowd_stage"] = np.select([work["ret_5d"].fillna(0)>=.50, work["ret_5d"].fillna(0)>=.20], ["Late / already extended","Acceleration"], default="Early / unconfirmed")
    return work.sort_values("prototype_score", ascending=False)


def runner_episodes(df, horizon=5, target=.50):
    if df.empty or len(df)<=horizon:
        return pd.DataFrame()
    d=df.copy().reset_index(drop=True); rows=[]
    for i in range(len(d)-horizon):
        entry=float(d.loc[i,"close"])
        if not entry or entry<=0: continue
        future=d.iloc[i+1:i+1+horizon]
        mfe=float(future["high"].max()/entry-1); mae=float(future["low"].min()/entry-1)
        if mfe>=target:
            rows.append({"signal_date":pd.to_datetime(d.loc[i,"date"]).date(),"entry_close":entry,"mfe_5d":mfe,"mae_5d":mae,"clean_50":bool(mae>-0.20)})
    if not rows: return pd.DataFrame()
    out=pd.DataFrame(rows); keep=[]; last=None
    for _,r in out.iterrows():
        dt=pd.Timestamp(r["signal_date"])
        if last is None or (dt-last).days>horizon:
            keep.append(r); last=dt
    return pd.DataFrame(keep)


def price_chart(df,ticker):
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=df["date"],open=df["open"],high=df["high"],low=df["low"],close=df["close"],name=ticker))
    fig.update_layout(height=460,margin=dict(l=10,r=10,t=35,b=10),xaxis_rangeslider_visible=False,title=f"{ticker} historical price")
    return fig


tabs=st.tabs(["Radar","Historical Runner Lab","Reddit Intelligence","Research Model","System Status"])

with tabs[0]:
    st.subheader("Early Attention Radar")
    st.caption("A live research snapshot of social attention plus basic market confirmation. The score is a heuristic until the historical backtest is complete.")
    c1,c2,c3=st.columns([2,1,1])
    with c1: subs_text=st.text_input("Subreddits", value=", ".join(DEFAULT_SUBREDDITS))
    with c2: hours=st.selectbox("Lookback",[6,12,24,48],index=2)
    with c3: top_n=st.selectbox("Market-check top",[10,15,20,30],index=2)
    subreddits=[x.strip().replace("r/","") for x in subs_text.split(",") if x.strip()]
    if st.button("Refresh research snapshot",type="primary",use_container_width=True): st.cache_data.clear()
    with st.spinner("Reading recent social posts and matching market state..."):
        attn,errs=build_attention_snapshot(subreddits,hours=hours); radar=add_market_and_score(attn,top_n=top_n) if not attn.empty else pd.DataFrame()
    if errs:
        with st.expander("Source warnings"):
            for e in errs: st.warning(e)
    if radar.empty: st.info("No usable cashtag mentions were returned from the selected sources. Try a longer lookback or fewer communities.")
    else:
        m1,m2,m3,m4=st.columns(4); m1.metric("Tickers detected",f"{len(attn):,}"); m2.metric("Posts with cashtags",f"{int(attn['mentions'].sum()):,}"); m3.metric("Most-discussed",radar.iloc[0]["ticker"]); m4.metric("Highest prototype score",f"{radar.iloc[0]['prototype_score']:.0f}/100")
        view=radar[["ticker","prototype_score","mentions","unique_authors","communities","price","ret_1d","ret_5d","rvol","dollar_volume","crowd_stage"]].copy()
        view.columns=["Ticker","Prototype score","Mentions","Unique authors","Communities","Price","1D return","5D return","RVOL","Dollar volume","Crowd stage"]
        st.dataframe(view.style.format({"Prototype score":"{:.0f}","Price":"${:.3f}","1D return":"{:.1%}","5D return":"{:.1%}","RVOL":"{:.2f}x","Dollar volume":"${:,.0f}"}),use_container_width=True,hide_index=True)
        st.caption("Prototype score = attention breadth + social activity + relative volume − late-stage price-extension penalty. It is not yet an empirically validated probability.")

with tabs[1]:
    st.subheader("Historical +50% Runner Lab"); st.caption("Start from the outcome and work backward. This is the runner-first half of the experiment.")
    a,b,c=st.columns(3)
    with a: ticker=st.text_input("Ticker",value="GME",key="runner_ticker").upper().strip()
    with b: start=st.date_input("Start",value=datetime(2020,1,1).date())
    with c: end=st.date_input("End",value=datetime(2022,1,1).date())
    if ticker and start<end:
        df=get_market_range(ticker,str(start),str(end+timedelta(days=1)))
        if df.empty: st.warning("No market history returned for this symbol/range.")
        else:
            st.plotly_chart(price_chart(df,ticker),use_container_width=True); episodes=runner_episodes(df)
            m1,m2,m3=st.columns(3); m1.metric("+50% episodes",len(episodes)); m2.metric("Best 5-day MFE",f"{episodes['mfe_5d'].max():.1%}" if not episodes.empty else "—"); m3.metric("Clean +50% episodes",int(episodes["clean_50"].sum()) if not episodes.empty else 0)
            if not episodes.empty:
                show=episodes.copy(); show.columns=["Signal date","Entry close","5D MFE","5D MAE","Clean +50"]
                st.dataframe(show.style.format({"Entry close":"${:.3f}","5D MFE":"{:.1%}","5D MAE":"{:.1%}"}),use_container_width=True,hide_index=True)
                st.info("Next research step: automatically reconstruct Reddit, filings, sector behavior, volume state, and poster history before each episode.")

with tabs[2]:
    st.subheader("Reddit Intelligence"); st.caption("Inspect information flow rather than trusting raw sentiment.")
    r1,r2,r3=st.columns(3)
    with r1: rsub=st.text_input("Subreddit",value="pennystocks",key="reddit_sub").replace("r/","")
    with r2: days=st.selectbox("Lookback days",[1,3,7,14,30],index=2)
    with r3: filter_ticker=st.text_input("Optional ticker filter",value="",key="reddit_filter").upper().replace("$","").strip()
    before=utc_now(); after=before-timedelta(days=int(days)); data=fetch_reddit_posts(rsub,after.isoformat(),before.isoformat(),100)
    if isinstance(data,dict) and data.get("error"): st.error(f"Reddit archive source error: {data['error']}")
    elif not data: st.info("No posts returned.")
    else:
        rows=[]
        for p in data:
            ts=datetime.fromtimestamp(p.get("created_utc",0),tz=timezone.utc) if p.get("created_utc") else None; ticks=post_tickers(p)
            if filter_ticker and filter_ticker not in ticks: continue
            rows.append({"Time UTC":ts,"Author":p.get("author"),"Tickers":", ".join(ticks),"Title":p.get("title") or "","URL":p.get("url") or ""})
        rdf=pd.DataFrame(rows)
        if rdf.empty: st.info("No posts matched the ticker filter.")
        else:
            st.metric("Posts inspected",len(rdf)); st.dataframe(rdf,use_container_width=True,hide_index=True,column_config={"URL":st.column_config.LinkColumn("URL")})
            st.caption("Historical model will use rolling poster history and timestamped comment arrival. It will not treat later final upvote counts as information known at post time.")

with tabs[3]:
    st.subheader("Research Model"); st.caption("What the eventual model must prove before we treat a signal as useful.")
    st.markdown("### Competing models")
    st.dataframe(pd.DataFrame([["A — Market only","Price, volume, volatility, liquidity, float/supply","Baseline"],["B — Social only","Attention velocity, authors, communities, poster fidelity, diffusion","Unvalidated"],["C — Catalyst only","SEC/company/clinical/contract novelty and verification","Unvalidated"],["D — Combined","Market + social + catalyst + network/sequence features","Target"]],columns=["Model","Inputs","Role"]),use_container_width=True,hide_index=True)
    st.markdown("### Outcome taxonomy")
    st.dataframe(pd.DataFrame([["Clean 50","+50% within 5 sessions before >20% adverse excursion"],["Tradable 50","+50% within 5 sessions with tolerable liquidity/drawdown"],["Wild 50","+50% achieved only after severe adverse excursion"],["False ignition","Attention/volume appears but move fails"],["Late crowd","Signal appears after price already repriced materially"]],columns=["Outcome","Definition"]),use_container_width=True,hide_index=True)
    st.markdown("### Go / no-go test"); st.write("The combined model must show out-of-time lift over the market-only model, acceptable MAE/liquidity, and stability across market regimes. 2026 remains an untouched holdout until feature choices are frozen.")

with tabs[4]:
    st.subheader("System Status")
    st.dataframe(pd.DataFrame([["Hosted dashboard","Online","This application"],["Recent Reddit archive","Connected on demand","Arctic Shift community API"],["Basic market history","Connected on demand","Yahoo Finance prototype adapter"],["SEC EDGAR","Planned next","Primary-source filing/catalyst engine"],["Point-in-time security master","Needed","Prevents survivorship bias"],["Delisted universe","Needed","Required before serious backtest"],["Poster fidelity model","Planned","Rolling, no future leakage"],["Intraday bars","Later","Only after daily proof of signal"],["Validated probability model","Not yet","Do not interpret prototype score as probability"]],columns=["Component","Status","Notes"]),use_container_width=True,hide_index=True)
    st.markdown("### Build sequence"); st.write("1. Hosted research dashboard → 2. historical market universe → 3. runner-first labels → 4. Reddit/SEC reconstruction → 5. market/social/catalyst ablation tests → 6. untouched holdout → 7. intraday/live scanner only if the edge survives.")
    st.warning("Low-priced securities can be extremely volatile, illiquid, diluted, halted, or manipulated. This site is a research instrument; it does not establish that a displayed ticker is suitable to buy.")
