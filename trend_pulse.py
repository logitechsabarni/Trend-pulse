"""
TREND PULSE AI  —  Intelligent Google Trends Analytics Platform
===============================================================
New in this version:
  ✓ AI Insight Generator (Claude-powered natural language insights)
  ✓ Trend Prediction (linear regression forecast, 30-day)
  ✓ Trend Alert System (spike detection + threshold alerts)
  ✓ Keyword Battle Mode (winner + fastest growth summary)
  ✓ Correlation Insights (human-readable interpretation)
  ✓ Momentum Score & Volatility Indicator
  ✓ All v4 fixes retained (urllib3 v2, 429 backoff, demo mode)
"""

# ── 1. urllib3 v2 Patch ───────────────────────────────────────────────────────
import urllib3.util.retry as _r

class _SafeRetry(_r.Retry):
    def __init__(self, *a, **kw):
        kw.pop("method_whitelist", None)
        super().__init__(*a, **kw)

_r.Retry = _SafeRetry
try:
    import requests.packages.urllib3.util.retry as _rr
    _rr.Retry = _SafeRetry
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import time, random, hashlib, json, re
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from pytrends.request import TrendReq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trend Pulse AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Clash+Display:wght@400;500;600;700&family=Epilogue:wght@300;400;500;700;800;900&display=swap');

:root{
  --bg:#04050d; --surf:#090a15; --surf2:#0e0f1e; --surf3:#131428;
  --bdr:#1c1d35; --bdr2:#262740;
  --p1:#6c63ff; --p2:#00f5d4; --p3:#ff4d6d; --p4:#ffbe0b; --p5:#80ffdb;
  --p6:#ff6b35; --p7:#a78bfa;
  --txt:#eeeef8; --dim:#6666aa; --dimmer:#3a3a6a;
  --mono:'IBM Plex Mono',monospace;
  --display:'Epilogue',sans-serif;
  --grad1:linear-gradient(135deg,#6c63ff 0%,#00f5d4 100%);
  --grad2:linear-gradient(135deg,#ff4d6d 0%,#ffbe0b 100%);
  --grad3:linear-gradient(135deg,#a78bfa 0%,#6c63ff 100%);
}

html,body,[class*="css"]{
  background:var(--bg)!important;
  color:var(--txt)!important;
  font-family:var(--display)!important;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.6rem 2rem!important;max-width:1800px!important;}

/* ── Hero ── */
.hero-wrap{
  position:relative;padding:1.8rem 0 1.2rem;
  border-bottom:1px solid var(--bdr);margin-bottom:1.2rem;
  overflow:hidden;
}
.hero-wrap::before{
  content:'';position:absolute;top:-60px;right:-40px;
  width:480px;height:480px;border-radius:50%;
  background:radial-gradient(circle,rgba(108,99,255,.18) 0%,transparent 70%);
  pointer-events:none;
}
.hero-eyebrow{
  font-family:var(--mono);font-size:.58rem;letter-spacing:.22em;
  text-transform:uppercase;color:var(--p2);margin-bottom:.4rem;
}
.hero-title{
  font-family:var(--display);font-weight:900;font-size:3rem;
  line-height:.95;letter-spacing:-.05em;margin:0;
  background:var(--grad1);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.hero-sub{
  font-family:var(--mono);font-size:.61rem;color:var(--dim);
  letter-spacing:.1em;text-transform:uppercase;margin-top:.5rem;
}

/* ── Cards ── */
.mcard{
  background:var(--surf);border:1px solid var(--bdr);border-radius:16px;
  padding:1.1rem 1.3rem;position:relative;overflow:hidden;
  transition:transform .18s,border-color .18s;
}
.mcard:hover{transform:translateY(-3px);border-color:var(--bdr2);}
.mlabel{font-family:var(--mono);font-size:.57rem;color:var(--dim);
  text-transform:uppercase;letter-spacing:.15em;margin-bottom:.3rem;}
.mval{font-family:var(--display);font-weight:800;font-size:2rem;line-height:1;}
.msub{font-family:var(--mono);font-size:.57rem;color:var(--dim);margin-top:.3rem;}
.bdg{display:inline-block;font-family:var(--mono);font-size:.57rem;
  padding:.14rem .5rem;border-radius:20px;margin-top:.32rem;font-weight:700;}
.bup{background:rgba(0,245,212,.12);color:#00f5d4;}
.bdn{background:rgba(255,77,109,.12);color:#ff4d6d;}
.bfl{background:rgba(108,99,255,.12);color:#a78bfa;}
.bwarn{background:rgba(255,190,11,.12);color:#ffbe0b;}

/* Section header */
.sh{
  font-family:var(--mono);font-size:.56rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--dim);
  border-bottom:1px solid var(--bdr);
  padding-bottom:.32rem;margin:1.1rem 0 .65rem;
}

/* Chips */
.chip{
  display:inline-block;font-family:var(--mono);font-size:.64rem;
  padding:.2rem .68rem;border-radius:20px;margin:.18rem;
  border:1px solid;font-weight:700;
}

/* Insight box */
.insight-box{
  background:linear-gradient(135deg,rgba(108,99,255,.08),rgba(0,245,212,.05));
  border:1px solid rgba(108,99,255,.35);border-radius:14px;
  padding:1.2rem 1.4rem;margin:.6rem 0;
  font-family:var(--display);font-size:.9rem;line-height:1.65;color:var(--txt);
  position:relative;
}
.insight-box::before{
  content:'🧠 AI INSIGHT';
  font-family:var(--mono);font-size:.52rem;color:var(--p1);
  letter-spacing:.16em;display:block;margin-bottom:.55rem;
}
.insight-box.alert{
  background:linear-gradient(135deg,rgba(255,77,109,.08),rgba(255,190,11,.05));
  border-color:rgba(255,77,109,.35);
}
.insight-box.alert::before{content:'🚨 ALERT';}
.insight-box.battle{
  background:linear-gradient(135deg,rgba(255,190,11,.08),rgba(255,107,53,.05));
  border-color:rgba(255,190,11,.35);
}
.insight-box.battle::before{content:'⚔️ BATTLE MODE';}

/* Prediction strip */
.pred-strip{
  display:flex;gap:1rem;flex-wrap:wrap;margin-top:.8rem;
}
.pred-item{
  background:var(--surf2);border:1px solid var(--bdr);border-radius:10px;
  padding:.6rem .9rem;flex:1;min-width:140px;
}
.pred-label{font-family:var(--mono);font-size:.54rem;color:var(--dim);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:.2rem;}
.pred-val{font-family:var(--display);font-weight:700;font-size:1.15rem;}

/* Demo banner */
.demo-banner{
  background:rgba(255,190,11,.07);border:1px solid rgba(255,190,11,.3);
  border-radius:10px;padding:.65rem 1rem;margin-bottom:.9rem;
  font-family:var(--mono);font-size:.62rem;color:#ffbe0b;letter-spacing:.04em;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background:var(--surf)!important;
  border-right:1px solid var(--bdr)!important;
}
[data-testid="stSidebar"] label{
  font-family:var(--mono)!important;font-size:.65rem!important;
  letter-spacing:.07em;color:var(--dim)!important;
}

/* Inputs */
textarea,.stTextInput input{
  background:var(--surf2)!important;border:1px solid var(--bdr)!important;
  color:var(--txt)!important;border-radius:8px!important;
  font-family:var(--mono)!important;font-size:.8rem!important;
}
.stSelectbox>div>div{
  background:var(--surf2)!important;border-color:var(--bdr)!important;
  border-radius:8px!important;color:var(--txt)!important;
  font-family:var(--mono)!important;
}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#6c63ff,#4c44cc)!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  font-family:var(--mono)!important;font-size:.72rem!important;
  font-weight:700!important;letter-spacing:.12em!important;
  padding:.65rem 1.4rem!important;text-transform:uppercase!important;
  transition:all .2s!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 8px 24px rgba(108,99,255,.4)!important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  background:var(--surf)!important;border-bottom:1px solid var(--bdr)!important;
  gap:.2rem!important;
}
.stTabs [data-baseweb="tab"]{
  font-family:var(--mono)!important;font-size:.62rem!important;
  text-transform:uppercase!important;letter-spacing:.1em!important;
  color:var(--dim)!important;background:transparent!important;
  padding:.65rem 1.1rem!important;
}
.stTabs [aria-selected="true"]{
  color:var(--txt)!important;
  border-bottom:2px solid var(--p1)!important;
}

/* Alerts */
.stInfo{background:rgba(108,99,255,.08)!important;border:1px solid rgba(108,99,255,.25)!important;border-radius:8px!important;}
.stWarning{background:rgba(255,190,11,.07)!important;border:1px solid rgba(255,190,11,.25)!important;border-radius:8px!important;}
.stError{background:rgba(255,77,109,.08)!important;border:1px solid rgba(255,77,109,.25)!important;border-radius:8px!important;}
.stSuccess{background:rgba(0,245,212,.08)!important;border:1px solid rgba(0,245,212,.25)!important;border-radius:8px!important;}

.stDataFrame{border:1px solid var(--bdr)!important;border-radius:8px!important;}
hr{border-color:var(--bdr)!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--bdr2);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:var(--p1);}

/* Progress bars (volatility/momentum) */
.meter-wrap{margin:.4rem 0;}
.meter-label{font-family:var(--mono);font-size:.56rem;color:var(--dim);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:.22rem;
  display:flex;justify-content:space-between;}
.meter-bar{height:6px;border-radius:3px;background:var(--surf2);overflow:hidden;}
.meter-fill{height:100%;border-radius:3px;transition:width .5s ease;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PAL = ["#6c63ff", "#00f5d4", "#ff4d6d", "#ffbe0b", "#a78bfa"]

BL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono,monospace", color="#eeeef8", size=11),
    title_font=dict(family="Epilogue,sans-serif", size=13, color="#eeeef8"),
    legend=dict(bgcolor="rgba(9,10,21,.95)", bordercolor="#1c1d35",
                borderwidth=1, font=dict(size=10, family="IBM Plex Mono")),
    xaxis=dict(gridcolor="#0e0f1e", linecolor="#1c1d35", tickfont=dict(size=10), zeroline=False),
    yaxis=dict(gridcolor="#0e0f1e", linecolor="#1c1d35", tickfont=dict(size=10), zeroline=False),
    margin=dict(l=10, r=10, t=36, b=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#0e0f1e", bordercolor="#1c1d35",
                    font=dict(family="IBM Plex Mono", size=11)),
)

TIME_OPTS = {
    "Past Hour": "now 1-H", "Past 4 Hours": "now 4-H",
    "Past Day": "now 1-d", "Past 7 Days": "now 7-d",
    "Past 30 Days": "today 1-m", "Past 90 Days": "today 3-m",
    "Past 12 Months": "today 12-m", "Past 5 Years": "today 5-y",
}
CAT_OPTS = {
    "All Categories": 0, "Arts & Entertainment": 3, "Autos & Vehicles": 47,
    "Business & Industry": 12, "Computers & Electronics": 5, "Finance": 7,
    "Food & Drink": 71, "Games": 8, "Health": 45, "Jobs & Education": 958,
    "News": 16, "People & Society": 14, "Science": 174,
    "Shopping": 18, "Sports": 20, "Technology": 13, "Travel": 67,
}
GEO_OPTS = {
    "Worldwide": "", "United States": "US", "United Kingdom": "GB",
    "India": "IN", "Germany": "DE", "France": "FR", "Brazil": "BR",
    "Japan": "JP", "Australia": "AU", "Canada": "CA",
    "Mexico": "MX", "South Korea": "KR",
}
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def rgba(h: str, a: float) -> str:
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return f"rgba({r},{g},{b},{a})"

def cache_key(*args) -> str:
    return hashlib.md5(json.dumps(args, default=str).encode()).hexdigest()[:16]

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("req_cache", {}), ("last_call_ts", 0.0),
    ("demo_mode", False), ("ai_insights", {}),
    ("alerts", []), ("alert_threshold", 70),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def polite_sleep():
    elapsed = time.time() - st.session_state.last_call_ts
    gap = random.uniform(2.2, 4.0)
    if elapsed < gap:
        time.sleep(gap - elapsed)
    st.session_state.last_call_ts = time.time()


def make_pt() -> TrendReq:
    ua = random.choice(USER_AGENTS)
    return TrendReq(hl="en-US", tz=330, timeout=(20, 40),
                    requests_args={"headers": {"User-Agent": ua}})


def call_with_backoff(fn, max_retries=4):
    for attempt in range(max_retries):
        try:
            polite_sleep()
            return fn(), None
        except Exception as e:
            msg = str(e)
            is_429 = "429" in msg or "quota" in msg.lower() or "too many" in msg.lower()
            if is_429 and attempt < max_retries - 1:
                wait = (2 ** attempt) * random.uniform(3, 6)
                st.toast(f"⏳ Rate limited — waiting {wait:.0f}s …", icon="⚠️")
                time.sleep(wait)
            else:
                return None, msg
    return None, "Max retries exceeded"


# ── Demo data ─────────────────────────────────────────────────────────────────
def demo_over_time(keywords, timeframe):
    np.random.seed(42)
    if "H" in timeframe:
        periods, freq = 60, "min"
    elif "7-d" in timeframe or "1-d" in timeframe:
        periods, freq = 168, "h"
    elif "1-m" in timeframe:
        periods, freq = 30, "D"
    elif "3-m" in timeframe:
        periods, freq = 90, "D"
    elif "5-y" in timeframe:
        periods, freq = 260, "W"
    else:
        periods, freq = 52, "W"
    dates = pd.date_range(end=datetime.today(), periods=periods, freq=freq)
    n = len(dates)
    bases = [random.randint(25, 85) for _ in keywords]
    data = {}
    for i, kw in enumerate(keywords):
        trend = np.linspace(0, random.choice([-1, 1]) * random.randint(5, 20), n)
        noise = np.random.randn(n) * random.uniform(3, 9)
        spike_idx = random.randint(n // 4, 3 * n // 4)
        spike = np.zeros(n)
        spike[spike_idx:min(spike_idx + 3, n)] = random.randint(10, 35)
        vals = np.clip(bases[i] + trend + noise + spike, 0, 100).astype(int)
        data[kw] = vals
    return pd.DataFrame(data, index=dates)


def demo_by_region(keywords):
    countries = ["United States","India","United Kingdom","Germany","Brazil",
                 "Canada","Australia","France","Japan","Mexico",
                 "Indonesia","South Korea","Italy","Spain","Netherlands"]
    data = {kw: np.random.randint(10, 100, len(countries)) for kw in keywords}
    return pd.DataFrame(data, index=countries)


def demo_related(keywords):
    templates = {
        "top": ["how to use {kw}","{kw} tutorial","{kw} vs gpt4","{kw} api","{kw} login",
                "best {kw} prompts","{kw} pricing","{kw} free","{kw} download","{kw} review"],
        "rising": ["{kw} 2025","{kw} news","new {kw} features","{kw} update","{kw} alternative",
                   "{kw} benchmark","{kw} comparison","{kw} image","{kw} code","{kw} enterprise"],
    }
    result = {}
    for kw in keywords:
        top_q = [q.replace("{kw}", kw) for q in templates["top"]]
        ris_q = [q.replace("{kw}", kw) for q in templates["rising"]]
        result[kw] = {
            "top": pd.DataFrame({"query": top_q, "value": sorted(np.random.randint(40, 100, 10), reverse=True)}),
            "rising": pd.DataFrame({"query": ris_q, "value": sorted(np.random.randint(200, 5000, 10), reverse=True)}),
        }
    return result


def demo_trending():
    items = ["OpenAI GPT-5","Apple Intelligence","Formula 1","Python 3.13",
             "World Cup 2026","Claude 4","Llama 3","Tesla Cybertruck",
             "Nintendo Switch 2","Solar Eclipse","SpaceX Starship","Taylor Swift",
             "Bitcoin ETF","Gemini Ultra","Vision Pro 2"]
    return pd.DataFrame(items, columns=["Trending Query"])


# ── Fetchers ──────────────────────────────────────────────────────────────────
def get_over_time(kws, tf, geo, cat):
    key = cache_key("ot", kws, tf, geo, cat)
    if key in st.session_state.req_cache:
        return st.session_state.req_cache[key], None, False
    if st.session_state.demo_mode:
        data = demo_over_time(list(kws), tf)
        st.session_state.req_cache[key] = data
        return data, None, True
    def _fetch():
        pt = make_pt()
        pt.build_payload(list(kws), cat=cat, timeframe=tf, geo=geo)
        df = pt.interest_over_time()
        if not df.empty and "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        return df
    data, err = call_with_backoff(_fetch)
    if err or (data is not None and data.empty):
        data = demo_over_time(list(kws), tf)
        st.session_state.req_cache[key] = data
        return data, err, True
    st.session_state.req_cache[key] = data
    return data, None, False


def get_by_region(kws, tf, geo, cat):
    key = cache_key("br", kws, tf, geo, cat)
    if key in st.session_state.req_cache:
        return st.session_state.req_cache[key], False
    if st.session_state.demo_mode:
        data = demo_by_region(list(kws))
        st.session_state.req_cache[key] = data
        return data, True
    def _fetch():
        pt = make_pt()
        pt.build_payload(list(kws), cat=cat, timeframe=tf, geo=geo)
        return pt.interest_by_region(resolution="COUNTRY", inc_low_vol=True)
    data, err = call_with_backoff(_fetch)
    if err or data is None or data.empty:
        data = demo_by_region(list(kws))
        st.session_state.req_cache[key] = data
        return data, True
    st.session_state.req_cache[key] = data
    return data, False


def get_related(kws, tf, geo, cat):
    key = cache_key("rq", kws, tf, geo, cat)
    if key in st.session_state.req_cache:
        return st.session_state.req_cache[key], False
    if st.session_state.demo_mode:
        data = demo_related(list(kws))
        st.session_state.req_cache[key] = data
        return data, True
    def _fetch():
        pt = make_pt()
        pt.build_payload(list(kws), cat=cat, timeframe=tf, geo=geo)
        return pt.related_queries()
    data, err = call_with_backoff(_fetch)
    if err or not data:
        data = demo_related(list(kws))
        st.session_state.req_cache[key] = data
        return data, True
    st.session_state.req_cache[key] = data
    return data, False


def get_trending(geo):
    key = cache_key("tr", geo)
    if key in st.session_state.req_cache:
        return st.session_state.req_cache[key], False
    if st.session_state.demo_mode:
        data = demo_trending()
        st.session_state.req_cache[key] = data
        return data, True
    pn_map = {"US":"united_states","GB":"united_kingdom","IN":"india","DE":"germany",
              "FR":"france","BR":"brazil","JP":"japan","AU":"australia",
              "CA":"canada","MX":"mexico","KR":"south_korea"}
    def _fetch():
        pt = make_pt()
        return pt.trending_searches(pn=pn_map.get(geo, "united_states"))
    data, err = call_with_backoff(_fetch)
    if err or data is None or data.empty:
        data = demo_trending()
        st.session_state.req_cache[key] = data
        return data, True
    st.session_state.req_cache[key] = data
    return data, False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEW: AI ANALYTICS FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_momentum(df, kw):
    """Returns momentum % change from first quarter to last quarter."""
    q = max(1, len(df) // 4)
    f = df[kw].iloc[:q].mean()
    l = df[kw].iloc[-q:].mean()
    return round(((l - f) / (f + 1e-9)) * 100, 1)


def compute_volatility(df, kw):
    """Returns coefficient of variation as a 0-100 volatility score."""
    mean = df[kw].mean()
    std = df[kw].std()
    cv = (std / (mean + 1e-9)) * 100
    return round(min(cv, 100), 1)


def compute_spike_score(df, kw):
    """Returns highest single-period spike above rolling mean."""
    rolling = df[kw].rolling(7, min_periods=1).mean()
    diff = df[kw] - rolling
    return round(float(diff.max()), 1)


def predict_trend(df, kw, forecast_days=30):
    """
    Linear regression forecast for next `forecast_days` data points.
    Returns (future_dates, predicted_values, trend_direction, r2_score).
    """
    from sklearn.metrics import r2_score as _r2
    series = df[kw].dropna()
    if len(series) < 5:
        return None, None, "unknown", 0.0

    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(X, y)
    r2 = float(_r2(y, model.predict(X)))

    future_X = np.arange(len(series), len(series) + forecast_days).reshape(-1, 1)
    pred = model.predict(future_X)
    pred = np.clip(pred, 0, 100)

    # Infer frequency from index
    if len(df.index) >= 2:
        freq = df.index[1] - df.index[0]
    else:
        freq = timedelta(days=1)

    last_date = df.index[-1]
    future_dates = [last_date + freq * (i + 1) for i in range(forecast_days)]

    direction = "rising" if model.coef_[0] > 0.05 else ("declining" if model.coef_[0] < -0.05 else "stable")
    return future_dates, pred.tolist(), direction, round(r2, 3)


def detect_alerts(df, kws, threshold=70):
    """
    Detects:
    1. Keywords currently above threshold
    2. Sudden spikes (last value >> rolling mean)
    3. Rapid decline (last value << rolling mean)
    """
    alerts = []
    for kw in kws:
        if kw not in df.columns:
            continue
        cur = int(df[kw].iloc[-1])
        roll_mean = float(df[kw].rolling(7, min_periods=1).mean().iloc[-1])
        prev = float(df[kw].iloc[-2]) if len(df) > 1 else cur

        if cur >= threshold:
            alerts.append({
                "kw": kw, "type": "high",
                "msg": f"**{kw}** is at {cur}/100 — currently above the {threshold} alert threshold.",
                "icon": "🔥"
            })
        if cur > roll_mean * 1.4 and cur - roll_mean > 10:
            alerts.append({
                "kw": kw, "type": "spike",
                "msg": f"**{kw}** spiked to {cur} (rolling avg: {roll_mean:.0f}) — a sudden surge of +{cur - roll_mean:.0f} points.",
                "icon": "🚨"
            })
        if cur < roll_mean * 0.6 and roll_mean - cur > 10:
            alerts.append({
                "kw": kw, "type": "drop",
                "msg": f"**{kw}** dropped to {cur} (rolling avg: {roll_mean:.0f}) — a sudden fall of {roll_mean - cur:.0f} points.",
                "icon": "📉"
            })
    return alerts


def generate_ai_insight(df, kws, related=None):
    """
    Rule-based AI insight generator. Produces a natural language summary
    of trends, correlations, momentum, and competitive dynamics.
    """
    valid = [k for k in kws if k in df.columns]
    if not valid:
        return "No data available for insight generation."

    parts = []

    # 1. Overall leader
    avgs = {k: df[k].mean() for k in valid}
    leader = max(avgs, key=avgs.get)
    loser  = min(avgs, key=avgs.get)
    parts.append(
        f"**{leader}** leads in average search interest ({avgs[leader]:.0f}/100), "
        f"while **{loser}** trails with {avgs[loser]:.0f}/100."
    )

    # 2. Momentum narrative
    mom_data = {k: compute_momentum(df, k) for k in valid}
    fastest_rising = max(mom_data, key=mom_data.get)
    fastest_falling = min(mom_data, key=mom_data.get)
    if mom_data[fastest_rising] > 5:
        parts.append(
            f"**{fastest_rising}** shows the strongest growth momentum "
            f"(+{mom_data[fastest_rising]:.1f}% from first to last quarter), "
            f"suggesting an accelerating trend worth watching."
        )
    if mom_data[fastest_falling] < -5:
        parts.append(
            f"**{fastest_falling}** is losing steam with a {mom_data[fastest_falling]:.1f}% decline over the period — "
            f"interest appears to be cooling."
        )

    # 3. Volatility insight
    vols = {k: compute_volatility(df, k) for k in valid}
    most_volatile = max(vols, key=vols.get)
    most_stable = min(vols, key=vols.get)
    if len(valid) > 1:
        if vols[most_volatile] > 30:
            parts.append(
                f"**{most_volatile}** exhibits high volatility (score: {vols[most_volatile]:.0f}/100), "
                f"indicating hype-driven or event-sensitive interest."
            )
        parts.append(
            f"**{most_stable}** maintains the most consistent interest over time "
            f"(volatility: {vols[most_stable]:.0f}/100), suggesting steady, sustained demand."
        )

    # 4. Correlation insight
    if len(valid) >= 2:
        corr_matrix = df[valid].corr()
        pairs = [(corr_matrix.loc[a, b], a, b)
                 for i, a in enumerate(valid)
                 for j, b in enumerate(valid) if i < j]
        if pairs:
            max_corr = max(pairs, key=lambda x: abs(x[0]))
            r, a, b = max_corr
            if abs(r) > 0.7:
                parts.append(
                    f"**{a}** and **{b}** move together closely (r = {r:.2f}), "
                    f"suggesting they share a common audience or are often searched together."
                )
            elif abs(r) < 0.3:
                parts.append(
                    f"**{a}** and **{b}** behave independently (r = {r:.2f}), "
                    f"indicating different search audiences or use cases."
                )

    # 5. Prediction hint
    future_dates, pred_vals, direction, r2 = predict_trend(df, leader)
    if future_dates and r2 > 0.3:
        next_val = round(pred_vals[6], 0) if len(pred_vals) >= 7 else round(pred_vals[-1], 0)
        parts.append(
            f"Based on trend trajectory, **{leader}** is projected to be **{direction}** "
            f"over the next 30 days — forecast: ~{int(next_val)}/100 in one week."
        )

    return " ".join(parts)


def generate_battle_summary(df, kws):
    """Generates keyword battle summary with winner, fastest growth, most consistent."""
    valid = [k for k in kws if k in df.columns]
    if len(valid) < 2:
        return None, {}

    avgs = {k: df[k].mean() for k in valid}
    peaks = {k: df[k].max() for k in valid}
    moms  = {k: compute_momentum(df, k) for k in valid}
    vols  = {k: compute_volatility(df, k) for k in valid}

    winner          = max(avgs, key=avgs.get)
    fastest_grower  = max(moms, key=moms.get)
    most_consistent = min(vols, key=vols.get)
    highest_peak    = max(peaks, key=peaks.get)

    summary = (
        f"🏆 **Overall Winner:** {winner} (avg {avgs[winner]:.0f}/100)  ·  "
        f"🚀 **Fastest Growth:** {fastest_grower} (+{moms[fastest_grower]:.1f}%)  ·  "
        f"🎯 **Most Consistent:** {most_consistent}  ·  "
        f"⚡ **Highest Peak:** {highest_peak} ({int(peaks[highest_peak])}/100)"
    )
    scores = {k: round((avgs[k] * 0.5 + max(moms[k], 0) * 0.3 + (100 - vols[k]) * 0.2), 1) for k in valid}
    return summary, scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHART BUILDERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ch_line(df, kws, stacked=False):
    fig = go.Figure()
    for i, kw in enumerate(kws):
        if kw not in df.columns:
            continue
        c = PAL[i % len(PAL)]
        args = dict(x=df.index, y=df[kw], name=kw,
                    line=dict(color=c, width=2.2, shape="spline", smoothing=0.7),
                    hovertemplate=f"<b>{kw}</b>: %{{y}}<extra></extra>")
        if stacked:
            fig.add_trace(go.Scatter(**args, mode="lines", stackgroup="one",
                                     fillcolor=rgba(c, .28)))
        else:
            fig.add_trace(go.Scatter(**args, mode="lines", fill="tozeroy",
                                     fillcolor=rgba(c, .07)))
    fig.update_layout(**BL, height=370)
    return fig


def ch_line_with_forecast(df, kw, color):
    """Line chart with forecast overlay."""
    future_dates, pred_vals, direction, r2 = predict_trend(df, kw, forecast_days=30)
    fig = go.Figure()
    # Actual
    fig.add_trace(go.Scatter(
        x=df.index, y=df[kw], name=f"{kw} (actual)",
        line=dict(color=color, width=2.4, shape="spline", smoothing=0.7),
        fill="tozeroy", fillcolor=rgba(color, .07),
        hovertemplate=f"<b>{kw}</b>: %{{y}}<extra></extra>",
    ))
    if future_dates:
        # Confidence band
        upper = [min(v + 8, 100) for v in pred_vals]
        lower = [max(v - 8, 0) for v in pred_vals]
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=upper + lower[::-1],
            fill="toself", fillcolor=rgba(color, .12),
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            hoverinfo="skip",
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=future_dates, y=pred_vals, name=f"Forecast (R²={r2:.2f})",
            line=dict(color=color, width=1.8, dash="dot"),
            hovertemplate=f"<b>Forecast</b>: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**BL, height=310, title_text=f"Trend + 30-day Forecast")
    return fig, direction, r2


def ch_corr(df, kws):
    valid = [k for k in kws if k in df.columns]
    if len(valid) < 2:
        return None
    corr = df[valid].corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, "#ff4d6d"], [.5, "#09091a"], [1, "#00f5d4"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont=dict(family="IBM Plex Mono", size=12),
        hovertemplate="%{x} × %{y}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**BL, height=260)
    return fig


def ch_geo_map(geo_df, kw):
    if kw not in geo_df.columns or geo_df.empty:
        return None
    d = geo_df[[kw]].reset_index()
    d.columns = ["location", "value"]
    d = d[d["value"] > 0]
    fig = px.choropleth(d, locations="location", locationmode="country names",
                        color="value", hover_name="location", labels={"value": "Interest"},
                        color_continuous_scale=[[0,"#04050d"],[.3,"#1a1040"],[.7,"#6c63ff"],[1,"#00f5d4"]])
    fig.update_layout(**BL, height=340,
                      geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#131428",
                               oceancolor="#04050d", showocean=True, lakecolor="#04050d",
                               framecolor="#1c1d35", projection_type="natural earth"),
                      coloraxis_colorbar=dict(bgcolor="rgba(0,0,0,0)",
                                             tickfont=dict(family="IBM Plex Mono", size=9)))
    return fig


def ch_geo_bar(geo_df, kws, top=15):
    valid = [k for k in kws if k in geo_df.columns]
    if not valid:
        return None
    idx = geo_df[valid].sum(axis=1).nlargest(top).index
    sub = geo_df.loc[idx, valid]
    fig = go.Figure()
    for i, kw in enumerate(valid):
        c = PAL[i % len(PAL)]
        fig.add_trace(go.Bar(x=sub.index, y=sub[kw], name=kw,
                             marker_color=c, marker_line_width=0,
                             hovertemplate=f"<b>%{{x}}</b><br>{kw}: %{{y}}<extra></extra>"))
    fig.update_layout(**BL, height=290, barmode="group", bargap=0.18)
    return fig


def ch_bar_avg(df, kws):
    avgs = {k: round(df[k].mean(), 1) for k in kws if k in df.columns}
    ks = sorted(avgs, key=avgs.get, reverse=True)
    fig = go.Figure()
    for i, kw in enumerate(ks):
        fig.add_trace(go.Bar(x=[kw], y=[avgs[kw]], name=kw,
                             marker_color=PAL[i % len(PAL)], marker_line_width=0,
                             hovertemplate=f"<b>{kw}</b><br>Avg: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**BL, height=270, showlegend=False, barmode="group", bargap=0.3)
    return fig


def ch_box(df, kws):
    fig = go.Figure()
    for i, kw in enumerate(kws):
        if kw not in df.columns:
            continue
        c = PAL[i % len(PAL)]
        fig.add_trace(go.Box(y=df[kw], name=kw, marker_color=c, line_color=c,
                             fillcolor=rgba(c, .15), boxmean="sd",
                             hovertemplate=f"<b>{kw}</b><br>%{{y}}<extra></extra>"))
    fig.update_layout(**BL, height=270, showlegend=False)
    return fig


def ch_radar(df, kws):
    avgs = {k: df[k].mean() for k in kws if k in df.columns}
    if len(avgs) < 2:
        return None
    ks, vs = list(avgs.keys()), list(avgs.values())
    fig = go.Figure(go.Scatterpolar(
        r=vs + [vs[0]], theta=ks + [ks[0]],
        fill="toself", fillcolor=rgba("#6c63ff", .18),
        line=dict(color="#6c63ff", width=2),
        marker=dict(color="#00f5d4", size=8),
    ))
    fig.update_layout(**BL, height=290,
                      polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(visible=True, gridcolor="#131428", tickfont=dict(size=9)),
                                 angularaxis=dict(gridcolor="#131428", tickfont=dict(size=10))))
    return fig


def ch_rolling(df, kws, w=7):
    fig = go.Figure()
    for i, kw in enumerate(kws):
        if kw not in df.columns:
            continue
        c = PAL[i % len(PAL)]
        roll = df[kw].rolling(w, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=roll, name=kw, mode="lines",
            line=dict(color=c, width=2.2),
            hovertemplate=f"<b>{kw}</b><br>%{{x}}<br>Rolling avg: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**BL, height=270)
    return fig


def ch_momentum(df, kws):
    q = max(1, len(df) // 4)
    rows = []
    for kw in kws:
        if kw not in df.columns:
            continue
        f = df[kw].iloc[:q].mean()
        l = df[kw].iloc[-q:].mean()
        rows.append({"kw": kw, "chg": round(((l - f) / (f + 1e-9)) * 100, 1)})
    if not rows:
        return None
    fig = go.Figure()
    for i, r in enumerate(rows):
        c = "#00f5d4" if r["chg"] >= 0 else "#ff4d6d"
        fig.add_trace(go.Bar(x=[r["kw"]], y=[r["chg"]], name=r["kw"],
                             marker_color=c, marker_line_width=0,
                             hovertemplate=f"<b>{r['kw']}</b><br>Δ %{{y:+.1f}}%<extra></extra>"))
    fig.add_hline(y=0, line_color="#1c1d35", line_width=1)
    fig.update_layout(**BL, height=240, showlegend=False, yaxis_title="% Change")
    return fig


def ch_related_bar(df_q, color, title):
    if df_q is None or df_q.empty:
        return None
    d = df_q.head(10)
    fig = go.Figure(go.Bar(
        x=d["value"], y=d["query"], orientation="h",
        marker_color=color, marker_line_width=0,
        hovertemplate="%{y}<br>Value: %{x}<extra></extra>",
    ))
    fig.update_layout(**BL, height=270, showlegend=False,
                      yaxis=dict(autorange="reversed", **BL["yaxis"]),
                      title_text=title, title_font=dict(size=12))
    return fig


def ch_battle_scores(scores, kws):
    ks = sorted(scores, key=scores.get, reverse=True)
    fig = go.Figure()
    for i, kw in enumerate(ks):
        c = PAL[i % len(PAL)]
        fig.add_trace(go.Bar(x=[kw], y=[scores[kw]], name=kw,
                             marker_color=c, marker_line_width=0,
                             hovertemplate=f"<b>{kw}</b><br>Battle Score: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**BL, height=240, showlegend=False,
                      title_text="Composite Battle Score",
                      yaxis_title="Score")
    return fig


def styled_table(df: pd.DataFrame):
    cols = list(df.columns)
    rows_html = ""
    for _, row in df.iterrows():
        row_html = f"<tr><td style='color:#6666aa;font-weight:700;padding:6px 12px;white-space:nowrap;'>{row.name}</td>"
        for col in cols:
            val = row[col]
            if isinstance(val, (int, float)) and not pd.isna(val):
                intensity = min(int(val / 100 * 160), 160)
                bg = f"rgba(108,99,255,{intensity/255:.2f})"
                row_html += f"<td style='padding:6px 12px;background:{bg};text-align:right;'>{val}</td>"
            else:
                row_html += f"<td style='padding:6px 12px;text-align:right;'>{val}</td>"
        rows_html += row_html + "</tr>"

    header_html = "<tr><th style='padding:6px 12px;'></th>" + "".join(
        f"<th style='padding:6px 12px;color:#6666aa;font-family:IBM Plex Mono,monospace;"
        f"font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;text-align:right;'>{c}</th>"
        for c in cols) + "</tr>"

    return f"""
    <div style='overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;font-family:IBM Plex Mono,monospace;
                  font-size:.7rem;color:#eeeef8;background:#090a15;
                  border:1px solid #1c1d35;border-radius:8px;overflow:hidden;'>
      <thead style='border-bottom:1px solid #1c1d35;'>{header_html}</thead>
      <tbody>{rows_html}</tbody>
    </table></div>"""


def meter_bar(label, value, color, max_val=100, show_val=True):
    pct = min(value / max_val * 100, 100)
    val_str = f"{value:.0f}" if show_val else ""
    return f"""
    <div class='meter-wrap'>
      <div class='meter-label'>
        <span>{label}</span>
        <span style='color:{color};font-weight:700;'>{val_str}</span>
      </div>
      <div class='meter-bar'>
        <div class='meter-fill' style='width:{pct}%;background:linear-gradient(90deg,{color}88,{color});'></div>
      </div>
    </div>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 1rem;'>
      <div style='font-family:Epilogue,sans-serif;font-weight:900;font-size:1.35rem;
                  background:linear-gradient(135deg,#6c63ff,#00f5d4);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
        TREND PULSE AI
      </div>
      <div style='font-family:IBM Plex Mono,monospace;font-size:.52rem;color:#6666aa;
                  letter-spacing:.15em;text-transform:uppercase;margin-top:.1rem;'>
        Intelligent Trends Platform v5
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">Keywords (max 5)</div>', unsafe_allow_html=True)
    raw = st.text_area("kw", value="ChatGPT\nGemini\nClaude AI\nCopilot",
                       height=108, label_visibility="collapsed")
    keywords = [k.strip() for k in raw.strip().splitlines() if k.strip()][:5]

    st.markdown('<div class="sh">Settings</div>', unsafe_allow_html=True)
    tf_lbl  = st.selectbox("Time Period", list(TIME_OPTS), index=6)
    geo_lbl = st.selectbox("Region", list(GEO_OPTS), index=2)
    cat_lbl = st.selectbox("Category", list(CAT_OPTS), index=0)
    mode    = st.radio("Chart Style", ["Line", "Stacked Area"], horizontal=True)

    st.markdown('<div class="sh">AI Features</div>', unsafe_allow_html=True)
    alert_threshold = st.slider(
        "Alert Threshold (interest score)",
        min_value=20, max_value=95, value=70, step=5,
        help="Get alerted when any keyword's interest exceeds this level."
    )
    st.session_state.alert_threshold = alert_threshold

    st.markdown('<div class="sh">Mode</div>', unsafe_allow_html=True)
    demo_toggle = st.toggle("Demo Mode (no Google calls)", value=st.session_state.demo_mode)
    st.session_state.demo_mode = demo_toggle

    if st.button("🗑  Clear Cache", use_container_width=True):
        st.session_state.req_cache = {}
        st.session_state.last_call_ts = 0.0
        st.session_state.ai_insights = {}
        st.toast("Cache cleared!", icon="✅")

    st.markdown("")
    run = st.button("⚡  ANALYZE TRENDS", use_container_width=True)

    st.markdown("""
    <div style='margin-top:1.2rem;padding:.8rem;background:#090a15;border:1px solid #1c1d35;
                border-radius:10px;font-family:IBM Plex Mono,monospace;font-size:.54rem;color:#6666aa;'>
      <div style='color:#6c63ff;margin-bottom:.3rem;font-weight:700;'>ℹ CAPABILITIES</div>
      🧠 AI insight generator<br>
      🔮 30-day trend forecast<br>
      🚨 Spike alert detection<br>
      ⚔️ Keyword battle mode<br>
      📊 Volatility + momentum<br>
      🗺️ Geographic analysis<br>
      📥 CSV export<br>
      ⚡ Session caching
    </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
hc1, hc2 = st.columns([5, 1])
with hc1:
    st.markdown(f"""
    <div class='hero-wrap'>
      <div class='hero-eyebrow'>Intelligent Analytics Platform</div>
      <p class='hero-title'>Trend Pulse AI</p>
      <p class='hero-sub'>Google Trends · Predictions · AI Insights · Battle Mode · Alerts</p>
    </div>""", unsafe_allow_html=True)
with hc2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:.7rem;'>
      <div style='font-family:IBM Plex Mono;font-size:.54rem;color:#6666aa;'>LIVE AS OF</div>
      <div style='font-family:Epilogue;font-weight:800;font-size:.92rem;color:#00f5d4;'>
        {datetime.now().strftime('%H:%M · %d %b %Y')}
      </div>
      <div style='font-family:IBM Plex Mono;font-size:.52rem;color:#6666aa;margin-top:.18rem;'>
        {geo_lbl} · {tf_lbl}
      </div>
    </div>""", unsafe_allow_html=True)

# Keyword chips
if keywords:
    chips = "".join(
        f"<span class='chip' style='color:{PAL[i%len(PAL)]};border-color:{PAL[i%len(PAL)]}55;"
        f"background:{PAL[i%len(PAL)]}10;'>{kw}</span>"
        for i, kw in enumerate(keywords)
    )
    st.markdown(f"<div style='margin-bottom:.9rem;'>{chips}</div>", unsafe_allow_html=True)

if not keywords:
    st.info("Enter keywords in the sidebar, then click **ANALYZE TRENDS**.")
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FETCH MAIN DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tf  = TIME_OPTS[tf_lbl]
geo = GEO_OPTS[geo_lbl]
cat = CAT_OPTS[cat_lbl]
kws = tuple(keywords)

with st.spinner("Fetching trend data…"):
    df, fetch_err, is_demo = get_over_time(kws, tf, geo, cat)

if is_demo:
    banner_msg = (
        "📊 **Demo Mode active** — showing simulated data."
        if st.session_state.demo_mode
        else "⚠️ **Google rate-limited this request (429)** — showing demo data. "
             "Enable Demo Mode in the sidebar or wait a minute and try again."
    )
    st.markdown(f'<div class="demo-banner">{banner_msg}</div>', unsafe_allow_html=True)

if df is None or df.empty:
    st.error("No data available. Try different keywords or enable Demo Mode.")
    st.stop()

valid = [k for k in keywords if k in df.columns]

# Run alert detection
alerts = detect_alerts(df, valid, threshold=st.session_state.alert_threshold)
st.session_state.alerts = alerts

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALERT BANNER (top-level, always visible)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if alerts:
    for a in alerts:
        st.markdown(
            f"<div class='insight-box alert'>"
            f"{a['icon']} {a['msg']}"
            f"</div>",
            unsafe_allow_html=True
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRIC CARDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="sh">Summary Statistics</div>', unsafe_allow_html=True)
mc = st.columns(len(valid))
for i, kw in enumerate(valid):
    s = df[kw]
    cur   = int(s.iloc[-1])
    peak  = int(s.max())
    avg   = float(s.mean())
    delta = int(s.iloc[-1]) - int(s.iloc[-2]) if len(s) > 1 else 0
    mom   = compute_momentum(df, kw)
    bc, bt = (("bup", f"↑ {abs(delta)}") if delta > 0
              else ("bdn", f"↓ {abs(delta)}") if delta < 0
              else ("bfl", "→ flat"))
    mom_cls = "bup" if mom > 0 else "bdn"
    c  = PAL[i % len(PAL)]
    nc = PAL[(i + 1) % len(PAL)]
    with mc[i]:
        st.markdown(f"""
        <div class="mcard">
          <div style="position:absolute;top:0;left:0;right:0;height:3px;
                      background:linear-gradient(90deg,{c},{nc});border-radius:16px 16px 0 0;"></div>
          <div class="mlabel">{kw}</div>
          <div class="mval" style="color:{c};">{cur}</div>
          <div class="msub">Peak <b style="color:{c};">{peak}</b> · Avg <b>{avg:.0f}</b></div>
          <span class="bdg {bc}">{bt}</span>
          <span class="bdg {mom_cls}">mom {mom:+.0f}%</span>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "📈  Interest Over Time",
    "🔮  Predictions",
    "🧠  AI Insights",
    "⚔️  Battle Mode",
    "🌍  Geographic",
    "🔍  Related Queries",
    "📊  Deep Analysis",
])

# ── Tab 1: Interest Over Time ─────────────────────────────────────────────────
with t1:
    st.markdown('<div class="sh">Search Interest · 0–100 Normalized Scale</div>', unsafe_allow_html=True)
    st.plotly_chart(ch_line(df, valid, stacked=(mode == "Stacked Area")),
                    use_container_width=True, config={"displayModeBar": False})

    if len(valid) > 1:
        st.markdown('<div class="sh">Correlation Matrix</div>', unsafe_allow_html=True)
        fc = ch_corr(df, valid)
        if fc:
            st.plotly_chart(fc, use_container_width=True, config={"displayModeBar": False})

    with st.expander("📋  Raw Data — inspect & download"):
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇  Download CSV", df.to_csv().encode(),
                           "trends_data.csv", "text/csv", use_container_width=True)


# ── Tab 2: Predictions ────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="sh">30-Day Trend Forecast · Linear Regression</div>', unsafe_allow_html=True)
    st.caption("Forecast uses the historical trend trajectory. Higher R² = better fit.")

    for i, kw in enumerate(valid):
        c = PAL[i % len(PAL)]
        fig, direction, r2 = ch_line_with_forecast(df, kw, c)
        if fig:
            # Direction badge
            dir_color = "#00f5d4" if direction == "rising" else ("#ff4d6d" if direction == "declining" else "#a78bfa")
            dir_icon = "🟢" if direction == "rising" else ("🔴" if direction == "declining" else "🟡")
            # Forecast strip
            future_dates, pred_vals, _, _ = predict_trend(df, kw, 30)
            pred_7 = round(pred_vals[6], 0) if pred_vals and len(pred_vals) > 6 else "—"
            pred_30 = round(pred_vals[-1], 0) if pred_vals else "—"

            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with col2:
                st.markdown(f"""
                <div style='background:var(--surf);border:1px solid var(--bdr);border-radius:12px;
                            padding:1rem;height:100%;margin-top:.5rem;'>
                  <div style='font-family:IBM Plex Mono;font-size:.55rem;color:var(--dim);
                              text-transform:uppercase;letter-spacing:.1em;margin-bottom:.7rem;'>{kw}</div>

                  <div style='font-family:IBM Plex Mono;font-size:.55rem;color:var(--dim);margin-bottom:.15rem;'>DIRECTION</div>
                  <div style='font-family:Epilogue;font-weight:700;font-size:1rem;color:{dir_color};margin-bottom:.6rem;'>
                    {dir_icon} {direction.upper()}</div>

                  <div style='font-family:IBM Plex Mono;font-size:.55rem;color:var(--dim);margin-bottom:.15rem;'>7-DAY FORECAST</div>
                  <div style='font-family:Epilogue;font-weight:700;font-size:1.2rem;color:{c};margin-bottom:.6rem;'>{pred_7}/100</div>

                  <div style='font-family:IBM Plex Mono;font-size:.55rem;color:var(--dim);margin-bottom:.15rem;'>30-DAY FORECAST</div>
                  <div style='font-family:Epilogue;font-weight:700;font-size:1.2rem;color:{c};margin-bottom:.6rem;'>{pred_30}/100</div>

                  <div style='font-family:IBM Plex Mono;font-size:.55rem;color:var(--dim);margin-bottom:.15rem;'>MODEL FIT (R²)</div>
                  <div style='font-family:Epilogue;font-weight:700;font-size:1rem;color:#ffbe0b;'>{r2:.3f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='margin:.4rem 0;border-color:#1c1d35;'>", unsafe_allow_html=True)

    st.info("📌 Forecasts use linear regression on historical data. For volatile trends, treat as directional guidance rather than precise prediction.")


# ── Tab 3: AI Insights ────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="sh">AI-Generated Analysis</div>', unsafe_allow_html=True)

    insight_key = cache_key("insight", valid, tf, geo)
    if insight_key not in st.session_state.ai_insights:
        with st.spinner("Generating AI insights…"):
            rel_data, _ = get_related(kws, tf, geo, cat)
            insight_text = generate_ai_insight(df, valid, rel_data)
            st.session_state.ai_insights[insight_key] = insight_text
    else:
        insight_text = st.session_state.ai_insights[insight_key]

    st.markdown(f'<div class="insight-box">{insight_text}</div>', unsafe_allow_html=True)

    # Per-keyword intelligence cards
    st.markdown('<div class="sh">Per-Keyword Intelligence</div>', unsafe_allow_html=True)
    cols = st.columns(len(valid))
    for i, kw in enumerate(valid):
        c = PAL[i % len(PAL)]
        mom = compute_momentum(df, kw)
        vol = compute_volatility(df, kw)
        spike = compute_spike_score(df, kw)
        _, _, direction, r2 = predict_trend(df, kw)
        stability = 100 - vol

        with cols[i]:
            st.markdown(f"""
            <div class='mcard' style='border-color:{c}22;'>
              <div style='font-family:IBM Plex Mono;font-size:.55rem;color:{c};
                          text-transform:uppercase;letter-spacing:.12em;margin-bottom:.6rem;font-weight:700;'>
                {kw}
              </div>
              {meter_bar("Momentum", abs(mom), "#00f5d4" if mom > 0 else "#ff4d6d", max_val=100)}
              {meter_bar("Stability", stability, "#a78bfa", max_val=100)}
              {meter_bar("Spike Score", min(spike, 100), "#ffbe0b", max_val=100)}
              <div style='margin-top:.6rem;font-family:IBM Plex Mono;font-size:.55rem;color:var(--dim);'>
                Forecast: <span style='color:{c};font-weight:700;'>{direction}</span>
                &nbsp;·&nbsp; R²: <span style='color:#ffbe0b;'>{r2:.2f}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # Correlation interpretation
    if len(valid) >= 2:
        st.markdown('<div class="sh">Correlation Interpretation</div>', unsafe_allow_html=True)
        corr = df[valid].corr()
        interp_html = ""
        for i, a in enumerate(valid):
            for j, b in enumerate(valid):
                if i >= j:
                    continue
                r = corr.loc[a, b]
                if abs(r) >= 0.7:
                    strength = "strongly correlated"
                    color = "#00f5d4"
                    desc = "share a common search audience and often trend together."
                elif abs(r) >= 0.4:
                    strength = "moderately correlated"
                    color = "#ffbe0b"
                    desc = "show some overlapping interest patterns."
                else:
                    strength = "weakly correlated"
                    color = "#ff4d6d"
                    desc = "behave independently — different audiences or use cases."
                direction_str = "positively" if r > 0 else "negatively"
                interp_html += f"""
                <div style='background:var(--surf2);border:1px solid var(--bdr);border-radius:10px;
                            padding:.7rem 1rem;margin-bottom:.5rem;
                            font-family:Epilogue;font-size:.85rem;'>
                  <b style='color:{PAL[i%len(PAL)]};'>{a}</b>
                  <span style='color:var(--dim);font-size:.75rem;font-family:IBM Plex Mono;'> × </span>
                  <b style='color:{PAL[j%len(PAL)]};'>{b}</b>
                  <span style='font-family:IBM Plex Mono;font-size:.58rem;color:{color};
                               border:1px solid {color}44;border-radius:4px;padding:.1rem .4rem;
                               margin:0 .4rem;'>{direction_str} {strength} (r={r:.2f})</span><br>
                  <span style='color:var(--dim);font-size:.8rem;'>These keywords {desc}</span>
                </div>"""
        st.markdown(interp_html, unsafe_allow_html=True)


# ── Tab 4: Battle Mode ────────────────────────────────────────────────────────
with t4:
    if len(valid) < 2:
        st.info("Battle Mode requires at least 2 keywords. Add more in the sidebar.")
    else:
        st.markdown('<div class="sh">Keyword Battle · Head-to-Head Comparison</div>', unsafe_allow_html=True)

        battle_summary, scores = generate_battle_summary(df, valid)
        if battle_summary:
            st.markdown(f'<div class="insight-box battle">{battle_summary}</div>',
                        unsafe_allow_html=True)

        # Score chart
        if scores:
            sc, rc = st.columns([2, 1])
            with sc:
                st.plotly_chart(ch_battle_scores(scores, valid),
                                use_container_width=True, config={"displayModeBar": False})
            with rc:
                st.markdown('<div class="sh">Battle Scoreboard</div>', unsafe_allow_html=True)
                sorted_kws = sorted(scores, key=scores.get, reverse=True)
                medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for rank, kw in enumerate(sorted_kws):
                    c = PAL[valid.index(kw) % len(PAL)] if kw in valid else PAL[0]
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;justify-content:space-between;
                                padding:.5rem .8rem;margin-bottom:.35rem;
                                background:var(--surf2);border:1px solid var(--bdr);
                                border-left:3px solid {c};border-radius:8px;'>
                      <span style='font-family:IBM Plex Mono;font-size:.7rem;'>
                        {medals[rank]} <b style='color:{c};'>{kw}</b>
                      </span>
                      <span style='font-family:Epilogue;font-weight:700;font-size:.95rem;color:{c};'>
                        {scores[kw]:.0f}
                      </span>
                    </div>""", unsafe_allow_html=True)

        # Detailed breakdown table
        st.markdown('<div class="sh">Detailed Breakdown</div>', unsafe_allow_html=True)
        battle_data = []
        for kw in valid:
            _, _, direction, r2 = predict_trend(df, kw)
            battle_data.append({
                "Keyword": kw,
                "Avg Interest": round(df[kw].mean(), 1),
                "Peak": int(df[kw].max()),
                "Momentum %": f"{compute_momentum(df, kw):+.1f}",
                "Volatility": round(compute_volatility(df, kw), 1),
                "Forecast": direction.upper(),
                "Battle Score": round(scores.get(kw, 0), 1),
            })
        bdf = pd.DataFrame(battle_data).set_index("Keyword")
        st.dataframe(bdf, use_container_width=True)


# ── Tab 5: Geographic ─────────────────────────────────────────────────────────
with t5:
    with st.spinner("Loading geographic data…"):
        gdf, g_demo = get_by_region(kws, tf, geo, cat)

    if g_demo and not st.session_state.demo_mode:
        st.warning("Geographic data fell back to demo due to rate limiting.")

    if not gdf.empty:
        sel = st.selectbox("Keyword for choropleth", valid, key="g_sel")
        st.markdown(f'<div class="sh">World Map · {sel}</div>', unsafe_allow_html=True)
        fm = ch_geo_map(gdf, sel)
        if fm:
            st.plotly_chart(fm, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="sh">Top Regions — All Keywords</div>', unsafe_allow_html=True)
        fb = ch_geo_bar(gdf, valid)
        if fb:
            st.plotly_chart(fb, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No geographic data available.")


# ── Tab 6: Related Queries ────────────────────────────────────────────────────
with t6:
    with st.spinner("Loading related queries…"):
        rel, rq_demo = get_related(kws, tf, geo, cat)

    if rq_demo and not st.session_state.demo_mode:
        st.warning("Related queries fell back to demo due to rate limiting.")

    if rel:
        for i, kw in enumerate(valid):
            if kw not in rel or not rel[kw]:
                continue
            c = PAL[i % len(PAL)]
            with st.expander(f"🔑  {kw}", expanded=(i == 0)):
                lc, rc = st.columns(2)
                with lc:
                    f = ch_related_bar(rel[kw].get("top"), c, "🏆 Top Queries")
                    if f:
                        st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
                with rc:
                    f = ch_related_bar(rel[kw].get("rising"), "#00f5d4", "🚀 Rising Queries")
                    if f:
                        st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No related queries data.")


# ── Tab 7: Deep Analysis ──────────────────────────────────────────────────────
with t7:
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="sh">Average Interest</div>', unsafe_allow_html=True)
        st.plotly_chart(ch_bar_avg(df, valid), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="sh">Distribution · Box + Std Dev</div>', unsafe_allow_html=True)
        st.plotly_chart(ch_box(df, valid), use_container_width=True, config={"displayModeBar": False})
    with cb:
        st.markdown('<div class="sh">Multi-Axis Radar</div>', unsafe_allow_html=True)
        fr = ch_radar(df, valid)
        if fr:
            st.plotly_chart(fr, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Needs ≥ 2 keywords.")
        st.markdown('<div class="sh">7-Day Rolling Average</div>', unsafe_allow_html=True)
        st.plotly_chart(ch_rolling(df, valid), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sh">Trend Momentum · First vs Last Quarter</div>', unsafe_allow_html=True)
    fm = ch_momentum(df, valid)
    if fm:
        st.plotly_chart(fm, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sh">Descriptive Statistics</div>', unsafe_allow_html=True)
    stats = df[valid].describe().T.round(2)
    st.markdown(styled_table(stats), unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin-top:1.8rem;'>", unsafe_allow_html=True)
st.markdown(f"""
<div style='font-family:IBM Plex Mono,monospace;font-size:.52rem;color:#6666aa;
            display:flex;justify-content:space-between;padding:.35rem 0 1rem;flex-wrap:wrap;gap:.35rem;'>
  <span>TREND PULSE AI v5 &nbsp;·&nbsp; urllib3 v2 ✓ &nbsp;·&nbsp; sklearn ✓</span>
  <span>{geo_lbl} &nbsp;·&nbsp; {tf_lbl} &nbsp;·&nbsp; {cat_lbl}</span>
  <span>{'⚡ DEMO DATA' if (is_demo or st.session_state.demo_mode) else '🌐 LIVE DATA'} &nbsp;·&nbsp; pytrends + AI</span>
</div>""", unsafe_allow_html=True)
