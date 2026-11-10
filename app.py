import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="Breakout Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BIG_BARS = 10**6
BATCH_SIZE = 60
MODE_FRESH = "Taze Buy sinyali (son N bar)"
MODE_ACTIVE = "Aktif Long (Buy durumunda)"

NO_MARK = "—"
MARK_OPTIONS = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
EMOJI_TO_MARK = {emoji: name for name, emoji in MARK_OPTIONS.items()}
MARK_CSS = {
    "green": "background-color: rgba(34, 197, 94, 0.28);",
    "yellow": "background-color: rgba(234, 179, 8, 0.32);",
    "red": "background-color: rgba(239, 68, 68, 0.30);",
}

MARKETS = {
    "us": {
        "label": "🇺🇸 Amerika (NASDAQ & NYSE)",
        "tv_market": "america",
        "exchanges": ["NASDAQ", "NYSE"],
        "suffix": "",
        "dot_to_dash": True,
        "tz": "America/New_York",
        "tz_label": "ET",
        "open_min": 570,
        "split_min": 810,
        "close_min": 960,
        "currency": "$",
        "common_only": True,
        "defaults": {"limit": 300, "min_cap": 1000, "min_price": 5.0, "min_volume": 300000},
    },
    "tr": {
        "label": "🇹🇷 Türkiye (BIST)",
        "tv_market": "turkey",
        "exchanges": ["BIST"],
        "suffix": ".IS",
        "dot_to_dash": False,
        "tz": "Europe/Istanbul",
        "tz_label": "TRT",
        "open_min": 600,
        "split_min": 840,
        "close_min": 1080,
        "currency": "₺",
        "common_only": False,
        "defaults": {"limit": 300, "min_cap": 1000, "min_price": 1.0, "min_volume": 500000},
    },
}

TF_ORDER = ["1h", "4h", "1d", "1w"]
TF_CONFIG = {
    "1h": {"interval": "1h", "period": "6mo", "label": "1 Saat"},
    "4h": {"interval": "1h", "period": "6mo", "label": "4 Saat"},
    "1d": {"interval": "1d", "period": "2y", "label": "1 Gün"},
    "1w": {"interval": "1wk", "period": "10y", "label": "1 Hafta"},
}
BT_PERIODS = {
    "1h": ["3mo", "6mo", "1y"],
    "4h": ["3mo", "6mo", "1y"],
    "1d": ["1y", "2y", "5y", "10y", "max"],
    "1w": ["5y", "10y", "max"],
}

TEXT_COLUMNS = {"name", "description", "exchange", "sector", "industry", "tv_symbol", "ticker", "symbol"}
MIN_COLUMNS = [
    "name", "description", "exchange", "sector", "industry",
    "close", "volume", "market_cap_basic",
]
BASE_COLUMNS = MIN_COLUMNS + [
    "price_earnings_ttm", "earnings_per_share_diluted_ttm", "dividends_yield_current",
    "beta_1_year", "relative_volume_10d_calc", "average_volume_90d_calc",
    "Perf.W", "Perf.1M", "Perf.3M", "Perf.Y",
    "price_52_week_high", "price_52_week_low", "Recommend.All",
]
EXTENDED_COLUMNS = BASE_COLUMNS + [
    "price_book_fq", "price_sales_current", "return_on_equity", "debt_to_equity",
    "gross_margin", "net_margin", "total_revenue_yoy_growth_ttm",
    "earnings_per_share_diluted_yoy_growth_ttm",
]
COLUMN_SETS = [EXTENDED_COLUMNS, BASE_COLUMNS, MIN_COLUMNS]

FUND_LABELS = {
    "market_cap_basic": "Piyasa Değeri",
    "price_earnings_ttm": "F/K",
    "earnings_per_share_diluted_ttm": "HBK",
    "dividends_yield_current": "Temettü Verimi %",
    "beta_1_year": "Beta (1Y)",
    "price_book_fq": "PD/DD",
    "price_sales_current": "F/S",
    "return_on_equity": "ROE %",
    "debt_to_equity": "Borç/Özkaynak",
    "gross_margin": "Brüt Marj %",
    "net_margin": "Net Marj %",
    "total_revenue_yoy_growth_ttm": "Gelir Büyümesi % (YoY)",
    "earnings_per_share_diluted_yoy_growth_ttm": "HBK Büyümesi % (YoY)",
    "Perf.W": "1H Perf %",
    "Perf.1M": "1A Perf %",
    "Perf.3M": "3A Perf %",
    "Perf.Y": "1Y Perf %",
}


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background: rgba(120, 130, 150, 0.10);
            border: 1px solid rgba(120, 130, 150, 0.25);
            padding: 0.7rem 0.9rem;
            border-radius: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def nz(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(number) or np.isinf(number):
        return float(default)
    return number


@st.cache_resource(show_spinner=False)
def get_http_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=8)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.tradingview.com",
            "Referer": "https://www.tradingview.com/",
        }
    )
    return session


@st.cache_data(ttl=900, show_spinner=False)
def fetch_universe(market, limit, min_cap, min_price, min_volume):
    mkt_config = MARKETS[market]
    tv_url = f"https://scanner.tradingview.com/{mkt_config['tv_market']}/scan"
    session = get_http_session()
    filters = [
        {"left": "type", "operation": "equal", "right": "stock"},
        {"left": "exchange", "operation": "in_range", "right": mkt_config["exchanges"]},
        {"left": "market_cap_basic", "operation": "egreater", "right": float(min_cap)},
        {"left": "close", "operation": "egreater", "right": float(min_price)},
        {"left": "volume", "operation": "egreater", "right": float(min_volume)},
    ]
    if mkt_config["common_only"]:
        filters.insert(2, {"left": "typespecs", "operation": "has", "right": ["common"]})
    last_error = ""
    for columns in COLUMN_SETS:
        payload = {
            "filter": filters,
            "options": {"lang": "en"},
            "markets": [mkt_config["tv_market"]],
            "symbols": {"query": {"types": []}, "tickers": []},
            "columns": columns,
            "sort": {"sortBy": "market_cap_basic", "sortOrder": "desc"},
            "range": [0, int(limit)],
        }
        try:
            response = session.post(tv_url, json=payload, timeout=25)
            if response.status_code != 200:
                last_error = f"TradingView Scanner HTTP {response.status_code}"
                continue
            data = response.json().get("data", [])
        except Exception as exc:
            last_error = f"TradingView Scanner hatası: {exc}"
            continue
        rows = []
        for item in data:
            values = item.get("d", [])
            if len(values) != len(columns):
                continue
            row = dict(zip(columns, values))
            row["tv_symbol"] = item.get("s", "")
            rows.append(row)
        if not rows:
            last_error = "TradingView Scanner boş sonuç döndürdü."
            continue
        df = pd.DataFrame(rows)
        for column in df.columns:
            if column not in TEXT_COLUMNS:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        df["ticker"] = df["name"].astype(str).str.upper()
        if mkt_config["dot_to_dash"]:
            df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        df = df[df["ticker"].str.match(r"^[A-Z][A-Z0-9\-]{0,5}$")]
        df = df.drop_duplicates("ticker").reset_index(drop=True)
        df["symbol"] = df["ticker"] + mkt_config["suffix"]
        df["sector"] = df["sector"].fillna("Diğer")
        df["industry"] = df["industry"].fillna("Diğer")
        df["description"] = df["description"].fillna(df["ticker"])
        return df, ""
    return pd.DataFrame(), last_error


def extract_frames(raw, symbols):
    frames = {}
    columns = ["Open", "High", "Low", "Close", "Volume"]
    if raw is None or raw.empty:
        return frames
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            raw = raw.swaplevel(0, 1, axis=1)
        raw = raw.sort_index(axis=1)
        available = set(raw.columns.get_level_values(0))
        for symbol in symbols:
            if symbol not in available:
                continue
            sub = raw[symbol]
            if not all(col in sub.columns for col in columns):
                continue
            sub = sub[columns].dropna(subset=["Open", "High", "Low", "Close"])
            if not sub.empty:
                frames[symbol] = sub
    elif len(symbols) == 1 and all(col in raw.columns for col in columns):
        sub = raw[columns].dropna(subset=["Open", "High", "Low", "Close"])
        if not sub.empty:
            frames[symbols[0]] = sub
    return frames


@st.cache_data(ttl=300, show_spinner=False, max_entries=24)
def download_batch(symbols, interval, period):
    raw = None
    for attempt in range(3):
        try:
            raw = yf.download(
                list(symbols),
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
                timeout=30,
            )
            if raw is not None and not raw.empty:
                break
        except Exception:
            raw = None
        time.sleep(1.2 * (attempt + 1))
    try:
        return extract_frames(raw, list(symbols))
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False, max_entries=32)
def fetch_history(ticker, tf, market):
    cfg = TF_CONFIG[tf]
    symbol = ticker + MARKETS[market]["suffix"]
    frames = download_batch((symbol,), cfg["interval"], cfg["period"])
    return frames.get(symbol)


@st.cache_data(ttl=300, show_spinner=False, max_entries=24)
def fetch_history_period(ticker, tf, market, period):
    cfg = TF_CONFIG[tf]
    symbol = ticker + MARKETS[market]["suffix"]
    frames = download_batch((symbol,), cfg["interval"], period)
    return frames.get(symbol)


def localize_market(df, tf, tz):
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if tf in ("1d", "1w"):
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize().tz_localize(tz)
    else:
        if idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert(tz)
        else:
            idx = idx.tz_convert(tz)
    df.index = idx
    return df


def resample_4h(df, split_min):
    work = df.copy()
    work["_d"] = work.index.normalize()
    work["_b"] = ((work.index.hour * 60 + work.index.minute) >= split_min).astype(int)
    work["_ts"] = work.index
    grouped = work.groupby(["_d", "_b"], sort=True).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
        _ts=("_ts", "first"),
    )
    grouped.index = pd.DatetimeIndex(grouped["_ts"])
    return grouped.drop(columns="_ts").sort_index()


def bar_end_time(ts, tf, mkt):
    day = ts.normalize()
    close_time = day + pd.Timedelta(minutes=mkt["close_min"])
    if tf == "1h":
        return min(ts + pd.Timedelta(hours=1), close_time)
    if tf == "4h":
        minutes = ts.hour * 60 + ts.minute
        if minutes < mkt["split_min"]:
            return day + pd.Timedelta(minutes=mkt["split_min"])
        return close_time
    if tf == "1d":
        return close_time
    monday = day - pd.Timedelta(days=ts.weekday())
    return monday + pd.Timedelta(days=4, minutes=mkt["close_min"])


def prepare_frame(raw, tf, closed_only, market):
    mkt = MARKETS[market]
    df = raw.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return df
    df = localize_market(df, tf, mkt["tz"])
    if tf == "4h":
        df = resample_4h(df, mkt["split_min"])
    df["Volume"] = df["Volume"].fillna(0.0)
    if closed_only and len(df) > 0:
        now = pd.Timestamp.now(tz=mkt["tz"])
        if bar_end_time(df.index[-1], tf, mkt) > now:
            df = df.iloc[:-1]
    return df


def rma(values, length):
    out = np.full(len(values), np.nan)
    if len(values) < length:
        return out
    out[length - 1] = np.mean(values[:length])
    alpha = 1.0 / length
    for i in range(length, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def true_range(high, low, close):
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def compute_slope(high, low, close, length, mult, method):
    n = len(close)
    if method == "Atr":
        prev_close = np.concatenate(([np.nan], close[:-1]))
        tr = np.where(
            np.isnan(prev_close),
            high - low,
            np.maximum.reduce(
                [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
            ),
        )
        return rma(tr, length) / length * mult
    series = pd.Series(close)
    if method == "Stdev":
        return (series.rolling(length).std(ddof=0) / length * mult).to_numpy()
    index = pd.Series(np.arange(n, dtype=float))
    covariance = (series * index).rolling(length).mean() - (
        series.rolling(length).mean() * index.rolling(length).mean()
    )
    variance = index.rolling(length).var(ddof=0)
    return (covariance.abs() / variance / 2.0 * mult).to_numpy()


def pivots(high, low, length):
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    window = 2 * length + 1
    if n < window:
        return ph, pl
    rolling_max = pd.Series(high).rolling(window).max().to_numpy()
    rolling_min = pd.Series(low).rolling(window).min().to_numpy()
    for j in range(window - 1, n):
        center_high = high[j - length]
        if center_high == rolling_max[j]:
            ph[j] = center_high
        center_low = low[j - length]
        if center_low == rolling_min[j]:
            pl[j] = center_low
    return ph, pl


def run_engine(high, low, close, length, mult, method):
    n = len(close)
    slope = np.nan_to_num(compute_slope(high, low, close, length, mult, method), nan=0.0)
    ph, pl = pivots(high, low, length)
    upper = 0.0
    lower = 0.0
    slope_ph = 0.0
    slope_pl = 0.0
    upos = 0
    dnos = 0
    seen_ph = False
    seen_pl = False
    line_up = np.full(n, np.nan)
    line_dn = np.full(n, np.nan)
    slope_up = np.zeros(n)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    for i in range(n):
        has_ph = not np.isnan(ph[i])
        has_pl = not np.isnan(pl[i])
        if has_ph:
            slope_ph = slope[i]
            upper = ph[i]
            seen_ph = True
        else:
            upper = upper - slope_ph
        if has_pl:
            slope_pl = slope[i]
            lower = pl[i]
            seen_pl = True
        else:
            lower = lower + slope_pl
        prev_up = upos
        prev_dn = dnos
        if has_ph:
            upos = 0
        elif seen_ph and close[i] > upper - slope_ph * length:
            upos = 1
        if has_pl:
            dnos = 0
        elif seen_pl and close[i] < lower + slope_pl * length:
            dnos = 1
        bull[i] = upos > prev_up
        bear[i] = dnos > prev_dn
        slope_up[i] = slope_ph
        if seen_ph:
            line_up[i] = upper - slope_ph * length
        if seen_pl:
            line_dn[i] = lower + slope_pl * length
    return {
        "bull": bull,
        "bear": bear,
        "line_up": line_up,
        "line_dn": line_dn,
        "slope_up": slope_up,
        "ph_mask": ~np.isnan(ph),
        "pl_mask": ~np.isnan(pl),
    }


def position_state(bull, bear):
    n = len(bull)
    pos = np.zeros(n, dtype=int)
    state = 0
    last_buy = -1
    last_sell = -1
    for i in range(n):
        if bull[i]:
            state = 1
            last_buy = i
        if bear[i]:
            state = 0
            last_sell = i
        pos[i] = state
    return pos, last_buy, last_sell


def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def adx_wilder(high, low, close, period=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )
    alpha = 1.0 / period
    atr = true_range(high, low, close).ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()


def ema_status(close, ema20, ema50, ema200):
    if close > ema20 > ema50 > ema200:
        return "Güçlü Yükseliş"
    if close > ema50 and ema20 > ema50:
        return "Yükseliş"
    if close > ema50:
        return "Toparlanma"
    if close > ema200:
        return "Zayıf"
    return "Düşüş"


def breakout_score(in_long, buy_ago, rvol, adx, rsi, close, ema20, ema50, ema200, macd_hist):
    score = 0.0
    if in_long and buy_ago < BIG_BARS:
        score += max(0.0, 25.0 - 5.0 * buy_ago)
    score += min(max(rvol, 0.0), 3.0) / 3.0 * 20.0
    score += min(max(adx, 0.0), 40.0) / 40.0 * 15.0
    if 50.0 <= rsi <= 70.0:
        score += 15.0
    elif 45.0 <= rsi < 50.0 or 70.0 < rsi <= 75.0:
        score += 8.0
    if close > ema50:
        score += 10.0
    if ema20 > ema50:
        score += 5.0
    if ema50 > ema200:
        score += 5.0
    if macd_hist > 0:
        score += 5.0
    return round(score, 1)


def analyze_frame(df, length, mult, method):
    if df is None or len(df) < max(2 * length + 40, 80):
        return None
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    engine = run_engine(high, low, close, length, mult, method)
    pos, last_buy, last_sell = position_state(engine["bull"], engine["bear"])
    i = len(df) - 1
    close_s = df["Close"]
    last_close = float(close[i])

    ema20 = nz(close_s.ewm(span=20, adjust=False).mean().iloc[-1], np.nan)
    ema50 = nz(close_s.ewm(span=50, adjust=False).mean().iloc[-1], np.nan)
    ema200 = nz(close_s.ewm(span=200, adjust=False).mean().iloc[-1], np.nan)
    macd_line = close_s.ewm(span=12, adjust=False).mean() - close_s.ewm(span=26, adjust=False).mean()
    macd_hist = nz((macd_line - macd_line.ewm(span=9, adjust=False).mean()).iloc[-1], 0.0)
    rsi = nz(rsi_wilder(close_s).iloc[-1], 50.0)
    adx = nz(adx_wilder(df["High"], df["Low"], close_s).iloc[-1], 0.0)
    atr = true_range(df["High"], df["Low"], close_s).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    atr_pct = nz(atr.iloc[-1], 0.0) / last_close * 100.0 if last_close > 0 else 0.0
    volume = df["Volume"]
    prev_mean = float(volume.iloc[-21:-1].mean()) if len(volume) > 21 else 0.0
    rvol = float(volume.iloc[-1] / prev_mean) if prev_mean > 0 else 0.0

    in_long = bool(pos[i] == 1)
    buy_ago = int(i - last_buy) if last_buy >= 0 else BIG_BARS
    sell_ago = int(i - last_sell) if last_sell >= 0 else BIG_BARS
    entry_price = float(close[last_buy]) if last_buy >= 0 else np.nan
    pnl = (last_close / entry_price - 1.0) * 100.0 if in_long and not np.isnan(entry_price) else np.nan
    line_up_now = engine["line_up"][i]
    line_dn_now = engine["line_dn"][i]
    dist_up = (last_close / line_up_now - 1.0) * 100.0 if not np.isnan(line_up_now) and line_up_now > 0 else np.nan
    slope_pct = engine["slope_up"][i] / last_close * 100.0 if last_close > 0 else 0.0
    buy_time = df.index[last_buy].strftime("%Y-%m-%d %H:%M") if last_buy >= 0 else ""

    return {
        "bar_time": df.index[i].strftime("%Y-%m-%d %H:%M"),
        "close": last_close,
        "in_long": in_long,
        "buy_ago": buy_ago,
        "sell_ago": sell_ago,
        "buy_time": buy_time,
        "entry_price": entry_price,
        "pnl": pnl,
        "line_up": float(line_up_now) if not np.isnan(line_up_now) else np.nan,
        "line_dn": float(line_dn_now) if not np.isnan(line_dn_now) else np.nan,
        "dist_up": dist_up,
        "slope_pct": slope_pct,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "adx": adx,
        "rvol": rvol,
        "atr_pct": atr_pct,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "ema_status": ema_status(last_close, ema20, ema50, ema200),
        "skor": breakout_score(in_long, buy_ago, rvol, adx, rsi, last_close, ema20, ema50, ema200, macd_hist),
    }


@st.cache_data(ttl=300, show_spinner=False, max_entries=64)
def scan_batch(tickers, tf, length, mult, method, closed_only, market):
    cfg = TF_CONFIG[tf]
    suffix = MARKETS[market]["suffix"]
    symbols = tuple(t + suffix for t in tickers)
    frames = download_batch(symbols, cfg["interval"], cfg["period"])
    rows = []
    for ticker in tickers:
        raw = frames.get(ticker + suffix)
        if raw is None or raw.empty:
            continue
        try:
            df = prepare_frame(raw, tf, closed_only, market)
            result = analyze_frame(df, length, mult, method)
        except Exception:
            continue
        if result is None:
            continue
        result["ticker"] = ticker
        result["tf"] = tf
        rows.append(result)
    return rows


def run_scan(params):
    market = params["market"]
    universe, error = fetch_universe(
        market, params["limit"], params["min_cap"] * 1_000_000, params["min_price"], params["min_volume"]
    )
    if universe.empty:
        st.error(f"Hisse listesi alınamadı. {error}")
        return None
    tickers = universe["ticker"].tolist()
    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    progress = st.progress(0.0, text="Tarama başlıyor...")
    rows = []
    failures = 0
    for index, batch in enumerate(batches):
        for tf in params["tfs"]:
            try:
                rows.extend(
                    scan_batch(
                        tuple(batch), tf, params["length"], params["mult"],
                        params["method"], params["closed_only"], market,
                    )
                )
            except Exception:
                failures += 1
        progress.progress(
            (index + 1) / len(batches),
            text=f"Taranıyor: {min((index + 1) * BATCH_SIZE, len(tickers))}/{len(tickers)} hisse",
        )
    progress.empty()
    if not rows:
        st.error("Hiçbir hisse için fiyat verisi alınamadı. Lütfen birkaç dakika sonra tekrar deneyin.")
        return None
    raw_df = pd.DataFrame(rows)
    mkt = MARKETS[market]
    return {
        "raw": raw_df,
        "uni": universe,
        "params": dict(params),
        "failures": failures,
        "time": pd.Timestamp.now(tz=mkt["tz"]).strftime("%Y-%m-%d %H:%M:%S ") + mkt["tz_label"],
    }


def marks_store():
    if "marks" not in st.session_state:
        st.session_state["marks"] = {}
    return st.session_state["marks"]


def mark_key(market, ticker):
    return f"{market}:{ticker}"


def get_mark_emoji(market, ticker):
    mark = marks_store().get(mark_key(market, ticker))
    return MARK_OPTIONS.get(mark, NO_MARK)


def apply_marks(market, tickers, mark):
    store = marks_store()
    for ticker in tickers:
        if mark is None:
            store.pop(mark_key(market, ticker), None)
        else:
            store[mark_key(market, ticker)] = mark


def get_nonce(key):
    return st.session_state.get("sel_nonce", {}).get(key, 0)


def bump_nonce(key):
    nonces = st.session_state.setdefault("sel_nonce", {})
    nonces[key] = nonces.get(key, 0) + 1


def render_marks_sidebar():
    store = marks_store()
    with st.sidebar.expander("6) 🏷️ İşaretler", expanded=False):
        summary = "  ".join(
            f"{emoji} {sum(1 for v in store.values() if v == name)}" for name, emoji in MARK_OPTIONS.items()
        )
        st.write(summary)
        rows = [
            {"market": k.split(":", 1)[0], "ticker": k.split(":", 1)[1], "mark": v}
            for k, v in store.items()
        ]
        export = pd.DataFrame(rows, columns=["market", "ticker", "mark"])
        st.download_button(
            "⬇️ İşaretleri indir (CSV)",
            export.to_csv(index=False).encode("utf-8-sig"),
            "isaretler.csv",
            "text/csv",
            key="marks_download",
            use_container_width=True,
        )
        uploaded = st.file_uploader("İşaretleri içe aktar (CSV)", type=["csv"], key="marks_upload")
        if uploaded is not None:
            signature = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("marks_signature") != signature:
                try:
                    data = pd.read_csv(uploaded)
                    for _, item in data.iterrows():
                        market_code = str(item["market"])
                        mark_name = str(item["mark"])
                        if market_code in MARKETS and mark_name in MARK_OPTIONS:
                            store[mark_key(market_code, str(item["ticker"]).upper())] = mark_name
                    st.session_state["marks_signature"] = signature
                    st.toast("İşaretler içe aktarıldı.")
                except Exception as exc:
                    st.error(f"CSV okunamadı: {exc}")
        if st.button("🗑️ Tüm işaretleri temizle", key="marks_clear", use_container_width=True):
            store.clear()
            st.rerun()
        st.caption("İşaretler oturum boyunca saklanır. Kalıcı olması için CSV olarak indirin.")


def sidebar_controls():
    scan = st.session_state.get("scan")
    st.sidebar.title("📈 Breakout Screener")
    market = st.sidebar.radio(
        "Piyasa",
        list(MARKETS.keys()),
        format_func=lambda k: MARKETS[k]["label"],
    )
    mkt = MARKETS[market]
    defaults = mkt["defaults"]
    st.sidebar.caption("LuxAlgo Trendlines with Breaks (Long Only)")

    with st.sidebar.expander("1) Evren ve Veri", expanded=True):
        limit = st.slider(
            "Taranacak hisse sayısı (piyasa değerine göre)", 100, 1500, defaults["limit"], 50,
            key=f"limit_{market}",
        )
        min_cap = st.number_input(
            f"Min. piyasa değeri (Milyon {mkt['currency']})", 0, 10_000_000, defaults["min_cap"], 100,
            key=f"min_cap_{market}",
        )
        min_price = st.number_input(
            f"Min. fiyat ({mkt['currency']})", 0.0, 100000.0, float(defaults["min_price"]), 0.5,
            key=f"min_price_{market}",
        )
        min_volume = st.number_input(
            "Min. günlük hacim (adet)", 0, 1_000_000_000, defaults["min_volume"], 50000,
            key=f"min_volume_{market}",
        )
        tfs = st.multiselect(
            "Zaman dilimleri",
            TF_ORDER,
            default=TF_ORDER,
            format_func=lambda tf: TF_CONFIG[tf]["label"],
        )

    with st.sidebar.expander("2) Pine Script Parametreleri", expanded=False):
        length = st.number_input("Swing Detection Lookback", 2, 60, 14, 1)
        mult = st.number_input("Slope", 0.0, 10.0, 1.0, 0.1)
        method = st.selectbox("Slope Calculation Method", ["Atr", "Stdev", "Linreg"])
        closed_only = st.checkbox("Yalnızca kapanmış barlar", value=True)

    run = st.sidebar.button("🚀 Taramayı Başlat", type="primary", use_container_width=True)
    clear = st.sidebar.button("🔄 Önbelleği Temizle", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sonuç Filtreleri")

    with st.sidebar.expander("3) Sinyal", expanded=True):
        mode = st.radio("Sinyal modu", [MODE_FRESH, MODE_ACTIVE])
        lookback = st.slider("Taze sinyal: son kaç bar", 0, 10, 2, 1, disabled=(mode != MODE_FRESH))
        if len(tfs) > 1:
            min_tf = st.slider("Min. uyumlu zaman dilimi sayısı", 1, len(tfs), 1, 1)
        else:
            min_tf = 1

    with st.sidebar.expander("4) Teknik", expanded=False):
        rsi_range = st.slider("RSI aralığı", 0, 100, (0, 100))
        min_rvol = st.slider("Min. göreceli hacim (RVOL)", 0.0, 5.0, 0.0, 0.1)
        min_adx = st.slider("Min. ADX", 0, 50, 0, 1)
        above_ema50 = st.checkbox("Fiyat EMA 50 üzerinde")
        min_score = st.slider("Min. Breakout Skoru", 0, 100, 0, 5)

    sectors = []
    if scan is not None:
        sectors = sorted(scan["uni"]["sector"].dropna().unique().tolist())
    with st.sidebar.expander("5) Temel", expanded=False):
        picked_sectors = st.multiselect("Sektör", sectors)
        pe_max = st.number_input("Maks. F/K (0 = kapalı)", 0.0, 1000.0, 0.0, 5.0)
        pos_eps = st.checkbox("Yalnızca pozitif HBK")
        max_dist52 = st.slider("52H zirvesine maks. uzaklık % (100 = kapalı)", 5, 100, 100, 5)

    render_marks_sidebar()

    params = {
        "market": market,
        "limit": int(limit),
        "min_cap": int(min_cap),
        "min_price": float(min_price),
        "min_volume": int(min_volume),
        "tfs": [tf for tf in TF_ORDER if tf in tfs],
        "length": int(length),
        "mult": float(mult),
        "method": method,
        "closed_only": bool(closed_only),
    }
    flt = {
        "mode": mode,
        "lookback": int(lookback),
        "min_tf": int(min_tf),
        "rsi": rsi_range,
        "min_rvol": float(min_rvol),
        "min_adx": float(min_adx),
        "above_ema50": bool(above_ema50),
        "min_score": float(min_score),
        "sectors": picked_sectors,
        "pe_max": float(pe_max),
        "pos_eps": bool(pos_eps),
        "max_dist52": int(max_dist52),
    }
    return params, flt, run, clear


def compute_views(scan, flt):
    tfs = scan["params"]["tfs"]
    fund = scan["uni"].set_index("ticker")
    df = scan["raw"]
    df = df[df["tf"].isin(tfs)].copy()
    if df.empty:
        return None
    if flt["mode"] == MODE_FRESH:
        signal = df["in_long"] & (df["buy_ago"] <= flt["lookback"])
    else:
        signal = df["in_long"]
    tech = (
        df["rsi"].between(flt["rsi"][0], flt["rsi"][1])
        & (df["rvol"] >= flt["min_rvol"])
        & (df["adx"] >= flt["min_adx"])
        & (df["skor"] >= flt["min_score"])
    )
    if flt["above_ema50"]:
        tech = tech & (df["close"] > df["ema50"])
    df["signal"] = signal
    df["match"] = signal & tech

    fund_ok = pd.Series(True, index=fund.index)
    if flt["sectors"]:
        fund_ok &= fund["sector"].isin(flt["sectors"])
    if flt["pe_max"] > 0 and "price_earnings_ttm" in fund.columns:
        fund_ok &= fund["price_earnings_ttm"].between(0, flt["pe_max"])
    if flt["pos_eps"] and "earnings_per_share_diluted_ttm" in fund.columns:
        fund_ok &= fund["earnings_per_share_diluted_ttm"] > 0
    if flt["max_dist52"] < 100 and "price_52_week_high" in fund.columns:
        distance = (fund["close"] / fund["price_52_week_high"] - 1.0) * 100.0
        fund_ok &= distance >= -float(flt["max_dist52"])
    df = df[df["ticker"].isin(fund_ok[fund_ok].index)]

    counts = df[df["match"]].groupby("ticker").size()
    min_tf = min(flt["min_tf"], len(tfs))
    keep = counts[counts >= min_tf].index
    matched = df[df["match"] & df["ticker"].isin(keep)].copy()
    context = df[df["ticker"].isin(keep)].copy()
    summary = build_summary(context, fund, tfs)
    return {"matched": matched, "context": context, "summary": summary, "fund": fund}


def recommend_label(value):
    number = nz(value, np.nan)
    if np.isnan(number):
        return "—"
    if number >= 0.5:
        return "Güçlü Al"
    if number >= 0.1:
        return "Al"
    if number > -0.1:
        return "Nötr"
    if number > -0.5:
        return "Sat"
    return "Güçlü Sat"


def build_summary(context, fund, tfs):
    rows = []
    for ticker, group in context.groupby("ticker"):
        info = fund.loc[ticker] if ticker in fund.index else None
        row = {
            "Hisse": ticker,
            "Şirket": str(info["description"]) if info is not None else ticker,
            "Sektör": str(info["sector"]) if info is not None else "Diğer",
        }
        best = None
        matches = 0
        for tf in tfs:
            part = group[group["tf"] == tf]
            if part.empty:
                row[tf] = "Veri yok"
                continue
            item = part.iloc[0]
            if item["match"]:
                matches += 1
                row[tf] = "🟢 Yeni" if item["buy_ago"] == 0 else f"🟢 {int(item['buy_ago'])} bar önce"
                if best is None or item["skor"] > best["skor"]:
                    best = item
            elif item["in_long"]:
                row[tf] = "⚪ Long (filtre dışı)"
            else:
                row[tf] = "—"
        if best is None:
            continue
        row["Uyum"] = matches
        row["Skor"] = float(best["skor"])
        row["En İyi TF"] = TF_CONFIG[best["tf"]]["label"]
        row["Fiyat"] = float(best["close"])
        row["RSI"] = float(best["rsi"])
        row["RVOL"] = float(best["rvol"])
        row["ADX"] = float(best["adx"])
        row["Çizgiye Mesafe %"] = nz(best["dist_up"], np.nan)
        if info is not None:
            cap = nz(info.get("market_cap_basic"), np.nan)
            row["Piyasa Değeri (Mlr)"] = cap / 1e9 if not np.isnan(cap) else np.nan
            row["F/K"] = nz(info.get("price_earnings_ttm"), np.nan)
            row["1A Perf %"] = nz(info.get("Perf.1M"), np.nan)
            high52 = nz(info.get("price_52_week_high"), np.nan)
            close_tv = nz(info.get("close"), np.nan)
            row["52H Uzaklık %"] = (close_tv / high52 - 1.0) * 100.0 if high52 > 0 and not np.isnan(close_tv) else np.nan
            row["Analist"] = recommend_label(info.get("Recommend.All"))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["Uyum", "Skor"], ascending=[False, False]).reset_index(drop=True)


def build_detail_table(matched):
    if matched.empty:
        return pd.DataFrame()
    table = pd.DataFrame(
        {
            "Hisse": matched["ticker"],
            "Zaman Dilimi": matched["tf"].map(lambda tf: TF_CONFIG[tf]["label"]),
            "Son Bar": matched["bar_time"],
            "Fiyat": matched["close"],
            "Buy Zamanı": matched["buy_time"],
            "Kaç Bar Önce": matched["buy_ago"],
            "Sinyal Getirisi %": matched["pnl"],
            "Direnç Çizgisi": matched["line_up"],
            "Çizgiye Mesafe %": matched["dist_up"],
            "Eğim %/bar": matched["slope_pct"],
            "RSI": matched["rsi"],
            "MACD Hist": matched["macd_hist"],
            "ADX": matched["adx"],
            "RVOL": matched["rvol"],
            "ATR %": matched["atr_pct"],
            "EMA Durumu": matched["ema_status"],
            "Skor": matched["skor"],
        }
    )
    return table.sort_values("Skor", ascending=False).reset_index(drop=True)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def row_style(row):
    mark = EMOJI_TO_MARK.get(row["İşaret"])
    css = MARK_CSS.get(mark, "")
    return [css] * len(row)


def table_filters(df, key):
    filtered = df
    with st.expander("🔎 Tablo filtreleri ve sıralama", expanded=False):
        search = st.text_input("Hisse / şirket ara", key=f"{key}_search")
        candidates = [c for c in df.columns if c not in ("Hisse", "Şirket")]
        signature = abs(hash(tuple(candidates))) % 10**8
        chosen = st.multiselect(
            "Filtre uygulanacak sütunlar (çoklu seçim)",
            candidates,
            key=f"{key}_cols_{signature}",
        )
        mask = pd.Series(True, index=df.index)
        if search.strip():
            term = search.strip().lower()
            match = df["Hisse"].astype(str).str.lower().str.contains(term, regex=False)
            if "Şirket" in df.columns:
                match = match | df["Şirket"].astype(str).str.lower().str.contains(term, regex=False)
            mask &= match
        for column in chosen:
            series = df[column]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                values = series.dropna()
                if values.empty:
                    continue
                low = float(values.min())
                high = float(values.max())
                if low == high:
                    st.caption(f"{column}: tüm satırlarda {low:g}")
                    continue
                widget_key = f"{key}_f_{column}_{round(low, 4)}_{round(high, 4)}"
                if pd.api.types.is_integer_dtype(series):
                    picked = st.slider(column, int(low), int(high), (int(low), int(high)), key=widget_key)
                else:
                    picked = st.slider(
                        column, low, high, (low, high), step=(high - low) / 200.0,
                        format="%.4g", key=widget_key,
                    )
                if picked[0] > low or picked[1] < high:
                    mask &= series.between(picked[0], picked[1])
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    continue
                option_sig = abs(hash(tuple(options))) % 10**8
                picked = st.multiselect(column, options, key=f"{key}_f_{column}_{option_sig}")
                if picked:
                    mask &= series.astype(str).isin(picked)
        filtered = df[mask]
        sort_col_key = f"{key}_sort_{signature}"
        left, right = st.columns([2, 1])
        sort_col = left.selectbox("Sırala", ["(varsayılan sıra)"] + list(df.columns), key=sort_col_key)
        direction = right.radio("Yön", ["Azalan", "Artan"], horizontal=True, key=f"{key}_dir_{signature}")
        if sort_col != "(varsayılan sıra)":
            filtered = filtered.sort_values(sort_col, ascending=(direction == "Artan"), na_position="last")
    return filtered


def render_marked_table(df, key, market, column_config, ticker_col="Hisse", file_name="tablo.csv"):
    if df is None or df.empty:
        st.info("Gösterilecek satır yok.")
        return
    work = df.copy().reset_index(drop=True)
    work = work.replace([np.inf, -np.inf], np.nan)
    work.insert(0, "İşaret", [get_mark_emoji(market, t) for t in work[ticker_col]])
    filtered = table_filters(work, key).reset_index(drop=True)
    st.caption(
        f"Gösterilen satır: {len(filtered)} / {len(work)}   ·   🟢 Yeşil   🟡 Sarı   🔴 Kırmızı   "
        "(satırların solundaki kutularla seçip aşağıdaki düğmelerle işaretleyin)"
    )
    if filtered.empty:
        st.info("Filtrelerle eşleşen satır yok.")
        return
    config = dict(column_config)
    config["İşaret"] = st.column_config.TextColumn("İşaret", width="small")
    row_signature = abs(hash(tuple(filtered[ticker_col].astype(str).tolist()))) % 10**8
    event = st.dataframe(
        filtered.style.apply(row_style, axis=1),
        column_config=config,
        hide_index=True,
        use_container_width=True,
        height=min(720, 60 + 36 * len(filtered)),
        on_select="rerun",
        selection_mode="multi-row",
        key=f"{key}_df_{get_nonce(key)}_{row_signature}",
    )
    try:
        selected_rows = list(event.selection.rows)
    except Exception:
        selected_rows = []
    scope = st.radio(
        "İşaret kapsamı",
        ["Seçili satırlar", "Tablodaki tüm satırlar"],
        horizontal=True,
        key=f"{key}_scope",
    )
    actions = [
        ("🟢 Yeşil", "green"),
        ("🟡 Sarı", "yellow"),
        ("🔴 Kırmızı", "red"),
        ("⚪ İşareti kaldır", None),
    ]
    columns = st.columns(len(actions) + 1)
    for column, (label, mark) in zip(columns, actions):
        if column.button(label, key=f"{key}_btn_{mark}", use_container_width=True):
            if scope == "Seçili satırlar":
                tickers = filtered.iloc[selected_rows][ticker_col].astype(str).tolist() if selected_rows else []
            else:
                tickers = filtered[ticker_col].astype(str).tolist()
            if not tickers:
                st.warning("Önce tabloda en az bir satır seçin.")
            else:
                apply_marks(market, tickers, mark)
                bump_nonce(key)
                st.rerun()
    columns[-1].download_button(
        "⬇️ CSV",
        csv_bytes(filtered),
        file_name,
        "text/csv",
        key=f"{key}_download",
        use_container_width=True,
    )


def render_summary_tab(views, tfs, market):
    summary = views["summary"]
    if summary.empty:
        st.info("Seçili filtrelerle eşleşen hisse bulunamadı. Filtreleri gevşetmeyi deneyin.")
        return
    config = {
        "Hisse": st.column_config.TextColumn("Hisse", width="small"),
        "Uyum": st.column_config.NumberColumn("Uyum (TF)", format="%d"),
        "Skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f"),
        "Fiyat": st.column_config.NumberColumn(format="%.2f"),
        "RSI": st.column_config.NumberColumn(format="%.1f"),
        "RVOL": st.column_config.NumberColumn(format="%.2f"),
        "ADX": st.column_config.NumberColumn(format="%.1f"),
        "Çizgiye Mesafe %": st.column_config.NumberColumn(format="%.2f"),
        "Piyasa Değeri (Mlr)": st.column_config.NumberColumn(format="%.1f"),
        "F/K": st.column_config.NumberColumn(format="%.1f"),
        "1A Perf %": st.column_config.NumberColumn(format="%.1f"),
        "52H Uzaklık %": st.column_config.NumberColumn(format="%.1f"),
    }
    first = ["Hisse", "Şirket", "Sektör", "Uyum"] + list(tfs)
    rest = [c for c in summary.columns if c not in first]
    render_marked_table(summary[first + rest], "tbl_summary", market, config, file_name="breakout_ozet.csv")


def render_detail_table_tab(views, market):
    table = build_detail_table(views["matched"])
    if table.empty:
        st.info("Eşleşen sinyal bulunamadı.")
        return
    config = {
        "Skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f"),
        "Fiyat": st.column_config.NumberColumn(format="%.2f"),
        "Direnç Çizgisi": st.column_config.NumberColumn(format="%.2f"),
        "Sinyal Getirisi %": st.column_config.NumberColumn(format="%.2f"),
        "Çizgiye Mesafe %": st.column_config.NumberColumn(format="%.2f"),
        "Eğim %/bar": st.column_config.NumberColumn(format="%.3f"),
        "RSI": st.column_config.NumberColumn(format="%.1f"),
        "MACD Hist": st.column_config.NumberColumn(format="%.3f"),
        "ADX": st.column_config.NumberColumn(format="%.1f"),
        "RVOL": st.column_config.NumberColumn(format="%.2f"),
        "ATR %": st.column_config.NumberColumn(format="%.2f"),
    }
    render_marked_table(table, "tbl_detail", market, config, file_name="breakout_detay.csv")


def style_figure(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def render_visual_tab(views, tfs):
    summary = views["summary"]
    matched = views["matched"]
    if summary.empty:
        st.info("Görselleştirme için eşleşen hisse yok.")
        return
    left, right = st.columns(2)
    top = summary.head(20).iloc[::-1]
    fig = px.bar(
        top, x="Skor", y="Hisse", orientation="h", color="Uyum",
        color_continuous_scale="Tealgrn", title="En Yüksek Breakout Skorları (Top 20)",
    )
    left.plotly_chart(style_figure(fig, 520), use_container_width=True)

    heat_source = summary.head(25)["Hisse"].tolist()
    pivot = (
        matched[matched["ticker"].isin(heat_source)]
        .pivot_table(index="ticker", columns="tf", values="skor", aggfunc="max")
        .reindex(index=heat_source, columns=[tf for tf in TF_ORDER if tf in tfs])
    )
    heat = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[TF_CONFIG[c]["label"] for c in pivot.columns],
            y=pivot.index.tolist(),
            colorscale="Tealgrn",
            zmin=0,
            zmax=100,
            text=np.round(pivot.values, 0),
            texttemplate="%{text}",
            hoverongaps=False,
            colorbar=dict(title="Skor"),
        )
    )
    heat.update_layout(title="Zaman Dilimi Isı Haritası (Top 25)", yaxis=dict(autorange="reversed"))
    right.plotly_chart(style_figure(heat, 520), use_container_width=True)

    left2, right2 = st.columns(2)
    scatter_data = matched.copy()
    scatter_data["Zaman Dilimi"] = scatter_data["tf"].map(lambda tf: TF_CONFIG[tf]["label"])
    scatter_data["size"] = scatter_data["skor"].clip(lower=5)
    scatter = px.scatter(
        scatter_data, x="rsi", y="rvol", size="size", color="Zaman Dilimi",
        hover_name="ticker", hover_data={"skor": True, "adx": ":.1f", "size": False},
        title="RSI – Göreceli Hacim (balon = skor)",
        labels={"rsi": "RSI", "rvol": "RVOL"},
    )
    scatter.add_vline(x=50, line_dash="dot", line_color="gray")
    scatter.add_vline(x=70, line_dash="dot", line_color="gray")
    left2.plotly_chart(style_figure(scatter, 420), use_container_width=True)

    sector_counts = summary["Sektör"].value_counts().reset_index()
    sector_counts.columns = ["Sektör", "Hisse Sayısı"]
    sector_fig = px.bar(
        sector_counts.iloc[::-1], x="Hisse Sayısı", y="Sektör", orientation="h",
        title="Sektör Dağılımı", color="Hisse Sayısı", color_continuous_scale="Blues",
    )
    right2.plotly_chart(style_figure(sector_fig, 420), use_container_width=True)

    left3, right3 = st.columns(2)
    hist = px.histogram(
        matched, x="skor", nbins=20, color=matched["tf"].map(lambda tf: TF_CONFIG[tf]["label"]),
        title="Skor Dağılımı", labels={"skor": "Skor", "color": "Zaman Dilimi"},
    )
    left3.plotly_chart(style_figure(hist, 380), use_container_width=True)
    dist_data = matched.dropna(subset=["dist_up"]).copy()
    dist_data["Zaman Dilimi"] = dist_data["tf"].map(lambda tf: TF_CONFIG[tf]["label"])
    dist_fig = px.scatter(
        dist_data, x="dist_up", y="adx", color="Zaman Dilimi", hover_name="ticker",
        title="Direnç Çizgisi Üstü Mesafe % – ADX",
        labels={"dist_up": "Çizgiye Mesafe %", "adx": "ADX"},
    )
    right3.plotly_chart(style_figure(dist_fig, 380), use_container_width=True)


def render_fundamental_tab(views, market):
    summary = views["summary"]
    fund = views["fund"]
    if summary.empty:
        st.info("Temel analiz için eşleşen hisse yok.")
        return
    tickers = summary["Hisse"].tolist()
    available = [c for c in FUND_LABELS if c in fund.columns]
    table = fund.loc[fund.index.isin(tickers), available].copy()
    if "Recommend.All" in fund.columns:
        table["Analist"] = fund.loc[table.index, "Recommend.All"].map(recommend_label)
    else:
        table["Analist"] = "—"
    if "market_cap_basic" in table.columns:
        table["market_cap_basic"] = table["market_cap_basic"] / 1e9
    table = table.rename(columns=FUND_LABELS).rename(columns={"Piyasa Değeri": "Piyasa Değeri (Mlr)"})
    table.insert(0, "Sektör", fund.loc[table.index, "sector"])
    table.insert(0, "Hisse", table.index)
    table["Skor"] = table["Hisse"].map(summary.set_index("Hisse")["Skor"])
    table = table.sort_values("Skor", ascending=False).reset_index(drop=True)
    render_marked_table(
        table,
        "tbl_fund",
        market,
        {"Skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.0f")},
        file_name="breakout_temel.csv",
    )
    left, right = st.columns(2)
    cap_col = "Piyasa Değeri (Mlr)"
    if cap_col in table.columns:
        cap_data = table.dropna(subset=[cap_col])
        cap_data = cap_data[cap_data[cap_col] > 0]
        if not cap_data.empty:
            cap_fig = px.scatter(
                cap_data, x=cap_col, y="Skor", color="Sektör",
                hover_name="Hisse", log_x=True, title="Piyasa Değeri – Skor",
            )
            left.plotly_chart(style_figure(cap_fig, 400), use_container_width=True)
    if "F/K" in table.columns:
        pe_data = table[(table["F/K"] > 0) & (table["F/K"] < 100)]
        if not pe_data.empty:
            pe_fig = px.histogram(pe_data, x="F/K", nbins=25, title="F/K Dağılımı (0–100)")
            right.plotly_chart(style_figure(pe_fig, 400), use_container_width=True)
    if "1A Perf %" in table.columns and "3A Perf %" in table.columns:
        perf = table.dropna(subset=["1A Perf %", "3A Perf %"])
        if not perf.empty:
            perf_fig = px.scatter(
                perf, x="1A Perf %", y="3A Perf %", color="Sektör", size=perf["Skor"].clip(lower=5),
                hover_name="Hisse", title="Momentum: 1 Aylık – 3 Aylık Performans",
            )
            perf_fig.add_hline(y=0, line_dash="dot", line_color="gray")
            perf_fig.add_vline(x=0, line_dash="dot", line_color="gray")
            st.plotly_chart(style_figure(perf_fig, 420), use_container_width=True)


def market_rangebreaks(tf, mkt):
    if tf in ("1h", "4h"):
        return [
            dict(bounds=["sat", "mon"]),
            dict(bounds=[mkt["close_min"] / 60.0, mkt["open_min"] / 60.0], pattern="hour"),
        ]
    if tf == "1d":
        return [dict(bounds=["sat", "mon"])]
    return []


def build_chart(df, engine, tf, ticker, view_bars, mkt):
    n = len(df)
    start = max(0, n - view_bars)
    view = df.iloc[start:]
    x = view.index
    line_up = engine["line_up"].copy()
    line_dn = engine["line_dn"].copy()
    line_up[engine["ph_mask"]] = np.nan
    line_dn[engine["pl_mask"]] = np.nan
    line_up = line_up[start:]
    line_dn = line_dn[start:]
    bull = engine["bull"][start:]
    bear = engine["bear"][start:]
    close_s = df["Close"]
    ema20 = close_s.ewm(span=20, adjust=False).mean().iloc[start:]
    ema50 = close_s.ewm(span=50, adjust=False).mean().iloc[start:]
    rsi = rsi_wilder(close_s).iloc[start:]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.62, 0.18, 0.20]
    )
    fig.add_trace(
        go.Candlestick(
            x=x, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
            name="Fiyat", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=line_up, mode="lines", name="Üst trend çizgisi (direnç)",
                   line=dict(color="#2dd4bf", width=2, dash="dash"), connectgaps=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=line_dn, mode="lines", name="Alt trend çizgisi (destek)",
                   line=dict(color="#f87171", width=2, dash="dash"), connectgaps=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=ema20, mode="lines", name="EMA 20", line=dict(color="#fbbf24", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=ema50, mode="lines", name="EMA 50", line=dict(color="#60a5fa", width=1)),
        row=1, col=1,
    )
    lows = view["Low"].to_numpy()
    highs = view["High"].to_numpy()
    if bull.any():
        fig.add_trace(
            go.Scatter(x=x[bull], y=lows[bull] * 0.992, mode="markers+text", text=["B"] * int(bull.sum()),
                       textposition="bottom center", name="Buy (B)",
                       marker=dict(symbol="triangle-up", size=12, color="#14b8a6")),
            row=1, col=1,
        )
    if bear.any():
        fig.add_trace(
            go.Scatter(x=x[bear], y=highs[bear] * 1.008, mode="markers+text", text=["S"] * int(bear.sum()),
                       textposition="top center", name="Sell (S)",
                       marker=dict(symbol="triangle-down", size=12, color="#ef4444")),
            row=1, col=1,
        )
    colors = np.where(view["Close"].to_numpy() >= view["Open"].to_numpy(), "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x, y=view["Volume"], marker_color=colors, name="Hacim", showlegend=False), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=x, y=rsi, mode="lines", name="RSI 14", line=dict(color="#c084fc", width=1.5)),
        row=3, col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=3, col=1)
    breaks = market_rangebreaks(tf, mkt)
    if breaks:
        fig.update_xaxes(rangebreaks=breaks)
    fig.update_layout(
        template="plotly_dark",
        height=780,
        xaxis_rangeslider_visible=False,
        title=f"{ticker} — {TF_CONFIG[tf]['label']}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def render_detail_tab(scan, views):
    summary = views["summary"]
    if summary.empty:
        st.info("Detay için eşleşen hisse yok.")
        return
    params = scan["params"]
    market = params["market"]
    mkt = MARKETS[market]
    cur = mkt["currency"]
    col1, col2, col3 = st.columns([2, 1, 1])
    ticker = col1.selectbox(
        "Hisse", summary["Hisse"].tolist(),
        format_func=lambda t: f"{t} — {summary.loc[summary['Hisse'] == t, 'Şirket'].iloc[0]}",
    )
    tf = col2.selectbox("Zaman dilimi", params["tfs"], format_func=lambda x: TF_CONFIG[x]["label"])
    view_bars = col3.slider("Gösterilecek bar", 60, 400, 180, 10)

    raw = fetch_history(ticker, tf, market)
    if raw is None or raw.empty:
        st.warning("Bu hisse için fiyat verisi alınamadı.")
        return
    try:
        df = prepare_frame(raw, tf, params["closed_only"], market)
        result = analyze_frame(df, params["length"], params["mult"], params["method"])
        if result is None:
            st.warning("Yeterli bar sayısı yok.")
            return
        engine = run_engine(
            df["High"].to_numpy(dtype=float), df["Low"].to_numpy(dtype=float),
            df["Close"].to_numpy(dtype=float), params["length"], params["mult"], params["method"],
        )
        fig = build_chart(df, engine, tf, ticker, view_bars, mkt)
    except Exception as exc:
        st.error(f"Grafik oluşturulamadı: {exc}")
        return

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Son Fiyat", f"{cur}{result['close']:.2f}")
    m2.metric("Durum", "Long" if result["in_long"] else "Flat",
              f"{result['buy_ago']} bar önce Buy" if result["in_long"] and result["buy_ago"] < BIG_BARS else None)
    m3.metric("Skor", f"{result['skor']:.0f}/100")
    m4.metric("RSI", f"{result['rsi']:.1f}")
    m5.metric("ADX", f"{result['adx']:.1f}")
    m6.metric("RVOL", f"{result['rvol']:.2f}x")
    st.plotly_chart(fig, use_container_width=True)

    info = views["fund"].loc[ticker] if ticker in views["fund"].index else None
    if info is not None:
        st.markdown(f"**{info['description']}** · {info['sector']} · {info['industry']}")
        cards = [
            ("Piyasa Değeri", f"{cur}{nz(info.get('market_cap_basic'), np.nan) / 1e9:.1f} Mlr"),
            ("F/K", f"{nz(info.get('price_earnings_ttm'), np.nan):.1f}"),
            ("HBK", f"{cur}{nz(info.get('earnings_per_share_diluted_ttm'), np.nan):.2f}"),
            ("Temettü", f"{nz(info.get('dividends_yield_current'), 0.0):.2f}%"),
            ("Beta", f"{nz(info.get('beta_1_year'), np.nan):.2f}"),
            ("Analist", recommend_label(info.get("Recommend.All"))),
        ]
        columns = st.columns(len(cards))
        for column, (label, value) in zip(columns, cards):
            column.metric(label, value.replace("nan", "—"))

    extra = pd.DataFrame(
        {
            "Gösterge": [
                "Direnç çizgisi", "Çizgiye mesafe %", "Eğim %/bar", "Sinyal getirisi %",
                "MACD histogram", "ATR %", "EMA 20", "EMA 50", "EMA 200", "EMA durumu", "Buy zamanı",
            ],
            "Değer": [
                f"{result['line_up']:.2f}" if not np.isnan(result["line_up"]) else "—",
                f"{result['dist_up']:.2f}" if not np.isnan(result["dist_up"]) else "—",
                f"{result['slope_pct']:.3f}",
                f"{result['pnl']:.2f}" if not np.isnan(result["pnl"]) else "—",
                f"{result['macd_hist']:.3f}",
                f"{result['atr_pct']:.2f}",
                f"{result['ema20']:.2f}" if not np.isnan(result["ema20"]) else "—",
                f"{result['ema50']:.2f}" if not np.isnan(result["ema50"]) else "—",
                f"{result['ema200']:.2f}" if not np.isnan(result["ema200"]) else "—",
                result["ema_status"],
                result["buy_time"] or "—",
            ],
        }
    )
    st.dataframe(extra, hide_index=True, use_container_width=True)


def normalize_ticker(text, market):
    mkt = MARKETS[market]
    value = text.strip().upper()
    if mkt["suffix"] and value.endswith(mkt["suffix"]):
        value = value[: -len(mkt["suffix"])]
    if mkt["dot_to_dash"]:
        value = value.replace(".", "-")
    if not re.match(r"^[A-Z][A-Z0-9\-]{0,9}$", value):
        return ""
    return value


def run_backtest(df, engine, capital, commission_pct, slippage_pct, fractional):
    n = len(df)
    open_p = df["Open"].to_numpy(dtype=float)
    close_p = df["Close"].to_numpy(dtype=float)
    bull = engine["bull"]
    bear = engine["bear"]
    comm = commission_pct / 100.0
    slip = slippage_pct / 100.0
    cash = float(capital)
    shares = 0.0
    in_pos = False
    entry_i = -1
    entry_price = 0.0
    entry_fee = 0.0
    pending = 0
    equity = np.zeros(n)
    exposure = np.zeros(n, dtype=bool)
    trades = []
    for i in range(n):
        if pending == 1 and not in_pos:
            price = open_p[i] * (1.0 + slip)
            qty = cash / (price * (1.0 + comm)) if price > 0 else 0.0
            if not fractional:
                qty = float(np.floor(qty))
            if qty > 0:
                cost = qty * price
                fee = cost * comm
                cash -= cost + fee
                shares = qty
                in_pos = True
                entry_i = i
                entry_price = price
                entry_fee = fee
        elif pending == -1 and in_pos:
            price = open_p[i] * (1.0 - slip)
            proceeds = shares * price
            fee = proceeds * comm
            cash += proceeds - fee
            invested = shares * entry_price + entry_fee
            pnl = proceeds - fee - invested
            trades.append(
                {
                    "entry_i": entry_i, "exit_i": i, "entry_price": entry_price, "exit_price": price,
                    "shares": shares, "pnl": pnl, "ret": pnl / invested * 100.0 if invested > 0 else 0.0,
                    "bars": i - entry_i, "status": "Kapalı",
                }
            )
            shares = 0.0
            in_pos = False
        equity[i] = cash + shares * close_p[i]
        exposure[i] = in_pos
        if bear[i]:
            pending = -1
        elif bull[i]:
            pending = 1
        else:
            pending = 0
    if in_pos:
        mark = close_p[-1]
        invested = shares * entry_price + entry_fee
        pnl = shares * mark - invested
        trades.append(
            {
                "entry_i": entry_i, "exit_i": n - 1, "entry_price": entry_price, "exit_price": float(mark),
                "shares": shares, "pnl": pnl, "ret": pnl / invested * 100.0 if invested > 0 else 0.0,
                "bars": n - 1 - entry_i, "status": "Açık",
            }
        )
    return equity, exposure, trades, pending


def build_trades_table(df, trades, tf):
    fmt = "%Y-%m-%d %H:%M" if tf in ("1h", "4h") else "%Y-%m-%d"
    rows = []
    for number, trade in enumerate(trades, start=1):
        rows.append(
            {
                "No": number,
                "Giriş Zamanı": df.index[trade["entry_i"]].strftime(fmt),
                "Çıkış Zamanı": df.index[trade["exit_i"]].strftime(fmt) if trade["status"] == "Kapalı" else "—",
                "Giriş Fiyatı": trade["entry_price"],
                "Çıkış / Son Fiyat": trade["exit_price"],
                "Adet": trade["shares"],
                "Bar": trade["bars"],
                "Getiri %": trade["ret"],
                "K/Z": trade["pnl"],
                "Durum": trade["status"],
            }
        )
    return pd.DataFrame(rows)


def backtest_stats(df, equity, exposure, trades, capital):
    final = float(equity[-1])
    total_ret = (final / capital - 1.0) * 100.0
    bh_ret = (float(df["Close"].iloc[-1]) / float(df["Open"].iloc[0]) - 1.0) * 100.0
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min() * 100.0)
    closed = [t for t in trades if t["status"] == "Kapalı"]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = -sum(t["pnl"] for t in losses)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = float("nan")
    all_returns = [t["ret"] for t in trades]
    days = (df.index[-1] - df.index[0]).days
    cagr = ((final / capital) ** (365.0 / days) - 1.0) * 100.0 if days >= 60 and final > 0 else float("nan")
    return {
        "capital": float(capital),
        "final": final,
        "total_ret": total_ret,
        "bh_ret": bh_ret,
        "max_dd": max_dd,
        "trade_count": len(trades),
        "closed_count": len(closed),
        "win_rate": len(wins) / len(closed) * 100.0 if closed else float("nan"),
        "profit_factor": profit_factor,
        "avg_ret": float(np.mean(all_returns)) if all_returns else float("nan"),
        "best": float(max(all_returns)) if all_returns else float("nan"),
        "worst": float(min(all_returns)) if all_returns else float("nan"),
        "exposure": float(exposure.mean() * 100.0),
        "cagr": cagr,
    }


def execute_backtest(ticker, tf, period, market, params, capital, commission, slippage, fractional):
    try:
        raw = fetch_history_period(ticker, tf, market, period)
        if raw is None or raw.empty:
            return {"error": f"{ticker} için fiyat verisi alınamadı. Sembolü ve piyasa seçimini kontrol edin."}
        df = prepare_frame(raw, tf, params["closed_only"], market)
        if df is None or len(df) < max(2 * params["length"] + 40, 80):
            return {"error": "Backtest için yeterli bar yok. Daha uzun bir veri aralığı seçin."}
        engine = run_engine(
            df["High"].to_numpy(dtype=float), df["Low"].to_numpy(dtype=float),
            df["Close"].to_numpy(dtype=float), params["length"], params["mult"], params["method"],
        )
        equity, exposure, trades, pending = run_backtest(df, engine, capital, commission, slippage, fractional)
        stats = backtest_stats(df, equity, exposure, trades, capital)
        shares_bh = capital / float(df["Open"].iloc[0])
        bh_equity = shares_bh * df["Close"].to_numpy(dtype=float)
        fmt = "%Y-%m-%d %H:%M" if tf in ("1h", "4h") else "%Y-%m-%d"
        return {
            "ticker": ticker,
            "tf": tf,
            "market": market,
            "period": period,
            "df": df,
            "engine": engine,
            "equity": equity,
            "bh_equity": bh_equity,
            "trades": trades,
            "trades_table": build_trades_table(df, trades, tf),
            "stats": stats,
            "pending": pending,
            "first": df.index[0].strftime(fmt),
            "last": df.index[-1].strftime(fmt),
            "bars": len(df),
            "pine": (params["length"], params["mult"], params["method"]),
            "costs": (capital, commission, slippage, fractional),
        }
    except Exception as exc:
        return {"error": f"Backtest hesaplanamadı: {exc}"}


def build_backtest_chart(result):
    df = result["df"]
    trades = result["trades"]
    equity = result["equity"]
    bh_equity = result["bh_equity"]
    tf = result["tf"]
    mkt = MARKETS[result["market"]]
    x = df.index
    peak = np.maximum.accumulate(equity)
    drawdown = (equity / peak - 1.0) * 100.0
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.52, 0.30, 0.18]
    )
    fig.add_trace(
        go.Candlestick(
            x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Fiyat", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )
    if trades:
        entry_x = [x[t["entry_i"]] for t in trades]
        entry_y = [t["entry_price"] for t in trades]
        fig.add_trace(
            go.Scatter(x=entry_x, y=entry_y, mode="markers", name="Alım",
                       marker=dict(symbol="triangle-up", size=12, color="#14b8a6", line=dict(width=1, color="white"))),
            row=1, col=1,
        )
        closed = [t for t in trades if t["status"] == "Kapalı"]
        if closed:
            fig.add_trace(
                go.Scatter(
                    x=[x[t["exit_i"]] for t in closed], y=[t["exit_price"] for t in closed],
                    mode="markers", name="Satım",
                    marker=dict(symbol="triangle-down", size=12, color="#ef4444", line=dict(width=1, color="white")),
                    text=[f"{t['ret']:+.2f}%" for t in closed], hovertemplate="%{y:.2f} (%{text})<extra>Satım</extra>",
                ),
                row=1, col=1,
            )
    fig.add_trace(
        go.Scatter(x=x, y=equity, mode="lines", name="Strateji özsermaye", line=dict(color="#2dd4bf", width=2)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=bh_equity, mode="lines", name="Al-Tut", line=dict(color="#fbbf24", width=1.5, dash="dot")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=drawdown, mode="lines", name="Düşüş %", fill="tozeroy",
                   line=dict(color="#f87171", width=1)),
        row=3, col=1,
    )
    breaks = market_rangebreaks(tf, mkt)
    if breaks:
        fig.update_xaxes(rangebreaks=breaks)
    fig.update_layout(
        template="plotly_dark",
        height=820,
        xaxis_rangeslider_visible=False,
        title=f"{result['ticker']} — {TF_CONFIG[tf]['label']} Backtest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def fmt_value(value, pattern="{:.2f}", suffix=""):
    number = nz(value, np.nan)
    if np.isnan(number):
        return "—"
    if np.isinf(number):
        return "∞"
    return pattern.format(number) + suffix


def render_backtest_result(result):
    if "error" in result:
        st.error(result["error"])
        return
    stats = result["stats"]
    cur = MARKETS[result["market"]]["currency"]
    capital, commission, slippage, fractional = result["costs"]
    length, mult, method = result["pine"]
    st.markdown(
        f"**{result['ticker']}** · {TF_CONFIG[result['tf']]['label']} · veri: {result['first']} → {result['last']} "
        f"({result['bars']} bar, istek: {result['period']})"
    )
    st.caption(
        f"Pine: Swing={length}, Slope={mult}, Yöntem={method} · Sermaye {cur}{capital:,.0f} · "
        f"Komisyon %{commission} · Kayma %{slippage} · {'Kesirli hisse' if fractional else 'Tam hisse'}"
    )
    row1 = st.columns(4)
    row1[0].metric(
        "Strateji Getirisi", f"{stats['total_ret']:.2f}%",
        f"{stats['total_ret'] - stats['bh_ret']:+.2f} puan (Al-Tut'a göre)",
    )
    row1[1].metric("Al-Tut Getirisi", f"{stats['bh_ret']:.2f}%")
    row1[2].metric(
        "Son Sermaye", f"{cur}{stats['final']:,.2f}", f"{stats['final'] - stats['capital']:+,.2f}",
    )
    row1[3].metric("Maks. Düşüş", f"{stats['max_dd']:.2f}%")
    row2 = st.columns(4)
    row2[0].metric("İşlem Sayısı", f"{stats['trade_count']}")
    row2[1].metric("Kazanma Oranı", fmt_value(stats["win_rate"], "{:.1f}", "%"))
    row2[2].metric("Profit Factor", fmt_value(stats["profit_factor"]))
    row2[3].metric("Ort. İşlem Getirisi", fmt_value(stats["avg_ret"], "{:.2f}", "%"))
    row3 = st.columns(4)
    row3[0].metric("En İyi İşlem", fmt_value(stats["best"], "{:.2f}", "%"))
    row3[1].metric("En Kötü İşlem", fmt_value(stats["worst"], "{:.2f}", "%"))
    row3[2].metric("Piyasada Kalma", f"{stats['exposure']:.1f}%")
    row3[3].metric("Yıllık Getiri (CAGR)", fmt_value(stats["cagr"], "{:.2f}", "%"))
    if result["pending"] == 1:
        st.info("Son bar **Buy** sinyali verdi. İşlem bir sonraki barın açılışında gerçekleşir.")
    elif result["pending"] == -1:
        st.info("Son bar **Sell** sinyali verdi. Açık pozisyon varsa bir sonraki barın açılışında kapanır.")
    if stats["trade_count"] == 0:
        st.warning("Seçilen dönemde hiç işlem oluşmadı.")
    st.plotly_chart(build_backtest_chart(result), use_container_width=True)
    table = result["trades_table"]
    if table.empty:
        return
    returns_fig = go.Figure(
        go.Bar(
            x=table["No"], y=table["Getiri %"],
            marker_color=np.where(table["Getiri %"] >= 0, "#26a69a", "#ef5350"),
            hovertemplate="İşlem %{x}: %{y:.2f}%<extra></extra>",
        )
    )
    returns_fig.update_layout(title="İşlem Başına Getiri %", xaxis_title="İşlem No", yaxis_title="Getiri %")
    st.plotly_chart(style_figure(returns_fig, 320), use_container_width=True)
    st.markdown("##### İşlem Listesi")
    st.dataframe(
        table,
        column_config={
            "Giriş Fiyatı": st.column_config.NumberColumn(format="%.2f"),
            "Çıkış / Son Fiyat": st.column_config.NumberColumn(format="%.2f"),
            "Adet": st.column_config.NumberColumn(format="%.4f"),
            "Getiri %": st.column_config.NumberColumn(format="%.2f"),
            "K/Z": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True,
        use_container_width=True,
        height=min(520, 60 + 36 * len(table)),
    )
    st.download_button(
        "⬇️ İşlem listesini CSV indir",
        csv_bytes(table),
        f"backtest_{result['ticker']}_{result['tf']}.csv",
        "text/csv",
        key="bt_download",
    )


def render_backtest_tab(scan, views, params):
    market = params["market"]
    mkt = MARKETS[market]
    st.markdown(
        "Seçilen hisse için LuxAlgo **Trendlines with Breaks** stratejisi (Long Only) geçmişte uygulanmış olsaydı "
        "sonuç ne olurdu? Sinyal bar kapanışında oluşur, işlem sonraki barın açılışında gerçekleşir."
    )
    options = []
    labels = {}
    if scan is not None and scan["params"]["market"] == market:
        uni = scan["uni"]
        priority = views["summary"]["Hisse"].tolist() if views is not None and not views["summary"].empty else []
        rest = [t for t in uni["ticker"].tolist() if t not in priority]
        options = priority + rest
        labels = dict(zip(uni["ticker"], uni["description"]))
    col1, col2, col3 = st.columns([2, 1, 1])
    picked = None
    if options:
        picked = col1.selectbox(
            "Hisse (filtre sonuçları üstte)", options,
            format_func=lambda t: f"{t} — {labels.get(t, t)}", key="bt_ticker",
        )
    manual = col1.text_input(
        "veya sembol yazın (örn. AAPL, THYAO)", key="bt_manual",
        help="Doluysa listedeki seçimin yerine geçer.",
    )
    tf = col2.selectbox("Zaman dilimi", TF_ORDER, index=2, format_func=lambda x: TF_CONFIG[x]["label"], key="bt_tf")
    period = col3.selectbox("Geçmiş veri", BT_PERIODS[tf], index=min(1, len(BT_PERIODS[tf]) - 1), key=f"bt_period_{tf}")
    c1, c2, c3, c4 = st.columns(4)
    capital = c1.number_input(f"Başlangıç sermayesi ({mkt['currency']})", 100.0, 1_000_000_000.0, 1000.0, 100.0, key="bt_capital")
    commission = c2.number_input("Komisyon % (her işlem yönü)", 0.0, 5.0, 0.04, 0.01, key="bt_commission")
    slippage = c3.number_input("Kayma % (her işlem yönü)", 0.0, 5.0, 0.0, 0.01, key="bt_slippage")
    fractional = c4.checkbox("Kesirli hisse", value=True, key="bt_fractional")
    st.caption(
        f"Pine parametreleri sol menüden alınır: Swing={params['length']}, Slope={params['mult']}, "
        f"Yöntem={params['method']}, Yalnızca kapanmış barlar={'Evet' if params['closed_only'] else 'Hayır'}"
    )
    if st.button("▶️ Backtest'i Çalıştır", type="primary", key="bt_run"):
        ticker = normalize_ticker(manual, market) if manual.strip() else picked
        if not ticker:
            st.error("Geçerli bir hisse seçin veya sembol yazın.")
        else:
            with st.spinner(f"{ticker} için backtest hesaplanıyor..."):
                st.session_state["bt_result"] = execute_backtest(
                    ticker, tf, period, market, params, float(capital), float(commission), float(slippage), bool(fractional),
                )
    result = st.session_state.get("bt_result")
    if result is not None:
        render_backtest_result(result)
    st.caption(
        "Backtest sonuçları geçmiş performansı gösterir, gelecekteki getiriyi garanti etmez. "
        "TradingView strateji test sonuçlarından küçük farklar olabilir."
    )


def render_method_tab():
    st.markdown(
        """
### Pine Script Mantığı

Strateji, LuxAlgo **Trendlines with Breaks** göstergesinin Long Only dönüşümüdür.

- **Pivotlar:** `Swing Detection Lookback` kadar sol ve sağ barın tepe/dip noktası pivot kabul edilir. Pivot, `length` bar sonra onaylanır.
- **Eğim:** `Atr`, `Stdev` veya `Linreg` yöntemiyle `length` ve `Slope` çarpanına göre hesaplanır.
- **Trend çizgileri:** Her yeni pivot yüksekte üst çizgi, her yeni pivot düşükte alt çizgi sıfırlanır; çizgi her barda eğim kadar ilerler.
- **Buy (B):** Üst trend çizgisi yukarı kırıldığında (`upos` 0 → 1) oluşur.
- **Sell (S):** Alt trend çizgisi aşağı kırıldığında (`dnos` 0 → 1) oluşur ve açık Long kapanır.

### Piyasalar

| Piyasa | Hisse listesi | Fiyat verisi | Seans |
|---|---|---|---|
| Amerika | TradingView `america` (NASDAQ, NYSE) | `yfinance` (sembol) | 09:30–16:00 ET |
| Türkiye | TradingView `turkey` (BIST) | `yfinance` (`.IS` eki) | 10:00–18:00 TRT |

4 saatlik mumlar seansa göre iki parçaya bölünür: ABD 09:30–13:30 / 13:30–16:00, BIST 10:00–14:00 / 14:00–18:00.

### Sinyal Modları

- **Taze Buy:** Son Buy sinyali son N bar içinde ve sonrasında Sell sinyali yok.
- **Aktif Long:** Son sinyal Buy, henüz Sell sinyali oluşmamış.

### Tablolar: Filtreleme ve İşaretleme

- Her tabloda **🔎 Tablo filtreleri ve sıralama** bölümü vardır: metin arama, istediğiniz sütunlar için aralık veya çoklu seçim filtresi, sıralama. Filtreler birlikte (VE mantığıyla) uygulanır.
- Satırların solundaki kutularla seçip **🟢 Yeşil / 🟡 Sarı / 🔴 Kırmızı** düğmeleriyle işaretleyin. Satırlar renklenir. "İşaret" sütunu da filtrelenebilir.
- İşaretler tüm tablolarda ortaktır. Sol menüden CSV olarak indirilip yüklenebilir.

### Backtest

- Long Only strateji: Buy sinyalinde alım, Sell sinyalinde satım. Sinyal bar kapanışında oluşur, işlem sonraki barın açılışında gerçekleşir.
- Sermayenin tamamı kullanılır, komisyon (varsayılan %0.04) ve isteğe bağlı kayma her işlem yönünde uygulanır.
- Özsermaye her barın kapanışına göre hesaplanır. Maks. düşüş bu eğriden bulunur.
- Al-Tut: veri setinin ilk barının açılışından son barın kapanışına.

### Breakout Skoru (0–100)

| Bileşen | Maks. Puan |
|---|---|
| Sinyal tazeliği (yeni sinyal = 25, her bar -5) | 25 |
| Göreceli hacim (RVOL, 3x = tam puan) | 20 |
| ADX (40 = tam puan) | 15 |
| RSI (50–70 arası tam puan) | 15 |
| Fiyat > EMA 50 | 10 |
| EMA 20 > EMA 50 | 5 |
| EMA 50 > EMA 200 | 5 |
| MACD histogram > 0 | 5 |

### Notlar

- "Yalnızca kapanmış barlar" açıkken oluşmakta olan bar hesaplamaya katılmaz.
- Veriler `yfinance` kaynaklıdır ve gecikmeli olabilir. Bu uygulama yatırım tavsiyesi değildir.
        """
    )


def render_welcome():
    st.markdown(
        """
        Sol menüden **piyasayı** (Amerika veya Türkiye BIST), evreni, zaman dilimlerini ve Pine parametrelerini seçip
        **🚀 Taramayı Başlat** düğmesine basın.

        Uygulama, TradingView Scanner üzerinden hisse listesini alır, `yfinance` ile fiyat verilerini çeker
        ve LuxAlgo *Trendlines with Breaks* mantığıyla **Buy** sinyalindeki hisseleri bulur.
        """
    )
    st.info("İlk tarama, hisse sayısına bağlı olarak 1–3 dakika sürebilir. Sonraki taramalar önbellekten hızlı gelir.")


def main():
    inject_css()
    st.title("📈 Breakout Screener")

    params, flt, run, clear = sidebar_controls()

    if clear:
        st.cache_data.clear()
        st.session_state.pop("scan", None)
        st.toast("Önbellek temizlendi.")

    if run:
        if not params["tfs"]:
            st.sidebar.error("En az bir zaman dilimi seçin.")
        else:
            result = run_scan(params)
            if result is not None:
                st.session_state["scan"] = result

    scan = st.session_state.get("scan")
    if scan is None:
        st.caption("LuxAlgo Trendlines with Breaks | 1h · 4h · 1d · 1w")
        render_welcome()
        st.markdown("---")
        st.subheader("🧪 Hızlı Backtest")
        render_backtest_tab(None, None, params)
        return

    scan_market = scan["params"]["market"]
    scan_mkt = MARKETS[scan_market]
    st.caption(f"{scan_mkt['label']} | LuxAlgo Trendlines with Breaks | 1h · 4h · 1d · 1w")

    if scan["params"] != params:
        st.warning("Sol menüdeki piyasa / evren / Pine ayarları son taramadan farklı. Yeni ayarlar için taramayı yeniden başlatın.")

    views = compute_views(scan, flt)
    if views is None:
        st.error("Tarama sonucu boş.")
        return

    summary = views["summary"]
    tfs = scan["params"]["tfs"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Taranan Hisse", f"{scan['raw']['ticker'].nunique()}")
    c2.metric("Eşleşen Hisse", f"{len(summary)}")
    c3.metric("Eşleşen Sinyal", f"{len(views['matched'])}")
    c4.metric("Tüm TF Uyumlu", f"{int((summary['Uyum'] == len(tfs)).sum()) if not summary.empty else 0}")
    c5.metric("Son Tarama", scan["time"])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "🎯 Sonuçlar", "📋 Sinyal Detayları", "📊 Görselleştirme", "🏦 Temel Analiz",
            "🔍 Hisse Detay", "🧪 Backtest", "ℹ️ Metodoloji",
        ]
    )
    with tab1:
        render_summary_tab(views, tfs, scan_market)
    with tab2:
        render_detail_table_tab(views, scan_market)
    with tab3:
        render_visual_tab(views, tfs)
    with tab4:
        render_fundamental_tab(views, scan_market)
    with tab5:
        render_detail_tab(scan, views)
    with tab6:
        render_backtest_tab(scan, views, params)
    with tab7:
        render_method_tab()


main()
