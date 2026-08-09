"""
Loading prices and building the features the ML model reads.

Everything here looks BACKWARDS in time only. The rule applied throughout is:
a feature for day i may use prices on day i or earlier, never later. The label
asks about day i + PREDICT_DAYS, so it is always strictly in the future.
"""

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- loading
def _read_price_csv(path):
    """Read one cached price file and strip the timezone off the index."""
    df = pd.read_csv(path, parse_dates=["Date"])
    # The cached files carry a market timezone. Dropping it makes every series
    # line up on a plain calendar date, which is all this project needs.
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df.set_index("Date").sort_index()


def load_prices(tickers=None, download_if_missing=True):
    """Return {ticker: DataFrame}. Reads the cached CSVs; downloads only if absent."""
    tickers = tickers or C.TICKERS
    out = {}
    for tick in tickers:
        path = C.DATA_DIR / f"{tick}_prices.csv"
        if path.exists():
            out[tick] = _read_price_csv(path)
        elif download_if_missing:
            import yfinance as yf
            df = yf.Ticker(tick).history(start=C.START_DATE, end=C.END_DATE)
            df.to_csv(path)
            out[tick] = _read_price_csv(path)
        else:
            raise FileNotFoundError(f"No cached prices for {tick} at {path}")
    return out


def load_vix(download_if_missing=True):
    """Return the VIX close as a Series. Used to label each day calm or shock."""
    path = C.DATA_DIR / "VIX.csv"
    if not path.exists() and download_if_missing:
        import yfinance as yf
        yf.Ticker(C.VIX_TICKER).history(start=C.START_DATE, end=C.END_DATE).to_csv(path)
    return _read_price_csv(path)["Close"].astype(float)


def regime_of(vix_series, date):
    """'SHOCK' if the VIX on (or last before) this date is above the threshold."""
    prior = vix_series.index[vix_series.index <= pd.Timestamp(date)]
    if len(prior) == 0:
        return "CALM", float("nan")
    level = float(vix_series.loc[prior[-1]])
    return ("SHOCK" if level > C.VIX_SHOCK_LEVEL else "CALM"), level


# ---------------------------------------------------------------- features
def rsi(series, period=14):
    """Relative Strength Index, rescaled to sit between 0 and 1."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 1.0 - 1.0 / (1.0 + gain / (loss + 1e-12))


def build_features(df, vix_aligned, mkt_close):
    """Create the 22 features for one stock. Every one looks backwards only."""
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    f = pd.DataFrame(index=df.index)

    # --- momentum
    f["rsi_14"] = rsi(close, 14)

    # --- trend (MACD)
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd_ratio"]        = macd / (close + 1e-12)
    f["macd_signal_ratio"] = signal / (close + 1e-12)

    # --- price against its own moving averages
    for n in (5, 10, 20, 50):
        f[f"ma{n}_ratio"] = close / (close.rolling(n).mean() + 1e-12)

    # --- Bollinger Bands
    ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    f["bb_upper_ratio"] = close / (ma20 + 2 * sd20 + 1e-12)
    f["bb_lower_ratio"] = close / (ma20 - 2 * sd20 + 1e-12)
    f["bb_width"]       = (4 * sd20) / (ma20 + 1e-12)

    # --- returns over several windows
    for n in (1, 5, 10, 20):
        f[f"ret_{n}d"] = close.pct_change(n)

    # --- volatility and volume
    f["vol_10d"]   = f["ret_1d"].rolling(10).std()
    f["vol_ratio"] = volume / (volume.rolling(20).mean() + 1e-12)

    prev = close.shift(1)
    tr   = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    f["atr_ratio"] = tr.rolling(14).mean() / (close + 1e-12)

    # --- market regime, as the model sees it (the 5 features that let the ML tier
    #     know what kind of market it is trading in, not just its own chart)
    v = vix_aligned.reindex(f.index).ffill()
    f["vix_level"]      = v / 100.0
    f["vix_chg_5d"]     = v.pct_change(5)
    f["vix_ma20_ratio"] = v / (v.rolling(20).mean() + 1e-12)

    mkt = mkt_close.reindex(f.index).ffill()
    f["mkt_ret_5d"] = mkt.pct_change(5)
    f["rel_ret_5d"] = f["ret_5d"] - f["mkt_ret_5d"]

    return f.replace([np.inf, -np.inf], np.nan).dropna()


def make_target(close, horizon=None):
    """1 if the close is higher `horizon` trading days later, else 0."""
    horizon = horizon or C.PREDICT_DAYS
    future  = close.shift(-horizon)
    # Comparing NaN with a number returns False, so casting before filtering used
    # to label the final ``horizon`` rows as DOWN even though no future close
    # exists.  Keep only rows whose outcome is genuinely observable first.
    valid = future.notna()
    return (future.loc[valid] > close.loc[valid]).astype(int)


def load_everything(tickers=None):
    """One call that returns everything downstream code needs.

    Returns a dict with prices, vix, features, targets and the feature names.
    """
    tickers = tickers or C.TICKERS
    prices  = load_prices(tickers)
    vix     = load_vix()

    closes_df   = pd.DataFrame({t: prices[t]["Close"].astype(float) for t in tickers}).dropna()
    mkt_close   = closes_df.mean(axis=1)                    # equal-weighted basket of the 5
    vix_aligned = vix.reindex(closes_df.index).ffill()

    feats, targets = {}, {}
    for tick in tickers:
        f = build_features(prices[tick], vix_aligned, mkt_close)
        y = make_target(prices[tick]["Close"].astype(float))
        idx = f.index.intersection(y.index)                 # keep rows that have BOTH
        feats[tick], targets[tick] = f.loc[idx], y.loc[idx]

    return {
        "prices":        prices,
        "vix":           vix,
        "closes_df":     closes_df,
        "mkt_close":     mkt_close,
        "features":      feats,
        "targets":       targets,
        "feature_names": list(feats[tickers[0]].columns),
        "tickers":       tickers,
    }
