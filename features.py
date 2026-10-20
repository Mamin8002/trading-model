"""
features.py — Технический индикаторный пайплайн (табличные признаки).

Вместо картинок графиков (слабый CNN-подход) строим плотный табличный
фичерсет: RSI, MACD, ATR, EMA/SMA разных длин, лог-доходности на нескольких
горизонтах, объёмные метрики, позиции в диапазоне,Patterns свечей и т.д.

Целевая переменная (multi-class):
    y = 1  если через `horizon` баров доходность >  + threshold(threshold_scale * ATR%)
    y = -1 если доходность <  - threshold
    y = 0  иначе (нейтрально / шум)
Порог адаптивен к волатильности, поэтому классы сбалансированнее между ТФ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Индикаторы (чистый pandas/numpy, без talib-зависимости)
# ---------------------------------------------------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window // 2).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def stoch_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    ll = df["low"].rolling(period, min_periods=period // 2).min()
    hh = df["high"].rolling(period, min_periods=period // 2).max()
    return 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)


def adlf_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Proxy money-flow: доля объёма в 'умных' свечах (закрытие в верхней части диапазона)."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    cl_pos = (df["close"] - df["low"]) / rng            # 0..1
    signed_vol = df["volume"] * (2 * cl_pos - 1)
    return signed_vol.rolling(window, min_periods=window // 2).sum() / df["volume"].rolling(window).sum().replace(0, np.nan)


# ---------------------------------------------------------------------------
# Основной конструктор признаков
# ---------------------------------------------------------------------------
FEATURE_PREFIXES = ("ret_", "ema_", "rsi", "macd", "atr_", "bb_", "stoch",
                    "vol_", "rng_", "pos_", "gap_", "cw_", "obv", "mf_", "tf_")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вход: OHLCV DataFrame (datetime, open, high, low, close, volume).
    Выход: DataFrame только с числовыми признаками (индекс сохранён).
    Все признаки считаются CAUSALLY (без заглядывания в будущее).
    """
    out = pd.DataFrame(index=df.index)
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    logc = np.log(c)

    # --- логарифмические доходности на нескольких горизонтах ---
    for k in (1, 2, 3, 5, 10, 20, 60):
        out[f"ret_{k}"] = logc - logc.shift(k)

    # --- EMA-системы и расстояния до них ---
    e9, e21, e50, e200 = ema(c, 9), ema(c, 21), ema(c, 50), ema(c, 200)
    out["ema_ratio_9_21"] = c / e21 - 1.0
    out["ema_ratio_21_50"] = e21 / e50 - 1.0
    out["ema_ratio_50_200"] = e50 / e200 - 1.0
    out["ema_slope_21"] = e21.pct_change(3)
    out["ema_slope_50"] = e50.pct_change(5)

    # --- RSI (быстрый/медленный) ---
    out["rsi_14"] = rsi(c, 14)
    out["rsi_7"] = rsi(c, 7)
    out["rsi_diff"] = out["rsi_7"] - out["rsi_14"]

    # --- MACD нормализованный ценой ---
    m_line, s_line, m_hist = macd(c)
    out["macd_line_n"] = m_line / c
    out["macd_signal_n"] = s_line / c
    out["macd_hist_n"] = m_hist / c
    out["macd_hist_chg"] = m_hist.diff() / c

    # --- ATR и волатильность ---
    a14 = atr(df, 14)
    out["atr_pct"] = a14 / c
    out["atr_ratio_14_50"] = a14 / atr(df, 50)
    ret1 = out["ret_1"]
    out["vol_std_20"] = ret1.rolling(20, min_periods=10).std()
    out["vol_std_60"] = ret1.rolling(60, min_periods=30).std()
    out["vol_regime"] = out["vol_std_20"] / out["vol_std_60"].replace(0, np.nan)

    # --- Bollinger ---
    ma20 = sma(c, 20)
    sd20 = c.rolling(20, min_periods=10).std()
    out["bb_pos"] = (c - ma20) / (2 * sd20).replace(0, np.nan)   # %B упрощённо
    out["bb_width"] = (4 * sd20) / ma20

    # --- Stochastic ---
    out["stoch_k_14"] = stoch_k(df, 14)
    out["stoch_k_slow"] = out["stoch_k_14"].rolling(3, min_periods=2).mean()

    # --- Объёмные метрики ---
    vol_ma20 = v.rolling(20, min_periods=10).mean()
    out["vol_rel_20"] = (v + 1.0) / (vol_ma20 + 1.0)
    out["vol_ret_interact"] = out["ret_1"] * np.log1p(v / (vol_ma20 + 1.0))
    obv = (np.sign(ret1.fillna(0.0)) * v).cumsum()
    out["obv_slope_10"] = (obv - obv.shift(10)) / (v.rolling(10).sum() + 1.0)
    out["mf_flow_20"] = adlf_proxy(df, 20)

    # --- Позиция закрытия в дневном/скользящем диапазоне ---
    roll_hi = h.rolling(96, min_periods=24).max()
    roll_lo = l.rolling(96, min_periods=24).min()
    out["pos_in_range_96"] = (c - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)
    dist_high = c / roll_hi - 1.0
    dist_low = c / roll_lo - 1.0
    out["dist_roll_high"] = dist_high
    out["dist_roll_low"] = dist_low

    # --- Свечные паттерны (continuous, не бинарные) ---
    body = c - o
    rng_bar = (h - l).replace(0, np.nan)
    out["cw_body_ratio"] = body / rng_bar                 # сила тела vs фитили
    out["cw_upper_wick"] = (h - np.maximum(c, o)) / rng_bar
    out["cw_lower_wick"] = (np.minimum(c, o) - l) / rng_bar
    out["cw_gap"] = o / c.shift(1) - 1.0                  # гэп открытия
    out["cw_close_vs_prev"] = c / c.shift(1) - 1.0

    # --- Внутридневная сезонность (для intraday ТФ) ---
    if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        mins = pd.to_datetime(df["datetime"]).dt.hour * 60 + pd.to_datetime(df["datetime"]).dt.minute
        out["tf_sin_mday"] = np.sin(2 * np.pi * mins / (24 * 60))
        out["tf_cos_mday"] = np.cos(2 * np.pi * mins / (24 * 60))
        out["tf_dow"] = pd.to_datetime(df["datetime"]).dt.dayofweek / 4.0

    # служебная колонка ATR для риск-менеджера (не фича модели)
    out["_atr_abs"] = a14
    out["_atr_pct"] = out["atr_pct"]

    return out


def feature_columns(fx: pd.DataFrame) -> list[str]:
    return [col for col in fx.columns if not col.startswith("_")]


# ---------------------------------------------------------------------------
# Целевая переменная
# ---------------------------------------------------------------------------
def make_target(df: pd.DataFrame, fx: pd.DataFrame, horizon_bars: int,
                atr_threshold_mult: float = 0.5) -> pd.Series:
    """
    Мультикласс: 1 (up) / 0 (flat) / -1 (down) по future log-return за `horizon`.
    Порог = atr_threshold_mult * ATR% за горизонт (адаптивно к волатильности).
    """
    logc = np.log(df["close"])
    fwd_ret = logc.shift(-horizon_bars) - logc
    # ожидаемый ATR-scale на horizon баров ~ atr_pct * sqrt(horizon)
    thr = fx["_atr_pct"] * np.sqrt(horizon_bars) * atr_threshold_mult
    y = pd.Series(0, index=df.index, dtype=int)
    y[fwd_ret > thr] = 1
    y[fwd_ret < -thr] = -1
    y[fwd_ret.isna()] = np.nan          # последние horizon баров не имеют цели
    return y


def build_dataset(df: pd.DataFrame, horizon_bars: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Полный пайплайн: OHLCV -> (X, y, feature_names)."""
    fx = build_features(df)
    y = make_target(df, fx, horizon_bars)
    feats = feature_columns(fx)
    X = fx[feats].copy()
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid].astype(int)
    return X, y, feats
