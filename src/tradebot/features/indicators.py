"""Причинные индикаторы для табличных признаков (docs/ml.md §2).

Гарантии причинности (architecture §6.1):
- значение индикатора в строке t зависит только от строк <= t;
- rolling-окна — trailing (min_periods=window, правый край включительно);
- EMA/RSI/ATR используют ewm(adjust=False) — рекурсия от начала серии,
  префикс-инвариантны (значение в t не меняется при продлении серии);
- никаких центрированных окон, shift(-k) и глобальных нормировок здесь нет.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "atr_wilder",
    "ema",
    "log_return",
    "macd_lines",
    "parkinson_vol",
    "rsi_wilder",
    "safe_ratio",
    "trailing_std",
]


def log_return(close: pd.Series, lag: int) -> pd.Series:
    """Лог-доходность за lag баров: ln(close_t / close_{t-lag})."""
    if lag < 1:
        raise ValueError("lag >= 1")
    return np.log(close / close.shift(lag))


def ema(series: pd.Series, span: int) -> pd.Series:
    """EMA (adjust=False): рекурсивная, причинная, префикс-инвариантная."""
    if span < 2:
        raise ValueError("span >= 2")
    return series.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int) -> pd.Series:
    """RSI по Уайлдеру (0..100). Первые period строк — NaN (min_periods).

    Вырожденные случаи: avg_loss == 0 и avg_gain > 0 -> 100; обе средние нулевые
    (постоянная цена) -> NaN (ряд исключается как непрогретый/неинформативный).
    """
    if period < 2:
        raise ValueError("period >= 2")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    both_zero = (avg_loss == 0.0) & (avg_gain == 0.0)
    only_gain = (avg_loss == 0.0) & (avg_gain > 0.0)
    rsi = rsi.where(~only_gain, 100.0)
    rsi = rsi.where(~both_zero, np.nan)
    return rsi


def macd_lines(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD: (macd, signal, hist) — все линии причинные (ewm adjust=False)."""
    macd = ema(close, fast) - ema(close, slow)
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """ATR по Уайлдеру (ewm alpha=1/period, min_periods=period)."""
    if period < 2:
        raise ValueError("period >= 2")
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def trailing_std(series: pd.Series, window: int) -> pd.Series:
    """Скользящее стандартное отклонение (ddof=1) по trailing-окну."""
    if window < 2:
        raise ValueError("window >= 2")
    return series.rolling(window, min_periods=window).std(ddof=1)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Оценка Паркинсона: sqrt(mean(ln(high/low)^2, window) / (4 ln 2))."""
    if window < 2:
        raise ValueError("window >= 2")
    log_hl = np.log(high / low.where(low > 0.0))
    squared = log_hl.pow(2)
    return np.sqrt(squared.rolling(window, min_periods=window).mean() / (4.0 * np.log(2.0)))


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Деление с защитой от нуля/NaN в знаменателе (результат NaN, не inf)."""
    return numerator / denominator.where(denominator != 0.0)
