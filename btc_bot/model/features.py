"""Признаки модели.

Наборы:
  * "raw"   - как в оригинальном ноутбуке: open, high, low, close, volume, turnover
              (один закрытый бар);
  * "ratio" - v1: цены относительно close текущего бара + volume, turnover (один бар);
  * "v2"    - Этап 5: лаг-доходности, форма бара, относительный объём, время суток.
              Требует окна ЗАКРЫТЫХ баров (warm-up 21 бар). Все признаки вычисляются
              только по барам <= t - будущие данные не используются.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

RAW_FEATURES = ["open", "high", "low", "close", "volume", "turnover"]
RATIO_FEATURES = ["open_rel", "high_rel", "low_rel", "volume", "turnover"]
V2_FEATURES = ["ret_1", "ret_3", "ret_h", "body", "range_rel",
               "upper_shadow", "lower_shadow", "vol_ratio_20", "tod_sin", "tod_cos"]
VALID_FEATURE_SETS = ("raw", "ratio", "v2")
SINGLE_BAR_SETS = ("raw", "ratio")     # достаточно одного закрытого бара
V2_VOL_WINDOW = 20                      # окно средней объёма (бары ДО текущего)


class InsufficientHistoryError(RuntimeError):
    """Не хватает закрытых баров для вычисления признаков (например, v2 warm-up)."""


def v2_min_bars(horizon_bars: int) -> int:
    """Минимальное число закрытых баров (включая текущий) для признаков v2."""
    return max(V2_VOL_WINDOW, 3, int(horizon_bars)) + 1


def feature_columns(feature_set: str, horizon_bars: int = 1) -> list[str]:
    if feature_set == "raw":
        return list(RAW_FEATURES)
    if feature_set == "ratio":
        return list(RATIO_FEATURES)
    if feature_set == "v2":
        return list(V2_FEATURES)
    raise ValueError(f"Неизвестный feature_set: {feature_set!r} (допустимы {VALID_FEATURE_SETS})")


def build_features(df: pd.DataFrame, feature_set: str, horizon_bars: int = 1) -> pd.DataFrame:
    """Матрица признаков из очищенного DataFrame свечей (порядок строк сохраняется).

    Для v2 первые max(20, 3, horizon_bars) строк содержат NaN (warm-up) - они
    исключаются позже в build_dataset (честно, без фабрикации данных).
    """
    if feature_set == "raw":
        out = df[RAW_FEATURES].astype("float64").reset_index(drop=True)
    elif feature_set == "ratio":
        close = df["close"].astype("float64").to_numpy()
        if np.any(close <= 0):
            raise ValueError("close <= 0 - некорректные данные для ratio-признаков")
        out = pd.DataFrame(
            {
                "open_rel": df["open"].astype("float64").to_numpy() / close,
                "high_rel": df["high"].astype("float64").to_numpy() / close,
                "low_rel": df["low"].astype("float64").to_numpy() / close,
                "volume": df["volume"].astype("float64").to_numpy(),
                "turnover": df["turnover"].astype("float64").to_numpy(),
            }
        )
    elif feature_set == "v2":
        close = df["close"].astype("float64")
        open_ = df["open"].astype("float64")
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        volume = df["volume"].astype("float64")
        h = int(horizon_bars)
        # доходности - только по ПРОШЛЫМ барам (shift > 0)
        ret_1 = close / close.shift(1) - 1.0
        ret_3 = close / close.shift(3) - 1.0
        ret_h = close / close.shift(h) - 1.0
        body = (close - open_) / close
        range_rel = (high - low) / close
        upper_shadow = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close
        lower_shadow = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close
        vol_mean_prev = volume.rolling(V2_VOL_WINDOW, min_periods=V2_VOL_WINDOW).mean().shift(1)
        vol_ratio_20 = volume / (vol_mean_prev + 1e-12)
        minutes = (df["start_ms"].astype("int64") // 60_000) % 1440  # минута дня UTC
        tod_sin = np.sin(2 * math.pi * minutes / 1440.0)
        tod_cos = np.cos(2 * math.pi * minutes / 1440.0)
        out = pd.DataFrame({
            "ret_1": ret_1, "ret_3": ret_3, "ret_h": ret_h, "body": body,
            "range_rel": range_rel, "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow, "vol_ratio_20": vol_ratio_20,
            "tod_sin": tod_sin, "tod_cos": tod_cos,
        })
    else:
        raise ValueError(f"Неизвестный feature_set: {feature_set!r}")
    if feature_set != "v2" and out.isna().any().any():
        raise ValueError("В признаках есть NaN - данные должны быть очищены до этого шага")
    return out


def _bar_dict(bar: Mapping) -> dict:
    return {k: float(bar[k]) for k in ("open", "high", "low", "close", "volume", "turnover")}


def bar_features(bar: Mapping, feature_set: str) -> dict[str, float]:
    """Признаки одного закрытого бара (только для single-bar наборов raw/ratio)."""
    if feature_set == "v2":
        raise InsufficientHistoryError(
            "Набор v2 требует окна закрытых баров - используйте history_features()")
    b = _bar_dict(bar)
    o, h, l, c, v, t = b["open"], b["high"], b["low"], b["close"], b["volume"], b["turnover"]
    if not (c > 0 and h >= l and h >= max(o, c) and l <= min(o, c)):
        raise ValueError(f"Некорректный бар для построения признаков: {dict(bar)}")
    if feature_set == "raw":
        return {"open": o, "high": h, "low": l, "close": c, "volume": v, "turnover": t}
    if feature_set == "ratio":
        return {"open_rel": o / c, "high_rel": h / c, "low_rel": l / c, "volume": v, "turnover": t}
    raise ValueError(f"Неизвестный feature_set: {feature_set!r}")


def history_features(bars: Sequence[Mapping], feature_set: str,
                     horizon_bars: int = 1) -> dict[str, float]:
    """Признаки ПОСЛЕДНЕГО бара из окна закрытых баров (bars[-1] = текущий закрытый бар).

    Для v2 требуется >= v2_min_bars(horizon_bars) баров; бары должны быть в
    хронологическом порядке. Исключение: InsufficientHistoryError.
    """
    if feature_set in SINGLE_BAR_SETS:
        return bar_features(bars[-1], feature_set)
    if feature_set != "v2":
        raise ValueError(f"Неизвестный feature_set: {feature_set!r}")
    need = v2_min_bars(horizon_bars)
    if len(bars) < need:
        raise InsufficientHistoryError(
            f"Для признаков v2 нужно >= {need} закрытых баров, получено {len(bars)}")
    arrs = {k: np.array([float(b[k]) for b in bars]) for k in
            ("start_ms", "open", "high", "low", "close", "volume")}
    if np.any(arrs["start_ms"][1:] <= arrs["start_ms"][:-1]):
        raise ValueError("history_features: бары должны быть строго по возрастанию start_ms")
    c = arrs["close"]; o = arrs["open"]; h = arrs["high"]; l = arrs["low"]; v = arrs["volume"]
    i = len(bars) - 1
    h_bars = int(horizon_bars)
    if c[i - 1] <= 0 or c[i - 3] <= 0 or c[i - h_bars] <= 0 or c[i] <= 0:
        raise InsufficientHistoryError("некорректные цены в окне истории")
    ret_1 = c[i] / c[i - 1] - 1.0
    ret_3 = c[i] / c[i - 3] - 1.0
    ret_h = c[i] / c[i - h_bars] - 1.0
    body = (c[i] - o[i]) / c[i]
    range_rel = (h[i] - l[i]) / c[i]
    upper_shadow = (h[i] - max(o[i], c[i])) / c[i]
    lower_shadow = (min(o[i], c[i]) - l[i]) / c[i]
    vol_mean_prev = float(np.mean(v[i - V2_VOL_WINDOW:i]))
    vol_ratio_20 = v[i] / (vol_mean_prev + 1e-12)
    minutes = (int(arrs["start_ms"][i]) // 60_000) % 1440
    tod_sin = math.sin(2 * math.pi * minutes / 1440.0)
    tod_cos = math.cos(2 * math.pi * minutes / 1440.0)
    return {"ret_1": ret_1, "ret_3": ret_3, "ret_h": ret_h, "body": body,
            "range_rel": range_rel, "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow, "vol_ratio_20": vol_ratio_20,
            "tod_sin": tod_sin, "tod_cos": tod_cos}


def bar_features_frame(bar: Mapping, feature_set: str, horizon_bars: int = 1) -> pd.DataFrame:
    """Однострочный DataFrame признаков (порядок колонок = feature_columns)."""
    if feature_set == "v2":
        raise InsufficientHistoryError(
            "Набор v2 требует окна закрытых баров - используйте history_features_frame()")
    feats = bar_features(bar, feature_set)
    cols = feature_columns(feature_set)
    return pd.DataFrame([[feats[c] for c in cols]], columns=cols)


def history_features_frame(bars: Sequence[Mapping], feature_set: str,
                           horizon_bars: int = 1) -> pd.DataFrame:
    feats = history_features(bars, feature_set, horizon_bars)
    cols = feature_columns(feature_set, horizon_bars)
    return pd.DataFrame([[feats[c] for c in cols]], columns=cols)
