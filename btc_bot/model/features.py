"""Признаки модели (подход исходного 5-минутного ноутбука + минимальная модификация).

Оригинал: X = df.drop(['timestamp']) -> сырые OHLCV-колонки.
Варианты:
  * "raw"   - точно как в оригинале: open, high, low, close, volume, turnover;
  * "ratio" - минимально необходимая модификация: цены относительно close текущего
              бара (open/close, high/close, low/close) + volume, turnover. Причина:
              деревья не экстраполируют за пределы уровней, встреченных в обучении,
              а цена BTC за 2024-2026 менялась в разы (38k -> 126k -> 84k).
Оба набора вычислимы по ОДНОМУ закрытому бару в реальном времени.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

RAW_FEATURES = ["open", "high", "low", "close", "volume", "turnover"]
RATIO_FEATURES = ["open_rel", "high_rel", "low_rel", "volume", "turnover"]
VALID_FEATURE_SETS = ("raw", "ratio")


def feature_columns(feature_set: str) -> list[str]:
    if feature_set == "raw":
        return list(RAW_FEATURES)
    if feature_set == "ratio":
        return list(RATIO_FEATURES)
    raise ValueError(f"Неизвестный feature_set: {feature_set!r} (допустимы {VALID_FEATURE_SETS})")


def build_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Матрица признаков из очищенного DataFrame свечей (порядок строк сохраняется)."""
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
    else:
        raise ValueError(f"Неизвестный feature_set: {feature_set!r}")
    if out.isna().any().any():
        raise ValueError("В признаках есть NaN - данные должны быть очищены до этого шага")
    return out


def bar_features(bar: Mapping, feature_set: str) -> dict[str, float]:
    """Признаки одного закрытого бара (для live-предсказания). bar: open/high/low/close/volume/turnover."""
    o = float(bar["open"]); h = float(bar["high"]); l = float(bar["low"])
    c = float(bar["close"]); v = float(bar["volume"]); t = float(bar["turnover"])
    if not (c > 0 and h >= l and h >= max(o, c) and l <= min(o, c)):
        raise ValueError(f"Некорректный бар для построения признаков: {dict(bar)}")
    if feature_set == "raw":
        return {"open": o, "high": h, "low": l, "close": c, "volume": v, "turnover": t}
    if feature_set == "ratio":
        return {"open_rel": o / c, "high_rel": h / c, "low_rel": l / c, "volume": v, "turnover": t}
    raise ValueError(f"Неизвестный feature_set: {feature_set!r}")


def bar_features_frame(bar: Mapping, feature_set: str) -> pd.DataFrame:
    """Однострочный DataFrame признаков (порядок колонок = feature_columns)."""
    feats = bar_features(bar, feature_set)
    cols = feature_columns(feature_set)
    return pd.DataFrame([[feats[c] for c in cols]], columns=cols)
