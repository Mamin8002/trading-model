"""FeatureBuilder: бары -> матрица причинных признаков (architecture §5.4, §6.1).

Контракт:
- вход: финальные бары, строго упорядоченные по ts_open (UTC, bar_open);
- выход: DataFrame с индексом ts_open и колонками в фиксированном порядке
  ``spec.feature_names``; строки прогрева содержат NaN и исключаются на уровне
  датасета (решения по ним не принимаются);
- признак в строке t использует только бары <= t (проверяется тестами
  префикс-инвариантности и возмущения будущего — tests/test_features.py);
- inf заменяются на NaN (строка исключается), никаких глобальных нормализаций
  по всей выборке нет и быть не должно (architecture §6.4).
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..data.bar import Bar
from .indicators import (
    atr_wilder,
    ema,
    log_return,
    macd_lines,
    parkinson_vol,
    rsi_wilder,
    safe_ratio,
    trailing_std,
)
from .spec import FeatureSpec

__all__ = ["FeatureBuilder"]


class FeatureBuilder:
    def __init__(self, spec: FeatureSpec, local_timezone: str = "UTC") -> None:
        if spec.version != "v1":
            raise ValueError(f"FeatureBuilder: неизвестная версия набора признаков {spec.version!r}")
        self.spec = spec
        self.local_timezone = ZoneInfo(local_timezone)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.spec.feature_names

    @classmethod
    def from_config(cls, features_cfg, local_timezone: str = "UTC") -> "FeatureBuilder":
        return cls(FeatureSpec.from_config(features_cfg), local_timezone=local_timezone)

    def build(self, bars: Sequence[Bar]) -> pd.DataFrame:
        self._validate_bars(bars)
        columns = list(self.spec.feature_names)
        if not bars:
            return pd.DataFrame(
                {name: pd.Series(dtype="float64") for name in columns},
                index=pd.DatetimeIndex([], tz="UTC", name="ts_open"),
            )

        index = pd.DatetimeIndex([b.ts_open for b in bars], tz="UTC", name="ts_open")
        open_ = pd.Series([b.open for b in bars], index=index, dtype="float64")
        high = pd.Series([b.high for b in bars], index=index, dtype="float64")
        low = pd.Series([b.low for b in bars], index=index, dtype="float64")
        close = pd.Series([b.close for b in bars], index=index, dtype="float64")
        volume = pd.Series([float(b.volume) for b in bars], index=index, dtype="float64")

        p = _Params(self.spec.params)
        features: dict[str, pd.Series] = {}

        # 1) доходности
        ret1 = log_return(close, 1)
        for lag in p.return_lags:
            features[f"ret_{lag}"] = log_return(close, lag) if lag != 1 else ret1

        # 2) EMA: отношение цены и наклон (причинные ewm adjust=False)
        for period in p.ema_periods:
            ema_p = ema(close, period)
            features[f"ema_ratio_{period}"] = safe_ratio(close, ema_p) - 1.0
            features[f"ema_slope_{period}"] = safe_ratio(ema_p, ema_p.shift(1)) - 1.0

        # 3) RSI (Wilder)
        features[f"rsi_{p.rsi_period}"] = rsi_wilder(close, p.rsi_period)

        # 4) MACD, нормированный на close (масштабная инвариантность между инструментами)
        macd, signal, hist = macd_lines(close, p.macd_fast, p.macd_slow, p.macd_signal)
        features["macd_norm"] = safe_ratio(macd, close)
        features["macd_signal_norm"] = safe_ratio(signal, close)
        features["macd_hist_norm"] = safe_ratio(hist, close)

        # 5) ATR / close
        features[f"atr_ratio_{p.atr_period}"] = safe_ratio(atr_wilder(high, low, close, p.atr_period), close)

        # 6) структура свечи (все — внутри бара t, нормировано на close)
        bar_range = high - low
        body_top = pd.concat([open_, close], axis=1).max(axis=1)
        body_bot = pd.concat([open_, close], axis=1).min(axis=1)
        features["bar_range"] = safe_ratio(bar_range, close)
        features["bar_body"] = safe_ratio(close - open_, close)
        features["upper_wick"] = safe_ratio(high - body_top, close)
        features["lower_wick"] = safe_ratio(body_bot - low, close)
        # вырожденный бар (high == low) -> 0.5 (серединное значение, документировано)
        features["close_position"] = safe_ratio(close - low, bar_range).where(bar_range > 0.0, 0.5)
        features["gap_prev"] = safe_ratio(open_, close.shift(1)) - 1.0

        # 7) волатильность
        vol_series: list[pd.Series] = []
        for window in p.vol_windows:
            vol_w = trailing_std(ret1, window)
            features[f"vol_{window}"] = vol_w
            vol_series.append(vol_w)
        if len(vol_series) >= 2:
            features["vol_ratio"] = safe_ratio(vol_series[0], vol_series[-1])
        for window in p.vol_windows:
            features[f"park_vol_{window}"] = parkinson_vol(high, low, window)

        # 8) объём (относительные характеристики — абсолютный объём не масштабируем)
        vol_mean_short = volume.rolling(p.volume_window, min_periods=p.volume_window).mean()
        vol_mean_long = volume.rolling(p.volume_trend_window, min_periods=p.volume_trend_window).mean()
        features[f"volume_ratio_{p.volume_window}"] = safe_ratio(volume, vol_mean_short)
        features[f"volume_trend_{p.volume_window}_{p.volume_trend_window}"] = (
            safe_ratio(vol_mean_short, vol_mean_long) - 1.0
        )

        # 9) внутридневное время (локальная зона биржи из конфига)
        local_index = index.tz_convert(self.local_timezone)
        minute_of_day = np.asarray(local_index.hour, dtype="float64") * 60.0 + np.asarray(
            local_index.minute, dtype="float64"
        )
        features["minute_of_day"] = pd.Series(minute_of_day, index=index)
        features["day_of_week"] = pd.Series(np.asarray(local_index.weekday, dtype="float64"), index=index)

        frame = pd.DataFrame(features, index=index)
        missing = [name for name in columns if name not in frame.columns]
        extra = [name for name in frame.columns if name not in columns]
        if missing or extra:
            raise AssertionError(
                f"Рассинхрон спецификации и построителя: missing={missing}, extra={extra}"
            )
        frame = frame[columns]
        frame = frame.replace([np.inf, -np.inf], np.nan)
        return frame

    @staticmethod
    def _validate_bars(bars: Sequence[Bar]) -> None:
        last: datetime | None = None
        for bar in bars:
            if not bar.is_final:
                raise ValueError(
                    "FeatureBuilder принимает только финальные бары (is_final=True): "
                    "незакрытая свеча мутирует и нарушает причинность (decisions D9)"
                )
            if last is not None and bar.ts_open <= last:
                raise ValueError(
                    f"Бары должны быть строго упорядочены и уникальны: {bar.ts_open} после {last}"
                )
            last = bar.ts_open


class _Params:
    """Типизированный доступ к параметрам спецификации (хранятся как dict)."""

    def __init__(self, raw: dict) -> None:
        self.rsi_period: int = int(raw["rsi_period"])
        self.macd_fast: int = int(raw["macd_fast"])
        self.macd_slow: int = int(raw["macd_slow"])
        self.macd_signal: int = int(raw["macd_signal"])
        self.atr_period: int = int(raw["atr_period"])
        self.ema_periods: list[int] = [int(v) for v in raw["ema_periods"]]
        self.return_lags: list[int] = [int(v) for v in raw["return_lags"]]
        self.vol_windows: list[int] = [int(v) for v in raw["vol_windows"]]
        self.volume_window: int = int(raw["volume_window"])
        self.volume_trend_window: int = int(raw["volume_trend_window"])
