"""Генератор СИНТЕТИЧЕСКИХ демонстрационных баров для тестов и ручных проверок.

ВНИМАНИЕ: это НЕ рыночные данные и не замена реальной истории. Генератор нужен
только для детерминированных проверок пайплайна (причинность признаков, сплиты,
обучение) без скачивания внешних данных. Все серии воспроизводимы по seed.

Модель генерации (упрощённая, для тестов):
- лог-доходности: смена режимов дрейфа каждые ~regime_len баров + AR(1)-волатильность
  (кластеризация), всё из numpy RandomState(seed);
- OHLC: open = close предыдущего бара × (1 + маленький гэп); high/low достраиваются
  вокруг тела так, чтобы инварианты low <= open/close <= high выполнялись строго;
- volume: базовый уровень × (1 + k·|r|) × логнормальный шум, целые штуки;
- временные метки: минутная сетка, только пн-пт, окно 08:00-16:00 UTC
  (условная «сессия»; календарь демо-режима, не расписание MOEX).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from .bar import UTC, Bar, BarSource, Timeframe

__all__ = ["generate_synthetic_bars", "trading_minute_grid"]


def trading_minute_grid(start: datetime, count: int, step_minutes: int = 1) -> list[datetime]:
    """Минутная сетка пн-пт 08:00-15:59 UTC (окно [08:00, 16:00)).

    Бары ставятся на слоты, кратные step_minutes от 08:00, начиная с первого слота
    не раньше start. Это условный демо-календарь (не расписание MOEX).
    """
    if start.tzinfo is None:
        raise ValueError("start должен быть timezone-aware")
    if step_minutes < 1 or 60 % step_minutes != 0:
        raise ValueError("step_minutes должен делить 60 (1, 2, 3, 4, 5, 6, 10, ...)")
    start_utc = start.astimezone(UTC)
    slots_per_day = (8 * 60) // step_minutes  # окно 8 часов
    slots: list[datetime] = []
    day = start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    guard_days = count // max(slots_per_day, 1) + 400
    for _ in range(guard_days):
        if len(slots) >= count:
            break
        if day.weekday() < 5:
            base = day.replace(hour=8)
            for k in range(slots_per_day):
                ts = base + timedelta(minutes=k * step_minutes)
                if ts >= start_utc:
                    slots.append(ts)
                    if len(slots) >= count:
                        break
        day += timedelta(days=1)
    if len(slots) < count:
        raise RuntimeError("trading_minute_grid: не хватило горизонта для генерации сетки")
    return slots[:count]


def generate_synthetic_bars(
    n_bars: int,
    *,
    seed: int = 12345,
    start: datetime | None = None,
    timeframe: "Timeframe | str" = "5m",
    symbol: str = "SYNTH@DEMO",
    start_price: float = 100.0,
    regime_len: int = 500,
) -> list[Bar]:
    """Детерминированные синтетические OHLCV-бары (см. предупреждение в docstring модуля)."""
    tf = Timeframe.parse(timeframe)
    if n_bars < 10:
        raise ValueError("n_bars должен быть >= 10")
    if start is None:
        start = datetime(2024, 1, 1, 8, 0, tzinfo=UTC)
    rng = np.random.RandomState(seed)

    timestamps = trading_minute_grid(start, n_bars, step_minutes=tf.minutes)

    # режимы дрейфа и AR(1)-волатильность
    n_regimes = max(1, n_bars // regime_len + 1)
    drifts = rng.uniform(-0.0004, 0.0004, size=n_regimes)
    regime_index = np.minimum(np.arange(n_bars) // regime_len, n_regimes - 1)
    drift = drifts[regime_index]

    sigma = np.empty(n_bars)
    sigma[0] = 0.0015
    shock = rng.normal(0.0, 0.0006, size=n_bars)
    for i in range(1, n_bars):
        sigma[i] = max(0.0004, 0.92 * sigma[i - 1] + 0.08 * 0.0015 + shock[i])
    returns = drift + sigma * rng.normal(0.0, 1.0, size=n_bars)

    closes = start_price * np.exp(np.cumsum(returns))
    opens = np.empty(n_bars)
    opens[0] = start_price
    gap = rng.normal(0.0, 0.0004, size=n_bars)
    opens[1:] = closes[:-1] * (1.0 + gap[1:])

    body_hi = np.maximum(opens, closes)
    body_lo = np.minimum(opens, closes)
    wick_up = np.abs(rng.normal(0.0, 1.0, size=n_bars)) * sigma * closes * 0.8
    wick_dn = np.abs(rng.normal(0.0, 1.0, size=n_bars)) * sigma * closes * 0.8
    highs = body_hi + wick_up
    lows = body_lo - wick_dn

    base_volume = 50_000
    volumes = np.rint(
        base_volume * (1.0 + 40.0 * np.abs(returns)) * np.exp(rng.normal(0.0, 0.25, size=n_bars))
    ).astype(np.int64)
    volumes = np.maximum(volumes, 1)

    bars: list[Bar] = []
    for i in range(n_bars):
        bars.append(
            Bar(
                ts_open=timestamps[i],
                open=float(round(opens[i], 4)),
                high=float(round(highs[i], 4)),
                low=float(round(lows[i], 4)),
                close=float(round(closes[i], 4)),
                volume=int(volumes[i]),
                source=BarSource.CSV,  # помечаем как "csv" — источник значения не меняет
                is_final=True,
                symbol=symbol,
            )
        )
    return bars
