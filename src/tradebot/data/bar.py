"""Контракт данных: Bar, Timeframe, BarSource (docs/architecture.md §3.2, §3.3).

Внутреннее представление свечи:
- ``ts_open`` — всегда timezone-aware, UTC, семантика "начало бара" (bar_open).
  Нормализация из bar_close и из локальных таймзон выполняется на границе загрузки
  (CSV-loader / Finam-адаптер), а не внутри ядра.
- цены — float (для ML-контура); в контуре заявок — только Decimal/строки
  (docs/decisions.md D1; конверсия выполняется в execution/, которого ещё нет).
- ``volume`` — целое число штук.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable

__all__ = [
    "SUPPORTED_TIMEFRAMES",
    "Bar",
    "BarSource",
    "Timeframe",
    "floor_to_grid",
]

UTC = timezone.utc

#: Таймфреймы из ТЗ (docs/architecture.md §3.3).
SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "10m", "15m", "1h")


class Timeframe(str, Enum):
    """Поддерживаемые таймфреймы. Значение — строковый код из ТЗ."""

    M1 = "1m"
    M5 = "5m"
    M10 = "10m"
    M15 = "15m"
    H1 = "1h"

    @classmethod
    def parse(cls, value: "str | Timeframe") -> "Timeframe":
        """Строгий парсер: только поддерживаемые таймфреймы, иначе ValueError."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"Таймфрейм должен быть строкой или Timeframe, получено: {type(value).__name__}"
            )
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            raise ValueError(
                f"Неподдерживаемый таймфрейм {value!r}. Поддерживаются: {', '.join(SUPPORTED_TIMEFRAMES)}"
            ) from None

    @property
    def minutes(self) -> int:
        return _TIMEFRAME_MINUTES[self.value]

    @property
    def duration(self) -> timedelta:
        return timedelta(minutes=self.minutes)


_TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "10m": 10, "15m": 15, "1h": 60}


class BarSource(str, Enum):
    """Источник бара.

    ``derived`` — бар, построенный локальным ресэмплингом (например 10m из 5m/1m),
    а не полученный из внешнего источника напрямую. Расширение enum из
    architecture §3.2 зафиксировано в docs/data.md.
    """

    CSV = "csv"
    FINAM_REST = "finam_rest"
    FINAM_WS = "finam_ws"
    DERIVED = "derived"


def floor_to_grid(ts: datetime, step_minutes: int) -> datetime:
    """Округление метки времени вниз до сетки шага (в UTC).

    Корректно для любого целочасового смещения локальной зоны (Europe/Moscow = UTC+3):
    часовые/15/10/5-минутные границы совпадают в UTC и локальном времени.
    """
    if ts.tzinfo is None:
        raise ValueError("floor_to_grid требует timezone-aware datetime (UTC)")
    if step_minutes <= 0:
        raise ValueError("step_minutes должен быть положительным")
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    total_minutes = int((ts.astimezone(UTC) - epoch).total_seconds() // 60)
    floored = total_minutes - (total_minutes % step_minutes)
    return epoch + timedelta(minutes=floored)


@dataclass(frozen=True)
class Bar:
    """Свеча OHLCV. Неизменяемая; hashable по значениям."""

    ts_open: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: BarSource = BarSource.CSV
    is_final: bool = True
    symbol: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ts_open, datetime):
            raise TypeError(f"ts_open должен быть datetime, получено {type(self.ts_open).__name__}")
        if self.ts_open.tzinfo is None:
            raise ValueError(
                "ts_open обязан быть timezone-aware (UTC). Naive-метки нормализуются "
                "на границе загрузки (см. docs/data.md)"
            )
        if self.ts_open.tzinfo != UTC:
            # Приводим к UTC, чтобы внутри ядра была одна зона.
            object.__setattr__(self, "ts_open", self.ts_open.astimezone(UTC))
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"Поле {name!r} должно быть числом, получено {type(value).__name__}")
            if not math.isfinite(value):
                raise ValueError(f"Поле {name!r} не конечно: {value!r}")
        if not isinstance(self.volume, int) or isinstance(self.volume, bool):
            raise TypeError(f"volume должен быть int (штуки), получено {type(self.volume).__name__}")
        if self.volume < 0:
            raise ValueError(f"volume не может быть отрицательным: {self.volume}")

    def ts_close(self, timeframe: "Timeframe | str") -> datetime:
        """Время окончания бара (ts_open + длительность таймфрейма)."""
        return self.ts_open + Timeframe.parse(timeframe).duration


def bars_are_sorted_unique(bars: Iterable[Bar]) -> bool:
    """Служебная проверка: метки строго возрастают и уникальны."""
    last: datetime | None = None
    for bar in bars:
        if last is not None and bar.ts_open <= last:
            return False
        last = bar.ts_open
    return True
