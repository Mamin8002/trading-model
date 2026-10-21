"""SessionCalendar — торговые сессии и часовой пояс (docs/architecture.md §3.3).

Принципы (криптовалютный режим, docs/data.md):
- Крипторынок работает КРУГЛОСУТОЧНО (24/7): базовое ожидание — непрерывная сетка
  баров в UTC. Для этого есть готовый календарь ``always_on_calendar()`` (окно
  00:00-24:00 во все дни недели) — рекомендованный календарь крипто-конфигов.
- Биржевые расписания (техобслуживания и пр.) мы НЕ хардкодим: если конкретная
  биржа/инструмент имеет окна торгов, календарь задаётся конфигурацией.
- Календарь может быть отключён (None): тогда полнота интервалов проверяется только
  по равномерной сетке, а разрывы сообщаются как warnings.
- Бары вне сессий не фабрикуются: календарь используется для (а) подсчёта ожидаемого
  числа слотов при ресэмплинге и (б) пометки баров вне сессий валидатором.

Поддерживаются сессионные окна внутри одних суток (start < end; конец "24:00"
разрешён и означает полночь, т.е. окно на все сутки). Время окон — локальное время
календаря (по умолчанию UTC — единая зона крипторынка).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from .bar import UTC, Timeframe, floor_to_grid

__all__ = ["DayRule", "SessionCalendar", "SessionWindow", "always_on_calendar"]

_HHMM_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_HHMM_END_RE = re.compile(r"^(?:([01]?\d|2[0-3]):([0-5]\d)|24:00)$")


@dataclass(frozen=True)
class SessionWindow:
    """Окно сессии в локальном времени календаря, start включительно, end исключительно.

    end может быть "24:00" (конец суток) — это позволяет выразить круглосуточное
    окно 00:00-24:00 для рынков 24/7 (криптовалюты).
    """

    start: timedelta
    end: timedelta

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError(
                "Сессионное окно должно удовлетворять start < end "
                f"(получено start={self.start}, end={self.end}); окна через полночь не поддерживаются"
            )

    @classmethod
    def parse(cls, start: str, end: str) -> "SessionWindow":
        return cls(_parse_hhmm(start, "start"), _parse_hhmm(end, "end", allow_24=True))


def _parse_hhmm(value: str, field: str, *, allow_24: bool = False) -> timedelta:
    pattern = _HHMM_END_RE if allow_24 else _HHMM_RE
    match = pattern.match(value.strip())
    if not match:
        hint = ', конец окна может быть "24:00"' if allow_24 else ""
        raise ValueError(f"Неверный формат времени {field}={value!r}, ожидается HH:MM (например 10:00){hint}")
    if allow_24 and match.group(1) is None:
        return timedelta(hours=24)
    hours, minutes = int(match.group(1)), int(match.group(2))
    return timedelta(hours=hours, minutes=minutes)


@dataclass(frozen=True)
class DayRule:
    """Правило для набора дней недели: 0=понедельник ... 6=воскресенье."""

    days: frozenset[int]
    windows: tuple[SessionWindow, ...]

    def __post_init__(self) -> None:
        for day in self.days:
            if not 0 <= day <= 6:
                raise ValueError(f"День недели вне диапазона 0..6: {day}")
        if not self.windows:
            raise ValueError("DayRule требует хотя бы одно сессионное окно")


class SessionCalendar:
    """Календарь сессий: отвечает на вопрос «является ли момент UTC торговым»."""

    def __init__(self, timezone: str, day_rules: Sequence[DayRule]) -> None:
        self.timezone = ZoneInfo(timezone)
        self._by_weekday: dict[int, tuple[SessionWindow, ...]] = {}
        for rule in day_rules:
            for day in rule.days:
                existing = self._by_weekday.get(day, ())
                self._by_weekday[day] = tuple(sorted(existing + rule.windows, key=lambda w: w.start))
        if not self._by_weekday:
            raise ValueError("Календарь сессий пуст: не задано ни одного дня")

    @classmethod
    def from_spec(
        cls,
        timezone: str,
        rules: Iterable[tuple[Iterable[int], Iterable[tuple[str, str]]]],
    ) -> "SessionCalendar":
        """Сборка из простой структуры: [(days, [(start_hhmm, end_hhmm), ...]), ...]."""
        day_rules = [
            DayRule(frozenset(days), tuple(SessionWindow.parse(s, e) for s, e in windows))
            for days, windows in rules
        ]
        return cls(timezone, day_rules)

    def to_local(self, ts_utc: datetime) -> datetime:
        if ts_utc.tzinfo is None:
            raise ValueError("Ожидается timezone-aware datetime (UTC)")
        return ts_utc.astimezone(self.timezone)

    def is_trading_time(self, ts_utc: datetime) -> bool:
        """True, если момент UTC попадает в одно из окон сессии (start <= t < end)."""
        local = self.to_local(ts_utc)
        midnight = timedelta(hours=local.hour, minutes=local.minute, seconds=local.second, microseconds=local.microsecond)
        for window in self._by_weekday.get(local.weekday(), ()):
            if window.start <= midnight < window.end:
                return True
        return False

    def grid_slots(
        self,
        start_utc: datetime,
        end_utc: datetime,
        step: "Timeframe | str",
    ) -> list[datetime]:
        """Все слоты сетки шага в [start_utc, end_utc) (start округляется вниз)."""
        step_minutes = Timeframe.parse(step).minutes
        slot = floor_to_grid(start_utc, step_minutes)
        out: list[datetime] = []
        while slot < end_utc:
            if slot >= start_utc:
                out.append(slot)
            slot += timedelta(minutes=step_minutes)
        return out

    def trading_slots(
        self,
        start_utc: datetime,
        end_utc: datetime,
        step: "Timeframe | str",
    ) -> list[datetime]:
        """Слоты сетки, являющиеся торговым временем (используется для ожидаемого покрытия)."""
        return [slot for slot in self.grid_slots(start_utc, end_utc, step) if self.is_trading_time(slot)]


def always_on_calendar(timezone: str = "UTC") -> SessionCalendar:
    """Календарь 24/7 — рекомендованный для крипторынка (торги без выходных).

    Окно 00:00-24:00 во все дни недели: любой момент является «торговым временем»,
    поэтому ожидаемое покрытие при валидации/ресэмплинге — непрерывная сетка UTC.
    """
    rule = DayRule(frozenset(range(7)), (SessionWindow(timedelta(0), timedelta(hours=24)),))
    return SessionCalendar(timezone, (rule,))
