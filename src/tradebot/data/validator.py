"""Валидация данных: модель проблем и проверки последовательности баров.

Уровни:
- ERROR  — данные нельзя использовать (схема, типы, дубликаты, порядок, OHLC,
  вне сетки таймфрейма). Загрузка с ошибками должна падать с понятным сообщением.
- WARNING — данные использовать можно, но факт требует внимания (пропуски, нулевые
  объёмы, бары вне configured-сессий). Ничего не чинится молча (docs/architecture.md §3.1).

Проверки (этап D1, roadmap):
E_DUPLICATE_TS, E_UNSORTED, E_OHLC, E_OFF_GRID, E_VOLUME_NEGATIVE — структурные;
W_GAP (пропуски против сетки/календаря), W_ZERO_VOLUME, W_OUT_OF_SESSION — предупреждения.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence

from .bar import Bar, Timeframe, floor_to_grid
from .sessions import SessionCalendar

__all__ = [
    "IssueCode",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "validate_bars",
]


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class IssueCode(str, Enum):
    # errors
    E_EMPTY_FILE = "E_EMPTY_FILE"
    E_HEADER_MISSING_COLS = "E_HEADER_MISSING_COLS"
    E_HEADER_UNKNOWN_COLS = "E_HEADER_UNKNOWN_COLS"
    E_DATETIME_PARSE = "E_DATETIME_PARSE"
    E_NUMBER_PARSE = "E_NUMBER_PARSE"
    E_PRICE_INVALID = "E_PRICE_INVALID"
    E_VOLUME_INVALID = "E_VOLUME_INVALID"
    E_DUPLICATE_TS = "E_DUPLICATE_TS"
    E_UNSORTED = "E_UNSORTED"
    E_OHLC = "E_OHLC"
    E_OFF_GRID = "E_OFF_GRID"
    # warnings
    W_GAP = "W_GAP"
    W_ZERO_VOLUME = "W_ZERO_VOLUME"
    W_OUT_OF_SESSION = "W_OUT_OF_SESSION"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: IssueCode
    message: str
    line: int | None = None
    ts: datetime | None = None

    def format(self) -> str:
        where = []
        if self.line is not None:
            where.append(f"строка {self.line}")
        if self.ts is not None:
            where.append(f"ts={self.ts.isoformat()}")
        prefix = f"[{self.severity.value.upper()} {self.code.value}]"
        suffix = f" ({'; '.join(where)})" if where else ""
        return f"{prefix} {self.message}{suffix}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(
        self,
        severity: Severity,
        code: IssueCode,
        message: str,
        line: int | None = None,
        ts: datetime | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(severity, code, message, line, ts))

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    def summary_text(self, max_issues: int = 50) -> str:
        lines = [i.format() for i in self.issues[:max_issues]]
        if len(self.issues) > max_issues:
            lines.append(f"... и ещё {len(self.issues) - max_issues} проблем(ы)")
        header = (
            f"ОШИБОК: {len(self.errors)}, ПРЕДУПРЕЖДЕНИЙ: {len(self.warnings)}"
        )
        return "\n".join([header, *lines]) if lines else header


def validate_bars(
    bars: Sequence[Bar],
    timeframe: "Timeframe | str",
    calendar: SessionCalendar | None = None,
    lines: Sequence[int] | None = None,
    warn_zero_volume: bool = True,
) -> ValidationReport:
    """Структурные проверки последовательности баров одного таймфрейма.

    Args:
        bars: последовательность баров (ts_open ожидается в UTC, bar_open-семантика).
        timeframe: таймфрейм баров.
        calendar: необязательный календарь сессий (уточняет ожидаемые слоты и флаг
            "вне сессии"); при отсутствии календаря пропуски считаются по равномерной сетке.
        lines: необязательное соответствие бар -> номер строки в источнике (для CSV).
        warn_zero_volume: помечать ли бары с нулевым объёмом.
    """
    tf = Timeframe.parse(timeframe)
    report = ValidationReport()
    if lines is not None and len(lines) != len(bars):
        raise ValueError("lines должен соответствовать длине bars")

    def line_of(idx: int) -> int | None:
        return lines[idx] if lines is not None else None

    seen: dict[datetime, int] = {}
    prev: Bar | None = None
    for idx, bar in enumerate(bars):
        line = line_of(idx)
        # дубликаты
        if bar.ts_open in seen:
            report.add(
                Severity.ERROR,
                IssueCode.E_DUPLICATE_TS,
                f"Дублирующаяся метка времени (первая встреча: строка {line_of(seen[bar.ts_open])})",
                line=line,
                ts=bar.ts_open,
            )
        else:
            seen[bar.ts_open] = idx
        # порядок
        if prev is not None and bar.ts_open < prev.ts_open:
            report.add(
                Severity.ERROR,
                IssueCode.E_UNSORTED,
                f"Нарушен порядок времени: {bar.ts_open.isoformat()} после {prev.ts_open.isoformat()}",
                line=line,
                ts=bar.ts_open,
            )
        # OHLC-инварианты: low <= min(open, close), high >= max(open, close), low <= high
        ohlc_ok = (
            bar.high >= bar.low
            and bar.high >= bar.open
            and bar.high >= bar.close
            and bar.low <= bar.open
            and bar.low <= bar.close
        )
        if not ohlc_ok:
            report.add(
                Severity.ERROR,
                IssueCode.E_OHLC,
                "Нарушены соотношения OHLC: требуется low <= open/close <= high и low <= high "
                f"(o={bar.open}, h={bar.high}, l={bar.low}, c={bar.close})",
                line=line,
                ts=bar.ts_open,
            )
        # Отрицательный volume сюда попасть не может: Bar.__post_init__ его отклоняет;
        # в CSV-загрузчике такие значения ловятся как E_VOLUME_INVALID до создания Bar.
        if warn_zero_volume and bar.volume == 0:
            report.add(
                Severity.WARNING,
                IssueCode.W_ZERO_VOLUME,
                "Нулевой объём бара",
                line=line,
                ts=bar.ts_open,
            )
        # соответствие сетке таймфрейма (кратность N минут от начала часа UTC;
        # для целочасовых локальных зон эквивалентно локальной сетке)
        if floor_to_grid(bar.ts_open, tf.minutes) != bar.ts_open:
            report.add(
                Severity.ERROR,
                IssueCode.E_OFF_GRID,
                f"Метка не на сетке таймфрейма {tf.value} (ожидается кратность {tf.minutes} мин)",
                line=line,
                ts=bar.ts_open,
            )
        # пропуски и сессии
        if prev is not None:
            step = tf.duration
            gap_start = prev.ts_open + step
            gap_end = bar.ts_open
            if gap_end > gap_start:
                if calendar is not None:
                    missing = calendar.grid_slots(gap_start, gap_end, tf)
                    missing = [s for s in missing if calendar.is_trading_time(s)]
                else:
                    missing = []
                    slot = gap_start
                    while slot < gap_end:
                        missing.append(slot)
                        slot += step
                if missing:
                    report.add(
                        Severity.WARNING,
                        IssueCode.W_GAP,
                        f"Пропуск: {len(missing)} бар(ов) между {prev.ts_open.isoformat()} и {bar.ts_open.isoformat()}"
                        + (" (по календарю сессий)" if calendar is not None else " (по равномерной сетке; календарь сессий не задан)"),
                        line=line,
                        ts=gap_start,
                    )
        if calendar is not None and not calendar.is_trading_time(bar.ts_open):
            report.add(
                Severity.WARNING,
                IssueCode.W_OUT_OF_SESSION,
                "Бар вне настроенных сессионных окон",
                line=line,
                ts=bar.ts_open,
            )
        prev = bar
    return report
