"""Загрузчик исторических CSV формата MOEX-выгрузок (docs/data.md).

Формат (подтверждён контрактом ТЗ и прецедентом WISEPLAT, docs/research.md S9h):

    datetime,open,high,low,close,volume
    2021-01-04 10:00:00,274.67,274.67,274.67,274.67,123460

Правила:
- колонки — ровно требуемый набор (порядок колонок не важен);
- ``datetime`` — naive-метка в локальной таймзоне источника (по умолчанию
  Europe/Moscow) либо ISO-метка со смещением; семантика метки задаётся ЯВНО
  (``timestamp_semantics``: bar_open | bar_close) и нормализуется к bar_open;
- числа — только точка как десятичный разделитель, без разделителей тысяч;
- volume — целое неотрицательное число штук;
- результат — список ``Bar`` (UTC, bar_open) + ``ValidationReport``;
  при наличии ошибок загрузка бросает ``DataValidationError`` (никакой тихой починки:
  данные не сортируются и не дедуплицируются молча — ошибки нужно исправить в источнике).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from .bar import UTC, Bar, BarSource, Timeframe
from .sessions import SessionCalendar
from .validator import (
    IssueCode,
    Severity,
    ValidationReport,
    validate_bars,
)

__all__ = [
    "CsvLoadResult",
    "DataValidationError",
    "REQUIRED_COLUMNS",
    "load_csv",
]

REQUIRED_COLUMNS: tuple[str, ...] = ("datetime", "open", "high", "low", "close", "volume")

_PRICE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_VOLUME_RE = re.compile(r"^[+-]?\d+(?:\.0*)?$")


class _PriceNotPositiveError(ValueError):
    """Цена <= 0 — отдельный код ошибки (E_PRICE_INVALID), не путать с нечисловой."""


class DataValidationError(Exception):
    """CSV не прошёл валидацию. Содержит полный отчёт (report)."""

    def __init__(self, report: ValidationReport, source: str) -> None:
        self.report = report
        self.source = source
        summary = report.summary_text(max_issues=20)
        super().__init__(f"Файл {source} не прошёл проверку данных:\n{summary}")


@dataclass
class CsvLoadResult:
    bars: list[Bar] = field(default_factory=list)
    report: ValidationReport = field(default_factory=ValidationReport)
    source_path: str = ""

    @property
    def ok(self) -> bool:
        return self.report.ok


def _parse_price(raw: str) -> float:
    value = raw.strip()
    if not _PRICE_RE.match(value):
        raise ValueError(
            f"нечисловое или недопустимое значение цены {raw!r} "
            "(разрешены только цифры, одна точка и экспонента; запятая и разделители тысяч запрещены)"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"значение цены не конечно: {raw!r}")
    if number <= 0:
        raise _PriceNotPositiveError(f"цена должна быть положительной, получено {raw!r}")
    return number


def _parse_volume(raw: str) -> int:
    value = raw.strip()
    if not _VOLUME_RE.match(value):
        raise ValueError(
            f"объём должен быть целым неотрицательным числом штук, получено {raw!r}"
        )
    number = int(float(value))
    if number < 0:
        raise ValueError(f"объём не может быть отрицательным: {raw!r}")
    return number


def _parse_datetime(
    raw: str,
    source_tz: ZoneInfo,
) -> datetime:
    value = raw.strip()
    if not value:
        raise ValueError("пустая метка времени")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"не удалось разобрать метку времени {raw!r} "
            "(ожидаются форматы 'YYYY-MM-DD HH:MM:SS' или ISO-8601, опционально со смещением)"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=source_tz)
    return parsed.astimezone(UTC)


def load_csv(
    path: "str | Path",
    *,
    timeframe: "Timeframe | str",
    symbol: str = "",
    source_timezone: str = "Europe/Moscow",
    timestamp_semantics: Literal["bar_open", "bar_close"] = "bar_open",
    calendar: SessionCalendar | None = None,
    warn_zero_volume: bool = True,
) -> CsvLoadResult:
    """Читает и валидирует CSV. Бросает DataValidationError при ошибках уровня ERROR."""
    tf = Timeframe.parse(timeframe)
    if timestamp_semantics not in ("bar_open", "bar_close"):
        raise ValueError(
            f"timestamp_semantics должен быть 'bar_open' или 'bar_close', получено {timestamp_semantics!r}"
        )
    source_path = str(path)
    file_path = Path(path)
    report = ValidationReport()
    result = CsvLoadResult(source_path=source_path, report=report)

    if not file_path.exists():
        report.add(Severity.ERROR, IssueCode.E_EMPTY_FILE, f"Файл не найден: {source_path}")
        raise DataValidationError(report, source_path)

    raw_bytes = file_path.read_bytes()
    if not raw_bytes.strip():
        report.add(Severity.ERROR, IssueCode.E_EMPTY_FILE, "Файл пуст (0 байт или только пробелы)")
        raise DataValidationError(report, source_path)

    try:
        source_tz = ZoneInfo(source_timezone)
    except Exception as exc:  # zoneinfo может бросить ZoneInfoNotFoundError
        raise ValueError(f"Неизвестная таймзона источника {source_timezone!r}: {exc}") from exc

    text = raw_bytes.decode("utf-8-sig", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.reader(text.splitlines())
    try:
        header = next(reader)
    except StopIteration:
        report.add(Severity.ERROR, IssueCode.E_HEADER_MISSING_COLS, "Файл не содержит заголовка")
        raise DataValidationError(report, source_path)

    header_clean = [h.strip().lstrip("\ufeff").lower() for h in header]
    missing = [c for c in REQUIRED_COLUMNS if c not in header_clean]
    unknown = [c for c in header_clean if c not in REQUIRED_COLUMNS]
    if missing:
        report.add(
            Severity.ERROR,
            IssueCode.E_HEADER_MISSING_COLS,
            f"В заголовке отсутствуют обязательные колонки: {', '.join(missing)} "
            f"(требуется ровно: {', '.join(REQUIRED_COLUMNS)})",
            line=1,
        )
    if unknown:
        report.add(
            Severity.ERROR,
            IssueCode.E_HEADER_UNKNOWN_COLS,
            f"Неизвестные колонки в заголовке: {', '.join(unknown)} "
            f"(требуется ровно: {', '.join(REQUIRED_COLUMNS)})",
            line=1,
        )
    if missing or unknown:
        raise DataValidationError(report, source_path)

    col_index = {name: header_clean.index(name) for name in REQUIRED_COLUMNS}
    n_cols = len(header_clean)

    bars: list[Bar] = []
    row_lines: list[int] = []
    line_no = 1  # заголовок
    for row in reader:
        line_no += 1
        if not row or (len(row) == 1 and not row[0].strip()):
            continue  # пустая строка — пропускаем, но не молча в данных: см. ниже
        if len(row) != n_cols:
            report.add(
                Severity.ERROR,
                IssueCode.E_NUMBER_PARSE,
                f"В строке {len(row)} полей, ожидается {n_cols}",
                line=line_no,
            )
            continue

        ts_raw = row[col_index["datetime"]]
        try:
            ts_utc = _parse_datetime(ts_raw, source_tz)
        except ValueError as exc:
            report.add(Severity.ERROR, IssueCode.E_DATETIME_PARSE, str(exc), line=line_no)
            continue
        if timestamp_semantics == "bar_close":
            ts_utc = ts_utc - tf.duration

        prices: dict[str, float] = {}
        row_failed = False
        for name in ("open", "high", "low", "close"):
            try:
                prices[name] = _parse_price(row[col_index[name]])
            except _PriceNotPositiveError as exc:
                report.add(Severity.ERROR, IssueCode.E_PRICE_INVALID, f"Колонка {name!r}: {exc}", line=line_no)
                row_failed = True
            except ValueError as exc:
                report.add(Severity.ERROR, IssueCode.E_NUMBER_PARSE, f"Колонка {name!r}: {exc}", line=line_no)
                row_failed = True
        try:
            volume = _parse_volume(row[col_index["volume"]])
        except ValueError as exc:
            report.add(Severity.ERROR, IssueCode.E_VOLUME_INVALID, f"Колонка 'volume': {exc}", line=line_no)
            row_failed = True
        if row_failed:
            continue

        bars.append(
            Bar(
                ts_open=ts_utc,
                open=prices["open"],
                high=prices["high"],
                low=prices["low"],
                close=prices["close"],
                volume=volume,
                source=BarSource.CSV,
                is_final=True,
                symbol=symbol,
            )
        )
        row_lines.append(line_no)

    if not bars and report.ok:
        report.add(
            Severity.ERROR,
            IssueCode.E_EMPTY_FILE,
            "Файл содержит заголовок, но не содержит ни одной валидной строки данных",
        )

    if report.ok:
        # структурные проверки (порядок, дубликаты, OHLC, сетка, пропуски, сессии)
        report.extend(
            validate_bars(
                bars,
                tf,
                calendar=calendar,
                lines=row_lines,
                warn_zero_volume=warn_zero_volume,
            )
        )

    if not report.ok:
        raise DataValidationError(report, source_path)

    result.bars = bars
    return result
