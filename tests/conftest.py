"""Общие фикстуры и тестовые утилиты (без сети: проект работает офлайн на CSV)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.data.bar import UTC, Bar, BarSource


def make_bar(
    ts_open: datetime,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 1000,
    *,
    source: BarSource = BarSource.CSV,
    is_final: bool = True,
    symbol: str = "TEST-USDT",
) -> Bar:
    if ts_open.tzinfo is None:
        ts_open = ts_open.replace(tzinfo=UTC)
    return Bar(
        ts_open=ts_open.astimezone(UTC),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source=source,
        is_final=is_final,
        symbol=symbol,
    )


def minute_series(
    start: datetime,
    count: int,
    *,
    step_minutes: int = 1,
    open0: float = 100.0,
    d_open: float = 0.01,
    symbol: str = "TEST-USDT",
    skip_indexes: set[int] | None = None,
) -> list[Bar]:
    """Детерминированная серия баров: open растёт линейно, high/low ±0.05, volume 1000+10i."""
    skip = skip_indexes or set()
    bars: list[Bar] = []
    for i in range(count):
        if i in skip:
            continue
        ts = start + timedelta(minutes=step_minutes * i)
        o = round(open0 + d_open * i, 6)
        bars.append(
            make_bar(
                ts,
                open_=o,
                high=o + 0.05,
                low=o - 0.05,
                close=o + 0.02,
                volume=1000 + 10 * i,
                symbol=symbol,
            )
        )
    return bars


def write_csv(path: Path, rows: list[str], header: str = "datetime,open,high,low,close,volume") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def utc_now() -> datetime:
    return datetime(2026, 10, 21, 12, 0, 0, tzinfo=timezone.utc)
