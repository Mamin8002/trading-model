"""Общие фикстуры и тестовые утилиты.

ВАЖНО: все значения токенов/счетов здесь — фиктивные строки для тестов,
не имеющие отношения к реальным учётным данным.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from tradebot.data.bar import UTC, Bar, BarSource
from tradebot.finam.transport import HttpResponse, HttpTransport


def make_bar(
    ts_open: datetime,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: int = 1000,
    *,
    source: BarSource = BarSource.CSV,
    is_final: bool = True,
    symbol: str = "TEST@TEST",
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
    symbol: str = "TEST@TEST",
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


@dataclass
class RecordedRequest:
    method: str
    url: str
    headers: dict[str, str]
    json_body: dict[str, Any] | None


@dataclass
class FakeTransport(HttpTransport):
    """Подменный транспорт для тестов адаптера Finam (без сети).

    ``responder(method, url, call_index) -> HttpResponse``. По умолчанию отдаёт
    запрограммированную очередь ответов, затем повторяет последний.
    Все запросы записываются в ``requests`` для проверок (заголовки, параметры).
    """

    responses: list[HttpResponse] = field(default_factory=list)
    responder: Callable[[str, str, int], HttpResponse] | None = None
    requests: list[RecordedRequest] = field(default_factory=list)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 15.0,
    ) -> HttpResponse:
        self.requests.append(RecordedRequest(method, url, dict(headers or {}), json_body))
        index = len(self.requests) - 1
        if self.responder is not None:
            return self.responder(method, url, index)
        if not self.responses:
            return HttpResponse(status=500, body=b'{"code":13,"message":"no fake response programmed"}')
        return self.responses[min(index, len(self.responses) - 1)]


def json_response(status: int, payload: Any) -> HttpResponse:
    return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))


def decimal(value: str) -> dict[str, str]:
    """Формат decimal REST-ответов Finam (research §2.7)."""
    return {"value": value}


def make_bars_payload(symbol: str, bars: list[dict[str, Any]]) -> dict[str, Any]:
    return {"symbol": symbol, "bars": bars}


def bar_payload(
    ts: str,
    open_: str,
    high: str,
    low: str,
    close: str,
    volume: str,
    is_data_snapshot: bool = False,
) -> dict[str, Any]:
    return {
        "timestamp": ts,
        "open": decimal(open_),
        "high": decimal(high),
        "low": decimal(low),
        "close": decimal(close),
        "volume": decimal(volume),
        "is_data_snapshot": is_data_snapshot,
    }


FAKE_SECRET = "FAKE-SECRET-DO-NOT-USE-000"
FAKE_ACCESS_KEY = "FAKE-ACCESS-KEY-DO-NOT-USE-000"
FAKE_ACCOUNT_ID = "FAKEACCOUNT000"


@pytest.fixture()
def fake_sleep_calls() -> list[float]:
    return []


@pytest.fixture()
def fake_sleep(fake_sleep_calls):
    def _sleep(seconds: float) -> None:
        fake_sleep_calls.append(seconds)

    return _sleep


@pytest.fixture()
def utc_now() -> datetime:
    return datetime(2026, 10, 21, 12, 0, 0, tzinfo=timezone.utc)
