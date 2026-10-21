"""Адаптер Finam REST-клиента к интерфейсам marketdata (docs/architecture.md §4.3).

Ключевые правила:
- нормализация Bar.timestamp из API к внутреннему bar_open требует ЯВНОГО указания
  семантики (``bar_timestamp_semantics``): документация её не фиксирует (research Q3),
  молчаливое угадывание запрещено (decisions D1). Значение по умолчанию отсутствует;
- 10m через API недоступен — ``fetch_history`` для "10m" бросает
  UnsupportedTimeframeError с инструкцией использовать локальный ресэмплинг (D3);
- Decimal -> float конверсия выполняется здесь (граница внешнего контура);
  volume обязан быть целым числом штук — нецелый объём отклоняется явно;
- бары с ``is_data_snapshot`` принимаются как обычные исторические значения
  (флаг документирован как признак снимка, research §2.4) и сохраняются в Bar.source=FINAM_REST.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from ..data.bar import Bar, BarSource, Timeframe
from ..marketdata.interfaces import (
    HistorySource,
    OrderBookSnapshot,
    OrderBookSnapshotSource,
    QuoteSnapshot,
    QuoteSnapshotSource,
)
from .errors import FinamInvalidResponseError, UnsupportedTimeframeError
from .rest_client import FinamRestClient

__all__ = ["FinamHistorySource", "FinamSnapshotSource", "client_from_config"]


def _decimal_to_bar_price(value: Decimal, field_name: str, timestamp: datetime) -> float:
    number = float(value)
    if not number == number or number in (float("inf"), float("-inf")):  # NaN/inf
        raise FinamInvalidResponseError(
            f"Поле {field_name!r} свечи {timestamp.isoformat()} не представимо как конечное число: {value!r}"
        )
    return number


def _decimal_to_bar_volume(value: Decimal, timestamp: datetime) -> int:
    if value != value.to_integral_value():
        raise FinamInvalidResponseError(
            f"Объём свечи {timestamp.isoformat()} не является целым числом штук: {value!r}. "
            "Внутренний контракт Bar.volume — int; дробные объёмы не угадываем и не округляем молча"
        )
    return int(value)


class FinamHistorySource(HistorySource):
    """История свечей из Finam REST API (read-only)."""

    def __init__(
        self,
        client: FinamRestClient,
        *,
        bar_timestamp_semantics: Literal["bar_open", "bar_close"] | None,
    ) -> None:
        if bar_timestamp_semantics not in ("bar_open", "bar_close"):
            raise ValueError(
                "bar_timestamp_semantics обязан быть явно задан ('bar_open' или 'bar_close'): "
                "семантика timestamp свечей API документацией не зафиксирована (research Q3). "
                "Задайте finam.bar_timestamp_semantics в конфиге или флаг CLI после эмпирической проверки"
            )
        self._client = client
        self._semantics = bar_timestamp_semantics

    def fetch_history(
        self,
        symbol: str,
        timeframe: "Timeframe | str",
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[Bar]:
        tf = Timeframe.parse(timeframe)
        if tf is Timeframe.M10:
            raise UnsupportedTimeframeError(
                "10m недоступен в Finam API (нет TIME_FRAME_M10, research §2.3). "
                "Загрузите 5m или 1m и постройте 10m локальным ресэмплингом через "
                "data.resampler.resample(...) (decisions D3)"
            )
        records = self._client.fetch_bars(symbol, tf, start_utc, end_utc)
        bars: list[Bar] = []
        for record in records:
            ts_open = record.timestamp
            if self._semantics == "bar_close":
                ts_open = ts_open - tf.duration
            bars.append(
                Bar(
                    ts_open=ts_open,
                    open=_decimal_to_bar_price(record.open, "open", record.timestamp),
                    high=_decimal_to_bar_price(record.high, "high", record.timestamp),
                    low=_decimal_to_bar_price(record.low, "low", record.timestamp),
                    close=_decimal_to_bar_price(record.close, "close", record.timestamp),
                    volume=_decimal_to_bar_volume(record.volume, record.timestamp),
                    source=BarSource.FINAM_REST,
                    is_final=True,
                    symbol=symbol,
                )
            )
        return bars


class FinamSnapshotSource(QuoteSnapshotSource, OrderBookSnapshotSource):
    """Снимки котировки и стакана (read-only; глубина стакана не гарантирована — Q4)."""

    def __init__(self, client: FinamRestClient) -> None:
        self._client = client

    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        return self._client.get_latest_quote(symbol)

    def get_order_book(self, symbol: str) -> OrderBookSnapshot:
        return self._client.get_order_book(symbol)


def client_from_config(
    config,  # FinamConfig (duck-typing, чтобы не тянуть цикл импортов)
    secret: str,
    *,
    transport=None,
) -> FinamRestClient:
    """Фабрика клиента из FinamConfig (секрет передаётся из окружения, не из конфига)."""
    return FinamRestClient(
        secret,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        rate_limit_per_minute=config.rate_limit_per_minute,
        max_get_attempts=config.max_get_attempts,
        transport=transport,
    )
