"""Интерфейсы источников рыночных данных (docs/architecture.md §2, §4.3).

Границы:
- ``HistorySource`` — история свечей (CSV-replay в TEST, Finam REST в LIVE);
- ``QuoteSnapshotSource`` / ``OrderBookSnapshotSource`` — снимки котировки и стакана
  (REST-методы документированы и подтверждены: docs/research.md §2.4, §2.5);
- ``StreamSource`` — контракт потоковых подписок. Документацией подтверждены
  WebSocket (wss://api.finam.ru/ws, SUBSCRIBE после HANDSHAKE_SUCCESS, конверты
  DATA/ERROR/EVENT; AsyncAPI v1.0.0) и gRPC-стримы SDK (SubscribeBars/SubscribeQuote/
  SubscribeOrderBook/SubscribeLatestTrades). РЕАЛИЗАЦИЯ потока — этап L2 roadmap:
  в этой фазе определён только интерфейс, чтобы транспорт выбирался позже без
  изменения потребителей (decisions D7: транспорт за интерфейсом).

Стакан: наличие и формат снимка ПОДТВЕРЖДЕНЫ документацией
(GET /v1/instruments/{symbol}/orderbook; rows: price, buy_size|sell_size,
action ADD/REMOVE/UPDATE, mpid, timestamp — research §2.5), поэтому снимок
реализован. Гарантированная глубина (число уровней) документацией НЕ задана
(research Q4) — потребители не должны рассчитывать на фиксированную глубину.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Iterator, Sequence

from ..data.bar import Bar, Timeframe

__all__ = [
    "CsvHistorySource",
    "HistorySource",
    "OrderBookLevelSide",
    "OrderBookRow",
    "OrderBookRowAction",
    "OrderBookSnapshot",
    "QuoteSnapshot",
    "QuoteSnapshotSource",
    "OrderBookSnapshotSource",
    "StreamSource",
]


class OrderBookLevelSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderBookRowAction(str, Enum):
    """Действие уровня стакана (подтверждено proto: ACTION_ADD/REMOVE/UPDATE)."""

    UNSPECIFIED = "ACTION_UNSPECIFIED"
    REMOVE = "ACTION_REMOVE"
    ADD = "ACTION_ADD"
    UPDATE = "ACTION_UPDATE"


@dataclass(frozen=True)
class OrderBookRow:
    price: Decimal
    size: Decimal
    side: OrderBookLevelSide
    action: OrderBookRowAction = OrderBookRowAction.UNSPECIFIED
    mpid: str | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    rows: tuple[OrderBookRow, ...]
    received_at: datetime

    @property
    def depth(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class QuoteSnapshot:
    """Снимок последней котировки (поля подтверждены research §2.4/S3e и proto Quote)."""

    symbol: str
    timestamp: datetime | None
    last: Decimal | None
    bid: Decimal | None
    ask: Decimal | None
    bid_size: Decimal | None
    ask_size: Decimal | None
    volume: Decimal | None


class HistorySource(ABC):
    """Источник исторических свечей. Возвращает финальные бары, bar_open, UTC."""

    @abstractmethod
    def fetch_history(
        self,
        symbol: str,
        timeframe: "Timeframe | str",
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[Bar]:
        """Свечи за [start_utc, end_utc) — границы как в документированном API
        (start включительно, end исключительно; research §2.3)."""
        raise NotImplementedError


class QuoteSnapshotSource(ABC):
    @abstractmethod
    def get_latest_quote(self, symbol: str) -> QuoteSnapshot:
        raise NotImplementedError


class OrderBookSnapshotSource(ABC):
    @abstractmethod
    def get_order_book(self, symbol: str) -> OrderBookSnapshot:
        raise NotImplementedError


class StreamSource(ABC):
    """Контракт потоковых подписок (реализация — этап L2, см. docstring модуля).

    Семантика (подтверждена документацией, research §2.4):
    - поток может повторно присылать свечу с тем же timestamp (мутация незакрытого
      бара); завершённой свеча считается только после появления бара с более
      поздним timestamp — потребитель обязан применять BarFinalizer (этап L2);
    - пропущенные во время обрыва события не доставляются: после переподключения
      требуется REST-снимок и сверка;
    - ``is_data_snapshot`` отличает начальный снимок от инкрементальных обновлений.
    """

    @abstractmethod
    def subscribe_bars(self, symbol: str, timeframe: "Timeframe | str") -> Iterator[Bar]:
        raise NotImplementedError

    @abstractmethod
    def subscribe_quotes(self, symbols: Sequence[str]) -> Iterator[QuoteSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class CsvHistorySource(HistorySource):
    """TEST-источник: история из локальных CSV (детерминированный replay).

    Загружает файлы лениво и кэширует результат; окна фильтрует по
    [start_utc, end_utc). Загрузка и валидация — через data.csv_loader.load_csv.
    """

    def __init__(
        self,
        files_by_timeframe: dict[str, "str | object"],
        *,
        source_timezone: str = "Europe/Moscow",
        timestamp_semantics: "str" = "bar_open",
        symbol: str = "",
        calendar: "object | None" = None,
    ) -> None:
        from ..data.csv_loader import load_csv  # локальный импорт против цикла

        self._load_csv = load_csv
        self._files = dict(files_by_timeframe)
        self._source_timezone = source_timezone
        self._timestamp_semantics = timestamp_semantics
        self._symbol = symbol
        self._calendar = calendar
        self._cache: dict[str, list[Bar]] = {}

    def _bars(self, timeframe: Timeframe) -> list[Bar]:
        key = timeframe.value
        if key not in self._cache:
            path = self._files.get(key)
            if path is None:
                raise FileNotFoundError(
                    f"Для таймфрейма {key} не указан CSV-файл (data.symbols[].files)"
                )
            loaded = self._load_csv(
                path,
                timeframe=timeframe,
                symbol=self._symbol,
                source_timezone=self._source_timezone,
                timestamp_semantics=self._timestamp_semantics,  # type: ignore[arg-type]
                calendar=self._calendar,  # type: ignore[arg-type]
            )
            self._cache[key] = loaded.bars
        return self._cache[key]

    def fetch_history(
        self,
        symbol: str,
        timeframe: "Timeframe | str",
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[Bar]:
        tf = Timeframe.parse(timeframe)
        if symbol and self._symbol and symbol != self._symbol:
            raise ValueError(
                f"CsvHistorySource настроен на символ {self._symbol!r}, запрошен {symbol!r}"
            )
        return [b for b in self._bars(tf) if start_utc <= b.ts_open < end_utc]
