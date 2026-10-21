"""Интерфейсы источников рыночных данных (docs/architecture.md §2, §4.3).

Это бирженезависимые контракты (порты): будущий адаптер криптобиржи реализует
их без изменения потребителей. Реализация адаптера НЕ выполнена — подключение
биржи отложено (exchange/base.py, docs/live.md); сейчас единственный источник
данных — локальные CSV (CsvHistorySource, TEST-режим).

Границы:
- ``HistorySource`` — история свечей (CSV-replay в TEST; REST биржи — в будущем);
- ``QuoteSnapshotSource`` / ``OrderBookSnapshotSource`` — снимки котировки и
  стакана (для будущего адаптера; конкретные форматы ответов биржи здесь НЕ
  фиксируются — приведение к этим типам делает адаптер);
- ``StreamSource`` — контракт потоковых подписок (WebSocket биржи — будущий
  этап; транспорт выбирается за интерфейсом, docs/decisions.md D7).

Стакан: гарантированная глубина (число уровней) НЕ предполагается —
потребители не должны рассчитывать на фиксированную глубину.
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
    """Действие уровня стакана (в терминах дельта-потоков биржи)."""

    UNSPECIFIED = "unspecified"
    REMOVE = "remove"
    ADD = "add"
    UPDATE = "update"


@dataclass(frozen=True)
class OrderBookRow:
    price: Decimal
    size: Decimal
    side: OrderBookLevelSide
    action: OrderBookRowAction = OrderBookRowAction.UNSPECIFIED
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
    """Снимок последней котировки (бирженезависимый набор полей)."""

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
        """Свечи за [start_utc, end_utc): start включительно, end исключительно."""
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
    """Контракт потоковых подписок (реализация — будущий этап, см. docstring модуля).

    Ожидаемая семантика криптобиржевых потоков (учесть в будущем адаптере):
    - поток может повторно присылать свечу с тем же timestamp (мутация незакрытого
      бара); завершённой свеча считается только после появления бара с более
      поздним timestamp или явного флага закрытия — потребитель обязан применять
      BarFinalizer;
    - пропущенные во время обрыва события не доставляются: после переподключения
      требуется REST-снимок и сверка;
    - начальный снимок (snapshot) отличается от инкрементальных обновлений.
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
        source_timezone: str = "UTC",
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
