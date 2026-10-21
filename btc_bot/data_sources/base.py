"""Общий интерфейс источника рыночных данных (для будущей поддержки других бирж).

Любая биржа подключается реализацией MarketDataSource; остальной код бота
(данные, модель, сигналы, UI) от конкретной биржи не зависит.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Kline:
    """Одна свеча (бар). Все цены - float, время - unix ms (UTC)."""

    start_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    is_closed: bool  # True, если свеча закрыта (confirm), False - ещё формируется


@dataclass(frozen=True)
class LastPrice:
    price: float
    ts_ms: int      # серверное время котировки (unix ms UTC)
    source: str     # имя источника ("bybit")
    symbol: str


class DataSourceError(RuntimeError):
    """Базовая ошибка источника данных."""


class DataSourceNetworkError(DataSourceError):
    """Сетевая ошибка / недоступность источника."""


class DataSourceAPIError(DataSourceError):
    """API вернул ошибку (retCode != 0, HTTP != 200 и т.п.)."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


ProgressCallback = Callable[[int, int, Optional[int]], None]
"""callback(n_pages_done, n_rows_total, last_start_ms)"""


class MarketDataSource(ABC):
    """Интерфейс публичного источника рыночных данных.

    Реализации обязаны:
      - возвращать свечи в порядке возрастания start_ms, без дубликатов;
      - корректно помечать незакрытую свечу (is_closed=False);
      - не фабриковать данные: пропуски баров остаются пропусками.
    """

    name: str = "abstract"

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str,
        category: str,
        interval: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
        max_pages: Optional[int] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> list[Kline]:
        """Исторические свечи через REST с пагинацией. Результат - по возрастанию start_ms."""

    @abstractmethod
    def get_last_price(self, symbol: str, category: str) -> LastPrice:
        """Последняя цена (для spot Bybit - REST tickers)."""

    @abstractmethod
    def get_server_time_ms(self) -> int:
        """Текущее серверное время источника в unix ms (для контроля часов/свежести)."""

    def describe(self) -> dict:
        return {"source": self.name}
