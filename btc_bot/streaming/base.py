"""Событийная модель потока рыночных данных (бирже-независимая).

Конкретный WS-клиент (Bybit) переводит сырые сообщения в эти события через
StreamProcessor; остальной код (сигналы, UI, журнал) работает только с ними.
Это же позволяет тестировать всю логику офлайн на реплее сообщений.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ConnState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class KlineBar:
    """Бар (закрытый или формирующийся)."""
    symbol: str
    interval: str
    start_ms: int
    end_ms: int          # момент закрытия (exclusive)
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    confirmed: bool      # True только для закрытой свечи (WS confirm=true / REST период истёк)
    server_ts_ms: int    # серверное время сообщения
    received_ms: int     # локальное время получения
    origin: str = "ws"   # ws | backfill | replay

    def as_feature_mapping(self) -> dict:
        return {
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume, "turnover": self.turnover,
        }


@dataclass(frozen=True)
class TickerUpdate:
    symbol: str
    price: float
    server_ts_ms: int
    received_ms: int


@dataclass(frozen=True)
class StateChange:
    state: ConnState
    detail: str
    received_ms: int


@dataclass(frozen=True)
class GapDetected:
    """Разрыв сетки закрытых баров (потеря соединения/пропуск биржи)."""
    expected_start_ms: int
    got_start_ms: int
    received_ms: int
    detail: str = ""


@dataclass(frozen=True)
class FreshnessChange:
    fresh: bool
    age_ms: int
    detail: str
    received_ms: int


@dataclass(frozen=True)
class ErrorEvent:
    detail: str
    received_ms: int
    fatal: bool = False


Event = KlineBar | TickerUpdate | StateChange | GapDetected | FreshnessChange | ErrorEvent


@dataclass
class StreamSnapshot:
    """Срез состояния потока для UI/двика."""
    conn_state: ConnState = ConnState.DISCONNECTED
    conn_detail: str = ""
    fresh: bool = False
    last_data_ms: Optional[int] = None
    age_ms: Optional[int] = None
    last_price: Optional[TickerUpdate] = None
    live_candle: Optional[KlineBar] = None
    confirmed_bars: list = field(default_factory=list)   # по возрастанию, последние в конце
    reconnect_count: int = 0
    gaps: list = field(default_factory=list)             # недавние GapDetected
    server_ts_ms: Optional[int] = None


class KlineStreamBase(ABC):
    """Интерфейс потокового источника (для будущих других бирж)."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...
