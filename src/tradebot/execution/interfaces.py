"""Контракты исполнительного контура фазы 4 (docs/simulation.md).

Реализован ТОЛЬКО TEST-симулятор: реальные заявки не выставляются и
подтверждения от биржи не имитируются. Типы спроектированы так, чтобы будущий
LIVE-шлюз криптобиржи (НЕ реализован — exchange/base.py) переиспользовал те же
OrderIntent/RiskDecision/Fill.

ВАЖНО про цены (ограничение ТЗ фазы 4):
- ``OrderIntent.reference_price`` — цена закрытия сигнальной свечи. Это ТОЛЬКО
  справочная метка для журнала; она НЕ используется как цена исполнения;
- фактическое исполнение в симуляторе — не раньше следующей доступной цены
  (open следующего бара ± явно настроенное проскальзывание), см. ``Fill``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

__all__ = [
    "BrokerGateway",
    "BrokerPosition",
    "ExitCause",
    "Fill",
    "HaltEvent",
    "OrderIntent",
    "OrderSubmission",
    "OrderSubmissionStatus",
    "PortfolioState",
    "ReconciliationMismatch",
    "RejectionCode",
    "RiskContext",
    "RiskDecision",
    "RiskRejection",
    "Side",
    "Signal",
]


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class ExitCause(str, Enum):
    """Причина выхода из позиции (аудит журнала)."""

    SIGNAL = "exit_signal"          # встречный/закрывающий сигнал модели
    STOP = "stop"                   # стоп исполнен в пределах бара
    TARGET = "target"               # цель исполнена
    STOP_TARGET_CONFLICT = "stop_target_conflict"  # обе границы в одном баре -> консервативное правило
    GAP_THROUGH_STOP = "gap_through_stop"          # открытие хуже стопа -> исполнение по open
    END_OF_DATA = "end_of_data"     # НЕ используется для фабрикации сделки: позиция остаётся открытой


class RejectionCode(str, Enum):
    """Коды отказов риск-контроля и симулятора (единый словарь для журнала)."""

    RISK_PARAM_NOT_SET = "risk_param_not_set"        # параметр не задан конфигом -> безопасный отказ
    SIZING_STOP_REQUIRED = "sizing_stop_required"    # рисковый сайзинг без стоп-цены в сигнале
    MAX_POSITION = "max_position_exceeded"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    DAILY_LOSS = "daily_loss_limit"
    DRAWDOWN = "drawdown_limit"
    ORDER_LIMIT = "order_count_limit"
    SHORT_NOT_ALLOWED = "short_not_allowed"
    NO_POSITION_TO_CLOSE = "no_position_to_close"
    STALE_DATA = "stale_data"
    INVALID_DATA = "invalid_data"
    POSITION_DESYNC = "position_inconsistency"
    HALTED = "strategy_halted"
    INVALID_LOT = "invalid_lot_multiple"
    INVALID_PRICE_STEP = "invalid_price_step"
    INVALID_STOP_TARGET = "invalid_stop_target"
    NO_NEXT_BAR = "no_next_bar"
    INSTRUMENT_PARAM_NOT_SET = "instrument_param_not_set"


@dataclass(frozen=True)
class RiskRejection:
    code: RejectionCode
    message: str
    limit: str | None = None     # имя лимита/параметра
    limit_value: float | None = None
    observed_value: float | None = None


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    rejections: tuple[RiskRejection, ...] = ()
    warnings: tuple[str, ...] = ()
    checks_performed: tuple[str, ...] = ()

    @property
    def reason_summary(self) -> str:
        if self.approved:
            return "approved"
        return "; ".join(f"{r.code.value}: {r.message}" for r in self.rejections)


@dataclass(frozen=True)
class Signal:
    """Сигнал на закрытии бара ``ts_open`` (решение принимается ПОСЛЕ закрытия бара).

    side=None означает «закрыть позицию» (exit/flat). proba — вероятность класса 1
    модельного источника (или None у скриптовых сигналов). model_version — строка
    версии для журнала (эксперимент+fingerprint артефакта или "scripted").
    """

    ts_open: datetime
    side: Side | None
    symbol: str
    timeframe: str
    model_version: str
    proba: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderIntent:
    """Намерение, прошедшее (или не прошедшее) риск-контроль.

    reference_price — close сигнального бара, СПРАВОЧНО для журнала (см. module docstring).
    """

    created_at: datetime          # ts_open сигнального бара (момент решения)
    symbol: str
    timeframe: str
    side: Side
    quantity_units: int
    signal: Signal
    reference_price: float
    stop_price: float | None = None
    target_price: float | None = None
    client_intent_id: str = ""


@dataclass(frozen=True)
class Fill:
    """Симулированное исполнение.

    ts — метка бара исполнения (open следующего бара после сигнала либо бар
    срабатывания стопа/цели); raw_price — цена до проскальзывания; price —
    фактическая симулированная цена (с проскальзыванием, ограниченная диапазоном
    бара); commission_quote — комиссия в валюте котировки.
    """

    ts: datetime
    side: Side
    price: float
    raw_price: float
    units: int
    commission_quote: float
    slippage_bps: float
    bar_ts_open: datetime
    cause: str                    # entry_signal | exit_signal | stop | target | stop_target_conflict | gap_through_stop
    intent_id: str = ""
    symbol: str = ""
    timeframe: str = ""


@dataclass
class PortfolioState:
    """Состояние счёта/позиции симулятора (единственный источник правды в TEST)."""

    cash: float
    units: int = 0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    peak_equity: float = 0.0
    day_key: date | None = None
    day_realized_pnl: float = 0.0
    day_orders: int = 0
    halted: bool = False
    halt_reason: str | None = None

    def equity(self, last_close: float) -> float:
        return self.cash + self.units * last_close


@dataclass(frozen=True)
class RiskContext:
    """Контекст решения: всё, что риск-контроль знает на момент сигнала."""

    decision_ts: datetime
    last_bar_ts_open: datetime
    last_close: float
    equity: float
    cash: float
    position_units: int
    data_age_seconds: float | None = None  # None = источник не сообщил (TEST-replay: 0 по построению)
    bar_is_valid: bool = True
    data_source_ok: bool = True


@dataclass(frozen=True)
class HaltEvent:
    ts: datetime
    reason: str
    detail: str


# ---------------------------------------------------------------------------
# Биржевой шлюз: контракт для БУДУЩЕГО адаптера криптобиржи. Реализации НЕТ:
# подключение биржи не выполнено (exchange/base.py), LIVE-гейт (live/gate.py,
# docs/live.md) блокирует ордерный контур до её появления. В TEST-режиме
# шлюз не используется и не подключается.
# ---------------------------------------------------------------------------


class OrderSubmissionStatus(str, Enum):
    """Статус результата отправки заявки.

    UNKNOWN_RESULT — отдельное состояние: ответ на отправку не получен;
    заявка МОГЛА быть принята. Повторная отправка вслепую запрещена —
    только сверка по client_order_id.
    """

    SUBMITTED = "submitted"
    REJECTED = "rejected"                # явный отказ биржи: заявка не принята
    UNKNOWN_RESULT = "unknown_result"    # ответ потерян/5xx/таймаут: исход неизвестен
    DUPLICATE_FOUND = "duplicate_found"  # заявка с таким client_order_id уже существует
    DISABLED = "disabled"                # ордерный контур выключен гейтом/kill-switch


@dataclass(frozen=True)
class OrderSubmission:
    status: OrderSubmissionStatus
    client_order_id: str
    order_id: str | None = None
    order_status: str | None = None      # статус заявки в терминах биржи
    message: str = ""
    retryable_after_window: bool = False  # True только для явного rate-limit отказа (запрос не принимался)


@dataclass(frozen=True)
class ReconciliationMismatch:
    symbol: str
    local_units: int
    broker_units: int
    detail: str = ""


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    quantity_units: int


class BrokerGateway(ABC):
    """Контракт биржевого шлюза: заявки, состояния, сверка (реализация — будущая).

    Обязательства будущей реализации адаптера криптобиржи (docs/live.md):
    - детерминированный client_order_id из намерения — защита от дублирования;
      перед отправкой — проверка активных заявок по этому id (clientOrderId);
    - потерянный ответ НЕ повторяется вслепую: статус UNKNOWN_RESULT +
      сверка resolve_unknown() по client_order_id;
    - повторы только после явной сверки состояния;
    - reconcile_positions() сравнивает локальный учёт со снимком аккаунта.
    """

    @abstractmethod
    def submit(self, intent: OrderIntent, *, limit_price: "Decimal | None" = None) -> OrderSubmission:
        raise NotImplementedError

    @abstractmethod
    def resolve_unknown(self, client_order_id: str) -> OrderSubmission | None:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, order_id: str) -> OrderSubmission:
        raise NotImplementedError

    @abstractmethod
    def active_orders(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def account_positions(self) -> list[BrokerPosition]:
        raise NotImplementedError

    @abstractmethod
    def reconcile_positions(self, local_units_by_symbol: dict[str, int]) -> list[ReconciliationMismatch]:
        raise NotImplementedError
