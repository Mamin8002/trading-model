"""Исполнительный контур: риск-контроль, сайзинг, симулируемый брокер, журнал.

ТОЛЬКО TEST-симуляция. Реальные заявки не выставляются, подтверждения
биржи не имитируются. LIVE-шлюз (адаптер криптобиржи) НЕ реализован —
контракт описан в interfaces.BrokerGateway и exchange/base.py (docs/live.md).
"""

from .errors import IntrabarDataRequired, RiskHaltError, SimConfigError
from .interfaces import (
    ExitCause,
    Fill,
    HaltEvent,
    OrderIntent,
    PortfolioState,
    RejectionCode,
    RiskContext,
    RiskDecision,
    RiskRejection,
    Side,
    Signal,
)
from .journal import JournalWriter, audit_line, iter_audit, read_journal
from .risk import RiskGuards
from .simulator import ExitPlan, SimulatedBroker, validate_simulation_config
from .sizing import SizingError, SizingResult, compute_position_size, estimate_commission

__all__ = [
    "ExitCause",
    "ExitPlan",
    "Fill",
    "HaltEvent",
    "IntrabarDataRequired",
    "JournalWriter",
    "OrderIntent",
    "PortfolioState",
    "RejectionCode",
    "RiskContext",
    "RiskDecision",
    "RiskGuards",
    "RiskHaltError",
    "RiskRejection",
    "Side",
    "Signal",
    "SimConfigError",
    "SimulatedBroker",
    "SizingError",
    "SizingResult",
    "audit_line",
    "compute_position_size",
    "estimate_commission",
    "iter_audit",
    "read_journal",
    "validate_simulation_config",
]
