"""Точка подключения криптобиржи (НЕ реализована — см. base.py, docs/live.md)."""

from .base import (
    EXCHANGE_INTEGRATION_AVAILABLE,
    ExchangeAdapter,
    ExchangeNotImplemented,
    get_exchange_adapter,
)

__all__ = [
    "EXCHANGE_INTEGRATION_AVAILABLE",
    "ExchangeAdapter",
    "ExchangeNotImplemented",
    "get_exchange_adapter",
]
