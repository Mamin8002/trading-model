"""Точка подключения криптобиржи: интерфейс адаптера и явный статус «НЕ реализовано».

Этот модуль — ЕДИНСТВЕННОЕ место, через которое в будущем подключается биржа
(docs/live.md, docs/roadmap.md). Сейчас:

- API-клиент, сетевые запросы, аутентификация и отправка ордеров НЕ реализованы;
- ``EXCHANGE_INTEGRATION_AVAILABLE = False`` — константа-факт, которую проверяет
  LIVE-гейт (live/gate.py): пока она False, LIVE-контур не может быть разрешён
  НИ при каких флагах/конфигах/переменных окружения (многофакторный гейт просто
  всегда имеет невыполненное условие);
- ``get_exchange_adapter()`` всегда бросает ``ExchangeNotImplemented`` — скрытых
  fallback-реализаций нет.

Что должен предоставить будущий адаптер (контракты уже определены и бирженезависимы):

- рыночные данные: marketdata/interfaces.py — ``HistorySource`` (свечи),
  ``QuoteSnapshotSource``/``OrderBookSnapshotSource`` (снимки), ``StreamSource``
  (потоки; бары считаются финальными только после закрытия — BarFinalizer);
- исполнение: execution/interfaces.py — ``BrokerGateway`` (заявки, сверка
  позиций, защита от дублей через детерминированный client_order_id);
- безопасность: config/env.py — секреты только из окружения с маскированием.

Отложенные решения (НЕ предполагаются по умолчанию — docs/decisions.md D15):
тип рынка (спот / маржа / фьючерсы), дробные количества, плечо, режим хеджирования,
валюта маржи. Они должны быть явно зафиксированы в конфигурации и документации
биржи при реализации адаптера; до этого бот работает ТОЛЬКО в TEST-режиме
на локальных CSV.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = [
    "EXCHANGE_INTEGRATION_AVAILABLE",
    "ExchangeAdapter",
    "ExchangeNotImplemented",
    "get_exchange_adapter",
]

#: Факт доступности интеграции с биржей. Проверяется LIVE-гейтом как обязательное
#: условие. Переключение в True допустимо ТОЛЬКО вместе с реальной, проверенной
#: реализацией адаптера (docs/live.md) — не раньше.
EXCHANGE_INTEGRATION_AVAILABLE = False


class ExchangeNotImplemented(RuntimeError):
    """Интеграция с криптобиржей не реализована. Сетевые вызовы невозможны."""


class ExchangeAdapter(ABC):
    """Контракт адаптера криптобиржи (реализаций нет — см. docstring модуля).

    Композиция существующих бирженезависимых портов: реализация предоставляет
    источник истории/потоков и (отдельно, за LIVE-гейтом) ордерный шлюз.
    """

    #: имя адаптера для конфигов/логов (например, имя биржи)
    name: str = ""

    @abstractmethod
    def history(self) -> object:
        """Источник истории: реализация marketdata.interfaces.HistorySource."""
        raise NotImplementedError

    @abstractmethod
    def stream(self) -> object:
        """Источник потоков: реализация marketdata.interfaces.StreamSource."""
        raise NotImplementedError

    @abstractmethod
    def gateway(self) -> object:
        """Ордерный шлюз: реализация execution.interfaces.BrokerGateway.

        Выдаётся ТОЛЬКО за пройденным LIVE-гейтом (live/gate.py).
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


def get_exchange_adapter(name: str | None = None) -> ExchangeAdapter:
    """Фабрика адаптера биржи. Сейчас ВСЕГДА бросает ExchangeNotImplemented.

    Функция существует, чтобы будущая интеграция подключалась в одной точке и
    чтобы вызывающий код (CLI/гейт) получал явную честную ошибку вместо
    молчаливого fallback.
    """
    raise ExchangeNotImplemented(
        "Подключение к криптобирже НЕ реализовано"
        + (f" (запрошен адаптер {name!r})" if name else "")
        + ": доступны только TEST-режим и локальные CSV. LIVE заблокирован "
        "многофакторным гейтом (docs/live.md). Реализация адаптера — отдельная задача "
        "(docs/roadmap.md)."
    )
