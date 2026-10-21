"""Типизированные ошибки Finam-адаптера и маскирование секретов.

Иерархия повторяет документированную карту HTTP-ошибок (docs/research.md §2.8):
400 -> FinamInvalidRequestError, 401 -> FinamAuthError, 403 -> FinamPermissionDeniedError,
404 -> FinamNotFoundError, 429 -> FinamRateLimitError, 500/503/504 -> FinamServerError,
сетевые -> FinamTransportError, битый ответ -> FinamInvalidResponseError.

Все сообщения перед выбросом проходят через SecretMasker: secret_key и access_key
никогда не попадают в исключения, логи и отчёты (research S4 «Чего не делать»).
"""

from __future__ import annotations

import logging

from ..config.env import GLOBAL_MASKER, MaskingFilter, SecretMasker

__all__ = [
    "FinamError",
    "FinamAuthError",
    "FinamPermissionDeniedError",
    "FinamInvalidRequestError",
    "FinamNotFoundError",
    "FinamRateLimitError",
    "FinamServerError",
    "FinamTransportError",
    "FinamInvalidResponseError",
    "UnsupportedTimeframeError",
    "HistoryDepthError",
    "get_logger",
]


class FinamError(Exception):
    """Базовая ошибка адаптера. Сообщение уже маскировано."""

    def __init__(self, message: str, masker: SecretMasker | None = None) -> None:
        super().__init__((masker or GLOBAL_MASKER).mask(str(message)))


class FinamAuthError(FinamError):
    """401: secret_key/access_key недействителен или истёк."""


class FinamPermissionDeniedError(FinamError):
    """403: недостаточно прав (тип токена, счёт, рыночные данные)."""


class FinamInvalidRequestError(FinamError):
    """400: неверный symbol/интервал/параметры. НЕ повторять — исправлять запрос."""


class FinamNotFoundError(FinamError):
    """404: счёт/инструмент/заявка не найдены."""


class FinamRateLimitError(FinamError):
    """429: исчерпана квота (200 запросов/мин на метод). Не повторять немедленно."""


class FinamServerError(FinamError):
    """500/503/504: ошибка/недоступность сервиса. Ограниченный повтор с backoff."""

    def __init__(self, message: str, status_code: int | None = None, masker: SecretMasker | None = None) -> None:
        super().__init__(message, masker)
        self.status_code = status_code


class FinamTransportError(FinamError):
    """Сетевая ошибка до получения ответа. GET можно повторять с backoff."""


class FinamInvalidResponseError(FinamError):
    """Ответ не соответствует документированной схеме — данные не угадываются."""


class UnsupportedTimeframeError(FinamError):
    """Таймфрейм не поддерживается API (например 10m: TIME_FRAME_M10 отсутствует
    в документированном enum — docs/research.md §2.3/§6, decisions D3)."""


class HistoryDepthError(FinamError):
    """Запрошенное окно истории превышает документированную глубину
    (M1 — 7 дней; M5/M15/H1 — 30 дней; research §2.3)."""


_LOGGER_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Логгер пространства 'tradebot' с фильтром маскирования секретов (один раз)."""
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        root = logging.getLogger("tradebot")
        root.addFilter(MaskingFilter(GLOBAL_MASKER))
        # Фильтр на родительском логгере применяется к записям детей только если
        # запись проходит через него; для надёжности вешаем и на каждый child.
        _LOGGER_CONFIGURED = True
    logger = logging.getLogger(name if name.startswith("tradebot") else f"tradebot.{name}")
    if not any(isinstance(f, MaskingFilter) for f in logger.filters):
        logger.addFilter(MaskingFilter(GLOBAL_MASKER))
    return logger
