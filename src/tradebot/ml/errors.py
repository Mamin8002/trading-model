"""Ошибки ML-контура."""

from __future__ import annotations

__all__ = ["InsufficientHistoryError", "ModelCompatibilityError", "TrainingError"]


class TrainingError(Exception):
    """Базовая ошибка обучения/оценки."""


class InsufficientHistoryError(TrainingError):
    """Истории недостаточно для честной walk-forward валидации.

    Требования задачи: в этом случае прогон останавливается с явным сообщением;
    метрики на обучающей выборке НЕ выдаются за оценку качества.
    """

    def __init__(self, message: str, *, required_bars: int | None = None, available_bars: int | None = None) -> None:
        super().__init__(message)
        self.required_bars = required_bars
        self.available_bars = available_bars


class ModelCompatibilityError(TrainingError):
    """Артефакт модели несовместим с запрошенным контекстом (ТФ/признаки/горизонт)."""

    def __init__(self, mismatches: list[str]) -> None:
        self.mismatches = mismatches
        super().__init__("Модель несовместима с контекстом:\n  - " + "\n  - ".join(mismatches))
