"""Базовые ориентиры (sanity baselines) — docs/decisions.md D4.

Обязательное условие промоушена модели: бустинг должен обыгрывать базовые
ориентиры НА ТОМ ЖЕ walk-forward разбиении:

1. ``majority``  — постоянный класс из обучающего окна фолда (оценка «угадывание
   большинства»; AUC вырождается в ~0.5, торговля = постоянная позиция/флэт);
2. ``momentum``  — «предыдущий знак»: класс = знак последней НАБЛЮДАННОЙ
   доходности close_t/close_{t-1} - 1 (прошлое, не будущее — утечки нет);
3. ``logreg``    — логистическая регрессия со стандартизацией, обученная строго
   на обучающем окне фолда (нормировка НЕ фитится на всей выборке — architecture §6.4).

Все ориентиры получают те же фолды и оцениваются теми же метриками, что и модель.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BaselineResult", "LogisticBaseline", "MajorityBaseline", "MomentumBaseline"]


@dataclass(frozen=True)
class BaselineResult:
    """Прогнозы ориентира на валидационном окне фолда."""

    name: str
    proba: np.ndarray  # P(y=1) для классификации; для регрессии — прогноз значения
    degenerate: bool = False  # True, если ориентир вырожден (например, один класс в train)
    note: str = ""


class MajorityBaseline:
    name = "majority"

    def fit(self, y_train: np.ndarray) -> None:
        self._p = float(np.mean(y_train)) if len(y_train) else 0.5
        self._degenerate = len(set(np.unique(y_train))) <= 1

    def predict_proba(self, n: int) -> BaselineResult:
        return BaselineResult(
            name=self.name,
            proba=np.full(n, self._p, dtype=np.float64),
            degenerate=self._degenerate,
            note="constant = доля класса 1 в обучающем окне фолда",
        )


class MomentumBaseline:
    """«Предыдущий знак»: p=1, если последний наблюдаемый бар вырос, иначе 0.

    Использует prev_ret (close_t/close_{t-1} - 1) — данные, доступные в момент
    решения t. Никакой будущей информации.
    """

    name = "momentum"

    def fit(self, y_train: np.ndarray) -> None:  # параметры не требуются
        pass

    def predict_proba(self, prev_ret: np.ndarray) -> BaselineResult:
        proba = (prev_ret > 0.0).astype(np.float64)
        proba = np.where(np.isclose(prev_ret, 0.0), 0.5, proba)
        return BaselineResult(name=self.name, proba=proba, note="sign(close_t/close_{t-1} - 1)")


class LogisticBaseline:
    """Логистическая регрессия + StandardScaler, фит только на train-окне фолда."""

    name = "logreg"

    def __init__(self, seed: int = 42, max_iter: int = 2000, c: float = 1.0) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self.c = c
        self._pipeline = None
        self._degenerate = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        classes = np.unique(y_train)
        self._degenerate = len(classes) <= 1
        if self._degenerate:
            self._pipeline = None
            self._constant = float(classes[0]) if len(classes) else 0.5
            return
        # NaN/inf быть не должно (датасет их исключает), но проверяем явно:
        if not np.isfinite(X_train).all():
            raise ValueError("LogisticBaseline: X_train содержит NaN/inf")
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # fit ТОЛЬКО на train-окне фолда
                ("clf", LogisticRegression(max_iter=self.max_iter, C=self.c, random_state=self.seed)),
            ]
        )
        self._pipeline.fit(X_train, y_train)

    def predict_proba(self, X_val: np.ndarray) -> BaselineResult:
        if self._pipeline is None:
            return BaselineResult(
                name=self.name,
                proba=np.full(len(X_val), getattr(self, "_constant", 0.5), dtype=np.float64),
                degenerate=True,
                note="train-окно содержит один класс — константный прогноз",
            )
        proba = self._pipeline.predict_proba(X_val)[:, 1]
        return BaselineResult(name=self.name, proba=proba, note="StandardScaler+LogisticRegression (fit на train-окне)")
