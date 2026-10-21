"""Загрузка артефакта модели и формирование предсказаний по закрытому бару.

Предсказание = (направление, уверенность); преобразование в сигнал BUY/SELL/HOLD
с учётом порога уверенности выполняется здесь же (action), но семантика SELL для
spot (закрыть/уменьшить длинную позицию) и привязка к позиции - уровень
сигнального движка (Этап 3/4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .features import bar_features_frame
from .train import load_artifact


class ModelMismatchError(RuntimeError):
    """Конфигурация бота не соответствует артефакту модели."""


@dataclass(frozen=True)
class ModelPrediction:
    action: str            # BUY | SELL | HOLD (HOLD = воздержание при низкой уверенности)
    raw_direction: int     # +1 (рост) / -1 (падение или ничья) - argmax модели
    confidence: float      # max(prob_up, 1-prob_up) ∈ [0.5, 1]
    prob_up: float         # вероятность класса +1
    model_version: str
    horizon_bars: int
    horizon_minutes: int
    feature_set: str
    features: dict         # значения признаков, на которых сделано предсказание


class Predictor:
    def __init__(self, model, card: dict, confidence_threshold: float):
        self.model = model
        self.card = card
        self.confidence_threshold = float(confidence_threshold)
        # классы модели закодированы 0/1 (0 = -1 "down/flat", 1 = +1 "up")
        self._classes = list(model.classes_)
        if set(self._classes) != {0, 1}:
            raise ModelMismatchError(
                f"Неожиданные классы модели: {self._classes} (ожидались [0, 1]: "
                f"0=down/flat, 1=up)"
            )

    @classmethod
    def load(cls, model_dir: str | Path, confidence_threshold: float = 0.55) -> "Predictor":
        model, card = load_artifact(model_dir)
        return cls(model, card, confidence_threshold)

    # -- card fields ----------------------------------------------------
    @property
    def model_version(self) -> str:
        return str(self.card["model_version"])

    @property
    def feature_set(self) -> str:
        return str(self.card["feature_set"])

    @property
    def horizon_bars(self) -> int:
        return int(self.card["horizon_bars"])

    @property
    def horizon_minutes(self) -> int:
        return int(self.card["horizon_minutes"])

    def ensure_compatible(self, *, symbol: str, category: str, interval: str, horizon_bars: int) -> None:
        problems = []
        if str(self.card.get("symbol")) != symbol:
            problems.append(f"symbol: модель={self.card.get('symbol')!r}, конфиг={symbol!r}")
        if str(self.card.get("category")) != category:
            problems.append(f"category: модель={self.card.get('category')!r}, конфиг={category!r}")
        if str(self.card.get("interval")) != str(interval):
            problems.append(f"interval: модель={self.card.get('interval')!r}, конфиг={interval!r}")
        if int(self.card.get("horizon_bars", -1)) != int(horizon_bars):
            problems.append(f"horizon_bars: модель={self.card.get('horizon_bars')!r}, конфиг={horizon_bars!r}")
        if problems:
            raise ModelMismatchError(
                "Артефакт модели не соответствует конфигурации - переобучите модель "
                "(scripts/train_model.py) или поправьте конфиг: " + "; ".join(problems)
            )

    # -- prediction ------------------------------------------------------
    def action_from_prob_up(self, prob_up: float) -> tuple[str, int, float]:
        # ничья (prob_up == 0.5) трактуется как -1 (конвенция исходного ноутбука);
        # при confidence < порога действие всегда HOLD независимо от направления
        raw_direction = 1 if prob_up > 0.5 else -1
        confidence = max(prob_up, 1.0 - prob_up)
        if confidence < self.confidence_threshold:
            action = "HOLD"
        else:
            action = "BUY" if raw_direction == 1 else "SELL"
        return action, raw_direction, confidence

    def predict_bar(self, bar: Mapping, *, signal_ts_ms: int | None = None) -> ModelPrediction:
        """bar: словарь закрытого бара (open/high/low/close/volume/turnover[, start_ms])."""
        feats_frame = bar_features_frame(bar, self.feature_set)
        proba = self.model.predict_proba(feats_frame)[0]
        prob_up = float(proba[self._classes.index(1)])
        action, raw_direction, confidence = self.action_from_prob_up(prob_up)
        _ = signal_ts_ms  # метку времени сигнала проставляет вызывающий код (Этап 3)
        return ModelPrediction(
            action=action,
            raw_direction=raw_direction,
            confidence=round(confidence, 6),
            prob_up=round(prob_up, 6),
            model_version=self.model_version,
            horizon_bars=self.horizon_bars,
            horizon_minutes=self.horizon_minutes,
            feature_set=self.feature_set,
            features={k: float(v) for k, v in feats_frame.iloc[0].items()},
        )
