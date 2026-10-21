"""Хронологическая walk-forward-валидация без утечки будущих данных.

Схема: expanding window. Первые train_fraction данных - стартовое обучение;
остаток делится на n_folds последовательных тестовых окон. На каждом окне
модель переобучается заново на всей истории ДО окна.

Embargo (защита от утечки): метка строки t становится известна только в момент
label_resolution(t) = конец бара, начинающегося в t + H*interval. В обучение
попадают только строки с label_resolution(t) <= test_start. При полном наборе
данных это эквивалентно исключению последних H баров перед границей окна.

Сравнение на каждом окне:
  * модель (accuracy, f1/precision/recall для класса +1, confusion matrix);
  * покрытие и точность при confidence >= threshold (HOLD-полоса);
  * baseline "majority" - постоянный прогноз преобладающего класса обучения;
  * baseline "momentum" - знак изменения close текущего бара (без обучения).
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from ..data_sources.bybit import INTERVAL_MS, period_end_ms
from .features import feature_columns
from .train import decode_predictions, encode_labels, make_model

ProgressCB = Callable[[int, int, dict], None]


def _fixed_step_ms(interval: str) -> Optional[int]:
    return INTERVAL_MS.get(interval)


def label_resolution_times(times: np.ndarray, interval: str, horizon_bars: int) -> np.ndarray:
    """Момент, когда метка каждой строки становится полностью известна (unix ms)."""
    step = _fixed_step_ms(interval)
    if step is not None:
        return times.astype("int64") + (horizon_bars + 1) * step
    out = np.empty(len(times), dtype="int64")
    for i, t in enumerate(times):
        cur = int(t)
        for _ in range(horizon_bars + 1):
            cur = period_end_ms(cur, interval)
        out[i] = cur
    return out


def shift_back_ms(t_ms: int, interval: str, n: int) -> int:
    """Сдвиг метки времени на n периодов назад (для baseline'ов)."""
    step = _fixed_step_ms(interval)
    if step is not None:
        return int(t_ms) - n * step
    ts = pd.Timestamp(int(t_ms), unit="ms", tz="UTC")
    if interval == "D":
        ts = ts - pd.offsets.Day(n)
    elif interval == "W":
        ts = ts - pd.offsets.Week(n)
    else:
        ts = ts.normalize() - pd.offsets.MonthBegin(n)
    return int(ts.timestamp() * 1000)


@dataclass
class FoldMetrics:
    fold: int
    train_start_ms: int
    train_end_ms: int
    train_n: int
    test_start_ms: int
    test_end_ms: int
    test_n: int
    accuracy: float
    f1_up: float
    precision_up: float
    recall_up: float
    confusion: dict
    test_class_balance: dict
    coverage_at_threshold: float
    accuracy_at_threshold: Optional[float]
    majority_accuracy: float
    majority_class: int
    momentum_accuracy: Optional[float]
    momentum_n_used: int

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "train": {"start_ms": self.train_start_ms, "end_ms": self.train_end_ms, "n": self.train_n},
            "test": {"start_ms": self.test_start_ms, "end_ms": self.test_end_ms, "n": self.test_n},
            "model": {
                "accuracy": self.accuracy, "f1_up": self.f1_up,
                "precision_up": self.precision_up, "recall_up": self.recall_up,
                "confusion": self.confusion, "test_class_balance": self.test_class_balance,
                "coverage_at_threshold": self.coverage_at_threshold,
                "accuracy_at_threshold": self.accuracy_at_threshold,
            },
            "baselines": {
                "majority": {"accuracy": self.majority_accuracy, "class": self.majority_class},
                "momentum": {"accuracy": self.momentum_accuracy, "n_used": self.momentum_n_used},
            },
        }


@dataclass
class WalkForwardResult:
    feature_set: str
    interval: str
    horizon_bars: int
    threshold: float
    n_folds: int
    train_fraction: float
    folds: list[FoldMetrics] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "feature_set": self.feature_set,
            "interval": self.interval,
            "horizon_bars": self.horizon_bars,
            "threshold": self.threshold,
            "n_folds": self.n_folds,
            "train_fraction": self.train_fraction,
            "embargo_rule": "label_resolution(t)=t+(H+1)*interval <= test_start",
            "elapsed_s": round(self.elapsed_s, 2),
            "folds": [f.to_dict() for f in self.folds],
            "aggregate": self.aggregate,
        }


def _weighted(values: list[tuple[float, int]]) -> float:
    total_w = sum(w for _, w in values)
    if total_w == 0:
        return float("nan")
    return sum(v * w for v, w in values) / total_w


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_up": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_up": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_up": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion": {
            "actual_-1": {"pred_-1": int(cm[0][0]), "pred_+1": int(cm[0][1])},
            "actual_+1": {"pred_-1": int(cm[1][0]), "pred_+1": int(cm[1][1])},
        },
    }


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    times: pd.Series,
    prices: pd.DataFrame,
    interval: str,
    horizon_bars: int,
    *,
    feature_set: str,
    n_folds: int = 8,
    train_fraction: float = 0.6,
    threshold: float = 0.55,
    seed: int = 42,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    progress_cb: Optional[ProgressCB] = None,
) -> WalkForwardResult:
    """Walk-forward с переобучением на каждом окне и embargo по времени разрешения метки."""
    cols = feature_columns(feature_set)  # заодно валидация имени набора
    if list(X.columns) != cols:
        raise ValueError(f"Колонки X {list(X.columns)} != ожидаемые {cols}")
    n = len(X)
    if n < 100:
        raise ValueError(f"Слишком мало данных для walk-forward: {n} строк (нужно >= 100)")
    if not (0.2 <= train_fraction < 0.95):
        raise ValueError("train_fraction должен быть в [0.2, 0.95)")

    times_np = times.to_numpy(dtype="int64")
    y_np = y.to_numpy(dtype="int64")
    resolution = label_resolution_times(times_np, interval, horizon_bars)
    close_by_time = dict(zip(prices["start_ms"].astype("int64"), prices["close"].astype("float64")))

    split0 = int(n * train_fraction)
    test_region = n - split0
    if test_region < n_folds * 10:
        raise ValueError(f"Тестовая область слишком мала: {test_region} строк для {n_folds} окон")
    fold_size = test_region // n_folds

    result = WalkForwardResult(
        feature_set=feature_set, interval=interval, horizon_bars=horizon_bars,
        threshold=threshold, n_folds=n_folds, train_fraction=train_fraction,
    )
    t0 = _time.monotonic()

    for k in range(n_folds):
        a = split0 + k * fold_size
        b = n if k == n_folds - 1 else a + fold_size
        test_start = int(times_np[a])

        train_mask = resolution <= test_start  # EMBARGO: метка известна до начала окна
        X_train = X.loc[train_mask]
        y_train = y_np[train_mask]
        if len(X_train) < 50:
            raise ValueError(f"Fold {k}: после embargo слишком мало обучающих строк: {len(X_train)}")

        model = make_model(seed, n_estimators, max_depth, learning_rate)
        model.fit(X_train, encode_labels(y_train))

        X_test = X.iloc[a:b]
        y_test = y_np[a:b]
        t_test = times_np[a:b]
        proba = model.predict_proba(X_test)
        classes = list(model.classes_)  # закодированные классы [0, 1]
        up_idx = classes.index(1)       # 1 = "up" = +1
        prob_up = proba[:, up_idx]
        y_pred = decode_predictions(np.argmax(proba, axis=1))
        confidence = np.maximum(prob_up, 1.0 - prob_up)

        m = classification_metrics(y_test, y_pred)
        above = confidence >= threshold
        acc_thr = float(accuracy_score(y_test[above], y_pred[above])) if above.any() else None

        vals, counts = np.unique(y_train, return_counts=True)
        majority_class = int(vals[np.argmax(counts)])
        maj_acc = float(accuracy_score(y_test, np.full_like(y_test, majority_class)))

        prev_close = np.array([close_by_time.get(shift_back_ms(int(t), interval, 1), np.nan) for t in t_test])
        cur_close = np.array([close_by_time.get(int(t), np.nan) for t in t_test])
        mom_ok = ~(np.isnan(prev_close) | np.isnan(cur_close))
        if mom_ok.any():
            mom_pred = np.where(cur_close[mom_ok] > prev_close[mom_ok], 1, -1)
            mom_acc = float(accuracy_score(y_test[mom_ok], mom_pred))
        else:
            mom_acc = None

        fm = FoldMetrics(
            fold=k,
            train_start_ms=int(times_np[train_mask][0]), train_end_ms=int(times_np[train_mask][-1]),
            train_n=int(train_mask.sum()),
            test_start_ms=test_start, test_end_ms=int(t_test[-1]), test_n=int(b - a),
            accuracy=m["accuracy"], f1_up=m["f1_up"], precision_up=m["precision_up"],
            recall_up=m["recall_up"], confusion=m["confusion"],
            test_class_balance={"up": int((y_test == 1).sum()), "down": int((y_test == -1).sum())},
            coverage_at_threshold=float(above.mean()), accuracy_at_threshold=acc_thr,
            majority_accuracy=maj_acc, majority_class=majority_class,
            momentum_accuracy=mom_acc, momentum_n_used=int(mom_ok.sum()),
        )
        result.folds.append(fm)
        if progress_cb:
            progress_cb(k, n_folds, {"accuracy": fm.accuracy, "test_n": fm.test_n,
                                     "majority": fm.majority_accuracy, "momentum": fm.momentum_accuracy})

    w = [(f.accuracy, f.test_n) for f in result.folds]
    del w  # агрегаты считаются ниже по каждому показателю отдельно
    result.aggregate = {
        "n_test_total": sum(f.test_n for f in result.folds),
        "accuracy": _weighted([(f.accuracy, f.test_n) for f in result.folds]),
        "f1_up": _weighted([(f.f1_up, f.test_n) for f in result.folds]),
        "precision_up": _weighted([(f.precision_up, f.test_n) for f in result.folds]),
        "recall_up": _weighted([(f.recall_up, f.test_n) for f in result.folds]),
        "coverage_at_threshold": _weighted([(f.coverage_at_threshold, f.test_n) for f in result.folds]),
        "accuracy_at_threshold": _weighted(
            [(f.accuracy_at_threshold, f.test_n) for f in result.folds if f.accuracy_at_threshold is not None]
        ),
        "majority_accuracy": _weighted([(f.majority_accuracy, f.test_n) for f in result.folds]),
        "momentum_accuracy": _weighted(
            [(f.momentum_accuracy, f.test_n) for f in result.folds if f.momentum_accuracy is not None]
        ),
    }
    result.elapsed_s = _time.monotonic() - t0
    return result


def evaluate_original_protocol(
    X: pd.DataFrame, y: pd.Series, *, seed: int = 42, n_estimators: int = 100,
    max_depth: int = 6, learning_rate: float = 0.3, split_fraction: float = 0.75,
) -> dict:
    """Воспроизведение протокола исходного ноутбука на свежих данных:
    единственный хронологический сплит 75/25, без embargo и без walk-forward."""
    n = len(X)
    split = int(split_fraction * n)
    model = make_model(seed, n_estimators, max_depth, learning_rate)
    model.fit(X.iloc[:split], encode_labels(y.to_numpy()[:split]))
    y_pred = decode_predictions(model.predict(X.iloc[split:]))
    m = classification_metrics(y.to_numpy()[split:], np.asarray(y_pred))
    return {
        "protocol": f"single chronological split {split_fraction:.0%}/{1 - split_fraction:.0%}, no embargo",
        "n_train": split, "n_test": n - split, **m,
    }
