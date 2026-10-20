"""
model.py — ML-модель: обучение, walk-forward валидация, инференс.

Архитектура: градиентный бустинг LightGBM (мультикласс up/flat/down).
Запасной лёгкий вариант — LogisticRegression (для окружений без LGBM).

Ключевые моменты:
  * Predict_proba -> уверенность сигнала (порог задаётся конфигом).
  * Walk-Forward Validation: последовательные train/test окна без shuffle,
    оценка accuracy/F1 на OOS-окнах.
  * Механизм ретренировки (retrain) для адаптации к смене рыночных режимов.
  * Сохранение/загрузка модели (joblib).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger("model")

# маппинг классов -1/0/1 -> 0/1/2 (LGBM требует непрерывные метки с нуля)
LABELS_RAW = [-1, 0, 1]


@dataclass
class ModelPrediction:
    direction: int          # -1 | 0 | 1
    confidence: float       # max proba соответствующего класса
    proba_up: float
    proba_down: float
    proba_flat: float


@dataclass
class TrainReport:
    n_train: int
    n_features: int
    class_balance: dict
    cv_accuracy: float = 0.0
    cv_f1: float = 0.0
    wf_windows: list = field(default_factory=list)


class TradeModel:
    def __init__(self, model_type: str = "lightgbm", random_state: int = 7):
        self.model_type = model_type if (model_type == "lightgbm" and HAS_LGBM) else "logreg"
        if model_type == "lightgbm" and not HAS_LGBM:
            log.warning("LightGBM недоступен — переключаемся на LogisticRegression")
        self.random_state = random_state
        self.model = None
        self.encoder = LabelEncoder().fit(LABELS_RAW)
        self.feature_names: Optional[list[str]] = None
        self.report: Optional[TrainReport] = None
        self.trained = False

    # ------------------------------------------------------------------ fit
    def _make_estimator(self):
        if self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                objective="multiclass", num_class=3,
                n_estimators=400, learning_rate=0.05,
                num_leaves=63, max_depth=-1,
                min_child_samples=40, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                class_weight="balanced", random_state=self.random_state,
                n_jobs=-1, verbosity=-1,
            )
        return LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                                  random_state=self.random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[tuple[pd.DataFrame, pd.Series]] = None) -> TrainReport:
        """Обучение на всём переданном датасете (X уже очищен от NaN)."""
        self.feature_names = list(X.columns)
        yc = self.encoder.transform(y.values)
        self.model = self._make_estimator()
        fit_kwargs = {}
        if self.model_type == "lightgbm" and eval_set is not None:
            Xe, ye = eval_set
            fit_kwargs = dict(eval_set=[(Xe, self.encoder.transform(ye.values))],
                              eval_metric="multi_logloss",
                              callbacks=[lgb.early_stopping(50, verbose=False)])
        self.model.fit(X.values, yc, **fit_kwargs)
        balance = {int(k): int(v) for k, v in zip(*np.unique(y.values, return_counts=True))}
        self.report = TrainReport(n_train=len(X), n_features=X.shape[1], class_balance=balance)
        self.trained = True
        log.info("Модель обучена: %s, samples=%d, features=%d, classes=%s",
                 self.model_type, len(X), X.shape[1], balance)
        return self.report

    # ------------------------------------------------------------- inference
    def predict(self, X_row: pd.DataFrame) -> ModelPrediction:
        """Инференс на одном (или нескольких) срезе признаков; берём последнюю строку."""
        if not self.trained:
            raise RuntimeError("Модель не обучена")
        if self.feature_names is not None:
            X_row = X_row[self.feature_names]
        proba = self.model.predict_proba(X_row.values)[-1]   # [p(-1), p(0), p(+1)]
        p_down, p_flat, p_up = float(proba[0]), float(proba[1]), float(proba[2])
        idx = int(np.argmax(proba))
        direction = LABELS_RAW[idx]
        return ModelPrediction(direction=direction, confidence=float(proba[idx]),
                               proba_up=p_up, proba_down=p_down, proba_flat=p_flat)

    # ------------------------------------------------ walk-forward validation
    def walk_forward_cv(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 4,
                        train_frac: float = 0.6) -> tuple[float, float, list[dict]]:
        """
        Классический expanding-window walk-forward:
        данные режутся на n_splits последовательных окон; каждое окно обучается
        на предыдущих train_frac*window баров и тестируется на следующих.
        Никакого shuffle — строго хронологический порядок.
        """
        n = len(X)
        window = n // (n_splits + 1)
        accs, f1s, details = [], [], []
        for i in range(n_splits):
            tr_end = window * (i + 2)
            te_start, te_end = tr_end, min(tr_end + window, n)
            if te_end - te_start < 50:
                break
            Xtr, ytr = X.iloc[:te_start - (te_start - tr_end)], y.iloc[:te_start - (te_start - tr_end)]
            # hold-out последние 20% train-окна как early-stopping set
            cut = int(len(Xtr) * 0.8)
            eval_set = (Xtr.iloc[cut:], ytr.iloc[cut:])
            Xtr, ytr = Xtr.iloc[:cut], ytr.iloc[:cut]
            Xte, yte = X.iloc[te_start:te_end], y.iloc[te_start:te_end]
            fold = TradeModel(self.model_type, self.random_state)
            fold.fit(Xtr, ytr, eval_set=eval_set)
            # batch prediction на OOS-окне:
            proba = fold.model.predict_proba(Xte.values)
            pred_cls = fold.encoder.inverse_transform(proba.argmax(axis=1))
            acc = accuracy_score(yte.values, pred_cls)
            f1 = f1_score(yte.values, pred_cls, average="weighted")
            accs.append(acc); f1s.append(f1)
            details.append({"fold": i, "train_n": len(Xtr), "test_n": len(Xte),
                            "accuracy": round(acc, 4), "f1_weighted": round(f1, 4)})
            log.info("WF fold %d: n_test=%d acc=%.3f f1=%.3f", i, len(Xte), acc, f1)
        # восстанавливаем основную модель обучением на полном окне
        self.fit(X, y)
        if self.report is not None:
            self.report.cv_accuracy = float(np.mean(accs)) if accs else 0.0
            self.report.cv_f1 = float(np.mean(f1s)) if f1s else 0.0
            self.report.wf_windows = details
        return self.report.cv_accuracy, self.report.cv_f1, details

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "encoder": self.encoder,
                     "feature_names": self.feature_names, "type": self.model_type}, path)
        log.info("Модель сохранена: %s", path)

    @classmethod
    def load(cls, path: str) -> "TradeModel":
        import joblib
        blob = joblib.load(path)
        obj = cls(model_type=blob["type"])
        obj.model, obj.encoder = blob["model"], blob["encoder"]
        obj.feature_names = blob["feature_names"]
        obj.trained = True
        return obj
