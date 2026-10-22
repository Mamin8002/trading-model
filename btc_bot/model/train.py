"""Обучение финальной модели и сохранение/загрузка артефакта с метаданными.

Артефакт: каталог data/models/<model_version>/
  * model.joblib      - обученный XGBClassifier
  * model_card.json   - метаданные (версия, инструмент, интервал, горизонт,
                        feature_set, параметры, данные обучения, метрики,
                        версии библиотек, ограничения)
  * walk_forward.json - полные результаты walk-forward-валидации
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"
WF_FILE = "walk_forward.json"
CARD_SCHEMA_VERSION = 1

# Соглашение о кодировании меток (XGBoost 3.x требует 0/1 для бинарной классификации):
#   +1 (рост, "up")            -> 1
#   -1 (падение/ничья, "down") -> 0
# В датасете (prep.build_dataset) метки хранятся как ±1 (как в исходном ноутбуке);
# кодирование выполняется только на модельном слое.
LABEL_ENCODING = {"up(+1)": 1, "down_or_flat(-1)": 0}


def encode_labels(y) -> np.ndarray:
    """±1 -> 0/1 для обучения XGBoost."""
    y_arr = np.asarray(y)
    encoded = np.where(y_arr == 1, 1, 0).astype("int64")
    if not set(np.unique(y_arr)) <= {-1, 1}:
        raise ValueError("Ожидались метки {-1, +1}")
    return encoded


def decode_predictions(pred_encoded) -> np.ndarray:
    """0/1 -> ±1."""
    return np.where(np.asarray(pred_encoded) == 1, 1, -1).astype("int64")


def make_model(seed: int = 42, n_estimators: int = 100, max_depth: int = 6,
               learning_rate: float = 0.3) -> XGBClassifier:
    """XGBClassifier с параметрами по умолчанию (как XGBClassifier() в оригинальном ноутбуке)
    + зафиксированный seed для воспроизводимости."""
    return XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        random_state=int(seed),
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
        verbosity=0,
    )


def train_final(X: pd.DataFrame, y: pd.Series, *, seed: int = 42,
                n_estimators: int = 100, max_depth: int = 6,
                learning_rate: float = 0.3) -> XGBClassifier:
    model = make_model(seed, n_estimators, max_depth, learning_rate)
    model.fit(X, encode_labels(y))
    return model


def make_model_version(symbol: str, category: str, interval: str, horizon_bars: int,
                       feature_set: str, ts: Optional[datetime] = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    itok = interval if interval in ("D", "W", "M") else f"{interval}m"
    return (f"xgb-{symbol}-{category}-{itok}-h{horizon_bars}-"
            f"{feature_set}-{ts:%Y%m%dT%H%M%SZ}")


def library_versions() -> dict[str, str]:
    import sklearn
    import xgboost
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
    }


def build_card(**kwargs: Any) -> dict:
    card = {
        "schema_version": CARD_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "kind": "btc-direction-classifier",
        "disclaimer": (
            "Сигналы модели - статистический вывод, не гарантия дохода и не персональная "
            "финансовая рекомендация. Accuracy не означает прибыльность: спред, комиссии "
            "и слиппедж не учитываются."
        ),
    }
    card.update(kwargs)
    return card


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_artifact(model: XGBClassifier, out_dir: str | Path, card: dict,
                  walk_forward: Optional[dict] = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILE
    card_path = out_dir / CARD_FILE

    buf_path = out_dir / (MODEL_FILE + ".tmp")
    joblib.dump(model, buf_path)
    os.replace(buf_path, model_path)
    _atomic_write_bytes(card_path, (json.dumps(card, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
    paths = {"model": model_path, "card": card_path}
    if walk_forward is not None:
        wf_path = out_dir / WF_FILE
        _atomic_write_bytes(wf_path, (json.dumps(walk_forward, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
        paths["walk_forward"] = wf_path
    return paths


def load_artifact(model_dir: str | Path) -> tuple[XGBClassifier, dict]:
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILE
    card_path = model_dir / CARD_FILE
    if not model_path.exists() or not card_path.exists():
        raise FileNotFoundError(f"Артефакт модели не найден в {model_dir} "
                                f"(ожидаются {MODEL_FILE} и {CARD_FILE})")
    with open(card_path, "r", encoding="utf-8") as f:
        card = json.load(f)
    if int(card.get("schema_version", 0)) != CARD_SCHEMA_VERSION:
        raise ValueError(f"Несовместимая схема model_card: {card.get('schema_version')}")
    model = joblib.load(model_path)
    return model, card


def find_latest_model_dir(models_root: str | Path) -> Optional[Path]:
    """Каталог последней ПРИНЯТОЙ модели (по created_at_utc; adopted=false пропускаются)."""
    root = Path(models_root)
    if not root.exists():
        return None
    best: Optional[tuple[str, Path]] = None
    for d in sorted(root.iterdir()):
        card_path = d / CARD_FILE
        if d.is_dir() and card_path.exists():
            try:
                with open(card_path, "r", encoding="utf-8") as f:
                    card = json.load(f)
                if card.get("adopted") is False:
                    continue  # артефакт сохранён для прозрачности, но НЕ принят gate'ом
                created = str(card.get("created_at_utc", ""))
            except (json.JSONDecodeError, OSError):
                continue
            key = created or d.name
            if best is None or key > best[0]:
                best = (key, d)
    return best[1] if best else None
