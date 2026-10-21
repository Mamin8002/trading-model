"""Модельный реестр: артефакт + метаданные + проверка совместимости (docs/ml.md §6).

Состав артефакта (каталог):
- ``model.lgb.txt``       — текстовая модель LightGBM (booster);
- ``metadata.json``       — период обучения, ТФ, горизонт, метка, сплиты, метрики
  по фолдам и OOS, допущения, версии библиотек, seed, хэш конфига, fingerprint датасета;
- ``feature_manifest.json`` — версия набора признаков, параметры, ТОЧНЫЙ список
  и порядок колонок, warmup;
- ``oos_predictions.csv`` — прогнозы out-of-sample (ts, y, proba/pred, position).

Загрузчик отклоняет несовместимый контекст (таймфрейм, горизонт, вид метки,
список признаков) — ModelCompatibilityError с перечнем расхождений: рассинхрон
«модель ↔ признаки» недопустим (architecture §5.4, decisions D10).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..data.bar import Bar
from ..features.builder import FeatureBuilder
from ..features.spec import FeatureSpec
from .errors import ModelCompatibilityError

__all__ = [
    "MODEL_ARTIFACT_NAME",
    "METADATA_NAME",
    "MANIFEST_NAME",
    "ModelBundle",
    "library_versions",
    "load_model_bundle",
    "save_model_bundle",
]

MODEL_ARTIFACT_NAME = "model.lgb.txt"
METADATA_NAME = "metadata.json"
MANIFEST_NAME = "feature_manifest.json"
PREDICTIONS_NAME = "oos_predictions.csv"
METADATA_SCHEMA_VERSION = 1


def library_versions() -> dict[str, str]:
    """Версии runtime-библиотек для метаданных (воспроизводимость, task 6)."""
    from importlib import metadata as importlib_metadata

    versions = {"python": sys.version.split()[0]}
    try:
        from .. import __version__ as tradebot_version

        versions["tradebot"] = tradebot_version
    except Exception:  # pragma: no cover
        versions["tradebot"] = "unknown"
    for package in ("numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow", "pydantic"):
        try:
            versions[package] = importlib_metadata.version(package)
        except Exception:
            versions[package] = "not-installed"
    return versions


@dataclass(frozen=True)
class ModelBundle:
    """Загруженная модель + метаданные + манифест признаков."""

    model_dir: Path
    booster: Any  # lightgbm.Booster (импорт отложен, чтобы реестр не требовал lightgbm на чтение метаданных)
    metadata: dict[str, Any]
    spec: FeatureSpec

    @property
    def timeframe(self) -> str:
        return str(self.metadata["timeframe"])

    @property
    def horizon_bars(self) -> int:
        return int(self.metadata["horizon_bars"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["feature_manifest"]["feature_names"])

    def validate_compatibility(
        self,
        *,
        timeframe: str | None = None,
        feature_names: Sequence[str] | None = None,
        horizon_bars: int | None = None,
        label_kind: str | None = None,
    ) -> None:
        """Проверяет совместимость контекста; перечисляет ВСЕ расхождения."""
        mismatches: list[str] = []
        if timeframe is not None and timeframe != self.timeframe:
            mismatches.append(f"таймфрейм: модель={self.timeframe!r}, запрошен={timeframe!r}")
        if horizon_bars is not None and horizon_bars != self.horizon_bars:
            mismatches.append(
                f"горизонт: модель={self.horizon_bars}, запрошен={horizon_bars}"
            )
        if label_kind is not None and label_kind != self.metadata.get("label", {}).get("kind"):
            mismatches.append(
                f"вид метки: модель={self.metadata.get('label', {}).get('kind')!r}, запрошен={label_kind!r}"
            )
        if feature_names is not None:
            expected = list(self.feature_names)
            given = list(feature_names)
            if given != expected:
                missing = [c for c in expected if c not in given]
                extra = [c for c in given if c not in expected]
                order_bad = given != expected and not missing and not extra
                detail = []
                if missing:
                    detail.append(f"отсутствуют: {missing[:6]}{'...' if len(missing) > 6 else ''}")
                if extra:
                    detail.append(f"лишние: {extra[:6]}{'...' if len(extra) > 6 else ''}")
                if order_bad:
                    detail.append("порядок колонок отличается")
                mismatches.append("набор признаков: " + "; ".join(detail))
        if mismatches:
            raise ModelCompatibilityError(mismatches)

    def predict_proba(self, bars: Sequence[Bar], *, local_timezone: str = "UTC") -> np.ndarray:
        """Прогноз P(y=1) (бинарная цель) или значения (регрессия) по барам.

        Признаки строятся тем же FeatureBuilder по манифесту модели; список и
        порядок колонок сверяются до прогноза. Строки прогрева (первые
        spec.warmup_bars) вызывающий обязан отбросить сам — их признаки неполны.
        """
        builder = FeatureBuilder(self.spec, local_timezone=local_timezone)
        frame = builder.build(list(bars))
        expected = list(self.feature_names)
        if list(frame.columns) != expected:
            self.validate_compatibility(feature_names=expected)  # гарантированно бросит
            raise ModelCompatibilityError(["рассинхрон колонок при прогнозе"])
        matrix = frame[expected].to_numpy(dtype=np.float64)
        predictions = self.booster.predict(matrix)
        return np.asarray(predictions, dtype=np.float64)


def save_model_bundle(
    model_dir: "str | Path",
    booster: Any,
    metadata: dict[str, Any],
    spec: FeatureSpec,
    *,
    oos_predictions_csv: str | None = None,
) -> Path:
    """Сохраняет артефакт модели. Возвращает путь каталога."""
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / MODEL_ARTIFACT_NAME))
    metadata = dict(metadata)
    metadata.setdefault("schema_version", METADATA_SCHEMA_VERSION)
    metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    metadata["feature_manifest"] = spec.to_manifest()
    (directory / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (directory / MANIFEST_NAME).write_text(
        json.dumps(spec.to_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if oos_predictions_csv:
        (directory / PREDICTIONS_NAME).write_text(oos_predictions_csv, encoding="utf-8")
    return directory


def load_model_bundle(model_dir: "str | Path", *, require_lightgbm: bool = True) -> ModelBundle:
    """Загружает артефакт; проверяет схему метаданных и манифест."""
    directory = Path(model_dir)
    for name in (MODEL_ARTIFACT_NAME, METADATA_NAME, MANIFEST_NAME):
        if not (directory / name).exists():
            raise FileNotFoundError(f"Артефакт модели неполон: отсутствует {name} в {directory}")
    metadata = json.loads((directory / METADATA_NAME).read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", 0)) != METADATA_SCHEMA_VERSION:
        raise ModelCompatibilityError(
            [f"schema_version метаданных: {metadata.get('schema_version')!r} != {METADATA_SCHEMA_VERSION}"]
        )
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    spec = FeatureSpec.from_manifest(manifest)
    booster = None
    if require_lightgbm:
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(directory / MODEL_ARTIFACT_NAME))
    return ModelBundle(model_dir=directory, booster=booster, metadata=metadata, spec=spec)
