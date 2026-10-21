"""DatasetAssembler: бары -> (X, y, контекст исполнения) для walk-forward (docs/ml.md §3-4).

Состав строки датасета:
- X[t]  — вектор признаков, рассчитанный по барам <= t (решение принимается на
  закрытии бара t);
- y[t]  — целевая переменная из close[t+H] (только для обучения/валидации);
- close_t, open_t1, open_t2 — цены для оценочной симуляции исполнения:
  сигнал на закрытии t -> вход по open[t+1] -> выход по open[t+2]
  (architecture §6.6: исполнение не раньше открытия следующего бара);
- prev_ret — доходность последнего НАБЛЮДЕННОГО бара (close_t/close_{t-1} - 1),
  используется только базовым ориентиром «предыдущий знак» (прошлое, не будущее).

Строки с NaN в признаках (прогрев), NaN в метке (хвост H баров + нейтральная
полоса для классификации) и без доступных цен исполнения (хвост 2 бара) —
исключаются; счётчики исключений попадают в DatasetInfo (ничего не фабрикается).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..data.bar import Bar, Timeframe
from ..features.builder import FeatureBuilder
from ..features.spec import FeatureSpec
from ..labels.target import TargetInfo, TargetSpec, build_target
from .errors import TrainingError

__all__ = ["DatasetInfo", "MlDataset", "assemble_dataset"]


@dataclass(frozen=True)
class DatasetInfo:
    n_bars: int
    n_rows: int
    warmup_dropped: int
    neutral_dropped: int
    tail_dropped: int
    execution_tail_dropped: int
    feature_manifest: dict[str, Any] = field(default_factory=dict)
    target_info: TargetInfo | None = None


@dataclass(frozen=True)
class MlDataset:
    X: np.ndarray                     # (n_rows, n_features) float64
    y: np.ndarray                     # (n_rows,) float64; классификация: 0/1
    ts: np.ndarray                    # (n_rows,) datetime64[us, UTC] — метки баров решения
    close_t: np.ndarray               # close бара решения
    open_t1: np.ndarray               # open следующего бара (точка входа)
    open_t2: np.ndarray               # open бара после следующего (точка выхода 1-баровой позиции)
    prev_ret: np.ndarray              # доходность предыдущего наблюдаемого бара (для базового ориентира)
    feature_names: tuple[str, ...]
    spec: FeatureSpec
    target: TargetSpec
    timeframe: str
    symbol: str
    info: DatasetInfo

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def is_classification(self) -> bool:
        return self.target.kind == "fwd_return_class"


def assemble_dataset(
    bars: Sequence[Bar],
    *,
    timeframe: "Timeframe | str",
    horizon_bars: int,
    lookback_bars: int,
    label_cfg,
    features_cfg,
    symbol: str = "",
    local_timezone: str = "Europe/Moscow",
) -> MlDataset:
    """Собирает датасет из финальных баров одного символа и таймфрейма."""
    tf = Timeframe.parse(timeframe)
    if not bars:
        raise TrainingError("assemble_dataset: пустая история баров")

    spec = FeatureSpec.from_config(features_cfg)
    if lookback_bars < spec.warmup_bars:
        raise TrainingError(
            f"lookback_bars={lookback_bars} таймфрейма {tf.value} меньше прогрева набора "
            f"признаков v1 ({spec.warmup_bars} баров): увеличьте lookback_bars в конфиге "
            "(значения должны оставаться явными, молчаливый подбор запрещён)"
        )

    builder = FeatureBuilder(spec, local_timezone=local_timezone)
    frame = builder.build(bars)

    close = pd.Series([b.close for b in bars], index=frame.index, dtype="float64")
    open_ = pd.Series([b.open for b in bars], index=frame.index, dtype="float64")

    target = TargetSpec.from_config(label_cfg, horizon_bars)
    y_series, target_info = build_target(close, target)

    open_t1 = open_.shift(-1)
    open_t2 = open_.shift(-2)
    prev_ret = close / close.shift(1) - 1.0

    features_ready = frame.notna().all(axis=1)
    label_ready = y_series.notna()
    execution_ready = open_t1.notna() & open_t2.notna() & close.notna()
    prev_ready = prev_ret.notna()

    n_bars = len(bars)
    warmup_dropped = int((~features_ready).sum())
    tail_dropped = int(target_info.n_tail_dropped)
    neutral_dropped = int(target_info.n_neutral_dropped)
    valid = features_ready & label_ready & execution_ready & prev_ready
    execution_tail_dropped = int((features_ready & label_ready & ~execution_ready).sum())

    if int(valid.sum()) == 0:
        raise TrainingError(
            "assemble_dataset: после исключения прогрева/хвостов/нейтральных строк "
            f"не осталось данных (баров: {n_bars}, warmup: {warmup_dropped}, "
            f"нейтраль: {neutral_dropped}, хвост метки: {tail_dropped})"
        )

    sub_frame = frame.loc[valid]
    info = DatasetInfo(
        n_bars=n_bars,
        n_rows=int(valid.sum()),
        warmup_dropped=warmup_dropped,
        neutral_dropped=neutral_dropped,
        tail_dropped=tail_dropped,
        execution_tail_dropped=execution_tail_dropped,
        feature_manifest=spec.to_manifest(),
        target_info=target_info,
    )
    return MlDataset(
        X=sub_frame.to_numpy(dtype=np.float64, copy=True),
        y=y_series.loc[valid].to_numpy(dtype=np.float64, copy=True),
        ts=np.array([ts.to_datetime64() for ts in sub_frame.index], dtype="datetime64[us]"),
        close_t=close.loc[valid].to_numpy(dtype=np.float64),
        open_t1=open_t1.loc[valid].to_numpy(dtype=np.float64),
        open_t2=open_t2.loc[valid].to_numpy(dtype=np.float64),
        prev_ret=prev_ret.loc[valid].to_numpy(dtype=np.float64),
        feature_names=spec.feature_names,
        spec=spec,
        target=target,
        timeframe=tf.value,
        symbol=symbol or (bars[0].symbol if bars else ""),
        info=info,
    )


def dataset_fingerprint(dataset: MlDataset) -> str:
    """Детерминированный идентификатор датасета для манифеста модели."""
    import hashlib

    hasher = hashlib.sha256()
    hasher.update(dataset.symbol.encode("utf-8"))
    hasher.update(dataset.timeframe.encode("utf-8"))
    hasher.update(str(dataset.n_samples).encode())
    hasher.update(str(dataset.ts[0]).encode())
    hasher.update(str(dataset.ts[-1]).encode())
    hasher.update(np.ascontiguousarray(dataset.X[:1]).tobytes())
    hasher.update(dataset.info.feature_manifest["version"].encode())
    return hasher.hexdigest()[:16]


def dataset_period(dataset: MlDataset) -> tuple[datetime, datetime]:
    first = pd.Timestamp(dataset.ts[0]).to_pydatetime()
    last = pd.Timestamp(dataset.ts[-1]).to_pydatetime()
    return first, last
