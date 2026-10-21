"""Целевая переменная (docs/decisions.md D5, docs/ml.md §3).

Для бара t и горизонта H (horizon_bars таймфрейма):

    fwd_return(t) = close[t+H] / close[t] - 1

Варианты (LabelConfig.kind):
- ``fwd_return_class``: y = 1 при fwd_return > +band; y = 0 при fwd_return < -band;
  в полосе [-band, +band] — NaN (нейтрально, исключается из обучения). band задаётся
  в базисных пунктах (neutral_band_bps) и по смыслу ≈ издержки сделки;
- ``fwd_return_reg``: y = fwd_return (регрессия);
- ``triple_barrier``: НЕ реализовано (отложено до этапа B1+, decisions D5) — явная
  ошибка, не молчаливая подмена.

ВАЖНО (architecture §6.2): метка использует close[t+H], т.е. будущее относительно
момента решения. Модуль импортируется ТОЛЬКО из ml/dataset (сборка датасета) и
тестов; в признаках, стратегии и inference-пути меток нет физически.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["TargetInfo", "TargetSpec", "build_target"]


@dataclass(frozen=True)
class TargetSpec:
    kind: str  # fwd_return_class | fwd_return_reg
    horizon_bars: int
    neutral_band_bps: float

    def __post_init__(self) -> None:
        if self.kind not in ("fwd_return_class", "fwd_return_reg"):
            raise ValueError(
                f"Неизвестный вид целевой переменной: {self.kind!r} "
                "(triple_barrier не реализован — см. from_config)"
            )
        if self.horizon_bars < 1:
            raise ValueError(f"horizon_bars должен быть >= 1, получено {self.horizon_bars}")
        if self.neutral_band_bps < 0:
            raise ValueError(f"neutral_band_bps >= 0, получено {self.neutral_band_bps}")

    @property
    def band(self) -> float:
        return self.neutral_band_bps / 10_000.0

    @classmethod
    def from_config(cls, label_cfg, horizon_bars: int) -> "TargetSpec":
        kind = getattr(label_cfg, "kind", "fwd_return_class")
        if kind == "triple_barrier":
            raise NotImplementedError(
                "Целевая переменная triple_barrier не реализована в фазе 3 (отложена до "
                "этапа B1+ согласно docs/decisions.md D5). Используйте fwd_return_class "
                "или fwd_return_reg"
            )
        if kind not in ("fwd_return_class", "fwd_return_reg"):
            raise ValueError(f"Неизвестный вид целевой переменной: {kind!r}")
        if horizon_bars < 1:
            raise ValueError(f"horizon_bars должен быть >= 1, получено {horizon_bars}")
        return cls(kind=kind, horizon_bars=horizon_bars, neutral_band_bps=float(label_cfg.neutral_band_bps))


@dataclass(frozen=True)
class TargetInfo:
    spec: TargetSpec
    n_total: int
    n_valid: int
    n_neutral_dropped: int
    n_tail_dropped: int
    class_balance: dict[str, float]


def compute_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """close[t+H]/close[t] - 1. Последние H строк — NaN (будущее недоступно).

    Это единственное место, где намеренно используется shift(-H); оно вызывается
    только при сборке датасета для обучения/валидации (см. предупреждение модуля).
    """
    if horizon < 1:
        raise ValueError("horizon >= 1")
    future = close.shift(-horizon)
    return future / close - 1.0


def build_target(close: pd.Series, spec: TargetSpec) -> tuple[pd.Series, TargetInfo]:
    """Возвращает (y, info). y — float Series с NaN для неиспользуемых строк.

    Для классификации NaN означают: (а) последние H строк — нет будущего close;
    (б) движение внутри нейтральной полосы — класс не определяется (исключается).
    """
    fwd = compute_forward_return(close, spec.horizon_bars)
    n_total = int(len(close))
    tail_nan = int(fwd.tail(spec.horizon_bars).isna().sum()) if n_total else 0

    if spec.kind == "fwd_return_reg":
        y = fwd.astype("float64")
        n_valid = int(y.notna().sum())
        info = TargetInfo(
            spec=spec,
            n_total=n_total,
            n_valid=n_valid,
            n_neutral_dropped=0,
            n_tail_dropped=tail_nan,
            class_balance={},
        )
        return y, info

    band = spec.band
    y = pd.Series(np.nan, index=close.index, dtype="float64")
    y[fwd > band] = 1.0
    y[fwd < -band] = 0.0
    neutral = fwd.notna() & (fwd >= -band) & (fwd <= band)
    n_neutral = int(neutral.sum())
    n_valid = int(y.notna().sum())
    balance: dict[str, float] = {}
    if n_valid > 0:
        ones = float((y == 1.0).sum())
        balance = {"up": ones / n_valid, "down": (n_valid - ones) / n_valid}
    info = TargetInfo(
        spec=spec,
        n_total=n_total,
        n_valid=n_valid,
        n_neutral_dropped=n_neutral,
        n_tail_dropped=tail_nan,
        class_balance=balance,
    )
    return y, info
