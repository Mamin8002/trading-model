"""Спецификация набора признаков (версионирование и манифест).

Набор ``v1`` (docs/ml.md §2, architecture §5.4): табличные причинные признаки
из OHLCV. Периоды индикаторов настраиваются конфигом (FeatureParamsConfig);
список колонок выводится из параметров детерминированно и сохраняется в манифест
модели — рассинхрон «модель ↔ признаки» отклоняется загрузчиком (decisions D10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FeatureSpec"]


@dataclass(frozen=True)
class FeatureSpec:
    """Версия набора + параметры + фиксированный порядок колонок + warmup."""

    version: str
    params: dict[str, Any] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()
    warmup_bars: int = 0

    @classmethod
    def from_config(cls, features_cfg) -> "FeatureSpec":
        """Строит спецификацию из FeaturesConfig (config/schema.py)."""
        if features_cfg.set != "v1":
            raise ValueError(
                f"Неизвестный набор признаков {features_cfg.set!r}; реализован только 'v1'"
            )
        p = features_cfg.params
        names: list[str] = []

        names += [f"ret_{lag}" for lag in p.return_lags]
        for period in p.ema_periods:
            names += [f"ema_ratio_{period}", f"ema_slope_{period}"]
        names.append(f"rsi_{p.rsi_period}")
        names += ["macd_norm", "macd_signal_norm", "macd_hist_norm"]
        names.append(f"atr_ratio_{p.atr_period}")
        names += ["bar_range", "bar_body", "upper_wick", "lower_wick", "close_position", "gap_prev"]
        names += [f"vol_{w}" for w in p.vol_windows]
        if len(p.vol_windows) >= 2:
            names.append("vol_ratio")
        names += [f"park_vol_{w}" for w in p.vol_windows]
        names += [
            f"volume_ratio_{p.volume_window}",
            f"volume_trend_{p.volume_window}_{p.volume_trend_window}",
        ]
        names += ["minute_of_day", "day_of_week"]

        warmup = max(
            max(p.return_lags),
            max(p.ema_periods) + 1,
            p.macd_slow + p.macd_signal,
            p.rsi_period * 2,
            p.atr_period * 2 + 1,
            max(p.vol_windows) + 1,
            p.volume_trend_window + 1,
            2,
        )
        params_dict = p.model_dump(mode="json")
        return cls(
            version="v1",
            params=params_dict,
            feature_names=tuple(names),
            warmup_bars=int(warmup),
        )

    def to_manifest(self) -> dict[str, Any]:
        """Манифест для сохранения вместе с моделью (docs/ml.md §6)."""
        return {
            "version": self.version,
            "params": self.params,
            "feature_names": list(self.feature_names),
            "warmup_bars": self.warmup_bars,
        }

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "FeatureSpec":
        return cls(
            version=str(manifest["version"]),
            params=dict(manifest.get("params", {})),
            feature_names=tuple(manifest["feature_names"]),
            warmup_bars=int(manifest.get("warmup_bars", 0)),
        )
