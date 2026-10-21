"""Источники сигналов для симулятора (docs/simulation.md §2).

Сигнал привязан к ``ts_open`` бара решения и интерпретируется ПОСЛЕ закрытия
этого бара; исполнение — не раньше следующего бара (правила — в EventSimulator).

- ``ModelSignalSource``: вероятности из сохранённого артефакта модели фазы 3
  (ModelBundle), пороги — из явного конфига ``simulation.signal``; версия модели
  для журнала = experiment_id:fingerprint артефакта;
- ``ScriptedSignalSource``: детерминированные сигналы для тестов/ручных проверок
  (model_version="scripted"); все метки обязаны совпадать с ts_open баров —
  иначе ошибка (никаких «подгонок» времени).

Стратегическое правило фазы 4: long-only, один символ, один вход за раз,
реверсы не поддерживаются (встречный BUY при открытой позиции игнорируется
с предупреждением в журнал).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from ..config.schema import SignalConfig
from ..data.bar import Bar
from ..execution.interfaces import Side, Signal

__all__ = ["ModelSignalSource", "ScriptedSignalSource", "SignalSource"]


class SignalSource(Protocol):
    model_version: str

    def signals_for(self, bars: Sequence[Bar]) -> dict[datetime, Signal]:
        """Карта ts_open бара -> сигнал (отсутствующий бар = нет сигнала)."""
        ...


@dataclass
class ScriptedSignalSource:
    """Детерминированные сигналы (тесты, ручные сценарии)."""

    signals: Sequence[Signal]
    symbol: str = ""
    timeframe: str = ""
    model_version: str = "scripted"

    def signals_for(self, bars: Sequence[Bar]) -> dict[datetime, Signal]:
        bar_stamps = {b.ts_open for b in bars}
        unknown = [s.ts_open.isoformat() for s in self.signals if s.ts_open not in bar_stamps]
        if unknown:
            raise ValueError(
                f"ScriptedSignalSource: метки сигналов не совпадают ни с одним баром: {unknown[:5]} "
                "(подгонка времени запрещена — исправьте сценарий)"
            )
        out: dict[datetime, Signal] = {}
        for signal in self.signals:
            filled = Signal(
                ts_open=signal.ts_open,
                side=signal.side,
                symbol=signal.symbol or self.symbol,
                timeframe=signal.timeframe or self.timeframe,
                model_version=signal.model_version or self.model_version,
                proba=signal.proba,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                meta=dict(signal.meta),
            )
            out[signal.ts_open] = filled
        return out


class ModelSignalSource:
    """Сигналы из артефакта модели фазы 3 (read-only, без сети)."""

    def __init__(
        self,
        bundle,  # ml.registry.ModelBundle (duck-typing против цикла импортов)
        signal_cfg: SignalConfig,
        *,
        symbol: str,
        timeframe: str,
        local_timezone: str = "UTC",
    ) -> None:
        self.bundle = bundle
        self.signal_cfg = signal_cfg
        self.symbol = symbol
        self.timeframe = timeframe
        self.local_timezone = local_timezone
        metadata = bundle.metadata
        self.model_version = (
            f"{metadata.get('experiment_id', 'unknown')}:{str(metadata.get('dataset', {}).get('fingerprint', ''))[:8]}"
        )

    def signals_for(self, bars: Sequence[Bar]) -> dict[datetime, Signal]:
        bars_list = list(bars)
        if not bars_list:
            return {}
        warmup = int(self.bundle.spec.warmup_bars)
        proba = self.bundle.predict_proba(bars_list, local_timezone=self.local_timezone)
        out: dict[datetime, Signal] = {}
        for index, bar in enumerate(bars_list):
            if index < warmup:
                continue  # прогрев: признаки неполные — сигналы не выдаются
            p = float(proba[index])
            side: Side | None = None
            if p >= self.signal_cfg.entry_threshold:
                side = Side.BUY
            elif self.signal_cfg.exit_threshold is not None and p <= self.signal_cfg.exit_threshold:
                side = None  # явный выход
            else:
                continue
            if side is None and self.signal_cfg.exit_threshold is None:
                continue
            # для выхода side=None; отличаем «выход» от «нет сигнала» флагом в meta
            is_exit = side is None
            out[bar.ts_open] = Signal(
                ts_open=bar.ts_open,
                side=side,
                symbol=self.symbol,
                timeframe=self.timeframe,
                model_version=self.model_version,
                proba=p,
                stop_price=None,   # стоп/цель рассчитывает движок из simulation.signal.*_bps
                target_price=None,
                meta={"kind": "exit" if is_exit else "entry", "threshold": self.signal_cfg.entry_threshold},
            )
        return out
