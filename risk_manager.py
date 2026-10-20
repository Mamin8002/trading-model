"""
risk_manager.py — Расчёт SL/TP, размера позиции и мониторинг активной сделки.

Подход:
  * Дистанция стопа = atr_sl_mult * ATR (адаптивна к волатильности актива).
  * Дистанция тейка = atr_tp_mult * ATR.
  * Position sizing: риск на сделку = risk_per_trade_pct% от текущего капитала.
        qty = (equity * risk%) / |entry - stop|   (округление до лота)
    с потолком max_position_pct% депозита.
  * Проверка TP/SL по барам внутри цикла сопровождения; таймаут удержания.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from config import BotConfig

log = logging.getLogger("risk")


@dataclass
class TradePlan:
    side: str                 # LONG | SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    qty: int                  # количество штук (лотность учтена в bot)
    risk_amount: float        # RUB, которые потеряем при стопе
    rr_ratio: float           # reward/risk


class RiskManager:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    # ------------------------------------------------------------- levels
    def compute_levels(self, entry: float, atr_abs: float, side: str) -> tuple[float, float]:
        """Стоп и тейк на основе ATR."""
        if atr_abs <= 0 or math.isnan(atr_abs):
            # страховка: фиксированный 0.5%/1% если ATR не посчитался
            sl_dist, tp_dist = entry * 0.005, entry * 0.01
        else:
            sl_dist = self.cfg.atr_sl_mult * atr_abs
            tp_dist = self.cfg.atr_tp_mult * atr_abs
        if side == "LONG":
            return entry - sl_dist, entry + tp_dist
        return entry + sl_dist, entry - tp_dist

    # ------------------------------------------------------------ sizing
    def position_size(self, equity: float, entry: float, stop: float,
                      lot_size: int = 1) -> int:
        """Классический fixed-fractional sizing с учётом расстояния до стопа."""
        risk_rub = equity * self.cfg.risk_per_trade_pct / 100.0
        per_unit_risk = abs(entry - stop)
        if per_unit_risk <= 0:
            return 0
        qty = risk_rub / per_unit_risk
        # потолок по доле депозита в позиции
        max_qty_by_notional = equity * self.cfg.max_position_pct / 100.0 / entry
        qty = min(qty, max_qty_by_notional)
        # округление вниз до целого лота
        lots = math.floor(qty / lot_size)
        return max(lots * lot_size, 0)

    # ------------------------------------------------------- build plan
    def make_plan(self, equity: float, entry: float, atr_abs: float, side: str,
                  lot_size: int = 1) -> Optional[TradePlan]:
        sl, tp = self.compute_levels(entry, atr_abs, side)
        qty = self.position_size(equity, entry, sl, lot_size)
        if qty <= 0:
            log.warning("Размер позиции = 0 (слишком широкий стоп или мал капитал). Сигнал пропущен.")
            return None
        risk_amount = qty * abs(entry - sl)
        rr = abs(tp - entry) / max(abs(entry - sl), 1e-9)
        return TradePlan(side=side, entry_price=entry, stop_loss=sl, take_profit=tp,
                         qty=qty, risk_amount=risk_amount, rr_ratio=rr)

    # -------------------------------------------- monitoring of open trade
    @staticmethod
    def check_exit(side: str, high: float, low: float, close: float,
                   stop: float, target: float) -> Optional[str]:
        """
        По данным закрытого бара определяет, был ли достигнут SL или TP.
        Возвращает 'SL' | 'TP' | None. При пересечении обоих за один бар — приоритет SL.
        """
        if side == "LONG":
            if low <= stop:
                return "SL"
            if high >= target:
                return "TP"
        else:
            if high >= stop:
                return "SL"
            if low <= target:
                return "TP"
        return None

    @staticmethod
    def exit_price_for(reason: str, plan_stop: float, plan_target: float,
                       bar_close: float, slippage_pct: float, side: str) -> float:
        """
        Реалистичная цена выхода:
          * TP/SL исполняются лимитно по уровню + проскальзывание против нас;
          * таймаут/рынок — по close + проскальзывание.
        """
        base = {"SL": plan_stop, "TP": plan_target}.get(reason, bar_close)
        slip = base * slippage_pct / 100.0
        # выход из LONG = продажа (проскальзывание вниз), из SHORT = покупка (вверх)
        return base - slip if side == "LONG" else base + slip
