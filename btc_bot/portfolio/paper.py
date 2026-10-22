"""Paper trading: виртуальные позиции БЕЗ отправки ордеров и без приватного API.

Принципы (спецификация Этапа 5):
  * действия: OPEN_VIRTUAL / HOLD / CLOSE / PARTIAL_CLOSE / ABSTAIN - каждое с причиной;
  * издержки задаются ЯВНО (fee_pct, slippage_pct) и применяются к каждой сделке;
  * правила выхода (TP/SL/max_hold/horizon/partial-TP) фиксируются СНИМКОМ в момент
    открытия позиции и не меняются молча во время работы;
  * сигнал модели НЕ открывает позицию автоматически без выполнения фильтров:
    данные fresh, уверенность >= порога, правила выхода определены, cooldown истёк,
    нет открытой конфликтующей позиции (max_open_positions=1, только LONG для spot);
  * spike-алерты (быстрые движения) НЕ открывают позиции - только сопровождение;
  * история хранится в журнале (отдельная коллекция paper_positions) и переживает
    перезапуск; PnL - по виртуальным сделкам с учётом комиссий и слиппеджа.

Учёт (cash-flow): вход qty*entry_eff + комиссия; выход qty_c*exit_eff - комиссия;
realized_pnl = proceeds - доля невозвращённой стоимости входа. Баланс =
initial_balance + сумма realized по всем позициям.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from .journal import Journal, _utc_iso

A_OPEN = "OPEN_VIRTUAL"
A_HOLD = "HOLD"
A_CLOSE = "CLOSE"
A_PARTIAL = "PARTIAL_CLOSE"
A_ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class PaperAction:
    action: str
    reason: str
    position_id: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[float] = None
    pnl_quote: Optional[float] = None
    rule: Optional[str] = None      # какое правило выхода сработало
    ts_ms: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class PaperTrader:
    def __init__(
        self,
        journal: Journal,
        *,
        symbol: str,
        category: str,
        fee_pct: float,
        slippage_pct: float,
        order_size_quote: float,
        exit_rules: Optional[dict],
        entry_min_confidence: Optional[float],
        cooldown_bars: int,
        max_open_positions: int,
        intrabar_exits: bool,
        exit_at_horizon: bool,
        partial_tp: Optional[dict],
        interval_step_ms: int,
        horizon_bars: int,
        clock_ms: Optional[Callable[[], int]] = None,
    ):
        self.journal = journal
        self.symbol = symbol
        self.category = category
        self.fee = fee_pct / 100.0
        self.slip = slippage_pct / 100.0
        self.order_size_quote = order_size_quote
        self.exit_rules = dict(exit_rules or {})
        self.entry_min_confidence = entry_min_confidence
        self.cooldown_bars = cooldown_bars
        self.max_open = max_open_positions
        self.intrabar_exits = intrabar_exits
        self.exit_at_horizon = exit_at_horizon
        self.partial_tp = dict(partial_tp or {})
        self.step_ms = interval_step_ms
        self.horizon_bars = horizon_bars
        self._clock = clock_ms or (lambda: int(time.time() * 1000))
        self.last_action: Optional[PaperAction] = None

    # -- helpers ---------------------------------------------------------
    def open_positions(self) -> list[dict]:
        return [p for p in self.journal.paper_positions("OPEN") if p["symbol"] == self.symbol]

    def _log(self, act: PaperAction) -> PaperAction:
        self.last_action = act
        self.journal.log_paper_action(act.action, act.to_dict())
        return act

    def _rules_defined(self) -> bool:
        return (self.exit_at_horizon
                or any(self.exit_rules.get(k) is not None
                       for k in ("take_profit_pct", "stop_loss_pct", "max_hold_bars")))

    def _cooldown_left(self, now_ms: int) -> int:
        closed = [p for p in self.journal.paper_positions("CLOSED") if p["symbol"] == self.symbol]
        if not closed or self.cooldown_bars <= 0:
            return 0
        last_close = max(int(p.get("closed_at_ms") or 0) for p in closed)
        elapsed = (now_ms - last_close) // self.step_ms
        return max(0, self.cooldown_bars - int(elapsed))

    def _do_close(self, pos: dict, fill_px: float, rule: str, reason: str,
                  ts_ms: int, quantity: Optional[float] = None) -> PaperAction:
        qty_open = float(pos["quantity_open"])
        qty = min(float(quantity), qty_open) if quantity is not None else qty_open
        exit_eff = fill_px * (1.0 - self.slip)
        proceeds = qty * exit_eff
        exit_fee = proceeds * self.fee
        cash_in = proceeds - exit_fee
        cost_open = float(pos["entry_cost_open_quote"])
        cost_portion = cost_open * (qty / qty_open)
        realized = cash_in - cost_portion
        pos["entry_cost_open_quote"] = cost_open - cost_portion
        pos["quantity_open"] = qty_open - qty
        pos["fills"].append({
            "ts_ms": ts_ms, "ts_utc": _utc_iso(ts_ms),
            "type": "paper_exit_partial" if qty < qty_open - 1e-12 else "paper_exit_full",
            "price": fill_px, "quantity": qty, "rule": rule, "reason": reason,
            "fee_quote": exit_fee, "realized_quote": realized,
        })
        pos["realized_pnl_quote"] = float(pos.get("realized_pnl_quote") or 0.0) + realized
        exits = [f for f in pos["fills"] if f["type"].startswith("paper_exit")]
        total_q = sum(f["quantity"] for f in exits)
        pos["avg_exit_price"] = (sum(f["price"] * f["quantity"] for f in exits) / total_q) if total_q else None
        fully = pos["quantity_open"] <= 1e-12
        hist_reason = f"{rule}: {reason}; результат {realized:+.4f} USDT"
        if fully:
            pos["status"] = "CLOSED"
            pos["quantity_open"] = 0.0
            pos["closed_at_ms"] = ts_ms
            pos["closed_at_utc"] = _utc_iso(ts_ms)
            r = float(pos["realized_pnl_quote"])
            pos["outcome"] = "WIN" if r > 0 else ("LOSS" if r < 0 else "FLAT")
            hist_reason += f" -> позиция ЗАКРЫТА ({pos['outcome']})"
        pos["status_history"].append({"ts_ms": ts_ms, "ts_utc": _utc_iso(ts_ms),
                                      "event": "PAPER_CLOSED" if fully else "PAPER_PARTIAL_CLOSE",
                                      "reason": hist_reason})
        self.journal.save_paper_position(pos)
        return self._log(PaperAction(
            action=A_CLOSE if fully else A_PARTIAL,
            reason=f"[{rule}] {reason}",
            position_id=pos["position_id"], price=fill_px, quantity=qty,
            pnl_quote=round(realized, 6), rule=rule, ts_ms=ts_ms))

    def _exit_levels(self, pos: dict) -> tuple[Optional[float], Optional[float]]:
        rules = pos.get("rules_snapshot") or {}
        entry = float(pos["entry_price"])
        tp = rules.get("take_profit_pct")
        sl = rules.get("stop_loss_pct")
        tp_lvl = entry * (1 + tp / 100.0) if tp is not None else None
        sl_lvl = entry * (1 - sl / 100.0) if sl is not None else None
        return tp_lvl, sl_lvl

    # -- события -----------------------------------------------------------
    def on_signal(self, sig, price_now: Optional[float]) -> Optional[PaperAction]:
        """Реакция на новый сигнал модели. sig: Signal (engine)."""
        ts = self._clock()
        px = float(price_now if price_now is not None else sig.price_at_signal)
        opens = self.open_positions()

        if not sig.data_fresh:
            return self._log(PaperAction(A_ABSTAIN, "данные не актуальны - виртуальные сделки не совершаются",
                                         ts_ms=ts))
        if opens:
            pos = opens[-1]
            if sig.action == "SELL":
                return self._do_close(pos, px, "model_counter_signal",
                                      f"встречный сигнал модели SELL (conf {sig.confidence:.3f})", ts)
            return self._log(PaperAction(
                A_HOLD, f"позиция {pos['position_id']} открыта в том же направлении; "
                        f"сигнал {sig.action} - удерживаем", position_id=pos["position_id"], ts_ms=ts))

        if sig.action == "SELL":
            return self._log(PaperAction(A_ABSTAIN,
                                         "SELL без открытой виртуальной позиции: шорт не открывается (spot)",
                                         ts_ms=ts))
        if sig.action != "BUY":
            return self._log(PaperAction(A_ABSTAIN, f"сигнал {sig.action} - входа нет", ts_ms=ts))

        # фильтры входа
        if self.entry_min_confidence is not None and sig.confidence < self.entry_min_confidence:
            return self._log(PaperAction(A_ABSTAIN,
                                         f"уверенность {sig.confidence:.3f} < entry_min_confidence "
                                         f"{self.entry_min_confidence}", ts_ms=ts))
        if not self._rules_defined():
            return self._log(PaperAction(A_ABSTAIN,
                                         "правила выхода не заданы (positions.exit_rules пустые и "
                                         "paper.exit_at_horizon=false) - вход воздержан", ts_ms=ts))
        if len(opens) >= self.max_open:
            return self._log(PaperAction(A_ABSTAIN, "лимит открытых виртуальных позиций исчерпан", ts_ms=ts))
        cd = self._cooldown_left(ts)
        if cd > 0:
            return self._log(PaperAction(A_ABSTAIN, f"cooldown после закрытия: ещё {cd} бар(ов)", ts_ms=ts))

        # OPEN_VIRTUAL
        entry_eff = px * (1.0 + self.slip)
        qty = self.order_size_quote / entry_eff
        entry_fee = qty * entry_eff * self.fee
        entry_cost = qty * entry_eff + entry_fee
        data = self.journal._load()
        pid = self.journal.next_paper_id(data)
        pos = {
            "position_id": pid, "kind": "paper",
            "symbol": self.symbol, "category": self.category, "direction": "LONG",
            "status": "OPEN",
            "opened_at_ms": ts, "opened_at_utc": _utc_iso(ts),
            "entry_price": px, "entry_eff_price": entry_eff,
            "quantity_initial": qty, "quantity_open": qty,
            "entry_cost_quote": entry_cost, "entry_cost_open_quote": entry_cost,
            "entry_fee_quote": entry_fee,
            "source_signal": {
                "signal_id": sig.signal_id, "action": sig.action, "confidence": sig.confidence,
                "signal_created_at_ms": sig.created_at_ms, "bar_end_ms": sig.bar_end_ms,
                "price_at_signal": sig.price_at_signal,
                "horizon_bars": sig.horizon_bars, "horizon_minutes": sig.horizon_minutes,
                "model_version": sig.model_version,
            },
            "rules_snapshot": {
                **self.exit_rules,
                "exit_at_horizon": self.exit_at_horizon,
                "horizon_bars": self.horizon_bars,
                "partial_tp": dict(self.partial_tp),
                "fee_pct": self.fee * 100, "slippage_pct": self.slip * 100,
            },
            "entry_threshold": self.entry_min_confidence,
            "fills": [{
                "ts_ms": ts, "ts_utc": _utc_iso(ts), "type": "paper_entry",
                "price": px, "quantity": qty, "fee_quote": entry_fee,
                "reason": f"BUY conf {sig.confidence:.3f}, сигнал {sig.signal_id}",
            }],
            "status_history": [{
                "ts_ms": ts, "ts_utc": _utc_iso(ts), "event": "PAPER_OPENED",
                "reason": (f"виртуальный вход по сигналу {sig.signal_id} "
                           f"(conf {sig.confidence:.3f}, модель {sig.model_version}); "
                           f"{qty:.8f} @ {px:,.2f} (eff {entry_eff:,.2f}), комиссия {entry_fee:.4f}"),
            }],
            "realized_pnl_quote": 0.0, "partial_done": False,
            "closed_at_ms": None, "avg_exit_price": None, "outcome": None,
        }
        data["paper_positions"].append(pos)
        self.journal._save(data)
        return self._log(PaperAction(A_OPEN, f"BUY conf {sig.confidence:.3f} -> виртуальная покупка",
                                     position_id=pid, price=px, quantity=qty, ts_ms=ts))

    def _check_exits(self, pos: dict, ts_ms: int, *, bar=None, tick_px: Optional[float] = None
                     ) -> Optional[PaperAction]:
        """Проверка правил выхода. Возвращает действие или None."""
        if pos.get("symbol") != self.symbol:
            return None
        tp_lvl, sl_lvl = self._exit_levels(pos)
        entry_ts = int(pos["opened_at_ms"])
        rules = pos.get("rules_snapshot") or {}

        # 1) SL/TP: тиковый режим (LIVE) или баровый (TEST/фолбэк)
        if tick_px is not None and self.intrabar_exits:
            if sl_lvl is not None and tick_px <= sl_lvl:
                return self._do_close(pos, sl_lvl, "stop_loss_pct",
                                      f"тик {tick_px:,.2f} <= уровня SL {sl_lvl:,.2f}", ts_ms)
            if tp_lvl is not None and tick_px >= tp_lvl:
                return self._do_close(pos, tp_lvl, "take_profit_pct",
                                      f"тик {tick_px:,.2f} >= уровня TP {tp_lvl:,.2f}", ts_ms)
        if bar is not None and bar.end_ms > entry_ts:  # бар входа не проверяем (вход на его close)
            if not self.intrabar_exits or tick_px is None:
                pass  # баровые SL/TP проверяются ниже независимо; тики лишь приоритетнее в LIVE
            if sl_lvl is not None and bar.low <= sl_lvl:
                fill = min(bar.open, sl_lvl)
                return self._do_close(pos, fill, "stop_loss_pct",
                                      f"бар {bar.start_ms}: low {bar.low:,.2f} <= SL {sl_lvl:,.2f}"
                                      f" (исполнение {fill:,.2f}, SL приоритетнее TP)", bar.end_ms)
            if tp_lvl is not None and bar.high >= tp_lvl:
                fill = max(bar.open, tp_lvl)
                return self._do_close(pos, fill, "take_profit_pct",
                                      f"бар {bar.start_ms}: high {bar.high:,.2f} >= TP {tp_lvl:,.2f}"
                                      f" (исполнение {fill:,.2f})", bar.end_ms)

        # 2) partial TP (один раз)
        ptp = rules.get("partial_tp") or {}
        if ptp.get("enabled") and not pos.get("partial_done") and ptp.get("at_pct"):
            lvl = float(pos["entry_price"]) * (1 + float(ptp["at_pct"]) / 100.0)
            hit_px = None
            if tick_px is not None and self.intrabar_exits and tick_px >= lvl:
                hit_px = tick_px
            elif bar is not None and bar.end_ms > entry_ts and bar.high >= lvl:
                hit_px = max(bar.open, lvl)
            if hit_px is not None:
                frac = float(ptp.get("fraction") or 0.5)
                qty_c = float(pos["quantity_open"]) * frac
                pos["partial_done"] = True
                act = self._do_close(pos, hit_px, "partial_tp",
                                     f"частичная фиксация {frac:.0%} на +{ptp['at_pct']}% "
                                     f"(уровень {lvl:,.2f})", ts_ms, quantity=qty_c)
                return act

        # 3) max_hold и горизонт (по времени, на закрытом баре)
        if bar is not None and bar.end_ms > entry_ts:
            elapsed = (bar.end_ms - entry_ts) // self.step_ms
            mh = rules.get("max_hold_bars")
            if mh is not None and elapsed >= int(mh):
                return self._do_close(pos, bar.close, "max_hold_bars",
                                      f"прошло {elapsed} баров >= max_hold {mh}", bar.end_ms)
            if rules.get("exit_at_horizon") and elapsed >= int(rules.get("horizon_bars") or self.horizon_bars):
                return self._do_close(pos, bar.close, "horizon",
                                      f"горизонт прогноза истёк ({elapsed} баров) - сигнал больше "
                                      f"не действителен", bar.end_ms)
        return None

    def on_bar(self, bar) -> Optional[PaperAction]:
        """Закрытый бар: баровые выходы (SL/TP если нет тиков, max_hold, horizon)."""
        for pos in self.open_positions():
            act = self._check_exits(pos, bar.end_ms, bar=bar)
            if act is not None:
                return act
        return None

    def on_tick(self, price: float, ts_ms: int) -> Optional[PaperAction]:
        """Тик сделки: тиковые SL/TP/partial-TP (только при intrabar_exits)."""
        if not self.intrabar_exits:
            return None
        for pos in self.open_positions():
            act = self._check_exits(pos, ts_ms, tick_px=float(price))
            if act is not None:
                return act
        return None
