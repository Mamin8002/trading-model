"""PositionAdvisor: оценка состояния позиции по новым сигналам и явным правилам выхода.

Правила (по спецификации):
  * при устаревших/неполных данных рекомендации по управлению позицией НЕ выдаются;
  * пока позиция открыта, новый сигнал сначала оценивает ЕЁ состояние:
    держать / рассмотреть закрытие / продолжить наблюдение;
  * встречный сигнал (SELL против LONG) -> сообщить, объяснить причину и указать,
    какие НАСТРОЕННЫЕ условия выхода сработали (или что они не заданы/не сработали);
  * для spot BTCUSDT SELL = закрыть/уменьшить подтверждённую длинную позицию;
    короткая позиция НЕ открывается;
  * если открытых позиций нет, а пришёл SELL -> фактическое состояние портфеля
    НЕИЗВЕСТНО (бот не утверждает, что нужно что-то закрывать);
  * уровни TP/SL/срок удержания не изобретаются: только явно заданные правила
    (конфиг или параметры регистрации позиции);
  * если правила выхода не заданы - рекомендация НА ВХОД воздерживается
    (сигнал модели при этом фиксируется и отображается).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .journal import Journal, PositionView

# коды действий советника
A_HOLD_POSITION = "ДЕРЖАТЬ"
A_CONSIDER_CLOSE = "РАССМОТРЕТЬ_ЗАКРЫТИЕ"
A_OBSERVE = "НАБЛЮДАТЬ"
A_ENTRY_POSSIBLE = "ВХОД_ВОЗМОЖЕН"
A_ENTRY_WITHHELD = "ВХОД_ВОЗДЕРЖАНИЕ"
A_PORTFOLIO_UNKNOWN = "ПОРТФЕЛЬ_НЕИЗВЕСТЕН"
A_NO_ADVICE_STALE = "НЕТ_РЕКОМЕНДАЦИЙ_ДАННЫЕ_УСТАРЕЛИ"


@dataclass
class Advice:
    action: str
    text: str
    reasons: list[str] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)
    position_id: Optional[str] = None
    signal_id: Optional[str] = None
    ts_ms: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class PositionAdvisor:
    def __init__(self, journal: Journal, *, default_exit_rules: Optional[dict],
                 interval_step_ms: int, clock_ms=None):
        self.journal = journal
        self.default_exit_rules = default_exit_rules or {}
        self.interval_step_ms = interval_step_ms
        self._clock = clock_ms or (lambda: int(time.time() * 1000))

    # -- helpers -----------------------------------------------------------
    def _rules_for(self, pos: Optional[PositionView]) -> dict:
        if pos is not None and pos.exit_rules:
            return pos.exit_rules
        return self.default_exit_rules

    @staticmethod
    def _rules_configured(rules: dict) -> bool:
        return any(rules.get(k) is not None for k in ("take_profit_pct", "stop_loss_pct", "max_hold_bars"))

    def rules_text(self, rules: dict, pos: Optional[PositionView] = None,
                    price_now: Optional[float] = None, bar_end_ms: Optional[int] = None) -> str:
        parts = []
        tp, sl, mh = rules.get("take_profit_pct"), rules.get("stop_loss_pct"), rules.get("max_hold_bars")
        if tp is not None:
            lvl = f" (уровень {pos.entry_price * (1 + tp / 100):,.2f})" if pos else ""
            parts.append(f"TP {tp}%{lvl}")
        if sl is not None:
            lvl = f" (уровень {pos.entry_price * (1 - sl / 100):,.2f})" if pos else ""
            parts.append(f"SL {sl}%{lvl}")
        if mh is not None and pos is not None and bar_end_ms is not None:
            elapsed = max(0, (bar_end_ms - pos.data["opened_at_ms"]) // self.interval_step_ms)
            parts.append(f"max_hold {mh} баров (прошло {elapsed})")
        elif mh is not None:
            parts.append(f"max_hold {mh} баров")
        return ", ".join(parts) if parts else "не заданы"

    def _check_rules(self, rules: dict, pos: PositionView, price_now: Optional[float],
                     bar_end_ms: int) -> list[str]:
        triggered: list[str] = []
        tp, sl, mh = rules.get("take_profit_pct"), rules.get("stop_loss_pct"), rules.get("max_hold_bars")
        if price_now is not None:
            if tp is not None and price_now >= pos.entry_price * (1 + tp / 100):
                triggered.append(
                    f"take_profit_pct={tp}%: цена {price_now:,.2f} >= уровня "
                    f"{pos.entry_price * (1 + tp / 100):,.2f} (вход {pos.entry_price:,.2f})")
            if sl is not None and price_now <= pos.entry_price * (1 - sl / 100):
                triggered.append(
                    f"stop_loss_pct={sl}%: цена {price_now:,.2f} <= уровня "
                    f"{pos.entry_price * (1 - sl / 100):,.2f} (вход {pos.entry_price:,.2f})")
        if mh is not None:
            elapsed = (bar_end_ms - pos.data["opened_at_ms"]) // self.interval_step_ms
            if elapsed >= mh:
                triggered.append(f"max_hold_bars={mh}: прошло {elapsed} баров с момента входа")
        return triggered

    # -- main ----------------------------------------------------------------
    def evaluate(self, signal, price_now: Optional[float] = None) -> Advice:
        """Оценка сигнала относительно зарегистрированных позиций. signal: Signal (dataclass)."""
        ts = self._clock()
        sid = getattr(signal, "signal_id", None)
        symbol = getattr(signal, "symbol", None)
        bar_end = int(getattr(signal, "bar_end_ms", ts))

        # 1) устаревшие/неполные данные -> никаких рекомендаций по управлению
        if not getattr(signal, "data_fresh", False):
            return Advice(
                action=A_NO_ADVICE_STALE,
                text=("Данные устарели или неполны - рекомендации по управлению позицией "
                      "не выдаются. Сигнал зафиксирован как НЕ актуальный."),
                reasons=list(getattr(signal, "reasons", [])),
                signal_id=sid, ts_ms=ts,
            )

        if price_now is None:
            price_now = float(getattr(signal, "price_at_signal", 0.0)) or None

        opens = self.journal.open_positions(symbol)
        n_consecutive = self.journal.consecutive_same_action(getattr(signal, "action", ""))
        hist_note = (f"История сигналов учтена: это {n_consecutive}-й подряд сигнал "
                     f"{getattr(signal, 'action', '?')}." if n_consecutive >= 2 else "")

        # 2) есть открытая зарегистрированная позиция -> сначала оцениваем ЕЁ
        if opens:
            pos = opens[-1]
            rules = self._rules_for(pos)
            triggered = self._check_rules(rules, pos, price_now, bar_end)
            rules_txt = self.rules_text(rules, pos, price_now, bar_end)
            unreal = (f"{pos.unrealized_pct(price_now):+.3f}%" if price_now else "н/д")
            common = (f"Позиция {pos.position_id} {pos.direction} {pos.symbol}: вход "
                      f"{pos.entry_price:,.2f}, открыто {pos.quantity_open:g} из "
                      f"{pos.data['quantity_initial']:g}, текущая {price_now:,.2f} ({unreal}). "
                      f"Настроенные условия выхода: {rules_txt}.")
            if len(opens) > 1:
                common += f" (всего открытых позиций: {len(opens)}, оценена последняя)"

            against = (pos.direction == "LONG" and getattr(signal, "action", "") == "SELL")
            if triggered or against:
                reasons = []
                if against:
                    reasons.append(
                        f"встречный сигнал модели: SELL (уверенность {signal.confidence:.3f}, "
                        f"бар закрыт {bar_end}) - для spot это рекомендация закрыть/уменьшить "
                        f"длинную позицию")
                reasons.extend(triggered)
                if not triggered:
                    reasons.append("настроенные условия выхода (TP/SL/max_hold) не сработали или не заданы - "
                                   "основание: встречный сигнал модели")
                if hist_note:
                    reasons.append(hist_note)
                return Advice(
                    action=A_CONSIDER_CLOSE,
                    text=(f"{common} РЕКОМЕНДАЦИЯ: рассмотреть закрытие/уменьшение позиции "
                          f"(не является приказом - исполнение только вручную)."),
                    reasons=reasons, triggered_rules=triggered,
                    position_id=pos.position_id, signal_id=sid, ts_ms=ts,
                )
            if getattr(signal, "action", "") == "BUY" and pos.direction == "LONG":
                reasons = ["сигнал модели в направлении позиции"]
                if hist_note:
                    reasons.append(hist_note)
                return Advice(
                    action=A_HOLD_POSITION,
                    text=f"{common} РЕКОМЕНДАЦИЯ: держать (условия выхода не сработали).",
                    reasons=reasons, position_id=pos.position_id, signal_id=sid, ts_ms=ts,
                )
            reasons = ["сигнал HOLD (воздержание модели)"]
            if hist_note:
                reasons.append(hist_note)
            return Advice(
                action=A_OBSERVE,
                text=f"{common} РЕКОМЕНДАЦИЯ: продолжить наблюдение.",
                reasons=reasons, position_id=pos.position_id, signal_id=sid, ts_ms=ts,
            )

        # 3) открытых позиций нет
        action = getattr(signal, "action", "")
        if action == "BUY":
            rules = self._rules_for(None)
            if self._rules_configured(rules):
                reasons = [f"условия выхода заданы: {self.rules_text(rules)}"]
                if hist_note:
                    reasons.append(hist_note)
                return Advice(
                    action=A_ENTRY_POSSIBLE,
                    text=("Сигнал BUY: вход возможен согласно настроенным правилам выхода. "
                          "Сигнал НЕ считается исполненным - зарегистрируйте фактический вход вручную "
                          "(position_cli.py register-entry). Не финансовая рекомендация."),
                    reasons=reasons, signal_id=sid, ts_ms=ts,
                )
            reasons = ["не заданы условия выхода (positions.exit_rules в конфиге или --tp/--sl/--max-hold "
                       "при регистрации) - уровни не изобретаются"]
            if hist_note:
                reasons.append(hist_note)
            return Advice(
                action=A_ENTRY_WITHHELD,
                text=("Сигнал BUY зафиксирован, но рекомендация на вход ВОЗДЕРЖАНА: "
                      "настройте явные условия выхода (конфиг positions.exit_rules) - "
                      "без них бот не выдаёт рекомендаций на вход."),
                reasons=reasons, signal_id=sid, ts_ms=ts,
            )
        if action == "SELL":
            reasons = ["зарегистрированных открытых позиций нет",
                       "без приватного API фактическое состояние портфеля неизвестно"]
            if hist_note:
                reasons.append(hist_note)
            return Advice(
                action=A_PORTFOLIO_UNKNOWN,
                text=("SELL для spot BTCUSDT означает рекомендацию закрыть/уменьшить ПОДТВЕРЖДЁННУЮ "
                      "длинную позицию. Зарегистрированных открытых позиций нет - фактическое "
                      "состояние портфеля НЕИЗВЕСТНО; бот не утверждает, что нужно что-то закрывать. "
                      "Короткая позиция НЕ открывается."),
                reasons=reasons, signal_id=sid, ts_ms=ts,
            )
        reasons = ["сигнал HOLD (воздержание модели), открытых позиций нет"]
        if hist_note:
            reasons.append(hist_note)
        return Advice(
            action=A_OBSERVE,
            text="Наблюдение: нет сигнала к действию и нет зарегистрированных позиций.",
            reasons=reasons, signal_id=sid, ts_ms=ts,
        )
