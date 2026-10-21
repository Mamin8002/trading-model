"""SimulatedBroker — правила симулируемого исполнения (docs/simulation.md §2, §4).

Ключевые правила (ограничения ТЗ фазы 4):
- исполнение НЕ РАНЬШЕ следующей доступной цены: вход/выход по сигналу —
  по open следующего бара после сигнального (fill_policy=next_open);
- цена сигнальной свечи используется только как справочная (reference_price);
- проскальзывание применяется НЕВЫГОДНОЙ стороной к каждой цене исполнения
  (buy: +slippage_bps, sell: −slippage_bps) и ограничивается диапазоном бара
  [low, high] — за пределами бара цена не выдумывается;
- комиссия: max(commission_bps·notional/10⁴, commission_min_quote) — только из
  явного конфига;
- стоп/цель внутри бара: см. evaluate_exit_on_bar и консервативное правило
  конфликта (stop_first) либо остановку с требованием внутрибарных данных
  (require_intrabar_data). Выгодный сценарий автоматически НЕ выбирается;
- гэп через стоп: исполнение по open бара (хуже стопа) — реалистично и
  консервативно для длинной позиции.

НЕ реализовано (ограничения документированы): частичные исполнения, очередь/
ликвидность стакана, внутрибарная траектория, шорты, маржа/плечо.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config.schema import InstrumentConfig, SimulationConfig
from ..data.bar import Bar
from .errors import IntrabarDataRequired, SimConfigError
from .interfaces import ExitCause, Fill, Side

__all__ = ["ExitPlan", "SimulatedBroker", "validate_simulation_config"]


def validate_simulation_config(
    sim_cfg: SimulationConfig | None,
    risk_cfg=None,  # RiskConfig | None
    *,
    require_risk: bool = True,
) -> None:
    """Проверка ЯВНОСТИ обязательных параметров до запуска (ТЗ: не выдумывать)."""
    missing: list[str] = []
    if sim_cfg is None:
        missing.append("simulation: секция отсутствует (commission_bps, slippage_bps обязательны)")
    else:
        if sim_cfg.commission_bps is None:
            missing.append("simulation.commission_bps — комиссия должна быть задана явно")
        if sim_cfg.slippage_bps is None:
            missing.append("simulation.slippage_bps — проскальзывание должно быть задано явно")
    if require_risk:
        if risk_cfg is None:
            missing.append("risk: секция отсутствует (initial_capital_quote и лимиты обязательны)")
        else:
            if risk_cfg.initial_capital_quote is None:
                missing.append("risk.initial_capital_quote — стартовый капитал задаётся явно")
            for field_name in (
                "max_position_units",
                "max_orders_per_day",
                "daily_loss_limit_quote",
                "max_drawdown_pct",
                "max_data_age_seconds",
            ):
                if getattr(risk_cfg, field_name) is None:
                    missing.append(f"risk.{field_name} — лимит должен быть задан явно (безопасный отказ по умолчанию)")
            if (
                risk_cfg.position_units is None
                and risk_cfg.risk_per_trade_quote is None
                and risk_cfg.risk_per_trade_pct is None
            ):
                missing.append(
                    "risk.position_units ИЛИ risk.risk_per_trade_quote/pct — режим размера позиции обязателен"
                )
    if missing:
        raise SimConfigError(missing)


@dataclass(frozen=True)
class ExitPlan:
    """План выхода, рассчитанный по бару (исполняется немедленно в этом баре)."""

    price: float           # цена ДО проскальзывания/ограничения диапазоном
    cause: ExitCause
    conflict: bool = False


class SimulatedBroker:
    def __init__(self, sim_cfg: SimulationConfig, instrument_cfg: InstrumentConfig | None) -> None:
        validate_simulation_config(sim_cfg, None, require_risk=False)  # только sim-поля
        self.sim_cfg = sim_cfg
        self.instrument_cfg = instrument_cfg

    # ------------------------------------------------------------------ цены

    @property
    def slippage_fraction(self) -> float:
        return (self.sim_cfg.slippage_bps or 0.0) / 10_000.0

    def slipped_price(self, raw_price: float, side: Side, bar: Bar) -> float:
        """Проскальзывание невыгодной стороной + ограничение диапазоном бара."""
        slip = self.slippage_fraction
        price = raw_price * (1.0 + slip) if side is Side.BUY else raw_price * (1.0 - slip)
        return min(max(price, bar.low), bar.high)

    def commission(self, price: float, units: int) -> float:
        notional = price * units
        commission = notional * (self.sim_cfg.commission_bps or 0.0) / 10_000.0
        if self.sim_cfg.commission_min_quote is not None:
            commission = max(commission, self.sim_cfg.commission_min_quote)
        return commission

    # ----------------------------------------------------------------- филлы

    def execute_at_bar_open(
        self,
        bar: Bar,
        side: Side,
        units: int,
        cause: str,
        *,
        symbol: str = "",
        timeframe: str = "",
        intent_id: str = "",
    ) -> Fill:
        """Исполнение по open бара (единственная точка входа для сигнальных заявок)."""
        if units <= 0:
            raise ValueError("units должен быть > 0")
        raw = bar.open
        price = self.slipped_price(raw, side, bar)
        return Fill(
            ts=bar.ts_open,
            side=side,
            price=price,
            raw_price=raw,
            units=units,
            commission_quote=self.commission(price, units),
            slippage_bps=float(self.sim_cfg.slippage_bps or 0.0),
            bar_ts_open=bar.ts_open,
            cause=cause,
            intent_id=intent_id,
            symbol=symbol or bar.symbol,
            timeframe=timeframe,
        )

    def execute_exit_in_bar(
        self,
        bar: Bar,
        plan: ExitPlan,
        units: int,
        *,
        symbol: str = "",
        timeframe: str = "",
        intent_id: str = "",
    ) -> Fill:
        """Исполнение выхода по рассчитанной в баре цене (стоп/цель/конфликт)."""
        side = Side.SELL  # фаза 4: long-only, выходы всегда продажи
        price = self.slipped_price(plan.price, side, bar)
        return Fill(
            ts=bar.ts_open,
            side=side,
            price=price,
            raw_price=plan.price,
            units=units,
            commission_quote=self.commission(price, units),
            slippage_bps=float(self.sim_cfg.slippage_bps or 0.0),
            bar_ts_open=bar.ts_open,
            cause=plan.cause.value,
            intent_id=intent_id,
            symbol=symbol or bar.symbol,
            timeframe=timeframe,
        )

    # -------------------------------------------------------- стопы и цели

    def evaluate_exit_on_bar(
        self,
        bar: Bar,
        *,
        entry_price: float,
        stop_price: float | None,
        target_price: float | None,
    ) -> ExitPlan | None:
        """Проверка стопа/цели для ДЛИННОЙ позиции на баре (long-only, фаза 4).

        Порядок (docs/simulation.md §4):
        - достигнуты обе границы в одном баре (low <= stop и high >= target):
          конфликт. stop_first -> план по стопу (консервативно);
          require_intrabar_data -> IntrabarDataRequired (остановка, данные запрашиваются);
        - только стоп: если open <= stop (гэп вниз) -> цена open (cause=gap_through_stop),
          иначе stop (cause=stop);
        - только цель: если open >= target (гэп вверх) -> цена open (реалистичное
          исполнение лимитного выхода по лучшей доступной цене), иначе target;
        - ничего не достигнуто -> None.

        Внутрибарная траектория неизвестна — кроме документированного правила
        конфликта никаких «выгодных» допущений не делается.
        """
        if stop_price is None and target_price is None:
            return None
        stop_hit = stop_price is not None and bar.low <= stop_price
        target_hit = target_price is not None and bar.high >= target_price

        if stop_hit and target_hit:
            if self.sim_cfg.conflict_policy == "require_intrabar_data":
                raise IntrabarDataRequired(bar.ts_open.isoformat(), float(stop_price or 0.0), float(target_price or 0.0))
            return ExitPlan(price=float(stop_price), cause=ExitCause.STOP_TARGET_CONFLICT, conflict=True)
        if stop_hit:
            assert stop_price is not None
            if bar.open <= stop_price:
                return ExitPlan(price=bar.open, cause=ExitCause.GAP_THROUGH_STOP)
            return ExitPlan(price=float(stop_price), cause=ExitCause.STOP)
        if target_hit:
            assert target_price is not None
            if bar.open >= target_price:
                return ExitPlan(price=bar.open, cause=ExitCause.TARGET)
            return ExitPlan(price=float(target_price), cause=ExitCause.TARGET)
        return None
