"""Тесты риск-контроля и сайзинга (фаза 4): отказ по каждому лимиту, явность
параметров, консервативное округление, сверка позиции, устаревшие данные."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradebot.config.schema import InstrumentConfig, RiskConfig, SimulationConfig
from tradebot.execution.errors import RiskHaltError
from tradebot.execution.interfaces import (
    OrderIntent,
    PortfolioState,
    RejectionCode,
    RiskContext,
    Side,
    Signal,
)
from tradebot.execution.risk import RiskGuards
from tradebot.execution.sizing import SizingError, compute_position_size, estimate_commission

UTC = timezone.utc
TS = datetime(2024, 1, 4, 10, 0, tzinfo=UTC)


def sim_cfg(commission_bps=0.0, commission_min=None, slippage_bps=0.0, **kw) -> SimulationConfig:
    return SimulationConfig(commission_bps=commission_bps, commission_min_quote=commission_min,
                            slippage_bps=slippage_bps, **kw)


def risk_cfg(**kw) -> RiskConfig:
    base = dict(
        initial_capital_quote=100_000.0,
        position_units=10,
        max_position_units=100,
        max_orders_per_day=10,
        daily_loss_limit_quote=1_000.0,
        max_drawdown_pct=10.0,
        max_data_age_seconds=60.0,
    )
    base.update(kw)
    return RiskConfig(**base)


def intent(side=Side.BUY, units=10, stop=None, target=None, ts=TS) -> OrderIntent:
    signal = Signal(ts_open=ts, side=side, symbol="TEST-USDT", timeframe="1m", model_version="test")
    return OrderIntent(
        created_at=ts, symbol="TEST-USDT", timeframe="1m", side=side, quantity_units=units,
        signal=signal, reference_price=100.0, stop_price=stop, target_price=target,
    )


def ctx(cash=100_000.0, equity=100_000.0, units=0, age=0.0, valid=True, source_ok=True) -> RiskContext:
    return RiskContext(
        decision_ts=TS, last_bar_ts_open=TS, last_close=100.0, equity=equity, cash=cash,
        position_units=units, data_age_seconds=age, bar_is_valid=valid, data_source_ok=source_ok,
    )


def portfolio(cash=100_000.0, units=0, **kw) -> PortfolioState:
    p = PortfolioState(cash=cash, peak_equity=kw.pop("peak_equity", 100_000.0), **kw)
    p.units = units
    return p


def guards(risk=None, instrument=None, sim=None) -> RiskGuards:
    return RiskGuards(risk or risk_cfg(), instrument, sim or sim_cfg(), local_timezone="UTC")


class TestSizingKnownValues:
    def test_risk_quote_exact(self):
        # риск 1000, вход 100, стоп 95 -> убыток/ед = 5 -> 200 единиц (лот не задан)
        result = compute_position_size(
            risk_cfg=risk_cfg(position_units=None, risk_per_trade_quote=1_000.0, max_position_units=1000),
            instrument_cfg=None, sim_cfg=sim_cfg(), entry_price=100.0, stop_price=95.0,
            cash=1_000_000.0, equity=1_000_000.0, current_units=0,
        )
        assert result.units == 200 and result.mode == "risk_quote" and result.raw_units == 200

    def test_risk_quote_lot_rounding_down(self):
        # то же, лот 7 -> floor(200/7)*7 = 196
        result = compute_position_size(
            risk_cfg=risk_cfg(position_units=None, risk_per_trade_quote=1_000.0, max_position_units=1000),
            instrument_cfg=InstrumentConfig(lot_size_units=7), sim_cfg=sim_cfg(),
            entry_price=100.0, stop_price=95.0, cash=1_000_000.0, equity=1_000_000.0, current_units=0,
        )
        assert result.units == 196

    def test_risk_pct_of_equity(self):
        # 1% от капитала 50000 = 500; loss/ед = 2.5 -> 200 единиц
        result = compute_position_size(
            risk_cfg=risk_cfg(position_units=None, risk_per_trade_pct=1.0, max_position_units=1000),
            instrument_cfg=None, sim_cfg=sim_cfg(), entry_price=50.0, stop_price=47.5,
            cash=1_000_000.0, equity=50_000.0, current_units=0,
        )
        assert result.units == 200 and result.mode == "risk_pct"

    def test_risk_mode_capped_by_capital(self):
        # риск даёт 200 единиц по 100 = 20000, кэш 10000 -> 100 единиц (capped_by=capital)
        result = compute_position_size(
            risk_cfg=risk_cfg(position_units=None, risk_per_trade_quote=1_000.0, max_position_units=1000),
            instrument_cfg=None, sim_cfg=sim_cfg(), entry_price=100.0, stop_price=95.0,
            cash=10_000.0, equity=10_000.0, current_units=0,
        )
        assert result.units == 100 and result.capped_by == "capital"

    def test_risk_mode_capped_by_max_position(self):
        result = compute_position_size(
            risk_cfg=risk_cfg(position_units=None, risk_per_trade_quote=1_000.0, max_position_units=150),
            instrument_cfg=None, sim_cfg=sim_cfg(), entry_price=100.0, stop_price=95.0,
            cash=1_000_000.0, equity=1_000_000.0, current_units=0,
        )
        assert result.units == 150 and result.capped_by == "max_position"

    def test_risk_mode_requires_stop(self):
        with pytest.raises(SizingError) as exc:
            compute_position_size(
                risk_cfg=risk_cfg(position_units=None, risk_per_trade_quote=1_000.0),
                instrument_cfg=None, sim_cfg=sim_cfg(), entry_price=100.0, stop_price=None,
                cash=1_000_000.0, equity=1_000_000.0, current_units=0,
            )
        assert exc.value.rejection.code is RejectionCode.SIZING_STOP_REQUIRED

    def test_fixed_mode_passes_through_and_lot_violation_rejected(self):
        result = compute_position_size(
            risk_cfg=risk_cfg(position_units=50), instrument_cfg=InstrumentConfig(lot_size_units=10),
            sim_cfg=sim_cfg(), entry_price=100.0, stop_price=None, cash=1_000_000.0,
            equity=1_000_000.0, current_units=0,
        )
        assert result.units == 50 and result.mode == "fixed" and result.capped_by is None
        with pytest.raises(SizingError) as exc:
            compute_position_size(
                risk_cfg=risk_cfg(position_units=55), instrument_cfg=InstrumentConfig(lot_size_units=10),
                sim_cfg=sim_cfg(), entry_price=100.0, stop_price=None, cash=1_000_000.0,
                equity=1_000_000.0, current_units=0,
            )
        assert exc.value.rejection.code is RejectionCode.INVALID_LOT

    def test_no_sizing_mode_is_safe_refusal(self):
        cfg = risk_cfg(position_units=None)
        with pytest.raises(SizingError) as exc:
            compute_position_size(
                risk_cfg=cfg, instrument_cfg=None, sim_cfg=sim_cfg(), entry_price=100.0,
                stop_price=None, cash=1_000_000.0, equity=1_000_000.0, current_units=0,
            )
        assert exc.value.rejection.code is RejectionCode.RISK_PARAM_NOT_SET

    def test_commission_requires_explicit_bps(self):
        with pytest.raises(SizingError):
            estimate_commission(1000.0, SimulationConfig())  # commission_bps=None

    def test_commission_min_applies(self):
        assert estimate_commission(1_000.0, sim_cfg(commission_bps=10.0, commission_min=5.0)) == 5.0
        assert estimate_commission(10_000.0, sim_cfg(commission_bps=10.0, commission_min=5.0)) == pytest.approx(10.0)


class TestRiskGuardsRejections:
    def _codes(self, decision):
        return {r.code for r in decision.rejections}

    def test_unset_limit_is_safe_refusal(self):
        for field_name in ("max_position_units", "max_orders_per_day", "daily_loss_limit_quote",
                           "max_drawdown_pct", "max_data_age_seconds"):
            cfg = risk_cfg(**{field_name: None})
            decision = guards(risk=cfg).evaluate(intent(), portfolio(), ctx())
            assert not decision.approved, field_name
            assert RejectionCode.RISK_PARAM_NOT_SET in self._codes(decision), field_name

    def test_max_position_exceeded(self):
        decision = guards().evaluate(intent(units=60), portfolio(units=50), ctx())
        assert not decision.approved
        assert RejectionCode.MAX_POSITION in self._codes(decision)

    def test_insufficient_capital(self):
        decision = guards().evaluate(intent(units=10), portfolio(cash=500.0), ctx(cash=500.0))
        assert RejectionCode.INSUFFICIENT_CAPITAL in self._codes(decision)

    def test_daily_loss_limit_blocks_entry_and_halts(self):
        p = portfolio(day_realized_pnl=-1_500.0)
        g = guards()
        decision = g.evaluate(intent(), p, ctx())
        assert RejectionCode.DAILY_LOSS in self._codes(decision)
        assert g.is_halted and g.halt_reason == "daily_loss_limit"

    def test_drawdown_limit_blocks_entry(self):
        p = portfolio(peak_equity=100_000.0)
        g = guards()
        decision = g.evaluate(intent(), p, ctx(equity=80_000.0))  # просадка 20% > 10%
        assert RejectionCode.DRAWDOWN in self._codes(decision)
        assert g.is_halted and g.halt_reason == "drawdown_limit"

    def test_order_limit_per_day(self):
        p = portfolio(day_orders=10)
        decision = guards().evaluate(intent(), p, ctx())
        assert RejectionCode.ORDER_LIMIT in self._codes(decision)

    def test_stale_data_rejected_and_halts(self):
        g = guards()
        decision = g.evaluate(intent(), portfolio(), ctx(age=120.0))
        assert RejectionCode.STALE_DATA in self._codes(decision)
        assert g.is_halted and g.halt_reason == "stale_data"

    def test_invalid_bar_rejected_and_halts(self):
        g = guards()
        decision = g.evaluate(intent(), portfolio(), ctx(valid=False))
        assert RejectionCode.INVALID_DATA in self._codes(decision)
        assert g.is_halted and g.halt_reason == "invalid_data"

    def test_source_error_halts(self):
        g = guards()
        decision = g.evaluate(intent(), portfolio(), ctx(source_ok=False))
        assert RejectionCode.INVALID_DATA in self._codes(decision)
        assert g.is_halted and g.halt_reason == "data_source_error"

    def test_halt_blocks_entries_but_allows_protective_exit(self):
        g = guards()
        g.halt("manual_kill_switch", "оператор")
        buy = g.evaluate(intent(side=Side.BUY), portfolio(), ctx())
        assert not buy.approved and RejectionCode.HALTED in self._codes(buy)
        sell = g.evaluate(intent(side=Side.SELL, units=10), portfolio(units=10), ctx(units=10))
        assert sell.approved, "защитный выход обязан проходить при halt"

    def test_sell_without_position_rejected(self):
        decision = guards().evaluate(intent(side=Side.SELL, units=10), portfolio(units=0), ctx())
        assert RejectionCode.NO_POSITION_TO_CLOSE in self._codes(decision)

    def test_sell_more_than_position_rejected(self):
        decision = guards().evaluate(intent(side=Side.SELL, units=15), portfolio(units=10), ctx(units=10))
        assert RejectionCode.SHORT_NOT_ALLOWED in self._codes(decision)

    def test_price_step_violation(self):
        g = guards(instrument=InstrumentConfig(price_step=0.05))
        decision = g.evaluate(intent(stop=99.99), portfolio(), ctx())  # 99.99 не кратна 0.05
        assert RejectionCode.INVALID_PRICE_STEP in self._codes(decision)

    def test_price_step_not_confirmed_is_warning_not_guess(self):
        g = guards(instrument=None)
        decision = g.evaluate(intent(stop=99.99), portfolio(), ctx())
        assert decision.approved
        assert any("price_step" in w for w in decision.warnings)

    def test_stop_above_entry_rejected(self):
        decision = guards().evaluate(intent(stop=101.0), portfolio(), ctx())
        assert RejectionCode.INVALID_STOP_TARGET in self._codes(decision)

    def test_approved_intent_lists_checks(self):
        decision = guards().evaluate(intent(), portfolio(), ctx())
        assert decision.approved
        for name in ("halt_flag", "data_age", "drawdown", "daily_loss", "orders_per_day", "max_position", "capital"):
            assert name in decision.checks_performed


class TestConsistency:
    def test_desync_halts(self):
        g = guards()
        assert g.verify_consistency(0)
        g.on_fill(portfolio(), 100.0, Side.BUY, 10, 1.0)  # журнал ожидает +10
        assert g.verify_consistency(10)
        assert not g.verify_consistency(7)  # портфель «повреждён»
        assert g.is_halted and g.halt_reason == "position_desync"

    def test_on_fill_updates_day_counters(self):
        p = portfolio()
        g = guards()
        g.on_fill(p, 100.0, Side.BUY, 10, 2.0)
        assert p.day_orders == 1
        assert p.day_realized_pnl == pytest.approx(-2.0)
        p.avg_entry_price = 100.0
        g.on_fill(p, 105.0, Side.SELL, 10, 2.0)
        assert p.realized_pnl == pytest.approx(50.0 - 4.0)
        assert p.day_realized_pnl == pytest.approx(48.0 - 2.0)
        assert p.total_commission == pytest.approx(4.0)

    def test_daily_loss_halt_triggers_on_fill(self):
        p = portfolio()
        p.avg_entry_price = 100.0
        g = guards(risk=risk_cfg(daily_loss_limit_quote=50.0))
        g.on_fill(p, 90.0, Side.SELL, 10, 1.0)  # убыток 100 + комиссия > 50
        assert g.is_halted and g.halt_reason == "daily_loss_limit"


class TestRollover:
    def test_day_counters_reset(self):
        p = portfolio()
        g = guards()
        p.day_orders = 5
        p.day_realized_pnl = -100.0
        g.rollover_day(p, TS + timedelta(days=1))
        assert p.day_orders == 0 and p.day_realized_pnl == 0.0
