"""Тесты событийного симулятора: время исполнения, издержки, стопы/цели,
консервативное правило конфликта, безопасные остановки (фаза 4).

Бары — детерминированные «ручные» серии; все ожидаемые значения посчитаны явно.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tradebot.backtest.engine import EventSimulator, run_simulation
from tradebot.config import AppConfig
from tradebot.data.bar import Bar, BarSource
from tradebot.execution.errors import SimConfigError
from tradebot.execution.interfaces import RejectionCode, Side, Signal
from tradebot.strategy.signals import ScriptedSignalSource

UTC = timezone.utc
START = datetime(2024, 1, 4, 8, 0, tzinfo=UTC)  # четверг


def make_bars(ohlc: list[tuple[float, float, float, float]], start=START, symbol="TEST-USDT") -> list[Bar]:
    bars = []
    for i, (o, h, l, c) in enumerate(ohlc):
        bars.append(
            Bar(ts_open=start + timedelta(minutes=i), open=o, high=h, low=l, close=c,
                volume=1000 + i, source=BarSource.CSV, symbol=symbol)
        )
    return bars


def buy_signal(bar: Bar) -> Signal:
    return Signal(ts_open=bar.ts_open, side=Side.BUY, symbol=bar.symbol, timeframe="1m",
                  model_version="test", proba=0.9)


def exit_signal(bar: Bar) -> Signal:
    return Signal(ts_open=bar.ts_open, side=None, symbol=bar.symbol, timeframe="1m",
                  model_version="test", proba=0.1, meta={"kind": "exit"})


BASE_CFG = {
    "run": {"mode": "TEST", "experiment_id": "sim-test", "model_dir": "models", "report_dir": "reports"},
    "data": {
        "csv_dir": "data/raw",
        "symbols": [{"ticker": "TEST-USDT", "files": {}}],
        "timestamp_semantics": "bar_open",
        "source_timezone": "UTC",
    },
    "timeframes": {"1m": {"enabled": True, "lookback_bars": 10, "horizon_bars": 1}},
    "risk": {
        "initial_capital_quote": 100_000.0,
        "position_units": 10,
        "max_position_units": 1_000,
        "max_orders_per_day": 100,
        "daily_loss_limit_quote": 100_000.0,
        "max_drawdown_pct": 50.0,
        "max_data_age_seconds": 3600.0,
        "allow_short": False,
        "halt_on_limit_breach": True,
    },
    "instrument": {"price_step": None, "lot_size_units": None, "quote_currency": None},
    "simulation": {
        "commission_bps": 0.0,
        "commission_min_quote": None,
        "slippage_bps": 0.0,
        "conflict_policy": "stop_first",
        "signal": {"entry_threshold": 0.5, "exit_threshold": None, "stop_bps": None, "target_bps": None},
    },
}


def build_config(**overrides) -> AppConfig:
    import copy
    import json

    raw = json.loads(json.dumps(copy.deepcopy(BASE_CFG)))

    def merge(section: str, values: dict) -> None:
        raw[section].update(values)

    for section, values in overrides.items():
        if section in ("risk", "instrument", "simulation"):
            if section == "simulation" and "signal" in values:
                raw["simulation"]["signal"].update(values.pop("signal"))
            merge(section, values)
        else:
            raw[section] = values
    return AppConfig.model_validate(raw)


FLAT = [(100.0, 100.5, 99.5, 100.0)] * 10


class TestExecutionTiming:
    def test_entry_and_exit_at_next_bar_open(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),   # b0: сигнал BUY (close 100)
            (101.0, 101.5, 100.5, 101.0),   # b1: исполнение входа по open=101 (НЕ по close b0)
            (101.0, 101.5, 100.5, 101.0),   # b2: сигнал выхода (close 101)
            (102.0, 102.5, 101.5, 102.0),   # b3: исполнение выхода по open=102
            (102.0, 102.5, 101.5, 102.0),   # b4
        ]
        bars = make_bars(ohlc)
        source = ScriptedSignalSource([buy_signal(bars[0]), exit_signal(bars[2])], symbol="TEST-USDT", timeframe="1m")
        run = run_simulation(build_config(), bars, source, timeframe="1m", symbol="TEST-USDT")
        assert len(run.fills) == 2
        entry, exit_ = run.fills
        assert entry.ts == bars[1].ts_open and entry.price == pytest.approx(101.0)
        assert entry.side is Side.BUY and entry.units == 10
        assert exit_.ts == bars[3].ts_open and exit_.price == pytest.approx(102.0)
        # цена закрытия сигнальной свечи НЕ использовалась как цена исполнения
        assert entry.price != bars[0].close
        assert exit_.price != bars[2].close
        assert run.portfolio.units == 0
        # PnL: (102-101)*10 = 10, комиссий нет
        assert run.portfolio.realized_pnl == pytest.approx(10.0)

    def test_signal_on_last_bar_is_rejected_no_next_bar(self):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([buy_signal(bars[-1])], symbol="TEST-USDT", timeframe="1m")
        run = run_simulation(build_config(), bars, source, timeframe="1m", symbol="TEST-USDT")
        assert run.fills == []
        assert any(r["code"] == RejectionCode.NO_NEXT_BAR.value for r in run.rejections)

    def test_open_position_at_end_not_closed_by_fabrication(self):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([buy_signal(bars[2])], symbol="TEST-USDT", timeframe="1m")
        run = run_simulation(build_config(), bars, source, timeframe="1m", symbol="TEST-USDT")
        assert len(run.fills) == 1  # только вход; фиктивной продажи нет
        assert run.portfolio.units == 10
        assert any("position_open_at_end" in w for w in run.warnings)


class TestCosts:
    def test_commission_bps_and_min(self):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([buy_signal(bars[0]), exit_signal(bars[2])], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(simulation={"commission_bps": 50.0, "commission_min_quote": None})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        entry, exit_ = run.fills
        # вход: 10 * 100 * 50bp = 5.0 ; выход: 10 * 100 * 50bp = 5.0 (цены flat=100)
        assert entry.commission_quote == pytest.approx(5.0)
        assert exit_.commission_quote == pytest.approx(5.0)
        assert run.portfolio.total_commission == pytest.approx(10.0)
        # realized PnL учитывает комиссии: gross 0 - 10
        assert run.portfolio.realized_pnl == pytest.approx(-10.0)

    def test_commission_minimum_applies(self):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([buy_signal(bars[0])], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(simulation={"commission_bps": 1.0, "commission_min_quote": 7.5})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        # 10*100*1bp = 0.1 -> минимум 7.5
        assert run.fills[0].commission_quote == pytest.approx(7.5)

    def test_slippage_adverse_and_clamped(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),       # b0 сигнал
            (100.0, 103.0, 99.0, 100.0),       # b1: вход buy slip 100bp -> 101.0 (в диапазоне)
            (101.0, 101.5, 100.5, 101.0),      # b2 выход-сигнал
            (101.0, 101.2, 100.8, 101.0),      # b3: выход sell slip 100bp -> 100.0, но low=100.8 -> кламп 100.8
        ]
        bars = make_bars(ohlc)
        source = ScriptedSignalSource([buy_signal(bars[0]), exit_signal(bars[2])], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(simulation={"slippage_bps": 100.0})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        entry, exit_ = run.fills
        assert entry.raw_price == pytest.approx(100.0)
        assert entry.price == pytest.approx(101.0)     # 100 * (1 + 0.01)
        assert exit_.price == pytest.approx(100.8)     # 101 * 0.99 = 99.99 -> кламп к low=100.8
        assert exit_.raw_price == pytest.approx(101.0)
        # оценочные издержки слиппеджа = |price-raw|*units
        slip_cost = abs(entry.price - entry.raw_price) * 10 + abs(exit_.price - exit_.raw_price) * 10
        assert slip_cost == pytest.approx((1.0 + 0.2) * 10)


class TestStopsTargets:
    def _run_with_stop_target(self, ohlc, stop=None, target=None, conflict_policy="stop_first"):
        bars = make_bars(ohlc)
        signal = Signal(ts_open=bars[0].ts_open, side=Side.BUY, symbol="TEST-USDT", timeframe="1m",
                        model_version="test", proba=0.9, stop_price=stop, target_price=target)
        source = ScriptedSignalSource([signal], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(simulation={"conflict_policy": conflict_policy})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        return run, bars

    def test_stop_hit_within_bar(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),    # b0 сигнал, стоп 99
            (100.0, 100.5, 99.5, 100.0),    # b1 вход по 100
            (100.0, 100.5, 98.5, 99.0),     # b2 low <= 99 -> стоп по 99
            (99.0, 99.5, 98.5, 99.0),
        ]
        run, bars = self._run_with_stop_target(ohlc, stop=99.0)
        exit_fill = run.fills[-1]
        assert exit_fill.cause == "stop"
        assert exit_fill.ts == bars[2].ts_open
        assert exit_fill.price == pytest.approx(99.0)
        assert run.episodes[0].exit_cause == "stop"

    def test_target_hit(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),    # b0 сигнал, цель 105
            (100.0, 100.5, 99.5, 100.0),    # b1 вход
            (104.0, 106.0, 103.5, 105.5),   # b2 high >= 105, open < 105 -> филл по цели 105
        ]
        run, bars = self._run_with_stop_target(ohlc, target=105.0)
        exit_fill = run.fills[-1]
        assert exit_fill.cause == "target"
        assert exit_fill.price == pytest.approx(105.0)

    def test_target_gap_up_fills_at_open(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 100.5, 99.5, 100.0),
            (107.0, 108.0, 106.5, 107.5),   # open выше цели 105 -> исполнение по open
        ]
        run, _ = self._run_with_stop_target(ohlc, target=105.0)
        assert run.fills[-1].price == pytest.approx(107.0)
        assert run.fills[-1].cause == "target"

    def test_conflict_stop_first_is_conservative(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),    # b0 сигнал: стоп 99, цель 105
            (100.0, 100.5, 99.5, 100.0),    # b1 вход
            (100.0, 106.0, 98.0, 103.0),    # b2: low<=99 И high>=105 -> конфликт
        ]
        run, bars = self._run_with_stop_target(ohlc, stop=99.0, target=105.0)
        exit_fill = run.fills[-1]
        assert exit_fill.cause == "stop_target_conflict"
        assert exit_fill.price == pytest.approx(99.0)  # консервативно: стоп, НЕ цель
        assert len(run.conflict_bars) == 1 and run.conflict_bars[0] == bars[2].ts_open
        assert any("stop_target_conflict" in w for w in run.warnings)

    def test_conflict_require_intrabar_halts_without_fill(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 106.0, 98.0, 103.0),    # конфликт
            (103.0, 104.0, 102.0, 103.5),
        ]
        run, _ = self._run_with_stop_target(ohlc, stop=99.0, target=105.0,
                                            conflict_policy="require_intrabar_data")
        assert run.completed is False
        assert any(h.reason == "intrabar_data_required" for h in run.halts)
        assert len(run.fills) == 1  # только вход; выход НЕ сфабрикован
        assert run.portfolio.units == 10

    def test_gap_through_stop_fills_at_open(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),    # b0 сигнал, стоп 99
            (100.0, 100.5, 99.5, 100.0),    # b1 вход
            (97.0, 97.5, 96.0, 96.5),       # b2: гэп — open ниже стопа -> филл по open 97
        ]
        run, _ = self._run_with_stop_target(ohlc, stop=99.0)
        exit_fill = run.fills[-1]
        assert exit_fill.cause == "gap_through_stop"
        assert exit_fill.price == pytest.approx(97.0)  # хуже стопа — реалистично

    def test_bps_stop_target_anchored_at_fill_price(self):
        # stop_bps=100 (1%), вход по open b1=101 -> стоп = 99.99; low b2 = 99.0 -> сработает
        bars = make_bars([
            (100.0, 100.5, 99.5, 100.0),
            (101.0, 101.5, 100.5, 101.0),
            (100.5, 100.6, 99.0, 99.5),
        ])
        source = ScriptedSignalSource([buy_signal(bars[0])], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(simulation={"signal": {"stop_bps": 100.0}})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        assert run.episodes[0].stop_price == pytest.approx(101.0 * 0.99)
        assert run.fills[-1].cause == "stop"

    def test_stop_rounded_conservatively_to_price_step(self):
        # шаг 0.05: стоп 100*(1-0.0033)=99.67 -> округление ВВЕРХ до 99.70 (сработает не позже)
        bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * 4)
        source = ScriptedSignalSource([buy_signal(bars[0])], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(instrument={"price_step": 0.05}, simulation={"signal": {"stop_bps": 33.0}})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        stop = run.episodes[0].stop_price
        assert stop == pytest.approx(99.70)
        assert abs(stop / 0.05 - round(stop / 0.05)) < 1e-9


class TestHaltsAndLimits:
    def test_max_position_rejection_not_silent_trim(self):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([buy_signal(bars[0])], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(risk={"position_units": 500, "max_position_units": 100})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        assert run.fills == []
        assert any(r["code"] == RejectionCode.MAX_POSITION.value for r in run.rejections)

    def test_insufficient_capital_rejection(self):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([buy_signal(bars[0])], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(risk={"initial_capital_quote": 500.0, "position_units": 10})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        assert run.fills == []
        assert any(r["code"] == RejectionCode.INSUFFICIENT_CAPITAL.value for r in run.rejections)

    def test_orders_per_day_limit(self):
        # три входа в один локальный день, лимит 2
        ohlc = [(100.0, 100.5, 99.5, 100.0)] * 12
        bars = make_bars(ohlc)
        signals = [
            buy_signal(bars[0]), exit_signal(bars[2]),
            buy_signal(bars[4]), exit_signal(bars[6]),
            buy_signal(bars[8]),
        ]
        source = ScriptedSignalSource(signals, symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(risk={"max_orders_per_day": 2})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        entries = [f for f in run.fills if f.side is Side.BUY]
        assert len(entries) == 2
        assert any(r["code"] == RejectionCode.ORDER_LIMIT.value for r in run.rejections)

    def test_daily_loss_limit_halts_new_entries(self):
        # убыточная сделка 10 шт: вход 100, выход 90 -> -100 > лимита 50
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),   # b0 buy
            (100.0, 100.5, 99.5, 100.0),   # b1 вход
            (90.0, 90.5, 89.5, 90.0),      # b2 exit-сигнал
            (90.0, 90.5, 89.5, 90.0),      # b3 выход по 90 -> дневной убыток -100
            (90.0, 90.5, 89.5, 90.0),      # b4 второй buy-сигнал
            (90.0, 90.5, 89.5, 90.0),      # b5 (вход заблокирован)
        ]
        bars = make_bars(ohlc)
        signals = [buy_signal(bars[0]), exit_signal(bars[2]), buy_signal(bars[4])]
        source = ScriptedSignalSource(signals, symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(risk={"daily_loss_limit_quote": 50.0, "position_units": 10})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        buys = [f for f in run.fills if f.side is Side.BUY]
        assert len(buys) == 1, "после дневного лимита новые входы запрещены"
        assert any(h.reason == "daily_loss_limit" for h in run.halts)
        assert any(r["code"] == RejectionCode.HALTED.value for r in run.rejections)

    def test_drawdown_limit_halts(self):
        ohlc = [
            (100.0, 100.5, 99.5, 100.0),  # b0 buy-сигнал (капитал 1000, 10 шт)
            (100.0, 100.5, 99.5, 100.0),  # b1 вход по 100 -> cash=0, units=10
            (95.0, 95.5, 94.5, 95.0),     # b2 close 95: equity=950, просадка 5% > 2% -> halt
            (95.0, 95.5, 94.5, 95.0),     # b3 exit-сигнал (защитный выход разрешён при halt)
            (95.0, 95.5, 94.5, 95.0),     # b4 исполнение выхода по 95
            (95.0, 95.5, 94.5, 95.0),     # b5 buy-сигнал -> отклонён (HALTED)
            (95.0, 95.5, 94.5, 95.0),     # b6
        ]
        bars = make_bars(ohlc)
        signals = [buy_signal(bars[0]), exit_signal(bars[3]), buy_signal(bars[5])]
        source = ScriptedSignalSource(signals, symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(risk={"initial_capital_quote": 1_000.0, "position_units": 10,
                                 "max_drawdown_pct": 2.0})
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        assert any(h.reason == "drawdown_limit" for h in run.halts)
        assert any(r["code"] == RejectionCode.HALTED.value for r in run.rejections)
        assert len([f for f in run.fills if f.side is Side.BUY]) == 1
        # защитный выход при остановке исполнен
        assert len([f for f in run.fills if f.side is Side.SELL]) == 1

    def test_invalid_bar_halts_simulation(self):
        bars = make_bars(FLAT)
        corrupt = replace(bars[5], high=98.0, low=99.0)  # high < low
        bars[5] = corrupt
        source = ScriptedSignalSource([buy_signal(bars[0])], symbol="TEST-USDT", timeframe="1m")
        run = run_simulation(build_config(), bars, source, timeframe="1m", symbol="TEST-USDT")
        assert run.completed is False
        assert any(h.reason == "invalid_data" for h in run.halts)
        assert run.bars_processed <= 5

    def test_position_desync_halts(self):
        bars = make_bars(FLAT)
        signals = [buy_signal(bars[0]), exit_signal(bars[4])]
        source = ScriptedSignalSource(signals, symbol="TEST-USDT", timeframe="1m")

        def corrupt_after_entry(index, bar, engine):
            if index == 2 and engine.portfolio.units > 0:
                engine.portfolio.units += 5  # инжектированное расхождение

        run = run_simulation(build_config(), bars, source, timeframe="1m", symbol="TEST-USDT",
                             bar_hook=corrupt_after_entry)
        assert run.completed is False
        assert any(h.reason == "position_desync" for h in run.halts)

    def test_missing_explicit_config_refuses_to_start(self):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([], symbol="TEST-USDT", timeframe="1m")
        import copy, json
        raw = json.loads(json.dumps(copy.deepcopy(BASE_CFG)))
        raw["simulation"]["commission_bps"] = None
        raw["risk"]["max_position_units"] = None
        cfg = AppConfig.model_validate(raw)
        with pytest.raises(SimConfigError) as exc:
            run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT")
        text = str(exc.value)
        assert "simulation.commission_bps" in text and "risk.max_position_units" in text


class TestMisc:
    def test_entry_signal_while_in_position_ignored(self):
        bars = make_bars(FLAT)
        signals = [buy_signal(bars[0]), buy_signal(bars[3])]
        source = ScriptedSignalSource(signals, symbol="TEST-USDT", timeframe="1m")
        run = run_simulation(build_config(), bars, source, timeframe="1m", symbol="TEST-USDT")
        assert len([f for f in run.fills if f.side is Side.BUY]) == 1
        assert any("игнорируется" in w for w in run.warnings)

    def test_scripted_signal_time_mismatch_rejected(self):
        bars = make_bars(FLAT)
        alien = Signal(ts_open=START + timedelta(seconds=30), side=Side.BUY, symbol="TEST-USDT",
                       timeframe="1m", model_version="test")
        source = ScriptedSignalSource([alien], symbol="TEST-USDT", timeframe="1m")
        with pytest.raises(ValueError, match="не совпадают"):
            run_simulation(build_config(), bars, source, timeframe="1m", symbol="TEST-USDT")

    def test_equity_identity_cash_plus_position(self):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([buy_signal(bars[2])], symbol="TEST-USDT", timeframe="1m")
        run = run_simulation(build_config(), bars, source, timeframe="1m", symbol="TEST-USDT")
        last_equity = run.equity_curve[-1][1]
        assert last_equity == pytest.approx(run.portfolio.cash + run.portfolio.units * bars[-1].close)
