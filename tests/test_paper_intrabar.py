"""Офлайн unit-тесты: intrabar (publicTrade/spikes) + paper trading (Этап 5b)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_bot.portfolio.journal import Journal  # noqa: E402
from btc_bot.portfolio.paper import (  # noqa: E402
    A_ABSTAIN, A_CLOSE, A_HOLD, A_OPEN, A_PARTIAL, PaperTrader,
)
from btc_bot.signals.engine import Signal  # noqa: E402
from btc_bot.streaming.base import SpikeEvent, TradeTick  # noqa: E402
from btc_bot.streaming.processor import IntrabarTracker, StreamProcessor  # noqa: E402

BASE = 1_800_000_000_000
STEP = 300_000


class FakeClock:
    def __init__(self, ms): self.ms = ms
    def __call__(self): return self.ms


def mk_signal(action="BUY", conf=0.62, fresh=True, sid="sig-1", price=100.0,
              bar_end=BASE + STEP):
    return Signal(signal_id=sid, created_at_ms=bar_end, bar_start_ms=bar_end - STEP,
                  bar_end_ms=bar_end, symbol="BTCUSDT", category="spot", interval="5",
                  action=action, confidence=conf, prob_up=conf if action == "BUY" else 1 - conf,
                  raw_direction=1 if action == "BUY" else -1, price_at_signal=price,
                  price_source="bar_close", model_version="xgb-test", horizon_bars=3,
                  horizon_minutes=15, feature_set="v2", data_fresh=fresh,
                  reasons=(), origin="replay", features={})


def make_trader(journal, clock, **kw):
    params = dict(symbol="BTCUSDT", category="spot", fee_pct=0.1, slippage_pct=0.02,
                  order_size_quote=100.0,
                  exit_rules={"take_profit_pct": 1.0, "stop_loss_pct": 0.5, "max_hold_bars": 10},
                  entry_min_confidence=None, cooldown_bars=0, max_open_positions=1,
                  intrabar_exits=True, exit_at_horizon=True,
                  partial_tp={"enabled": False, "at_pct": None, "fraction": 0.5},
                  interval_step_ms=STEP, horizon_bars=3, clock_ms=clock)
    params.update(kw)
    return PaperTrader(journal, **params)


@pytest.fixture()
def journal(tmp_path):
    return Journal(tmp_path / "j.json")


def test_publictrade_parsing():
    clock = FakeClock(BASE)
    proc = StreamProcessor("BTCUSDT", "5", clock_ms=clock)
    msg = {"topic": "publicTrade.BTCUSDT", "type": "snapshot", "ts": BASE + 1000,
           "data": [{"T": BASE + 900, "s": "BTCUSDT", "S": "Buy", "v": "0.001",
                     "p": "84500.50", "i": "t1", "BT": False, "seq": 1}]}
    clock.ms = BASE + 1000  # локальное время получения
    events = proc.on_ws_message(msg)
    ticks = [e for e in events if isinstance(e, TradeTick)]
    assert len(ticks) == 1 and ticks[0].price == 84500.50 and ticks[0].side == "Buy"
    assert ticks[0].trade_ts_ms == BASE + 900
    assert proc.last_data_ms == BASE + 1000      # свежесть по времени получения
    assert proc.server_ts_ms == BASE + 1000      # серверное время сообщения
    assert proc.is_fresh(clock.ms)


def test_spike_detector_and_cooldown():
    tr = IntrabarTracker(window_s=60, spike_pct=0.15, cooldown_s=300)
    for i in range(7):  # медленный дрейф
        assert tr.on_tick(BASE + i * 10_000, 100.0 * (1 + 0.0005 * i / 6)) is None
    ev = tr.on_tick(BASE + 70_000, 100.2)
    assert isinstance(ev, SpikeEvent) and ev.direction == "up" and ev.move_pct >= 0.15
    assert tr.on_tick(BASE + 100_000, 100.25) is None  # cooldown


def test_paper_open_math_and_snapshot(journal):
    clock = FakeClock(BASE + STEP)
    tr = make_trader(journal, clock)
    act = tr.on_signal(mk_signal(price=100.0), price_now=100.0)
    assert act.action == A_OPEN
    p = journal.paper_positions("OPEN")[0]
    entry_eff = 100.0 * 1.0002
    qty = 100.0 / entry_eff
    assert p["entry_eff_price"] == pytest.approx(entry_eff)
    assert p["quantity_initial"] == pytest.approx(qty)
    assert p["entry_fee_quote"] == pytest.approx(qty * entry_eff * 0.001)
    assert p["rules_snapshot"]["take_profit_pct"] == 1.0
    assert p["rules_snapshot"]["exit_at_horizon"] is True
    assert p["source_signal"]["signal_id"] == "sig-1"


def test_paper_abstain_paths(journal):
    clock = FakeClock(BASE + STEP)
    tr = make_trader(journal, clock)
    a = tr.on_signal(mk_signal(action="SELL", conf=0.7), price_now=100.0)
    assert a.action == A_ABSTAIN and "шорт" in a.reason
    a = tr.on_signal(mk_signal(fresh=False), price_now=100.0)
    assert a.action == A_ABSTAIN and "не актуальны" in a.reason
    assert not journal.paper_positions()


def test_paper_close_tick_sl_math(journal):
    clock = FakeClock(BASE + STEP)
    tr = make_trader(journal, clock)
    tr.on_signal(mk_signal(price=100.0), price_now=100.0)
    act = tr.on_tick(99.4, clock.ms + 30_000)  # SL-уровень 99.5
    assert act.action == A_CLOSE and act.rule == "stop_loss_pct"
    assert act.price == pytest.approx(99.5)
    entry_eff = 100.0 * 1.0002
    qty = 100.0 / entry_eff
    exp = qty * (99.5 * 0.9998) * 0.999 - (qty * entry_eff + qty * entry_eff * 0.001)
    p = journal.paper_positions("CLOSED")[0]
    assert p["realized_pnl_quote"] == pytest.approx(exp, abs=1e-9)
    assert p["outcome"] == "LOSS"


def test_paper_hold_and_counter_close(journal):
    clock = FakeClock(BASE + STEP)
    tr = make_trader(journal, clock)
    tr.on_signal(mk_signal(price=100.0, sid="a"), price_now=100.0)
    clock.ms = BASE + 2 * STEP
    a = tr.on_signal(mk_signal(price=100.2, sid="b"), price_now=100.2)
    assert a.action == A_HOLD and len(journal.paper_positions("OPEN")) == 1
    clock.ms = BASE + 3 * STEP
    a = tr.on_signal(mk_signal(action="SELL", conf=0.7, price=100.5, sid="c"), price_now=100.5)
    assert a.action == A_CLOSE and a.rule == "model_counter_signal"


def test_paper_partial_tp_once(journal):
    clock = FakeClock(BASE + STEP)
    tr = make_trader(journal, clock,
                     partial_tp={"enabled": True, "at_pct": 0.5, "fraction": 0.5})
    tr.on_signal(mk_signal(price=100.0), price_now=100.0)
    qty0 = journal.paper_positions("OPEN")[0]["quantity_open"]
    a1 = tr.on_tick(100.6, clock.ms + 1000)
    assert a1.action == A_PARTIAL and a1.rule == "partial_tp"
    assert journal.paper_positions("OPEN")[0]["quantity_open"] == pytest.approx(qty0 * 0.5)
    assert tr.on_tick(100.8, clock.ms + 2000) is None  # повторно не срабатывает
    a3 = tr.on_tick(101.2, clock.ms + 3000)
    assert a3.action == A_CLOSE and a3.rule == "take_profit_pct"


def test_paper_cooldown(journal):
    clock = FakeClock(BASE + STEP)
    tr = make_trader(journal, clock, cooldown_bars=2,
                     exit_rules={"take_profit_pct": None, "stop_loss_pct": None,
                                 "max_hold_bars": None})
    tr.on_signal(mk_signal(price=100.0, sid="a"), price_now=100.0)
    clock.ms = BASE + 2 * STEP
    tr.on_signal(mk_signal(action="SELL", conf=0.7, price=100.0, sid="b"), price_now=100.0)
    clock.ms = BASE + 3 * STEP
    a = tr.on_signal(mk_signal(price=100.0, sid="c"), price_now=100.0)
    assert a.action == A_ABSTAIN and "cooldown" in a.reason
    clock.ms = BASE + 4 * STEP
    a = tr.on_signal(mk_signal(price=100.0, sid="d"), price_now=100.0)
    assert a.action == A_OPEN


def test_paper_persistence_restart(journal, tmp_path):
    clock = FakeClock(BASE + STEP)
    tr1 = make_trader(journal, clock)
    tr1.on_signal(mk_signal(price=100.0), price_now=100.0)
    # "перезапуск": новые Journal и PaperTrader из того же файла
    j2 = Journal(journal.path)
    clock2 = FakeClock(BASE + 2 * STEP)
    tr2 = make_trader(j2, clock2)
    opens = tr2.open_positions()
    assert len(opens) == 1 and opens[0]["source_signal"]["signal_id"] == "sig-1"
    act = tr2.on_tick(99.4, clock2.ms)
    assert act.action == A_CLOSE and act.rule == "stop_loss_pct"
    j3 = Journal(journal.path)
    assert len(j3.paper_positions("CLOSED")) == 1 and not j3.paper_positions("OPEN")
