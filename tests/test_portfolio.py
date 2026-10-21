"""Офлайн unit-тесты журнала и советника (Этап 4). Запуск: pytest tests/ -q"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_bot.portfolio.advisor import (  # noqa: E402
    A_CONSIDER_CLOSE, A_ENTRY_POSSIBLE, A_ENTRY_WITHHELD, A_HOLD_POSITION,
    A_NO_ADVICE_STALE, A_OBSERVE, A_PORTFOLIO_UNKNOWN, PositionAdvisor,
)
from btc_bot.portfolio.journal import Journal, JournalError  # noqa: E402
from btc_bot.signals.engine import Signal  # noqa: E402

STEP = 300_000
BASE = 1_800_000_000_000


def mk_signal(action="BUY", conf=0.6, fresh=True, sid="sig-1", price=100.0, bar_start=BASE):
    return Signal(signal_id=sid, created_at_ms=bar_start + STEP, bar_start_ms=bar_start,
                  bar_end_ms=bar_start + STEP, symbol="BTCUSDT", category="spot", interval="5",
                  action=action, confidence=conf, prob_up=conf if action == "BUY" else 1 - conf,
                  raw_direction=1 if action == "BUY" else -1, price_at_signal=price,
                  price_source="bar_close", model_version="xgb-test", horizon_bars=1,
                  horizon_minutes=5, feature_set="ratio", data_fresh=fresh,
                  reasons=(), origin="replay", features={})


@pytest.fixture()
def journal(tmp_path):
    return Journal(tmp_path / "journal.json")


def advisor(journal, rules=None):
    return PositionAdvisor(journal, default_exit_rules=rules or {}, interval_step_ms=STEP)


def test_signal_not_executed_until_manual_entry(journal):
    journal.record_signal(mk_signal(sid="s1", conf=0.61, price=84500.0))
    sig = journal.find_signal("s1")
    assert sig["entry_confirmed"] is False and sig["position_id"] is None
    pos = journal.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                                 price=84510.0, quantity=0.01, signal_id="s1",
                                 exit_rules={"take_profit_pct": 1.5})
    sig = journal.find_signal("s1")
    assert sig["entry_confirmed"] is True and sig["position_id"] == pos.position_id
    src = pos.data["source_signal"]
    assert src["confidence"] == pytest.approx(0.61)
    assert src["price_at_signal"] == 84500.0
    assert src["horizon_minutes"] == 5 and src["model_version"] == "xgb-test"
    with pytest.raises(JournalError):
        journal.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                               price=1.0, quantity=1.0, signal_id="s1")


def test_partial_and_full_close_math(journal):
    pos = journal.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                                 price=100.0, quantity=0.05)
    p1 = journal.close_position(pos.position_id, price=110.0, quantity=0.02)
    assert p1.status == "OPEN" and p1.quantity_open == pytest.approx(0.03)
    assert p1.data["realized_pnl_quote"] == pytest.approx(0.2)
    p2 = journal.close_position(pos.position_id, price=90.0)
    assert p2.status == "CLOSED" and p2.quantity_open == 0.0
    assert p2.data["avg_exit_price"] == pytest.approx(98.0)
    assert p2.data["realized_pnl_quote"] == pytest.approx(-0.1)
    assert p2.data["realized_pnl_pct"] == pytest.approx(-2.0)  # -0.1 USDT на 5 USDT ноты = -2%
    assert p2.data["outcome"] == "LOSS"
    with pytest.raises(JournalError):
        journal.close_position(pos.position_id, price=100.0)


def test_persistence_and_corruption_guard(tmp_path):
    path = tmp_path / "j.json"
    j1 = Journal(path)
    j1.record_signal(mk_signal(sid="sp"))
    j1.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                      price=100.0, quantity=0.1, signal_id="sp")
    j2 = Journal(path)  # "перезапуск"
    assert j2.stats() == j1.stats()
    assert j2.open_positions()[0].entry_price == 100.0
    bad = tmp_path / "bad.json"
    bad.write_text("{broken", encoding="utf-8")
    with pytest.raises(JournalError):
        Journal(bad)._load()
    assert bad.read_text(encoding="utf-8") == "{broken"  # не перезаписан молча


def test_short_rejected_for_spot(journal):
    with pytest.raises(JournalError):
        journal.register_entry(symbol="BTCUSDT", category="spot", direction="SHORT",
                               price=100.0, quantity=1.0)


def test_advisor_open_position_actions(journal):
    pos = journal.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                                 price=100.0, quantity=1.0, opened_at_ms=BASE - 10 * STEP,
                                 exit_rules={"take_profit_pct": 1.0, "stop_loss_pct": 0.5,
                                             "max_hold_bars": 100})
    adv = advisor(journal)
    a = adv.evaluate(mk_signal(action="SELL", sid="x1"), price_now=101.5)
    assert a.action == A_CONSIDER_CLOSE and a.position_id == pos.position_id
    assert any("take_profit_pct" in t for t in a.triggered_rules)
    assert any("встречный сигнал" in r for r in a.reasons)
    a = adv.evaluate(mk_signal(action="BUY", sid="x2"), price_now=100.2)
    assert a.action == A_HOLD_POSITION
    a = adv.evaluate(mk_signal(action="HOLD", conf=0.52, sid="x3"), price_now=100.2)
    assert a.action == A_OBSERVE
    # SL пробит при HOLD-сигнале -> всё равно CONSIDER_CLOSE (условие выхода сработало)
    a = adv.evaluate(mk_signal(action="HOLD", conf=0.52, sid="x4"), price_now=99.4)
    assert a.action == A_CONSIDER_CLOSE
    assert any("stop_loss_pct" in t for t in a.triggered_rules)


def test_advisor_no_position_and_rules_gating(journal):
    adv0 = advisor(journal, rules=None)
    a = adv0.evaluate(mk_signal(action="SELL", sid="y1"), price_now=100.0)
    assert a.action == A_PORTFOLIO_UNKNOWN and "НЕИЗВЕСТНО" in a.text
    a = adv0.evaluate(mk_signal(action="BUY", sid="y2"), price_now=100.0)
    assert a.action == A_ENTRY_WITHHELD and "ВОЗДЕРЖАНА" in a.text
    adv1 = advisor(journal, rules={"take_profit_pct": 2.0})
    a = adv1.evaluate(mk_signal(action="BUY", sid="y3"), price_now=100.0)
    assert a.action == A_ENTRY_POSSIBLE


def test_advisor_stale_data_blocks_management(journal):
    journal.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                           price=100.0, quantity=1.0, exit_rules={"stop_loss_pct": 0.5})
    adv = advisor(journal)
    a = adv.evaluate(mk_signal(action="SELL", sid="z1", fresh=False), price_now=90.0)
    assert a.action == A_NO_ADVICE_STALE


def test_events_log_and_outcome(journal):
    journal.record_signal(mk_signal(action="BUY", sid="e1", price=123.45))
    pos = journal.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                                 price=123.5, quantity=0.2, signal_id="e1")
    journal.close_position(pos.position_id, price=125.0)
    kinds = {e["kind"] for e in journal.events(100)}
    assert {"signal_recorded", "entry_registered", "position_closed"} <= kinds
    closed = [e for e in journal.events(100) if e["kind"] == "position_closed"][-1]
    assert "Результат" in closed["reason"]
    assert journal.get_position(pos.position_id).data["outcome"] == "WIN"
