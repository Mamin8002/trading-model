#!/usr/bin/env python3
r"""Ручная проверка Этапа 5b: внутрисвечной анализ (publicTrade/spikes) + paper trading.

Запуск (PowerShell, из корня проекта):
    python scripts\manual_checks\check_stage5b_paper.py

Полностью офлайн (без сети и секретов), на демонстрационных/синтетических данных.
Файлы создаются только во временных каталогах. Ордера не отправляются - paper
trading виртуальный; A9 структурно подтверждает отсутствие торговых/приватных
эндпоинтов и сетевых импортов в новом коде.

Проверяются:
  A1 парсинг publicTrade (формат документации Bybit: массив сделок T/p/v/S), свежесть
  A2 spike-детектор: быстрое движение -> АЛЕРТ (не сигнал входа), cooldown, медленный дрейф без алертов
  A3 paper OPEN_VIRTUAL: снимок правил/модели/сигнала, математика qty/комиссии/слиппеджа (ручной расчёт)
  A4 HOLD при повторном BUY/HOLD; ABSTAIN: SELL без позиции (шорт НЕ открывается), stale-сигнал
  A5 CLOSE-пути: тиковый SL, SL приоритетнее TP в баре, TP, max_hold, горизонт, встречный сигнал - с правилом и причиной
  A6 PARTIAL_CLOSE: частичная фиксация доли, остаток открыт, повторная частичка не выполняется
  A7 cooldown_bars и max_open_positions (конфликтующие позиции не открываются)
  A8 персистентность: открытая виртуальная позиция переживает "перезапуск" и продолжает сопровождаться
  A9 скан: portfolio/paper без сетевых импортов; во всём коде нет торговых/приватных эндпоинтов и ключей
  B  end-to-end: MonitorBot.run_test_replay + PaperTrader на синтетических данных -
     баланс = initial + Σ realized, у закрытых позиций есть правило/причина и outcome

Ожидаемый вывод: [PASS] A1..A9, B, "RESULT: PASS (проверок пройдено: 10)", код 0.
Время выполнения: ~15-30 с.
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btc_bot.bot import MonitorBot  # noqa: E402
from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import build_dataset  # noqa: E402
from btc_bot.data.store import save_klines_csv  # noqa: E402
from btc_bot.model.features import build_features  # noqa: E402
from btc_bot.model.predictor import Predictor  # noqa: E402
from btc_bot.model.train import build_card, library_versions, make_model_version, save_artifact, train_final  # noqa: E402
from btc_bot.portfolio.journal import Journal  # noqa: E402
from btc_bot.portfolio.paper import A_ABSTAIN, A_CLOSE, A_HOLD, A_OPEN, A_PARTIAL, PaperTrader  # noqa: E402
from btc_bot.signals.engine import Signal  # noqa: E402
from btc_bot.streaming.base import ConnState, KlineBar, SpikeEvent, TradeTick  # noqa: E402
from btc_bot.streaming.processor import IntrabarTracker, StreamProcessor  # noqa: E402
from btc_bot.ui.console import ConsoleUI, event_paper_text, event_spike_text, format_snapshot_text  # noqa: E402

INTERVAL = "5"
STEP = 300_000
BASE = 1_800_000_200_000 - (1_800_000_200_000 % STEP)

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


class FakeClock:
    def __init__(self, ms: int):
        self.ms = ms

    def __call__(self) -> int:
        return self.ms


def mk_signal(action="BUY", conf=0.62, fresh=True, bar_end=BASE + STEP, sid="sig-p1",
              price=100.0, version="xgb-test-h3") -> Signal:
    return Signal(signal_id=sid, created_at_ms=bar_end, bar_start_ms=bar_end - STEP,
                  bar_end_ms=bar_end, symbol="BTCUSDT", category="spot", interval=INTERVAL,
                  action=action, confidence=conf, prob_up=conf if action == "BUY" else 1 - conf,
                  raw_direction=1 if action == "BUY" else -1, price_at_signal=price,
                  price_source="bar_close", model_version=version, horizon_bars=3,
                  horizon_minutes=15, feature_set="v2", data_fresh=fresh,
                  reasons=(), origin="replay", features={})


def mk_bar(i: int, o: float, h: float, l: float, c: float) -> KlineBar:
    start = BASE + i * STEP
    return KlineBar(symbol="BTCUSDT", interval=INTERVAL, start_ms=start,
                    end_ms=start + STEP, open=o, high=h, low=l, close=c,
                    volume=1.0, turnover=100.0, confirmed=True,
                    server_ts_ms=start + STEP, received_ms=start + STEP)


def make_trader(tmp: Path, clock: FakeClock, *, rules=None, cooldown=0, intrabar=True,
                exit_at_horizon=True, partial=None, horizon=3, journal=None,
                entry_min=None) -> tuple[PaperTrader, Journal]:
    j = journal or Journal(tmp / f"paper_{id(clock)}_{len(list(tmp.iterdir()))}.json")
    t = PaperTrader(
        j, symbol="BTCUSDT", category="spot",
        fee_pct=0.1, slippage_pct=0.02, order_size_quote=100.0,
        exit_rules=rules if rules is not None else {"take_profit_pct": 1.0, "stop_loss_pct": 0.5,
                                                    "max_hold_bars": 10},
        entry_min_confidence=entry_min, cooldown_bars=cooldown, max_open_positions=1,
        intrabar_exits=intrabar, exit_at_horizon=exit_at_horizon,
        partial_tp=partial or {"enabled": False, "at_pct": None, "fraction": 0.5},
        interval_step_ms=STEP, horizon_bars=horizon, clock_ms=clock)
    return t, j


# ------------------------------------------------------------------ A1
def check_a1_publictrade_parsing() -> None:
    clock = FakeClock(BASE)
    proc = StreamProcessor("BTCUSDT", INTERVAL, clock_ms=clock)
    msg = {  # формат из документации Bybit (publicTrade spot)
        "topic": "publicTrade.BTCUSDT", "type": "snapshot", "ts": BASE + 1000,
        "data": [
            {"T": BASE + 900, "s": "BTCUSDT", "S": "Buy", "v": "0.001", "p": "84500.50",
             "i": "trade1", "BT": False, "seq": 1},
            {"T": BASE + 950, "s": "BTCUSDT", "S": "Sell", "v": "0.002", "p": "84499.00",
             "i": "trade2", "BT": False, "seq": 2},
        ],
    }
    clock.ms = BASE + 1000
    events = proc.on_ws_message(msg)
    ticks = [e for e in events if isinstance(e, TradeTick)]
    ok = (len(ticks) == 2 and abs(ticks[0].price - 84500.50) < 1e-9
          and ticks[0].side == "Buy" and abs(ticks[0].size - 0.001) < 1e-12
          and ticks[0].trade_ts_ms == BASE + 900
          and proc.last_data_ms == BASE + 1000 and proc.is_fresh(clock.ms)
          and proc.last_trade is not None)
    record("A1 publicTrade: массив сделок T/p/v/S -> TradeTick; свежесть данных обновляется",
           ok, f"ticks={len(ticks)}, last_price_from_trade={proc.last_trade.price if proc.last_trade else None}")


# ------------------------------------------------------------------ A2
def check_a2_spike_detector() -> None:
    tr = IntrabarTracker(window_s=60, spike_pct=0.15, cooldown_s=300)
    t0 = BASE
    events = []
    # медленный дрейф +0.05% за минуту -> без алертов
    for i in range(7):
        ev = tr.on_tick(t0 + i * 10_000, 100.0 * (1 + 0.0005 * i / 6))
        events.append(ev)
    ok_drift = not any(events)
    # резкий скачок до +0.2% -> алерт up
    ev1 = tr.on_tick(t0 + 70_000, 100.2)
    ok_spike = (isinstance(ev1, SpikeEvent) and ev1.direction == "up" and ev1.move_pct >= 0.15)
    # повтор через 30 с (тот же уровень) -> cooldown подавляет
    ev2 = tr.on_tick(t0 + 100_000, 100.25)
    ok_cooldown = ev2 is None
    # после cooldown новое резкое движение вниз -> алерт down
    tr2 = IntrabarTracker(window_s=60, spike_pct=0.15, cooldown_s=300)
    for i in range(5):
        tr2.on_tick(t0 + i * 10_000, 100.0)
    ev3 = tr2.on_tick(t0 + 50_000, 99.7)
    ok_down = isinstance(ev3, SpikeEvent) and ev3.direction == "down"
    txt = event_spike_text(ev1)
    ok_text = ("АЛЕРТ" in txt and "не сигнал входа" in txt and "%" in txt
               and f"{ev1.move_pct:+.3f}" in txt)
    record("A2 spike-детектор: дрейф без алертов; скачок -> АЛЕРТ (up/down) с cooldown; текст = 'не сигнал входа'",
           ok_drift and ok_spike and ok_cooldown and ok_down and ok_text,
           f"drift_quiet={ok_drift}, spike_up={ok_spike}, cooldown={ok_cooldown}, down={ok_down}, text={ok_text}")


# ------------------------------------------------------------------ A3
def check_a3_paper_open(tmp: Path) -> None:
    clock = FakeClock(BASE + STEP)
    tr, j = make_trader(tmp, clock)
    act = tr.on_signal(mk_signal(conf=0.62, price=100.0), price_now=100.0)
    pos = j.paper_positions("OPEN")
    entry_eff = 100.0 * (1 + 0.0002)
    qty = 100.0 / entry_eff
    fee = qty * entry_eff * 0.001
    cost = qty * entry_eff + fee
    p = pos[0]
    ok_math = (abs(p["entry_eff_price"] - entry_eff) < 1e-9
               and abs(p["quantity_initial"] - qty) < 1e-9
               and abs(p["entry_fee_quote"] - fee) < 1e-9
               and abs(p["entry_cost_quote"] - cost) < 1e-9)
    snap = p["rules_snapshot"]
    ok_snapshot = (snap["take_profit_pct"] == 1.0 and snap["stop_loss_pct"] == 0.5
                   and snap["max_hold_bars"] == 10 and snap["exit_at_horizon"] is True
                   and snap["horizon_bars"] == 3 and snap["fee_pct"] == 0.1
                   and p["source_signal"]["signal_id"] == "sig-p1"
                   and p["source_signal"]["model_version"] == "xgb-test-h3"
                   and abs(p["source_signal"]["confidence"] - 0.62) < 1e-9)
    ok_action = (act.action == A_OPEN and act.position_id == p["position_id"]
                 and len(j.paper_actions(10)) == 1
                 and "PAPER_OPENED" in p["status_history"][0]["event"])
    record("A3 OPEN_VIRTUAL: qty/комиссия/слиппедж сверены с ручным расчётом; снимок правил+модели+сигнала; действие в журнале",
           ok_math and ok_snapshot and ok_action,
           f"qty={qty:.8f}, fee={fee:.5f}, cost={cost:.5f}; action={act.action}")


# ------------------------------------------------------------------ A4
def check_a4_hold_abstain(tmp: Path) -> None:
    clock = FakeClock(BASE + STEP)
    tr, j = make_trader(tmp, clock)
    tr.on_signal(mk_signal(conf=0.62, price=100.0, sid="s4a"), price_now=100.0)
    clock.ms = BASE + 2 * STEP
    a2 = tr.on_signal(mk_signal(action="BUY", conf=0.66, price=100.2, sid="s4b"), price_now=100.2)
    clock.ms = BASE + 3 * STEP
    a3 = tr.on_signal(mk_signal(action="HOLD", conf=0.52, price=100.1, sid="s4c"), price_now=100.1)
    ok_open = a2.action == A_HOLD and a3.action == A_HOLD and len(j.paper_positions("OPEN")) == 1
    # SELL без позиции -> ABSTAIN (шорт не открывается)
    clock2 = FakeClock(BASE + STEP)
    tr2, j2 = make_trader(tmp, clock2)
    a4 = tr2.on_signal(mk_signal(action="SELL", conf=0.7, price=100.0, sid="s4d"), price_now=100.0)
    ok_sell = a4.action == A_ABSTAIN and "шорт" in a4.reason and not j2.paper_positions()
    # stale-сигнал -> ABSTAIN даже при BUY
    clock3 = FakeClock(BASE + STEP)
    tr3, j3 = make_trader(tmp, clock3)
    a5 = tr3.on_signal(mk_signal(action="BUY", conf=0.8, fresh=False, price=100.0, sid="s4e"), price_now=100.0)
    ok_stale = a5.action == A_ABSTAIN and "не актуальны" in a5.reason and not j3.paper_positions()
    record("A4 HOLD при повторном BUY/HOLD (позиция одна); ABSTAIN: SELL без позиции (шорт не открывается), stale-сигнал",
           ok_open and ok_sell and ok_stale, f"hold={ok_open}, sell_abstain={ok_sell}, stale_abstain={ok_stale}")


# ------------------------------------------------------------------ A5
def check_a5_close_paths(tmp: Path) -> None:
    # (a) тиковый SL
    clock = FakeClock(BASE + STEP)
    tr, j = make_trader(tmp, clock)
    tr.on_signal(mk_signal(price=100.0, sid="s5a"), price_now=100.0)
    clock.ms = BASE + STEP + 30_000
    act = tr.on_tick(99.4, clock.ms)  # SL-уровень 99.5
    pos = j.paper_positions("CLOSED")[0]
    entry_eff = 100.0 * 1.0002
    qty = 100.0 / entry_eff
    exit_eff = 99.5 * 0.9998
    exp_realized = qty * exit_eff * 0.999 - (qty * entry_eff + qty * entry_eff * 0.001)
    ok_a = (act is not None and act.action == A_CLOSE and act.rule == "stop_loss_pct"
            and abs(act.price - 99.5) < 1e-9 and abs(pos["realized_pnl_quote"] - exp_realized) < 1e-6
            and pos["outcome"] == "LOSS")
    # (b) SL приоритетнее TP в одном баре (баровый режим)
    clock = FakeClock(BASE + STEP)
    tr_b, j_b = make_trader(tmp, clock, intrabar=False)
    tr_b.on_signal(mk_signal(price=100.0, sid="s5b"), price_now=100.0)
    bar = mk_bar(1, 100.0, 101.5, 99.0, 100.5)  # high >= TP(101.0) И low <= SL(99.5)
    act_b = tr_b.on_bar(bar)
    ok_b = act_b is not None and act_b.action == A_CLOSE and act_b.rule == "stop_loss_pct" \
        and abs(act_b.price - min(100.0, 99.5)) < 1e-9
    # (c) TP по бару
    clock = FakeClock(BASE + STEP)
    tr_c, j_c = make_trader(tmp, clock, intrabar=False)
    tr_c.on_signal(mk_signal(price=100.0, sid="s5c"), price_now=100.0)
    act_c = tr_c.on_bar(mk_bar(1, 100.2, 101.4, 100.0, 101.2))
    ok_c = act_c is not None and act_c.rule == "take_profit_pct" and abs(act_c.price - 101.0) < 1e-9
    # (d) max_hold
    clock = FakeClock(BASE + STEP)
    tr_d, j_d = make_trader(tmp, clock, rules={"take_profit_pct": None, "stop_loss_pct": None,
                                               "max_hold_bars": 2}, intrabar=False)
    tr_d.on_signal(mk_signal(price=100.0, sid="s5d"), price_now=100.0)
    tr_d.on_bar(mk_bar(1, 100, 100.2, 99.9, 100.1))
    act_d = tr_d.on_bar(mk_bar(2, 100.1, 100.3, 100.0, 100.2))
    ok_d = act_d is not None and act_d.rule == "max_hold_bars" and act_d.action == A_CLOSE
    # (e) выход по горизонту (H=3)
    clock = FakeClock(BASE + STEP)
    tr_e, j_e = make_trader(tmp, clock, rules={"take_profit_pct": None, "stop_loss_pct": None,
                                               "max_hold_bars": None}, intrabar=False, horizon=3)
    tr_e.on_signal(mk_signal(price=100.0, sid="s5e"), price_now=100.0)
    tr_e.on_bar(mk_bar(1, 100, 100.1, 99.9, 100.05))
    tr_e.on_bar(mk_bar(2, 100.05, 100.2, 100.0, 100.1))
    act_e = tr_e.on_bar(mk_bar(3, 100.1, 100.3, 100.0, 100.25))
    ok_e = act_e is not None and act_e.rule == "horizon" and "горизонт" in act_e.reason
    # (f) встречный сигнал SELL -> CLOSE
    clock = FakeClock(BASE + STEP)
    tr_f, j_f = make_trader(tmp, clock)
    tr_f.on_signal(mk_signal(price=100.0, sid="s5f"), price_now=100.0)
    clock.ms = BASE + 2 * STEP
    act_f = tr_f.on_signal(mk_signal(action="SELL", conf=0.7, price=100.5, sid="s5g"), price_now=100.5)
    ok_f = act_f is not None and act_f.action == A_CLOSE and act_f.rule == "model_counter_signal"
    ok = ok_a and ok_b and ok_c and ok_d and ok_e and ok_f
    record("A5 CLOSE-пути: тиковый SL (PnL сверен вручную), SL>TP в баре, TP, max_hold, горизонт, встречный сигнал - каждое с правилом",
           ok, f"tick_sl={ok_a}, sl_priority={ok_b}, tp={ok_c}, max_hold={ok_d}, horizon={ok_e}, counter={ok_f}; "
              f"SL PnL={pos['realized_pnl_quote']:+.5f} (ожидали {exp_realized:+.5f})")


# ------------------------------------------------------------------ A6
def check_a6_partial_close(tmp: Path) -> None:
    clock = FakeClock(BASE + STEP)
    tr, j = make_trader(tmp, clock, partial={"enabled": True, "at_pct": 0.5, "fraction": 0.5})
    tr.on_signal(mk_signal(price=100.0, sid="s6a"), price_now=100.0)
    p0 = j.paper_positions("OPEN")[0]
    qty0 = p0["quantity_open"]
    clock.ms = BASE + STEP + 20_000
    act1 = tr.on_tick(100.6, clock.ms)   # >= уровня partial TP 100.5
    p1 = j.paper_positions("OPEN")
    ok1 = (act1 is not None and act1.action == A_PARTIAL and act1.rule == "partial_tp"
           and len(p1) == 1 and abs(p1[0]["quantity_open"] - qty0 * 0.5) < 1e-9
           and p1[0]["partial_done"] is True)
    act2 = tr.on_tick(100.8, clock.ms + 1000)  # повторная частичка НЕ выполняется; TP 101.0 ещё не достигнут
    ok2 = act2 is None and abs(j.paper_positions("OPEN")[0]["quantity_open"] - qty0 * 0.5) < 1e-9
    act3 = tr.on_tick(101.2, clock.ms + 2000)  # полный TP
    ok3 = act3 is not None and act3.action == A_CLOSE and act3.rule == "take_profit_pct"
    closed = j.paper_positions("CLOSED")
    realized_sum = sum(f.get("realized_quote", 0.0) for f in closed[0]["fills"] if f["type"].startswith("paper_exit"))
    ok4 = (len(closed) == 1 and abs(closed[0]["realized_pnl_quote"] - realized_sum) < 1e-9
           and closed[0]["outcome"] == "WIN")
    record("A6 PARTIAL_CLOSE: доля закрыта по уровню, остаток открыт, повторно не срабатывает, затем полный TP; realized согласован",
           ok1 and ok2 and ok3 and ok4,
           f"partial={ok1}, no_repeat={ok2}, final_tp={ok3}, accounting={ok4}")


# ------------------------------------------------------------------ A7
def check_a7_cooldown_and_max_open(tmp: Path) -> None:
    clock = FakeClock(BASE + STEP)
    tr, j = make_trader(tmp, clock, cooldown=2, rules={"take_profit_pct": None, "stop_loss_pct": None,
                                                       "max_hold_bars": None}, intrabar=False)
    tr.on_signal(mk_signal(price=100.0, sid="s7a"), price_now=100.0)
    clock.ms = BASE + 2 * STEP
    act_close = tr.on_signal(mk_signal(action="SELL", conf=0.7, price=100.0, sid="s7b"), price_now=100.0)
    ok_close = act_close.action == A_CLOSE
    # cooldown: через 1 бар - ABSTAIN, через 2 бара - OPEN
    clock.ms = BASE + 3 * STEP
    a1 = tr.on_signal(mk_signal(price=100.0, sid="s7c"), price_now=100.0)
    ok_cd = a1.action == A_ABSTAIN and "cooldown" in a1.reason
    clock.ms = BASE + 4 * STEP
    a2 = tr.on_signal(mk_signal(price=100.0, sid="s7d"), price_now=100.0)
    ok_open_after = a2.action == A_OPEN
    # max_open=1: BUY при открытой -> HOLD, второй позиции нет
    clock.ms = BASE + 5 * STEP
    a3 = tr.on_signal(mk_signal(price=100.1, sid="s7e"), price_now=100.1)
    ok_max = a3.action == A_HOLD and len(j.paper_positions("OPEN")) == 1
    record("A7 cooldown_bars соблюдается; max_open_positions=1 - конфликтующие позиции не открываются",
           ok_close and ok_cd and ok_open_after and ok_max,
           f"close={ok_close}, cooldown_abstain={ok_cd}, open_after={ok_open_after}, max_open={ok_max}")


# ------------------------------------------------------------------ A8
def check_a8_persistence(tmp: Path) -> None:
    path = tmp / "paper_persist.json"
    clock = FakeClock(BASE + STEP)
    j1 = Journal(path)
    tr1 = PaperTrader(j1, symbol="BTCUSDT", category="spot", fee_pct=0.1, slippage_pct=0.02,
                      order_size_quote=100.0,
                      exit_rules={"take_profit_pct": 1.0, "stop_loss_pct": 0.5, "max_hold_bars": 10},
                      entry_min_confidence=None, cooldown_bars=0, max_open_positions=1,
                      intrabar_exits=True, exit_at_horizon=True,
                      partial_tp={"enabled": False, "at_pct": None, "fraction": 0.5},
                      interval_step_ms=STEP, horizon_bars=3, clock_ms=clock)
    tr1.on_signal(mk_signal(price=100.0, sid="s8a"), price_now=100.0)
    # "перезапуск": новый Journal + новый PaperTrader из того же файла
    j2 = Journal(path)
    clock2 = FakeClock(BASE + 2 * STEP)
    tr2 = PaperTrader(j2, symbol="BTCUSDT", category="spot", fee_pct=0.1, slippage_pct=0.02,
                      order_size_quote=100.0,
                      exit_rules={"take_profit_pct": 1.0, "stop_loss_pct": 0.5, "max_hold_bars": 10},
                      entry_min_confidence=None, cooldown_bars=0, max_open_positions=1,
                      intrabar_exits=True, exit_at_horizon=True,
                      partial_tp={"enabled": False, "at_pct": None, "fraction": 0.5},
                      interval_step_ms=STEP, horizon_bars=3, clock_ms=clock2)
    opens = tr2.open_positions()
    ok_load = len(opens) == 1 and opens[0]["source_signal"]["signal_id"] == "s8a"
    act = tr2.on_tick(99.4, clock2.ms)  # SL сработал и после перезапуска
    ok_close = act is not None and act.action == A_CLOSE and act.rule == "stop_loss_pct"
    j3 = Journal(path)
    ok_final = len(j3.paper_positions("CLOSED")) == 1 and len(j3.paper_positions("OPEN")) == 0
    record("A8 персистентность: открытая виртуальная позиция переживает перезапуск и продолжает сопровождаться",
           ok_load and ok_close and ok_final, f"reload={ok_load}, sl_after_restart={ok_close}, final={ok_final}")


# ------------------------------------------------------------------ A9
def check_a9_no_trading_calls() -> None:
    net_imports = ["import requests", "import websockets", "import urllib", "from urllib",
                   "http.client", "socket.socket"]
    targets = [PROJECT_ROOT / "btc_bot" / "portfolio" / "paper.py",
               PROJECT_ROOT / "btc_bot" / "portfolio" / "journal.py",
               PROJECT_ROOT / "btc_bot" / "portfolio" / "advisor.py",
               PROJECT_ROOT / "scripts" / "position_cli.py"]
    net_hits = []
    for f in targets:
        text = f.read_text(encoding="utf-8")
        for pat in net_imports:
            if pat in text:
                net_hits.append(f"{f.name}:{pat}")
    p1, p2, p3 = "/v5/" + "order", "/v5/" + "position", "/v5/" + "account"
    p4, p5, p6 = "v5/" + "private", "X-" + "BAPI", "api_" + "key"
    forbidden = [p1, p2, p3, p4, p5, p6, "api_" + "secret", "place_" + "order",
                 "create_" + "order", "cancel_" + "order"]
    trade_hits = []
    for root in (PROJECT_ROOT / "btc_bot", PROJECT_ROOT / "scripts"):
        for py in sorted(root.rglob("*.py")):
            if "manual_checks" in py.parts:
                continue
            text = py.read_text(encoding="utf-8")
            for pat in forbidden:
                if pat in text:
                    trade_hits.append(f"{py.relative_to(PROJECT_ROOT)}:{pat}")
    record("A9 paper/journal/advisor/CLI офлайн; во всём коде нет торговых/приватных эндпоинтов и ключей",
           not net_hits and not trade_hits, f"net={net_hits}, trade={trade_hits}")


# ------------------------------------------------------------------ B
def check_b_replay_end_to_end(tmp: Path) -> None:
    # синтетический рынок -> CSV -> MonitorBot.run_test_replay + PaperTrader
    rng = np.random.default_rng(101)
    rows, price = [], 60_000.0
    n = 80
    for i in range(n):
        o = price
        c = o * (1 + rng.normal(0, 0.0015))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0007)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0007)))
        v = float(rng.uniform(5, 40))
        rows.append({"start_ms": BASE + i * STEP, "open": o, "high": h, "low": l,
                     "close": c, "volume": v, "turnover": v * c, "is_closed": True})
        price = c
    df = pd.DataFrame(rows)
    csv_path = tmp / "replay.csv"
    from btc_bot.data.store import start_ms_to_iso
    df["time_utc"] = [start_ms_to_iso(int(t)) for t in df["start_ms"]]
    save_klines_csv(df, csv_path, source="synthetic-demo", category="spot",
                    symbol="BTCUSDT", interval=INTERVAL)

    # маленькая модель с низким порогом, чтобы BUY/SELL действия возникали
    X, y, _, _ = build_dataset(df, 1, INTERVAL, features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=30)
    card = build_card(model_version=make_model_version("BTCUSDT", "spot", INTERVAL, 1, "ratio"),
                      algorithm="xgboost", symbol="BTCUSDT", category="spot", interval=INTERVAL,
                      horizon_bars=1, horizon_minutes=5, feature_set="ratio",
                      features=["open_rel", "high_rel", "low_rel", "volume", "turnover"],
                      confidence_threshold=0.52, library_versions=library_versions())
    save_artifact(model, tmp / "mb", card)
    pred = Predictor.load(tmp / "mb", 0.52)

    cfg = load_config(overrides={"mode": "TEST", "model": {"horizon_bars": 1, "confidence_threshold": 0.52}})
    journal = Journal(tmp / "replay_journal.json")
    clock_holder = {}
    tr = PaperTrader(journal, symbol="BTCUSDT", category="spot",
                     fee_pct=0.1, slippage_pct=0.02, order_size_quote=100.0,
                     exit_rules={"take_profit_pct": 0.25, "stop_loss_pct": 0.15, "max_hold_bars": 4},
                     entry_min_confidence=None, cooldown_bars=1, max_open_positions=1,
                     intrabar_exits=False, exit_at_horizon=True,
                     partial_tp={"enabled": False, "at_pct": None, "fraction": 0.5},
                     interval_step_ms=STEP, horizon_bars=1)
    ui = ConsoleUI(stream=io.StringIO(), force_line_mode=True)
    bot = MonitorBot(cfg, pred, ui, journal=journal, advisor=None, paper=tr)
    rc = bot.run_test_replay(csv_path, n_bars=60, speed_bps=0.0, demo_stale=False)

    st = journal.paper_stats(cfg.paper.initial_balance_quote)
    problems = []
    if rc != 0:
        problems.append(f"rc={rc}")
    n_pos = len(journal.paper_positions())
    n_open_actions = sum(1 for a in journal.paper_actions(10000) if a["kind"] == "OPEN_VIRTUAL")
    if n_open_actions != n_pos:
        problems.append(f"OPEN-действий {n_open_actions} != позиций {n_pos}")
    if abs(st["balance_quote"] - (st["initial_balance_quote"] + st["realized_pnl_quote"])) > 1e-6:
        problems.append("баланс не сходится с realized")
    for p in journal.paper_positions("CLOSED"):
        if p.get("outcome") not in ("WIN", "LOSS", "FLAT"):
            problems.append(f"{p['position_id']}: нет outcome")
        for f in p["fills"]:
            if f["type"].startswith("paper_exit") and not f.get("rule"):
                problems.append(f"{p['position_id']}: выход без правила")
        rsum = sum(f.get("realized_quote", 0.0) for f in p["fills"] if f["type"].startswith("paper_exit"))
        if abs(rsum - float(p.get("realized_pnl_quote") or 0.0)) > 1e-6:
            problems.append(f"{p['position_id']}: realized не согласован с fills")
    view = bot.build_view()
    text = format_snapshot_text(view)
    if "PAPER TRADING" not in text:
        problems.append("paper-блок отсутствует в дашборде")
    line = event_paper_text(tr.last_action) if tr.last_action else ""
    if tr.last_action and "PAPER" not in line:
        problems.append("event_paper_text некорректен")
    record("B end-to-end реплей + PaperTrader: баланс=initial+realized, OPEN-действия=позиции, у выходов есть правило и outcome, paper-блок в UI",
           not problems,
           "; ".join(problems) if problems else
           f"сигналов={len(bot.engine.signals)}, вирт. позиций={n_pos} "
           f"(открыто {st['n_open']}, закрыто {st['n_closed']} W{st['wins']}/L{st['losses']}), "
           f"баланс={st['balance_quote']:,.4f} USDT, действий={st['n_actions']}")


def main() -> int:
    print("== Этап 5b: внутрисвечной анализ (publicTrade/spikes) + paper trading (офлайн) ==")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_a1_publictrade_parsing()
        check_a2_spike_detector()
        check_a3_paper_open(tmp)
        check_a4_hold_abstain(tmp)
        check_a5_close_paths(tmp)
        check_a6_partial_close(tmp)
        check_a7_cooldown_and_max_open(tmp)
        check_a8_persistence(tmp)
    check_a9_no_trading_calls()
    with tempfile.TemporaryDirectory() as td:
        check_b_replay_end_to_end(Path(td))

    failed = [name for name, ok, _ in _results if not ok]
    print()
    if failed:
        print(f"RESULT: FAIL ({len(failed)} проваленных: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (проверок пройдено: {len(_results)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
