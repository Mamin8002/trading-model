#!/usr/bin/env python3
r"""Ручная проверка Этапа 4: история сигналов, ручные позиции, сопровождение.

Запуск (PowerShell, из корня проекта):
    python scripts\manual_checks\check_stage4_positions.py

Полностью офлайн (без сети и секретов); файлы создаются только во временных каталогах.

Проверяются:
  A1  запись сигнала + регистрация входа с привязкой к сигналу (сигнал НЕ исполнен до регистрации)
  A2  частичное и полное закрытие: количества, средняя цена выхода, реализованный PnL, статусы
  A3  персистентность журнала после "перезапуска" + защита от молчаливой потери при повреждении
  A4  открытая позиция + новый сигнал: держать / рассмотреть закрытие / наблюдать;
      срабатывание НАСТРОЕННЫХ условий выхода (TP/SL/max_hold) с объяснением причин
  A5  SELL без зарегистрированной позиции -> "состояние портфеля неизвестно", без приказа закрыть
  A6  BUY без заданных условий выхода -> воздержание от рекомендации на вход; с условиями -> вход возможен
  A7  устаревшие данные -> рекомендации по управлению позицией не выдаются
  A8  регистрация SHORT для spot отклоняется (SELL != шорт)
  A9  event-log: фиксации сигналов (цена на момент выдачи), входов/выходов, причин и результата
  A10 UI-блок позиций и рекомендации в дашборде
  A11 структурный скан: portfolio/CLI без сети и без торговых эндпоинтов/ключей

Ожидаемый вывод: [PASS] A1..A11, "RESULT: PASS (проверок пройдено: 11)", код возврата 0.
"""

from __future__ import annotations

import io
import json
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
from btc_bot.model.features import build_features  # noqa: E402
from btc_bot.model.predictor import Predictor  # noqa: E402
from btc_bot.model.train import build_card, library_versions, make_model_version, save_artifact, train_final  # noqa: E402
from btc_bot.portfolio.advisor import (  # noqa: E402
    A_CONSIDER_CLOSE, A_ENTRY_POSSIBLE, A_ENTRY_WITHHELD, A_HOLD_POSITION,
    A_NO_ADVICE_STALE, A_OBSERVE, A_PORTFOLIO_UNKNOWN, PositionAdvisor,
)
from btc_bot.portfolio.journal import Journal, JournalError  # noqa: E402
from btc_bot.signals.engine import Signal  # noqa: E402
from btc_bot.streaming.processor import StreamProcessor  # noqa: E402
from btc_bot.ui.console import ConsoleUI, format_snapshot_text  # noqa: E402

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


def mk_signal(action="BUY", conf=0.60, fresh=True, bar_start=BASE, price=100.0,
              sid="sig-test-000001", version="xgb-test-5m-h1", horizon=1, reasons=()) -> Signal:
    return Signal(
        signal_id=sid, created_at_ms=bar_start + STEP, bar_start_ms=bar_start,
        bar_end_ms=bar_start + STEP, symbol="BTCUSDT", category="spot", interval=INTERVAL,
        action=action, confidence=conf,
        prob_up=conf if action == "BUY" else (1.0 - conf),
        raw_direction=1 if action == "BUY" else -1,
        price_at_signal=price, price_source="bar_close", model_version=version,
        horizon_bars=horizon, horizon_minutes=horizon * 5, feature_set="ratio",
        data_fresh=fresh, reasons=tuple(reasons), origin="replay", features={},
    )


def make_advisor(journal: Journal, default_rules=None, clock=None) -> PositionAdvisor:
    return PositionAdvisor(journal, default_exit_rules=default_rules or {},
                           interval_step_ms=STEP, clock_ms=clock or FakeClock(BASE + STEP))


# ----------------------------------------------------------------------- A1
def check_a1(tmp: Path) -> None:
    j = Journal(tmp / "j1.json")
    sig = mk_signal(action="BUY", conf=0.61, price=84_500.0, sid="sig-a1")
    entry = j.record_signal(sig)
    ok_not_executed = entry["entry_confirmed"] is False and entry["position_id"] is None
    pos = j.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                           price=84_510.0, quantity=0.01, opened_at_ms=BASE + 2 * STEP,
                           signal_id="sig-a1",
                           exit_rules={"take_profit_pct": 1.5, "stop_loss_pct": 1.0, "max_hold_bars": 12},
                           note="проверка")
    sig_after = j.find_signal("sig-a1")
    src = pos.data["source_signal"]
    ok_link = (sig_after["entry_confirmed"] is True and sig_after["position_id"] == pos.position_id
               and src["signal_id"] == "sig-a1" and src["action"] == "BUY"
               and abs(src["confidence"] - 0.61) < 1e-9
               and src["signal_created_at_ms"] == BASE + STEP
               and src["price_at_signal"] == 84_500.0
               and src["horizon_minutes"] == 5 and src["model_version"] == "xgb-test-5m-h1")
    ok_state = (pos.status == "OPEN" and pos.quantity_open == 0.01
                and pos.exit_rules["take_profit_pct"] == 1.5)
    # повторная привязка того же сигнала запрещена
    try:
        j.register_entry(symbol="BTCUSDT", category="spot", direction="LONG", price=1.0,
                         quantity=1.0, signal_id="sig-a1")
        ok_dup = False
    except JournalError:
        ok_dup = True
    record("A1 сигнал не исполнен до регистрации; вход связан с сигналом (время/цена/горизонт/версия); повторная привязка запрещена",
           ok_not_executed and ok_link and ok_state and ok_dup,
           f"not_executed={ok_not_executed}, link={ok_link}, state={ok_state}, dup_guard={ok_dup}")


# ----------------------------------------------------------------------- A2
def check_a2(tmp: Path) -> None:
    j = Journal(tmp / "j2.json")
    pos = j.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                           price=100.0, quantity=0.05, opened_at_ms=BASE)
    p1 = j.close_position(pos.position_id, price=110.0, quantity=0.02, closed_at_ms=BASE + 10 * STEP)
    ok_partial = (p1.status == "OPEN" and abs(p1.quantity_open - 0.03) < 1e-12
                  and abs(p1.data["realized_pnl_quote"] - 0.2) < 1e-9)
    p2 = j.close_position(pos.position_id, price=90.0, closed_at_ms=BASE + 20 * STEP)  # остаток
    exp_avg = (110.0 * 0.02 + 90.0 * 0.03) / 0.05          # = 98.0
    exp_pnl = 0.2 + (90.0 - 100.0) * 0.03                    # = -0.1
    exp_pct = exp_pnl / (100.0 * 0.05) * 100.0               # = -0.2%
    ok_full = (p2.status == "CLOSED" and p2.quantity_open == 0.0
               and abs(p2.data["avg_exit_price"] - exp_avg) < 1e-9
               and abs(p2.data["realized_pnl_quote"] - exp_pnl) < 1e-9
               and abs(p2.data["realized_pnl_pct"] - exp_pct) < 1e-9
               and p2.data["outcome"] == "LOSS")
    ok_over = False
    pos3 = j.register_entry(symbol="BTCUSDT", category="spot", direction="LONG", price=100.0, quantity=0.01)
    try:
        j.close_position(pos3.position_id, price=100.0, quantity=0.02)
    except JournalError:
        ok_over = True
    ok_closed_again = False
    try:
        j.close_position(pos.position_id, price=100.0)
    except JournalError:
        ok_closed_again = True
    record("A2 частичное/полное закрытие: остатки, средняя цена выхода 98.0, PnL -0.1 USDT (-2.0% от ноты 5 USDT) LOSS; guard перевыставления",
           ok_partial and ok_full and ok_over and ok_closed_again,
           f"partial={ok_partial}, full={ok_full}, overclose_guard={ok_over}, reclose_guard={ok_closed_again}")


# ----------------------------------------------------------------------- A3
def check_a3(tmp: Path) -> None:
    path = tmp / "j3.json"
    j = Journal(path)
    j.record_signal(mk_signal(sid="sig-a3"))
    j.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                     price=100.0, quantity=0.1, signal_id="sig-a3")
    st1 = j.stats()
    j2 = Journal(path)  # "перезапуск"
    st2 = j2.stats()
    ok_persist = (st1 == st2 and st2["n_positions"] == 1 and st2["n_open"] == 1
                  and j2.open_positions()[0].entry_price == 100.0
                  and j2.find_signal("sig-a3")["entry_confirmed"] is True)
    # повреждённый журнал: ошибка, файл НЕ перезаписывается молча
    bad = tmp / "j3bad.json"
    bad.write_text("{not json", encoding="utf-8")
    try:
        Journal(bad)
        _ = Journal(bad)._load()
        ok_corrupt = False
    except JournalError:
        ok_corrupt = bad.read_text(encoding="utf-8") == "{not json"
    record("A3 персистентность после перезапуска + повреждённый журнал не перезаписывается молча",
           ok_persist and ok_corrupt, f"persist={ok_persist}, corrupt_guard={ok_corrupt}")


# ----------------------------------------------------------------------- A4
def check_a4(tmp: Path) -> None:
    j = Journal(tmp / "j4.json")
    clock = FakeClock(BASE + STEP)
    adv = make_advisor(j, clock=clock)
    pos = j.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                           price=100.0, quantity=1.0, opened_at_ms=BASE - 10 * STEP,
                           exit_rules={"take_profit_pct": 1.0, "stop_loss_pct": 0.5, "max_hold_bars": 100})
    # встречный SELL + цена выше TP -> CONSIDER_CLOSE с BOTH причинами
    a1 = adv.evaluate(mk_signal(action="SELL", conf=0.62, sid="s4a"), price_now=101.5)
    ok1 = (a1.action == A_CONSIDER_CLOSE and a1.position_id == pos.position_id
           and any("встречный сигнал" in r for r in a1.reasons)
           and any("take_profit_pct" in t for t in a1.triggered_rules))
    # BUY в направлении позиции, правила не сработали -> ДЕРЖАТЬ
    a2 = adv.evaluate(mk_signal(action="BUY", conf=0.60, sid="s4b"), price_now=100.2)
    ok2 = a2.action == A_HOLD_POSITION and a2.position_id == pos.position_id
    # HOLD -> НАБЛЮДАТЬ
    a3 = adv.evaluate(mk_signal(action="HOLD", conf=0.52, sid="s4c"), price_now=100.2)
    ok3 = a3.action == A_OBSERVE
    # SELL без сработавших правил -> CONSIDER_CLOSE c пометкой, что правила не сработали
    a4 = adv.evaluate(mk_signal(action="SELL", conf=0.58, sid="s4d"), price_now=100.2)
    ok4 = (a4.action == A_CONSIDER_CLOSE
           and any("не сработали" in r for r in a4.reasons) and not a4.triggered_rules)
    # SL пробит (99.4 <= уровня 99.5): даже при сигнале HOLD -> CONSIDER_CLOSE (условие выхода сработало)
    a5 = adv.evaluate(mk_signal(action="HOLD", conf=0.52, sid="s4e"), price_now=99.4)
    ok5 = (a5.action == A_CONSIDER_CLOSE and any("stop_loss_pct" in t for t in a5.triggered_rules))
    a5b = adv.evaluate(mk_signal(action="SELL", conf=0.56, sid="s4f"), price_now=99.4)
    ok5 = (ok5 and a5b.action == A_CONSIDER_CLOSE
           and any("stop_loss_pct" in t for t in a5b.triggered_rules)
           and any("встречный сигнал" in r for r in a5b.reasons))
    # max_hold: позиция открыта 10 баров назад, лимит 5
    j2 = Journal(tmp / "j4b.json")
    adv2 = make_advisor(j2, clock=clock)
    pos2 = j2.register_entry(symbol="BTCUSDT", category="spot", direction="LONG", price=100.0,
                             quantity=1.0, opened_at_ms=BASE - 10 * STEP,
                             exit_rules={"max_hold_bars": 5})
    a6 = adv2.evaluate(mk_signal(action="BUY", conf=0.60, sid="s4g"), price_now=100.1)
    ok6 = a6.action == A_CONSIDER_CLOSE and any("max_hold_bars" in t for t in a6.triggered_rules)
    record("A4 сопровождение открытой позиции: ДЕРЖАТЬ/НАБЛЮДАТЬ/РАССМОТРЕТЬ ЗАКРЫТИЕ; TP, SL и max_hold срабатывают с объяснением",
           ok1 and ok2 and ok3 and ok4 and ok5 and ok6,
           f"sell_tp={ok1}, hold={ok2}, observe={ok3}, sell_norules={ok4}, sl={ok5}, max_hold={ok6}")


# ----------------------------------------------------------------------- A5
def check_a5(tmp: Path) -> None:
    j = Journal(tmp / "j5.json")
    adv = make_advisor(j)
    a = adv.evaluate(mk_signal(action="SELL", conf=0.70, sid="s5"), price_now=100.0)
    ok = (a.action == A_PORTFOLIO_UNKNOWN
          and "НЕИЗВЕСТНО" in a.text
          and "Короткая позиция НЕ открывается" in a.text
          and not j.positions()
          and a.action != A_CONSIDER_CLOSE)
    record("A5 SELL без зарегистрированной позиции: портфель НЕИЗВЕСТЕН, приказа закрыть нет, шорт не открывается",
           ok, f"action={a.action}")


# ----------------------------------------------------------------------- A6
def check_a6(tmp: Path) -> None:
    j = Journal(tmp / "j6.json")
    adv_no_rules = make_advisor(j, default_rules=None)
    a1 = adv_no_rules.evaluate(mk_signal(action="BUY", conf=0.65, sid="s6a"), price_now=100.0)
    ok1 = a1.action == A_ENTRY_WITHHELD and "ВОЗДЕРЖАНА" in a1.text and any("условия выхода" in r for r in a1.reasons)
    adv_rules = make_advisor(j, default_rules={"take_profit_pct": 2.0, "stop_loss_pct": 1.0, "max_hold_bars": None})
    a2 = adv_rules.evaluate(mk_signal(action="BUY", conf=0.65, sid="s6b"), price_now=100.0)
    ok2 = a2.action == A_ENTRY_POSSIBLE and "вручную" in a2.text
    # история сигналов учитывается: 3-й BUY подряд отмечается в причинах
    for _ in range(2):
        j.record_signal(mk_signal(action="BUY", sid=f"s6x{_}"))
    a3 = adv_rules.evaluate(mk_signal(action="BUY", conf=0.65, sid="s6c"), price_now=100.0)
    j.record_signal(mk_signal(action="BUY", conf=0.65, sid="s6c"))
    ok3 = any("подряд" in r for r in a3.reasons)
    record("A6 BUY без условий выхода -> воздержание от рекомендации на вход; с условиями -> вход возможен; история сигналов учитывается",
           ok1 and ok2 and ok3, f"withheld={ok1}, possible={ok2}, history={ok3}")


# ----------------------------------------------------------------------- A7
def check_a7(tmp: Path) -> None:
    j = Journal(tmp / "j7.json")
    adv = make_advisor(j)
    j.register_entry(symbol="BTCUSDT", category="spot", direction="LONG", price=100.0, quantity=1.0,
                     exit_rules={"stop_loss_pct": 0.5})
    stale = mk_signal(action="SELL", conf=0.70, sid="s7", fresh=False,
                      reasons=("устаревший бар",))
    a = adv.evaluate(stale, price_now=90.0)  # даже при цене ниже SL
    ok = a.action == A_NO_ADVICE_STALE and "не выдаются" in a.text
    record("A7 устаревшие данные: рекомендации по управлению позицией НЕ выдаются (даже если SL формально пробит)",
           ok, f"action={a.action}")


# ----------------------------------------------------------------------- A8
def check_a8(tmp: Path) -> None:
    j = Journal(tmp / "j8.json")
    try:
        j.register_entry(symbol="BTCUSDT", category="spot", direction="SHORT", price=100.0, quantity=1.0)
        ok = False
        msg = "регистрация SHORT прошла (не должна)"
    except JournalError as e:
        ok = "SHORT запрещена" in str(e)
        msg = str(e)[:80]
    record("A8 регистрация SHORT для spot отклоняется (SELL != открытие шорта)", ok, msg)


# ----------------------------------------------------------------------- A9
def check_a9(tmp: Path) -> None:
    j = Journal(tmp / "j9.json")
    j.record_signal(mk_signal(action="BUY", conf=0.66, price=123.45, sid="s9"))
    pos = j.register_entry(symbol="BTCUSDT", category="spot", direction="LONG", price=123.5,
                           quantity=0.2, signal_id="s9")
    j.close_position(pos.position_id, price=124.0, quantity=0.1, note="частично")
    j.close_position(pos.position_id, price=125.0, note="остаток")
    kinds = [e["kind"] for e in j.events(100)]
    sig = j.find_signal("s9")
    close_ev = [e for e in j.events(100) if e["kind"] == "position_closed"][-1]
    ok = ("signal_recorded" in kinds and "entry_registered" in kinds
          and "position_partial_close" in kinds and "position_closed" in kinds
          and sig["price_at_signal"] == 123.45
          and "Результат" in close_ev["reason"] and close_ev["status"] == "CLOSED"
          and j.get_position(pos.position_id).data["outcome"] in ("WIN", "LOSS", "FLAT"))
    record("A9 event-log: сигналы (цена на момент выдачи), подтверждения входа/выхода, причины смены статуса, результат позиции",
           ok, f"kinds={sorted(set(kinds))}, outcome={j.get_position(pos.position_id).data['outcome']}")


# ---------------------------------------------------------------------- A10
def check_a10(tmp: Path) -> None:
    # UI: позиции + рекомендация в дашборде (детерминированно: регистрируем позицию заранее)
    def synthetic(n, seed=31):
        rng = np.random.default_rng(seed)
        rows, price = [], 60_000.0
        for i in range(n):
            o = price
            c = o * (1 + rng.normal(0, 0.002))
            h = max(o, c) * (1 + abs(rng.normal(0, 0.0008)))
            l = min(o, c) * (1 - abs(rng.normal(0, 0.0008)))
            v = float(rng.uniform(1, 40))
            rows.append({"start_ms": BASE + i * STEP, "open": o, "high": h, "low": l,
                         "close": c, "volume": v, "turnover": v * c, "is_closed": True})
            price = c
        return pd.DataFrame(rows)

    df = synthetic(500)
    X, y, _, _ = build_dataset(df, 1, INTERVAL, features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=25)
    card = build_card(model_version=make_model_version("BTCUSDT", "spot", INTERVAL, 1, "ratio"),
                      algorithm="xgboost", symbol="BTCUSDT", category="spot", interval=INTERVAL,
                      horizon_bars=1, horizon_minutes=5, feature_set="ratio",
                      features=["open_rel", "high_rel", "low_rel", "volume", "turnover"],
                      confidence_threshold=0.55, library_versions=library_versions())
    save_artifact(model, tmp / "m10", card)
    pred = Predictor.load(tmp / "m10", 0.55)

    clock = FakeClock(BASE)
    cfg = load_config(overrides={"mode": "TEST", "model": {"horizon_bars": 1, "confidence_threshold": 0.55}})
    journal = Journal(tmp / "j10.json")
    pos = journal.register_entry(symbol="BTCUSDT", category="spot", direction="LONG",
                                 price=60_000.0, quantity=0.5, opened_at_ms=BASE - 5 * STEP,
                                 exit_rules={"take_profit_pct": 2.0, "stop_loss_pct": 1.0, "max_hold_bars": 50})
    advisor = make_advisor(journal, clock=clock)
    ui = ConsoleUI(stream=io.StringIO(), force_line_mode=True)
    bot = MonitorBot(cfg, pred, ui, clock_ms=clock, journal=journal, advisor=advisor)

    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    clock.ms = BASE + STEP
    row = df.iloc[0].to_dict()
    msg = {"topic": f"kline.{INTERVAL}.BTCUSDT", "ts": clock.ms,
           "data": [{"start": BASE, "end": BASE + STEP - 1, "interval": INTERVAL,
                     "open": str(row["open"]), "close": str(row["close"]),
                     "high": str(row["high"]), "low": str(row["low"]),
                     "volume": str(row["volume"]), "turnover": str(row["turnover"]),
                     "confirm": True, "timestamp": clock.ms}]}
    sigs = bot.dispatch(bot.processor.on_ws_message(msg))
    text = format_snapshot_text(bot.build_view())
    ok = (len(sigs) == 1
          and journal.stats()["n_signals"] >= 1
          and pos.position_id in text
          and "Открытые позиции" in text
          and "Рекомендация [" in text
          and bot.last_advice is not None
          and bot.last_advice.position_id == pos.position_id
          and "условия выхода" in text)
    record("A10 UI: блок открытых позиций + рекомендация по сигналу; сигнал записан в журнал",
           ok, f"signals={len(sigs)}, advice={bot.last_advice.action if bot.last_advice else None}, "
              f"pos_in_ui={pos.position_id in text}")


# ---------------------------------------------------------------------- A11
def check_a11_no_network_no_trading() -> None:
    net_imports = ["import requests", "import websockets", "import urllib", "from urllib",
                   "http.client", "socket.socket"]
    targets = [PROJECT_ROOT / "btc_bot" / "portfolio" / "journal.py",
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
    forbidden = [p1, p2, p3, p4, p5, p6, "api_" + "secret", "place_" + "order"]
    trade_hits = []
    for root in (PROJECT_ROOT / "btc_bot", PROJECT_ROOT / "scripts"):
        for py in sorted(root.rglob("*.py")):
            if "manual_checks" in py.parts:
                continue
            text = py.read_text(encoding="utf-8")
            for pat in forbidden:
                if pat in text:
                    trade_hits.append(f"{py.relative_to(PROJECT_ROOT)}:{pat}")
    record("A11 portfolio/CLI офлайн (без сетевых импортов); во всём коде нет торговых/приватных эндпоинтов и ключей",
           not net_hits and not trade_hits,
           f"net_hits={net_hits}, trade_hits={trade_hits}")


def main() -> int:
    print("== Этап 4: журнал сигналов, ручные позиции, сопровождение (офлайн) ==")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        check_a1(tmp)
        check_a2(tmp)
        check_a3(tmp)
        check_a4(tmp)
        check_a5(tmp)
        check_a6(tmp)
        check_a7(tmp)
        check_a8(tmp)
        check_a9(tmp)
        check_a10(tmp)
    check_a11_no_network_no_trading()

    failed = [name for name, ok, _ in _results if not ok]
    print()
    if failed:
        print(f"RESULT: FAIL ({len(failed)} проваленных: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (проверок пройдено: {len(_results)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
