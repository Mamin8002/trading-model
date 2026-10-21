#!/usr/bin/env python3
r"""Ручная проверка Этапа 3: поток данных, закрытие свечи, потеря соединения, устаревшие данные.

Запуск (PowerShell, из корня проекта btc_bybit_bot):
    python scripts\manual_checks\check_stage3_stream.py
Опции:
    --live          дополнительная read-only проверка реального WS Bybit (~45 с)
    --wait-confirm  с --live: ждать закрытия реальной свечи (до ~7 минут)

По умолчанию работает БЕЗ внешней сети и без секретов: сценарии A1-A8 - детерминированный
реплей сообщений с фейковыми часами; A9 - структурный скан кода на отсутствие торговых
вызовов; A10 - полный LIVE-тракт на ЛОКАЛЬНОМ fake-WS-сервере (127.0.0.1, эфемерный порт).
Файлы не создаются (тестовые модели - во временных каталогах).

Ожидаемый вывод: строки [PASS]/[FAIL], итог "RESULT: PASS (проверок пройдено: 10)",
код возврата 0. Время выполнения: ~15-25 с.
"""

from __future__ import annotations

import argparse
import io
import json
import queue
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btc_bot.bot import MonitorBot  # noqa: E402
from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import build_dataset  # noqa: E402
from btc_bot.data_sources.base import Kline  # noqa: E402
from btc_bot.model.features import build_features  # noqa: E402
from btc_bot.model.predictor import Predictor  # noqa: E402
from btc_bot.model.train import build_card, library_versions, make_model_version, save_artifact, train_final  # noqa: E402
from btc_bot.signals.engine import SignalEngine  # noqa: E402
from btc_bot.streaming.base import (  # noqa: E402
    ConnState,
    FreshnessChange,
    GapDetected,
    KlineBar,
    StateChange,
    TickerUpdate,
)
from btc_bot.streaming.bybit_ws import backoff_delay_s  # noqa: E402
from btc_bot.streaming.processor import StreamProcessor  # noqa: E402
from btc_bot.ui.console import ConsoleUI, format_snapshot_text  # noqa: E402

INTERVAL = "5"
STEP = 300_000
BASE = 1_800_000_200_000 - (1_800_000_200_000 % STEP)  # выровнено по 5 минутам

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


class FakeClock:
    def __init__(self, ms: int):
        self.ms = ms

    def __call__(self) -> int:
        return self.ms


def synthetic_bars(n: int, seed: int = 21) -> pd.DataFrame:
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


def make_test_predictor(tmpdir: Path, threshold: float = 0.55) -> Predictor:
    df = synthetic_bars(500)
    X, y, _, _ = build_dataset(df, 1, INTERVAL, features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=25)
    card = build_card(
        model_version=make_model_version("BTCUSDT", "spot", INTERVAL, 1, "ratio"),
        algorithm="xgboost", symbol="BTCUSDT", category="spot", interval=INTERVAL,
        horizon_bars=1, horizon_minutes=5, feature_set="ratio",
        features=["open_rel", "high_rel", "low_rel", "volume", "turnover"],
        confidence_threshold=threshold, library_versions=library_versions(),
    )
    save_artifact(model, tmpdir, card)
    return Predictor.load(tmpdir, confidence_threshold=threshold)


def kline_msg(start_ms: int, row: dict, confirm: bool, ts_ms: int) -> dict:
    return {
        "topic": f"kline.{INTERVAL}.BTCUSDT", "type": "snapshot", "ts": ts_ms,
        "data": [{"start": start_ms, "end": start_ms + STEP - 1, "interval": INTERVAL,
                  "open": str(row["open"]), "close": str(row["close"]),
                  "high": str(row["high"]), "low": str(row["low"]),
                  "volume": str(row["volume"]), "turnover": str(row["turnover"]),
                  "confirm": confirm, "timestamp": ts_ms}],
    }


def make_bot(predictor: Predictor, clock: FakeClock) -> MonitorBot:
    cfg = load_config(overrides={"mode": "TEST"})
    ui = ConsoleUI(stream=io.StringIO(), force_line_mode=True)
    return MonitorBot(cfg, predictor, ui, clock_ms=clock)


# --------------------------------------------------------------- сценарии A
def check_a1_full_pipeline(tmp: Path) -> None:
    clock = FakeClock(BASE)
    pred = make_test_predictor(tmp / "m1")
    bot = make_bot(pred, clock)
    df = synthetic_bars(6)

    events = bot.processor.on_connection_state(ConnState.CONNECTING, "тест")
    events += bot.processor.on_ws_message(
        {"op": "subscribe", "success": True, "ret_msg": "subscribe", "ts": clock.ms})
    bot.dispatch(events)
    ok_connected = any(isinstance(e, StateChange) and e.state == ConnState.CONNECTED for e in events)

    clock.ms += 500
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    ok_fresh = bot.processor.is_fresh(clock.ms)

    clock.ms += 500
    bot.dispatch(bot.processor.on_ws_message(
        {"topic": "tickers.BTCUSDT", "ts": clock.ms,
         "data": [{"symbol": "BTCUSDT", "lastPrice": "60123.4"}]}))
    ok_price = (bot.processor.last_price is not None
                and abs(bot.processor.last_price.price - 60123.4) < 1e-9)

    # формирующаяся свеча -> сигнала НЕТ
    clock.ms = BASE + STEP // 2
    sigs = bot.dispatch(bot.processor.on_ws_message(kline_msg(BASE, df.iloc[0].to_dict(), False, clock.ms)))
    ok_no_signal_open = (not sigs and bot.engine.last_signal is None
                         and bot.processor.live_candle is not None)

    # закрытие свечи (confirm=true) -> сигнал ЕСТЬ
    clock.ms = BASE + STEP
    row0 = df.iloc[0].to_dict()
    sigs = bot.dispatch(bot.processor.on_ws_message(kline_msg(BASE, row0, True, clock.ms)))
    ok_confirm_signal = len(sigs) == 1
    s = sigs[0] if sigs else None
    ok_fields = bool(s) and (
        s.action in ("BUY", "SELL", "HOLD")
        and 0.5 <= s.confidence <= 1.0
        and s.model_version == pred.model_version
        and s.created_at_ms == clock.ms
        and s.bar_start_ms == BASE and s.bar_end_ms == BASE + STEP
        and s.data_fresh is True
        and s.symbol == "BTCUSDT" and s.category == "spot" and s.interval == INTERVAL
        and s.horizon_minutes == 5
    )
    ok_price_src = bool(s) and s.price_source == "ticker" and abs(s.price_at_signal - 60123.4) < 1e-9

    # дубликат закрытого бара -> без нового сигнала
    sigs2 = bot.dispatch(bot.processor.on_ws_message(kline_msg(BASE, row0, True, clock.ms + 10)))
    ok_dedup = not sigs2

    record("A1 полный тракт: CONNECTED, FRESH, цена, confirm=false без сигнала, confirm=true -> сигнал, дедуп",
           ok_connected and ok_fresh and ok_price and ok_no_signal_open and ok_confirm_signal
           and bool(ok_fields) and ok_price_src and ok_dedup,
           f"connected={ok_connected}, fresh={ok_fresh}, price={ok_price}, no_sig_open={ok_no_signal_open}, "
           f"signal={ok_confirm_signal}, fields={ok_fields}, ticker_price={ok_price_src}, dedup={ok_dedup}"
           + (f"; сигнал: {s.action} conf={s.confidence:.3f} ts={s.created_at_ms} ver={s.model_version}" if s else ""))


def check_a2_gap_incomplete(tmp: Path) -> None:
    clock = FakeClock(BASE)
    pred = make_test_predictor(tmp / "m2")
    bot = make_bot(pred, clock)
    df = synthetic_bars(6)
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))

    clock.ms = BASE + STEP
    bot.dispatch(bot.processor.on_ws_message(kline_msg(BASE, df.iloc[0].to_dict(), True, clock.ms)))
    # пропускаем бар BASE+STEP: сразу-confirm бар BASE+2*STEP
    clock.ms = BASE + 3 * STEP
    events = bot.processor.on_ws_message(
        kline_msg(BASE + 2 * STEP, df.iloc[2].to_dict(), True, clock.ms))
    bot.dispatch(events)
    gap_events = [e for e in events if isinstance(e, GapDetected)]
    sig_after_gap = bot.engine.last_signal
    ok = (len(gap_events) == 1
          and sig_after_gap is not None and sig_after_gap.data_fresh is False
          and any("разрыв" in r for r in sig_after_gap.reasons))
    # следующий бар без разрыва -> снова fresh
    clock.ms = BASE + 4 * STEP
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    bot.dispatch(bot.processor.on_ws_message(
        kline_msg(BASE + 3 * STEP, df.iloc[3].to_dict(), True, clock.ms)))
    ok_next = bot.engine.last_signal is not None and bot.engine.last_signal.data_fresh is True
    record("A2 разрыв сетки: GapDetected, сигнал помечен НЕ актуальным (неполные данные), следующий - fresh",
           ok and ok_next, f"gap_events={len(gap_events)}, stale_reason={[r for r in (sig_after_gap.reasons if sig_after_gap else [])]}, next_fresh={ok_next}")


def check_a3_stale(tmp: Path) -> None:
    clock = FakeClock(BASE)
    pred = make_test_predictor(tmp / "m3")
    bot = make_bot(pred, clock)
    df = synthetic_bars(6)
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))

    # тишина дольше stale-порога (3 интервала = 15 мин)
    clock.ms = BASE + 16 * 60_000 + STEP
    events = bot.processor.tick(now_ms=clock.ms)
    bot.dispatch(events)
    stale_ev = [e for e in events if isinstance(e, FreshnessChange) and not e.fresh]
    ok_stale_state = bool(stale_ev) and not bot.processor.is_fresh(clock.ms)

    # запоздалый закрытый бар -> сигнал НЕ актуален
    sigs = bot.dispatch(bot.processor.on_ws_message(
        kline_msg(BASE, df.iloc[0].to_dict(), True, clock.ms)))
    s = sigs[0] if sigs else None
    ok_signal_stale = (s is not None and s.data_fresh is False
                       and any("устаревший бар" in r for r in s.reasons))

    # UI честно показывает STALE
    view = bot.build_view()
    text = format_snapshot_text(view)
    ok_ui = ("STALE" in text) and (view["fresh"] is False)
    record("A3 устаревшие данные: FreshnessChange(STALE), сигнал помечен неактуальным, UI показывает STALE",
           ok_stale_state and bool(ok_signal_stale) and ok_ui,
           f"stale_event={bool(stale_ev)}, signal_stale={ok_signal_stale}, ui={ok_ui}")


def check_a4_disconnect_reconnect(tmp: Path) -> None:
    clock = FakeClock(BASE)
    pred = make_test_predictor(tmp / "m4")
    bot = make_bot(pred, clock)
    df = synthetic_bars(6)
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    clock.ms = BASE + STEP
    bot.dispatch(bot.processor.on_ws_message(kline_msg(BASE, df.iloc[0].to_dict(), True, clock.ms)))

    events = bot.processor.on_connection_state(ConnState.RECONNECTING, "обрыв (тест)")
    bot.dispatch(events)
    ok_reconn_state = any(isinstance(e, StateChange) and e.state == ConnState.RECONNECTING for e in events)
    ok_counted = bot.processor.reconnect_count == 1

    # во время обрыва новых баров нет -> сигналов нет
    n_before = len(bot.engine.signals)
    clock.ms += 60_000
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    ok_no_signals = len(bot.engine.signals) == n_before

    # переподключение + ack -> CONNECTED, снова идут сигналы
    bot.dispatch(bot.processor.on_connection_state(ConnState.CONNECTING, "переподключение"))
    clock.ms += 1_000
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    ok_connected = bot.processor.conn_state == ConnState.CONNECTED
    # backfill-бар (пропущенный во время обрыва) + свежий бар
    backfilled = [Kline(BASE + STEP, 10, 11, 9, 10.5, 1.0, 100.0, True),
                  Kline(BASE + 2 * STEP, 10.5, 11.5, 10, 11.0, 1.0, 100.0, True)]
    bf_signals = bot.dispatch(bot.processor.ingest_backfilled(backfilled))  # -> list[Signal]
    clock.ms = BASE + 3 * STEP
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    sigs = bot.dispatch(bot.processor.on_ws_message(
        kline_msg(BASE + 3 * STEP, df.iloc[3].to_dict(), True, clock.ms)))
    ok_resumed = (len(bf_signals) == 2 and len(sigs) == 1 and sigs[0].data_fresh is True
                  and all(s.origin == "backfill" for s in bf_signals))
    record("A4 обрыв/переподключение: RECONNECTING, нет сигналов во время обрыва, CONNECTED, backfill+сигнал fresh",
           ok_reconn_state and ok_counted and ok_no_signals and ok_connected and ok_resumed,
           f"reconnecting={ok_reconn_state}, count={bot.processor.reconnect_count}, "
           f"no_sig={ok_no_signals}, connected={ok_connected}, resumed={ok_resumed}")


def check_a5_hold_band(tmp: Path) -> None:
    clock = FakeClock(BASE)
    # порог выше максимально возможной уверенности -> HOLD детерминирован
    pred_strict = make_test_predictor(tmp / "m5", threshold=1.0001)
    bot = make_bot(pred_strict, clock)
    df = synthetic_bars(4)
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    clock.ms = BASE + STEP
    sigs = bot.dispatch(bot.processor.on_ws_message(kline_msg(BASE, df.iloc[0].to_dict(), True, clock.ms)))
    s = sigs[0] if sigs else None
    ok = (s is not None and s.action == "HOLD" and s.confidence <= 1.0
          and any("воздержание" in r for r in s.reasons)
          and s.created_at_ms == clock.ms and s.model_version == pred_strict.model_version)
    record("A5 HOLD/воздержание при уверенности ниже порога (время и версия модели фиксируются)",
           bool(ok), f"action={s.action if s else None}, conf={s.confidence if s else None}, "
                     f"reasons={list(s.reasons) if s else None}")


def check_a6_backoff() -> None:
    seq = [backoff_delay_s(i, base_s=1.0, max_s=60.0) for i in range(8)]
    ok = seq == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    record("A6 reconnect backoff: 1,2,4,...,60 c (монотонно, с капом, в пределах лимитов Bybit)",
           ok, f"seq={seq}")


def check_a7_backfill_dedup_order() -> None:
    clock = FakeClock(BASE)
    proc = StreamProcessor("BTCUSDT", INTERVAL, clock_ms=clock)
    kl_new = Kline(BASE + 2 * STEP, 10, 11, 9, 10.5, 1.0, 100.0, True)
    kl_old = Kline(BASE, 9, 10, 8, 9.5, 1.0, 100.0, True)
    ev1 = proc.ingest_backfilled([kl_new])
    ev2 = proc.ingest_backfilled([kl_old])          # поздний старый бар -> вставка по порядку
    ev3 = proc.ingest_backfilled([kl_new])          # дубликат -> игнор
    starts = [b.start_ms for b in proc.confirmed]
    ok = (len(ev1) == 1 and len(ev2) == 1 and not ev3
          and starts == [BASE, BASE + 2 * STEP]
          and all(isinstance(e, KlineBar) for e in ev1 + ev2))
    record("A7 backfill: вставка с сохранением порядка, дубликаты игнорируются", ok,
           f"starts_ok={starts == [BASE, BASE + 2 * STEP]}, dup_ignored={not ev3}")


def check_a8_ui_text(tmp: Path) -> None:
    clock = FakeClock(BASE)
    pred = make_test_predictor(tmp / "m8")
    bot = make_bot(pred, clock)
    df = synthetic_bars(6)
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    clock.ms += 100
    bot.dispatch(bot.processor.on_ws_message(
        {"topic": "tickers.BTCUSDT", "ts": clock.ms, "data": [{"symbol": "BTCUSDT", "lastPrice": "60500.5"}]}))
    clock.ms = BASE + STEP
    bot.dispatch(bot.processor.on_ws_message(kline_msg(BASE, df.iloc[0].to_dict(), True, clock.ms)))
    text = format_snapshot_text(bot.build_view())
    required = ["CONNECTED", "FRESH", "60,500.50", pred.model_version, "уверенность",
                "ТЕКУЩИЙ СИГНАЛ", "История сигналов", "не гарантия дохода"]
    missing = [r for r in required if r not in text]
    sig = bot.engine.last_signal
    if sig:
        if sig.action not in text:
            missing.append(f"action {sig.action}")
    record("A8 UI-текст содержит: цену, соединение, свежесть, сигнал+уверенность+время, версию модели, дисклеймер",
           not missing, f"missing={missing}" if missing else f"все {len(required)} элементов на месте")


def check_a10_live_pipeline_local(tmp: Path) -> None:
    """Полный LIVE-тракт (BybitWsClient -> очередь -> StreamProcessor -> SignalEngine)
    на локальном fake-WS-сервере с документированным протоколом Bybit. Без внешней сети."""
    import threading

    from websockets.sync.server import serve

    step = 300_000
    now_ms = int(time.time() * 1000)
    bar_start = (now_ms // step - 1) * step  # только что закрывшийся 5m-бар
    bar = {"start": bar_start, "end": bar_start + step - 1, "interval": "5",
           "open": "84500", "close": "84600", "high": "84650", "low": "84450",
           "volume": "25.5", "turnover": "2157000", "confirm": True, "timestamp": now_ms}

    def handler(ws):
        try:
            req = json.loads(ws.recv(timeout=10))
            if req.get("op") != "subscribe":
                return
            ws.send(json.dumps({"success": True, "ret_msg": "subscribe",
                                "op": "subscribe", "conn_id": "fake"}))
            ws.send(json.dumps({"topic": "tickers.BTCUSDT", "ts": now_ms, "type": "snapshot",
                                "data": {"symbol": "BTCUSDT", "lastPrice": "84600.5"}}))  # spot: объект!
            open_bar = dict(bar, confirm=False, close="84590", timestamp=now_ms - 1000)
            ws.send(json.dumps({"topic": "kline.5.BTCUSDT", "ts": now_ms - 1000,
                                "type": "snapshot", "data": [open_bar]}))
            ws.send(json.dumps({"topic": "kline.5.BTCUSDT", "ts": now_ms,
                                "type": "snapshot", "data": [bar]}))
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv(timeout=2)
                except TimeoutError:
                    continue
                if raw is None:
                    continue
                data = json.loads(raw)
                if data.get("op") == "ping":
                    ws.send(json.dumps({"success": True, "ret_msg": "pong",
                                        "op": "ping", "conn_id": "fake"}))
        except Exception:  # noqa: BLE001 - клиент может закрыть соединение первым
            pass

    ok = False
    detail = ""
    try:
        with serve(handler, "127.0.0.1", 0) as server:
            port = server.socket.getsockname()[1]
            threading.Thread(target=server.serve_forever, daemon=True).start()
            pred = make_test_predictor(tmp / "m10")
            cfg = load_config(overrides={
                "mode": "LIVE",
                "exchange": {"ws_public_url": f"ws://127.0.0.1:{port}/v5/public/spot"},
            })
            ui = ConsoleUI(stream=io.StringIO(), force_line_mode=True)
            bot = MonitorBot(cfg, pred, ui, rest_source=None)  # REST-seed отключён (офлайн)
            rc = bot.run_live(duration_s=4)
            server.shutdown()
        sig = bot.engine.last_signal
        ok = (rc == 0
              and bot.processor.conn_state == ConnState.CONNECTED
              and len(bot.processor.confirmed) >= 1
              and bot.processor.last_price is not None
              and sig is not None and sig.data_fresh
              and sig.action in ("BUY", "SELL", "HOLD")
              and sig.model_version == pred.model_version)
        detail = (f"rc={rc}, conn={bot.processor.conn_state.value}, "
                  f"confirmed={len(bot.processor.confirmed)}, "
                  f"price={bot.processor.last_price.price if bot.processor.last_price else None}, "
                  f"signal={None if sig is None else (sig.action, sig.confidence, sig.data_fresh)}")
    except Exception as e:  # noqa: BLE001
        detail = f"исключение: {type(e).__name__}: {e}"
    record("A10 LIVE-тракт на локальном fake-WS (офлайн): клиент->процессор->движок->fresh-сигнал",
           ok, detail)


def check_a9_no_trading_calls() -> None:
    # структурный скан: никаких торговых/приватных эндпоинтов и ключей в коде бота
    p1, p2, p3 = "/v5/" + "order", "/v5/" + "position", "/v5/" + "account"
    p4, p5, p6 = "v5/" + "private", "X-" + "BAPI", "api_" + "key"
    p7, p8 = "api_" + "secret", "place_" + "order"
    forbidden = [p1, p2, p3, p4, p5, p6, p7, p8, "/v5/" + "asset", "/v5/" + "user"]
    roots = [PROJECT_ROOT / "btc_bot", PROJECT_ROOT / "scripts"]
    hits = []
    for root in roots:
        for py in sorted(root.rglob("*.py")):
            if "manual_checks" in py.parts:
                continue  # этот чек-скрипт содержит паттерны в виде конкатенаций
            text = py.read_text(encoding="utf-8")
            for pat in forbidden:
                if pat in text:
                    hits.append(f"{py.relative_to(PROJECT_ROOT)}:{pat}")
    record("A9 в коде бота нет торговых/приватных эндпоинтов и ключей (структурный скан)",
           not hits, f"найдено: {hits}" if hits else f"просканировано {len(list((PROJECT_ROOT / 'btc_bot').rglob('*.py')))} файлов пакета")


# --------------------------------------------------------------- часть B (live)
def check_b_live(wait_confirm: bool) -> None:
    """Read-only проверка реального WS Bybit через ТОТ ЖЕ тракт, что и бот
    (BybitWsClient -> StreamProcessor), на реальных сообщениях биржи."""
    from btc_bot.streaming.bybit_ws import BybitWsClient
    cfg = load_config()
    q: "queue.Queue" = queue.Queue(maxsize=5000)
    client = BybitWsClient(cfg.exchange.ws_public_url, cfg.exchange.symbol,
                           cfg.exchange.interval, q, recv_timeout_s=1.0)
    proc = StreamProcessor(cfg.exchange.symbol, cfg.exchange.interval,
                           stale_after_intervals=cfg.signals.stale_after_intervals)
    got = {"connected": False, "ticker": False, "kline": False, "confirm": False, "price": None}
    deadline = time.monotonic() + (cfg.exchange.interval_minutes * 60 + 120 if wait_confirm else 45)
    client.start()
    try:
        while time.monotonic() < deadline:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            kind = item[0]
            if kind == "msg":
                for ev in proc.on_ws_message(item[1]):
                    if isinstance(ev, StateChange) and ev.state == ConnState.CONNECTED:
                        got["connected"] = True
                    elif isinstance(ev, TickerUpdate):
                        got["ticker"] = True
                        got["price"] = ev.price
                    elif isinstance(ev, KlineBar):
                        got["kline"] = True
                        if ev.confirmed:
                            got["confirm"] = True
                if proc.live_candle is not None:
                    got["kline"] = True  # формирующаяся свеча тоже доказывает kline-поток
            elif kind == "state":
                proc.on_connection_state(item[1], item[2])
            if got["connected"] and got["ticker"] and got["kline"] and (got["confirm"] or not wait_confirm):
                break
    finally:
        client.stop()
    need = ["connected", "ticker", "kline"] + (["confirm"] if wait_confirm else [])
    missing = [k for k in need if not got[k]]
    record("B1 LIVE WS Bybit (read-only): subscribe ack + ticker + kline"
           + (" + закрытие свечи (confirm=true)" if wait_confirm else ""),
           not missing,
           f"missing={missing}" if missing else
           f"цена={got['price']}, confirm={got['confirm']}; торговые вызовы не использовались")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="добавить read-only проверку реального WS")
    ap.add_argument("--wait-confirm", action="store_true", help="с --live: ждать закрытия реальной свечи (до ~7 мин)")
    args = ap.parse_args()

    print("== Этап 3: поток данных, сигналы, соединение, свежесть (офлайн-сценарии) ==")
    with tempfile.TemporaryDirectory() as td_str:
        tmp = Path(td_str)
        check_a1_full_pipeline(tmp)
        check_a2_gap_incomplete(tmp)
        check_a3_stale(tmp)
        check_a4_disconnect_reconnect(tmp)
        check_a5_hold_band(tmp)
    check_a6_backoff()
    check_a7_backfill_dedup_order()
    with tempfile.TemporaryDirectory() as td_str:
        tmp = Path(td_str)
        check_a8_ui_text(tmp)
    check_a9_no_trading_calls()
    with tempfile.TemporaryDirectory() as td_str:
        tmp = Path(td_str)
        check_a10_live_pipeline_local(tmp)

    if args.live:
        print("-- Часть B: LIVE read-only проверка публичного WS Bybit --")
        check_b_live(args.wait_confirm)
    else:
        print("[SKIP] B1 LIVE WS :: не запрашивалась (запустите с --live; по умолчанию проверка офлайн)")

    failed = [name for name, ok, _ in _results if not ok]
    print()
    if failed:
        print(f"RESULT: FAIL ({len(failed)} проваленных: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (проверок пройдено: {len(_results)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
