"""Офлайн unit-тесты стрим-слоя и сигнального движка. Запуск: pytest tests/ -q"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_bot.bot import MonitorBot  # noqa: E402
from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import build_dataset  # noqa: E402
from btc_bot.data_sources.base import Kline  # noqa: E402
from btc_bot.model.features import build_features  # noqa: E402
from btc_bot.model.predictor import Predictor  # noqa: E402
from btc_bot.model.train import build_card, library_versions, make_model_version, save_artifact, train_final  # noqa: E402
from btc_bot.streaming.base import ConnState, FreshnessChange, GapDetected, KlineBar  # noqa: E402
from btc_bot.streaming.bybit_ws import backoff_delay_s  # noqa: E402
from btc_bot.streaming.processor import StreamProcessor  # noqa: E402
from btc_bot.ui.console import ConsoleUI, format_snapshot_text  # noqa: E402

INTERVAL = "5"
STEP = 300_000
BASE = 1_800_000_000_000


class FakeClock:
    def __init__(self, ms): self.ms = ms
    def __call__(self): return self.ms


def synthetic(n, seed=21):
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


@pytest.fixture(scope="module")
def predictor():
    df = synthetic(500)
    X, y, _, _ = build_dataset(df, 1, INTERVAL, features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=25)
    card = build_card(model_version=make_model_version("BTCUSDT", "spot", INTERVAL, 1, "ratio"),
                      algorithm="xgboost", symbol="BTCUSDT", category="spot", interval=INTERVAL,
                      horizon_bars=1, horizon_minutes=5, feature_set="ratio",
                      features=["open_rel", "high_rel", "low_rel", "volume", "turnover"],
                      confidence_threshold=0.55, library_versions=library_versions())
    with tempfile.TemporaryDirectory() as td:
        save_artifact(model, td, card)
        yield Predictor.load(td, 0.55)


def make_bot(predictor, clock):
    cfg = load_config(overrides={"mode": "TEST", "model": {"horizon_bars": 1, "confidence_threshold": 0.55}})
    return MonitorBot(cfg, predictor, ConsoleUI(stream=io.StringIO(), force_line_mode=True),
                      clock_ms=clock)


def kmsg(start_ms, row, confirm, ts_ms):
    return {"topic": f"kline.{INTERVAL}.BTCUSDT", "ts": ts_ms,
            "data": [{"start": start_ms, "end": start_ms + STEP - 1, "interval": INTERVAL,
                      "open": str(row["open"]), "close": str(row["close"]),
                      "high": str(row["high"]), "low": str(row["low"]),
                      "volume": str(row["volume"]), "turnover": str(row["turnover"]),
                      "confirm": confirm, "timestamp": ts_ms}]}


def test_signal_only_on_confirm(predictor):
    clock = FakeClock(BASE)
    bot = make_bot(predictor, clock)
    df = synthetic(4)
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    clock.ms = BASE + STEP // 2
    assert bot.dispatch(bot.processor.on_ws_message(kmsg(BASE, df.iloc[0].to_dict(), False, clock.ms))) == []
    assert bot.processor.live_candle is not None
    clock.ms = BASE + STEP
    sigs = bot.dispatch(bot.processor.on_ws_message(kmsg(BASE, df.iloc[0].to_dict(), True, clock.ms)))
    assert len(sigs) == 1
    s = sigs[0]
    assert s.action in ("BUY", "SELL", "HOLD") and 0.5 <= s.confidence <= 1.0
    assert s.model_version == predictor.model_version and s.data_fresh
    assert s.created_at_ms == clock.ms and s.bar_end_ms == BASE + STEP
    # дубликат -> без сигнала
    assert bot.dispatch(bot.processor.on_ws_message(kmsg(BASE, df.iloc[0].to_dict(), True, clock.ms))) == []


def test_stale_and_gap_mark_signal_not_fresh(predictor):
    clock = FakeClock(BASE)
    bot = make_bot(predictor, clock)
    df = synthetic(6)
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    clock.ms = BASE + 21 * 60_000  # тишина 21 мин: возраст бара 16 мин > порога 15 мин
    events = bot.processor.tick(now_ms=clock.ms)
    bot.dispatch(events)
    assert any(isinstance(e, FreshnessChange) and not e.fresh for e in events)
    sigs = bot.dispatch(bot.processor.on_ws_message(kmsg(BASE, df.iloc[0].to_dict(), True, clock.ms)))
    assert sigs and not sigs[0].data_fresh
    assert any("устаревший" in r for r in sigs[0].reasons)

    # gap-сценарий новым ботом
    clock2 = FakeClock(BASE)
    bot2 = make_bot(predictor, clock2)
    bot2.dispatch(bot2.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock2.ms}))
    bot2.dispatch(bot2.processor.tick(now_ms=clock2.ms))
    clock2.ms = BASE + STEP
    bot2.dispatch(bot2.processor.on_ws_message(kmsg(BASE, df.iloc[0].to_dict(), True, clock2.ms)))
    clock2.ms = BASE + 3 * STEP
    evs = bot2.processor.on_ws_message(kmsg(BASE + 2 * STEP, df.iloc[2].to_dict(), True, clock2.ms))
    bot2.dispatch(evs)
    assert any(isinstance(e, GapDetected) for e in evs)
    assert bot2.engine.last_signal is not None and not bot2.engine.last_signal.data_fresh


def test_backoff_schedule():
    assert [backoff_delay_s(i, 1.0, 60.0) for i in range(8)] == [1, 2, 4, 8, 16, 32, 60, 60]


def test_backfill_order_and_dedup():
    proc = StreamProcessor("BTCUSDT", INTERVAL, clock_ms=FakeClock(BASE))
    kl_new = Kline(BASE + 2 * STEP, 10, 11, 9, 10.5, 1.0, 100.0, True)
    kl_old = Kline(BASE, 9, 10, 8, 9.5, 1.0, 100.0, True)
    assert len(proc.ingest_backfilled([kl_new])) == 1
    assert len(proc.ingest_backfilled([kl_old])) == 1
    assert proc.ingest_backfilled([kl_new]) == []
    assert [b.start_ms for b in proc.confirmed] == [BASE, BASE + 2 * STEP]


def test_ui_text_contains_required_fields(predictor):
    clock = FakeClock(BASE)
    bot = make_bot(predictor, clock)
    df = synthetic(4)
    bot.dispatch(bot.processor.on_ws_message({"op": "subscribe", "success": True, "ts": clock.ms}))
    bot.dispatch(bot.processor.tick(now_ms=clock.ms))
    bot.dispatch(bot.processor.on_ws_message(
        {"topic": "tickers.BTCUSDT", "ts": clock.ms, "data": [{"symbol": "BTCUSDT", "lastPrice": "60500.5"}]}))
    clock.ms = BASE + STEP
    bot.dispatch(bot.processor.on_ws_message(kmsg(BASE, df.iloc[0].to_dict(), True, clock.ms)))
    text = format_snapshot_text(bot.build_view())
    for needle in ("CONNECTED", "FRESH", "60,500.50", predictor.model_version,
                   "уверенность", "не гарантия дохода"):
        assert needle in text
