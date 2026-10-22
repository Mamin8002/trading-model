"""Офлайн-тест полного LIVE-тракта на локальном fake-WS-сервере (без внешней сети)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from websockets.sync.server import serve  # noqa: E402

from btc_bot.bot import MonitorBot  # noqa: E402
from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import build_dataset  # noqa: E402
from btc_bot.model.features import build_features  # noqa: E402
from btc_bot.model.predictor import Predictor  # noqa: E402
from btc_bot.model.train import (  # noqa: E402
    build_card, library_versions, make_model_version, save_artifact, train_final,
)
from btc_bot.streaming.base import ConnState  # noqa: E402
from btc_bot.ui.console import ConsoleUI  # noqa: E402

STEP = 300_000


@pytest.fixture(scope="module")
def predictor():
    """Маленькая детерминированная модель на синтетике (явно НЕ реальные данные)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(21)
    rows, price = [], 60_000.0
    base = 1_800_000_000_000
    for i in range(500):
        o = price
        c = o * (1 + rng.normal(0, 0.002))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0008)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0008)))
        v = float(rng.uniform(1, 40))
        rows.append({"start_ms": base + i * STEP, "open": o, "high": h, "low": l,
                     "close": c, "volume": v, "turnover": v * c, "is_closed": True})
        price = c
    df = pd.DataFrame(rows)
    X, y, _, _ = build_dataset(df, 1, "5", features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=25)
    card = build_card(model_version=make_model_version("BTCUSDT", "spot", "5", 1, "ratio"),
                      algorithm="xgboost", symbol="BTCUSDT", category="spot", interval="5",
                      horizon_bars=1, horizon_minutes=5, feature_set="ratio",
                      features=["open_rel", "high_rel", "low_rel", "volume", "turnover"],
                      confidence_threshold=0.55, library_versions=library_versions())
    with tempfile.TemporaryDirectory() as td:
        save_artifact(model, td, card)
        yield Predictor.load(td, 0.55)


def test_run_live_on_local_fake_server(predictor):  # noqa: F811
    now_ms = int(time.time() * 1000)
    bar_start = (now_ms // STEP - 1) * STEP
    bar = {"start": bar_start, "end": bar_start + STEP - 1, "interval": "5",
           "open": "84500", "close": "84600", "high": "84650", "low": "84450",
           "volume": "25.5", "turnover": "2157000", "confirm": True, "timestamp": now_ms}

    def handler(ws):
        try:
            req = json.loads(ws.recv(timeout=10))
            assert req.get("op") == "subscribe"
            ws.send(json.dumps({"success": True, "ret_msg": "subscribe",
                                "op": "subscribe", "conn_id": "fake"}))
            # spot ticker: data - ОБЪЕКТ (не массив); kline: data - массив
            ws.send(json.dumps({"topic": "tickers.BTCUSDT", "ts": now_ms, "type": "snapshot",
                                "data": {"symbol": "BTCUSDT", "lastPrice": "84600.5"}}))
            ws.send(json.dumps({"topic": "kline.5.BTCUSDT", "ts": now_ms - 1000,
                                "type": "snapshot",
                                "data": [dict(bar, confirm=False, close="84590")]}))
            ws.send(json.dumps({"topic": "kline.5.BTCUSDT", "ts": now_ms,
                                "type": "snapshot", "data": [bar]}))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv(timeout=2)
                except TimeoutError:
                    continue
                if raw is None:
                    continue
                if json.loads(raw).get("op") == "ping":
                    ws.send(json.dumps({"success": True, "ret_msg": "pong", "op": "ping"}))
        except Exception:  # noqa: BLE001
            pass

    with serve(handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        cfg = load_config(overrides={
            "mode": "LIVE",
            "model": {"horizon_bars": 1, "confidence_threshold": 0.55},
            "exchange": {"ws_public_url": f"ws://127.0.0.1:{port}/v5/public/spot"},
        })
        ui = ConsoleUI(stream=io.StringIO(), force_line_mode=True)
        bot = MonitorBot(cfg, predictor, ui, rest_source=None)
        rc = bot.run_live(duration_s=3)
        server.shutdown()

    assert rc == 0
    assert bot.processor.conn_state == ConnState.CONNECTED
    assert len(bot.processor.confirmed) >= 1
    assert bot.processor.last_price is not None
    assert abs(bot.processor.last_price.price - 84600.5) < 1e-9
    sig = bot.engine.last_signal
    assert sig is not None and sig.data_fresh
    assert sig.action in ("BUY", "SELL", "HOLD")
    assert sig.model_version == predictor.model_version
    # seed-история не должна порождать сигналы (rest_source=None, но проверим счётчик)
    assert len(bot.engine.signals) == len(bot.processor.confirmed)
