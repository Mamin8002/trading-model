"""Офлайн unit-тесты режимов консоли (line/dash/block). Запуск: pytest tests/ -q"""

from __future__ import annotations

import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_bot.signals.engine import Signal  # noqa: E402
from btc_bot.ui.console import ConsoleUI, format_snapshot_text, format_status_line  # noqa: E402

BASE = 1_800_000_000_000
STEP = 300_000


def make_sig():
    return Signal(signal_id="sig-ui", created_at_ms=BASE + STEP, bar_start_ms=BASE,
                  bar_end_ms=BASE + STEP, symbol="BTCUSDT", category="spot", interval="5",
                  action="BUY", confidence=0.582, prob_up=0.582, raw_direction=1,
                  price_at_signal=84_500.0, price_source="ticker", model_version="xgb-ui-test",
                  horizon_bars=1, horizon_minutes=5, feature_set="ratio", data_fresh=True,
                  reasons=(), origin="ws", features={})


def make_view(price: float) -> dict:
    sig = make_sig()
    return {
        "mode": "LIVE", "exchange": {"category": "spot", "symbol": "BTCUSDT", "interval": "5"},
        "model": {"version": "xgb-ui-test", "horizon_minutes": 5, "threshold": 0.55, "feature_set": "ratio"},
        "conn_state": "CONNECTED", "conn_detail": "ok", "reconnect_count": 0,
        "fresh": True, "age_s": 2.0, "stale_threshold_s": 900.0, "server_ts_ms": BASE,
        "price": price, "price_age_s": 0.5, "live_candle": None, "bars": [],
        "display_bars": 5, "last_signal": sig, "signal_history": [sig],
        "positions": [], "portfolio_stats": None, "journal_path": None,
        "last_advice": None, "warnings": [], "rendered_at_ms": BASE + STEP,
    }


def test_line_mode_updates_in_place():
    s = io.StringIO()
    ui = ConsoleUI(stream=s, mode="line", refresh_s=0.0)
    ui.render(make_view(84_500.0), force=True)
    ui.render(make_view(84_600.0), force=True)
    out = s.getvalue()
    assert "\n" not in out            # строки НЕ добавляются
    assert out.count("\r") >= 2       # обновление на месте
    assert "84,600" in out
    ui.log_event("[тест] событие")
    out2 = s.getvalue()
    assert out2.count("\n") == 1      # событие - ровно одна новая строка
    assert "84,600" in out2           # статусная строка восстановлена после события


def test_dash_mode_uses_alternate_screen():
    s = io.StringIO()
    ui = ConsoleUI(stream=s, mode="dash", refresh_s=0.0)
    ui.begin()
    ui.render(make_view(84_500.0), force=True)
    ui.render(make_view(84_600.0), force=True)
    ui.end()
    out = s.getvalue()
    assert out.startswith("\x1b[?1049h")
    assert out.endswith("\x1b[?25h\x1b[?1049l")
    assert out.count("BTC MONITOR") == 2


def test_block_mode_prints_only_on_change():
    s = io.StringIO()
    ui = ConsoleUI(stream=s, mode="block", refresh_s=0.0)
    v = make_view(84_500.0)
    ui.render(v)
    ui.render(v)
    assert s.getvalue().count("BTC MONITOR") == 1
    ui.render(make_view(84_700.0))
    assert s.getvalue().count("BTC MONITOR") == 2


def test_status_line_content():
    txt = format_status_line(make_view(84_600.0))
    for needle in ("CONNECTED", "84,600", "BUY", "0.582"):
        assert needle in txt
    full = format_snapshot_text(make_view(84_600.0))
    for needle in ("ТЕКУЩИЙ СИГНАЛ", "xgb-ui-test", "не гарантия дохода"):
        assert needle in full
