"""Офлайн unit-тесты модулей данных (prep, store). Запуск: pytest tests/ -q"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_bot.data.prep import build_dataset, clean_klines  # noqa: E402
from btc_bot.data.store import load_klines_csv, save_klines_csv  # noqa: E402
from btc_bot.data_sources.base import Kline  # noqa: E402

BASE = 1_700_000_100_000  # выровнено по 5 минутам
STEP = 300_000


def mk(i, c, o=None, closed=True):
    o = c if o is None else o
    return {
        "start_ms": BASE + i * STEP, "open": o, "high": max(o, c) + 0.5,
        "low": min(o, c) - 0.5, "close": c, "volume": 1.0, "turnover": 100.0,
        "is_closed": closed,
    }


def test_clean_dedup_sort_and_unconfirmed():
    df = pd.DataFrame([mk(2, 10.0), mk(0, 9.0), mk(2, 10.0), mk(1, 9.5), mk(3, 10.0, closed=False)])
    out, rep = clean_klines(df, "5")
    assert rep.n_duplicates_removed == 1
    assert rep.n_unconfirmed_excluded == 1
    assert list(out["start_ms"]) == [BASE, BASE + STEP, BASE + 2 * STEP]


def test_clean_gaps_detected_not_filled():
    df = pd.DataFrame([mk(0, 10.0), mk(1, 10.0), mk(4, 10.0)])
    out, rep = clean_klines(df, "5")
    assert len(out) == 3  # ничего не добавлено
    assert rep.total_missing_bars == 2
    assert rep.gaps[0].missing_bars == 2


def test_clean_invalid_rows_removed():
    bad = mk(1, 10.0)
    bad["high"] = 1.0  # high < low
    df = pd.DataFrame([mk(0, 10.0), bad])
    out, rep = clean_klines(df, "5")
    assert rep.n_invalid_removed == 1
    assert len(out) == 1


def test_build_dataset_horizon_and_tie():
    closes = [10.0, 11.0, 10.5, 10.5, 12.0]
    df, rep = clean_klines(pd.DataFrame([mk(i, c) for i, c in enumerate(closes)]), "5")
    assert rep.n_output == 5
    X, y, times, drep = build_dataset(df, horizon_bars=1, interval="5")
    assert list(y) == [1, -1, -1, 1]  # ничья 10.5==10.5 -> -1 (как в оригинале)
    assert drep.n_dropped_no_future_bar == 1
    X2, y2, _, drep2 = build_dataset(df, horizon_bars=2, interval="5")
    assert list(y2) == [1, -1, 1]
    assert drep2.n_dropped_no_future_bar == 2


def test_build_dataset_missing_future_bar_dropped():
    rows = [mk(0, 10.0), mk(1, 11.0), mk(2, 10.5), mk(4, 12.0)]  # бар 3 пропущен
    df, _ = clean_klines(pd.DataFrame(rows), "5")
    X, y, times, drep = build_dataset(df, horizon_bars=1, interval="5")
    assert list(y) == [1, -1]          # только t0->t1 и t1->t2
    assert drep.n_dropped_no_future_bar == 2  # t2 (нет t3) и t4 (нет t5)


def test_store_roundtrip_and_metadata():
    kl = [Kline(BASE + i * STEP, 10 + i, 11 + i, 9 + i, 10.5 + i, 1.0, 100.0, i < 2)
          for i in range(3)]
    with tempfile.TemporaryDirectory() as td:
        csv_path, meta_path = save_klines_csv(
            kl, Path(td) / "s.csv", source="bybit", category="spot",
            symbol="BTCUSDT", interval="5",
        )
        df = load_klines_csv(csv_path)
        assert len(df) == 3
        assert bool(df["is_closed"].iloc[2]) is False
        import json
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        assert meta["symbol"] == "BTCUSDT" and meta["n_rows"] == 3
        assert "fetched_at_utc" in meta and "period_start_ms" in meta
