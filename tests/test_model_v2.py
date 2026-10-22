"""Офлайн unit-тесты модели v2, торгового симулятора и adopt-gate (Этап 5a)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_bot.data.prep import build_dataset  # noqa: E402
from btc_bot.model.features import (  # noqa: E402
    V2_FEATURES, InsufficientHistoryError, build_features, feature_columns,
    history_features, v2_min_bars,
)
from btc_bot.model.trading_sim import (  # noqa: E402
    DEFAULT_THR_GRID, ExitRulesSim, TradingCosts, decide_adoption,
    select_threshold, simulate_trades,
)
from btc_bot.model.train import build_card, library_versions, make_model_version, save_artifact, train_final, find_latest_model_dir  # noqa: E402
from btc_bot.model.predictor import ModelMismatchError, Predictor  # noqa: E402
from btc_bot.model.walk_forward import walk_forward_validate  # noqa: E402

BASE = 1_800_000_000_000
STEP = 300_000
H = 3


def synthetic(n: int, seed: int = 51) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, price = [], 60_000.0
    for i in range(n):
        o = price
        c = o * (1 + rng.normal(0, 0.002))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0008)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0008)))
        v = float(rng.uniform(5, 50))
        rows.append({"start_ms": BASE + i * STEP, "open": o, "high": h, "low": l,
                     "close": c, "volume": v, "turnover": v * c, "is_closed": True})
        price = c
    return pd.DataFrame(rows)


def test_v2_features_tail_invariant():
    df = synthetic(120)
    full = build_features(df, "v2", horizon_bars=H)
    k = 100
    cut = build_features(df.iloc[:k].reset_index(drop=True), "v2", horizon_bars=H)
    assert np.allclose(full.iloc[k - 5:k].to_numpy(dtype="float64"),
                       cut.iloc[k - 5:k].to_numpy(dtype="float64"), atol=1e-12, equal_nan=True)
    assert int(full.apply(lambda c: c.first_valid_index()).max()) == 20


def test_dataset_h3_warmup_and_target():
    n = 60
    df = synthetic(n, seed=53)
    X, y, times, rep = build_dataset(df, H, "5", features=build_features(df, "v2", horizon_bars=H))
    assert rep.n_dropped_warmup == 20
    assert rep.n_dropped_no_future_bar == H
    assert rep.n_rows == n - 20 - H
    assert not X.isna().any().any()
    assert list(X.columns) == V2_FEATURES
    close_by_time = dict(zip(df["start_ms"], df["close"]))
    for idx in (5, 20, 30):
        t = int(times.iloc[idx])
        fut = close_by_time.get(t + H * STEP)
        expected = 1 if (fut is not None and fut > close_by_time[t]) else -1
        assert int(y.iloc[idx]) == expected


def test_history_features_matches_df_and_guards():
    df = synthetic(80, seed=57)
    feats = build_features(df, "v2", horizon_bars=H)
    bars = [{"start_ms": int(r.start_ms), "open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": r.volume, "turnover": r.turnover} for r in df.itertuples()]
    live = history_features(bars, "v2", horizon_bars=H)
    row = feats.iloc[-1]
    for c in V2_FEATURES:
        assert abs(live[c] - float(row[c])) < 1e-12
    with pytest.raises(InsufficientHistoryError):
        history_features(bars[-(v2_min_bars(H) - 1):], "v2", horizon_bars=H)


def test_simulator_math_time_exit_and_sl_priority():
    costs = TradingCosts(fee_pct=0.1, slippage_pct=0.02, order_size_quote=100.0)
    ohlc = pd.DataFrame({"start_ms": [BASE, BASE + STEP], "open": [100.0, 101.0],
                         "high": [100.5, 102.5], "low": [99.5, 100.8], "close": [100.0, 102.0]})
    r = simulate_trades(ohlc, np.array([1.0, 1.0]), 0.55, costs, ExitRulesSim(), 1, "5")
    entry_eff = 100.0 * 1.0002
    exit_eff = 102.0 * 0.9998
    assert r["n_trades"] == 1 and r["exit_reasons"] == {"horizon": 1}
    assert abs(r["avg_net_pct"] / 100.0 - ((exit_eff / entry_eff - 1.0) - 0.002)) < 1e-12

    ohlc2 = pd.DataFrame({"start_ms": [BASE, BASE + STEP, BASE + 2 * STEP],
                          "open": [100.0, 98.0, 98.5], "high": [100.5, 101.5, 99.0],
                          "low": [99.5, 97.5, 98.0], "close": [100.0, 98.5, 98.5]})
    rules = ExitRulesSim(take_profit_pct=1.0, stop_loss_pct=1.0)
    r2 = simulate_trades(ohlc2, np.array([1.0, 1.0, 1.0]), 0.55, costs, rules, 2, "5")
    assert r2["n_trades"] == 1 and r2["exit_reasons"] == {"stop_loss": 1}
    exit_eff2 = min(98.0, 99.0) * 0.9998
    assert abs(r2["avg_net_pct"] / 100.0 - ((exit_eff2 / entry_eff - 1.0) - 0.002)) < 1e-12


def test_simulator_skips_entries_near_window_end():
    costs = TradingCosts(0.1, 0.02, 100.0)
    ohlc = pd.DataFrame({"start_ms": [BASE, BASE + STEP, BASE + 2 * STEP],
                         "open": [100.0] * 3, "high": [101.0] * 3,
                         "low": [99.0] * 3, "close": [100.0, 100.5, 101.0]})
    r = simulate_trades(ohlc, np.array([1.0, 1.0, 1.0]), 0.55, costs, ExitRulesSim(), 1, "5")
    # бар0: вход -> выход на баре1 (горизонт); бар1 пропуск (бар выхода), бар2: вход за окном -> skip
    assert r["n_trades"] == 1 and r["skipped_entries"] == 1


def test_decide_adoption_gate():
    assert decide_adoption(0.010, 0.000, -0.010) is True
    assert decide_adoption(0.000, 0.010, -0.010) is False
    assert decide_adoption(0.020, 0.010, 0.030) is False   # хуже baseline
    assert decide_adoption(0.010, 0.010, -0.010) is False  # равенство
    assert decide_adoption(float("nan"), 0.0, -1.0) is False


def test_select_threshold_in_grid_and_deterministic():
    rng = np.random.default_rng(7)
    ohlc = synthetic(400, seed=59)[["start_ms", "open", "high", "low", "close"]]
    prob = rng.uniform(0.35, 0.65, size=400)
    costs = TradingCosts(0.1, 0.02, 100.0)
    t1, _ = select_threshold(ohlc, prob, costs, ExitRulesSim(), 1, "5", min_trades=3)
    t2, _ = select_threshold(ohlc, prob, costs, ExitRulesSim(), 1, "5", min_trades=3)
    assert t1 == t2 and t1 in DEFAULT_THR_GRID


def test_walk_forward_selected_mode_with_trades():
    df = synthetic(600, seed=61)
    X, y, times, _ = build_dataset(df, 1, "5", features=build_features(df, "ratio"))
    ohlc = df[["start_ms", "open", "high", "low", "close"]]
    costs = TradingCosts(0.1, 0.02, 100.0)
    wf = walk_forward_validate(X, y, times, df[["start_ms", "close"]], "5", 1,
                               feature_set="ratio", n_folds=3, train_fraction=0.5,
                               seed=42, n_estimators=20, ohlc=ohlc, trading=costs,
                               threshold_mode="selected", min_trades_for_thr=3)
    assert len(wf.folds) == 3
    for f in wf.folds:
        assert f.effective_threshold in DEFAULT_THR_GRID
        assert f.trades is not None and f.majority_trades is not None
        assert f.majority_trades["n_trades"] > 0  # baseline входит на каждом баре
    assert wf.aggregate["trades"] is not None


def test_predictor_v2_history_and_guards():
    df = synthetic(400, seed=67)
    X, y, _, _ = build_dataset(df, H, "5", features=build_features(df, "v2", horizon_bars=H))
    model = train_final(X, y, seed=42, n_estimators=15)
    card = build_card(model_version=make_model_version("BTCUSDT", "spot", "5", H, "v2"),
                      algorithm="xgboost", symbol="BTCUSDT", category="spot", interval="5",
                      horizon_bars=H, horizon_minutes=H * 5, feature_set="v2",
                      features=feature_columns("v2", H), confidence_threshold=0.6,
                      adopted=True, library_versions=library_versions())
    bars = [{"start_ms": int(r.start_ms), "open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": r.volume, "turnover": r.turnover} for r in df.itertuples()]
    with tempfile.TemporaryDirectory() as td:
        save_artifact(model, td, card)
        pred = Predictor.load(td)  # threshold None -> из card
        assert abs(pred.confidence_threshold - 0.6) < 1e-12
        assert pred.horizon_minutes == 15
        p = pred.predict_history(bars[-30:])
        assert p.action in ("BUY", "SELL", "HOLD") and p.feature_set == "v2"
        with pytest.raises(ModelMismatchError):
            pred.predict_bar(bars[-1])
        with pytest.raises(ModelMismatchError):
            pred.ensure_compatible(symbol="BTCUSDT", category="spot", interval="5", horizon_bars=1)


def test_find_latest_respects_adopted_flag():
    df = synthetic(150, seed=71)
    X, y, _, _ = build_dataset(df, 1, "5", features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=10)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "models"
        card_old = build_card(model_version="m-old", algorithm="xgboost", symbol="BTCUSDT",
                              category="spot", interval="5", horizon_bars=1, horizon_minutes=5,
                              feature_set="ratio", features=feature_columns("ratio"),
                              confidence_threshold=0.55, adopted=True,
                              library_versions=library_versions())
        card_old["created_at_utc"] = "2026-01-01T00:00:00Z"
        save_artifact(model, root / "m-old", card_old)
        card_new = dict(card_old)
        card_new.update({"model_version": "m-new", "adopted": False,
                         "created_at_utc": "2026-06-01T00:00:00Z"})
        save_artifact(model, root / "m-new", card_new)
        assert find_latest_model_dir(root).name == "m-old"
