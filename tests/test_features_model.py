"""Офлайн unit-тесты модельного слоя. Запуск: pytest tests/ -q"""

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
from btc_bot.model.features import RATIO_FEATURES, RAW_FEATURES, bar_features, build_features  # noqa: E402
from btc_bot.model.predictor import ModelMismatchError, Predictor  # noqa: E402
from btc_bot.model.train import build_card, library_versions, make_model_version, save_artifact, train_final  # noqa: E402
from btc_bot.model.walk_forward import label_resolution_times, walk_forward_validate  # noqa: E402

BASE = 1_700_000_100_000
STEP = 300_000


def synthetic(n: int, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, price = [], 40_000.0
    for i in range(n):
        o = price
        c = o * (1 + rng.normal(0, 0.002))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.001)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.001)))
        v = float(rng.uniform(1, 30))
        rows.append({"start_ms": BASE + i * STEP, "open": o, "high": h, "low": l,
                     "close": c, "volume": v, "turnover": v * c, "is_closed": True})
        price = c
    return pd.DataFrame(rows)


def test_feature_sets_columns_and_values():
    df = synthetic(5)
    raw = build_features(df, "raw")
    ratio = build_features(df, "ratio")
    assert list(raw.columns) == RAW_FEATURES
    assert list(ratio.columns) == RATIO_FEATURES
    row = df.iloc[2]
    assert np.isclose(ratio["high_rel"].iloc[2], row.high / row.close)
    feats = bar_features(row.to_dict(), "ratio")
    assert np.isclose(feats["low_rel"], row.low / row.close)
    with pytest.raises(ValueError):
        bar_features(row.to_dict(), "unknown")


def test_label_resolution_and_embargo_rule():
    times = np.array([BASE, BASE + STEP, BASE + 2 * STEP])
    res = label_resolution_times(times, "5", horizon_bars=1)
    # метка бара t разрешается в конце следующего бара: t + 2*step
    assert list(res) == [BASE + 2 * STEP, BASE + 3 * STEP, BASE + 4 * STEP]


def test_walk_forward_structure_no_overlap():
    df = synthetic(500, seed=5)
    X, y, times, _ = build_dataset(df, 1, "5", features=build_features(df, "ratio"))
    wf = walk_forward_validate(X, y, times, df[["start_ms", "close"]], "5", 1,
                               feature_set="ratio", n_folds=3, train_fraction=0.5,
                               threshold=0.55, seed=42, n_estimators=15)
    assert len(wf.folds) == 3
    prev_end = None
    for f in wf.folds:
        assert f.train_n > 0 and f.test_n > 0
        assert 0.0 <= f.accuracy <= 1.0
        res = label_resolution_times(np.array([f.train_end_ms]), "5", 1)[0]
        assert res <= f.test_start_ms  # embargo
        if prev_end is not None:
            assert f.test_start_ms > prev_end
        prev_end = f.test_end_ms
    assert wf.aggregate["n_test_total"] == sum(f.test_n for f in wf.folds)


def test_artifact_roundtrip_and_mismatch_guard():
    df = synthetic(300, seed=9)
    X, y, _, _ = build_dataset(df, 1, "5", features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=15)
    card = build_card(model_version=make_model_version("BTCUSDT", "spot", "5", 1, "ratio"),
                      algorithm="xgboost", symbol="BTCUSDT", category="spot", interval="5",
                      horizon_bars=1, horizon_minutes=5, feature_set="ratio",
                      features=RATIO_FEATURES, confidence_threshold=0.55,
                      library_versions=library_versions())
    with tempfile.TemporaryDirectory() as td:
        save_artifact(model, td, card)
        pred = Predictor.load(td, 0.55)
        pred.ensure_compatible(symbol="BTCUSDT", category="spot", interval="5", horizon_bars=1)
        with pytest.raises(ModelMismatchError):
            pred.ensure_compatible(symbol="ETHUSDT", category="spot", interval="5", horizon_bars=1)
        p = pred.predict_bar(df.iloc[-1].to_dict())
        assert p.action in ("BUY", "SELL", "HOLD")
        assert 0.5 <= p.confidence <= 1.0
        assert p.model_version == card["model_version"]


def test_action_mapping_hold_band():
    df = synthetic(200, seed=15)
    X, y, _, _ = build_dataset(df, 1, "5", features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=10)
    card = build_card(model_version="t", algorithm="xgboost", symbol="BTCUSDT", category="spot",
                      interval="5", horizon_bars=1, horizon_minutes=5, feature_set="ratio",
                      features=RATIO_FEATURES, confidence_threshold=0.6,
                      library_versions=library_versions())
    with tempfile.TemporaryDirectory() as td:
        save_artifact(model, td, card)
        pred = Predictor.load(td, 0.6)
        assert pred.action_from_prob_up(0.55) == ("HOLD", 1, pytest.approx(0.55))
        assert pred.action_from_prob_up(0.80)[:2] == ("BUY", 1)
        assert pred.action_from_prob_up(0.10)[:2] == ("SELL", -1)
