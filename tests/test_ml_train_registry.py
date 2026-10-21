"""End-to-end тесты обучения, реестра моделей и совместимости (критерии приёмки фазы 3).

Данные — синтетические (data/synthetic.py), обучение — маленькими параметрами для
скорости; все прогоны детерминированы (seed, n_jobs=1, deterministic=True).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tradebot.config import AppConfig
from tradebot.data.synthetic import generate_synthetic_bars
from tradebot.ml.errors import InsufficientHistoryError, ModelCompatibilityError
from tradebot.ml.registry import load_model_bundle
from tradebot.ml.train import run_training

BASE_CONFIG = {
    "run": {"mode": "TEST", "experiment_id": "test-ml"},
    "data": {
        "csv_dir": "data/raw",
        "symbols": [{"ticker": "SYNTH", "mic": "DEMO", "files": {}}],
        "timestamp_semantics": "bar_open",
        "source_timezone": "Europe/Moscow",
    },
    "timeframes": {
        "5m": {
            "enabled": True,
            "lookback_bars": 240,
            "horizon_bars": 4,
            "label": {"kind": "fwd_return_class", "neutral_band_bps": 5.0},
        },
        "15m": {"enabled": True, "lookback_bars": 240, "horizon_bars": 2},
    },
    "model": {
        "kind": "lightgbm",
        "seed": 42,
        "params": {
            "n_estimators": 25,
            "num_leaves": 7,
            "min_child_samples": 20,
            "learning_rate": 0.1,
        },
    },
    "validation": {
        "walk_forward": {"n_splits": 3, "min_train_bars": 300, "min_val_bars": 60},
        "thresholds": {"min_oos_auc": 0.5, "min_oos_trades": 1},
    },
}


def make_config(**section_overrides) -> AppConfig:
    """Копия BASE_CONFIG с заменой ЦЕЛЫХ секций верхнего уровня."""
    import copy

    raw = copy.deepcopy(BASE_CONFIG)
    for section, value in section_overrides.items():
        raw[section] = value
    return AppConfig.model_validate(raw)


@pytest.fixture(scope="module")
def bars():
    return generate_synthetic_bars(1600, seed=17, timeframe="5m")


@pytest.fixture(scope="module")
def run_result(bars, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    config = make_config()
    return run_training(
        config,
        "5m",
        bars=bars,
        model_dir=out / "models",
        report_dir=out / "reports",
    )


class TestRunTraining:
    def test_folds_and_oos_present(self, run_result):
        assert len(run_result.folds) == 3
        total_val = sum(f.n_val for f in run_result.folds)
        assert run_result.oos_metrics["n"] == total_val
        for fold in run_result.folds:
            assert fold.model_metrics["n"] == fold.n_val
            assert fold.embargo_bars >= run_result.horizon_bars

    def test_baselines_computed_same_folds(self, run_result):
        assert set(run_result.baselines_oos_metrics) >= {"majority", "momentum", "logreg"}
        for name in ("majority", "momentum", "logreg"):
            assert run_result.baselines_oos_metrics[name]["n"] == run_result.oos_metrics["n"]

    def test_verdict_structure(self, run_result):
        verdict = run_result.verdict
        assert verdict["text"]
        assert "прибыльн" not in verdict["text"].lower() or "НЕ является" in verdict["text"]
        checks = {c["check"]: c["passed"] for c in verdict["checks"]}
        assert "min_oos_auc" in checks and "min_oos_trades" in checks

    def test_audit_present(self, run_result):
        audit = run_result.audit
        assert audit and audit.get("applicable") is True
        assert audit["auc_random_split"] is not None
        assert audit["delta"] is not None

    def test_assumptions_and_annualization(self, run_result):
        a = run_result.assumptions
        assert a["commission_bps"] == 3.0   # дефолт EvaluationConfig
        assert a["slippage_bps"] == 2.0
        assert a["bars_per_year"] > 0
        assert run_result.oos_trading_annualized["bars_per_year_source"] in ("derived", "config")

    def test_report_written_with_required_sections(self, run_result):
        text = Path(run_result.report_path).read_text(encoding="utf-8")
        for fragment in (
            "Дисклеймер",
            "Walk-forward разбиение",
            "Допущения симуляции и годового пересчёта",
            "комиссия",
            "слиппедж",
            "bars_per_year",
            "Аудит утечек",
            "Вердикт",
            "обещанием",
            "majority",
        ):
            assert fragment.lower() in text.lower(), f"в отчёте нет фрагмента {fragment!r}"

    def test_insufficient_history(self, tmp_path):
        config = make_config()
        small = generate_synthetic_bars(400, seed=18, timeframe="5m")
        with pytest.raises(InsufficientHistoryError):
            run_training(config, "5m", bars=small, model_dir=tmp_path / "m", report_dir=tmp_path / "r")

    def test_catboost_not_wired_raises_explicit(self, bars, tmp_path):
        import copy

        raw = copy.deepcopy(BASE_CONFIG)
        raw["model"]["kind"] = "catboost"
        config = AppConfig.model_validate(raw)
        with pytest.raises(NotImplementedError, match="D4"):
            run_training(config, "5m", bars=bars[:500], model_dir=tmp_path / "m", report_dir=tmp_path / "r")

    def test_regression_kind_runs(self, bars, tmp_path):
        import copy

        raw = copy.deepcopy(BASE_CONFIG)
        raw["timeframes"]["5m"]["label"] = {"kind": "fwd_return_reg", "neutral_band_bps": 0.0}
        raw["validation"]["thresholds"] = {}
        config = AppConfig.model_validate(raw)
        result = run_training(config, "5m", bars=bars, model_dir=tmp_path / "m", report_dir=tmp_path / "r")
        assert "mae" in result.oos_metrics and "r2" in result.oos_metrics


class TestDeterminism:
    def test_same_seed_same_results(self, bars, tmp_path):
        config = make_config()
        r1 = run_training(config, "5m", bars=bars, model_dir=tmp_path / "m1", report_dir=tmp_path / "r1")
        r2 = run_training(config, "5m", bars=bars, model_dir=tmp_path / "m2", report_dir=tmp_path / "r2")
        assert r1.oos_metrics == r2.oos_metrics
        p1 = np.concatenate([f.val_proba for f in r1.folds])
        p2 = np.concatenate([f.val_proba for f in r2.folds])
        assert np.array_equal(p1, p2)


class TestRegistry:
    def test_artifact_files_and_metadata(self, run_result):
        model_dir = Path(run_result.model_dir)
        assert (model_dir / "model.lgb.txt").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "feature_manifest.json").exists()
        assert (model_dir / "oos_predictions.csv").exists()
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["timeframe"] == "5m"
        assert metadata["horizon_bars"] == 4
        assert metadata["lookback_bars"] == 240
        assert metadata["label"]["kind"] == "fwd_return_class"
        assert metadata["model"]["seed"] == 42
        assert metadata["training_period"]["first_ts"] < metadata["training_period"]["last_ts"]
        assert metadata["versions"]["lightgbm"] not in ("", "not-installed")
        assert metadata["versions"]["python"]
        assert len(metadata["walk_forward"]["folds"]) == 3
        assert "oos_metrics" in metadata["results"]
        manifest = json.loads((model_dir / "feature_manifest.json").read_text(encoding="utf-8"))
        assert manifest["version"] == "v1"
        assert len(manifest["feature_names"]) == 31

    def test_load_and_predict(self, run_result, bars):
        bundle = load_model_bundle(run_result.model_dir)
        assert bundle.timeframe == "5m"
        bundle.validate_compatibility(
            timeframe="5m", feature_names=bundle.feature_names, horizon_bars=4, label_kind="fwd_return_class"
        )
        proba = bundle.predict_proba(bars[100:160], local_timezone="Europe/Moscow")
        assert proba.shape == (60,)
        assert np.isfinite(proba).all()
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_rejects_incompatible_timeframe(self, run_result):
        bundle = load_model_bundle(run_result.model_dir)
        with pytest.raises(ModelCompatibilityError) as exc:
            bundle.validate_compatibility(timeframe="15m")
        assert "таймфрейм" in str(exc.value)

    def test_rejects_incompatible_horizon(self, run_result):
        bundle = load_model_bundle(run_result.model_dir)
        with pytest.raises(ModelCompatibilityError) as exc:
            bundle.validate_compatibility(horizon_bars=99)
        assert "горизонт" in str(exc.value)

    def test_rejects_feature_mismatch(self, run_result):
        bundle = load_model_bundle(run_result.model_dir)
        wrong = list(bundle.feature_names)
        wrong[0] = "unknown_feature"
        with pytest.raises(ModelCompatibilityError) as exc:
            bundle.validate_compatibility(feature_names=wrong)
        assert "признак" in str(exc.value)
        # порядок колонок тоже важен
        reordered = list(reversed(list(bundle.feature_names)))
        with pytest.raises(ModelCompatibilityError):
            bundle.validate_compatibility(feature_names=reordered)

    def test_missing_files_detected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_bundle(tmp_path)
