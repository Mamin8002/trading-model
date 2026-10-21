"""Тесты ML-секций конфигурации (обратная совместимость фазы 2 обязательна)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tradebot.config import ConfigError, load_config

ML_EXAMPLE = Path("configs/ml_example.yaml")


def load_raw(path: Path = ML_EXAMPLE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_cfg(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class TestMlExampleConfig:
    def test_loads(self):
        config = load_config(ML_EXAMPLE)
        assert config.features.set == "v1"
        assert config.model.kind == "lightgbm"
        assert config.model.params.n_estimators == 300
        assert config.validation.walk_forward.n_splits == 5
        assert config.validation.thresholds.min_oos_auc == 0.52
        assert config.evaluation.commission_bps == 3.0

    def test_phase2_configs_still_valid(self):
        # обратная совместимость: старые конфиги фазы 2 не требуют ML-секций
        c1 = load_config("configs/example.yaml")
        c2 = load_config("configs/demo.yaml")
        assert c1.model.seed == 42  # дефолты проставлены
        assert c2.validation.walk_forward.n_splits == 4


class TestStrictness:
    def test_embargo_less_than_horizon_rejected(self, tmp_path):
        raw = load_raw()
        raw["validation"]["walk_forward"]["embargo_bars"] = 2  # horizon 5m = 6
        with pytest.raises(ConfigError, match="embargo"):
            load_config(write_cfg(tmp_path, raw))

    def test_embargo_none_is_auto(self, tmp_path):
        raw = load_raw()
        raw["validation"]["walk_forward"]["embargo_bars"] = None
        config = load_config(write_cfg(tmp_path, raw))
        assert config.validation.walk_forward.embargo_bars is None

    def test_embargo_above_horizon_ok(self, tmp_path):
        raw = load_raw()
        raw["validation"]["walk_forward"]["embargo_bars"] = 12
        load_config(write_cfg(tmp_path, raw))

    def test_macd_fast_must_be_less_than_slow(self, tmp_path):
        raw = load_raw()
        raw["features"]["params"]["macd_fast"] = 30
        with pytest.raises(ConfigError, match="macd_fast"):
            load_config(write_cfg(tmp_path, raw))

    def test_vol_windows_must_be_ascending(self, tmp_path):
        raw = load_raw()
        raw["features"]["params"]["vol_windows"] = [48, 12]
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_unknown_feature_set_rejected(self, tmp_path):
        raw = load_raw()
        raw["features"]["set"] = "v99"
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_n_splits_below_two_rejected(self, tmp_path):
        raw = load_raw()
        raw["validation"]["walk_forward"]["n_splits"] = 1
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_subsample_without_freq_rejected(self, tmp_path):
        raw = load_raw()
        raw["model"]["params"]["subsample"] = 0.8
        with pytest.raises(ConfigError, match="subsample"):
            load_config(write_cfg(tmp_path, raw))

    def test_negative_bps_rejected(self, tmp_path):
        raw = load_raw()
        raw["evaluation"]["commission_bps"] = -1.0
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_threshold_auc_bounds(self, tmp_path):
        raw = load_raw()
        raw["validation"]["thresholds"]["min_oos_auc"] = 0.3  # < 0.5 — бессмыслица
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_catboost_allowed_in_schema(self, tmp_path):
        raw = load_raw()
        raw["model"]["kind"] = "catboost"
        config = load_config(write_cfg(tmp_path, raw))
        assert config.model.kind == "catboost"  # отклонится в runtime с объяснением (D4)

    def test_seed_explicit(self, tmp_path):
        raw = load_raw()
        raw["model"]["seed"] = 7
        assert load_config(write_cfg(tmp_path, raw)).model.seed == 7
