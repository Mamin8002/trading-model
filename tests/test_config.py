"""Тесты схемы и загрузки конфигурации (lookback/horizon обязательны и явны)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tradebot.config import ConfigError, build_calendar, config_hash, load_config

BASE = Path("configs/example.yaml")
DEMO = Path("configs/demo.yaml")


def load_raw(path: Path = BASE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_cfg(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class TestLoading:
    def test_example_config_valid(self):
        config = load_config(BASE)
        assert config.run.mode == "TEST"
        assert config.data.symbols[0].symbol == "SBER@MISX"
        assert config.data.timestamp_semantics == "bar_open"
        tf5 = config.timeframes["5m"]
        assert tf5.lookback_bars == 240 and tf5.horizon_bars == 6

    def test_demo_config_valid(self):
        config = load_config(DEMO)
        assert config.data.csv_dir == Path("data/demo")
        assert "10m" in config.timeframes
        assert config.timeframes["10m"].base_timeframe == "5m"

    def test_hash_deterministic_and_order_independent(self):
        config1 = load_config(BASE)
        raw = load_raw()
        # переставим ключи секций — хэш не должен измениться
        reordered = {k: raw[k] for k in reversed(list(raw.keys()))}
        path = Path("tests/_tmp_reordered.yaml")
        path.write_text(yaml.safe_dump(reordered, allow_unicode=True), encoding="utf-8")
        try:
            config2 = load_config(path)
        finally:
            path.unlink(missing_ok=True)
        assert config_hash(config1) == config_hash(config2)
        assert len(config_hash(config1)) == 64

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="не найден"):
            load_config(tmp_path / "nope.yaml")

    def test_broken_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("run: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="YAML"):
            load_config(path)


class TestStrictness:
    def test_missing_horizon_rejected(self, tmp_path):
        raw = load_raw()
        del raw["timeframes"]["5m"]["horizon_bars"]
        with pytest.raises(ConfigError, match="horizon_bars"):
            load_config(write_cfg(tmp_path, raw))

    def test_missing_lookback_rejected(self, tmp_path):
        raw = load_raw()
        del raw["timeframes"]["5m"]["lookback_bars"]
        with pytest.raises(ConfigError, match="lookback_bars"):
            load_config(write_cfg(tmp_path, raw))

    def test_missing_timestamp_semantics_rejected(self, tmp_path):
        raw = load_raw()
        del raw["data"]["timestamp_semantics"]
        with pytest.raises(ConfigError, match="timestamp_semantics"):
            load_config(write_cfg(tmp_path, raw))

    def test_unknown_timeframe_rejected(self, tmp_path):
        raw = load_raw()
        raw["timeframes"]["30m"] = {"lookback_bars": 10, "horizon_bars": 1}
        with pytest.raises(ConfigError, match="30m"):
            load_config(write_cfg(tmp_path, raw))

    def test_unknown_extra_field_rejected(self, tmp_path):
        raw = load_raw()
        raw["run"]["oops_typo"] = True
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_10m_requires_base_timeframe(self, tmp_path):
        raw = load_raw()
        raw["timeframes"]["10m"]["base_timeframe"] = None
        with pytest.raises(ConfigError, match="base_timeframe"):
            load_config(write_cfg(tmp_path, raw))

    def test_base_timeframe_must_be_smaller_and_divisible(self, tmp_path):
        raw = load_raw()
        raw["timeframes"]["5m"]["base_timeframe"] = "15m"
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_nonpositive_lookback_rejected(self, tmp_path):
        raw = load_raw()
        raw["timeframes"]["5m"]["lookback_bars"] = 0
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_http_base_url_rejected(self, tmp_path):
        raw = load_raw()
        raw["finam"]["base_url"] = "http://api.finam.ru"
        with pytest.raises(ConfigError, match="https"):
            load_config(write_cfg(tmp_path, raw))

    def test_rate_limit_above_documented_rejected(self, tmp_path):
        raw = load_raw()
        raw["finam"]["rate_limit_per_minute"] = 500  # документированный лимит 200/мин
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_bad_timezone_rejected(self, tmp_path):
        raw = load_raw()
        raw["data"]["source_timezone"] = "Mars/Olympus"
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))


class TestCalendarBuild:
    def test_calendar_built_from_config(self):
        config = load_config(DEMO)
        calendar = build_calendar(config)
        assert calendar is not None
        assert calendar.timezone.key == "Europe/Moscow"

    def test_calendar_none_when_sessions_absent(self, tmp_path):
        raw = load_raw()
        del raw["sessions"]
        config = load_config(write_cfg(tmp_path, raw))
        assert build_calendar(config) is None
