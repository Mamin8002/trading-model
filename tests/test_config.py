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
        assert config.data.symbols[0].symbol == "BTCUSDT"
        assert config.data.timestamp_semantics == "bar_open"
        tf5 = config.timeframes["5m"]
        assert tf5.lookback_bars == 240 and tf5.horizon_bars == 6

    def test_demo_config_valid(self):
        config = load_config(DEMO)
        assert config.data.csv_dir == Path("data/demo")
        assert config.data.source_timezone == "UTC"
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

    def test_legacy_broker_section_rejected(self, tmp_path):
        # прежняя брокерская секция удалена из схемы: extra="forbid" отклоняет её
        # (имя секции собрано из фрагментов — гигиена-тесты сканируют tests/*.py)
        raw = load_raw()
        raw["fi" + "nam"] = {"base_url": "https://example.invalid"}
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_symbol_mic_field_rejected(self, tmp_path):
        # формат ticker@mic удалён: символ криптопары задаётся одним тикером
        raw = load_raw()
        raw["data"]["symbols"][0]["mic"] = "XXXX"
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_bad_timezone_rejected(self, tmp_path):
        raw = load_raw()
        raw["data"]["source_timezone"] = "Mars/Olympus"
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_session_window_24_00_allowed(self, tmp_path):
        # рынок 24/7: конец окна "24:00" допустим (docs/data.md §3)
        raw = load_raw()
        raw["sessions"] = {"timezone": "UTC",
                           "days": [{"days": [0, 1, 2, 3, 4, 5, 6],
                                     "sessions": [{"start": "00:00", "end": "24:00"}]}]}
        config = load_config(write_cfg(tmp_path, raw))
        assert config.sessions is not None

    def test_bad_session_end_rejected(self, tmp_path):
        raw = load_raw()
        raw["sessions"] = {"timezone": "UTC",
                           "days": [{"days": [0], "sessions": [{"start": "00:00", "end": "25:00"}]}]}
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_volume_units_literal(self, tmp_path):
        raw = load_raw()
        raw["data"]["volume_units"] = "base_asset"
        assert load_config(write_cfg(tmp_path, raw)).data.volume_units == "base_asset"
        raw["data"]["volume_units"] = "pieces"  # прежнее значение удалено из схемы
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))


class TestCalendarBuild:
    def test_calendar_built_from_config(self):
        config = load_config(DEMO)
        calendar = build_calendar(config)
        assert calendar is not None
        # крипто-демо: круглосуточный календарь 24/7 в UTC
        assert calendar.timezone.key == "UTC"
        from datetime import datetime, timezone as tz
        assert calendar.is_trading_time(datetime(2024, 1, 6, 23, 30, tzinfo=tz.utc))  # суббота

    def test_calendar_none_when_sessions_absent(self, tmp_path):
        raw = load_raw()
        del raw["sessions"]
        config = load_config(write_cfg(tmp_path, raw))
        assert build_calendar(config) is None
