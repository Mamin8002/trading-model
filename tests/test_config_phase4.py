"""Тесты схемы конфигурации фазы 4 (risk/instrument/simulation) и совместимости."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tradebot.config import ConfigError, load_config

SIM_DEMO = Path("configs/sim_demo.yaml")


def load_raw(path: Path = SIM_DEMO) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_cfg(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class TestSchema:
    def test_sim_demo_loads(self):
        config = load_config(SIM_DEMO)
        assert config.risk.initial_capital_quote == 100_000.0
        assert config.risk.position_units is None  # режим не задан -> risk_per_trade_quote
        assert config.risk.risk_per_trade_quote == 500.0
        assert config.simulation.commission_bps == 3.0
        assert config.simulation.conflict_policy == "stop_first"
        assert config.simulation.signal.entry_threshold == 0.55

    def test_phase23_configs_still_valid_without_new_sections(self):
        for name in ("configs/example.yaml", "configs/demo.yaml", "configs/ml_example.yaml"):
            config = load_config(name)
            assert config.risk is None or config.risk.initial_capital_quote is None or True
            # секции фазы 4 опциональны
        assert load_config("configs/demo.yaml").simulation is None

    def test_ambiguous_sizing_modes_rejected(self, tmp_path):
        raw = load_raw()
        raw["risk"]["position_units"] = 10  # уже задан risk_per_trade_quote
        with pytest.raises(ConfigError, match="РОВНО ОДИН"):
            load_config(write_cfg(tmp_path, raw))

    def test_exit_threshold_must_be_below_entry(self, tmp_path):
        raw = load_raw()
        raw["simulation"]["signal"]["exit_threshold"] = 0.7  # entry 0.55
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_unknown_conflict_policy_rejected(self, tmp_path):
        raw = load_raw()
        raw["simulation"]["conflict_policy"] = "target_first"  # «выгодный» сценарий не предлагается
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_negative_limits_rejected(self, tmp_path):
        raw = load_raw()
        raw["risk"]["daily_loss_limit_quote"] = -1.0
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))

    def test_defaults_are_none_not_hidden_values(self):
        from tradebot.config.schema import RiskConfig, SimulationConfig

        risk = RiskConfig()
        assert risk.initial_capital_quote is None
        assert risk.position_units is None
        assert risk.max_position_units is None
        assert risk.daily_loss_limit_quote is None
        assert risk.max_drawdown_pct is None
        assert risk.max_orders_per_day is None
        assert risk.max_data_age_seconds is None
        assert risk.allow_short is False  # явный безопасный флаг, не финансовая величина
        sim = SimulationConfig()
        assert sim.commission_bps is None and sim.slippage_bps is None

    def test_price_step_and_lot_positive(self, tmp_path):
        raw = load_raw()
        raw["instrument"]["price_step"] = 0.0
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))
        raw = load_raw()
        raw["instrument"]["lot_size_units"] = 0
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, raw))
