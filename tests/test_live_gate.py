"""Тесты многофакторного LIVE-гейта (docs/live.md).

Ключевые свойства крипто-версии:
- интеграция с биржей НЕ реализована (exchange/base.py) -> условие №0 гейта не
  выполнено НИКОГДА: LIVE заблокирован даже при полном явном «включении»;
- по умолчанию LIVE закрыт: гейт отказывает без явных множественных включений;
- отказ происходит ДО любых сетевых вызовов (гейт — чистая функция);
- частичное включение не даёт обхода: требуется ВСЁ одновременно;
- kill-switch блокирует даже полностью разрешающую конфигурацию.
"""

from __future__ import annotations

import copy

import pytest

from tradebot.config import AppConfig
from tradebot.exchange.base import EXCHANGE_INTEGRATION_AVAILABLE
from tradebot.live.gate import LiveGateError, evaluate_live_gate

FULL_LIVE_CFG = {
    "run": {"mode": "LIVE", "experiment_id": "live-test"},
    "data": {
        "csv_dir": "data/raw",
        "symbols": [{"ticker": "BTCUSDT", "files": {}}],
        "timestamp_semantics": "bar_open",
        "source_timezone": "UTC",
    },
    "timeframes": {"5m": {"enabled": True, "lookback_bars": 240, "horizon_bars": 6}},
    "risk": {
        "initial_capital_quote": 100_000.0,
        "position_units": 10,
        "max_position_units": 100,
        "max_orders_per_day": 10,
        "daily_loss_limit_quote": 1_000.0,
        "max_drawdown_pct": 5.0,
        "max_data_age_seconds": 60.0,
    },
    "instrument": {"price_step": 0.01, "lot_size_units": 1, "quote_currency": "USDT"},
    "simulation": {"commission_bps": 3.0, "slippage_bps": 2.0},
    "live": {
        "allow_orders": True,
        "kill_switch_file": ".kill",
        "max_order_quantity_units": 10,
        "ack_env_var": "TRADEBOT_LIVE_ACK",
        "ack_expected_value": "ENABLE-LIVE",
    },
}

FULL_ENV = {
    "TRADEBOT_LIVE_ACK": "ENABLE-LIVE",
}

EXCHANGE_BLOCK_FRAGMENT = "НЕ реализована"


def make_config(**section_overrides) -> AppConfig:
    raw = copy.deepcopy(FULL_LIVE_CFG)
    for section, value in section_overrides.items():
        if value is None:
            raw.pop(section, None)
        else:
            raw[section] = value
    return AppConfig.model_validate(raw)


class TestGate:
    def test_exchange_integration_is_declared_unavailable(self):
        # константа-факт: пока адаптера биржи нет, LIVE не может быть разрешён
        assert EXCHANGE_INTEGRATION_AVAILABLE is False

    def test_full_explicit_enable_still_blocked_without_exchange(self):
        # даже ВСЕ флаги/конфиги/env одновременно НЕ открывают LIVE:
        # условие №0 (интеграция с биржей) не выполнено
        result = evaluate_live_gate(make_config(), cli_flag=True, env=FULL_ENV, kill_switch_exists=False)
        assert not result.allowed
        assert any(EXCHANGE_BLOCK_FRAGMENT in f for f in result.failures), result.failures
        # прочие условия при этом выполнены — других отказов нет
        assert all(EXCHANGE_BLOCK_FRAGMENT in f for f in result.failures)

    def test_default_is_denied(self):
        # «голый» TEST-конфиг (без live-секции) — гейт закрыт
        from tradebot.config import load_config

        config = load_config("configs/demo.yaml")
        result = evaluate_live_gate(config, cli_flag=False, env={}, kill_switch_exists=False)
        assert not result.allowed
        assert len(result.failures) >= 3  # биржа, mode, флаг, live-секция...

    @pytest.mark.parametrize(
        ("mutation", "expected_fragment"),
        [
            (lambda cfg: cfg.update(run={**cfg["run"], "mode": "TEST"}), "run.mode"),
            (lambda cfg: cfg.update(live={**cfg["live"], "allow_orders": False}), "allow_orders"),
            (lambda cfg: cfg.update(live={**cfg["live"], "max_order_quantity_units": None}), "max_order_quantity_units"),
            (lambda cfg: cfg.pop("live"), "live"),
            (lambda cfg: cfg.update(instrument={**cfg["instrument"], "price_step": None}), "price_step"),
            (lambda cfg: cfg.update(instrument={**cfg["instrument"], "lot_size_units": None}), "lot_size_units"),
            (lambda cfg: cfg.update(risk={**cfg["risk"], "max_drawdown_pct": None}), "max_drawdown_pct"),
        ],
    )
    def test_each_condition_is_required(self, mutation, expected_fragment):
        raw = copy.deepcopy(FULL_LIVE_CFG)
        mutation(raw)
        result = evaluate_live_gate(AppConfig.model_validate(raw), cli_flag=True, env=FULL_ENV, kill_switch_exists=False)
        assert not result.allowed
        assert any(expected_fragment in f for f in result.failures), result.failures

    def test_cli_flag_required(self):
        result = evaluate_live_gate(make_config(), cli_flag=False, env=FULL_ENV, kill_switch_exists=False)
        assert not result.allowed
        assert any("--allow-live" in f for f in result.failures)

    def test_env_ack_required(self):
        env = {k: v for k, v in FULL_ENV.items() if k != "TRADEBOT_LIVE_ACK"}
        result = evaluate_live_gate(make_config(), cli_flag=True, env=env, kill_switch_exists=False)
        assert not result.allowed
        assert any("TRADEBOT_LIVE_ACK" in f for f in result.failures)

    def test_wrong_ack_value_denied(self):
        env = dict(FULL_ENV, TRADEBOT_LIVE_ACK="yes-please")
        result = evaluate_live_gate(make_config(), cli_flag=True, env=env, kill_switch_exists=False)
        assert not result.allowed

    def test_kill_switch_blocks_even_fully_enabled(self):
        result = evaluate_live_gate(make_config(), cli_flag=True, env=FULL_ENV, kill_switch_exists=True)
        assert not result.allowed
        assert any("kill-switch" in f for f in result.failures)

    def test_raise_if_denied_lists_all_failures(self):
        result = evaluate_live_gate(make_config(), cli_flag=False, env={}, kill_switch_exists=True)
        with pytest.raises(LiveGateError) as exc:
            result.raise_if_denied()
        assert len(exc.value.failures) >= 3

    def test_gate_failure_precedes_any_network(self):
        # гейт — чистая функция: не создаёт клиентов и не читает секреты из файлов;
        # при отказе вызывающий CLI завершается до какого-либо сетевого кода
        result = evaluate_live_gate(make_config(live=None), cli_flag=True, env=FULL_ENV, kill_switch_exists=False)
        assert not result.allowed

    def test_checks_include_exchange_factor(self):
        result = evaluate_live_gate(make_config(), cli_flag=True, env=FULL_ENV, kill_switch_exists=False)
        assert "exchange_integration" in result.checks_performed
