"""Интеграционные проверки: полный TEST-путь (CSV -> обучение -> симуляция) и
безопасная блокировка LIVE-контура (биржевой адаптер не реализован).

Все прогоны офлайн: сеть не используется, реальные заявки невозможны.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from tradebot.cli.run import main as run_main
from tradebot.data.synthetic import generate_synthetic_bars
from tradebot.execution.journal import read_journal

TS = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    csv_dir = root / "csv"
    csv_dir.mkdir()
    # непрерывная круглосуточная серия (крипторежим 24/7)
    bars_5m = generate_synthetic_bars(640, seed=7, timeframe="5m", symbol="SYNTH-USDT", continuous=True)
    with (csv_dir / "SYNTH_M5.csv").open("w", encoding="utf-8", newline="") as fh:
        fh.write("datetime,open,high,low,close,volume\n")
        for bar in bars_5m:
            fh.write(f"{bar.ts_open.strftime('%Y-%m-%d %H:%M:%S')},{bar.open},{bar.high},"
                     f"{bar.low},{bar.close},{bar.volume}\n")
    config = {
        "run": {"mode": "TEST", "experiment_id": "integration-test", "report_dir": str(root)},
        "data": {"csv_dir": str(csv_dir),
                 "symbols": [{"ticker": "SYNTH-USDT", "files": {"5m": "SYNTH_M5.csv"}}],
                 "timestamp_semantics": "bar_open", "source_timezone": "UTC"},
        "timeframes": {"5m": {"enabled": True, "lookback_bars": 120, "horizon_bars": 6,
                              "label": {"kind": "fwd_return_class", "neutral_band_bps": 5.0}}},
        "model": {"kind": "lightgbm", "seed": 7,
                  "params": {"n_estimators": 40, "min_child_samples": 20}},
        "validation": {"walk_forward": {"n_splits": 2, "min_train_bars": 300, "min_val_bars": 60}},
        "risk": {"initial_capital_quote": 100_000.0, "position_units": 10,
                 "max_position_units": 100, "max_orders_per_day": 20,
                 "daily_loss_limit_quote": 2_000.0, "max_drawdown_pct": 10.0,
                 "max_data_age_seconds": 3600},
        "instrument": {"price_step": 0.01, "lot_size_units": 1},
        "simulation": {"commission_bps": 3.0, "commission_min_quote": 0.05, "slippage_bps": 2.0,
                       "signal": {"entry_threshold": 0.55, "exit_threshold": 0.45,
                                  "stop_bps": 100.0, "target_bps": 200.0}},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    exit_code = run_main([
        "test", "--config", str(config_path), "--timeframe", "5m", "--train",
        "--model-output-dir", str(root / "models"), "--report-output-dir", str(root / "reports"),
    ])
    return exit_code, root


class TestFullTestPipelineCLI:
    """Полный TEST-путь: CSV -> обучение -> симуляция -> отчёт + журнал (офлайн)."""

    def test_pipeline_exit_code(self, pipeline):
        exit_code, _ = pipeline
        assert exit_code == 0

    def test_report_journal_and_model_created(self, pipeline):
        _, root = pipeline
        reports = list((root / "reports").glob("test_run_integration-test_5m.md"))
        assert reports, "отчёт TEST-прогона не создан"
        text = reports[0].read_text(encoding="utf-8")
        assert "Допущения симуляции" in text and "Реальные заявки не выставлялись" in text
        journals = list((root / "reports").glob("journal_integration-test_5m.jsonl"))
        assert journals, "журнал не создан"
        records = read_journal(journals[0])
        kinds = {r["kind"] for r in records}
        assert {"state", "risk_decision"} <= kinds
        models = list((root / "models").glob("integration-test_5m_*"))
        assert models, "артефакт модели не сохранён"
        bundle_files = {fp.name for fp in models[0].iterdir()}
        assert {"model.lgb.txt", "metadata.json", "feature_manifest.json"} <= bundle_files


class TestBadConfigScenarios:
    """Ошибочная/неполная конфигурация: отказ БЕЗ заявок."""

    BASE = {
        "run": {"mode": "TEST", "experiment_id": "bad-config"},
        "data": {"csv_dir": "data/demo",
                 "symbols": [{"ticker": "DEMO", "files": {"1m": "DEMO_M1.csv"}}],
                 "timestamp_semantics": "bar_open", "source_timezone": "UTC"},
        "timeframes": {"1m": {"enabled": True, "lookback_bars": 30, "horizon_bars": 3}},
        "risk": {"initial_capital_quote": 100_000.0, "position_units": 1},
    }

    def _write(self, tmp_path, config: dict, name="config.yaml") -> str:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return str(path)

    def test_missing_simulation_exits_2(self, tmp_path, capsys):
        path = self._write(tmp_path, dict(self.BASE))
        assert run_main(["test", "--config", path, "--timeframe", "1m", "--train"]) == 2
        assert "НЕПОЛНАЯ КОНФИГУРАЦИЯ" in capsys.readouterr().err

    def test_missing_risk_exits_2(self, tmp_path):
        config = dict(self.BASE)
        config.pop("risk")
        config["simulation"] = {"commission_bps": 3.0, "slippage_bps": 2.0}
        path = self._write(tmp_path, config)
        assert run_main(["test", "--config", path, "--timeframe", "1m", "--train"]) == 2

    def test_live_mode_rejected_by_test_command(self, tmp_path, capsys):
        config = dict(self.BASE)
        config["run"] = {"mode": "LIVE", "experiment_id": "bad"}
        path = self._write(tmp_path, config)
        assert run_main(["test", "--config", path, "--timeframe", "1m", "--train"]) == 2
        assert "mode='LIVE'" in capsys.readouterr().err


class TestLivePreflightBlocked:
    """live-preflight: отказ всегда — интеграция с биржей не реализована."""

    def _live_config(self, tmp_path) -> str:
        config = json.loads(json.dumps(TestBadConfigScenarios.BASE))
        config["run"]["mode"] = "LIVE"
        config["risk"] = {
            "initial_capital_quote": 100_000.0, "position_units": 10,
            "max_position_units": 100, "max_orders_per_day": 10,
            "daily_loss_limit_quote": 1_000.0, "max_drawdown_pct": 5.0,
            "max_data_age_seconds": 60.0,
        }
        config["instrument"] = {"price_step": 0.01, "lot_size_units": 1}
        config["simulation"] = {"commission_bps": 3.0, "slippage_bps": 2.0}
        config["live"] = {"allow_orders": True, "max_order_quantity_units": 5,
                          "kill_switch_file": "/nonexistent/preflight/.kill"}
        path = tmp_path / "live.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return str(path)

    def test_denied_without_flag_and_env(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("TRADEBOT_LIVE_ACK", raising=False)
        path = self._live_config(tmp_path)
        assert run_main(["live-preflight", "--config", path]) == 2
        err = capsys.readouterr().err
        assert "гейт НЕ пройден" in err

    def test_denied_even_with_full_explicit_enable(self, tmp_path, capsys, monkeypatch):
        # флаг + env + полная live-секция: всё равно отказ, причина — отсутствие
        # биржевой интеграции; сетевой код в CLI даже не импортируется
        monkeypatch.setenv("TRADEBOT_LIVE_ACK", "ENABLE-LIVE")
        path = self._live_config(tmp_path)
        assert run_main(["live-preflight", "--config", path, "--allow-live"]) == 2
        err = capsys.readouterr().err
        assert "гейт НЕ пройден" in err
        assert "НЕ реализована" in err and "бирж" in err.lower()

    def test_no_network_modules_imported_by_preflight(self):
        # в cli.run не осталось сетевых клиентов: только конфиг + гейт
        import inspect

        import tradebot.cli.run as run_module

        source = inspect.getsource(run_module)
        assert "http" not in source.lower()
        assert "urlopen" not in source
