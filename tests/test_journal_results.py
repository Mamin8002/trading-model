"""Тесты журнала решений/сделок и расчёта результатов (фаза 4)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradebot.backtest.engine import run_simulation
from tradebot.backtest.results import compute_results, render_simulation_report
from tradebot.config import load_config
from tradebot.config.env import SecretMasker
from tradebot.execution.interfaces import Side, Signal
from tradebot.execution.journal import JOURNAL_SCHEMA_VERSION, JournalWriter, audit_line, read_journal
from tradebot.strategy.signals import ScriptedSignalSource

from test_simulator_engine import BASE_CFG, FLAT, build_config, buy_signal, exit_signal, make_bars

UTC = timezone.utc


class TestJournal:
    def test_records_and_audit_trail(self, tmp_path):
        bars = make_bars(FLAT)
        signals = [buy_signal(bars[0]), exit_signal(bars[3])]
        source = ScriptedSignalSource(signals, symbol="TEST-USDT", timeframe="1m")
        path = tmp_path / "journal.jsonl"
        with JournalWriter(path, symbol="TEST-USDT", timeframe="1m", model_version="test") as journal:
            run = run_simulation(build_config(), bars, source, timeframe="1m",
                                 symbol="TEST-USDT", journal=journal)
        records = read_journal(path)
        kinds = [r["kind"] for r in records]
        assert kinds.count("signal") == 2
        assert kinds.count("risk_decision") >= 2
        assert kinds.count("intent") == 2
        assert kinds.count("fill") == 2
        assert kinds.count("state") >= 1
        for record in records:
            assert record["schema_version"] == JOURNAL_SCHEMA_VERSION
            assert record["symbol"] == "TEST-USDT" and record["timeframe"] == "1m"
            assert record["model_version"] == "test"
            assert "written_at" in record
        # fill содержит план и факт
        fills = [r for r in records if r["kind"] == "fill"]
        for fill in fills:
            for key in ("ts" if "ts" in fill else "event_ts", "side", "units", "raw_price",
                        "price", "slippage_bps", "commission_quote", "cause"):
                assert key in fill, f"в fill нет поля {key}"
        # intent помечает reference_price как справочную
        intents = [r for r in records if r["kind"] == "intent"]
        assert all("НЕ цена исполнения" in r["reference_price_note"] for r in intents)
        # аудит: строки читаемы и объясняют решения
        lines = [audit_line(r) for r in records]
        assert any("APPROVED" in line for line in lines)
        assert any("FILL" in line for line in lines)

    def test_rejection_reason_recoverable(self, tmp_path):
        bars = make_bars(FLAT)
        source = ScriptedSignalSource([buy_signal(bars[0])], symbol="TEST-USDT", timeframe="1m")
        cfg = build_config(risk={"position_units": 500, "max_position_units": 100})
        path = tmp_path / "journal.jsonl"
        with JournalWriter(path, symbol="TEST-USDT", timeframe="1m", model_version="test") as journal:
            run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT", journal=journal)
        records = read_journal(path)
        decisions = [r for r in records if r["kind"] == "risk_decision" and not r["approved"]]
        assert decisions, "отказ не попал в журнал"
        rejection = decisions[0]["rejections"][0]
        assert rejection["code"] == "max_position_exceeded"
        assert rejection["limit"] == "risk.max_position_units"
        assert rejection["limit_value"] == 100.0
        # по журналу видно, ЧТО нарушено и чем
        assert rejection["message"]

    def test_secrets_never_reach_journal(self, tmp_path):
        masker = SecretMasker()
        masker.register("SUPER-SECRET-VALUE-123", label="SECRET_KEY")
        path = tmp_path / "journal.jsonl"
        with JournalWriter(path, symbol="TEST-USDT", timeframe="1m", model_version="test", masker=masker) as journal:
            journal.record("warning", message="случайно секрет SUPER-SECRET-VALUE-123 в тексте")
        text = path.read_text(encoding="utf-8")
        assert "SUPER-SECRET-VALUE-123" not in text
        assert "***SECRET_KEY***" in text

    def test_unknown_kind_rejected(self, tmp_path):
        with JournalWriter(tmp_path / "j.jsonl", symbol="S", timeframe="1m", model_version="t") as journal:
            with pytest.raises(ValueError):
                journal.record("unknown_kind")

    def test_corrupted_line_detected(self, tmp_path):
        path = tmp_path / "j.jsonl"
        path.write_text('{"kind": "state"}\n{broken\n', encoding="utf-8")
        with pytest.raises(ValueError, match="повреждён"):
            read_journal(path)


class TestResults:
    def _run(self, cfg=None, signals=None, bars=None):
        bars = bars or make_bars(FLAT)
        signals = signals if signals is not None else [buy_signal(bars[0]), exit_signal(bars[2])]
        source = ScriptedSignalSource(signals, symbol="TEST-USDT", timeframe="1m")
        run = run_simulation(cfg or build_config(), bars, source, timeframe="1m", symbol="TEST-USDT")
        return run, compute_results(run, cfg or build_config(), bars), bars

    def test_hand_computed_round_trip(self):
        # flat-серия: вход и выход по 100, 10 шт, комиссии 0 -> net 0
        run, results, bars = self._run()
        assert results.n_entries == 1 and results.n_round_trips == 1
        assert results.net_return == pytest.approx(0.0)
        assert results.total_commission == pytest.approx(0.0)
        trade = results.trades[0]
        assert trade.entry_price == pytest.approx(100.0) and trade.exit_price == pytest.approx(100.0)
        assert trade.net_pnl == pytest.approx(0.0)
        assert trade.cause == "exit_signal"

    def test_costs_in_net_result(self):
        cfg = build_config(simulation={"commission_bps": 50.0, "slippage_bps": 0.0})
        run, results, bars = self._run(cfg=cfg)
        # две сделки по notional 1000 -> комиссии 5 + 5
        assert results.total_commission == pytest.approx(10.0)
        # чистый результат = gross(0) - комиссии
        assert results.final_equity == pytest.approx(results.initial_capital - 10.0)
        assert results.net_return == pytest.approx(-10.0 / results.initial_capital)
        assert results.turnover_quote == pytest.approx(2000.0)

    def test_hit_rate_and_profit_factor_none_without_closed_trades(self):
        bars = make_bars(FLAT)
        run, results, _ = self._run(signals=[], bars=bars)
        assert results.hit_rate is None and results.profit_factor is None
        assert results.n_entries == 0

    def test_report_sections_and_assumptions(self):
        run, results, bars = self._run()
        report = render_simulation_report(results, build_config(), "journal.jsonl")
        for fragment in (
            "Дисклеймер", "Рисковые лимиты", "Допущения симуляции", "commission_bps",
            "slippage_bps", "stop_target_conflict", "годовой пересчёт", "bars_per_year",
            "Отказы риск-контроля", "Ограничения симулятора", "не являются обещанием доходности",
            "mark-to-market",
        ):
            assert fragment.lower() in report.lower(), f"нет фрагмента {fragment!r}"
        # неза заданные параметры инструмента видны как НЕ ЗАДАНЫ (не выдуманы)
        assert "НЕ ЗАДАН" in report or "None" in report

    def test_annualization_source_reported(self):
        run, results, bars = self._run()
        assert results.assumptions["annualization"]["bars_per_year_source"] in ("derived", "config")
        assert results.assumptions["annualization"]["bars_per_year"] > 0

    def test_demo_config_simulates(self, tmp_path):
        # configs/sim_demo.yaml грузится и даёт валидные секции фазы 4
        config = load_config("configs/sim_demo.yaml")
        assert config.risk is not None and config.risk.initial_capital_quote == 100_000.0
        assert config.simulation is not None and config.simulation.commission_bps == 3.0
        assert config.instrument is not None and config.instrument.price_step == 0.01
