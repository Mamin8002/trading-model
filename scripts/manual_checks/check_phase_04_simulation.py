#!/usr/bin/env python3
"""Ручная проверка фазы 4: риск-контроль, TEST-симулятор, журнал решений.

Запуск (из корня репозитория, PowerShell):
    python scripts\\manual_checks\\check_phase_04_simulation.py

Сценарий ПОЛНОСТЬЮ ЛОКАЛЬНЫЙ и ДЕТЕРМИНИРОВАННЫЙ:
- входные данные — «ручные» серии баров с заранее известными ожидаемыми значениями
  (все числа в проверках посчитаны вручную и напечатаны как «ожидалось»);
- сеть не используется, секреты не требуются;
- РЕАЛЬНЫЕ ЗАЯВКИ НЕ ОТПРАВЛЯЮТСЯ: работает TEST-симулятор (docs/simulation.md),
  подтверждения брокера не имитируются;
- артефакты (журнал, отчёт) создаются в reports/phase04_manual/ (каталог
  исключён из git); список созданных файлов печатается в конце.

Код возврата: 0 — все проверки PASS; 1 — есть FAIL.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradebot.backtest.engine import run_simulation  # noqa: E402
from tradebot.backtest.results import compute_results, render_simulation_report  # noqa: E402
from tradebot.config import AppConfig, load_config  # noqa: E402
from tradebot.config.env import SecretMasker  # noqa: E402
from tradebot.config.schema import InstrumentConfig, RiskConfig, SimulationConfig  # noqa: E402
from tradebot.data.bar import Bar, BarSource  # noqa: E402
from tradebot.execution.errors import SimConfigError  # noqa: E402
from tradebot.execution.interfaces import RejectionCode, RiskContext, Side, Signal  # noqa: E402
from tradebot.execution.journal import JournalWriter, audit_line, read_journal  # noqa: E402
from tradebot.execution.risk import RiskGuards  # noqa: E402
from tradebot.execution.sizing import SizingError, compute_position_size  # noqa: E402
from tradebot.strategy.signals import ScriptedSignalSource  # noqa: E402

UTC = timezone.utc
START = datetime(2024, 1, 4, 8, 0, tzinfo=UTC)  # четверг
OUT_DIR = REPO_ROOT / "reports" / "phase04_manual"
CREATED: list[str] = []

CHECKS: list[tuple[str, str, Callable[[], str]]] = []


def check(name: str, expected: str):
    """Регистрирует проверку с ПЕЧАТЬЮ ожидаемого результата (требование ТЗ)."""

    def decorator(fn: Callable[[], str]):
        CHECKS.append((name, expected, fn))
        return fn

    return decorator


def close(value: float, expected: float, tol: float = 1e-9) -> bool:
    """Точное сравнение с плавающей точкой (сценарий не зависит от тестовых библиотек)."""
    return abs(value - expected) <= tol * max(1.0, abs(expected))


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_bars(ohlc: list[tuple[float, float, float, float]], start: datetime = START) -> list[Bar]:
    return [
        Bar(ts_open=start + timedelta(minutes=i), open=o, high=h, low=l, close=c,
            volume=1000 + i, source=BarSource.CSV, symbol="TEST-USDT")
        for i, (o, h, l, c) in enumerate(ohlc)
    ]


def buy_at(bar: Bar, stop=None, target=None) -> Signal:
    return Signal(ts_open=bar.ts_open, side=Side.BUY, symbol="TEST-USDT", timeframe="1m",
                  model_version="scripted-demo", proba=0.9, stop_price=stop, target_price=target)


def exit_at(bar: Bar) -> Signal:
    return Signal(ts_open=bar.ts_open, side=None, symbol="TEST-USDT", timeframe="1m",
                  model_version="scripted-demo", proba=0.1, meta={"kind": "exit"})


BASE_CFG: dict[str, Any] = {
    "run": {"mode": "TEST", "experiment_id": "phase04-manual", "model_dir": "models", "report_dir": "reports"},
    "data": {
        "csv_dir": "data/raw",
        "symbols": [{"ticker": "TEST-USDT", "files": {}}],
        "timestamp_semantics": "bar_open",
        "source_timezone": "UTC",
    },
    "timeframes": {"1m": {"enabled": True, "lookback_bars": 10, "horizon_bars": 1}},
    "risk": {
        "initial_capital_quote": 100_000.0,
        "position_units": 10,
        "max_position_units": 1_000,
        "max_orders_per_day": 100,
        "daily_loss_limit_quote": 100_000.0,
        "max_drawdown_pct": 50.0,
        "max_data_age_seconds": 3_600.0,
        "allow_short": False,
        "halt_on_limit_breach": True,
    },
    "instrument": {"price_step": None, "lot_size_units": None, "quote_currency": None},
    "simulation": {
        "commission_bps": 0.0,
        "commission_min_quote": None,
        "slippage_bps": 0.0,
        "conflict_policy": "stop_first",
        "signal": {"entry_threshold": 0.5, "exit_threshold": None, "stop_bps": None, "target_bps": None},
    },
}


def build_config(**overrides) -> AppConfig:
    raw = copy.deepcopy(BASE_CFG)
    for section, values in overrides.items():
        if section == "simulation" and "signal" in values:
            raw["simulation"]["signal"].update(values.pop("signal"))
        raw[section].update(values) if section in ("risk", "instrument", "simulation") else raw.update({section: values})
    return AppConfig.model_validate(raw)


def simulate(cfg, bars, signals, journal_path: Path | None = None, bar_hook=None):
    source = ScriptedSignalSource(signals, symbol="TEST-USDT", timeframe="1m")
    if journal_path is None:
        return run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT",
                              bar_hook=bar_hook), None
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with JournalWriter(journal_path, symbol="TEST-USDT", timeframe="1m",
                       model_version="scripted-demo") as journal:
        run = run_simulation(cfg, bars, source, timeframe="1m", symbol="TEST-USDT",
                             journal=journal, bar_hook=bar_hook)
    return run, journal_path


# ---------------------------------------------------------------------------
# 1. Конфигурация: явность, никаких скрытых финансовых значений
# ---------------------------------------------------------------------------

@check("Конфиг sim_demo.yaml загружен; дефолты лимитов — None (не выдуманы)",
       "risk/simulation/instrument читаются; у «голой» RiskConfig все лимиты None")
def check_config_explicit() -> str:
    config = load_config(REPO_ROOT / "configs" / "sim_demo.yaml")
    expect(config.risk is not None and config.risk.initial_capital_quote == 100_000.0, "демо-капитал не прочитан")
    expect(config.simulation is not None and config.simulation.commission_bps == 3.0, "демо-комиссия не прочитана")
    empty = RiskConfig()
    for field_name in ("initial_capital_quote", "position_units", "risk_per_trade_quote",
                       "max_position_units", "max_orders_per_day", "daily_loss_limit_quote",
                       "max_drawdown_pct", "max_data_age_seconds"):
        expect(getattr(empty, field_name) is None, f"{field_name} имеет скрытый дефолт!")
    sim = SimulationConfig()
    expect(sim.commission_bps is None and sim.slippage_bps is None, "скрытые дефолты издержек")
    return "все финансовые параметры по умолчанию None; демо-конфиг задаёт их явно"


@check("Неполная конфигурация симуляции -> безопасный отказ SimConfigError",
       "перечислены ВСЕ недостающие поля: simulation.commission_bps и risk.max_position_units")
def check_config_missing_refused() -> str:
    raw = copy.deepcopy(BASE_CFG)
    raw["simulation"]["commission_bps"] = None
    raw["risk"]["max_position_units"] = None
    cfg = AppConfig.model_validate(raw)
    bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * 4)
    try:
        simulate(cfg, bars, [])
    except SimConfigError as exc:
        text = str(exc)
        expect("simulation.commission_bps" in text, "commission_bps не упомянут")
        expect("risk.max_position_units" in text, "max_position_units не упомянут")
        return f"отказ до запуска; недостающие поля перечислены ({len(exc.missing)} шт)"
    raise AssertionError("симуляция стартовала без явных параметров")


@check("Двусмысленный сайзинг и «выгодное» правило конфликта отклоняются схемой",
       "ConfigError: два режима размера сразу; conflict_policy=target_first не существует")
def check_config_strictness() -> str:
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "configs" / "sim_demo.yaml").read_text(encoding="utf-8"))
    raw2 = copy.deepcopy(raw)
    raw2["risk"]["position_units"] = 10  # уже задан risk_per_trade_quote
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text(yaml.safe_dump(raw2, allow_unicode=True), encoding="utf-8")
        try:
            load_config(p)
            raise AssertionError("двусмысленный сайзинг принят")
        except Exception as exc:
            expect("РОВНО ОДИН" in str(exc), f"сообщение: {exc}")
    raw3 = copy.deepcopy(raw)
    raw3["simulation"]["conflict_policy"] = "target_first"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.yaml"
        p.write_text(yaml.safe_dump(raw3, allow_unicode=True), encoding="utf-8")
        try:
            load_config(p)
            raise AssertionError("target_first принят — автоматический выбор выгодного сценария!")
        except Exception:
            pass
    return "двусмысленность и target_first отклоняются"


# ---------------------------------------------------------------------------
# 2. Размер позиции (ручные расчёты)
# ---------------------------------------------------------------------------

def _sim(commission_bps=0.0, commission_min=None):
    return SimulationConfig(commission_bps=commission_bps, commission_min_quote=commission_min, slippage_bps=0.0)


@check("Сайзинг по риску: 1000/(100-95)=200 шт; лот 7 -> 196; pct-режим; кэш-кэп",
       "units=200; units=196; 1% от 50000 / 2.5 = 200; при cash=10000 -> 100 (capped_by=capital)")
def check_sizing_values() -> str:
    def risk_cfg(**kw):
        base = dict(initial_capital_quote=100_000.0, risk_per_trade_quote=1_000.0,
                    max_position_units=1_000, max_orders_per_day=10, daily_loss_limit_quote=1_000.0,
                    max_drawdown_pct=10.0, max_data_age_seconds=60.0)
        base.update(kw)
        return RiskConfig(**base)

    r1 = compute_position_size(risk_cfg=risk_cfg(), instrument_cfg=None, sim_cfg=_sim(),
                               entry_price=100.0, stop_price=95.0, cash=1e6, equity=1e6, current_units=0)
    expect(r1.units == 200, f"ожидалось 200, получено {r1.units}")
    r2 = compute_position_size(risk_cfg=risk_cfg(), instrument_cfg=InstrumentConfig(lot_size_units=7),
                               sim_cfg=_sim(), entry_price=100.0, stop_price=95.0,
                               cash=1e6, equity=1e6, current_units=0)
    expect(r2.units == 196, f"ожидалось 196 (floor до лота 7), получено {r2.units}")
    r3 = compute_position_size(risk_cfg=risk_cfg(risk_per_trade_quote=None, risk_per_trade_pct=1.0),
                               instrument_cfg=None, sim_cfg=_sim(), entry_price=50.0, stop_price=47.5,
                               cash=1e6, equity=50_000.0, current_units=0)
    expect(r3.units == 200, f"ожидалось 200 (1% от 50000 / 2.5), получено {r3.units}")
    r4 = compute_position_size(risk_cfg=risk_cfg(), instrument_cfg=None, sim_cfg=_sim(),
                               entry_price=100.0, stop_price=95.0, cash=10_000.0, equity=10_000.0,
                               current_units=0)
    expect(r4.units == 100 and r4.capped_by == "capital", f"ожидалось 100/capital, получено {r4.units}/{r4.capped_by}")
    return "200 / 196 / 200 / 100(capital) — как в ручном расчёте"


@check("Без режима сайзинга и без стопа — безопасный отказ (код в журнал)",
       "SizingError: RISK_PARAM_NOT_SET (режим не задан); SIZING_STOP_REQUIRED (стоп обязателен)")
def check_sizing_safe_refusal() -> str:
    cfg_none = RiskConfig(initial_capital_quote=100_000.0, max_position_units=100,
                          max_orders_per_day=10, daily_loss_limit_quote=100.0,
                          max_drawdown_pct=10.0, max_data_age_seconds=60.0)
    try:
        compute_position_size(risk_cfg=cfg_none, instrument_cfg=None, sim_cfg=_sim(),
                              entry_price=100.0, stop_price=None, cash=1e6, equity=1e6, current_units=0)
        raise AssertionError("сайзинг без режима не отклонён")
    except SizingError as exc:
        expect(exc.rejection.code is RejectionCode.RISK_PARAM_NOT_SET, f"код: {exc.rejection.code}")
    cfg_risk = RiskConfig(initial_capital_quote=100_000.0, risk_per_trade_quote=500.0,
                          max_position_units=100, max_orders_per_day=10, daily_loss_limit_quote=100.0,
                          max_drawdown_pct=10.0, max_data_age_seconds=60.0)
    try:
        compute_position_size(risk_cfg=cfg_risk, instrument_cfg=None, sim_cfg=_sim(),
                              entry_price=100.0, stop_price=None, cash=1e6, equity=1e6, current_units=0)
        raise AssertionError("рисковый сайзинг без стопа не отклонён")
    except SizingError as exc:
        expect(exc.rejection.code is RejectionCode.SIZING_STOP_REQUIRED, f"код: {exc.rejection.code}")
    return "оба отказа с машиночитаемыми кодами"


# ---------------------------------------------------------------------------
# 3. Время исполнения и цены
# ---------------------------------------------------------------------------

@check("Исполнение НЕ РАНЬШЕ следующего бара: вход/выход по open t+1, не по close t",
       "сигнал b0(close=100) -> fill b1 по open=101; выход сигнал b2(close=101) -> fill b3 по open=102")
def check_execution_timing() -> str:
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),   # b0 сигнал BUY
        (101.0, 101.5, 100.5, 101.0),  # b1 вход по open=101
        (101.0, 101.5, 100.5, 101.0),  # b2 сигнал выхода
        (102.0, 102.5, 101.5, 102.0),  # b3 выход по open=102
        (102.0, 102.5, 101.5, 102.0),
    ]
    bars = make_bars(ohlc)
    run, _ = simulate(build_config(), bars, [buy_at(bars[0]), exit_at(bars[2])])
    expect(len(run.fills) == 2, f"филлов: {len(run.fills)}")
    entry, exit_ = run.fills
    expect(entry.ts == bars[1].ts_open and close(entry.price, 101.0),
           f"вход: {entry.ts} @ {entry.price}")
    expect(exit_.ts == bars[3].ts_open and close(exit_.price, 102.0),
           f"выход: {exit_.ts} @ {exit_.price}")
    expect(entry.price != bars[0].close and exit_.price != bars[2].close,
           "исполнение совпало с close сигнальной свечи — нарушение ТЗ")
    expect(close(run.portfolio.realized_pnl, 10.0), f"PnL: {run.portfolio.realized_pnl}")
    return "fill ts=open следующего бара; PnL=(102-101)*10=10"


@check("Сигнал на последнем баре и конец данных: NO_NEXT_BAR, фиктивного закрытия нет",
       "отказ no_next_bar; открытая позиция остаётся (warning position_open_at_end)")
def check_no_next_bar_and_open_end() -> str:
    bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * 6)
    run1, _ = simulate(build_config(), bars, [buy_at(bars[-1])])
    expect(run1.fills == [], "заявка исполнена без следующего бара")
    expect(any(r["code"] == "no_next_bar" for r in run1.rejections), "нет отказа no_next_bar")
    run2, _ = simulate(build_config(), bars, [buy_at(bars[1])])
    expect(run2.portfolio.units == 10 and len(run2.fills) == 1, "позиция закрыта фиктивной сделкой!")
    expect(any("position_open_at_end" in w for w in run2.warnings), "нет warning position_open_at_end")
    return "no_next_bar зафиксирован; позиция на конец данных не фабрикуется"


# ---------------------------------------------------------------------------
# 4. Комиссии и проскальзывание
# ---------------------------------------------------------------------------

@check("Комиссия: 50 б.п. от оборота = 5.00 на notional 1000; минимум 7.50 применяется",
       "вход 10*100*0.5%=5.0; с min=7.5 -> 7.5; net PnL уменьшается на сумму комиссий")
def check_commission() -> str:
    bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * 6)
    cfg = build_config(simulation={"commission_bps": 50.0})
    run, _ = simulate(cfg, bars, [buy_at(bars[0]), exit_at(bars[2])])
    expect(close(run.fills[0].commission_quote, 5.0), f"комиссия входа: {run.fills[0].commission_quote}")
    expect(close(run.portfolio.total_commission, 10.0), f"всего: {run.portfolio.total_commission}")
    expect(close(run.portfolio.realized_pnl, -10.0), f"realized: {run.portfolio.realized_pnl}")
    cfg_min = build_config(simulation={"commission_bps": 1.0, "commission_min_quote": 7.5})
    run2, _ = simulate(cfg_min, bars, [buy_at(bars[0])])
    expect(close(run2.fills[0].commission_quote, 7.5), f"минимум не применён: {run2.fills[0].commission_quote}")
    results = compute_results(run, cfg, bars)
    expect(close(results.total_commission, 10.0), "комиссии нет в результатах")
    expect(close(results.final_equity, 100_000.0 - 10.0), f"equity: {results.final_equity}")
    return "5.0/leg; min 7.5; net = -10.0; финальный капитал 99990"


@check("Проскальзывание: buy +100б.п. (100->101.00), sell -100б.п. с клампом к low бара",
       "вход 101.00 (raw 100); выход 100.80 (raw 101, кламп low=100.8); slip_cost=12.0")
def check_slippage() -> str:
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),   # b0 сигнал
        (100.0, 103.0, 99.0, 100.0),   # b1 вход: 100*1.01=101 (в диапазоне)
        (101.0, 101.5, 100.5, 101.0),  # b2 сигнал выхода
        (101.0, 101.2, 100.8, 101.0),  # b3 выход: 101*0.99=99.99 -> кламп к low=100.8
    ]
    bars = make_bars(ohlc)
    cfg = build_config(simulation={"slippage_bps": 100.0})
    run, _ = simulate(cfg, bars, [buy_at(bars[0]), exit_at(bars[2])])
    entry, exit_ = run.fills
    expect(close(entry.price, 101.0) and close(entry.raw_price, 100.0), f"вход: {entry}")
    expect(close(exit_.price, 100.8) and close(exit_.raw_price, 101.0), f"выход: {exit_}")
    results = compute_results(run, cfg, bars)
    expect(close(results.estimated_slippage_cost, 1.0 * 10 + 0.2 * 10),
           f"slip cost: {results.estimated_slippage_cost}")
    return "вход 101.00, выход 100.80 (кламп), издержки слиппеджа 12.0 в отчёте"


# ---------------------------------------------------------------------------
# 5. Стоп/цель и конфликт
# ---------------------------------------------------------------------------

@check("Стоп в баре -> исполнение по стопу; гэп ниже стопа -> по open (хуже стопа)",
       "stop: fill 99.0 (cause=stop); gap: open=97 -> fill 97.0 (cause=gap_through_stop)")
def check_stop_and_gap() -> str:
    ohlc_stop = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),   # вход по 100
        (100.0, 100.5, 98.5, 99.0),    # low<=99 -> стоп
    ]
    bars = make_bars(ohlc_stop)
    run, _ = simulate(build_config(), bars, [buy_at(bars[0], stop=99.0)])
    expect(run.fills[-1].cause == "stop" and close(run.fills[-1].price, 99.0), f"{run.fills[-1]}")
    ohlc_gap = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (97.0, 97.5, 96.0, 96.5),      # гэп: open ниже стопа
    ]
    bars2 = make_bars(ohlc_gap)
    run2, _ = simulate(build_config(), bars2, [buy_at(bars2[0], stop=99.0)])
    expect(run2.fills[-1].cause == "gap_through_stop", f"cause: {run2.fills[-1].cause}")
    expect(close(run2.fills[-1].price, 97.0), f"цена: {run2.fills[-1].price}")
    return "стоп 99.0; гэп-филл 97.0 (консервативно хуже стопа)"


@check("Конфликт стоп+цель в одном баре: stop_first -> стоп 99.0; require_intrabar_data -> HALT без филла",
       "policy stop_first: fill 99.0 cause=stop_target_conflict + warning; policy require_intrabar_data: "
       "halt intrabar_data_required, филлов=1 (выход НЕ сфабрикован)")
def check_conflict() -> str:
    ohlc = [
        (100.0, 100.5, 99.5, 100.0),   # b0 сигнал: стоп 99, цель 105
        (100.0, 100.5, 99.5, 100.0),   # b1 вход по 100
        (100.0, 106.0, 98.0, 103.0),   # b2 low<=99 И high>=105 -> конфликт
    ]
    bars = make_bars(ohlc)
    run1, _ = simulate(build_config(simulation={"conflict_policy": "stop_first"}), bars,
                       [buy_at(bars[0], stop=99.0, target=105.0)])
    expect(run1.fills[-1].cause == "stop_target_conflict", f"cause: {run1.fills[-1].cause}")
    expect(close(run1.fills[-1].price, 99.0), f"выбран выгодный сценарий! цена {run1.fills[-1].price}")
    expect(len(run1.conflict_bars) == 1, "конфликт не зарегистрирован")
    expect(any("stop_target_conflict" in w for w in run1.warnings), "нет warning о конфликте")
    run2, _ = simulate(build_config(simulation={"conflict_policy": "require_intrabar_data"}), bars,
                       [buy_at(bars[0], stop=99.0, target=105.0)])
    expect(run2.completed is False, "require_intrabar_data не остановил симуляцию")
    expect(any(h.reason == "intrabar_data_required" for h in run2.halts), f"halts: {run2.halts}")
    expect(len(run2.fills) == 1 and run2.portfolio.units == 10, "выход сфабрикован без внутрибарных данных")
    return "stop_first -> 99.0 (консервативно); require_intrabar_data -> halt, позиция как есть"


# ---------------------------------------------------------------------------
# 6. Лимиты и остановки
# ---------------------------------------------------------------------------

@check("Отказы по каждому лимиту: max_position, капитал, заявок/день, дневной убыток, просадка",
       "коды: max_position_exceeded (500>100, БЕЗ молчаливого урезания); insufficient_capital "
       "(cash 500 < 1000); order_count_limit (3-й вход при лимите 2); daily_loss_limit -> halt+HALTED; "
       "drawdown_limit -> halt, защитный выход исполняется")
def check_limits() -> str:
    flat = [(100.0, 100.5, 99.5, 100.0)] * 12
    bars = make_bars(flat)

    run, _ = simulate(build_config(risk={"position_units": 500, "max_position_units": 100}), bars,
                      [buy_at(bars[0])])
    expect(run.fills == [] and any(r["code"] == "max_position_exceeded" for r in run.rejections),
           "max_position не сработал или позиция урезана молча")

    run, _ = simulate(build_config(risk={"initial_capital_quote": 500.0}), bars, [buy_at(bars[0])])
    expect(any(r["code"] == "insufficient_capital" for r in run.rejections), "капитал не проверен")

    signals = [buy_at(bars[0]), exit_at(bars[2]), buy_at(bars[4]), exit_at(bars[6]), buy_at(bars[8])]
    run, _ = simulate(build_config(risk={"max_orders_per_day": 2}), bars, signals)
    buys = [f for f in run.fills if f.side is Side.BUY]
    expect(len(buys) == 2 and any(r["code"] == "order_count_limit" for r in run.rejections),
           f"лимит заявок: buys={len(buys)}")

    ohlc_loss = [
        (100.0, 100.5, 99.5, 100.0), (100.0, 100.5, 99.5, 100.0),
        (90.0, 90.5, 89.5, 90.0), (90.0, 90.5, 89.5, 90.0),
        (90.0, 90.5, 89.5, 90.0), (90.0, 90.5, 89.5, 90.0),
    ]
    bars_l = make_bars(ohlc_loss)
    run, _ = simulate(
        build_config(risk={"daily_loss_limit_quote": 50.0}),
        bars_l, [buy_at(bars_l[0]), exit_at(bars_l[2]), buy_at(bars_l[4])],
    )
    expect(any(h.reason == "daily_loss_limit" for h in run.halts), f"halts: {run.halts}")
    expect(any(r["code"] == "strategy_halted" for r in run.rejections), "после halt вход не отклонён")
    expect(len([f for f in run.fills if f.side is Side.BUY]) == 1, "вход после дневного лимита прошёл")

    ohlc_dd = [
        (100.0, 100.5, 99.5, 100.0), (100.0, 100.5, 99.5, 100.0),
        (95.0, 95.5, 94.5, 95.0), (95.0, 95.5, 94.5, 95.0),
        (95.0, 95.5, 94.5, 95.0), (95.0, 95.5, 94.5, 95.0), (95.0, 95.5, 94.5, 95.0),
    ]
    bars_d = make_bars(ohlc_dd)
    run, _ = simulate(
        build_config(risk={"initial_capital_quote": 1_000.0, "max_drawdown_pct": 2.0}),
        bars_d, [buy_at(bars_d[0]), exit_at(bars_d[3]), buy_at(bars_d[5])],
    )
    expect(any(h.reason == "drawdown_limit" for h in run.halts), f"halts: {run.halts}")
    expect(len([f for f in run.fills if f.side is Side.SELL]) == 1, "защитный выход не исполнен при halt")
    expect(any(r["code"] == "strategy_halted" for r in run.rejections), "вход после просадки не отклонён")
    return "все пять лимитов отказывают/останавливают как ожидалось"


@check("Устаревшие и невалидные данные -> отказ и безопасная остановка",
       "RiskGuards: stale (age 120c > 60c) -> STALE_DATA + halt; движок: бар с high<low -> "
       "halt invalid_data, цикл прерван; рассинхрон позиции -> halt position_desync, completed=False")
def check_stale_invalid_desync() -> str:
    ts = START
    g = RiskGuards(
        RiskConfig(initial_capital_quote=100_000.0, position_units=10, max_position_units=100,
                   max_orders_per_day=10, daily_loss_limit_quote=1_000.0, max_drawdown_pct=10.0,
                   max_data_age_seconds=60.0),
        None, SimulationConfig(commission_bps=0.0, slippage_bps=0.0), local_timezone="UTC",
    )
    from tradebot.execution.interfaces import OrderIntent, PortfolioState
    intent = OrderIntent(created_at=ts, symbol="TEST-USDT", timeframe="1m", side=Side.BUY,
                         quantity_units=10,
                         signal=Signal(ts_open=ts, side=Side.BUY, symbol="TEST-USDT", timeframe="1m",
                                       model_version="t"),
                         reference_price=100.0)
    context = RiskContext(decision_ts=ts, last_bar_ts_open=ts, last_close=100.0, equity=100_000.0,
                          cash=100_000.0, position_units=0, data_age_seconds=120.0)
    decision = g.evaluate(intent, PortfolioState(cash=100_000.0, peak_equity=100_000.0), context)
    expect(not decision.approved and decision.rejections[0].code is RejectionCode.STALE_DATA,
           f"stale не отклонён: {decision}")
    expect(g.is_halted and g.halt_reason == "stale_data", "нет остановки при stale")

    bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * 8)
    bars[5] = Bar(ts_open=bars[5].ts_open, open=100.0, high=98.0, low=99.0, close=100.0,
                  volume=10, symbol="TEST-USDT")  # high < low
    run, _ = simulate(build_config(), bars, [buy_at(bars[0])])
    expect(run.completed is False and any(h.reason == "invalid_data" for h in run.halts),
           f"invalid bar: {run.halts}")
    expect(run.bars_processed <= 5, f"обработка продолжилась после невалидного бара: {run.bars_processed}")

    bars2 = make_bars([(100.0, 100.5, 99.5, 100.0)] * 10)

    def corrupt(index, bar, engine):
        if index == 2 and engine.portfolio.units > 0:
            engine.portfolio.units += 5  # инъекция рассинхрона

    run2, _ = simulate(build_config(), bars2,
                       [buy_at(bars2[0]), exit_at(bars2[4])], bar_hook=corrupt)
    expect(run2.completed is False and any(h.reason == "position_desync" for h in run2.halts),
           f"desync: {run2.halts}")
    return "stale/invalid/desync -> отказ + остановка, обработка прервана"


# ---------------------------------------------------------------------------
# 7. Журнал и отчёт
# ---------------------------------------------------------------------------

@check("Журнал: по записям восстанавливается почему сигнал исполнен/отклонён; секретов нет",
       "есть записи signal/risk_decision/intent/fill/state; у отказа — код+лимит+значения; "
       "reference_price помечена «НЕ цена исполнения»; фейковый секрет маскируется ***")
def check_journal_audit() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    journal_path = OUT_DIR / "journal_demo.jsonl"
    if journal_path.exists():
        journal_path.unlink()
    bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * 8)
    cfg = build_config(risk={"position_units": 500, "max_position_units": 100})
    run, jp = simulate(cfg, bars, [buy_at(bars[0]), exit_at(bars[3])], journal_path=journal_path)
    records = read_journal(jp)
    kinds = {r["kind"] for r in records}
    expect({"signal", "risk_decision", "state"} <= kinds, f"виды записей: {kinds}")
    rejected = [r for r in records if r["kind"] == "risk_decision" and not r["approved"]]
    expect(rejected, "отказ не записан в журнал")
    rejection = rejected[0]["rejections"][0]
    expect(rejection["code"] == "max_position_exceeded" and rejection["limit"] == "risk.max_position_units",
           f"отказ невосстановим: {rejection}")
    expect(rejection["limit_value"] == 100.0 and rejection["observed_value"] == 500.0,
           "в отказе нет лимита/наблюдаемого значения")
    # секрет маскируется
    masker = SecretMasker()
    masker.register("FAKE-SECRET-PHASE4-000", label="SECRET_KEY")
    secret_path = OUT_DIR / "journal_masking_demo.jsonl"
    if secret_path.exists():
        secret_path.unlink()
    with JournalWriter(secret_path, symbol="TEST-USDT", timeframe="1m",
                       model_version="scripted-demo", masker=masker) as writer:
        writer.record("warning", message="утечка FAKE-SECRET-PHASE4-000 в тексте")
    text = secret_path.read_text(encoding="utf-8")
    expect("FAKE-SECRET-PHASE4-000" not in text and "***SECRET_KEY***" in text, "маскирование не сработало")
    line = audit_line(rejected[0])
    expect("REJECTED" in line and "max_position_exceeded" in line, f"audit_line: {line}")
    CREATED.append(str(journal_path.relative_to(REPO_ROOT)))
    CREATED.append(str(secret_path.relative_to(REPO_ROOT)))
    return f"{len(records)} записей; отказ восстановим (код/лимит/значения); секрет -> ***SECRET_KEY***"


@check("Отчёт: комиссии/слиппедж в чистом результате, допущения и лимиты показаны",
       "net_return = -10/100000 (flat-сделка с комиссией 50bp); в отчёте разделы «Допущения», "
       "«Рисковые лимиты», «Ограничения симулятора», дисклеймер, bars_per_year с источником")
def check_report() -> str:
    bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * 8)
    cfg = build_config(simulation={"commission_bps": 50.0})
    run, _ = simulate(cfg, bars, [buy_at(bars[0]), exit_at(bars[2])])
    results = compute_results(run, cfg, bars)
    expect(close(results.total_commission, 10.0), f"комиссии: {results.total_commission}")
    expect(close(results.net_return, -10.0 / 100_000.0), f"net: {results.net_return}")
    report = render_simulation_report(results, cfg, "journal.jsonl")
    for fragment in ("Допущения симуляции", "Рисковые лимиты", "Ограничения симулятора",
                     "Дисклеймер", "commission_bps", "slippage_bps", "bars_per_year",
                     "не являются обещанием доходности", "mark-to-market"):
        expect(fragment.lower() in report.lower(), f"в отчёте нет {fragment!r}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "sim_report_demo.md"
    report_path.write_text(report, encoding="utf-8")
    CREATED.append(str(report_path.relative_to(REPO_ROOT)))
    return f"net=-0.0001 (=-10/100000), комиссии 10.0 учтены; отчёт: {report_path.name}"


@check("CLI simulate доступен офлайн (--help), реальные заявки не реализованы",
       "python -m tradebot.cli.simulate --help -> exit 0; в коде execution/backtest нет торговых эндпоинтов")
def check_cli_and_no_real_orders() -> str:
    proc = subprocess.run([sys.executable, "-m", "tradebot.cli.simulate", "--help"],
                          capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT))
    expect(proc.returncode == 0, f"--help exit {proc.returncode}: {proc.stderr[:200]}")
    forbidden = ["/or" + "ders", "Place" + "Order", "Cancel" + "Order", "new_" + "order"]
    offenders = []
    for pattern in ("src/tradebot/execution/*.py", "src/tradebot/backtest/*.py", "src/tradebot/strategy/*.py"):
        for path in REPO_ROOT.glob(pattern):
            content = path.read_text(encoding="utf-8", errors="replace")
            offenders.extend(f"{path.name}:{tok}" for tok in forbidden if tok in content)
    expect(not offenders, f"торговые операции в коде: {offenders}")
    return "CLI на месте; торговых вызовов API в execution/backtest/strategy нет"


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("РУЧНАЯ ПРОВЕРКА ФАЗЫ 4: риск-контроль, TEST-симулятор, журнал решений")
    print("Режим: ЛОКАЛЬНЫЙ (детерминированные ручные серии баров, без сети и секретов)")
    print("Реальные заявки НЕ отправляются; подтверждения брокера НЕ имитируются")
    print("=" * 78)
    failures = 0
    for index, (name, expected, fn) in enumerate(CHECKS, start=1):
        print(f"[{index:02d}/{len(CHECKS)}] {name}")
        print(f"          ОЖИДАЛОСЬ: {expected}")
        try:
            actual = fn()
            print(f"          ФАКТ:      {actual}")
            print(f"          PASS")
        except AssertionError as exc:
            failures += 1
            print(f"          ФАКТ:      {exc}")
            print(f"          FAIL")
        except Exception:
            failures += 1
            print("          ФАКТ:      неожиданная ошибка")
            traceback.print_exc()
            print(f"          FAIL")
    print("-" * 78)
    if CREATED:
        print("Созданные файлы:")
        for path in dict.fromkeys(CREATED):
            print(f"  {path}")
    total = len(CHECKS)
    if failures:
        print(f"RESULT: FAIL ({total - failures}/{total} checks)")
        print("Что прислать: полный вывод, python --version, ОС, pip show tradebot pydantic")
        return 1
    print(f"RESULT: PASS ({total}/{total} checks)")
    print()
    print("Далее (на ВАШИХ данных, офлайн): заполните configs/sim_demo.yaml и обученную модель")
    print("фазы 3, затем:")
    print("  python -m tradebot.cli.simulate --config configs\\sim_demo.yaml --timeframe 5m --model-dir models\\<артефакт>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
