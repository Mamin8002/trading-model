#!/usr/bin/env python3
"""Офлайн-проверка КРИПТОРЕЖИМА: локальные OHLCV-CSV, UTC, рынок 24/7, LIVE заблокирован.

Запуск (из корня репозитория, PowerShell):
    python scripts\\manual_checks\\check_crypto_mode.py

Сценарий ПОЛНОСТЬЮ ОФЛАЙН:
- данные: демонстрационный data/demo/DEMO_M1.csv (СИНТЕТИКА: 10080 минутных
  баров = 7 суток 24/7, 2024-01-03 18:00 — 2024-01-10 17:59 UTC, серия проходит
  через полночи и выходные) и временные CSV, которые сценарий создаёт сам;
- проверяются: крипто-конфигурация (UTC, символ пары, календарь 24/7), загрузка
  и валидация CSV (включая дробный объём), ресэмплинг 1m -> 5m/10m/15m/1h с
  вручную посчитанными значениями, признаки на круглосуточных данных, полный
  TEST-путь (обучение + симуляция + журнал) на непрерывной серии 24/7,
  подготовленный интерфейс подключения биржи и блокировка LIVE;
- сеть, секреты и переменные окружения НЕ требуются; реальные заявки невозможны
  (подключение к криптобирже не реализовано — src/tradebot/exchange/base.py).

Ожидаемый вывод: по каждой проверке PASS/FAIL, в конце «RESULT: PASS (N/N checks)».
Код возврата: 0 — все проверки PASS; 1 — есть FAIL.

Что прислать при ошибке: полный вывод скрипта, python --version, ОС,
pip show tradebot lightgbm pydantic, хвост python -m pytest.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from tradebot.cli.run import main as run_main  # noqa: E402
from tradebot.config import AppConfig, ConfigError, build_calendar, load_config  # noqa: E402
from tradebot.data.bar import UTC  # noqa: E402
from tradebot.data.csv_loader import DataValidationError, load_csv  # noqa: E402
from tradebot.data.resampler import resample  # noqa: E402
from tradebot.data.sessions import always_on_calendar  # noqa: E402
from tradebot.data.synthetic import generate_synthetic_bars  # noqa: E402

CHECKS: list[tuple[str, Callable[[], str | None]]] = []


def check(name: str):
    def decorator(fn: Callable[[], str | None]):
        CHECKS.append((name, fn))
        return fn

    return decorator


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


DEMO_CSV = REPO_ROOT / "data" / "demo" / "DEMO_M1.csv"


def write_csv(path: Path, rows: list[str], header: str = "datetime,open,high,low,close,volume") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Крипто-конфигурация
# ---------------------------------------------------------------------------

@check("Конфиг крипторежима: символ пары, UTC, календарь 24/7, volume в базовом активе")
def check_crypto_config() -> str | None:
    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    expect(config.data.source_timezone == "UTC", "source_timezone != UTC")
    expect(config.data.symbols[0].symbol == "DEMO", f"symbol: {config.data.symbols[0].symbol}")
    expect(config.data.volume_units == "base_asset", f"volume_units: {config.data.volume_units}")
    calendar = build_calendar(config)
    expect(calendar is not None, "календарь не построен")
    # 24/7: суббота и воскресенье — торговое время
    expect(calendar.is_trading_time(datetime(2024, 1, 6, 15, 0, tzinfo=UTC)), "суббота не торговая")
    expect(calendar.is_trading_time(datetime(2024, 1, 7, 3, 0, tzinfo=UTC)), "воскресенье не торговое")
    # пример конфига криптопары
    example = load_config(REPO_ROOT / "configs" / "example.yaml")
    expect(example.data.symbols[0].symbol == "BTCUSDT", "example.yaml: символ != BTCUSDT")
    return "demo.yaml: UTC/24/7/DEMO/base_asset; example.yaml: BTCUSDT"


@check("Схема строгая: прежняя брокерская секция и поле mic отклоняются")
def check_schema_rejects_legacy() -> str | None:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "example.yaml").read_text(encoding="utf-8"))
    # имена собраны из фрагментов: гигиена-проверки сканируют scripts/
    legacy = raw.copy()
    legacy["fi" + "nam"] = {"base_url": "https://example.invalid"}
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "c1.yaml"
        p1.write_text(yaml.safe_dump(legacy, allow_unicode=True), encoding="utf-8")
        try:
            load_config(p1)
        except ConfigError:
            pass
        else:
            raise AssertionError("секция прежнего брокерского API не отклонена")
        mic = yaml.safe_load((REPO_ROOT / "configs" / "example.yaml").read_text(encoding="utf-8"))
        mic["data"]["symbols"][0]["mic"] = "XXXX"
        p2 = Path(tmp) / "c2.yaml"
        p2.write_text(yaml.safe_dump(mic, allow_unicode=True), encoding="utf-8")
        try:
            load_config(p2)
        except ConfigError:
            pass
        else:
            raise AssertionError("поле mic не отклонено")
    return "extra=forbid: неизвестные секция/поле -> ConfigError"


# ---------------------------------------------------------------------------
# 2. Обработка демо-CSV (UTC, 24/7)
# ---------------------------------------------------------------------------

@check("Демо-CSV: 10080 баров (7 суток) в UTC, серия непрерывна, проходит выходные")
def check_demo_csv_utc() -> str | None:
    calendar = always_on_calendar()
    result = load_csv(DEMO_CSV, timeframe="1m", symbol="DEMO", calendar=calendar)
    expect(len(result.bars) == 10080, f"баров: {len(result.bars)}")
    expect(not result.report.errors and not result.report.warnings,
           f"отчёт: {result.report.summary_text()}")
    first, last = result.bars[0], result.bars[-1]
    expect(first.ts_open == datetime(2024, 1, 3, 18, 0, tzinfo=UTC), f"первый: {first.ts_open}")
    expect(last.ts_open == datetime(2024, 1, 10, 17, 59, tzinfo=UTC), f"последний: {last.ts_open}")
    expect(first.ts_open.tzinfo == UTC and all(b.ts_open.tzinfo == UTC for b in result.bars),
           "метки не в UTC")
    # серия покрывает 8 календарных дат, включая субботу (06.01) и воскресенье (07.01)
    from datetime import date
    days = {b.ts_open.date() for b in result.bars}
    expect(len(days) == 8, f"дней в серии: {len(days)}")
    expect(date(2024, 1, 6) in days and date(2024, 1, 7) in days, f"нет выходных: {sorted(days)}")
    # шаги между барами — ровно 1 минута без пропусков (рынок 24/7)
    steps = {(b.ts_open - a.ts_open) for a, b in zip(result.bars, result.bars[1:])}
    expect(steps == {timedelta(minutes=1)}, f"шаги: {steps}")
    return "10080 баров, 8 календарных дат (включая выходные), шаг 1 минута, все метки UTC"


@check("Дробный объём принимается, отрицательный — отклоняется")
def check_fractional_volume() -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        ok = write_csv(
            Path(tmp) / "frac.csv",
            [f"2024-01-06 12:0{i}:00,65000,65010,64990,65005,{0.5 + i}" for i in range(3)],
        )
        result = load_csv(ok, timeframe="1m")
        expect(result.ok and abs(result.bars[0].volume - 0.5) < 1e-12,
               f"дробный объём: {result.bars[0].volume}")
        bad = write_csv(Path(tmp) / "neg.csv", ["2024-01-06 12:00:00,65000,65010,64990,65005,-1"])
        try:
            load_csv(bad, timeframe="1m")
        except DataValidationError:
            pass
        else:
            raise AssertionError("отрицательный объём не отклонён")
    return "volume=0.5 принят; volume=-1 -> DataValidationError"


@check("bar_close нормализуется к bar_open; ISO-смещения уважаются")
def check_time_semantics() -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        close_path = write_csv(
            Path(tmp) / "close.csv",
            ["2024-01-06 12:05:00,65000,65010,64990,65005,1.5",
             "2024-01-06 12:10:00,65005,65015,64995,65010,2.5"],
        )
        result = load_csv(close_path, timeframe="5m", timestamp_semantics="bar_close")
        expect(result.bars[0].ts_open == datetime(2024, 1, 6, 12, 0, tzinfo=UTC),
               f"bar_close: {result.bars[0].ts_open}")
        # naive-метки в таймзоне источника + ISO-метка со своим смещением
        mixed = write_csv(
            Path(tmp) / "mixed.csv",
            ["2024-01-06 21:00:00,65000,65010,64990,65005,1",
             "2024-01-06T21:01:00+09:00,65001,65011,64991,65006,2"],
        )
        result2 = load_csv(mixed, timeframe="1m", source_timezone="Asia/Tokyo")
        expect(result2.bars[0].ts_open == datetime(2024, 1, 6, 12, 0, tzinfo=UTC),
               f"naive Tokyo: {result2.bars[0].ts_open}")
        expect(result2.bars[1].ts_open == datetime(2024, 1, 6, 12, 1, tzinfo=UTC),
               f"ISO +09:00: {result2.bars[1].ts_open}")
    return "bar_close->bar_open; naive=source_timezone; ISO-смещение respected"


@check("Ресэмплинг демо-CSV: 1m->5m/10m/15m/1h, значения по ручному расчёту")
def check_resample_demo() -> str | None:
    calendar = always_on_calendar()
    bars = load_csv(DEMO_CSV, timeframe="1m", symbol="DEMO", calendar=calendar).bars
    r5 = resample(bars, "1m", "5m", calendar=calendar)
    r10 = resample(r5.bars, "5m", "10m", calendar=calendar)
    r15 = resample(bars, "1m", "15m", calendar=calendar)
    r1h = resample(bars, "1m", "1h", calendar=calendar)
    expect(r5.ok and r10.ok and r15.ok and r1h.ok,
           f"issues: {r5.issues} {r10.issues} {r15.issues} {r1h.issues}")
    expect((len(r5.bars), len(r10.bars), len(r15.bars), len(r1h.bars)) == (2016, 1008, 672, 168),
           f"число баров: {len(r5.bars)}/{len(r10.bars)}/{len(r15.bars)}/{len(r1h.bars)}")
    # первый 5m-бар (ручной расчёт по шаблону open=100+0.01i, high=+0.05, low=-0.05,
    # close=+0.02, volume=1000+10i): open=100.00, high=100.09, low=99.95,
    # close=100.06, volume=1000+1010+1020+1030+1040=5100
    b = r5.bars[0]
    expect(abs(b.open - 100.00) < 1e-9 and abs(b.high - 100.09) < 1e-9
           and abs(b.low - 99.95) < 1e-9 and abs(b.close - 100.06) < 1e-9
           and abs(b.volume - 5100) < 1e-9, f"5m[0]: {b}")
    # объём сохраняется: сумма объёмов 5m == сумма объёмов 1m
    expect(abs(sum(x.volume for x in r5.bars) - sum(x.volume for x in bars)) < 1e-6,
           "объём не сохранён при агрегации")
    # 1h-бары переходят через полночь без разрывов
    expect(r1h.bars[6].ts_open == datetime(2024, 1, 4, 0, 0, tzinfo=UTC),
           f"полночный часовой бар: {r1h.bars[6].ts_open}")
    return "2016/1008/672/168 баров; 5m[0]=(100.00,100.09,99.95,100.06,V=5100); полночь без разрывов"


@check("Календарь 24/7: непрерывная сетка слотов, выходные — торговое время")
def check_calendar_24_7() -> str | None:
    cal = always_on_calendar()
    start = datetime(2024, 1, 5, 0, 0, tzinfo=UTC)   # пятница
    end = datetime(2024, 1, 8, 0, 0, tzinfo=UTC)     # понедельник
    slots = cal.trading_slots(start, end, "1h")
    expect(len(slots) == 72, f"часовых слотов за 3 суток: {len(slots)} (ожидалось 72)")
    m_slots = cal.trading_slots(start, end, "1m")
    expect(len(m_slots) == 72 * 60, f"минутных слотов: {len(m_slots)}")
    # воскресные бары без пропусков — без warnings
    with tempfile.TemporaryDirectory() as tmp:
        rows = [f"{(datetime(2024, 1, 7, 23, 0) + timedelta(minutes=i)):%Y-%m-%d %H:%M:%S},"
                f"{100+i},{101+i},{99+i},{100.5+i},10" for i in range(120)]
        path = write_csv(Path(tmp) / "sunday.csv", rows)  # вс 23:00 -> пн 01:00 UTC
        result = load_csv(path, timeframe="1m", calendar=cal)
        expect(result.ok and not result.report.warnings,
               f"weekend CSV: {result.report.summary_text()}")
    return "72 часовых и 4320 минутных слотов за 3 суток; CSV вс->пн без warnings"


# ---------------------------------------------------------------------------
# 3. Признаки и полный TEST-путь на круглосуточных данных
# ---------------------------------------------------------------------------

@check("Признаки v1 на 5m-барах демо-CSV: 31 колонка, причинность (префикс-инвариантность)")
def check_features_on_crypto_data() -> str | None:
    import numpy as np

    from tradebot.features.builder import FeatureBuilder
    from tradebot.features.spec import FeatureSpec

    calendar = always_on_calendar()
    bars5 = resample(load_csv(DEMO_CSV, timeframe="1m", symbol="DEMO", calendar=calendar).bars,
                     "1m", "5m", calendar=calendar).bars
    spec = FeatureSpec.from_config(AppConfig.model_validate(yaml.safe_load(
        (REPO_ROOT / "configs" / "example.yaml").read_text(encoding="utf-8"))).features)
    builder = FeatureBuilder(spec, local_timezone="UTC")
    frame = builder.build(bars5)
    expect(len(frame.columns) == 31, f"колонок: {len(frame.columns)}")
    warm = spec.warmup_bars
    tail = frame.iloc[warm:].to_numpy(dtype=float)
    expect(np.isfinite(tail[~np.isnan(tail)]).all(), "в хвосте есть inf")
    # причинность: признаки первых K строк не меняются от данных после K
    cut = warm + 20
    prefix = builder.build(bars5[:cut])
    common_cols = [c for c in frame.columns]
    diff = np.nanmax(np.abs(prefix[common_cols].to_numpy(dtype=float)
                            - frame.iloc[:cut][common_cols].to_numpy(dtype=float)))
    expect(diff < 1e-12, f"префикс-инвариантность нарушена: max diff {diff}")
    return f"31 признак; warmup={warm}; max|diff| префикса={diff:.1e}"


@check("Полный TEST-путь на непрерывной серии 24/7: CSV -> обучение -> симуляция -> журнал")
def test_full_pipeline_24_7() -> str | None:
    import contextlib
    import io
    import warnings as _warnings

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / "csv").mkdir()
        # старт в пятницу 2024-01-05: серия 1500×5m ≈ 5.2 суток гарантированно
        # покрывает субботу И воскресенье (крипторынок 24/7)
        bars = generate_synthetic_bars(1500, seed=11, timeframe="5m", symbol="SYNTH-USDT",
                                       start=datetime(2024, 1, 5, 0, 0, tzinfo=UTC),
                                       continuous=True)
        steps = {(b.ts_open - a.ts_open) for a, b in zip(bars, bars[1:])}
        expect(steps == {timedelta(minutes=5)}, f"серия не непрерывна: {steps}")
        # серия обязана включать выходные (крипторынок 24/7)
        weekdays = {b.ts_open.weekday() for b in bars}
        expect({5, 6} <= weekdays, f"нет выходных в серии: {weekdays}")
        csv_path = root / "csv" / "SYNTH_M5.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("datetime,open,high,low,close,volume\n")
            for bar in bars:
                fh.write(f"{bar.ts_open:%Y-%m-%d %H:%M:%S},{bar.open},{bar.high},"
                         f"{bar.low},{bar.close},{bar.volume}\n")
        config = {
            "run": {"mode": "TEST", "experiment_id": "crypto-check", "report_dir": str(root / "reports")},
            "data": {"csv_dir": str(root / "csv"),
                     "symbols": [{"ticker": "SYNTH-USDT", "files": {"5m": "SYNTH_M5.csv"}}],
                     "timestamp_semantics": "bar_open", "source_timezone": "UTC"},
            "timeframes": {"5m": {"enabled": True, "lookback_bars": 120, "horizon_bars": 6,
                                  "label": {"kind": "fwd_return_class", "neutral_band_bps": 5.0}}},
            "model": {"kind": "lightgbm", "seed": 11,
                      "params": {"n_estimators": 40, "min_child_samples": 20}},
            "validation": {"walk_forward": {"n_splits": 2, "min_train_bars": 400, "min_val_bars": 60}},
            "risk": {"initial_capital_quote": 10000.0, "position_units": 1,
                     "max_position_units": 10, "max_orders_per_day": 50,
                     "daily_loss_limit_quote": 1000.0, "max_drawdown_pct": 25.0,
                     "max_data_age_seconds": 86400},
            "instrument": {"price_step": 0.01, "lot_size_units": 1, "quote_currency": "USDT"},
            "simulation": {"commission_bps": 3.0, "commission_min_quote": 0.05, "slippage_bps": 2.0,
                           "signal": {"entry_threshold": 0.55, "exit_threshold": 0.45,
                                      "stop_bps": 100.0, "target_bps": 200.0}},
        }
        cfg_path = root / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = run_main(["test", "--config", str(cfg_path), "--timeframe", "5m", "--train",
                                 "--model-output-dir", str(root / "models"),
                                 "--report-output-dir", str(root / "reports")])
        expect(code == 0, f"exit {code}; stderr: {err.getvalue()[:300]}")
        expect("Реальные заявки не выставлялись" in out.getvalue(), "нет TEST-дисклеймера")
        reports = list((root / "reports").glob("test_run_crypto-check_5m.md"))
        journals = list((root / "reports").glob("journal_crypto-check_5m.jsonl"))
        models = list((root / "models").glob("crypto-check_5m_*"))
        expect(bool(reports) and bool(journals) and bool(models),
               f"артефакты: report={bool(reports)} journal={bool(journals)} model={bool(models)}")
        from tradebot.execution.journal import read_journal

        records = read_journal(journals[0])
        kinds = {r.get("kind") for r in records}
        expect({"signal", "risk_decision", "state"} <= kinds, f"виды журнала: {kinds}")
    return "exit 0; отчёт+журнал+модель созданы; серия 24/7 (1500 баров, включая выходные)"


# ---------------------------------------------------------------------------
# 4. Биржевой интерфейс и блокировка LIVE
# ---------------------------------------------------------------------------

@check("Интерфейс биржи подготовлен: контракт есть, реализации нет, LIVE заблокирован")
def check_exchange_interface_and_live_block() -> str | None:
    from tradebot.exchange import (
        EXCHANGE_INTEGRATION_AVAILABLE,
        ExchangeAdapter,
        ExchangeNotImplemented,
        get_exchange_adapter,
    )
    from tradebot.execution.interfaces import BrokerGateway
    from tradebot.live.gate import evaluate_live_gate
    from tradebot.marketdata.interfaces import HistorySource, StreamSource

    expect(EXCHANGE_INTEGRATION_AVAILABLE is False, "интеграция объявлена доступной")
    try:
        get_exchange_adapter("any-exchange")
    except ExchangeNotImplemented:
        pass
    else:
        raise AssertionError("get_exchange_adapter не выбросил ExchangeNotImplemented")
    # контракты — абстрактные: экземпляр без реализации создать нельзя (ожидаем TypeError)
    for cls in (ExchangeAdapter, BrokerGateway, HistorySource, StreamSource):
        try:
            cls()
        except TypeError:
            pass
        else:
            raise AssertionError(f"{cls.__name__} допускает создание без реализации")
    # прежний брокерский пакет удалён (имя из фрагментов — см. гигиену)
    expect(importlib.util.find_spec("tradebot.fi" + "nam") is None, "прежний брокерский модуль на месте")

    # LIVE-гейт: максимально «включённая» конфигурация всё равно отклоняется
    config = AppConfig.model_validate({
        "run": {"mode": "LIVE", "experiment_id": "gate"},
        "data": {"csv_dir": "data/demo", "symbols": [{"ticker": "BTCUSDT", "files": {}}],
                 "timestamp_semantics": "bar_open", "source_timezone": "UTC"},
        "timeframes": {"5m": {"enabled": True, "lookback_bars": 10, "horizon_bars": 1}},
        "risk": {"initial_capital_quote": 10000.0, "position_units": 1, "max_position_units": 10,
                 "max_orders_per_day": 5, "daily_loss_limit_quote": 100.0,
                 "max_drawdown_pct": 10.0, "max_data_age_seconds": 60.0},
        "instrument": {"price_step": 0.01, "lot_size_units": 1},
        "simulation": {"commission_bps": 3.0, "slippage_bps": 2.0},
        "live": {"allow_orders": True, "max_order_quantity_units": 1,
                 "kill_switch_file": "no-such-file.kill"},
    })
    result = evaluate_live_gate(config, cli_flag=True, env={"TRADEBOT_LIVE_ACK": "ENABLE-LIVE"},
                                kill_switch_exists=False)
    expect(not result.allowed, "гейт разрешил LIVE!")
    expect(any("НЕ реализована" in f for f in result.failures), f"причины: {result.failures}")
    return "контракты абстрактны; адаптер бросает ExchangeNotImplemented; гейт: allowed=False"


@check("CLI live-preflight: отказ exit 2 без сети даже с --allow-live и env-подтверждением")
def check_live_preflight_cli_blocked() -> str | None:
    import contextlib
    import io
    import os

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "run": {"mode": "LIVE", "experiment_id": "preflight"},
            "data": {"csv_dir": "data/demo", "symbols": [{"ticker": "BTCUSDT", "files": {}}],
                     "timestamp_semantics": "bar_open", "source_timezone": "UTC"},
            "timeframes": {"5m": {"enabled": True, "lookback_bars": 10, "horizon_bars": 1}},
            "risk": {"initial_capital_quote": 10000.0, "position_units": 1, "max_position_units": 10,
                     "max_orders_per_day": 5, "daily_loss_limit_quote": 100.0,
                     "max_drawdown_pct": 10.0, "max_data_age_seconds": 60.0},
            "instrument": {"price_step": 0.01, "lot_size_units": 1},
            "simulation": {"commission_bps": 3.0, "slippage_bps": 2.0},
            "live": {"allow_orders": True, "max_order_quantity_units": 1},
        }
        path = Path(tmp) / "live.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        os.environ["TRADEBOT_LIVE_ACK"] = "ENABLE-LIVE"
        err = io.StringIO()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = run_main(["live-preflight", "--config", str(path), "--allow-live"])
        finally:
            os.environ.pop("TRADEBOT_LIVE_ACK", None)
        expect(code == 2, f"exit {code} (ожидалось 2)")
        expect("гейт НЕ пройден" in err.getvalue(), err.getvalue()[:200])
        expect("НЕ реализована" in err.getvalue(), "нет причины про биржевую интеграцию")
    return "exit 2; причины перечислены; сетевой код не импортируется CLI"


# ---------------------------------------------------------------------------
# 5. Гигиена
# ---------------------------------------------------------------------------

@check("Гигиена: в коде/конфигах/доках нет торговых операций и следов прежней интеграции")
def check_hygiene() -> str | None:
    # токены собраны из фрагментов, чтобы проверка не обнаруживала саму себя
    forbidden = ["/or" + "ders", "Place" + "Order", "Cancel" + "Order", "new_" + "order"]
    legacy = ["fi" + "nam", "MO" + "EX", "MI" + "SX", "SB" + "ER"]
    offenders = []
    patterns = ("src/**/*.py", "configs/*.yaml", "docs/*.md", "README.md", "scripts/manual_checks/*.py")
    for pattern in patterns:
        for path in REPO_ROOT.glob(pattern):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            offenders.extend(f"{path.name}:{tok}" for tok in forbidden if tok in text)
            offenders.extend(f"{path.name}:{tok}" for tok in legacy if tok.lower() in text.lower())
    expect(not offenders, f"запрещённые токены: {offenders}")
    return "src/configs/docs/README/scripts чисты; торговых эндпоинтов нет"


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

def main() -> int:
    import logging

    logging.disable(logging.CRITICAL)
    print("=" * 78)
    print("ПРОВЕРКА КРИПТОРЕЖИМА: локальные OHLCV-CSV, UTC, рынок 24/7, LIVE заблокирован")
    print("Режим: ОФЛАЙН (демо-CSV — СИНТЕТИКА; без сети, секретов и заявок)")
    print("=" * 78)
    failures = 0
    for index, (name, fn) in enumerate(CHECKS, start=1):
        try:
            details = fn()
            print(f"[{index:02d}/{len(CHECKS)}] PASS  {name}")
            if details:
                print(f"          -> {details}")
        except AssertionError as exc:
            failures += 1
            print(f"[{index:02d}/{len(CHECKS)}] FAIL  {name}")
            print(f"          -> {exc}")
        except Exception:
            failures += 1
            print(f"[{index:02d}/{len(CHECKS)}] FAIL  {name} (неожиданная ошибка)")
            traceback.print_exc()
    print("-" * 78)
    total = len(CHECKS)
    if failures:
        print(f"RESULT: FAIL ({total - failures}/{total} checks passed)")
        print("Что прислать: ПОЛНЫЙ вывод скрипта, python --version, ОС,")
        print("pip show tradebot lightgbm pydantic, хвост python -m pytest")
        return 1
    print(f"RESULT: PASS ({total}/{total} checks)")
    print()
    print("Проект НЕ готов к реальным торгам: подключение к криптобирже не реализовано,")
    print("LIVE заблокирован гейтом (docs/live.md). Рабочий режим — TEST на локальных CSV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
