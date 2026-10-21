#!/usr/bin/env python3
"""Ручная проверка слоя данных (крипторежим): конфиг, CSV-валидация, агрегация таймфреймов.

Запуск (из корня репозитория, PowerShell):
    python scripts\\manual_checks\\check_phase_02_data.py

Сценарий ПОЛНОСТЬЮ ЛОКАЛЬНЫЙ:
- не требует сети, секретов и переменных окружения;
- демонстрационный CSV — СИНТЕТИЧЕСКИЕ данные с детерминированным шаблоном
  (data/demo/DEMO_M1.csv: 7 суток непрерывных минутных баров 24/7, явно помечены),
  ожидаемые результаты агрегации посчитаны вручную;
- дополнительно проверяются крипто-соглашения: метки времени UTC, календарь 24/7,
  дробный объём, блокировка LIVE (биржевой адаптер не реализован).

Дополнительные офлайн-команды на демо-данных:
    python -m tradebot.cli.validate_data --config configs\\demo.yaml
    python -m tradebot.cli.validate_data --config configs\\demo.yaml --resample 10m

Код возврата: 0 — все проверки PASS; 1 — есть FAIL.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from tradebot.config import ConfigError, build_calendar, config_hash, load_config  # noqa: E402
from tradebot.data.bar import UTC, Timeframe  # noqa: E402
from tradebot.data.csv_loader import DataValidationError, load_csv  # noqa: E402
from tradebot.data.resampler import ResampleIssueCode, resample  # noqa: E402
from tradebot.data.sessions import SessionCalendar, always_on_calendar  # noqa: E402
from tradebot.data.validator import IssueCode  # noqa: E402

# ---------------------------------------------------------------------------
# Инфраструктура проверки
# ---------------------------------------------------------------------------

CHECKS: list[tuple[str, Callable[[], str | None]]] = []


def check(name: str):
    def decorator(fn: Callable[[], str | None]):
        CHECKS.append((name, fn))
        return fn

    return decorator


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def demo_series(count: int, start: datetime, skip: set[int] | None = None):
    """Синтетическая серия: open_i=100+i, high=+2, low=-1, close=+1, volume=10*(i+1)."""
    from tradebot.data.bar import Bar, BarSource

    skip = skip or set()
    bars = []
    for i in range(count):
        if i in skip:
            continue
        o = 100.0 + i
        bars.append(
            Bar(
                ts_open=start + timedelta(minutes=i),
                open=o,
                high=o + 2.0,
                low=o - 1.0,
                close=o + 1.0,
                volume=10 * (i + 1),
                source=BarSource.CSV,
                symbol="DEMO",
            )
        )
    return bars


START = datetime(2024, 1, 4, 7, 0, tzinfo=UTC)  # четверг, 07:00 UTC


# ---------------------------------------------------------------------------
# 1. Конфигурация
# ---------------------------------------------------------------------------

@check("Конфиг demo.yaml загружается; lookback/horizon явные; UTC и календарь 24/7")
def check_config_demo() -> str | None:
    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    for code, tf_cfg in config.timeframes.items():
        expect(tf_cfg.lookback_bars >= 1, f"{code}: lookback_bars не задан")
        expect(tf_cfg.horizon_bars >= 1, f"{code}: horizon_bars не задан")
    expect(config.timeframes["10m"].base_timeframe == "5m", "10m должен строиться из 5m")
    expect(config.data.timestamp_semantics == "bar_open", "семантика метки должна быть явной")
    expect(config.data.source_timezone == "UTC", "крипторежим: source_timezone ожидается UTC")
    calendar = build_calendar(config)
    expect(calendar is not None, "демо-календарь не построен")
    expect(calendar.is_trading_time(datetime(2024, 1, 6, 23, 30, tzinfo=UTC)),
           "календарь demo.yaml не покрывает субботу (ожидается 24/7)")
    h1 = config_hash(config)
    h2 = config_hash(load_config(REPO_ROOT / "configs" / "demo.yaml"))
    expect(h1 == h2 and len(h1) == 64, "хэш конфига не детерминирован")
    return f"timeframes={sorted(config.timeframes)}, tz=UTC, календарь=24/7, hash={h1[:12]}"


@check("Конфиг example.yaml загружается; прежняя брокерская секция отклоняется схемой")
def check_config_example() -> str | None:
    config = load_config(REPO_ROOT / "configs" / "example.yaml")
    expect(config.data.symbols[0].symbol == "BTCUSDT", "symbol != BTCUSDT")
    expect(config.data.volume_units == "base_asset", "volume_units != base_asset")
    raw = yaml.safe_load((REPO_ROOT / "configs" / "example.yaml").read_text(encoding="utf-8"))
    expect("mic" not in raw["data"]["symbols"][0], "в символах не должно быть поля mic")
    # имя удалённой секции собрано из фрагментов: гигиена-проверки сканируют scripts/
    raw["fi" + "nam"] = {"base_url": "https://example.invalid"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.yaml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        try:
            load_config(path)
        except ConfigError:
            pass
        else:
            raise AssertionError("конфиг с удалённой брокерской секцией НЕ был отклонён")
    return "ok (схема строгая: неизвестные секции отклоняются)"


@check("Конфиг без horizon_bars отклоняется (значения не подбираются молча)")
def check_config_missing_horizon() -> str | None:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "demo.yaml").read_text(encoding="utf-8"))
    del raw["timeframes"]["5m"]["horizon_bars"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.yaml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        try:
            load_config(path)
        except ConfigError as exc:
            expect("horizon_bars" in str(exc), "ошибка не упоминает horizon_bars")
            return "отклонён с понятной ошибкой"
    raise AssertionError("конфиг без horizon_bars НЕ был отклонён")


@check("Неизвестный таймфрейм и лишнее поле отклоняются (extra=forbid)")
def check_config_strict() -> str | None:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "demo.yaml").read_text(encoding="utf-8"))
    raw["timeframes"]["7m"] = {"lookback_bars": 5, "horizon_bars": 1}
    raw["run"]["typo_field"] = 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.yaml"
        path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
        try:
            load_config(path)
        except ConfigError:
            return "отклонён"
    raise AssertionError("строгость схемы не сработала")


# ---------------------------------------------------------------------------
# 2. CSV: загрузка и валидация
# ---------------------------------------------------------------------------

@check("Демо-CSV (10080 синтетических 1m-баров = 7 суток 24/7, UTC)")
def check_csv_demo() -> str | None:
    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    calendar = build_calendar(config)
    result = load_csv(
        REPO_ROOT / "data" / "demo" / "DEMO_M1.csv",
        timeframe="1m",
        symbol="DEMO",
        source_timezone="UTC",
        timestamp_semantics="bar_open",
        calendar=calendar,
    )
    expect(len(result.bars) == 10080, f"ожидалось 10080 баров, получено {len(result.bars)}")
    expect(not result.report.errors, f"ошибки: {result.report.summary_text()}")
    expect(not result.report.warnings, f"предупреждения: {result.report.summary_text()}")
    first, last = result.bars[0], result.bars[-1]
    expect(first.ts_open == datetime(2024, 1, 3, 18, 0, tzinfo=UTC), f"первый бар: {first.ts_open}")
    expect(last.ts_open == datetime(2024, 1, 10, 17, 59, tzinfo=UTC), f"последний бар: {last.ts_open}")
    expect(first.open == 100.0 and first.volume == 1000, "значения первой строки не совпали")
    expect(last.volume == 101790, f"объём последней строки: {last.volume}")
    return "10080 баров 2024-01-03 18:00 .. 2024-01-10 17:59 UTC; ошибок/предупреждений нет"


@check("Семантика bar_close сдвигает метку на начало бара")
def check_csv_bar_close() -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "close_stamped.csv"
        rows = ["datetime,open,high,low,close,volume"]
        for i in range(5):
            ts = datetime(2024, 1, 4, 10, 5 * (i + 1))  # метки КОНЦА 5m-баров (UTC naive)
            rows.append(f"{ts:%Y-%m-%d %H:%M:%S},{100+i},{101+i},{99+i},{100.5+i},{10*(i+1)}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        result = load_csv(path, timeframe="5m", timestamp_semantics="bar_close", source_timezone="UTC")
        expect(len(result.bars) == 5, "ожидалось 5 баров")
        expect(
            result.bars[0].ts_open == datetime(2024, 1, 4, 10, 0, tzinfo=UTC),
            f"первый бар: {result.bars[0].ts_open} (ожидалось 10:00 UTC)",
        )
    return "нормализация bar_close -> bar_open работает"


@check("Дробный объём принимается (криптопары: объём в базовом активе)")
def check_csv_fractional_volume() -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "frac.csv"
        rows = ["datetime,open,high,low,close,volume"]
        for i in range(3):
            ts = datetime(2024, 1, 4, 10, i)
            rows.append(f"{ts:%Y-%m-%d %H:%M:%S},{65000+i},{65010+i},{64990+i},{65005+i},{0.12345 + i}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        result = load_csv(path, timeframe="1m", timestamp_semantics="bar_open")
        expect(len(result.bars) == 3, "ожидалось 3 бара")
        expect(abs(result.bars[0].volume - 0.12345) < 1e-12, f"volume: {result.bars[0].volume}")
    return "volume=0.12345 загружен без округления"


@check("Некорректные CSV дают понятные ошибки (9 негативных кейсов)")
def check_csv_negative() -> str | None:
    cases: list[tuple[str, list[str], IssueCode, str]] = [
        ("дубликат метки", ["2024-01-04 10:00:00,100,101,99,100.5,10", "2024-01-04 10:00:00,100,101,99,100.5,10"], IssueCode.E_DUPLICATE_TS, ""),
        ("не отсортировано", ["2024-01-04 10:01:00,100,101,99,100.5,10", "2024-01-04 10:00:00,100,101,99,100.5,10"], IssueCode.E_UNSORTED, ""),
        ("high < low", ["2024-01-04 10:00:00,100,98,99,97,10"], IssueCode.E_OHLC, ""),
        ("close выше high", ["2024-01-04 10:00:00,100,100.2,99,101.5,10"], IssueCode.E_OHLC, ""),
        ("нечисловая цена", ["2024-01-04 10:00:00,abc,101,99,100.5,10"], IssueCode.E_NUMBER_PARSE, ""),
        ("пустой файл", [], IssueCode.E_EMPTY_FILE, ""),
        ("нет колонки volume", ["2024-01-04 10:00:00,100,101,99,100.5"], IssueCode.E_HEADER_MISSING_COLS, "datetime,open,high,low,close"),
        ("лишняя колонка", ["2024-01-04 10:00:00,100,101,99,100.5,10,x"], IssueCode.E_HEADER_UNKNOWN_COLS, "datetime,open,high,low,close,volume,extra"),
        ("отрицательный объём", ["2024-01-04 10:00:00,100,101,99,100.5,-10"], IssueCode.E_VOLUME_INVALID, ""),
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for title, rows, expected_code, header in cases:
            if title == "пустой файл":
                path = tmp / "empty.csv"
                path.write_bytes(b"")
            else:
                path = _write_bad_csv(tmp, rows, header=header or "datetime,open,high,low,close,volume")
            try:
                load_csv(path, timeframe="1m")
            except DataValidationError as exc:
                codes = {i.code for i in exc.report.errors}
                expect(expected_code in codes, f"{title}: ожидался {expected_code.value}, получены {sorted(c.value for c in codes)}")
                expect(str(exc).strip() != "", f"{title}: пустое сообщение об ошибке")
            else:
                raise AssertionError(f"{title}: ошибка НЕ была выброшена")
    return f"{len(cases)} кейсов: дубликат/порядок/OHLC/типы/пустой файл/колонки/объём"


def _write_bad_csv(tmp: Path, rows: list[str], header: str = "datetime,open,high,low,close,volume") -> Path:
    path = tmp / "bad.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


@check("Пропуски свечей: warning W_GAP, данные не заполняются молча")
def check_csv_gaps() -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        rows = [
            "2024-01-04 10:00:00,100,101,99,100.5,10",
            "2024-01-04 10:04:00,100,101,99,100.5,10",  # пропуск 3 минут
        ]
        path = _write_bad_csv(Path(tmp), rows)
        result = load_csv(path, timeframe="1m")
        gaps = [i for i in result.report.warnings if i.code is IssueCode.W_GAP]
        expect(len(gaps) == 1, f"ожидался 1 W_GAP, получено {len(gaps)}")
        expect("3 бар(ов)" in gaps[0].message, f"сообщение: {gaps[0].message}")
        expect(len(result.bars) == 2, "бары не должны дозаполняться")
    return "W_GAP c количеством пропущенных баров; заполнение не выполняется"


# ---------------------------------------------------------------------------
# 3. Агрегация таймфреймов (известные значения, посчитаны вручную)
# ---------------------------------------------------------------------------

@check("Агрегация 1m -> 5m/10m/15m на 30 барах: значения совпадают с ручным расчётом")
def check_aggregation_small() -> str | None:
    bars = demo_series(30, START)
    r5 = resample(bars, "1m", "5m")
    r10 = resample(bars, "1m", "10m")
    r15 = resample(bars, "1m", "15m")
    expect(r5.ok and r10.ok and r15.ok, f"issues: {r5.issues} {r10.issues} {r15.issues}")
    expect((len(r5.bars), len(r10.bars), len(r15.bars)) == (6, 3, 2), "неверное число баров")
    # 5m первый бар: open=100, high=max(102..106)=106, low=99, close=105, volume=10*(1+..+5)=150
    b = r5.bars[0]
    expect((b.open, b.high, b.low, b.close, b.volume) == (100.0, 106.0, 99.0, 105.0, 150), f"5m: {b}")
    # 10m первый: high=111, close=110, volume=550
    b = r10.bars[0]
    expect((b.open, b.high, b.low, b.close, b.volume) == (100.0, 111.0, 99.0, 110.0, 550), f"10m: {b}")
    # 15m первый: high=116, close=115, volume=1200
    b = r15.bars[0]
    expect((b.open, b.high, b.low, b.close, b.volume) == (100.0, 116.0, 99.0, 115.0, 1200), f"15m: {b}")
    return "5m:(100,106,99,105,V=150) 10m:(...,111,99,110,V=550) 15m:(...,116,99,115,V=1200)"


@check("Агрегация 1m -> 1h на 60 барах: ровно 1 бар, значения по ручному расчёту")
def check_aggregation_1h() -> str | None:
    bars = demo_series(60, START)
    result = resample(bars, "1m", "1h")
    expect(result.ok and result.pending is None, f"issues={result.issues}, pending={result.pending}")
    expect(len(result.bars) == 1, f"ожидался 1 часовой бар, получено {len(result.bars)}")
    b = result.bars[0]
    # open=100, high=159+2=161, low=99, close=159+1=160, volume=10*sum(1..60)=18300
    expect((b.open, b.high, b.low, b.close, b.volume) == (100.0, 161.0, 99.0, 160.0, 18300), f"1h: {b}")
    return "1h: (100,161,99,160,V=18300)"


@check("Эквивалентность путей: 1m->10m == 1m->5m->10m (decisions D3)")
def check_resample_equivalence() -> str | None:
    bars = demo_series(60, START)
    direct = resample(bars, "1m", "10m")
    two_step = resample(resample(bars, "1m", "5m").bars, "5m", "10m")
    expect(len(direct.bars) == len(two_step.bars) == 6, "число баров различается")
    for a, b in zip(direct.bars, two_step.bars):
        expect(
            (a.ts_open, a.open, a.high, a.low, a.close, a.volume)
            == (b.ts_open, b.open, b.high, b.low, b.close, b.volume),
            f"расхождение на {a.ts_open}: {a} != {b}",
        )
    return "6 баров идентичны на обоих путях"


@check("Неполный хвост: pending отдельно, is_final=False, НЕ среди полных баров")
def check_pending_tail() -> str | None:
    bars = demo_series(65, START)  # 60 полных минут + 5 минут хвоста
    result = resample(bars, "1m", "1h")
    expect(len(result.bars) == 1, f"ожидался 1 полный часовой бар, получено {len(result.bars)}")
    pending = result.pending
    expect(pending is not None, "pending отсутствует")
    expect(pending.is_final is False, "pending помечен как финальный — нарушение контракта")
    expect(all(b.is_final for b in result.bars), "в bars попал неполный бар")
    # pending из 5 баров i=60..64: open=160, high=164+2=166, low=160-1=159, close=164+1=165, V=10*(61+..+65)=3150
    expect(
        (pending.open, pending.high, pending.low, pending.close, pending.volume) == (160.0, 166.0, 159.0, 165.0, 3150),
        f"pending: {pending}",
    )
    return "полный бар=1, pending=(160,166,159,165,V=3150,is_final=False)"


@check("Дыра в данных: неполный интервал не выдаётся за полный и не заполняется")
def check_coverage_gap() -> str | None:
    bars = demo_series(12, START, skip={3})  # пропуск минуты внутри первого 10m
    result = resample(bars, "1m", "10m")
    expect(result.bars == [], "неполный интервал попал в полные бары")
    codes = [i.code for i in result.issues]
    expect(ResampleIssueCode.COVERAGE_INCOMPLETE in codes, f"нет issue о неполном покрытии: {codes}")
    issue = next(i for i in result.issues if i.code is ResampleIssueCode.COVERAGE_INCOMPLETE)
    expect(issue.expected == 10 and issue.found == 9, f"expected/found: {issue.expected}/{issue.found}")
    lenient = resample(bars, "1m", "10m", allow_incomplete=True)
    expect(lenient.bars and lenient.bars[0].is_final is False, "allow_incomplete должен помечать is_final=False")
    return "coverage_incomplete (10/9); allow_incomplete -> is_final=False"


@check("Календарь сессий: интервал через перерыв собирается полностью (без фиктивных баров)")
def check_sessions_break() -> str | None:
    # условные окна 07:00-07:05 и 07:10-07:15 UTC; бары только в окнах
    calendar = SessionCalendar.from_spec("UTC", [(range(0, 7), [("07:00", "07:05"), ("07:10", "07:15")])])
    bars = []
    for minute in list(range(0, 5)) + list(range(10, 15)):
        o = 100.0 + minute
        from tradebot.data.bar import Bar

        bars.append(
            Bar(
                ts_open=START + timedelta(minutes=minute),
                open=o,
                high=o + 2.0,
                low=o - 1.0,
                close=o + 1.0,
                volume=10 * (minute + 1),
                symbol="DEMO",
            )
        )
    result = resample(bars, "1m", "10m", calendar=calendar)
    complete = [b for b in result.bars if b.ts_open == START]
    expect(len(complete) == 1, f"интервал 07:00-07:10 должен быть полным: bars={result.bars}")
    expect(
        not any(i.code is ResampleIssueCode.COVERAGE_INCOMPLETE for i in result.issues),
        f"ложная проблема покрытия: {result.issues}",
    )
    expect(result.pending is not None and result.pending.ts_open == START + timedelta(minutes=10), "второй интервал должен быть pending")
    return "перерыв 07:05-07:10 учтён календарём: 5 слотов = 5 баров -> полный интервал"


# ---------------------------------------------------------------------------
# 4. Крипторежим: 24/7 и блокировка LIVE
# ---------------------------------------------------------------------------

@check("Календарь 24/7: выходные и полночь — торговое время; сетка непрерывна")
def check_crypto_24_7() -> str | None:
    cal = always_on_calendar()
    saturday = datetime(2024, 1, 6, 23, 59, tzinfo=UTC)
    sunday_midnight = datetime(2024, 1, 7, 0, 0, tzinfo=UTC)
    expect(cal.is_trading_time(saturday), "суббота 23:59 не торговое время")
    expect(cal.is_trading_time(sunday_midnight), "воскресенье 00:00 не торговое время")
    slots = cal.trading_slots(datetime(2024, 1, 5, tzinfo=UTC), datetime(2024, 1, 8, tzinfo=UTC), "1h")
    expect(len(slots) == 72, f"ожидалось 72 часовых слотов за 3 суток, получено {len(slots)}")
    # воскресные бары без пропусков загружаются без warnings
    with tempfile.TemporaryDirectory() as tmp:
        rows = ["datetime,open,high,low,close,volume"]
        base = datetime(2024, 1, 7, 0, 0)  # воскресенье
        for i in range(90):  # 1.5 часа через полночь не требуется, но покрываем час+
            ts = base + timedelta(minutes=i)
            rows.append(f"{ts:%Y-%m-%d %H:%M:%S},{100+i},{101+i},{99+i},{100.5+i},{10}")
        path = Path(tmp) / "sunday.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        result = load_csv(path, timeframe="1m", calendar=cal, timestamp_semantics="bar_open")
        expect(result.ok and not result.report.warnings, f"warnings: {result.report.summary_text()}")
    return "суббота/воскресенье/полночь — торговые; 72 слота за 3 суток; воскресный CSV без warnings"


@check("Биржевой интерфейс подготовлен и НЕ реализован; LIVE-гейт блокирует заявки")
def check_exchange_interface_blocked() -> str | None:
    import importlib.util

    from tradebot.exchange import (
        EXCHANGE_INTEGRATION_AVAILABLE,
        ExchangeAdapter,
        ExchangeNotImplemented,
        get_exchange_adapter,
    )
    from tradebot.live.gate import evaluate_live_gate

    expect(EXCHANGE_INTEGRATION_AVAILABLE is False, "интеграция объявлена доступной — это не так")
    try:
        get_exchange_adapter("anything")
    except ExchangeNotImplemented:
        pass
    else:
        raise AssertionError("get_exchange_adapter НЕ выбросил ExchangeNotImplemented")
    expect(issubclass(ExchangeAdapter, object) and hasattr(ExchangeAdapter, "gateway"),
           "контракт адаптера неполный")
    # прежний брокерский пакет удалён (имя собрано из фрагментов — см. гигиену)
    expect(importlib.util.find_spec("tradebot.fi" + "nam") is None, "модуль прежней интеграции всё ещё импортируем")

    # LIVE-гейт: даже «полное включение» не открывает ордерный контур
    config = {
        "run": {"mode": "LIVE", "experiment_id": "gate-check"},
        "data": {"csv_dir": "data/demo",
                 "symbols": [{"ticker": "BTCUSDT", "files": {}}],
                 "timestamp_semantics": "bar_open", "source_timezone": "UTC"},
        "timeframes": {"5m": {"enabled": True, "lookback_bars": 10, "horizon_bars": 1}},
        "risk": {"initial_capital_quote": 100_000.0, "position_units": 1, "max_position_units": 10,
                 "max_orders_per_day": 5, "daily_loss_limit_quote": 1_000.0,
                 "max_drawdown_pct": 10.0, "max_data_age_seconds": 60.0},
        "instrument": {"price_step": 0.01, "lot_size_units": 1},
        "simulation": {"commission_bps": 3.0, "slippage_bps": 2.0},
        "live": {"allow_orders": True, "max_order_quantity_units": 1,
                 "kill_switch_file": "no-such-file.kill"},
    }
    from tradebot.config import AppConfig

    app = AppConfig.model_validate(config)
    result = evaluate_live_gate(app, cli_flag=True, env={"TRADEBOT_LIVE_ACK": "ENABLE-LIVE"},
                                kill_switch_exists=False)
    expect(not result.allowed, "гейт разрешил LIVE без биржевой интеграции!")
    expect(any("НЕ реализована" in f for f in result.failures), f"нет причины про биржу: {result.failures}")
    expect("exchange_integration" in result.checks_performed, "фактор биржи не проверяется гейтом")
    return "адаптер бросает ExchangeNotImplemented; гейт: allowed=False, причина — интеграция не реализована"


# ---------------------------------------------------------------------------
# 5. Гигиена
# ---------------------------------------------------------------------------

@check("Гигиена: .gitignore/.env.example; в коде нет торговых операций и следов прежней интеграции")
def check_hygiene() -> str | None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    expect(".env" in gitignore, ".gitignore не покрывает .env")
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    expect("REPLACE_WITH_YOUR_OWN" in example, ".env.example без фиктивных заглушек")
    expect("eyJ" not in example, ".env.example содержит JWT-подобное значение")
    # токены собраны из фрагментов, чтобы проверка не обнаруживала саму себя.
    # Торговые операции в проекте отсутствуют: подключение биржи не реализовано
    # (exchange/base.py) — allowlist пуст.
    forbidden = ["/or" + "ders", "Place" + "Order", "Cancel" + "Order", "sltp-" + "orders", "new_" + "order"]
    legacy = ["fi" + "nam", "MO" + "EX", "MI" + "SX"]
    allowed_files: set[str] = set()
    offenders = []
    for pattern in ("src/**/*.py", "scripts/**/*.py", "configs/*.yaml"):
        for path in REPO_ROOT.glob(pattern):
            if path.name in allowed_files:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            offenders.extend(f"{path.name}:{tok}" for tok in forbidden if tok in text)
            offenders.extend(f"{path.name}:{tok}" for tok in legacy if tok.lower() in text.lower())
    expect(not offenders, f"запрещённые токены: {offenders}")
    import os

    real = os.environ.get("EXCHANGE_API_SECRET", "").strip()
    if real:
        for pattern in ("src/**/*.py", "scripts/**/*.py", "configs/*.yaml"):
            for path in REPO_ROOT.glob(pattern):
                expect(real not in path.read_text(encoding="utf-8", errors="replace"), f"реальный секрет в {path}")
    return "секреты/торговые операции/следы прежней интеграции отсутствуют; .env исключён из git"


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

def main() -> int:
    import logging

    # Детерминированный вывод сценария: предупреждения логгеров не должны
    # смешиваться с протоколом проверок.
    logging.disable(logging.CRITICAL)
    print("=" * 78)
    print("РУЧНАЯ ПРОВЕРКА СЛОЯ ДАННЫХ (КРИПТОРЕЖИМ): конфиг, CSV, агрегация ТФ, 24/7")
    print("Режим: ЛОКАЛЬНЫЙ (без сети и секретов)")
    print("Демо-данные: СИНТЕТИЧЕСКИЕ (data/demo/DEMO_M1.csv), ожидаемые значения — ручной расчёт")
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
        except Exception as exc:  # неожиданные ошибки — тоже FAIL с трейсбеком
            failures += 1
            print(f"[{index:02d}/{len(CHECKS)}] FAIL  {name} (неожиданная ошибка)")
            traceback.print_exc()
    print("-" * 78)
    total = len(CHECKS)
    if failures:
        print(f"RESULT: FAIL ({total - failures}/{total} checks passed)")
        print("Что прислать разработчику: полный вывод этого скрипта, версию Python")
        print("(python --version), ОС и вывод: pip show tradebot pydantic PyYAML")
        return 1
    print(f"RESULT: PASS ({total}/{total} checks)")
    print()
    print("Дополнительно (офлайн, на демо-данных):")
    print("  python -m tradebot.cli.validate_data --config configs\\demo.yaml --resample 10m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
