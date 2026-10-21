#!/usr/bin/env python3
"""Ручная проверка фазы 0-2: конфиг, CSV-валидация, агрегация таймфреймов, Finam-адаптер.

Запуск (из корня репозитория, PowerShell):
    python scripts\\manual_checks\\check_phase_02_data.py

Сценарий ПОЛНОСТЬЮ ЛОКАЛЬНЫЙ:
- не обращается к Finam API и не требует секретов/переменных окружения;
- адаптер Finam проверяется на ПОДМЕННОМ транспорте (FakeTransport): все «ответы
  сервера» — заранее известные фиктивные JSON-структуры, повторяющие формы из
  официальной документации (docs/research.md §2);
- демонстрационные CSV — СИНТЕТИЧЕСКИЕ данные с детерминированным шаблоном
  (явно помечены), ожидаемые результаты агрегации посчитаны вручную.

Необязательная проверка РЕАЛЬНОГО read-only подключения (потребует ВАШИ учётные
данные — secret_key с портала Finam; автоматически НЕ запускается):
    $env:TRADE_API_SECRET = "<ваш secret_key>"
    python -m tradebot.cli.smoke_readonly --symbol SBER@MISX --timeframe 5m --hours 24 --bar-timestamp-semantics bar_open

Код возврата: 0 — все проверки PASS; 1 — есть FAIL.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from tradebot.config import ConfigError, build_calendar, config_hash, load_config  # noqa: E402
from tradebot.data.bar import UTC, Timeframe  # noqa: E402
from tradebot.data.csv_loader import DataValidationError, load_csv  # noqa: E402
from tradebot.data.resampler import ResampleIssueCode, resample  # noqa: E402
from tradebot.data.sessions import SessionCalendar  # noqa: E402
from tradebot.data.validator import IssueCode  # noqa: E402
from tradebot.finam.errors import (  # noqa: E402
    FinamInvalidRequestError,
    UnsupportedTimeframeError,
)
from tradebot.finam.rest_client import HISTORY_DEPTH_LIMITS, FinamRestClient  # noqa: E402
from tradebot.finam.source import FinamHistorySource, FinamSnapshotSource  # noqa: E402
from tradebot.finam.transport import HttpResponse, HttpTransport  # noqa: E402

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


@dataclass
class FakeTransport(HttpTransport):
    """Подменный HTTP-транспорт: очередь фиктивных ответов + журнал запросов."""

    responses: list[HttpResponse] = field(default_factory=list)
    requests: list[tuple[str, str, dict[str, str], dict[str, Any] | None]] = field(default_factory=list)

    def request(self, method, url, *, headers=None, json_body=None, timeout=15.0):  # noqa: ANN001
        self.requests.append((method, url, dict(headers or {}), json_body))
        index = len(self.requests) - 1
        if not self.responses:
            return HttpResponse(status=500, body=b'{"message":"fake: no response programmed"}')
        return self.responses[min(index, len(self.responses) - 1)]


def json_response(status: int, payload: Any) -> HttpResponse:
    import json as _json

    return HttpResponse(status=status, body=_json.dumps(payload).encode("utf-8"))


FAKE_SECRET = "FAKE-SECRET-DO-NOT-USE-000"
FAKE_TOKEN = "FAKE-ACCESS-KEY-DO-NOT-USE-000"
FAKE_ACCOUNT = "FAKEACCOUNT000"


def auth_pair(expires_in_minutes: int = 15) -> list[HttpResponse]:
    expires = (datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        json_response(200, {"token": FAKE_TOKEN}),
        json_response(
            200,
            {
                "created_at": "2026-10-20T07:00:00Z",
                "expires_at": expires,
                "md_permissions": [{"quote_level": "QUOTE_LEVEL_1", "delay_minutes": 0, "mic": "MISX"}],
                "account_ids": [FAKE_ACCOUNT],
                "readonly": False,
            },
        ),
    ]


def make_client(transport: FakeTransport, sleeps: list[float] | None = None) -> FinamRestClient:
    import random

    return FinamRestClient(
        FAKE_SECRET,
        transport=transport,
        base_url="https://api.finam.ru",
        timeout_seconds=5.0,
        max_get_attempts=3,
        clock=lambda: 0.0,
        sleep=(sleeps.append if sleeps is not None else (lambda s: None)),
        rng=random.Random(7),
    )


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
                symbol="DEMO@DEMO",
            )
        )
    return bars


START = datetime(2024, 1, 4, 7, 0, tzinfo=UTC)  # 10:00 Europe/Moscow, четверг


# ---------------------------------------------------------------------------
# 1. Конфигурация
# ---------------------------------------------------------------------------

@check("Конфиг demo.yaml загружается; lookback/horizon явные для каждого ТФ")
def check_config_demo() -> str | None:
    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    for code, tf_cfg in config.timeframes.items():
        expect(tf_cfg.lookback_bars >= 1, f"{code}: lookback_bars не задан")
        expect(tf_cfg.horizon_bars >= 1, f"{code}: horizon_bars не задан")
    expect(config.timeframes["10m"].base_timeframe == "5m", "10m должен строиться из 5m")
    expect(config.data.timestamp_semantics == "bar_open", "семантика метки должна быть явной")
    h1 = config_hash(config)
    h2 = config_hash(load_config(REPO_ROOT / "configs" / "demo.yaml"))
    expect(h1 == h2 and len(h1) == 64, "хэш конфига не детерминирован")
    return f"timeframes={sorted(config.timeframes)}, hash={h1[:12]}"


@check("Конфиг example.yaml загружается (схема соответствует architecture §3.4)")
def check_config_example() -> str | None:
    config = load_config(REPO_ROOT / "configs" / "example.yaml")
    expect(config.data.symbols[0].symbol == "SBER@MISX", "symbol != SBER@MISX")
    expect(config.finam.rate_limit_per_minute <= 200, "лимит выше документированного")
    expect(config.finam.bar_timestamp_semantics is None, "семантика API-баров не должна угадываться по умолчанию")
    return "ok"


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

@check("Демо-CSV (120 синтетических 1m-баров) загружается без ошибок")
def check_csv_demo() -> str | None:
    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    calendar = build_calendar(config)
    result = load_csv(
        REPO_ROOT / "data" / "demo" / "DEMO_M1.csv",
        timeframe="1m",
        symbol="DEMO@DEMO",
        source_timezone="Europe/Moscow",
        timestamp_semantics="bar_open",
        calendar=calendar,
    )
    expect(len(result.bars) == 120, f"ожидалось 120 баров, получено {len(result.bars)}")
    expect(not result.report.errors, f"ошибки: {result.report.summary_text()}")
    expect(not result.report.warnings, f"предупреждения: {result.report.summary_text()}")
    first = result.bars[0]
    expect(first.ts_open == datetime(2024, 1, 4, 7, 0, tzinfo=UTC), "10:00 MSK должно стать 07:00 UTC")
    expect(first.open == 100.0 and first.volume == 1000, "значения первой строки не совпали")
    return "120 баров, 10:00 MSK -> 07:00 UTC"


@check("Семантика bar_close сдвигает метку на начало бара")
def check_csv_bar_close() -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "close_stamped.csv"
        rows = ["datetime,open,high,low,close,volume"]
        for i in range(5):
            ts = datetime(2024, 1, 4, 10, 5 * (i + 1))  # метки КОНЦА 5m-баров (MSK naive)
            rows.append(f"{ts:%Y-%m-%d %H:%M:%S},{100+i},{101+i},{99+i},{100.5+i},{10*(i+1)}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        result = load_csv(path, timeframe="5m", timestamp_semantics="bar_close", source_timezone="Europe/Moscow")
        expect(len(result.bars) == 5, "ожидалось 5 баров")
        expect(
            result.bars[0].ts_open == datetime(2024, 1, 4, 7, 0, tzinfo=UTC),
            f"первый бар: {result.bars[0].ts_open} (ожидалось 07:00 UTC)",
        )
    return "нормализация bar_close -> bar_open работает"


def _write_bad_csv(tmp: Path, rows: list[str], header: str = "datetime,open,high,low,close,volume") -> Path:
    path = tmp / "bad.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


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
    calendar = SessionCalendar.from_spec("Europe/Moscow", [(range(0, 7), [("10:00", "10:05"), ("10:10", "10:15")])])
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
                symbol="DEMO@DEMO",
            )
        )
    result = resample(bars, "1m", "10m", calendar=calendar)
    complete = [b for b in result.bars if b.ts_open == START]
    expect(len(complete) == 1, f"интервал 10:00-10:10 должен быть полным: bars={result.bars}")
    expect(
        not any(i.code is ResampleIssueCode.COVERAGE_INCOMPLETE for i in result.issues),
        f"ложная проблема покрытия: {result.issues}",
    )
    expect(result.pending is not None and result.pending.ts_open == START + timedelta(minutes=10), "второй интервал должен быть pending")
    return "перерыв 10:05-10:10 учтён календарём: 5 слотов = 5 баров -> полный интервал"


# ---------------------------------------------------------------------------
# 4. Finam-адаптер на подменном транспорте (без сети и секретов)
# ---------------------------------------------------------------------------

@check("Finam: аутентификация по документированному потоку POST /v1/sessions(+details)")
def check_finam_auth() -> str | None:
    transport = FakeTransport(responses=auth_pair())
    client = make_client(transport)
    details = client.authenticate()
    expect(details.account_ids == (FAKE_ACCOUNT,), "account_ids не прочитаны")
    expect(details.readonly is False, "readonly не прочитан")
    method, url, headers, body = transport.requests[0]
    expect(method == "POST" and url == "https://api.finam.ru/v1/sessions", f"неверный auth-запрос: {method} {url}")
    expect(body == {"secret": FAKE_SECRET}, "тело auth-запроса не соответствует документации")
    expect(FAKE_SECRET not in url, "secret не должен попадать в URL")
    method2, url2, _, body2 = transport.requests[1]
    expect(url2.endswith("/v1/sessions/details") and body2 == {"token": FAKE_TOKEN}, "неверный details-запрос")
    expect(FAKE_ACCOUNT not in repr(details) and FAKE_ACCOUNT not in str(details), "account id утекает в repr/str")
    return "token->details, secret только в теле, account_ids скрыты из repr/str"


@check("Finam: GET bars — путь, timeframe enum, RFC3339 интервал [start, end), Bearer-заголовок")
def check_finam_bars_request() -> str | None:
    payload = {
        "symbol": "SBER@MISX",
        "bars": [
            {
                "timestamp": "2026-10-20T07:00:00Z",
                "open": {"value": "313.58"},
                "high": {"value": "314.10"},
                "low": {"value": "313.01"},
                "close": {"value": "313.90"},
                "volume": {"value": "123456"},
                "is_data_snapshot": False,
            }
        ],
    }
    transport = FakeTransport(responses=auth_pair() + [json_response(200, payload)])
    client = make_client(transport)
    start = datetime(2026, 10, 20, 7, 0, tzinfo=UTC)
    records = client.fetch_bars("SBER@MISX", "5m", start, start + timedelta(hours=1))
    expect(len(records) == 1, "бар не разобран")
    expect(records[0].open == Decimal("313.58"), f"decimal потерян: {records[0].open}")
    expect(records[0].volume == Decimal("123456"), "volume потерян")
    method, url, headers, _ = transport.requests[-1]
    expect(method == "GET" and "/v1/instruments/SBER%40MISX/bars" in url, f"неверный путь: {url}")
    expect("timeframe=TIME_FRAME_M5" in url, f"нет enum-таймфрейма: {url}")
    expect("interval.start_time=2026-10-20T07%3A00%3A00Z" in url, f"нет start_time: {url}")
    expect("interval.end_time=2026-10-20T08%3A00%3A00Z" in url, f"нет end_time: {url}")
    expect(headers.get("Authorization") == f"Bearer {FAKE_TOKEN}", "нет Bearer access_key")
    expect(FAKE_SECRET not in url and FAKE_SECRET not in str(headers), "secret утекает в GET-запрос")
    return "TIME_FRAME_M5, RFC3339 Z, decimal {value} сохранён, Bearer-заголовок"


@check("Finam: 10m отклонён (нет TIME_FRAME_M10) с инструкцией про ресэмплинг")
def check_finam_10m_unsupported() -> str | None:
    transport = FakeTransport(responses=auth_pair())
    client = make_client(transport)
    client.authenticate()
    try:
        client.fetch_bars("SBER@MISX", "10m", START, START + timedelta(hours=1))
    except UnsupportedTimeframeError as exc:
        expect("TIME_FRAME_M10" in str(exc) and "ресэмплинг" in str(exc), f"сообщение: {exc}")
        return "UnsupportedTimeframeError с отсылкой к data.resampler"
    raise AssertionError("10m НЕ был отклонён")


@check("Finam: глубина истории ограничена документированными пределами (M1 7д, M5 30д)")
def check_finam_depth() -> str | None:
    transport = FakeTransport(responses=auth_pair())
    client = make_client(transport)
    client.authenticate()
    expect(HISTORY_DEPTH_LIMITS["1m"] == timedelta(days=7), "лимит M1 != 7 дней")
    expect(HISTORY_DEPTH_LIMITS["5m"] == timedelta(days=30), "лимит M5 != 30 дней")
    for tf, days in (("1m", 8), ("5m", 31), ("1h", 31)):
        try:
            client.fetch_bars("SBER@MISX", tf, START, START + timedelta(days=days))
        except Exception as exc:
            expect(type(exc).__name__ == "HistoryDepthError", f"{tf}/{days}д: {type(exc).__name__}")
        else:
            raise AssertionError(f"{tf} на {days} дней НЕ отклонён")
    return "M1>7д, M5>30д, 1h>30д отклоняются до запроса"


@check("Finam: снимок стакана и котировки (документированные поля), глубина не гарантируется")
def check_finam_snapshots() -> str | None:
    quote_payload = {
        "symbol": "SBER@MISX",
        "quote": {
            "timestamp": "2026-10-20T07:00:01Z",
            "last": {"value": "313.58"},
            "bid": {"value": "313.57"},
            "ask": {"value": "313.59"},
            "bid_size": {"value": "120"},
            "ask_size": {"value": "90"},
            "volume": {"value": "98765"},
        },
    }
    book_payload = {
        "symbol": "SBER@MISX",
        "orderbook": {
            "rows": [
                {"price": {"value": "313.57"}, "buy_size": {"value": "120"}, "action": "ACTION_UPDATE", "mpid": "MC000"},
                {"price": {"value": "313.59"}, "sell_size": {"value": "90"}, "action": "ACTION_ADD"},
            ]
        },
    }
    transport = FakeTransport(responses=auth_pair() + [json_response(200, quote_payload), json_response(200, book_payload)])
    source = FinamSnapshotSource(make_client(transport))
    quote = source.get_latest_quote("SBER@MISX")
    expect(quote.last == Decimal("313.58") and quote.bid_size == Decimal("120"), f"quote: {quote}")
    book = source.get_order_book("SBER@MISX")
    expect(book.depth == 2, f"depth: {book.depth}")
    expect(book.rows[0].side.value == "buy" and book.rows[1].side.value == "sell", "стороны перепутаны")
    return "quote(last/bid/ask/sizes), orderbook 2 уровня (buy/sell, action, mpid) — только фактическая глубина"


@check("Finam: 429 -> ожидание минутного окна и один повтор (не немедленный ретрай)")
def check_finam_429() -> str | None:
    sleeps: list[float] = []
    transport = FakeTransport(
        responses=auth_pair()
        + [
            json_response(429, {"code": 8, "message": "too many requests"}),
            json_response(200, {"symbol": "SBER@MISX", "bars": []}),
        ]
    )
    client = make_client(transport, sleeps=sleeps)
    records = client.fetch_bars("SBER@MISX", "5m", START, START + timedelta(hours=1))
    expect(records == [], "ожидался пустой список баров после успешного повтора")
    expect(any(s >= 60.0 for s in sleeps), f"не было ожидания окна (~60 c): {sleeps}")
    return f"повтор после ожидания {sleeps[0]:.0f} c"


@check("Finam: 400 не повторяется; сообщения ошибок маскируют секреты")
def check_finam_400_and_masking() -> str | None:
    body = {"code": 3, "message": f"bad request secret={FAKE_SECRET} token={FAKE_TOKEN}"}
    transport = FakeTransport(responses=auth_pair() + [json_response(400, body)])
    client = make_client(transport)
    try:
        client.fetch_bars("SBER@MISX", "5m", START, START + timedelta(hours=1))
    except FinamInvalidRequestError as exc:
        message = str(exc)
        expect(FAKE_SECRET not in message, "secret НЕ замаскирован в ошибке")
        expect(FAKE_TOKEN not in message, "access_key НЕ замаскирован в ошибке")
        expect("***SECRET_KEY***" in message and "***ACCESS_KEY***" in message, f"маскирование: {message}")
    else:
        raise AssertionError("400 не вызвал FinamInvalidRequestError")
    gets = [r for r in transport.requests if r[0] == "GET" and "/bars" in r[1]]
    expect(len(gets) == 1, f"400 повторялся {len(gets)} раз — должен без повторов")
    return "один запрос, secret/token заменены на ***"


@check("Finam: нормализация баров требует ЯВНОЙ семантики timestamp (research Q3)")
def check_finam_semantics_explicit() -> str | None:
    ts_payload = {
        "symbol": "SBER@MISX",
        "bars": [
            {
                "timestamp": "2026-10-20T07:05:00Z",
                "open": {"value": "100"},
                "high": {"value": "101"},
                "low": {"value": "99"},
                "close": {"value": "100.5"},
                "volume": {"value": "10"},
                "is_data_snapshot": False,
            }
        ],
    }
    try:
        FinamHistorySource(make_client(FakeTransport(responses=auth_pair())), bar_timestamp_semantics=None)
    except ValueError as exc:
        expect("bar_timestamp_semantics" in str(exc), f"сообщение: {exc}")
    else:
        raise AssertionError("None-семантика принята — молчаливое угадывание запрещено")
    for semantics, expected in (
        ("bar_open", datetime(2026, 10, 20, 7, 5, tzinfo=UTC)),
        ("bar_close", datetime(2026, 10, 20, 7, 0, tzinfo=UTC)),
    ):
        transport = FakeTransport(responses=auth_pair() + [json_response(200, ts_payload)])
        source = FinamHistorySource(make_client(transport), bar_timestamp_semantics=semantics)
        bars = source.fetch_history("SBER@MISX", "5m", datetime(2026, 10, 20, tzinfo=UTC), datetime(2026, 10, 21, tzinfo=UTC))
        expect(bars[0].ts_open == expected, f"{semantics}: {bars[0].ts_open} != {expected}")
        expect(bars[0].volume == 10 and isinstance(bars[0].volume, int), "volume не int")
    return "None -> ValueError; bar_open/bar_close нормализуются корректно"


@check("Finam: naive-timestamp и дробный volume отклоняются (данные не угадываются)")
def check_finam_strict_parsing() -> str | None:
    naive = {
        "symbol": "SBER@MISX",
        "bars": [
            {"timestamp": "2026-10-20T07:05:00", "open": {"value": "1"}, "high": {"value": "2"},
             "low": {"value": "0.5"}, "close": {"value": "1.5"}, "volume": {"value": "10"}}
        ],
    }
    transport = FakeTransport(responses=auth_pair() + [json_response(200, naive)])
    source = FinamHistorySource(make_client(transport), bar_timestamp_semantics="bar_open")
    try:
        source.fetch_history("SBER@MISX", "5m", START, START + timedelta(hours=1))
    except Exception as exc:
        expect(type(exc).__name__ == "FinamInvalidResponseError", f"naive ts: {type(exc).__name__}")
    else:
        raise AssertionError("naive timestamp принят")
    fractional = {
        "symbol": "SBER@MISX",
        "bars": [
            {"timestamp": "2026-10-20T07:00:00Z", "open": {"value": "1"}, "high": {"value": "2"},
             "low": {"value": "0.5"}, "close": {"value": "1.5"}, "volume": {"value": "10.5"}}
        ],
    }
    transport2 = FakeTransport(responses=auth_pair() + [json_response(200, fractional)])
    source2 = FinamHistorySource(make_client(transport2), bar_timestamp_semantics="bar_open")
    try:
        source2.fetch_history("SBER@MISX", "1m", START, START + timedelta(minutes=30))
    except Exception as exc:
        expect(type(exc).__name__ == "FinamInvalidResponseError", f"fractional volume: {type(exc).__name__}")
    else:
        raise AssertionError("дробный volume принят (округление молча запрещено)")
    return "naive ts и volume=10.5 отклонены как FinamInvalidResponseError"


# ---------------------------------------------------------------------------
# 5. Гигиена
# ---------------------------------------------------------------------------

@check("Гигиена: .gitignore/.env.example; в коде нет торговых операций и секретов")
def check_hygiene() -> str | None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    expect(".env" in gitignore, ".gitignore не покрывает .env")
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    expect("REPLACE_WITH_YOUR_OWN" in example, ".env.example без фиктивных заглушек")
    expect("eyJ" not in example, ".env.example содержит JWT-подобное значение")
    # токены собраны из фрагментов, чтобы проверка не обнаруживала саму себя.
    # ФАЗА 5: торговые операции разрешены ТОЛЬКО в отдельном ордерном адаптере
    # src/tradebot/finam/broker.py (за многофакторным LIVE-гейтом, docs/live.md) —
    # файл в явном allowlist; остальные модули по-прежнему «без торговли».
    forbidden = ["/or" + "ders", "Place" + "Order", "Cancel" + "Order", "sltp-" + "orders", "new_" + "order"]
    allowed_files = {"broker.py"}
    offenders = []
    for pattern in ("src/**/*.py", "scripts/**/*.py", "configs/*.yaml"):
        for path in REPO_ROOT.glob(pattern):
            if path.name in allowed_files:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            offenders.extend(f"{path.name}:{tok}" for tok in forbidden if tok in text)
    expect(not offenders, f"упоминания торговых операций: {offenders}")
    import os

    real = os.environ.get("TRADE_API_SECRET", "").strip()
    if real:
        for pattern in ("src/**/*.py", "scripts/**/*.py", "configs/*.yaml"):
            for path in REPO_ROOT.glob(pattern):
                expect(real not in path.read_text(encoding="utf-8", errors="replace"), f"реальный секрет в {path}")
    return "секреты/торговые операции отсутствуют; .env исключён из git"


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

def main() -> int:
    import logging

    # Детерминированный вывод сценария: предупреждения логгеров (например ожидание
    # окна после фиктивного 429) не должны смешиваться с протоколом проверок.
    logging.disable(logging.CRITICAL)
    print("=" * 78)
    print("РУЧНАЯ ПРОВЕРКА ФАЗЫ 0-2: конфиг, CSV, агрегация ТФ, read-only Finam-адаптер")
    print("Режим: ЛОКАЛЬНЫЙ (без сети и секретов; Finam — подменный транспорт)")
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
    print("Необязательная проверка РЕАЛЬНОГО API (потребует ВАШИ учётные данные,")
    print("автоматически не запускается; только read-only, заявки не выставляются):")
    print('  $env:TRADE_API_SECRET = "<ваш secret_key>"')
    print("  python -m tradebot.cli.smoke_readonly --symbol SBER@MISX --timeframe 5m --hours 24 --bar-timestamp-semantics bar_open")
    return 0


if __name__ == "__main__":
    sys.exit(main())
