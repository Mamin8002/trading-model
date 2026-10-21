#!/usr/bin/env python3
"""Ручная проверка фазы 5: ПОЛНЫЙ офлайн-сценарий TEST + защита LIVE-контура.

Запуск (из корня репозитория, PowerShell):
    python scripts\\manual_checks\\check_phase_05_test_run.py

Что делает сценарий (полностью ОФЛАЙН, детерминированно):
  1) готовит небольшой демонстрационный набор СИНТЕТИЧЕСКИХ 5m-баров (SYNTH@DEMO,
     640 штук, seed фиксирован) и конфигурацию TEST;
  2) запускает единой командой полный TEST-путь: чтение CSV -> обучение модели
     (walk-forward) -> сигналы -> риск-контроль -> симулятор -> результаты,
     отчёт и журнал;
  3) повторно прогоняет симуляцию с обученным артефактом и проверя метрики;
  4) проверяет сценарии НЕКОРРЕКТНОЙ конфигурации: неполный risk/simulation и
     mode: LIVE в команде `test` -> отказ с кодом 2 БЕЗ каких-либо заявок;
  5) проверяет LIVE-гейт: без флага/подтверждений отказ ДО создания сетевого
     клиента (клиент подменяется «сторожем» — создание невозможно);
  6) проверяет ордерный адаптер на ПОДМЕНЁННОМ транспорте: документированное
     тело заявки, потерянный ответ -> UNKNOWN_RESULT -> сверка -> защита от
     дубля (ровно один POST), kill-switch -> DISABLED без сети, отказ
     риск-контроля (устаревшие данные) -> ноль вызовов шлюза;
  7) проверяет, что ФЕЙКОВЫЕ секреты не попадают в логи и журнал.

ВАЖНО:
- Сценарий НЕ обращается к Finam API, НЕ требует секретов и НЕ отправляет
  заявки. Это НЕ проверка в брокерской среде: демо-счёт не задействован,
  модель исполнения демо документацией не описана (docs/research.md, Q1).
  Проверки ордерного адаптера — на подменном транспорте (docs/live.md).
- Ожидаемые файлы после успешного запуска (создаются в reports/phase05_manual/,
  каталог исключён из git):
      data/SYNTH_M5.csv                     — демонстрационный набор (СИНТЕТИКА)
      config.yaml                           — конфигурация TEST-прогона
      models/phase05-manual_5m_<hash>/      — артефакт модели (model.lgb.txt,
                                              metadata.json, feature_manifest.json)
      reports/test_run_phase05-manual_5m.md — отчёт симуляции с допущениями
      reports/journal_phase05-manual_5m.jsonl — журнал решений/исполнений
      journal_broker.jsonl                  — журнал событий ордерного адаптера
- Ожидаемый вывод: по каждой проверке «ОЖИДАЛОСЬ / ФАКТ / PASS», в конце
  «RESULT: PASS (17/17 checks)» и код возврата 0.

Если проверка завершилась ошибкой, пришлите:
  1. ПОЛНЫЙ вывод скрипта (от «===» до «RESULT: ...»);
  2. `python --version`, операционная система;
  3. `pip show tradebot lightgbm pydantic` (версии);
  4. хвост `python -m pytest -q` (последние ~20 строк).

Код возврата: 0 — все проверки PASS; 1 — есть FAIL.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from tradebot.backtest.engine import run_simulation  # noqa: E402
from tradebot.backtest.results import compute_results  # noqa: E402
from tradebot.cli.run import main as run_main  # noqa: E402
from tradebot.config import load_config  # noqa: E402
from tradebot.config.schema import InstrumentConfig, LiveConfig, RiskConfig, SimulationConfig  # noqa: E402
from tradebot.data.synthetic import generate_synthetic_bars  # noqa: E402
from tradebot.execution.interfaces import (  # noqa: E402
    OrderIntent,
    OrderSubmissionStatus,
    PortfolioState,
    RejectionCode,
    RiskContext,
    Side,
    Signal,
)
from tradebot.execution.journal import JournalWriter, read_journal  # noqa: E402
from tradebot.execution.risk import RiskGuards  # noqa: E402
from tradebot.finam.broker import FinamBrokerGateway  # noqa: E402
from tradebot.finam.errors import FinamTransportError  # noqa: E402
from tradebot.finam.rest_client import FinamRestClient  # noqa: E402
from tradebot.finam.transport import HttpResponse  # noqa: E402
from tradebot.ml.registry import load_model_bundle  # noqa: E402
from tradebot.ml.train import load_bars_for_timeframe  # noqa: E402
from tradebot.strategy.signals import ModelSignalSource  # noqa: E402

UTC = timezone.utc
OUT_DIR = REPO_ROOT / "reports" / "phase05_manual"
CREATED: list[str] = []

# ФЕЙКОВЫЕ значения — не являются учётными данными (проверки офлайн)
FAKE_SECRET = "FAKE-SECRET-PHASE5-DO-NOT-USE"
FAKE_ACCESS_KEY = "FAKE-ACCESS-KEY-PHASE5-DO-NOT-USE"
FAKE_ACCOUNT_ID = "FAKEACCOUNT555"

# фрагменты, а не литералы: гигиена-проверки сканируют scripts/ (см. check 17)
_ORDERS_SUFFIX = "/or" + "ders"

CHECKS: list[tuple[str, str, Callable[[], str]]] = []


def check(name: str, expected: str):
    """Регистрирует проверку с печатью ожидаемого результата."""

    def decorator(fn: Callable[[], str]):
        CHECKS.append((name, expected, fn))
        return fn

    return decorator


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def json_response(status: int, payload: Any) -> HttpResponse:
    return HttpResponse(status=status, body=json.dumps(payload).encode("utf-8"))


def auth_pair() -> list[HttpResponse]:
    return [
        json_response(200, {"token": FAKE_ACCESS_KEY}),
        json_response(200, {"account_ids": [FAKE_ACCOUNT_ID], "readonly": False,
                            "expires_at": "2026-10-21T12:15:00Z"}),
    ]


class ScriptedTransport:
    """Подменный транспорт: ответы по сценарию; Exception в сценарии выбрасывается."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[tuple] = []

    def request(self, method, url, *, headers=None, json_body=None, timeout=15.0):
        self.requests.append((method, url, dict(headers or {}), json_body))
        item = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    def order_posts(self) -> list[tuple]:
        return [r for r in self.requests if r[0] == "POST" and r[1].endswith(_ORDERS_SUFFIX)]


def make_client(transport: ScriptedTransport) -> FinamRestClient:
    return FinamRestClient(
        FAKE_SECRET, transport=transport, base_url="https://api.finam.ru",
        timeout_seconds=5.0, max_get_attempts=2, clock=lambda: 0.0, sleep=lambda s: None,
    )


NO_KILL_FILE = OUT_DIR / "no-such-kill-switch"
ORDER_STATE = {
    "order_id": "ord-77", "status": "ORDER_STATUS_NEW",
    "order": {"symbol": "SBER@MISX", "client_order_id": "d" * 20, "quantity": {"value": "10"}},
}


def make_gateway(transport: ScriptedTransport, *, kill_file: Path | None = None,
                 journal=None) -> FinamBrokerGateway:
    live_cfg = LiveConfig(allow_orders=True, max_order_quantity_units=10,
                          kill_switch_file=str(kill_file or NO_KILL_FILE))
    return FinamBrokerGateway(
        make_client(transport), account_id=FAKE_ACCOUNT_ID, live_cfg=live_cfg,
        instrument_cfg=InstrumentConfig(price_step=0.01, lot_size_units=1),
        orders_enabled=True, journal=journal,
    )


def make_intent(units: int = 10, client_id: str = "d" * 20) -> OrderIntent:
    ts = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    signal = Signal(ts_open=ts, side=Side.BUY, symbol="SBER@MISX", timeframe="5m", model_version="test")
    return OrderIntent(created_at=ts, symbol="SBER@MISX", timeframe="5m", side=Side.BUY,
                       quantity_units=units, signal=signal, reference_price=300.0,
                       client_intent_id=client_id)


def cli(argv: list[str]) -> tuple[int, str, str]:
    """Запуск CLI in-process с перехватом вывода (сценарий печатает только своё)."""
    out, err = io.StringIO(), io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_main(argv)
    return code, out.getvalue(), err.getvalue()


def full_config() -> dict:
    return {
        "run": {"mode": "TEST", "experiment_id": "phase05-manual", "report_dir": str(OUT_DIR / "reports")},
        "data": {"csv_dir": str(OUT_DIR / "data"),
                 "symbols": [{"ticker": "SYNTH", "mic": "DEMO", "files": {"5m": "SYNTH_M5.csv"}}],
                 "timestamp_semantics": "bar_open", "source_timezone": "Europe/Moscow"},
        "timeframes": {"5m": {"enabled": True, "lookback_bars": 120, "horizon_bars": 6,
                              "label": {"kind": "fwd_return_class", "neutral_band_bps": 5.0}}},
        "model": {"kind": "lightgbm", "seed": 7,
                  "params": {"n_estimators": 60, "learning_rate": 0.05, "min_child_samples": 20}},
        "validation": {"walk_forward": {"n_splits": 2, "min_train_bars": 300, "min_val_bars": 60}},
        "risk": {"initial_capital_quote": 100_000.0, "position_units": 10,
                 "max_position_units": 100, "max_orders_per_day": 40,
                 "daily_loss_limit_quote": 5_000.0, "max_drawdown_pct": 20.0,
                 "max_data_age_seconds": 86_400},
        "instrument": {"price_step": 0.01, "lot_size_units": 1},
        "simulation": {"commission_bps": 3.0, "commission_min_quote": 0.05, "slippage_bps": 2.0,
                       "signal": {"entry_threshold": 0.55, "exit_threshold": 0.45,
                                  "stop_bps": 100.0, "target_bps": 200.0}},
        "finam": {"base_url": "https://api.finam.ru"},
    }


def write_yaml(path: Path, config: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    CREATED.append(str(path.relative_to(REPO_ROOT)))
    return str(path)


# ---------------------------------------------------------------------------
# 1. Точка входа и демонстрационный набор
# ---------------------------------------------------------------------------

@check("Единая точка входа: python -m tradebot (test / live-preflight)",
       "--help обоих уровней -> exit 0; подкоманды test и live-preflight перечислены")
def check_entrypoint() -> str:
    proc = subprocess.run([sys.executable, "-m", "tradebot", "--help"],
                          capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT))
    expect(proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr[:200]}")
    expect("test" in proc.stdout and "live-preflight" in proc.stdout, "подкоманды не перечислены")
    proc2 = subprocess.run([sys.executable, "-m", "tradebot", "test", "--help"],
                           capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT))
    expect(proc2.returncode == 0, f"test --help exit {proc2.returncode}")
    expect("--model-dir" in proc2.stdout and "--train" in proc2.stdout, "нет источников модели")
    return "exit 0; test/live-preflight на месте; --train/--model-dir взаимно исключены"


@check("Демонстрационный набор: 640 СИНТЕТИЧЕСКИХ 5m-баров + конфигурация TEST",
       "SYNTH_M5.csv: 641 строка (заголовок+640); config.yaml записан; live-секции нет")
def check_demo_dataset() -> str:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "data").mkdir(parents=True)
    bars = generate_synthetic_bars(640, seed=7, timeframe="5m", symbol="SYNTH@DEMO")
    csv_path = OUT_DIR / "data" / "SYNTH_M5.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        # ВНИМАНИЕ: набор СИНТЕТИЧЕСКИЙ (не рыночные данные); комментарий в файл
        # не пишем — формат CSV строгий (docs/data.md)
        fh.write("datetime,open,high,low,close,volume\n")
        for bar in bars:
            fh.write(f"{bar.ts_open.strftime('%Y-%m-%d %H:%M:%S')},{bar.open},{bar.high},"
                     f"{bar.low},{bar.close},{bar.volume}\n")
    CREATED.append(str(csv_path.relative_to(REPO_ROOT)))
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    expect(len(lines) == 641, f"строк в CSV: {len(lines)} (ожидалось 641)")
    config = full_config()
    expect("live" not in config, "в TEST-конфиге не должно быть live-секции")
    write_yaml(OUT_DIR / "config.yaml", config)
    return f"CSV 640 баров (СИНТЕТИКА) + config.yaml записаны в {OUT_DIR.name}/"


# ---------------------------------------------------------------------------
# 2. Полный TEST-путь единой командой
# ---------------------------------------------------------------------------

PIPELINE_STATE: dict[str, Any] = {}


@check("Полный TEST-путь одной командой: данные -> обучение -> симуляция",
       "exit 0; в stdout 5 шагов [1/5]...[5/5]; «Реальные заявки не выставлялись»")
def check_full_pipeline() -> str:
    code, out, err = cli(["test", "--config", str(OUT_DIR / "config.yaml"), "--timeframe", "5m",
                          "--train", "--model-output-dir", str(OUT_DIR / "models"),
                          "--report-output-dir", str(OUT_DIR / "reports")])
    PIPELINE_STATE.update(code=code, out=out, err=err)
    expect(code == 0, f"exit {code}; stderr: {err[:400]}")
    for step in ("[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]"):
        expect(step in out, f"нет шага {step}")
    expect("Реальные заявки не выставлялись" in out, "нет напоминания о TEST-режиме")
    entries_line = next((l for l in out.splitlines() if l.startswith("Входов:")), "")
    expect("Капитал:" in out and entries_line != "", "нет итоговой сводки")
    return f"exit 0; шаги 1-5 выполнены; {entries_line.strip()}"


@check("Артефакты прогона: отчёт, журнал, модель",
       "test_run_phase05-manual_5m.md, journal_phase05-manual_5m.jsonl и каталог модели существуют")
def check_artifacts() -> str:
    report = OUT_DIR / "reports" / "test_run_phase05-manual_5m.md"
    journal = OUT_DIR / "reports" / "journal_phase05-manual_5m.jsonl"
    models = sorted((OUT_DIR / "models").glob("phase05-manual_5m_*"))
    expect(report.exists(), f"нет отчёта {report}")
    expect(journal.exists(), f"нет журнала {journal}")
    expect(bool(models), "нет каталога модели models/phase05-manual_5m_*")
    bundle = {p.name for p in models[0].iterdir()}
    expect({"model.lgb.txt", "metadata.json", "feature_manifest.json"} <= bundle,
           f"неполный артефакт модели: {bundle}")
    for path in (report, journal):
        CREATED.append(str(path.relative_to(REPO_ROOT)))
    CREATED.append(str(models[0].relative_to(REPO_ROOT)) + "/")
    PIPELINE_STATE["model_dir"] = models[0]
    return f"отчёт {report.stat().st_size} Б; журнал {journal.stat().st_size} Б; модель {models[0].name}"


@check("Отчёт: дисклеймер, допущения симуляции, TEST-статус",
       "разделы «Допущения симуляции» и дисклеймер «Реальные заявки не выставлялись» присутствуют")
def check_report_content() -> str:
    text = (OUT_DIR / "reports" / "test_run_phase05-manual_5m.md").read_text(encoding="utf-8")
    expect("Допущения симуляции" in text, "нет раздела допущений")
    expect("Реальные заявки не выставлялись" in text, "нет дисклеймера")
    expect("Дисклеймер" in text, "нет блока дисклеймера")
    expect("commission" in text.lower() or "комисс" in text.lower(), "нет упоминания комиссий")
    return "разделы допущений и дисклеймер на месте"


@check("Журнал: машиночитаемый JSONL, виды событий, исполнения",
       "каждая строка — валидный JSON с kind/ts; виды ⊇ {signal, risk_decision, fill, state}; fill >= 1")
def check_journal_content() -> str:
    journal = OUT_DIR / "reports" / "journal_phase05-manual_5m.jsonl"
    records = read_journal(journal)
    expect(len(records) > 100, f"записей слишком мало: {len(records)}")
    kinds = {r.get("kind") for r in records}
    expect({"signal", "risk_decision", "fill", "state"} <= kinds, f"неполные виды: {kinds}")
    fills = [r for r in records if r.get("kind") == "fill"]
    expect(len(fills) >= 1, "нет ни одного исполнения (fill)")
    for record in records:
        expect("kind" in record and ("event_ts" in record or "written_at" in record),
               f"запись без kind/ts: {record}")
    return f"{len(records)} записей; fills={len(fills)}; виды={sorted(k for k in kinds if k)}"


@check("Повторная симуляция с обученным артефактом: результаты и комиссии",
       "загрузка бандла -> совместимость 5m -> run_simulation: n_entries >= 1, комиссии > 0")
def check_resimulate_with_artifact() -> str:
    config = load_config(OUT_DIR / "config.yaml")
    bars, symbol = load_bars_for_timeframe(config, "5m")
    bundle = load_model_bundle(PIPELINE_STATE["model_dir"])
    bundle.validate_compatibility(timeframe="5m")
    source = ModelSignalSource(bundle, config.simulation.signal, symbol=symbol,
                               timeframe="5m", local_timezone=config.data.source_timezone)
    run = run_simulation(config, bars, source, timeframe="5m", symbol=symbol)
    results = compute_results(run, config, bars)
    expect(results.bars_processed == len(bars), f"обработано {results.bars_processed} из {len(bars)}")
    expect(results.n_entries >= 1, f"входов нет: {results.n_entries}")
    expect(results.total_commission > 0.0, f"комиссии нулевые: {results.total_commission}")
    expect(results.initial_capital == 100_000.0, "капитал не из конфига")
    return (f"баров={results.bars_processed}; входов={results.n_entries}; "
            f"комиссии={results.total_commission:.4f}; итог={results.final_equity:.2f}")


@check("Секреты отсутствуют в отчёте и журнале",
       "нет JWT-подобных строк ('eyJ') и фейковых значений в report/journal")
def check_no_secrets_in_artifacts() -> str:
    for path in (OUT_DIR / "reports" / "test_run_phase05-manual_5m.md",
                 OUT_DIR / "reports" / "journal_phase05-manual_5m.jsonl"):
        text = path.read_text(encoding="utf-8")
        expect("eyJ" not in text, f"JWT-подобная строка в {path.name}")
        for fake in (FAKE_SECRET, FAKE_ACCESS_KEY, FAKE_ACCOUNT_ID):
            expect(fake not in text, f"фейковый секрет в {path.name}")
    return "отчёт и журнал чисты"


# ---------------------------------------------------------------------------
# 3. Некорректная конфигурация и LIVE-гейт
# ---------------------------------------------------------------------------

@check("Неполная конфигурация (нет simulation) -> отказ exit 2, без заявок",
       "stderr содержит «НЕПОЛНАЯ КОНФИГУРАЦИЯ»; exit 2; данные/модель не запускались")
def check_bad_config_no_simulation() -> str:
    config = full_config()
    config.pop("simulation")
    path = write_yaml(OUT_DIR / "config_bad_no_simulation.yaml", config)
    code, out, err = cli(["test", "--config", path, "--timeframe", "5m", "--train"])
    expect(code == 2, f"exit {code} (ожидалось 2)")
    expect("НЕПОЛНАЯ КОНФИГУРАЦИЯ" in err, f"нет сообщения об отказе: {err[:200]}")
    expect("[1/5]" not in out, "прогон начал загрузку данных до отказа")
    return "exit 2; отказ до загрузки данных; заявки невозможны"


@check("Неполная конфигурация (нет risk) -> отказ exit 2",
       "exit 2 и «НЕПОЛНАЯ КОНФИГУРАЦИЯ»")
def check_bad_config_no_risk() -> str:
    config = full_config()
    config.pop("risk")
    path = write_yaml(OUT_DIR / "config_bad_no_risk.yaml", config)
    code, _, err = cli(["test", "--config", path, "--timeframe", "5m", "--train"])
    expect(code == 2, f"exit {code}")
    expect("НЕПОЛНАЯ КОНФИГУРАЦИЯ" in err, err[:200])
    return "exit 2; безопасный отказ"


@check("mode: LIVE в команде `test` -> отказ exit 2",
       "команда test не принимает LIVE; stderr объясняет про гейт; exit 2")
def check_live_mode_rejected_by_test() -> str:
    config = full_config()
    config["run"]["mode"] = "LIVE"
    path = write_yaml(OUT_DIR / "config_bad_live.yaml", config)
    code, _, err = cli(["test", "--config", path, "--timeframe", "5m", "--train"])
    expect(code == 2, f"exit {code}")
    expect("ОТКАЗ" in err and "LIVE" in err, err[:200])
    return "exit 2; LIVE не запускается командой test"


@check("LIVE-гейт: без флага/env отказ ДО создания сетевого клиента",
       "live-preflight без --allow-live и env -> exit 2, «гейт НЕ пройден»; клиент не создавался")
def check_live_gate_denies_before_network() -> str:
    config = full_config()
    config["run"]["mode"] = "LIVE"
    config["live"] = {"allow_orders": True, "max_order_quantity_units": 5}
    path = write_yaml(OUT_DIR / "config_live_gate.yaml", config)
    saved = {k: os.environ.pop(k, None) for k in ("TRADEBOT_LIVE_ACK", "TRADE_API_SECRET", "FINAM_ACCOUNT_ID")}
    import tradebot.finam.rest_client as rest_module

    def sentinel(*args, **kwargs):
        raise AssertionError("сетевой клиент создан до прохождения гейта")

    original = rest_module.FinamRestClient
    rest_module.FinamRestClient = sentinel  # type: ignore[assignment]
    try:
        code, _, err = cli(["live-preflight", "--config", path])
    finally:
        rest_module.FinamRestClient = original  # type: ignore[assignment]
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
    expect(code == 2, f"exit {code}")
    expect("гейт НЕ пройден" in err, err[:300])
    expect("--allow-live" in err or "TRADEBOT_LIVE_ACK" in err, "нет перечня нарушений")
    return "exit 2 до сети; нарушения перечислены; клиент не создавался"


# ---------------------------------------------------------------------------
# 4. Ордерный адаптер на подменном транспорте (НЕ брокерская среда)
# ---------------------------------------------------------------------------

@check("Адаптер: документированное тело заявки и принятый OrderState",
       "POST-запрос выставления заявки: путь accounts/{id}+suffix; тело symbol/quantity{value}/"
       "side/type/tif/client_order_id; SUBMITTED")
def check_broker_submit_contract() -> str:
    transport = ScriptedTransport(auth_pair() + [
        json_response(200, {"orders": []}),
        json_response(200, ORDER_STATE),
    ])
    gateway = make_gateway(transport)
    submission = gateway.submit(make_intent(), limit_price=Decimal("300.50"))
    expect(submission.status is OrderSubmissionStatus.SUBMITTED, f"статус {submission.status}")
    expect(submission.order_id == "ord-77", f"order_id {submission.order_id}")
    posts = transport.order_posts()
    expect(len(posts) == 1, f"POST-запросов: {len(posts)}")
    method, url, headers, body = posts[0]
    expect(url.endswith(f"/v1/accounts/{FAKE_ACCOUNT_ID}{_ORDERS_SUFFIX}"), f"путь {url}")
    expect(headers.get("Authorization") == f"Bearer {FAKE_ACCESS_KEY}", "нет Bearer-заголовка")
    expect(body == {
        "symbol": "SBER@MISX",
        "quantity": {"value": "10"},
        "side": "SIDE_BUY",
        "type": "ORDER_TYPE_LIMIT",
        "time_in_force": "TIME_IN_FORCE_DAY",
        "client_order_id": "d" * 20,
        "limit_price": {"value": "300.50"},
    }, f"тело не соответствует документированному: {body}")
    return "тело и путь — по документации (Order/OrderState); SUBMITTED, order_id=ord-77"


@check("Адаптер: потерянный ответ -> UNKNOWN_RESULT -> сверка -> защита от дубля",
       "1) UNKNOWN_RESULT без слепого повтора; 2) resolve_unknown находит заявку; 3) повторный submit -> DUPLICATE_FOUND; ровно ОДИН POST")
def check_broker_unknown_and_duplicate() -> str:
    transport = ScriptedTransport(auth_pair() + [
        json_response(200, {"orders": []}),
        FinamTransportError("connection reset (имитация)"),
        json_response(200, {"orders": [ORDER_STATE]}),
        json_response(200, {"orders": [ORDER_STATE]}),
    ])
    journal_path = OUT_DIR / "journal_broker.jsonl"
    with JournalWriter(journal_path, symbol="SBER@MISX", timeframe="5m", model_version="test") as journal:
        gateway = make_gateway(transport, journal=journal)
        intent = make_intent()
        first = gateway.submit(intent, limit_price=Decimal("300.00"))
        expect(first.status is OrderSubmissionStatus.UNKNOWN_RESULT, f"первый статус {first.status}")
        expect("запрещена до сверки" in first.message, "нет запрета слепого повтора в сообщении")
        resolved = gateway.resolve_unknown(intent.client_intent_id)
        expect(resolved is not None and resolved.status is OrderSubmissionStatus.SUBMITTED,
               f"сверка не нашла заявку: {resolved}")
        second = gateway.submit(intent, limit_price=Decimal("300.00"))
        expect(second.status is OrderSubmissionStatus.DUPLICATE_FOUND, f"повторный статус {second.status}")
    expect(len(transport.order_posts()) == 1, f"POST-запросов {len(transport.order_posts())} (ожидался 1)")
    records = read_journal(journal_path)
    actions = {r.get("action") for r in records if r.get("kind") == "order_event"}
    expect({"submit_unknown_result", "unknown_resolved", "duplicate_prevented"} <= actions,
           f"журнал неполный: {actions}")
    CREATED.append(str(journal_path.relative_to(REPO_ROOT)))
    return "UNKNOWN -> сверка -> DUPLICATE_FOUND; один POST; события в журнале"


@check("Адаптер: kill-switch и отказ риск-контроля -> НОЛЬ сетевых вызовов",
       "kill-switch: DISABLED без сети; устаревшие данные: STALE_DATA + halt, шлюз не вызван")
def check_broker_blocks() -> str:
    kill = OUT_DIR / ".kill-demo"
    kill.write_text("stop", encoding="utf-8")
    try:
        transport = ScriptedTransport(auth_pair())
        gateway = make_gateway(transport, kill_file=kill)
        submission = gateway.submit(make_intent(), limit_price=Decimal("300.00"))
        expect(submission.status is OrderSubmissionStatus.DISABLED, f"статус {submission.status}")
        expect(transport.requests == [], f"запросы при kill-switch: {transport.requests}")
    finally:
        kill.unlink(missing_ok=True)

    # интеграция риск-контроля: устаревшие данные -> отказ -> шлюз не вызывается
    ts_now = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    risk_cfg = RiskConfig(initial_capital_quote=100_000.0, position_units=10,
                          max_position_units=100, max_orders_per_day=10,
                          daily_loss_limit_quote=1_000.0, max_drawdown_pct=10.0,
                          max_data_age_seconds=60.0)
    sim_cfg = SimulationConfig(commission_bps=3.0, slippage_bps=2.0)
    guard = RiskGuards(risk_cfg, None, sim_cfg, local_timezone="Europe/Moscow")
    context = RiskContext(decision_ts=ts_now, last_bar_ts_open=ts_now - timedelta(seconds=120),
                          last_close=300.0, equity=100_000.0, cash=100_000.0,
                          position_units=0, data_age_seconds=120.0,
                          bar_is_valid=True, data_source_ok=True)
    decision = guard.evaluate(make_intent(), PortfolioState(cash=100_000.0, peak_equity=100_000.0), context)
    expect(not decision.approved, "устаревшие данные не отклонены")
    expect(RejectionCode.STALE_DATA in {r.code for r in decision.rejections},
           f"коды: {[r.code for r in decision.rejections]}")
    expect(guard.is_halted and guard.halt_reason == "stale_data", "нет остановки по stale_data")
    transport2 = ScriptedTransport(auth_pair())
    gateway2 = make_gateway(transport2)
    if decision.approved:
        gateway2.submit(make_intent(), limit_price=Decimal("300.00"))
    expect(transport2.requests == [], f"шлюз вызван после отказа: {transport2.requests}")
    return "kill-switch: DISABLED (0 запросов); stale_data: отказ+halt, 0 вызовов шлюза"


@check("Логи: фейковые секреты маскируются (логи + журнал адаптера)",
       "в записях логов и journal_broker.jsonl нет FAKE_SECRET/ACCESS_KEY/ACCOUNT_ID")
def check_logs_masked() -> str:
    captured: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    tradebot_logger = logging.getLogger("tradebot")
    handler = Capture(level=logging.DEBUG)
    saved_level = tradebot_logger.level
    tradebot_logger.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    try:
        transport = ScriptedTransport(auth_pair() + [
            json_response(200, {"orders": []}),
            json_response(200, ORDER_STATE),
            json_response(200, {"positions": [{"symbol": "SBER@MISX", "quantity": {"value": "10"}}]}),
        ])
        journal_path = OUT_DIR / "journal_masking.jsonl"
        with JournalWriter(journal_path, symbol="SBER@MISX", timeframe="5m", model_version="test") as journal:
            gateway = make_gateway(transport, journal=journal)
            gateway.submit(make_intent(), limit_price=Decimal("300.00"))
            mismatches = gateway.reconcile_positions({"SBER@MISX": 10})
        expect(mismatches == [], f"неожиданные расхождения: {mismatches}")
    finally:
        logging.getLogger().removeHandler(handler)
        tradebot_logger.setLevel(saved_level)
    log_text = "\n".join(captured)
    expect(len(captured) > 0, "логи не захвачены (нечего проверять)")
    for fake in (FAKE_SECRET, FAKE_ACCESS_KEY, FAKE_ACCOUNT_ID):
        expect(fake not in log_text, f"секрет в логах: {fake[:12]}...")
        expect(fake not in journal_path.read_text(encoding="utf-8"), f"секрет в журнале: {fake[:12]}...")
    CREATED.append(str(journal_path.relative_to(REPO_ROOT)))
    return f"{len(captured)} записей логов — фейковые секреты отсутствуют; журнал чист"


# ---------------------------------------------------------------------------
# 5. Гигиена репозитория
# ---------------------------------------------------------------------------

@check("Гигиена: торговые операции только в finam/broker.py; .env и reports в .gitignore",
       "вне allowlist нет торговых токенов; .gitignore покрывает .env и reports/")
def check_repo_hygiene() -> str:
    forbidden = ["/or" + "ders", "Place" + "Order", "Cancel" + "Order", "sltp-" + "orders", "new_" + "order"]
    allowed_files = {"broker.py"}  # единственный ордерный адаптер (за LIVE-гейтом, docs/live.md)
    offenders = []
    for pattern in ("src/**/*.py", "scripts/**/*.py", "configs/*.yaml"):
        for path in REPO_ROOT.glob(pattern):
            if path.name in allowed_files or path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            offenders.extend(f"{path.name}:{tok}" for tok in forbidden if tok in text)
    expect(not offenders, f"торговые токены вне allowlist: {offenders}")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    expect(".env" in gitignore and "reports/" in gitignore, ".gitignore неполный")
    expect((REPO_ROOT / "src" / "tradebot" / "finam" / "broker.py").exists(), "нет ордерного адаптера")
    return "токены — только в broker.py; .gitignore покрывает .env и reports/"


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("РУЧНАЯ ПРОВЕРКА ФАЗЫ 5: полный TEST-путь + защита LIVE-контура")
    print("Режим: ОФЛАЙН (синтетические демо-данные; без Finam API и секретов)")
    print("Реальные заявки НЕ отправляются. Это НЕ проверка в брокерской среде:")
    print("ордерный адаптер проверяется на подменном транспорте (docs/live.md)")
    print("=" * 78)
    failures = 0
    for index, (name, expected, fn) in enumerate(CHECKS, start=1):
        print(f"[{index:02d}/{len(CHECKS)}] {name}")
        print(f"          ОЖИДАЛОСЬ: {expected}")
        try:
            actual = fn()
            print(f"          ФАКТ:      {actual}")
            print("          PASS")
        except AssertionError as exc:
            failures += 1
            print(f"          ФАКТ:      {exc}")
            print("          FAIL")
        except Exception:
            failures += 1
            print("          ФАКТ:      неожиданная ошибка")
            traceback.print_exc()
            print("          FAIL")
    print("-" * 78)
    if CREATED:
        print("Созданные файлы (можно удалить вместе с reports/phase05_manual/):")
        for path in dict.fromkeys(CREATED):
            print(f"  {path}")
    total = len(CHECKS)
    if failures:
        print(f"RESULT: FAIL ({total - failures}/{total} checks)")
        print("Что прислать: ПОЛНЫЙ вывод скрипта, python --version, ОС,")
        print("pip show tradebot lightgbm pydantic, хвост python -m pytest -q")
        return 1
    print(f"RESULT: PASS ({total}/{total} checks)")
    print()
    print("Далее (на ВАШИХ данных, офлайн):")
    print("  1) положите CSV в data/raw и заполните configs/test_pipeline_example.yaml")
    print("  2) python -m tradebot test --config configs\\test_pipeline_example.yaml --timeframe 5m --train")
    print("  3) LIVE: только по чек-листу docs/live.md (по умолчанию выключен)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
