#!/usr/bin/env python3
"""Ручная проверка полного офлайн-сценария TEST + защита LIVE-контура (крипторежим).

Запуск (из корня репозитория, PowerShell):
    python scripts\\manual_checks\\check_phase_05_test_run.py

Что делает сценарий (полностью ОФЛАЙН, детерминированно):
  1) готовит небольшой демонстрационный набор СИНТЕТИЧЕСКИХ 5m-баров (SYNTH-USDT,
     640 штук, непрерывная сетка 24/7, seed фиксирован) и конфигурацию TEST;
  2) запускает единой командой полный TEST-путь: чтение CSV -> обучение модели
     (walk-forward) -> сигналы -> риск-контроль -> симулятор -> результаты,
     отчёт и журнал;
  3) повторно прогоняет симуляцию с обученным артефактом и проверяет метрики;
  4) проверяет сценарии НЕКОРРЕКТНОЙ конфигурации: неполный risk/simulation и
     mode: LIVE в команде `test` -> отказ с кодом 2 БЕЗ каких-либо заявок;
  5) проверяет LIVE-гейт: интеграция с криптобиржей НЕ реализована
     (exchange/base.py), поэтому гейт отказывает даже при полном явном
     «включении» (флаг + env + live-секция) — сетевых вызовов не существует;
  6) проверяет, что ФЕЙКОВЫЕ значения секретов не попадают в отчёт/журнал,
     а в коде нет торговых операций и следов прежней брокерской интеграции.

ВАЖНО:
- Сценарий НЕ обращается к сети, НЕ требует секретов и НЕ отправляет заявки.
  Подключение к криптобирже не реализовано: LIVE-торговля невозможна (docs/live.md).
- Ожидаемые файлы после успешного запуска (создаются в reports/phase05_manual/,
  каталог исключён из git):
      data/SYNTH_M5.csv                     — демонстрационный набор (СИНТЕТИКА)
      config.yaml                           — конфигурация TEST-прогона
      models/phase05-manual_5m_<hash>/      — артефакт модели (model.lgb.txt,
                                              metadata.json, feature_manifest.json)
      reports/test_run_phase05-manual_5m.md — отчёт симуляции с допущениями
      reports/journal_phase05-manual_5m.jsonl — журнал решений/исполнений
- Ожидаемый вывод: по каждой проверке «ОЖИДАЛОСЬ / ФАКТ / PASS», в конце
  «RESULT: PASS (13/13 checks)» и код возврата 0.

Если проверка завершилась ошибкой, пришлите:
  1. ПОЛНЫЙ вывод скрипта (от «===» до «RESULT: ...»);
  2. `python --version`, операционная система;
  3. `pip show tradebot lightgbm pydantic` (версии);
  4. хвост `python -m pytest` (последние ~20 строк).

Код возврата: 0 — все проверки PASS; 1 — есть FAIL.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import shutil
import subprocess
import sys
import traceback
import warnings
from datetime import datetime, timezone
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
from tradebot.data.synthetic import generate_synthetic_bars  # noqa: E402
from tradebot.execution.journal import read_journal  # noqa: E402
from tradebot.ml.registry import load_model_bundle  # noqa: E402
from tradebot.ml.train import load_bars_for_timeframe  # noqa: E402
from tradebot.strategy.signals import ModelSignalSource  # noqa: E402

UTC = timezone.utc
OUT_DIR = REPO_ROOT / "reports" / "phase05_manual"
CREATED: list[str] = []

# ФЕЙКОВЫЕ значения — не являются учётными данными (проверки офлайн).
# Используются только в негативной проверке «их нет в артефактах».
FAKE_SECRET = "FAKE-SECRET-PHASE5-DO-NOT-USE"
FAKE_ACCESS_KEY = "FAKE-ACCESS-KEY-PHASE5-DO-NOT-USE"
FAKE_ACCOUNT_ID = "FAKEACCOUNT555"

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
                 "symbols": [{"ticker": "SYNTH-USDT", "files": {"5m": "SYNTH_M5.csv"}}],
                 "timestamp_semantics": "bar_open", "source_timezone": "UTC"},
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


@check("Демонстрационный набор: 640 СИНТЕТИЧЕСКИХ 5m-баров 24/7 + конфигурация TEST",
       "SYNTH_M5.csv: 641 строка (заголовок+640), сетка непрерывная (крипторежим); config.yaml записан; live-секции нет")
def check_demo_dataset() -> str:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "data").mkdir(parents=True)
    bars = generate_synthetic_bars(640, seed=7, timeframe="5m", symbol="SYNTH-USDT", continuous=True)
    # непрерывность сетки 24/7: шаг между соседними барами постоянен
    from datetime import timedelta
    steps = {(b.ts_open - a.ts_open) for a, b in zip(bars, bars[1:])}
    expect(steps == {timedelta(minutes=5)}, f"сетка не непрерывна: {steps}")
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
    return f"CSV 640 баров 24/7 (СИНТЕТИКА) + config.yaml записаны в {OUT_DIR.name}/"


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


@check("LIVE-гейт: отказ даже при полном «включении» — биржа не подключена",
       "live-preflight с флагом/env/live-секцией -> exit 2, «гейт НЕ пройден», причина — интеграция не реализована")
def check_live_gate_always_denies() -> str:
    config = full_config()
    config["run"]["mode"] = "LIVE"
    config["live"] = {"allow_orders": True, "max_order_quantity_units": 5,
                      "kill_switch_file": str(OUT_DIR / "no-such-kill-switch")}
    path = write_yaml(OUT_DIR / "config_live_gate.yaml", config)
    os.environ["TRADEBOT_LIVE_ACK"] = "ENABLE-LIVE"
    try:
        code_full, _, err_full = cli(["live-preflight", "--config", path, "--allow-live"])
        saved = os.environ.pop("TRADEBOT_LIVE_ACK", None)
        code_plain, _, err_plain = cli(["live-preflight", "--config", path])
        if saved is not None:
            os.environ["TRADEBOT_LIVE_ACK"] = saved
    finally:
        os.environ.pop("TRADEBOT_LIVE_ACK", None)
    expect(code_full == 2, f"exit {code_full} при полном включении (ожидалось 2)")
    expect("гейт НЕ пройден" in err_full, err_full[:300])
    expect("НЕ реализована" in err_full, f"нет причины про биржевую интеграцию: {err_full[:300]}")
    expect(code_plain == 2 and "гейт НЕ пройден" in err_plain, "без флага гейт тоже должен отказывать")
    return "exit 2 в обоих случаях; причина — интеграция с биржей не реализована; сеть не задействована"


# ---------------------------------------------------------------------------
# 4. Гигиена репозитория
# ---------------------------------------------------------------------------

@check("Гигиена: торговых операций нет нигде; .env и reports в .gitignore",
       "ни в одном файле src/scripts/configs нет торговых токенов и следов прежней интеграции; .gitignore покрывает .env и reports/")
def check_repo_hygiene() -> str:
    # токены собраны из фрагментов, чтобы проверка не обнаруживала саму себя
    forbidden = ["/or" + "ders", "Place" + "Order", "Cancel" + "Order", "sltp-" + "orders", "new_" + "order"]
    legacy = ["fi" + "nam", "MO" + "EX", "MI" + "SX"]
    offenders = []
    for pattern in ("src/**/*.py", "scripts/**/*.py", "configs/*.yaml"):
        for path in REPO_ROOT.glob(pattern):
            if path.name == Path(__file__).name:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            offenders.extend(f"{path.name}:{tok}" for tok in forbidden if tok in text)
            offenders.extend(f"{path.name}:{tok}" for tok in legacy if tok.lower() in text.lower())
    expect(not offenders, f"запрещённые токены: {offenders}")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    expect(".env" in gitignore and "reports/" in gitignore, ".gitignore неполный")
    # прежний брокерский пакет удалён и не импортируется (имя из фрагментов)
    import importlib.util
    expect(importlib.util.find_spec("tradebot.fi" + "nam") is None, "прежний брокерский модуль импортируем")
    from tradebot.exchange import EXCHANGE_INTEGRATION_AVAILABLE
    expect(EXCHANGE_INTEGRATION_AVAILABLE is False, "интеграция объявлена доступной")
    return "торговых токенов нет; прежняя интеграция удалена; .gitignore покрывает .env и reports/"


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("РУЧНАЯ ПРОВЕРКА: полный TEST-путь + защита LIVE-контура (крипторежим)")
    print("Режим: ОФЛАЙН (синтетические демо-данные; без сети и секретов)")
    print("Реальные заявки НЕ отправляются: подключение к криптобирже не реализовано,")
    print("LIVE заблокирован многофакторным гейтом (docs/live.md)")
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
        print("pip show tradebot lightgbm pydantic, хвост python -m pytest")
        return 1
    print(f"RESULT: PASS ({total}/{total} checks)")
    print()
    print("Далее (на ВАШИХ данных, офлайн):")
    print("  1) положите CSV криптопар в data/raw и заполните configs\\test_pipeline_example.yaml")
    print("  2) python -m tradebot test --config configs\\test_pipeline_example.yaml --timeframe 5m --train")
    print("  3) LIVE: недоступен — подключение к бирже не реализовано (docs/live.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
