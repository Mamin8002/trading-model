#!/usr/bin/env python3
"""Ручная проверка фазы 3: признаки, целевая переменная, walk-forward, обучение, реестр.

Запуск (из корня репозитория, PowerShell):
    python scripts\\manual_checks\\check_phase_03_ml.py

Сценарий ПОЛНОСТЬЮ ЛОКАЛЬНЫЙ:
- данные — СИНТЕТИЧЕСКИЙ детерминированный набор (tradebot.data.synthetic),
  внешние загрузки и сеть не используются;
- секреты и Finam API не задействуются (торговые операции в проекте не реализованы);
- артефакты создаются в models/phase03_manual/ и reports/phase03_manual/
  (оба каталога исключены из git); в конце печатается список созданных файлов.

Ожидаемый результат: все проверки PASS, итог "RESULT: PASS", код возврата 0.
При любой неудаче — "RESULT: FAIL" и код возврата 1.

Проверка обучения на ВАШИХ данных (после заполнения configs/ml_example.yaml):
    python -m tradebot.cli.train --config configs\\ml_example.yaml --timeframe 5m
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tradebot.config import AppConfig, load_config  # noqa: E402
from tradebot.config.schema import FeaturesConfig, LabelConfig  # noqa: E402
from tradebot.data.synthetic import generate_synthetic_bars  # noqa: E402
from tradebot.features.builder import FeatureBuilder  # noqa: E402
from tradebot.features.spec import FeatureSpec  # noqa: E402
from tradebot.labels.target import TargetSpec, build_target  # noqa: E402
from tradebot.ml.dataset import assemble_dataset  # noqa: E402
from tradebot.ml.errors import InsufficientHistoryError, ModelCompatibilityError, TrainingError  # noqa: E402
from tradebot.ml.metrics import simulate_positions  # noqa: E402
from tradebot.ml.registry import load_model_bundle  # noqa: E402
from tradebot.ml.splits import assert_no_label_overlap, purged_walk_forward  # noqa: E402
from tradebot.ml.train import run_training  # noqa: E402

CHECKS: list[tuple[str, Callable[[], str | None]]] = []
CREATED_FILES: list[str] = []


def check(name: str):
    def decorator(fn: Callable[[], str | None]):
        CHECKS.append((name, fn))
        return fn

    return decorator


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


MODEL_DIR = REPO_ROOT / "models" / "phase03_manual"
REPORT_DIR = REPO_ROOT / "reports" / "phase03_manual"

CONFIG_DICT: dict[str, Any] = {
    "run": {"mode": "TEST", "experiment_id": "phase03-manual", "model_dir": "models/phase03_manual", "report_dir": "reports/phase03_manual"},
    "data": {
        "csv_dir": "data/raw",
        "symbols": [{"ticker": "SYNTH", "mic": "DEMO", "files": {}}],
        "timestamp_semantics": "bar_open",
        "source_timezone": "Europe/Moscow",
    },
    "timeframes": {
        "5m": {
            "enabled": True,
            "lookback_bars": 240,
            "horizon_bars": 6,
            "label": {"kind": "fwd_return_class", "neutral_band_bps": 5.0},
        },
        "15m": {"enabled": False, "lookback_bars": 240, "horizon_bars": 3},
    },
    "features": {"set": "v1"},
    "model": {
        "kind": "lightgbm",
        "seed": 42,
        "params": {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 30, "learning_rate": 0.1},
    },
    "validation": {
        "walk_forward": {"n_splits": 3, "min_train_bars": 500, "min_val_bars": 100, "audit_random_split": True},
        "thresholds": {"min_oos_auc": 0.52, "min_oos_trades": 3},
    },
    "evaluation": {"commission_bps": 3.0, "slippage_bps": 2.0, "decision_threshold": 0.5},
}


def make_config() -> AppConfig:
    return AppConfig.model_validate(json.loads(json.dumps(CONFIG_DICT)))


BARS = generate_synthetic_bars(2600, seed=7, timeframe="5m", symbol="SYNTH@DEMO")
HORIZON = 6


def _track_created(path: "str | Path") -> None:
    p = Path(path)
    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.is_file():
                CREATED_FILES.append(str(child.relative_to(REPO_ROOT)))
    else:
        CREATED_FILES.append(str(p.relative_to(REPO_ROOT)))


# ---------------------------------------------------------------------------
# 1. Признаки
# ---------------------------------------------------------------------------

@check("Синтетический набор детерминирован (один seed -> идентичные бары)")
def check_synthetic_determinism() -> str | None:
    again = generate_synthetic_bars(2600, seed=7, timeframe="5m", symbol="SYNTH@DEMO")
    expect(len(again) == len(BARS) == 2600, "число баров не совпало")
    for a, b in zip(BARS[:50], again[:50]):
        expect((a.ts_open, a.open, a.high, a.low, a.close, a.volume) == (b.ts_open, b.open, b.high, b.low, b.close, b.volume),
               "бары различаются при одном seed")
    return f"2600 баров 5m, {BARS[0].ts_open} .. {BARS[-1].ts_open} (SYNTHETIC/DEMO)"


@check("Признаки v1: 31 колонка, warmup, значения после прогрева без NaN")
def check_features_basic() -> str | None:
    spec = FeatureSpec.from_config(FeaturesConfig())
    frame = FeatureBuilder(spec, local_timezone="Europe/Moscow").build(BARS)
    expect(len(frame.columns) == 31, f"колонок: {len(frame.columns)}")
    expect(list(frame.columns) == list(spec.feature_names), "порядок колонок не совпадает с манифестом")
    expect(frame.iloc[: spec.warmup_bars].isna().any().any(), "в warmup-зоне ожидались NaN")
    tail = frame.iloc[spec.warmup_bars:]
    expect(bool(tail.notna().all().all()), "после warmup остались NaN")
    values = frame.to_numpy(dtype="float64")
    expect(not np.isinf(values[~np.isnan(values)]).any(), "в признаках inf")
    return f"warmup={spec.warmup_bars}, колонок=31, NaN после прогрева нет"


@check("Причинность: префикс-инвариантность + возмущение будущего не меняют признаки в t")
def check_features_causality() -> str | None:
    spec = FeatureSpec.from_config(FeaturesConfig())
    builder = FeatureBuilder(spec, local_timezone="Europe/Moscow")
    full = builder.build(BARS)
    # 1) обрезанная серия даёт те же значения в прошлых строках
    for cut in (spec.warmup_bars + 20, 800, len(BARS) - 25):
        prefix = builder.build(BARS[:cut])
        left = prefix.to_numpy(dtype="float64")
        right = full.iloc[:cut].to_numpy(dtype="float64")
        expect(np.allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True),
               f"префикс-инвариантность нарушена на cut={cut}")
    # 2) сильное возмущение ВСЕХ баров после t не меняет строки <= t
    from dataclasses import replace

    rng = np.random.RandomState(3)
    for t in (spec.warmup_bars + 5, 1000, len(BARS) - 20):
        perturbed = list(BARS)
        for i in range(t + 1, len(perturbed)):
            factor = 1.0 + rng.uniform(-0.3, 0.3)
            b = perturbed[i]
            perturbed[i] = replace(
                b,
                open=b.open * factor,
                high=max(b.high, b.open, b.close) * factor * 1.001,
                low=min(b.low, b.open, b.close) * factor * 0.999,
                close=b.close * factor,
                volume=int(b.volume * rng.uniform(0.1, 9.0)) + 1,
            )
        pert_frame = builder.build(perturbed)
        left = full.iloc[: t + 1].to_numpy(dtype="float64")
        right = pert_frame.iloc[: t + 1].to_numpy(dtype="float64")
        expect(np.allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True),
               f"признаки в t={t} изменились после возмущения будущего — УТЕЧКА")
    return "3 префикс-проверки + 3 возмущения будущего — изменений в прошлом нет"


@check("Целевая переменная: fwd_return и нейтральная полоса на вручную посчитанном примере")
def check_target_known_values() -> str | None:
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 105.0, 99.0, 100.2])
    spec = TargetSpec(kind="fwd_return_class", horizon_bars=1, neutral_band_bps=100.0)  # полоса ±1%
    y, info = build_target(close, spec)
    expect(float(y.iloc[3]) == 1.0, "+5% должен дать класс 1")
    expect(float(y.iloc[4]) == 0.0, "-5.8% должен дать класс 0")
    expect(np.isnan(y.iloc[0]), "движение 0% внутри полосы -> нейтраль (NaN)")
    expect(np.isnan(y.iloc[6]), "последняя строка (нет будущего) -> NaN")
    expect(info.n_neutral_dropped == 3 and info.n_tail_dropped == 1, f"счётчики: {info}")
    return "класс 1 при >+1%, класс 0 при <-1%, полоса и хвост -> NaN"


@check("Датасет: X без NaN, счётчики исключений, защита lookback < warmup")
def check_dataset() -> str | None:
    dataset = assemble_dataset(
        BARS,
        timeframe="5m",
        horizon_bars=HORIZON,
        lookback_bars=240,
        label_cfg=LabelConfig(kind="fwd_return_class", neutral_band_bps=5.0),
        features_cfg=FeaturesConfig(),
        symbol="SYNTH@DEMO",
    )
    expect(np.isfinite(dataset.X).all(), "X содержит NaN/inf")
    expect(dataset.X.shape[1] == 31, "ширина X != 31")
    expect(set(np.unique(dataset.y)) <= {0.0, 1.0}, "y не бинарный")
    expect(dataset.info.warmup_dropped > 0 and dataset.info.tail_dropped >= HORIZON,
           f"счётчики: {dataset.info}")
    expect(dataset.info.n_rows <= dataset.info.n_bars, "строк больше, чем баров")
    try:
        assemble_dataset(
            BARS[:400], timeframe="5m", horizon_bars=HORIZON, lookback_bars=10,
            label_cfg=LabelConfig(), features_cfg=FeaturesConfig(),
        )
    except TrainingError as exc:
        expect("lookback_bars" in str(exc), f"сообщение: {exc}")
    else:
        raise AssertionError("lookback < warmup не отклонён")
    return f"строк={dataset.info.n_rows} из {dataset.info.n_bars} баров (warmup={dataset.info.warmup_dropped}, нейтраль={dataset.info.neutral_dropped}, хвост={dataset.info.tail_dropped})"


# ---------------------------------------------------------------------------
# 2. Walk-forward разбиение
# ---------------------------------------------------------------------------

@check("Сплиты: хронология, expanding, embargo>=H, max(train)+H < min(val)")
def check_splits() -> str | None:
    dataset = assemble_dataset(
        BARS, timeframe="5m", horizon_bars=HORIZON, lookback_bars=240,
        label_cfg=LabelConfig(kind="fwd_return_class", neutral_band_bps=5.0),
        features_cfg=FeaturesConfig(), symbol="SYNTH@DEMO",
    )
    folds = purged_walk_forward(dataset.n_samples, n_splits=3, horizon_bars=HORIZON,
                                min_train_bars=500, min_val_bars=100)
    expect(len(folds) == 3, f"фолдов: {len(folds)}")
    prev_val_max = -1
    prev_train_size = -1
    boundaries = []
    for fold in folds:
        assert_no_label_overlap(fold)
        expect(fold.train_idx.max() + HORIZON < fold.val_idx.min(),
               f"фолд {fold.fold_id}: пересечение целевых горизонтов")
        expect(fold.train_idx[0] == 0, "train-окно не от начала (expanding)")
        expect(fold.n_train > prev_train_size, "train-окно не расширяется")
        expect(fold.val_idx.min() > prev_val_max, "валидационные окна пересекаются/не по времени")
        expect(fold.embargo_bars >= HORIZON, "embargo < горизонта")
        prev_val_max = fold.val_idx.max()
        prev_train_size = fold.n_train
        boundaries.append(f"F{fold.fold_id}: train[0..{fold.train_idx.max()}] embargo={fold.embargo_bars} val[{fold.val_idx.min()}..{fold.val_idx.max()}]")
    return "; ".join(boundaries)


@check("Недостаток истории: явная ошибка, train-метрики НЕ выдаются за качество")
def check_insufficient_history() -> str | None:
    try:
        purged_walk_forward(300, n_splits=5, horizon_bars=HORIZON, min_train_bars=500, min_val_bars=100)
    except InsufficientHistoryError as exc:
        expect(exc.required_bars is not None and exc.available_bars == 300, "нет данных о требуемом объёме")
        expect("обучающей" in str(exc), "сообщение не предупреждает про train-метрики")
        return f"required={exc.required_bars}, available=300 — остановка до обучения"
    raise AssertionError("InsufficientHistoryError не выброшена")


# ---------------------------------------------------------------------------
# 3. Обучение и отчёт
# ---------------------------------------------------------------------------

_RUN_CACHE: dict[str, Any] = {}


def _run_training_once(tag: str, model_dir: Path, report_dir: Path):
    config = make_config()
    started = time.time()
    result = run_training(config, "5m", bars=BARS, model_dir=model_dir, report_dir=report_dir)
    elapsed = time.time() - started
    _RUN_CACHE[tag] = {"result": result, "elapsed": elapsed}
    return result


@check("Обучение: 3 фолда, метрики каждого окна + OOS, базовые ориентиры, аудит")
def check_training_run() -> str | None:
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    result = _run_training_once("main", MODEL_DIR, REPORT_DIR)
    expect(len(result.folds) == 3, f"фолдов: {len(result.folds)}")
    total_val = sum(f.n_val for f in result.folds)
    expect(result.oos_metrics["n"] == total_val, "OOS n != сумма валидационных окон")
    for fold in result.folds:
        expect("roc_auc" in fold.model_metrics, f"фолд {fold.fold_id}: нет ROC-AUC")
        expect(fold.model_metrics["n"] == fold.n_val, "метрики фолда посчитаны не на val-окне")
    for name in ("majority", "momentum", "logreg"):
        expect(name in result.baselines_oos_metrics, f"нет базового ориентира {name}")
        expect(result.baselines_oos_metrics[name]["n"] == result.oos_metrics["n"],
               f"{name}: OOS n не совпадает")
    expect(result.audit is not None and result.audit.get("applicable"), "аудит random-split не выполнен")
    expect("delta" in result.audit, "аудит без дельты")
    auc = result.oos_metrics["roc_auc"]
    base_aucs = {n: result.baselines_oos_metrics[n]["roc_auc"] for n in ("majority", "momentum", "logreg")}
    return (f"OOS n={result.oos_metrics['n']}, AUC модели={auc:.4f}, ориентиры="
            + ", ".join(f"{n}={v:.4f}" if v is not None else f"{n}=н/д" for n, v in base_aucs.items())
            + f"; аудит delta={result.audit['delta']:.4f} flagged={result.audit['flagged']}")


@check("Торговые метрики OOS с издержками + допущения годового пересчёта")
def check_trading_metrics() -> str | None:
    result = _RUN_CACHE["main"]["result"]
    trading = result.oos_trading
    for key in ("n_trades", "net_return", "max_drawdown", "turnover", "gross_return", "cost_return"):
        expect(key in trading, f"нет метрики {key}")
    expect(trading["cost_return"] >= 0.0, "издержки отрицательные")
    expect(trading["turnover"] >= 0.0, "оборот отрицательный")
    if trading["n_trades"] > 0:
        expect(trading["turnover"] >= 2.0 * trading["n_trades"] - 1,
               "оборот меньше числа сделок (вход+выход)")
    assumptions = result.assumptions
    expect(assumptions["commission_bps"] == 3.0 and assumptions["slippage_bps"] == 2.0, "издержки не из конфига")
    expect(assumptions["bars_per_year"] > 0, "bars_per_year не определён")
    ann = result.oos_trading_annualized
    expect("cagr" in ann and "sharpe" in ann, "нет годовых метрик")
    # независимая сверка симулятора на маленьком ручном примере
    stats = simulate_positions(
        np.array([100.0, 101.0, 102.0, 103.0]),
        np.array([101.0, 102.0, 103.0, 104.0]),
        np.array([1.0, 1.0, 0.0, 0.0]),
        commission_bps=10.0,
        slippage_bps=0.0,
    )
    expected_net = 1.009 * (1.0 + (102.0 / 101.0 - 1.0)) * 0.999 - 1.0
    expect(abs(stats.net_return - expected_net) < 1e-9, f"ручная сверка симулятора: {stats.net_return} != {expected_net}")
    return (f"сделок={trading['n_trades']}, net={trading['net_return']:.4f}, maxDD={trading['max_drawdown']:.4f}, "
            f"turnover={trading['turnover']:.0f}; bars_per_year={assumptions['bars_per_year']:.0f} ({assumptions['bars_per_year_source']})")


@check("Детерминированность: повторный прогон с тем же seed даёт идентичные прогнозы")
def check_determinism() -> str | None:
    second_dir = MODEL_DIR.parent / "phase03_manual_repeat"
    second_reports = REPORT_DIR.parent / "phase03_manual_repeat"
    for stale in (second_dir, second_reports):
        if stale.exists():
            shutil.rmtree(stale)
    result2 = _run_training_once("repeat", second_dir, second_reports)
    result1 = _RUN_CACHE["main"]["result"]
    expect(result1.oos_metrics == result2.oos_metrics, "OOS-метрики различаются между прогонами")
    p1 = np.concatenate([f.val_proba for f in result1.folds])
    p2 = np.concatenate([f.val_proba for f in result2.folds])
    expect(np.array_equal(p1, p2), "прогнозы различаются между прогонами (seed/детерминизм)")
    _track_created(second_dir)
    _track_created(second_reports)
    return f"повторный прогон ({_RUN_CACHE['repeat']['elapsed']:.1f} c) идентичен основному"


@check("Вердикт по заранее заданным порогам (без утверждений о прибыльности)")
def check_verdict() -> str | None:
    verdict = _RUN_CACHE["main"]["result"].verdict
    checks = {c["check"]: c["passed"] for c in verdict["checks"]}
    expect("min_oos_auc" in checks and "min_oos_trades" in checks, f"checks: {checks}")
    expect("НЕ является утверждением о прибыльности" in verdict["text"], f"текст вердикта: {verdict['text']}")
    return f"passed={verdict['passed']}; {verdict['text']}"


# ---------------------------------------------------------------------------
# 4. Артефакты и реестр
# ---------------------------------------------------------------------------

@check("Артефакт модели: 4 файла; метаданные (ТФ, горизонт, признаки, версии, seed, период)")
def check_artifacts() -> str | None:
    result = _RUN_CACHE["main"]["result"]
    model_dir = Path(result.model_dir)
    for name in ("model.lgb.txt", "metadata.json", "feature_manifest.json", "oos_predictions.csv"):
        expect((model_dir / name).exists(), f"нет файла {name}")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    expect(metadata["timeframe"] == "5m", f"timeframe: {metadata['timeframe']}")
    expect(metadata["horizon_bars"] == HORIZON, f"horizon: {metadata['horizon_bars']}")
    expect(metadata["label"]["kind"] == "fwd_return_class", "метка")
    expect(metadata["model"]["seed"] == 42, "seed")
    expect(len(metadata["dataset"]["feature_names"]) == 31, "список признаков")
    for lib in ("python", "tradebot", "numpy", "pandas", "lightgbm", "scikit-learn"):
        expect(metadata["versions"].get(lib) not in (None, "", "not-installed"), f"нет версии {lib}")
    period = metadata["training_period"]
    expect(period["first_ts"] < period["last_ts"], "период обучения некорректен")
    expect(len(metadata["walk_forward"]["folds"]) == 3, "сводка фолдов")
    expect(metadata["config_hash"] and metadata["dataset"]["fingerprint"], "нет хэша конфига/fingerprint")
    predictions = pd.read_csv(model_dir / "oos_predictions.csv")
    expect(len(predictions) == result.oos_metrics["n"], "oos_predictions.csv не соответствует OOS")
    _track_created(model_dir)
    _track_created(Path(result.report_path))
    return f"{model_dir.name}: model+metadata+manifest+predictions; период {period['first_ts'][:16]}..{period['last_ts'][:16]}"


@check("Отчёт: допущения (комиссии/слиппедж/годовой пересчёт), таблица фолдов, дисклеймер")
def check_report() -> str | None:
    result = _RUN_CACHE["main"]["result"]
    text = Path(result.report_path).read_text(encoding="utf-8")
    for fragment in (
        "Дисклеймер", "Walk-forward разбиение", "Допущения симуляции и годового пересчёта",
        "комиссия", "слиппедж", "bars_per_year", "Аудит утечек", "Вердикт",
        "обещанием", "majority", "momentum", "logreg", "Воспроизводимость",
    ):
        expect(fragment.lower() in text.lower(), f"в отчёте нет фрагмента {fragment!r}")
    expect("| фолд |" in text.lower(), "нет таблицы фолдов")
    return f"{Path(result.report_path).name} ({len(text)} символов), все обязательные разделы на месте"


@check("Реестр: загрузка, совместимость ОК; несовместимый ТФ/признаки/горизонт отклоняются")
def check_registry() -> str | None:
    result = _RUN_CACHE["main"]["result"]
    bundle = load_model_bundle(result.model_dir)
    bundle.validate_compatibility(timeframe="5m", feature_names=bundle.feature_names,
                                  horizon_bars=HORIZON, label_kind="fwd_return_class")
    for kwargs, label in (
        ({"timeframe": "15m"}, "таймфрейм 15m"),
        ({"horizon_bars": 12}, "горизонт 12"),
        ({"feature_names": list(bundle.feature_names)[:-1] + ["unknown"]}, "чужой список признаков"),
        ({"label_kind": "fwd_return_reg"}, "другой вид метки"),
    ):
        try:
            bundle.validate_compatibility(**kwargs)
        except ModelCompatibilityError as exc:
            expect(len(exc.mismatches) >= 1, "пустой список расхождений")
        else:
            raise AssertionError(f"несовместимый контекст принят: {label}")
    proba = bundle.predict_proba(BARS[100:180], local_timezone="Europe/Moscow")
    expect(proba.shape == (80,), f"shape прогноза: {proba.shape}")
    expect(bool(np.isfinite(proba).all()) and bool(((proba >= 0) & (proba <= 1)).all()), "прогноз вне [0,1]")
    return "совместимость проверяется; 4 вида несоответствий отклонены; predict_proba работает"


@check("Гигиена: признаки не импортируют метки; артефакты в gitignore-каталогах")
def check_hygiene() -> str | None:
    for path in (REPO_ROOT / "src" / "tradebot" / "features").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        expect("labels" not in text, f"{path.name}: ссылка на модуль меток (labels)")
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    expect("models/" in gitignore and "reports/" in gitignore, "артефакты не исключены из git")
    expect(str(MODEL_DIR.relative_to(REPO_ROOT)).replace("\\", "/").startswith("models/"), "модель вне models/")
    return "features не зависят от labels; models/ и reports/ в .gitignore"


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 78)
    print("РУЧНАЯ ПРОВЕРКА ФАЗЫ 3: признаки, метки, walk-forward, обучение, реестр")
    print("Режим: ЛОКАЛЬНЫЙ (синтетические данные, без сети и секретов)")
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
    if CREATED_FILES:
        print("Созданные файлы:")
        for path in dict.fromkeys(CREATED_FILES):
            print(f"  {path}")
    total = len(CHECKS)
    if failures:
        print(f"RESULT: FAIL ({total - failures}/{total} checks)")
        print("Что прислать: полный вывод, python --version, pip show tradebot lightgbm pandas numpy scikit-learn")
        return 1
    print(f"RESULT: PASS ({total}/{total} checks)")
    print(f"Время основного прогона обучения: {_RUN_CACHE.get('main', {}).get('elapsed', float('nan')):.1f} c")
    print()
    print("Далее (на ВАШИХ данных): заполните configs/ml_example.yaml и выполните")
    print("  python -m tradebot.cli.train --config configs\\ml_example.yaml --timeframe 5m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
