#!/usr/bin/env python3
"""Ручная проверка Этапа 2b: обучение и walk-forward-валидация на воспроизводимом наборе.

Запуск (PowerShell, из корня проекта btc_bybit_bot):
    python scripts/manual_checks/check_stage2_model.py

Сеть и секреты НЕ используются: все проверки офлайн.
Часть B требует загруженных данных Этапа 2a (data/processed/*.clean.csv либо
data/manual_checks/*.clean.csv - подойдёт образец из check_stage2_data.py).

Создаваемые файлы:
    data/models/_manual_check_stage2b/<model_version>/model.joblib
    data/models/_manual_check_stage2b/<model_version>/model_card.json
    data/models/_manual_check_stage2b/<model_version>/walk_forward.json

Ожидаемый вывод: строки [PASS]/[FAIL] и итог "RESULT: PASS", код возврата 0.
Время выполнения: ~20-60 с (зависит от CPU и объёма данных).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import build_dataset, clean_klines  # noqa: E402
from btc_bot.data.store import load_klines_csv  # noqa: E402
from btc_bot.model.features import (  # noqa: E402
    RATIO_FEATURES,
    RAW_FEATURES,
    bar_features,
    build_features,
    feature_columns,
)
from btc_bot.model.predictor import ModelMismatchError, Predictor  # noqa: E402
from btc_bot.model.train import (  # noqa: E402
    build_card,
    find_latest_model_dir,
    library_versions,
    make_model_version,
    save_artifact,
    train_final,
)
from btc_bot.model.walk_forward import (  # noqa: E402
    label_resolution_times,
    walk_forward_validate,
)

INTERVAL = "5"
STEP_MS = 300_000
BASE = 1_700_000_100_000

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f" :: {detail}"
    print(line)


def synthetic_bars(n: int, seed: int = 7) -> pd.DataFrame:
    """Синтетическое случайное блуждание (явно НЕ реальные данные)."""
    rng = np.random.default_rng(seed)
    rows = []
    price = 50_000.0
    for i in range(n):
        o = price
        c = o * (1 + rng.normal(0, 0.002))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0008)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0008)))
        v = float(rng.uniform(1, 50))
        rows.append({"start_ms": BASE + i * STEP_MS, "open": o, "high": h, "low": l,
                     "close": c, "volume": v, "turnover": v * c, "is_closed": True})
        price = c
    return pd.DataFrame(rows)


# --------------------------------------------------------------- часть A
def check_a1_features() -> None:
    df = synthetic_bars(10)
    raw = build_features(df, "raw")
    ratio = build_features(df, "ratio")
    ok_cols = list(raw.columns) == RAW_FEATURES and list(ratio.columns) == RATIO_FEATURES
    r0 = df.iloc[0]
    ok_vals = (abs(float(ratio["open_rel"].iloc[0]) - r0.open / r0.close) < 1e-12
               and abs(float(ratio["high_rel"].iloc[0]) - r0.high / r0.close) < 1e-12
               and float(ratio["high_rel"].iloc[0]) >= 1.0 >= float(ratio["low_rel"].iloc[0]))
    ok_nofloat_err = not (raw.isna().any().any() or ratio.isna().any().any())
    bar = df.iloc[3].to_dict()
    ok_live = bar_features(bar, "ratio") == {k: float(v) for k, v in
                                             build_features(df.iloc[[3]], "ratio").iloc[0].items()}
    try:
        bar_features({**bar, "close": 0.0}, "ratio")
        ok_guard = False
    except ValueError:
        ok_guard = True
    record("A1 признаки raw/ratio: колонки, значения, single-bar == df-строка, guard close<=0",
           ok_cols and ok_vals and ok_nofloat_err and ok_live and ok_guard,
           f"cols_ok={ok_cols}, vals_ok={ok_vals}, live_ok={ok_live}, guard_ok={ok_guard}")


def check_a2_embargo_and_folds() -> None:
    n = 600
    df = synthetic_bars(n, seed=11)
    X, y, times, rep = build_dataset(df, horizon_bars=1, interval=INTERVAL,
                                     features=build_features(df, "ratio"))
    prices = df[["start_ms", "close"]]
    wf = walk_forward_validate(X, y, times, prices, INTERVAL, horizon_bars=1,
                               feature_set="ratio", n_folds=3, train_fraction=0.5,
                               threshold=0.55, seed=42, n_estimators=20)
    problems = []
    if len(wf.folds) != 3:
        problems.append(f"ожидали 3 окна, получили {len(wf.folds)}")
    prev_end = None
    for f in wf.folds:
        # embargo: все обучающие строки разрешаются меткой не позже начала окна
        res_train_end = int(label_resolution_times(np.array([f.train_end_ms]), INTERVAL, 1)[0])
        if res_train_end > f.test_start_ms:
            problems.append(f"окно {f.fold}: нарушение embargo (train_end resolution {res_train_end} > test_start {f.test_start_ms})")
        if f.train_n <= 0 or f.test_n <= 0:
            problems.append(f"окно {f.fold}: пустое train/test")
        if prev_end is not None and f.test_start_ms <= prev_end:
            problems.append(f"окно {f.fold}: тестовые окна перекрываются")
        prev_end = f.test_end_ms
        for key in ("accuracy", "majority_accuracy"):
            v = getattr(f, key)
            if not (0.0 <= v <= 1.0):
                problems.append(f"окно {f.fold}: {key}={v} вне [0,1]")
    a = wf.aggregate
    if not (0.0 <= a["accuracy"] <= 1.0) or a["n_test_total"] <= 0:
        problems.append(f"агрегат некорректен: {a}")
    record("A2 walk-forward: 3 окна, embargo по времени разрешения метки, нет перекрытий",
           not problems, "; ".join(problems) if problems else
           f"train_n={[f.train_n for f in wf.folds]}, test_n={[f.test_n for f in wf.folds]}, "
           f"acc={[round(f.accuracy, 3) for f in wf.folds]} (синтетика - уровень случайности ожидаем)")


def check_a3_artifact_roundtrip() -> None:
    df = synthetic_bars(400, seed=13)
    X, y, times, _ = build_dataset(df, 1, INTERVAL, features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=20)
    version = make_model_version("BTCUSDT", "spot", INTERVAL, 1, "ratio",
                                 ts=pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime())
    card = build_card(
        model_version=version, algorithm="xgboost", symbol="BTCUSDT", category="spot",
        interval=INTERVAL, horizon_bars=1, horizon_minutes=5, feature_set="ratio",
        features=feature_columns("ratio"), confidence_threshold=0.55,
        library_versions=library_versions(),
    )
    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td) / version
        paths = save_artifact(model, model_dir, card, walk_forward={"folds": []})
        pred = Predictor.load(model_dir, confidence_threshold=0.55)
        ok_files = all(Path(p).exists() for p in paths.values())
        ok_version = pred.model_version == version
        try:
            pred.ensure_compatible(symbol="BTCUSDT", category="spot", interval=INTERVAL, horizon_bars=1)
            ok_compat = True
        except ModelMismatchError:
            ok_compat = False
        try:
            pred.ensure_compatible(symbol="BTCUSDT", category="spot", interval=INTERVAL, horizon_bars=6)
            ok_mismatch = False
        except ModelMismatchError:
            ok_mismatch = True
        bar = df.iloc[-1].to_dict()
        p = pred.predict_bar(bar)
        ok_pred = (p.action in ("BUY", "SELL", "HOLD") and 0.5 <= p.confidence <= 1.0
                   and 0.0 <= p.prob_up <= 1.0 and p.model_version == version
                   and p.horizon_minutes == 5 and p.raw_direction in (1, -1))
        latest = find_latest_model_dir(td)
        ok_latest = latest is not None and latest.name == version
    record("A3 артефакт: save/load, model_card, ensure_compatible (+ отклонение чужого horizon), predict_bar",
           ok_files and ok_version and ok_compat and ok_mismatch and ok_pred and ok_latest,
           f"files={ok_files}, version={ok_version}, compat={ok_compat}, mismatch_guard={ok_mismatch}, "
           f"pred={ok_pred}, latest={ok_latest}; пример: action={p.action}, conf={p.confidence:.3f}")


def check_a4_hold_band() -> None:
    # шумовые данные: метка - монетка, независимая от признаков -> уверенность ~0.5
    rng = np.random.default_rng(42)
    n = 800
    df = synthetic_bars(n, seed=17)
    X = build_features(df, "ratio")
    y = pd.Series(rng.choice([-1, 1], size=n), name="y")
    times = df["start_ms"]
    # последние H строк не имеют будущего бара - для шума это не важно, обрежем
    keep = slice(0, n - 1)
    model = train_final(X.iloc[keep], y.iloc[keep], seed=42, n_estimators=30)
    card = build_card(model_version="noise-test", algorithm="xgboost", symbol="BTCUSDT",
                      category="spot", interval=INTERVAL, horizon_bars=1, horizon_minutes=5,
                      feature_set="ratio", features=feature_columns("ratio"),
                      confidence_threshold=0.55, library_versions=library_versions())
    with tempfile.TemporaryDirectory() as td:
        save_artifact(model, Path(td), card)
        pred = Predictor.load(td, confidence_threshold=0.55)
        actions = [pred.predict_bar(df.iloc[i].to_dict()).action for i in range(0, 200, 4)]
    n_hold = actions.count("HOLD")
    # прямая проверка маппинга
    a1, d1, c1 = pred.action_from_prob_up(0.52)
    a2, d2, c2 = pred.action_from_prob_up(0.90)
    a3, d3, c3 = pred.action_from_prob_up(0.08)
    ok_map = (a1, d1) == ("HOLD", 1) and (a2, d2) == ("BUY", 1) and (a3, d3) == ("SELL", -1) \
        and abs(c1 - 0.52) < 1e-9 and abs(c2 - 0.90) < 1e-9 and abs(c3 - 0.92) < 1e-9
    record("A4 HOLD-полоса: маппинг вероятностей + воздержание на шумовой модели",
           ok_map and n_hold > 0,
           f"mapping_ok={ok_map}, HOLD на шумовых данных: {n_hold}/{len(actions)} "
           f"(BUY={actions.count('BUY')}, SELL={actions.count('SELL')})")


# --------------------------------------------------------------- часть B
def find_real_dataset() -> Path | None:
    candidates = sorted((PROJECT_ROOT / "data" / "processed").glob("*.clean.csv"),
                        key=lambda p: p.stat().st_size, reverse=True)
    if candidates:
        return candidates[0]
    samples = sorted((PROJECT_ROOT / "data" / "manual_checks").glob("*.clean.csv"),
                     key=lambda p: p.stat().st_size, reverse=True)
    return samples[0] if samples else None


def check_b_real_smoke(cfg) -> None:
    thr = cfg.model.confidence_threshold if cfg.model.confidence_threshold is not None else 0.55
    data_path = find_real_dataset()
    if data_path is None:
        record("B1 реальные данные: загрузка + walk-forward + артефакт", False,
               "нет данных: сначала запустите scripts/download_history.py "
               "(или scripts/manual_checks/check_stage2_data.py для малого образца)")
        return
    df_raw = load_klines_csv(data_path)
    df_clean, prep = clean_klines(df_raw, cfg.exchange.interval)
    if len(df_clean) > 20_000:
        df_clean = df_clean.tail(20_000).reset_index(drop=True)
    problems = []
    if len(df_clean) < 500:
        problems.append(f"слишком мало баров: {len(df_clean)}")
    H = cfg.model.horizon_bars
    X, y, times, ds = build_dataset(df_clean, H, cfg.exchange.interval,
                                    features=build_features(df_clean, "ratio"))
    prices = df_clean[["start_ms", "close"]]
    wf = walk_forward_validate(X, y, times, prices, cfg.exchange.interval, horizon_bars=H,
                               feature_set="ratio", n_folds=3, train_fraction=0.5,
                               threshold=thr,
                               seed=cfg.model.random_seed, n_estimators=40)
    prev_end = None
    for f in wf.folds:
        res = int(label_resolution_times(np.array([f.train_end_ms]), cfg.exchange.interval, H)[0])
        if res > f.test_start_ms:
            problems.append(f"окно {f.fold}: embargo нарушен")
        if prev_end is not None and f.test_start_ms <= prev_end:
            problems.append(f"окно {f.fold}: перекрытие окон")
        prev_end = f.test_end_ms
        if not (0.0 <= f.accuracy <= 1.0):
            problems.append(f"окно {f.fold}: accuracy вне [0,1]")

    # финальное обучение на подмножестве + артефакт в отдельном тестовом каталоге
    model = train_final(X, y, seed=cfg.model.random_seed, n_estimators=40)
    version = make_model_version(cfg.exchange.symbol, cfg.exchange.category,
                                 cfg.exchange.interval, H, "ratio")
    models_root = PROJECT_ROOT / "data" / "models" / "_manual_check_stage2b"
    if models_root.exists():
        shutil.rmtree(models_root)
    card = build_card(
        model_version=version, algorithm="xgboost",
        params={"n_estimators": 40, "seed": cfg.model.random_seed, "note": "manual check subset"},
        symbol=cfg.exchange.symbol, category=cfg.exchange.category, interval=cfg.exchange.interval,
        horizon_bars=H, horizon_minutes=H * cfg.exchange.interval_minutes,
        feature_set="ratio", features=feature_columns("ratio"),
        confidence_threshold=thr,
        training_data={"file": str(data_path), "n_bars": int(ds.n_rows)},
        walk_forward_summary=wf.aggregate, library_versions=library_versions(),
    )
    save_artifact(model, models_root / version, card, walk_forward=wf.to_dict())
    pred = Predictor.load(models_root / version, thr)
    pred.ensure_compatible(symbol=cfg.exchange.symbol, category=cfg.exchange.category,
                           interval=cfg.exchange.interval, horizon_bars=H)
    last_bar = df_clean.iloc[-1].to_dict()
    p = pred.predict_bar(last_bar)
    if p.action not in ("BUY", "SELL", "HOLD") or not (0.5 <= p.confidence <= 1.0):
        problems.append(f"некорректное предсказание: {p}")
    found = find_latest_model_dir(models_root)
    if found is None or found.name != version:
        problems.append("find_latest_model_dir не нашла сохранённую модель")

    agg = wf.aggregate
    record("B1 реальные данные: walk-forward(3 окна, embargo) + артефакт + предсказание последнего бара",
           not problems,
           "; ".join(problems) if problems else
           f"файл={data_path.name} (последние {len(df_clean)} баров), "
           f"acc по окнам={[round(f.accuracy, 4) for f in wf.folds]}, "
           f"majority={agg['majority_accuracy']:.4f}, momentum={agg['momentum_accuracy']:.4f}; "
           f"последний бар {pd.Timestamp(int(times.iloc[-1]), unit='ms', tz='UTC'):%Y-%m-%d %H:%M} UTC -> "
           f"{p.action} conf={p.confidence:.3f} version={p.model_version}; "
           f"артефакт: data/models/_manual_check_stage2b/{version}/")


def check_b2_train_script_help() -> None:
    proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "train_model.py"), "--help"],
                          capture_output=True, text=True, timeout=120)
    ok = proc.returncode == 0 and "--feature-set" in proc.stdout and "--quick" in proc.stdout
    record("B2 scripts/train_model.py --help работает (офлайн, без сети)", ok,
           f"rc={proc.returncode}" + ("" if ok else f", stderr={proc.stderr[:200]}"))


def main() -> int:
    print("== Этап 2b: обучение и walk-forward-валидация (офлайн) ==")
    cfg = load_config()
    print(f"Конфиг: {cfg.exchange.category}/{cfg.exchange.symbol} interval={cfg.exchange.interval}, "
          f"horizon_bars={cfg.model.horizon_bars} ({cfg.horizon_minutes} мин), "
          f"threshold={cfg.model.confidence_threshold if cfg.model.confidence_threshold is not None else '0.55 (дефолт)'},"
          f" seed={cfg.model.random_seed}")
    check_a1_features()
    check_a2_embargo_and_folds()
    check_a3_artifact_roundtrip()
    check_a4_hold_band()
    check_b_real_smoke(cfg)
    check_b2_train_script_help()

    failed = [name for name, ok, _ in _results if not ok]
    print()
    if failed:
        print(f"RESULT: FAIL ({len(failed)} проваленных: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (проверок пройдено: {len(_results)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
