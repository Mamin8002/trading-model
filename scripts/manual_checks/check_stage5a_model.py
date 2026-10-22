#!/usr/bin/env python3
r"""Ручная проверка Этапа 5a: модель v2 (горизонт 10-15 мин), отсутствие утечек, торговый симулятор.

Запуск (PowerShell, из корня проекта):
    python scripts\manual_checks\check_stage5a_model.py

Полностью офлайн (без сети и секретов). Часть A - синтетические детерминированные
данные; часть B - подмножество загруженных реальных данных (data/processed/*.clean.csv
или образец из data/manual_checks/). Файлы создаются только во временных каталогах.

Проверяются:
  A1 v2-признаки не зависят от будущих баров (tail-invariance)
  A2 горизонт H=3 (15 мин): цель, warm-up (21 строка), исключение строк без будущего бара
  A3 history_features == строка build_features; guard недостаточной истории
  A4 торговый симулятор: ручной расчёт net с комиссиями/слиппеджем, приоритет SL над TP,
     запрет входа в последних H барах окна (нет будущих данных)
  A5 выбор порога БЕЗ утечки: сетка, детерминизм, walk-forward c threshold_mode=selected
  A6 adopt-gate: замена только если v2 лучше И текущей модели, И majority-baseline
  A7 adopted=false артефакты не выбираются как production
  B1 реальные данные (подмножество): walk-forward H=3 v2 с симуляцией сделок, embargo,
     метрики конечны, majority совершает сделки
  B2 финальный пайплайн: select_final_threshold + train + card + predict_history (15 мин),
     predict_bar для v2 отклоняется, ensure_compatible по горизонту

Ожидаемый вывод: [PASS] A1..A7, B1..B2, "RESULT: PASS (проверок пройдено: 9)", код 0.
Время выполнения: ~30-90 с (зависит от CPU).
"""

from __future__ import annotations

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
    V2_FEATURES,
    InsufficientHistoryError,
    build_features,
    feature_columns,
    history_features,
    v2_min_bars,
)
from btc_bot.model.predictor import ModelMismatchError, Predictor  # noqa: E402
from btc_bot.model.trading_sim import (  # noqa: E402
    DEFAULT_THR_GRID,
    ExitRulesSim,
    TradingCosts,
    decide_adoption,
    select_threshold,
    simulate_trades,
)
from btc_bot.model.train import (  # noqa: E402
    build_card,
    find_latest_model_dir,
    library_versions,
    make_model_version,
    save_artifact,
    train_final,
)
from btc_bot.model.walk_forward import label_resolution_times, walk_forward_validate  # noqa: E402

INTERVAL = "5"
STEP = 300_000
H = 3  # 15 минут
BASE = 1_800_000_200_000 - (1_800_000_200_000 % STEP)

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


def synthetic_bars(n: int, seed: int = 51, start_price: float = 60_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, price = [], start_price
    for i in range(n):
        o = price
        c = o * (1 + rng.normal(0, 0.002))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0008)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0008)))
        v = float(rng.uniform(5, 50))
        rows.append({"start_ms": BASE + i * STEP, "open": o, "high": h, "low": l,
                     "close": c, "volume": v, "turnover": v * c, "is_closed": True})
        price = c
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- A1
def check_a1_no_future_dependence() -> None:
    df = synthetic_bars(120)
    full = build_features(df, "v2", horizon_bars=H)
    k = 100
    cut = build_features(df.iloc[:k].reset_index(drop=True), "v2", horizon_bars=H)
    # строка k-1 (и несколько предыдущих) должны совпадать: будущие бары не влияют
    tail_full = full.iloc[k - 5:k]
    tail_cut = cut.iloc[k - 5:k]
    ok = np.allclose(tail_full.to_numpy(dtype="float64"), tail_cut.to_numpy(dtype="float64"),
                     rtol=0, atol=1e-12, equal_nan=True)
    # warm-up NaN только в начале
    first_valid = full.apply(lambda col: col.first_valid_index()).max()
    ok_warmup = int(first_valid) == 20  # 21-я строка (индекс 20) - первая полная
    record("A1 v2-признаки не зависят от будущих баров (обрезание хвоста не меняет признаки)",
           bool(ok and ok_warmup), f"tail_equal={ok}, first_valid_row={int(first_valid)} (ожидали 20)")


# ----------------------------------------------------------------------- A2
def check_a2_horizon_and_warmup() -> None:
    n = 60
    df = synthetic_bars(n, seed=53)
    feats = build_features(df, "v2", horizon_bars=H)
    X, y, times, rep = build_dataset(df, horizon_bars=H, interval=INTERVAL, features=feats)
    # независимый пересчёт цели для нескольких строк (n_rows = 60-20-3 = 37)
    close_by_time = dict(zip(df["start_ms"], df["close"]))
    ok_target = True
    for idx in (0, 15, 36):
        t = int(times.iloc[idx])
        fut = close_by_time.get(t + H * STEP)
        exp = 1 if (fut is not None and fut > close_by_time[t]) else -1
        if int(y.iloc[idx]) != exp:
            ok_target = False
    ok_warm = rep.n_dropped_warmup == 20          # строки 0..19: vol_ratio_20 валиден с индекса 20
    ok_future = rep.n_dropped_no_future_bar == H  # последние 3 строки без будущего бара
    ok_nan = not X.isna().any().any()
    # строк = n - 20 (warmup) - H (без будущего бара); множества не пересекаются при n=60
    ok_rows = rep.n_rows == (n - 20 - H)
    ok_cols = list(X.columns) == V2_FEATURES
    record("A2 горизонт H=3 (15 мин): цель по close(t+3Δ), warm-up 20 строк исключено, без будущего - 3, NaN нет",
           ok_target and ok_warm and ok_future and ok_nan and ok_rows and ok_cols,
           f"target_ok={ok_target}, warmup={rep.n_dropped_warmup}, no_future={rep.n_dropped_no_future_bar}, "
           f"rows={rep.n_rows} (ожидали {n - 20 - H}), cols_ok={ok_cols}")


# ----------------------------------------------------------------------- A3
def check_a3_history_features() -> None:
    df = synthetic_bars(80, seed=57)
    feats = build_features(df, "v2", horizon_bars=H)
    bars = [{"start_ms": int(r.start_ms), "open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": r.volume, "turnover": r.turnover}
            for r in df.itertuples()]
    i = len(bars) - 1
    live = history_features(bars, "v2", horizon_bars=H)
    row = feats.iloc[i]
    ok_eq = all(abs(live[c] - float(row[c])) < 1e-12 for c in V2_FEATURES)
    ok_insufficient = False
    try:
        history_features(bars[-(v2_min_bars(H) - 1):], "v2", horizon_bars=H)
    except InsufficientHistoryError:
        ok_insufficient = True
    # window 25 баров (минимум 21) - достаточно
    ok_min = len(history_features(bars[-v2_min_bars(H):], "v2", horizon_bars=H)) == len(V2_FEATURES)
    record("A3 history_features(live) == build_features(история); guard нехватки баров",
           ok_eq and ok_insufficient and ok_min,
           f"equal={ok_eq}, insufficient_guard={ok_insufficient}, min_window_ok={ok_min}")


# ----------------------------------------------------------------------- A4
def check_a4_simulator_math() -> None:
    costs = TradingCosts(fee_pct=0.1, slippage_pct=0.02, order_size_quote=100.0)
    # (a) выход по горизонту H=1: close 100 -> 102
    ohlc = pd.DataFrame({
        "start_ms": [BASE, BASE + STEP],
        "open": [100.0, 101.0], "high": [100.5, 102.5], "low": [99.5, 100.8],
        "close": [100.0, 102.0],
    })
    r = simulate_trades(ohlc, np.array([1.0, 1.0]), 0.55, costs, ExitRulesSim(), 1, INTERVAL)
    entry_eff = 100.0 * (1 + 0.0002)
    exit_eff = 102.0 * (1 - 0.0002)
    exp_net = (exit_eff / entry_eff - 1.0) - 2 * 0.001
    ok_a = (r["n_trades"] == 1 and abs(r["avg_net_pct"] / 100.0 - exp_net) < 1e-12
            and r["exit_reasons"] == {"horizon": 1})
    # (b) SL приоритетнее TP в одном баре; гэп через уровень -> исполнение по open
    ohlc2 = pd.DataFrame({
        "start_ms": [BASE, BASE + STEP, BASE + 2 * STEP],
        "open": [100.0, 98.0, 98.5], "high": [100.5, 101.5, 99.0],
        "low": [99.5, 97.5, 98.0], "close": [100.0, 98.5, 98.5],
    })
    rules = ExitRulesSim(take_profit_pct=1.0, stop_loss_pct=1.0)  # TP=101, SL=99; бар2 задевает оба
    r2 = simulate_trades(ohlc2, np.array([1.0, 1.0, 1.0]), 0.55, costs, rules, 2, INTERVAL)
    sl_level = 99.0
    exp_exit_px = min(98.0, sl_level)  # гэп вниз -> по open
    exp_exit_eff = exp_exit_px * (1 - 0.0002)
    exp_net2 = (exp_exit_eff / entry_eff - 1.0) - 0.002
    ok_b = (r2["n_trades"] == 1 and r2["exit_reasons"] == {"stop_loss": 1}
            and abs(r2["avg_net_pct"] / 100.0 - exp_net2) < 1e-12)
    # (c) вход в последних H барах окна запрещён (горизонт выходит за окно)
    ohlc3 = pd.DataFrame({
        "start_ms": [BASE, BASE + STEP, BASE + 2 * STEP],
        "open": [100.0, 100.0, 100.0], "high": [101.0, 101.0, 101.0],
        "low": [99.0, 99.0, 99.0], "close": [100.0, 100.5, 101.0],
    })
    r3 = simulate_trades(ohlc3, np.array([1.0, 1.0, 1.0]), 0.55, costs, ExitRulesSim(), 1, INTERVAL)
    # H=1: бар0 вход -> бар1 выход; бар1 пропуск (бар выхода); бар2 вход за окном -> SKIP
    ok_c = r3["n_trades"] == 1 and r3["skipped_entries"] == 1
    record("A4 симулятор: net=gross-2*fee-слиппедж (сверено вручную), SL>TP в баре, гэп по open, entry-skip у границы окна",
           ok_a and ok_b and ok_c,
           f"horizon_exit={ok_a} (net {r['avg_net_pct']:+.4f}%), sl_priority={ok_b} "
           f"(net {r2['avg_net_pct']:+.4f}%), edge_skip={ok_c}")


# ----------------------------------------------------------------------- A5
def check_a5_threshold_selection() -> None:
    rng = np.random.default_rng(7)
    n = 400
    ohlc = synthetic_bars(n, seed=59)[["start_ms", "open", "high", "low", "close"]]
    prob_noise = rng.uniform(0.35, 0.65, size=n)
    costs = TradingCosts(0.1, 0.02, 100.0)
    thr1, info1 = select_threshold(ohlc, prob_noise, costs, ExitRulesSim(), 1, INTERVAL, min_trades=3)
    thr2, _ = select_threshold(ohlc, prob_noise, costs, ExitRulesSim(), 1, INTERVAL, min_trades=3)
    ok_grid = thr1 in DEFAULT_THR_GRID and thr1 == thr2  # детерминизм + сетка
    # walk-forward c selected на синтетике: пороги по окнам из сетки, embargo соблюдается
    df = synthetic_bars(600, seed=61)
    feats = build_features(df, "ratio")
    X, y, times, _ = build_dataset(df, 1, INTERVAL, features=feats)
    ohlc_full = df[["start_ms", "open", "high", "low", "close"]]
    wf = walk_forward_validate(X, y, times, df[["start_ms", "close"]], INTERVAL, 1,
                               feature_set="ratio", n_folds=3, train_fraction=0.5,
                               seed=42, n_estimators=20, ohlc=ohlc_full, trading=costs,
                               threshold_mode="selected", min_trades_for_thr=3)
    thrs = [f.effective_threshold for f in wf.folds]
    ok_wf = (len(wf.folds) == 3 and all(t in DEFAULT_THR_GRID for t in thrs)
             and all(f.trades is not None and f.majority_trades is not None for f in wf.folds))
    prev_end = None
    ok_emb = True
    for f in wf.folds:
        res = int(label_resolution_times(np.array([f.train_end_ms]), INTERVAL, 1)[0])
        if res > f.test_start_ms:
            ok_emb = False
        if prev_end is not None and f.test_start_ms <= prev_end:
            ok_emb = False
        prev_end = f.test_end_ms
    record("A5 выбор порога без утечки: сетка 0.50-0.75, детерминирован, только внутренний срез; walk-forward selected + embargo",
           ok_grid and ok_wf and ok_emb,
           f"thr={thr1} (det={thr1 == thr2}), fold_thrs={thrs}, embargo={ok_emb}, info={info1.get('reason')}")


# ----------------------------------------------------------------------- A6
def check_a6_gate() -> None:
    cases = [
        (0.010, 0.000, -0.010, True),    # лучше и текущей, и baseline
        (0.000, 0.010, -0.010, False),   # хуже текущей
        (0.020, 0.010, 0.030, False),    # лучше текущей, но хуже baseline
        (0.010, 0.010, -0.010, False),   # равенство -> не заменяем
        (float("nan"), 0.0, -1.0, False),  # нет сделок -> не заменяем
    ]
    ok = all(decide_adoption(a, b, c) == exp for a, b, c, exp in cases)
    record("A6 adopt-gate: замена только если net/сделку строго лучше текущей модели И majority (равенство/NaN -> нет)",
           ok, f"cases={len(cases)}")


# ----------------------------------------------------------------------- A7
def check_a7_adopted_flag(tmp: Path) -> None:
    df = synthetic_bars(200, seed=67)
    X, y, _, _ = build_dataset(df, 1, INTERVAL, features=build_features(df, "ratio"))
    model = train_final(X, y, seed=42, n_estimators=10)
    root = tmp / "models7"
    # старая ПРИНЯТАЯ (created_at раньше)
    card_old = build_card(model_version="m-old", algorithm="xgboost", symbol="BTCUSDT",
                          category="spot", interval=INTERVAL, horizon_bars=1, horizon_minutes=5,
                          feature_set="ratio", features=feature_columns("ratio"),
                          confidence_threshold=0.55, adopted=True, library_versions=library_versions())
    card_old["created_at_utc"] = "2026-01-01T00:00:00Z"
    save_artifact(model, root / "m-old", card_old)
    # новая НЕ принятая (created_at позже) - не должна выбираться
    card_new = dict(card_old)
    card_new.update({"model_version": "m-new-rejected", "adopted": False,
                     "created_at_utc": "2026-06-01T00:00:00Z"})
    save_artifact(model, root / "m-new-rejected", card_new)
    latest = find_latest_model_dir(root)
    ok = latest is not None and latest.name == "m-old"
    record("A7 find_latest_model_dir игнорирует adopted=false (production = принятый артефакт)",
           ok, f"selected={latest.name if latest else None}")


# ----------------------------------------------------------------------- B1
def find_real_dataset() -> Path | None:
    for folder in ("processed", "manual_checks"):
        cands = sorted((PROJECT_ROOT / "data" / folder).glob("*.clean.csv"),
                       key=lambda p: p.stat().st_size, reverse=True)
        if cands:
            return cands[0]
    return None


def check_b1_real_walkforward() -> None:
    data_path = find_real_dataset()
    if data_path is None:
        record("B1 реальные данные: walk-forward H=3 v2 + симуляция сделок", False,
               "нет данных: сначала scripts/download_history.py")
        return
    df_raw = load_klines_csv(data_path)
    df_clean, _ = clean_klines(df_raw, INTERVAL)
    if len(df_clean) > 30_000:
        df_clean = df_clean.tail(30_000).reset_index(drop=True)
    problems = []
    feats = build_features(df_clean, "v2", horizon_bars=H)
    X, y, times, ds = build_dataset(df_clean, H, INTERVAL, features=feats)
    if ds.n_dropped_warmup != 20:
        problems.append(f"warmup={ds.n_dropped_warmup} (ожидали 20)")
    ohlc = df_clean[["start_ms", "open", "high", "low", "close"]]
    costs = TradingCosts(0.1, 0.02, 100.0)
    wf = walk_forward_validate(X, y, times, df_clean[["start_ms", "close"]], INTERVAL, H,
                               feature_set="v2", n_folds=3, train_fraction=0.5,
                               seed=42, n_estimators=50, ohlc=ohlc, trading=costs,
                               exit_rules=ExitRulesSim(), threshold_mode="selected",
                               min_trades_for_thr=3)
    prev_end = None
    for f in wf.folds:
        res = int(label_resolution_times(np.array([f.train_end_ms]), INTERVAL, H)[0])
        if res > f.test_start_ms:
            problems.append(f"окно {f.fold}: embargo нарушен")
        if prev_end is not None and f.test_start_ms <= prev_end:
            problems.append(f"окно {f.fold}: перекрытие")
        prev_end = f.test_end_ms
        if not (0.0 <= f.accuracy <= 1.0):
            problems.append(f"окно {f.fold}: acc вне [0,1]")
        if f.effective_threshold not in DEFAULT_THR_GRID:
            problems.append(f"окно {f.fold}: порог вне сетки")
        for key in ("trades", "majority_trades", "momentum_trades"):
            t = getattr(f, key)
            if t is None:
                problems.append(f"окно {f.fold}: нет {key}")
    agg = wf.aggregate
    if not agg.get("trades") or not agg.get("majority_trades"):
        problems.append("агрегаты сделок отсутствуют")
    else:
        if agg["majority_trades"]["n_trades"] <= 0:
            problems.append("majority-baseline не совершил сделок (ожидался вход на каждом баре)")
        for v in (agg["trades"]["avg_net_pct"], agg["majority_trades"]["avg_net_pct"]):
            if v != v:  # NaN
                problems.append("NaN в агрегатах net")
    record("B1 реальные данные: walk-forward H=3 (15 мин) v2 + торговая симуляция, embargo, пороги из сетки",
           not problems,
           "; ".join(problems) if problems else
           f"файл={data_path.name} ({len(df_clean)} баров), окна acc={[round(f.accuracy, 4) for f in wf.folds]}, "
           f"пороги={[f.effective_threshold for f in wf.folds]}, "
           f"сделки v2={agg['trades']['n_trades']} net={agg['trades']['avg_net_pct']:+.4f}%/сделку, "
           f"majority={agg['majority_trades']['n_trades']} net={agg['majority_trades']['avg_net_pct']:+.4f}%")


# ----------------------------------------------------------------------- B2
def check_b2_final_pipeline(tmp: Path) -> None:
    df = synthetic_bars(800, seed=71)
    feats = build_features(df, "v2", horizon_bars=H)
    X, y, times, ds = build_dataset(df, H, INTERVAL, features=feats)
    ohlc_indexed = df[["start_ms", "open", "high", "low", "close"]].set_index("start_ms").sort_index()
    costs = TradingCosts(0.1, 0.02, 100.0)
    # импорт функции финального порога из скрипта (offline)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train_model import select_final_threshold  # noqa: E402
    thr, info = select_final_threshold(X, y, times, ohlc_indexed, INTERVAL, H, costs,
                                       ExitRulesSim(), 42, 40, 6, 0.3, min_trades=3)
    ok_thr = thr in DEFAULT_THR_GRID
    model = train_final(X, y, seed=42, n_estimators=40)
    version = make_model_version("BTCUSDT", "spot", INTERVAL, H, "v2")
    card = build_card(model_version=version, algorithm="xgboost", symbol="BTCUSDT",
                      category="spot", interval=INTERVAL, horizon_bars=H, horizon_minutes=H * 5,
                      target="вероятность направления на H баров", feature_set="v2",
                      features=feature_columns("v2", H), confidence_threshold=thr,
                      adopted=True, library_versions=library_versions())
    mdir = tmp / "models_b2" / version
    save_artifact(model, mdir, card)
    pred = Predictor.load(mdir)  # threshold None -> из card
    ok_card_thr = abs(pred.confidence_threshold - thr) < 1e-12
    pred.ensure_compatible(symbol="BTCUSDT", category="spot", interval=INTERVAL, horizon_bars=H)
    ok_h15 = pred.horizon_minutes == 15
    try:
        pred.ensure_compatible(symbol="BTCUSDT", category="spot", interval=INTERVAL, horizon_bars=1)
        ok_guard = False
    except ModelMismatchError:
        ok_guard = True
    bars = [{"start_ms": int(r.start_ms), "open": r.open, "high": r.high, "low": r.low,
             "close": r.close, "volume": r.volume, "turnover": r.turnover}
            for r in df.itertuples()]
    p = pred.predict_history(bars[-30:])
    ok_pred = (p.action in ("BUY", "SELL", "HOLD") and 0.5 <= p.confidence <= 1.0
               and p.horizon_minutes == 15 and p.feature_set == "v2"
               and p.model_version == version)
    ok_bar_reject = False
    try:
        pred.predict_bar(bars[-1])
    except ModelMismatchError:
        ok_bar_reject = True
    record("B2 финальный пайплайн: порог без утечки -> train -> card(thr, adopted, target) -> predict_history (15 мин); "
           "predict_bar для v2 отклоняется; guard горизонта",
           ok_thr and ok_card_thr and ok_h15 and ok_guard and ok_pred and ok_bar_reject,
           f"thr={thr}, card_thr={ok_card_thr}, horizon={p.horizon_minutes}мин, "
           f"action={p.action} conf={p.confidence:.3f}, guards={ok_guard and ok_bar_reject}")


def main() -> int:
    print("== Этап 5a: модель v2 (10-15 мин), утечки, торговый симулятор, adopt-gate (офлайн) ==")
    cfg = load_config()
    print(f"Конфиг: horizon_bars={cfg.model.horizon_bars} ({cfg.horizon_minutes} мин), "
          f"feature_set={cfg.model.feature_set}, trading: fee={cfg.trading.fee_pct}% "
          f"slip={cfg.trading.slippage_pct}% size={cfg.trading.order_size_quote} USDT")
    check_a1_no_future_dependence()
    check_a2_horizon_and_warmup()
    check_a3_history_features()
    check_a4_simulator_math()
    check_a5_threshold_selection()
    check_a6_gate()
    with tempfile.TemporaryDirectory() as td:
        check_a7_adopted_flag(Path(td))
    check_b1_real_walkforward()
    with tempfile.TemporaryDirectory() as td:
        check_b2_final_pipeline(Path(td))

    failed = [name for name, ok, _ in _results if not ok]
    print()
    if failed:
        print(f"RESULT: FAIL ({len(failed)} проваленных: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (проверок пройдено: {len(_results)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
