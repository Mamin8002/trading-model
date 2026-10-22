#!/usr/bin/env python3
"""Этап 5a: модель v2 (горизонт 10-15 мин) + честная торговая оценка + adopt-gate.

Протокол (предрегистрирован в docs/stage5_improvement_plan.md):
  * цель: вероятность НАПРАВЛЕНИЯ движения цены через H баров (H = model.horizon_bars,
    default 3 x 5 мин = 15 мин);
  * варианты: v1_fixed  = признаки ratio, фиксированный порог 0.55 (протокол текущей
              production-модели);
              v1_sel    = признаки ratio, порог выбирается БЕЗ утечки (внутренний срез);
              v2_sel    = признаки v2 (лаги/форма/объём/время), порог без утечки;
  * торговая симуляция с комиссиями и слиппеджем (trading.* конфига) и правилами
    выхода (positions.exit_rules); SL приоритетнее TP в одном баре (консервативно);
  * adopt-gate: v2 заменяет v1 только если net-return-per-trade(v2) >
    net-return-per-trade(v1_fixed) И > majority-baseline. Иначе production остаётся
    на признаках ratio (переобученных на новый горизонт), v2 сохраняется с
    adopted=false;
  * отчёт: docs/validation_report_v2.md (отчёт v1 не перезаписывается).

Использование (из корня проекта):
    python scripts\train_model.py             # полный прогон (2-4 мин)
    python scripts\train_model.py --quick     # смоук (3 окна, 40k баров, 60 деревьев)
    python scripts\train_model.py --feature-set v2   # явный выбор набора

Только офлайн-операции: сеть не используется, данные - из data/processed/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import build_dataset, clean_klines  # noqa: E402
from btc_bot.data.store import load_klines_csv  # noqa: E402
from btc_bot.model.features import build_features, feature_columns  # noqa: E402
from btc_bot.model.trading_sim import (  # noqa: E402
    ExitRulesSim,
    TradingCosts,
    decide_adoption,
    select_threshold,
)
from btc_bot.model.train import (  # noqa: E402
    build_card,
    library_versions,
    make_model,
    make_model_version,
    save_artifact,
    train_final,
)
from btc_bot.model.walk_forward import (  # noqa: E402
    WalkForwardResult,
    evaluate_original_protocol,
    label_resolution_times,
    walk_forward_validate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_model")

V1_FIXED_THRESHOLD = 0.55  # протокольная константа текущей production-модели


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Модель v2 + walk-forward с торговой симуляцией (офлайн).")
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None,
                   help="путь к clean CSV (по умолчанию - самый свежий в data/processed/)")
    p.add_argument("--feature-set", default=None, choices=["auto", "ratio", "v2"],
                   help="по умолчанию - model.feature_set из конфига (auto = сравнение + gate)")
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="смоук: 3 окна, 60 деревьев, последние 40k баров")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--report", default=None, help="по умолчанию docs/validation_report_v2.md")
    p.add_argument("--no-final", action="store_true", help="только валидация, без финальных артефактов")
    return p.parse_args()


def find_latest_clean_csv(processed_dir: Path) -> Optional[Path]:
    """Самый свежий ПОЛНЫЙ clean-файл по дате загрузки (частичные _PARTIAL пропускаются)."""
    from btc_bot.data.finders import find_latest_clean_csv as _find
    return _find(processed_dir)


def fmt_ms(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def progress_printer(name: str):
    def cb(k: int, n: int, info: dict) -> None:
        net = info.get("net_pct")
        net_txt = f"{net:+.4f}%" if isinstance(net, float) else "-"
        log.info("[%s] окно %d/%d: n_test=%d acc=%.4f thr=%.2f trades=%s net/сделку=%s (majority acc=%.4f)",
                 name, k + 1, n, info["test_n"], info["accuracy"], info["threshold"],
                 info.get("trades"), net_txt, info["majority"])
    return cb


def select_final_threshold(X, y, times, ohlc_indexed, interval, horizon_bars, costs, rules,
                           seed, n_estimators, max_depth, learning_rate,
                           val_fraction=0.15, min_trades=5):
    """Порог для финальной модели: внутренний срез ВСЕХ данных (последние val_fraction),
    fit на части до среза с embargo, выбор по net-per-trade. Без утечки в тест."""
    n = len(X)
    times_np = times.to_numpy(dtype="int64")
    y_np = y.to_numpy(dtype="int64")
    cut = int(n * (1.0 - val_fraction))
    val_start = int(times_np[cut])
    resolution = label_resolution_times(times_np, interval, horizon_bars)
    tr_idx = np.where((np.arange(n) < cut) & (resolution <= val_start))[0]
    val_idx = np.arange(cut, n)
    from btc_bot.model.train import encode_labels
    sel_model = make_model(seed, max(20, n_estimators // 2), max_depth, learning_rate)
    sel_model.fit(X.iloc[tr_idx], encode_labels(y_np[tr_idx]))
    proba = sel_model.predict_proba(X.iloc[val_idx])
    prob_up = proba[:, list(sel_model.classes_).index(1)]
    ohlc_val = ohlc_indexed.loc[times_np[val_idx]].reset_index()
    thr, info = select_threshold(ohlc_val, prob_up, costs, rules, horizon_bars, interval,
                                 min_trades=min_trades)
    info["n_val_rows"] = int(len(val_idx))
    return float(thr), info


def trades_line(agg_trades: Optional[dict]) -> str:
    if not agg_trades:
        return "нет (симуляция не запускалась)"
    a = agg_trades
    net = a["avg_net_pct"]
    net_txt = f"{net:+.4f}%" if net == net else "н/д"
    return (f"trades={a['n_trades']}, net/сделку={net_txt}, total={a['total_net_usd']:+.2f} USDT, "
            f"win={a['win_rate']:.1%}, maxDD={a['max_dd_fraction']:.4f}, gross/сделку={a['avg_gross_pct']:+.4f}%")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    ex, mc = cfg.exchange, cfg.model

    feature_set_choice = args.feature_set or mc.feature_set
    if feature_set_choice == "raw":
        feature_set_choice = "ratio"  # raw оставлен только в отчёте v1; здесь не сравнивается
    n_estimators = 60 if args.quick else (args.n_estimators or mc.n_estimators)
    n_folds = 3 if args.quick else args.folds
    resolved_thr = mc.confidence_threshold if mc.confidence_threshold is not None else V1_FIXED_THRESHOLD
    H = mc.horizon_bars

    costs = TradingCosts(fee_pct=cfg.trading.fee_pct, slippage_pct=cfg.trading.slippage_pct,
                         order_size_quote=cfg.trading.order_size_quote)
    er = cfg.positions.exit_rules
    rules = ExitRulesSim(take_profit_pct=er.take_profit_pct, stop_loss_pct=er.stop_loss_pct,
                         max_hold_bars=er.max_hold_bars)

    # -- данные ------------------------------------------------------------
    data_path = Path(args.data) if args.data else find_latest_clean_csv(cfg.resolve(cfg.data.processed_dir))
    if data_path is None or not data_path.exists():
        proc_dir = cfg.resolve(cfg.data.processed_dir)
        if list(proc_dir.glob("*_PARTIAL*.clean.csv")):
            log.error("Найдены только ЧАСТИЧНЫЕ файлы (загрузка прервана). Продолжите загрузку: "
                      "python scripts/download_history.py --resume <файл data/raw/*_PARTIAL_*.csv>")
        else:
            log.error("Clean-данные не найдены. Сначала выполните: python scripts/download_history.py")
        return 2
    log.info("Данные: %s", data_path)

    df_raw = load_klines_csv(data_path)
    df_clean, prep_report = clean_klines(df_raw, ex.interval)
    if args.quick and len(df_clean) > 40_000:
        df_clean = df_clean.tail(40_000).reset_index(drop=True)
        log.info("--quick: использую последние 40000 баров")
    log.info("Баров после очистки: %d (дубли: %d, пропуски: %d, незакрытые: %d)",
             prep_report.n_output, prep_report.n_duplicates_removed,
             prep_report.total_missing_bars, prep_report.n_unconfirmed_excluded)
    prices = df_clean[["start_ms", "close"]]
    ohlc = df_clean[["start_ms", "open", "high", "low", "close"]]
    ohlc_indexed = ohlc.set_index("start_ms").sort_index()

    def make_dataset(feature_set: str):
        feats = build_features(df_clean, feature_set, horizon_bars=H)
        return build_dataset(df_clean, horizon_bars=H, interval=ex.interval, features=feats)

    common = dict(
        interval=ex.interval, horizon_bars=H, n_folds=n_folds,
        train_fraction=args.train_fraction, seed=mc.random_seed,
        n_estimators=n_estimators, max_depth=mc.max_depth, learning_rate=mc.learning_rate,
        ohlc=ohlc, trading=costs, exit_rules=rules,
    )

    # -- варианты ------------------------------------------------------------
    variants: dict[str, WalkForwardResult] = {}
    datasets: dict[str, tuple] = {}
    plan: list[tuple[str, str, str]] = []  # (имя, feature_set, threshold_mode)
    if feature_set_choice == "auto":
        plan = [("v1_fixed", "ratio", "fixed"), ("v1_sel", "ratio", "selected"),
                ("v2_sel", "v2", "selected")]
    else:
        plan = [(f"{feature_set_choice}_sel", feature_set_choice, "selected")]

    for name, fs, tmode in plan:
        if fs not in datasets:
            datasets[fs] = make_dataset(fs)
        X, y, times, ds_rep = datasets[fs]
        log.info("Вариант %s (%s, threshold_mode=%s): строк=%d (warm-up исключено %d, без будущего бара %d)",
                 name, fs, tmode, ds_rep.n_rows, ds_rep.n_dropped_warmup, ds_rep.n_dropped_no_future_bar)
        wf = walk_forward_validate(
            X, y, times, prices, feature_set=fs,
            threshold=resolved_thr if tmode == "fixed" else V1_FIXED_THRESHOLD,
            threshold_mode=tmode, progress_cb=progress_printer(name), **common)
        variants[name] = wf
        a = wf.aggregate
        log.info("[%s] ИТОГ: acc=%.4f | %s", name, a["accuracy"], trades_line(a.get("trades")))

    # baseline-сделки берутся из любого варианта (они одинаковы: majority/momentum
    # не зависят от признаков); используем первый
    base_wf = next(iter(variants.values()))
    maj_tr = base_wf.aggregate.get("majority_trades")
    mom_tr = base_wf.aggregate.get("momentum_trades")
    log.info("[majority] %s", trades_line(maj_tr))
    log.info("[momentum] %s", trades_line(mom_tr))

    # -- adopt-gate (предрегистрирован) ---------------------------------------
    def primary(wf: WalkForwardResult) -> float:
        t = wf.aggregate.get("trades")
        return float(t["avg_net_pct"]) if t and t["n_trades"] > 0 else float("nan")

    gate = None
    if feature_set_choice == "auto":
        p_v2 = primary(variants["v2_sel"])
        p_v1 = primary(variants["v1_fixed"])
        p_maj = float(maj_tr["avg_net_pct"]) if maj_tr and maj_tr["n_trades"] else float("nan")
        adopted = decide_adoption(p_v2, p_v1, p_maj)
        gate = {
            "criterion": "net return per trade на 8 отложенных walk-forward окнах (после комиссий и слиппеджа)",
            "rule": "v2 принимается только если primary(v2) > primary(v1_fixed) И > primary(majority)",
            "primary_v2": p_v2, "primary_v1_fixed": p_v1, "primary_majority": p_maj,
            "secondary_acc_v2": variants["v2_sel"].aggregate["accuracy"],
            "secondary_acc_v1_fixed": variants["v1_fixed"].aggregate["accuracy"],
            "v2_adopted": bool(adopted),
        }
        winner_fs = "v2" if adopted else "ratio"
        winner_name = "v2_sel" if adopted else "v1_sel"
        loser_fs, loser_name = ("ratio", "v1_sel") if adopted else ("v2", "v2_sel")
        log.info("ADOPT-GATE: v2 net=%s vs v1_fixed net=%s vs majority net=%s -> v2 %s",
                 p_v2, p_v1, p_maj, "ПРИНЯТ" if adopted else "НЕ ПРИНЯТ (production остаётся на ratio)")
    else:
        winner_fs = feature_set_choice
        winner_name = f"{feature_set_choice}_sel"
        loser_fs = loser_name = None
        gate = {"criterion": "feature_set задан явно - gate не применяется", "v2_adopted": None}

    wf_winner = variants[winner_name]

    # -- протокол оригинала (справочно) ----------------------------------------
    X_proto, y_proto, _, _ = make_dataset("ratio")
    original_protocol = evaluate_original_protocol(
        X_proto, y_proto, seed=mc.random_seed, n_estimators=n_estimators,
        max_depth=mc.max_depth, learning_rate=mc.learning_rate)
    log.info("Протокол оригинала (single split 75/25, ratio): acc=%.4f", original_protocol["accuracy"])

    # -- финальные артефакты ----------------------------------------------------
    out: dict[str, Path] = {}
    model_version = None
    models_root = Path(args.out_dir) if args.out_dir else cfg.resolve(cfg.data.models_dir)
    if not args.no_final:
        X_final, y_final, times_final, ds_final = datasets[winner_fs]
        final_thr, thr_info = select_final_threshold(
            X_final, y_final, times_final, ohlc_indexed, ex.interval, H, costs, rules,
            mc.random_seed, n_estimators, mc.max_depth, mc.learning_rate)
        log.info("Финальный порог (выбран без утечки на внутреннем срезе): %.2f (%s)",
                 final_thr, thr_info.get("reason"))
        model = train_final(X_final, y_final, seed=mc.random_seed, n_estimators=n_estimators,
                            max_depth=mc.max_depth, learning_rate=mc.learning_rate)
        model_version = make_model_version(ex.symbol, ex.category, ex.interval, H, winner_fs)

        def make_card(fs: str, adopted: bool, wf: WalkForwardResult, thr: float, thr_info: dict):
            return build_card(
                model_version=None,  # проставим ниже
                algorithm="xgboost",
                params={"n_estimators": n_estimators, "max_depth": mc.max_depth,
                        "learning_rate": mc.learning_rate, "random_seed": mc.random_seed},
                symbol=ex.symbol, category=ex.category, interval=ex.interval,
                horizon_bars=H, horizon_minutes=H * ex.interval_minutes,
                target="вероятность направления: y=+1 если close(t+H*interval) > close(t), иначе -1",
                feature_set=fs, features=feature_columns(fs, H),
                confidence_threshold=thr,
                threshold_selection={"method": "внутренний срез 15% обучающих данных, embargo, критерий net-per-trade",
                                     **thr_info},
                trading_costs={"fee_pct": costs.fee_pct, "slippage_pct": costs.slippage_pct,
                               "order_size_quote": costs.order_size_quote},
                exit_rules_snapshot={"take_profit_pct": rules.take_profit_pct,
                                     "stop_loss_pct": rules.stop_loss_pct,
                                     "max_hold_bars": rules.max_hold_bars},
                adopted=adopted,
                adopt_gate=gate,
                training_data={
                    "file": str(data_path), "n_bars": int(ds_final.n_rows),
                    "period_start_utc": fmt_ms(int(times_final.iloc[0])),
                    "period_end_utc": fmt_ms(int(times_final.iloc[-1])),
                    "prep_report": prep_report.to_dict(),
                    "dataset_report": ds_final.to_dict(),
                },
                walk_forward_summary=wf.aggregate,
                walk_forward_params={"n_folds": n_folds, "train_fraction": args.train_fraction,
                                     "threshold_mode": wf.threshold_mode, "quick_mode": bool(args.quick),
                                     "embargo_rule": wf.to_dict()["embargo_rule"]},
                variant_comparison={name: w.aggregate for name, w in variants.items()},
                original_protocol_single_split=original_protocol,
                library_versions=library_versions(),
                limitations=[
                    "Accuracy/net-метрики НЕ означают прибыльность: при издержках ~0.2% round-trip "
                    "ожидаемый net-результат на 5-15-мин горизонте вероятнее всего отрицательный.",
                    "Симуляция выходов - по закрытым барам (тиковый уровень не моделируется; очередь и "
                    "частичные исполнения не учитываются).",
                    "Выбор порога/набора по тем же окнам даёт небольшой оптимизм (selection bias).",
                    "Горизонт прогноза: H баров; сигнал действителен до close(t+H*interval).",
                    "Модель деградирует при смене рыночного режима; требуется периодическое переобучение.",
                ],
            )

        # победитель (принят в production)
        card_w = make_card(winner_fs, True, wf_winner, final_thr, thr_info)
        card_w["model_version"] = model_version
        w_dir = models_root / model_version
        out = save_artifact(model, w_dir, card_w, walk_forward=wf_winner.to_dict())
        log.info("ПРИНЯТЫЙ артефакт: %s (adopted=true)", w_dir)

        # проигравший (для прозрачности; adopted=false - бот его не подхватит)
        if loser_fs is not None and not args.quick:
            X_l, y_l, times_l, ds_l = datasets[loser_fs]
            thr_l, thr_info_l = select_final_threshold(
                X_l, y_l, times_l, ohlc_indexed, ex.interval, H, costs, rules,
                mc.random_seed, n_estimators, mc.max_depth, mc.learning_rate)
            model_l = train_final(X_l, y_l, seed=mc.random_seed, n_estimators=n_estimators,
                                  max_depth=mc.max_depth, learning_rate=mc.learning_rate)
            version_l = make_model_version(ex.symbol, ex.category, ex.interval, H, loser_fs)
            card_l = make_card(loser_fs, False, variants[loser_name], thr_l, thr_info_l)
            card_l["model_version"] = version_l
            save_artifact(model_l, models_root / version_l, card_l,
                          walk_forward=variants[loser_name].to_dict())
            log.info("НЕ принятый артефакт (adopted=false, для прозрачности): %s", version_l)

    # -- отчёт -------------------------------------------------------------------
    report_path = Path(args.report) if args.report else PROJECT_ROOT / "docs" / "validation_report_v2.md"
    write_report(report_path, args, cfg, H, data_path, prep_report, datasets, variants,
                 maj_tr, mom_tr, gate, winner_fs, model_version, models_root,
                 original_protocol, costs, rules)
    log.info("Отчёт сохранён: %s", report_path)

    print()
    print("=== ИТОГ ЭТАПА 5a ===")
    for name, wf in variants.items():
        a = wf.aggregate
        print(f"[{name:9s}] acc={a['accuracy']:.4f} | {trades_line(a.get('trades'))}")
    print(f"[majority ] {trades_line(maj_tr)}")
    print(f"[momentum ] {trades_line(mom_tr)}")
    if gate and gate.get("v2_adopted") is not None:
        print(f"ADOPT-GATE: v2 {'ПРИНЯТ' if gate['v2_adopted'] else 'НЕ ПРИНЯТ'} "
              f"(primary: v2={gate['primary_v2']}, v1_fixed={gate['primary_v1_fixed']}, "
              f"majority={gate['primary_majority']})")
    if model_version:
        print(f"Принятый артефакт: {out['model']}")
        print(f"Model version: {model_version}")
    print(f"Отчёт: {report_path}")
    print("RESULT: OK")
    return 0


def write_report(report_path, args, cfg, H, data_path, prep_report, datasets, variants,
                 maj_tr, mom_tr, gate, winner_fs, model_version, models_root,
                 original_protocol, costs, rules) -> None:
    ex, mc = cfg.exchange, cfg.model
    now = datetime.now(timezone.utc)
    md = []
    md.append("# Отчёт валидации модели v2 (Этап 5a)")
    md.append("")
    md.append(f"*Сгенерировано: {now.isoformat(timespec='seconds')}Z скриптом `scripts/train_model.py`"
              + (" (режим --quick: укороченный прогон, НЕ финальный артефакт)" if args.quick else "") + "*")
    md.append("")
    md.append("> Сигналы модели — статистический вывод, не гарантия дохода и не персональная")
    md.append("> финансовая рекомендация. **Net-метрики учитывают комиссии и слиппедж;**")
    md.append("> отрицательный net-результат при данных издержках — ожидаем и честно показан.")
    md.append("")
    md.append("## 1. Что прогнозирует модель")
    md.append("")
    md.append(f"- **Вероятность направления** движения цены на горизонте **H={H} бар(ов) = "
              f"{H * ex.interval_minutes} минут**: y=+1 если close(t+H·Δ) > close(t), иначе −1.")
    md.append("- Это не ожидаемая доходность и не вероятность касания уровня (triple-barrier) —")
    md.append("  выбор зафиксирован в плане Этапа 5 (`docs/stage5_improvement_plan.md`).")
    md.append(f"- Инструмент: {ex.category}/{ex.symbol}, interval={ex.interval} мин; данные: `{data_path.name}`.")
    md.append(f"- Издержки симуляции: fee {costs.fee_pct}%/сторона, slippage {costs.slippage_pct}%/сторона, "
              f"объём сделки {costs.order_size_quote} USDT.")
    md.append(f"- Правила выхода симуляции: TP={rules.take_profit_pct}, SL={rules.stop_loss_pct}, "
              f"max_hold={rules.max_hold_bars} (null = выход по горизонту H). SL приоритетнее TP в одном баре.")
    md.append("")
    md.append("## 2. Протокол сравнения (без утечки будущих данных)")
    md.append("")
    md.append(f"- Walk-forward: expanding window, {args.folds if not args.quick else 3} окон, "
              f"train_fraction={args.train_fraction}, embargo t+(H+1)·Δ ≤ test_start.")
    md.append("- Порог входа в вариантах `_sel` выбирается на внутреннем срезе обучающего окна")
    md.append("  (последние 15%, отделены embargo), критерий — net return на сделку; на тесте заморожен.")
    md.append("- Торговый симулятор одинаков для модели и baseline'ов: вход по close сигнального бара")
    md.append("  ± слиппедж, выход по H или SL/TP/max_hold по закрытым барам; сделки, выходящие за")
    md.append("  окно, не открываются; одна позиция в момент времени.")
    md.append("- Variants: **v1_fixed** (ratio, порог 0.55 — протокол текущей production-модели),")
    md.append("  **v1_sel** (ratio, порог без утечки), **v2_sel** (лаг-признаки v2, порог без утечки).")
    md.append("")
    md.append("## 3. Результаты по окнам")
    md.append("")
    for name, wf in variants.items():
        X_ds = datasets[wf.feature_set][3]
        md.append(f"### Вариант `{name}` (признаки {wf.feature_set}; строк {X_ds.n_rows}, "
                  f"warm-up исключено {X_ds.n_dropped_warmup})")
        md.append("")
        md.append("| Окно | Тест (UTC) | n_test | Acc | Порог | Сделок | Net/сделку | Win% | MaxDD |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for f in wf.folds:
            tr = f.trades or {}
            net = tr.get("avg_net_pct")
            md.append(
                f"| {f.fold + 1} | {fmt_ms(f.test_start_ms)} → {fmt_ms(f.test_end_ms)} | {f.test_n} "
                f"| {f.accuracy:.4f} | {f.effective_threshold:.2f} | {tr.get('n_trades', '-')} "
                f"| {('%+.4f%%' % net) if isinstance(net, float) else '-'} "
                f"| {('%.1f%%' % (tr.get('win_rate', 0) * 100)) if tr else '-'} "
                f"| {('%.4f' % tr.get('max_dd_fraction', 0)) if tr else '-'} |")
        a = wf.aggregate
        t = a.get("trades") or {}
        net = t.get("avg_net_pct")
        md.append(
            f"| **Σ** | все окна | {a['n_test_total']} | **{a['accuracy']:.4f}** | - "
            f"| {t.get('n_trades', '-')} | {('%+.4f%%' % net) if isinstance(net, float) and net == net else '-'} "
            f"| {('%.1f%%' % (t.get('win_rate', 0) * 100)) if t else '-'} "
            f"| {('%.4f' % t.get('max_dd_fraction', 0)) if t else '-'} |")
        md.append("")
    md.append("## 4. Агрегированное сравнение")
    md.append("")
    md.append("| Вариант | Acc | Покрытие | Сделок | Net/сделку | Total net, USDT | Gross/сделку | Win% |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name, wf in variants.items():
        a = wf.aggregate
        t = a.get("trades") or {}
        md.append(f"| {name} | {a['accuracy']:.4f} | {a['coverage_at_threshold']:.1%} "
                  f"| {t.get('n_trades', '-')} | {_pct(t.get('avg_net_pct'))} "
                  f"| {('%+.2f' % t['total_net_usd']) if t else '-'} | {_pct(t.get('avg_gross_pct'))} "
                  f"| {('%.1f%%' % (t['win_rate'] * 100)) if t else '-'} |")
    for label, t in (("majority (baseline)", maj_tr), ("momentum (baseline)", mom_tr)):
        if t:
            md.append(f"| {label} | - | 100% | {t['n_trades']} | {_pct(t.get('avg_net_pct'))} "
                      f"| {t['total_net_usd']:+.2f} | {_pct(t.get('avg_gross_pct'))} "
                      f"| {t['win_rate']:.1%} |")
    md.append("")
    md.append("## 5. Adopt-gate (предрегистрированное правило замены модели)")
    md.append("")
    if gate:
        md.append(f"- Критерий: {gate.get('criterion')}")
        md.append(f"- Правило: {gate.get('rule')}")
        if gate.get("v2_adopted") is not None:
            md.append(f"- Значения: v2={gate['primary_v2']}, v1_fixed={gate['primary_v1_fixed']}, "
                      f"majority={gate['primary_majority']} (net % на сделку)")
            md.append(f"- **Решение: v2 {'ПРИНЯТ' if gate['v2_adopted'] else 'НЕ ПРИНЯТ'}**.")
        else:
            md.append(f"- {gate.get('criterion')}")
    if model_version:
        md.append("")
        md.append(f"Production-артефакт: `data/models/{model_version}/` (feature_set **{winner_fs}**, adopted=true).")
        md.append("Не принятый вариант (при наличии) сохранён рядом с adopted=false — бот его не подхватывает.")
    md.append("")
    md.append("## 6. Протокол исходного ноутбука (справочно)")
    md.append("")
    md.append(f"Single split 75/25, ratio-признаки, без embargo: acc={original_protocol['accuracy']:.4f}, "
              f"f1(+1)={original_protocol['f1_up']:.4f} (n_test={original_protocol['n_test']}).")
    md.append("")
    md.append("## 7. Ограничения (честно)")
    md.append("")
    md.append("- **Прибыльность не гарантируется и скорее всего отсутствует**: при издержках ~"
              f"{2 * (costs.fee_pct + costs.slippage_pct):.2f}% round-trip net-результат всех вариантов, "
              "вероятно, отрицательный — таблица выше показывает фактические числа.")
    md.append("- Выходы симулируются по закрытым барам (консервативно); тиковая точность, очередь и")
    md.append("  частичные исполнения не моделируются.")
    md.append("- Выбор порога и набора по тем же walk-forward окнам даёт небольшой оптимизм (selection bias).")
    md.append("- Majority/momentum baseline'ы входят в каждой строке (максимальная частота сделок) —")
    md.append("  их net почти полностью состоит из издержек; это нижняя граница, а не конкурентная стратегия.")
    md.append("- Результат зависит от рыночного режима; рекомендуется переобучение раз в 1-3 месяца и")
    md.append("  мониторинг live-качества. Автопереобучение на результатах позиций запрещено.")
    md.append("- Версии библиотек: " + json.dumps(library_versions(), ensure_ascii=False))
    md.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")


def _pct(v) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{f:+.4f}%" if f == f else "-"


if __name__ == "__main__":
    sys.exit(main())
