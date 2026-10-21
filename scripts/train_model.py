#!/usr/bin/env python3
"""Этап 2b: переобучение модели на свежих данных Bybit + walk-forward-валидация.

Использование (из корня проекта btc_bybit_bot):
    python scripts/train_model.py                 # полная валидация + финальная модель
    python scripts/train_model.py --quick         # быстрый прогон (3 окна, 60 деревьев)
    python scripts/train_model.py --feature-set ratio --folds 8

Только офлайн-операции: сеть не используется, данные - из data/processed/.

Создаваемые файлы:
    data/models/<model_version>/model.joblib       - артефакт модели
    data/models/<model_version>/model_card.json    - метаданные (версия, данные, метрики)
    data/models/<model_version>/walk_forward.json  - полные результаты валидации
    docs/validation_report.md                      - отчёт (переписывается при каждом запуске)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import build_dataset, clean_klines  # noqa: E402
from btc_bot.data.store import load_klines_csv  # noqa: E402
from btc_bot.model.features import build_features, feature_columns  # noqa: E402
from btc_bot.model.train import (  # noqa: E402
    build_card,
    library_versions,
    make_model_version,
    save_artifact,
    train_final,
)
from btc_bot.model.walk_forward import (  # noqa: E402
    WalkForwardResult,
    evaluate_original_protocol,
    walk_forward_validate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_model")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Переобучение модели + walk-forward-валидация (офлайн).")
    p.add_argument("--config", default=None)
    p.add_argument("--data", default=None,
                   help="путь к clean CSV (по умолчанию - самый свежий в data/processed/)")
    p.add_argument("--feature-set", default=None, choices=["auto", "raw", "ratio"],
                   help="по умолчанию - model.feature_set из конфига (auto = сравнить оба)")
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--n-estimators", type=int, default=None, help="по умолчанию из конфига")
    p.add_argument("--quick", action="store_true", help="быстрый прогон: 3 окна, 60 деревьев, последние 40k баров")
    p.add_argument("--out-dir", default=None, help="каталог для артефакта (по умолчанию data/models/)")
    p.add_argument("--report", default=None, help="путь отчёта (по умолчанию docs/validation_report.md)")
    p.add_argument("--no-final", action="store_true", help="только валидация, без финального обучения")
    return p.parse_args()


def find_latest_clean_csv(processed_dir: Path) -> Path | None:
    files = sorted(processed_dir.glob("*.clean.csv"))
    return files[-1] if files else None


def fmt_ms(ms: int) -> str:
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def progress_printer(feature_set: str):
    def cb(k: int, n: int, info: dict) -> None:
        log.info("[%s] окно %d/%d: n_test=%d acc=%.4f (majority=%.4f, momentum=%.4f)",
                 feature_set, k + 1, n, info["test_n"], info["accuracy"],
                 info["majority"], info["momentum"] if info["momentum"] is not None else float("nan"))
    return cb


def folds_table_md(wf: WalkForwardResult, threshold: float) -> str:
    lines = [
        f"| Окно | Тест (UTC) | n_test | Accuracy | F1(+1) | Prec(+1) | Rec(+1) | Покрытие@{threshold} | Acc@{threshold} | Majority | Momentum |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in wf.folds:
        acc_thr = f"{f.accuracy_at_threshold:.4f}" if f.accuracy_at_threshold is not None else "н/д"
        mom = f"{f.momentum_accuracy:.4f}" if f.momentum_accuracy is not None else "н/д"
        lines.append(
            f"| {f.fold + 1} | {fmt_ms(f.test_start_ms)} → {fmt_ms(f.test_end_ms)} | {f.test_n} "
            f"| {f.accuracy:.4f} | {f.f1_up:.4f} | {f.precision_up:.4f} | {f.recall_up:.4f} "
            f"| {f.coverage_at_threshold:.1%} | {acc_thr} | {f.majority_accuracy:.4f} | {mom} |"
        )
    a = wf.aggregate
    acc_thr = f"{a['accuracy_at_threshold']:.4f}" if not np.isnan(a["accuracy_at_threshold"]) else "н/д"
    mom = f"{a['momentum_accuracy']:.4f}" if not np.isnan(a["momentum_accuracy"]) else "н/д"
    lines.append(
        f"| **Σ** | все окна | {a['n_test_total']} | **{a['accuracy']:.4f}** | {a['f1_up']:.4f} "
        f"| {a['precision_up']:.4f} | {a['recall_up']:.4f} | {a['coverage_at_threshold']:.1%} "
        f"| {acc_thr} | {a['majority_accuracy']:.4f} | {mom} |"
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    ex, mc = cfg.exchange, cfg.model

    feature_set_choice = args.feature_set or mc.feature_set
    n_estimators = 60 if args.quick else (args.n_estimators or mc.n_estimators)
    n_folds = 3 if args.quick else args.folds
    threshold = mc.confidence_threshold

    # -- данные ------------------------------------------------------------
    data_path = Path(args.data) if args.data else find_latest_clean_csv(cfg.resolve(cfg.data.processed_dir))
    if data_path is None or not data_path.exists():
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

    def make_dataset(feature_set: str):
        """Датасет, выровненный по строкам df_clean (признаки + цель + времена)."""
        feats = build_features(df_clean, feature_set)
        return build_dataset(df_clean, horizon_bars=mc.horizon_bars,
                             interval=ex.interval, features=feats)

    variants_to_run = ["raw", "ratio"] if feature_set_choice == "auto" else [feature_set_choice]

    ds_raw_variant = make_dataset("raw")
    _, y, times, ds_report = ds_raw_variant
    log.info("Набор: строк=%d (исключено без будущего бара: %d), up=%d, down/flat=%d, период %s .. %s UTC",
             ds_report.n_rows, ds_report.n_dropped_no_future_bar, ds_report.n_up, ds_report.n_down_or_flat,
             fmt_ms(int(times.iloc[0])), fmt_ms(int(times.iloc[-1])))

    common = dict(
        interval=ex.interval, horizon_bars=mc.horizon_bars, n_folds=n_folds,
        train_fraction=args.train_fraction, threshold=threshold, seed=mc.random_seed,
        n_estimators=n_estimators, max_depth=mc.max_depth, learning_rate=mc.learning_rate,
    )

    # -- walk-forward: raw и/или ratio -------------------------------------
    wf_results: dict[str, WalkForwardResult] = {}
    for fs in variants_to_run:
        X = ds_raw_variant[0] if fs == "raw" else make_dataset(fs)[0]
        log.info("Walk-forward [%s]: %d окон, train_fraction=%.2f, n_estimators=%d ...",
                 fs, n_folds, args.train_fraction, n_estimators)
        wf = walk_forward_validate(X, y, times, prices, feature_set=fs,
                                   progress_cb=progress_printer(fs), **common)
        wf_results[fs] = wf
        a = wf.aggregate
        acc_thr = a["accuracy_at_threshold"]
        log.info("[%s] ИТОГ: acc=%.4f f1=%.4f coverage@thr=%.1f%% acc@thr=%s | majority=%.4f momentum=%.4f",
                 fs, a["accuracy"], a["f1_up"], a["coverage_at_threshold"] * 100,
                 f"{acc_thr:.4f}" if acc_thr == acc_thr else "н/д",
                 a["majority_accuracy"], a["momentum_accuracy"])

    # -- выбор feature_set ---------------------------------------------------
    if feature_set_choice == "auto":
        acc_raw = wf_results["raw"].aggregate["accuracy"]
        acc_ratio = wf_results["ratio"].aggregate["accuracy"]
        # при равенстве (|Δ| < 0.001) предпочитаем ratio - стационарные признаки устойчивее к дрейфу
        chosen = "ratio" if acc_ratio >= acc_raw - 0.001 else "raw"
        selection_note = (f"auto: raw acc={acc_raw:.4f}, ratio acc={acc_ratio:.4f} -> выбран {chosen} "
                          f"(при |Δ|<0.001 предпочитается ratio как более устойчивый к дрейфу уровней)")
    else:
        chosen = feature_set_choice
        selection_note = f"feature_set задан явно: {chosen}"
    wf_chosen = wf_results[chosen]
    log.info("Выбор feature_set: %s", selection_note)

    # -- протокол оригинального ноутбука (single split 75/25, raw, без embargo)
    X_for_protocol = ds_raw_variant[0]
    original_protocol = evaluate_original_protocol(
        X_for_protocol, y, seed=mc.random_seed, n_estimators=n_estimators,
        max_depth=mc.max_depth, learning_rate=mc.learning_rate,
    )
    log.info("Протокол оригинала (single split 75/25, raw): acc=%.4f f1=%.4f",
             original_protocol["accuracy"], original_protocol["f1_up"])

    out: dict[str, Path] = {}
    model_version = None
    if not args.no_final:
        # -- финальная модель на всех данных ---------------------------------
        X_final = ds_raw_variant[0] if chosen == "raw" else make_dataset(chosen)[0]
        model = train_final(X_final, y, seed=mc.random_seed, n_estimators=n_estimators,
                            max_depth=mc.max_depth, learning_rate=mc.learning_rate)
        model_version = make_model_version(ex.symbol, ex.category, ex.interval,
                                           mc.horizon_bars, chosen)
        models_root = Path(args.out_dir) if args.out_dir else cfg.resolve(cfg.data.models_dir)
        model_dir = models_root / model_version
        card = build_card(
            model_version=model_version,
            algorithm="xgboost",
            params={"n_estimators": n_estimators, "max_depth": mc.max_depth,
                    "learning_rate": mc.learning_rate, "random_seed": mc.random_seed},
            symbol=ex.symbol, category=ex.category, interval=ex.interval,
            horizon_bars=mc.horizon_bars,
            horizon_minutes=mc.horizon_bars * ex.interval_minutes,
            feature_set=chosen,
            features=feature_columns(chosen),
            confidence_threshold=threshold,
            training_data={
                "file": str(data_path),
                "n_bars": int(ds_report.n_rows),
                "period_start_ms": int(times.iloc[0]), "period_end_ms": int(times.iloc[-1]),
                "period_start_utc": fmt_ms(int(times.iloc[0])), "period_end_utc": fmt_ms(int(times.iloc[-1])),
                "prep_report": prep_report.to_dict(),
                "dataset_report": ds_report.to_dict(),
            },
            walk_forward_summary=wf_chosen.aggregate,
            walk_forward_params={
                "n_folds": n_folds,
                "train_fraction": args.train_fraction,
                "threshold": threshold,
                "embargo_rule": wf_chosen.to_dict()["embargo_rule"],
                "quick_mode": bool(args.quick),
                "n_estimators": n_estimators,
            },
            variant_comparison={fs: wf.aggregate for fs, wf in wf_results.items()},
            feature_set_selection=selection_note,
            original_protocol_single_split=original_protocol,
            library_versions=library_versions(),
            limitations=[
                "Accuracy выше baseline НЕ означает прибыльность: комиссии, спред и слиппедж не учитываются.",
                "Горизонт прогноза - 1..N баров (см. horizon_minutes); далее сигнал недействителен.",
                "Модель обучена на исторических данных и может деградировать при смене рыночного режима.",
                "Исходная модель из репозитория Sumit2514/Bitcoin-Price-Prediction невоспроизводима "
                "(данные отсутствуют), сравнение выполнено по протоколу, а не по артефакту.",
            ],
        )
        out = save_artifact(model, model_dir, card, walk_forward=wf_chosen.to_dict())
        log.info("Артефакт модели сохранён: %s", model_dir)

    # -- отчёт -----------------------------------------------------------------
    report_path = Path(args.report) if args.report else PROJECT_ROOT / "docs" / "validation_report.md"
    now = datetime.now(timezone.utc)
    md = []
    md.append("# Отчёт валидации модели (Этап 2b)")
    md.append("")
    md.append(f"*Сгенерировано: {now.isoformat(timespec='seconds')}Z скриптом `scripts/train_model.py`"
              + (" (режим --quick: укороченный прогон, НЕ финальный артефакт)" if args.quick else "") + "*")
    md.append("")
    md.append("> Сигналы модели — статистический вывод, не гарантия дохода и не персональная")
    md.append("> финансовая рекомендация. **Accuracy не равна прибыльности** (комиссии/спред/слиппедж не учтены).")
    md.append("")
    md.append("## 1. Данные и постановка")
    md.append("")
    md.append(f"- Источник: Bybit public REST `/v5/market/kline`, **{ex.category}/{ex.symbol}**, interval={ex.interval} мин.")
    md.append(f"- Файл: `{data_path.name}`; период: {fmt_ms(int(times.iloc[0]))} → {fmt_ms(int(times.iloc[-1]))} UTC")
    md.append(f"- Баров после очистки: {prep_report.n_output} (дубликатов удалено: {prep_report.n_duplicates_removed}, "
              f"пропусков: {prep_report.total_missing_bars}, незакрытых исключено: {prep_report.n_unconfirmed_excluded})")
    md.append(f"- Обучающих строк: {ds_report.n_rows} (исключено без будущего бара: {ds_report.n_dropped_no_future_bar}); "
              f"классы: up={ds_report.n_up} ({ds_report.n_up / max(ds_report.n_rows, 1):.1%}), "
              f"down/flat={ds_report.n_down_or_flat} ({ds_report.n_down_or_flat / max(ds_report.n_rows, 1):.1%})")
    md.append(f"- Цель (как в исходном ноутбуке): y=+1 если close(t+H·interval) > close(t), иначе −1; H={mc.horizon_bars} "
              f"(горизонт {mc.horizon_bars * ex.interval_minutes} мин)")
    md.append(f"- Признаки: набор(ы) {', '.join(variants_to_run)}; выбранный: **{chosen}** = {feature_columns(chosen)}")
    md.append(f"- Алгоритм: XGBClassifier(n_estimators={n_estimators}, max_depth={mc.max_depth}, "
              f"learning_rate={mc.learning_rate}, seed={mc.random_seed}) — параметры оригинала + фиксированный seed")
    md.append("")
    md.append("## 2. Walk-forward-валидация (chronological, expanding window, embargo)")
    md.append("")
    md.append(f"Схема: стартовое обучение {args.train_fraction:.0%} → {n_folds} последовательных тестовых окон; "
              f"переобучение на каждом окне. Embargo: в обучение попадают только строки, чья метка "
              f"полностью известна до начала окна (t + (H+1)·interval ≤ test_start) — утечка будущих данных исключена.")
    md.append("")
    for fs, wf in wf_results.items():
        md.append(f"### Набор признаков `{fs}` ({', '.join(feature_columns(fs))})")
        md.append("")
        md.append(folds_table_md(wf, threshold))
        md.append("")
        md.append(f"*Время счёта: {wf.elapsed_s:.1f} c.*")
        md.append("")
    md.append("## 3. Протокол исходного ноутбука на свежих данных")
    md.append("")
    md.append("Оригинальный артефакт модели в репозитории отсутствует (данные и модель не сохранены), поэтому")
    md.append("«исходная версия» воспроизведена протоколом: сырые OHLCV-признаки, единственный сплит 75/25, без embargo.")
    md.append("")
    md.append(f"- n_train={original_protocol['n_train']}, n_test={original_protocol['n_test']}")
    md.append(f"- Accuracy={original_protocol['accuracy']:.4f}, F1(+1)={original_protocol['f1_up']:.4f}, "
              f"Precision(+1)={original_protocol['precision_up']:.4f}, Recall(+1)={original_protocol['recall_up']:.4f}")
    md.append("")
    md.append("Метрики из README исходного репозитория проверить нельзя — датасетов автора там нет.")
    md.append("")
    md.append("## 4. Выбор и итоговое сравнение")
    md.append("")
    md.append(f"- {selection_note}")
    for fs, wf in wf_results.items():
        a = wf.aggregate
        md.append(f"- `{fs}`: acc={a['accuracy']:.4f}, f1(+1)={a['f1_up']:.4f}, majority={a['majority_accuracy']:.4f}, "
                  f"momentum={a['momentum_accuracy']:.4f}, покрытие@{threshold}={a['coverage_at_threshold']:.1%}, "
                  f"acc@{threshold}=" + (f"{a['accuracy_at_threshold']:.4f}" if a['accuracy_at_threshold'] == a['accuracy_at_threshold'] else "н/д"))
    md.append("")
    # статистическая значимость отличия от majority-baseline (агрегат)
    a_ch = wf_chosen.aggregate
    n_tot = max(int(a_ch["n_test_total"]), 1)
    se = float(np.sqrt(max(a_ch["accuracy"] * (1 - a_ch["accuracy"]), 1e-12) / n_tot))
    z = (a_ch["accuracy"] - a_ch["majority_accuracy"]) / se if se > 0 else float("nan")
    md.append(f"Статзначимость: агрегированное отличие `{chosen}` от majority-baseline "
              f"= {a_ch['accuracy'] - a_ch['majority_accuracy']:+.4f} (≈{z:.1f}σ при n={n_tot}, SE≈{se:.5f}). "
              f"Отличие статистически заметно, но МАЛО по величине (см. раздел 5).")
    md.append("")
    if model_version:
        md.append(f"**Сохранённый артефакт:** `data/models/{model_version}/` (model.joblib + model_card.json + walk_forward.json)")
    else:
        md.append("**Финальная модель не обучалась (--no-final).**")
    md.append("")
    md.append("## 5. Ограничения")
    md.append("")
    md.append("- Accuracy выше baseline **не означает прибыльность**: комиссии, спред и слиппедж не учитывались.")
    closes_np = df_clean["close"].to_numpy(dtype="float64")
    if len(closes_np) > 2:
        mean_abs_ret = float(np.mean(np.abs(np.diff(closes_np) / closes_np[:-1])))
        gross_edge = (2.0 * a_ch["accuracy"] - 1.0) * mean_abs_ret
        md.append(f"- Реалистичная оценка: среднее |движение| бара ≈ {mean_abs_ret:.4%}; ожидаемый валовой выигрыш "
                  f"на сигнал ≈ (2·acc−1)·E|Δ| ≈ **{gross_edge:.5%}** при acc={a_ch['accuracy']:.4f}. "
                  f"Комиссия спот-сделки round-trip (taker ~0.1%×2) ≈ 0.2% — на два порядка больше. "
                  f"Модель пригодна для исследовательского мониторинга и демонстрации, но не для комиссионной "
                  f"торговли на этом горизонте.")
    md.append("- Задача предсказания знака 5-минутного движения близка к эффективному рынку: реалистичный уровень — чуть выше 50%.")
    md.append("- Сырые уровни цен (набор `raw`) не экстраполируются деревьями за пределы обучающего диапазона — при дрейфе цены качество деградирует; walk-forward-переобучение обязательно.")
    md.append("- Выбор feature_set по тем же walk-forward-окнам, на которых он сравнивается, даёт небольшой оптимизм (selection bias).")
    md.append("- Результат зависит от рыночного режима в периоде обучения; рекомендуется периодическое переобучение и мониторинг live-качества.")
    md.append("- Версии библиотек: " + json.dumps(library_versions(), ensure_ascii=False))
    md.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")
    log.info("Отчёт сохранён: %s", report_path)

    print()
    print("=== ИТОГ ЭТАПА 2b ===")
    for fs, wf in wf_results.items():
        a = wf.aggregate
        print(f"[{fs:5s}] walk-forward acc={a['accuracy']:.4f} | f1(+1)={a['f1_up']:.4f} | "
              f"majority={a['majority_accuracy']:.4f} | momentum={a['momentum_accuracy']:.4f} | "
              f"покрытие@{threshold}={a['coverage_at_threshold']:.1%}")
    print(f"[оригинальный протокол 75/25, raw] acc={original_protocol['accuracy']:.4f}")
    print(f"Выбранный feature_set: {chosen}")
    if model_version:
        print(f"Артефакт: {out['model']}")
        print(f"Model card: {out['card']}")
        print(f"Model version: {model_version}")
    print(f"Отчёт: {report_path}")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
