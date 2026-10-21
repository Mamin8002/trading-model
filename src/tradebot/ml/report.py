"""Генерация markdown-отчёта об обучении (docs/ml.md §6).

Отчёт ОБЯЗАН содержать: допущения (комиссии, слиппедж, модель исполнения,
годовой пересчёт), сравнение с базовыми ориентирами, метрики по фолдам и OOS,
аудит утечек, вердикт по порогам и дисклеймер: результаты диагностические,
не являются обещанием доходности (decisions D15).
"""

from __future__ import annotations

from typing import Any

from .dataset import MlDataset
from .train import TrainingRunResult

__all__ = ["render_markdown"]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "н/д"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metrics_table(rows: list[tuple[str, dict[str, Any]]], classification: bool) -> list[str]:
    if classification:
        header = "| источник | n | accuracy | bal.acc | ROC-AUC | log-loss |"
        sep = "|---|---|---|---|---|---|"
        lines = [header, sep]
        for name, m in rows:
            lines.append(
                f"| {name} | {m.get('n')} | {_fmt(m.get('accuracy'))} | {_fmt(m.get('balanced_accuracy'))} "
                f"| {_fmt(m.get('roc_auc'))} | {_fmt(m.get('log_loss'))} |"
            )
    else:
        header = "| источник | n | MAE | RMSE | R² |"
        sep = "|---|---|---|---|---|"
        lines = [header, sep]
        for name, m in rows:
            lines.append(f"| {name} | {m.get('n')} | {_fmt(m.get('mae'), 6)} | {_fmt(m.get('rmse'), 6)} | {_fmt(m.get('r2'))} |")
    return lines


def _trading_table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "| источник | баров | сделок | exposure | валовая | издержки | чистая | maxDD | hit-rate | PF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in rows:
        lines.append(
            f"| {name} | {s.get('n_bars')} | {s.get('n_trades')} | {_fmt(s.get('exposure'), 3)} "
            f"| {_fmt(s.get('gross_return'))} | {_fmt(s.get('cost_return'))} | {_fmt(s.get('net_return'))} "
            f"| {_fmt(s.get('max_drawdown'))} | {_fmt(s.get('hit_rate'), 3)} | {_fmt(s.get('profit_factor'), 2)} |"
        )
    return lines


def render_markdown(result: TrainingRunResult, dataset: MlDataset) -> str:
    classification = dataset.is_classification
    lines: list[str] = []
    add = lines.append

    add(f"# Отчёт об обучении: {result.experiment_id} ({result.symbol}, {result.timeframe})")
    add("")
    add("> **Дисклеймер.** Все торговые метрики ниже — результаты упрощённой диагностической")
    add("> симуляции на истории. Они НЕ являются обещанием или оценкой будущей доходности.")
    add("> Система не объявляется production-ready (docs/decisions.md D15).")
    add("")
    add("## 1. Параметры прогона")
    add("")
    add(f"- символ: `{result.symbol}`; таймфрейм: `{result.timeframe}`")
    add(f"- горизонт метки H: {result.horizon_bars} баров; lookback: {result.lookback_bars} баров")
    add(f"- метка: `{result.label['kind']}`, нейтральная полоса: {result.label['neutral_band_bps']} б.п.")
    add(f"- модель: LightGBM (основной кандидат, decisions D4); seed: {result.seed}")
    add(f"- период данных: {result.training_period[0]} .. {result.training_period[1]}")
    add(f"- хэш конфига: `{result.config_hash[:16]}`; fingerprint датасета: `{result.dataset_fingerprint}`")
    info = result.dataset_info
    add(
        f"- строк датасета: {info['n_rows']} из {info['n_bars']} баров "
        f"(прогрев исключён: {info['warmup_dropped']}, нейтраль: {info['neutral_dropped']}, "
        f"хвост метки: {info['label_tail_dropped']}, хвост исполнения: {info['execution_tail_dropped']})"
    )
    balance = info.get("target_info", {}).get("class_balance") or {}
    if balance:
        add(f"- баланс классов: up={_fmt(balance.get('up'), 3)}, down={_fmt(balance.get('down'), 3)}")
    add("")
    add("## 2. Признаки (набор v1)")
    add("")
    add(f"- версия набора: `{dataset.spec.version}`; число колонок: {len(dataset.feature_names)};")
    add(f"- warmup (минимум баров до первого валидного признака): {dataset.spec.warmup_bars};")
    add("- все признаки причинные: значение в t использует только бары ≤ t")
    add("  (подтверждено тестами префикс-инвариантности и возмущения будущего);")
    add("- полный список колонок — в `feature_manifest.json` артефакта модели.")
    add("")
    add("## 3. Целевая переменная")
    add("")
    add(f"- `fwd_return(t) = close[t+{result.horizon_bars}] / close[t] − 1` по закрытым свечам {result.timeframe};")
    if classification:
        add(f"- класс 1 при fwd_return > +band, класс 0 при < −band, band = {result.label['neutral_band_bps']} б.п.;")
        add("- строки внутри полосы исключены из обучения (их число — в разделе 1);")
    add("- будущая информация используется ТОЛЬКО в метке и никогда — в признаках.")
    add("")
    add("## 4. Walk-forward разбиение (purged, expanding)")
    add("")
    add("| фолд | train (диапазон / строк) | embargo (баров) | val (диапазон / строк) | best_iter |")
    add("|---|---|---|---|---|")
    for fold in result.folds:
        add(
            f"| {fold.fold_id} | {fold.train_range[0]} .. {fold.train_range[1]} / {fold.n_train} "
            f"| {fold.embargo_bars} | {fold.val_range[0]} .. {fold.val_range[1]} / {fold.n_val} "
            f"| {fold.best_iteration if fold.best_iteration else '—'} |"
        )
    add("")
    add("Инвариант `max(train) + H < min(val)` проверен программно для каждого фолда")
    add("(пересечение целевых горизонтов исключено; случайное перемешивание не используется).")
    add("")
    add("## 5. Метрики ML по фолдам (валидационные окна)")
    add("")
    for fold in result.folds:
        add(f"### Фолд {fold.fold_id}")
        add("")
        rows = [("модель (val)", fold.model_metrics)]
        for name in ("majority", "momentum", "logreg"):
            metrics = fold.baseline_metrics.get(name)
            # note-only словари (без "n") — ориентиры, неприменимые к данной задаче
            if metrics is not None and "n" in metrics:
                rows.append((f"{name} (val)", metrics))
        lines.extend(_metrics_table(rows, classification))
        train_diag = fold.model_train_metrics_diagnostic
        if classification:
            add("")
            add(
                f"_Диагностика переобучения (не оценка качества): train ROC-AUC={_fmt(train_diag.get('roc_auc'))} "
                f"vs val ROC-AUC={_fmt(fold.model_metrics.get('roc_auc'))}_"
            )
        add("")
    add("## 6. Итоговый out-of-sample (объединение валидационных окон)")
    add("")
    oos_rows = [("МОДЕЛЬ (OOS)", result.oos_metrics)]
    for name in ("majority", "momentum", "logreg"):
        if name in result.baselines_oos_metrics:
            oos_rows.append((f"{name} (OOS)", result.baselines_oos_metrics[name]))
    lines.extend(_metrics_table(oos_rows, classification))
    add("")
    comparison = result.verdict.get("baseline_comparison", {})
    if comparison:
        add("Сравнение с базовыми ориентирами (decisions D4 — модель обязана обыгрывать ориентиры):")
        add("")
        for name, comp in comparison.items():
            add(f"- vs `{name}`: AUC модели {_fmt(result.oos_metrics.get('roc_auc'))} vs {_fmt(comp.get('auc'))} -> "
                f"{'модель выше' if comp.get('model_beats') else ('НЕ выше' if comp.get('model_beats') is False else 'н/д')}")
        add("")
    add("## 7. Торговые метрики OOS (упрощённая диагностическая симуляция)")
    add("")
    trading_rows = [("МОДЕЛЬ", result.oos_trading)]
    for name in ("majority", "momentum", "logreg"):
        if name in result.baselines_oos_trading:
            trading_rows.append((name, result.baselines_oos_trading[name]))
    lines.extend(_trading_table(trading_rows))
    add("")
    ann = result.oos_trading_annualized
    add(f"- годовой пересчёт: CAGR={_fmt(ann.get('cagr'))}, Sharpe={_fmt(ann.get('sharpe'), 2)} "
        f"(bars_per_year={_fmt(ann.get('bars_per_year'), 1)}, источник: {ann.get('bars_per_year_source')})")
    add("")
    add("### Допущения симуляции и годового пересчёта (обязательный раздел)")
    add("")
    a = result.assumptions
    add(f"- комиссия: {a['commission_bps']} б.п.; слиппедж: {a['slippage_bps']} б.п. — ОЦЕНОЧНЫЕ допущения,")
    add("  а не фактические тарифы брокера (research Q9);")
    add(f"- исполнение: {a['execution']};")
    add("- издержки удерживаются с оборота |Δпозиции|, включая ликвидацию в конце сегмента;")
    add("- сегменты фолдов независимы: позиция не переносится через разрывы между окнами;")
    bpr_note = (
        "из конфига"
        if a["bars_per_year_source"] == "config"
        else f"медиана {a['bars_per_day_median']} баров/день × {a['trading_days_per_year']} торговых дней/год"
    )
    add(f"- годовой пересчёт: bars_per_year={_fmt(a['bars_per_year'], 1)} ({bpr_note});")
    add(f"- безрисковая ставка для Sharpe: {a['risk_free_annual']} (годовых);")
    add("- полный симулятор (стопы, внутрибарные филлы, бюджет) — этап B1 roadmap; до него")
    add("  торговые цифры носят исключительно диагностический характер.")
    add("")
    add("## 8. Аудит утечек (random-split vs walk-forward)")
    add("")
    audit = result.audit or {}
    if not audit.get("enabled"):
        add("Аудит отключён конфигом (validation.walk_forward.audit_random_split=false).")
    elif not audit.get("applicable", True):
        add(f"Аудит не применим: {audit.get('note', '')}.")
    else:
        add(f"- AUC на нехронологическом сплите (аудит): {_fmt(audit.get('auc_random_split'))}")
        add(f"- AUC walk-forward OOS: {_fmt(audit.get('auc_walk_forward_oos'))}")
        add(f"- дельта: {_fmt(audit.get('delta'))} (порог флага: {audit.get('flag_threshold')})")
        add(f"- вывод: {audit.get('interpretation')}")
    add("")
    add("## 9. Вердикт по заранее заданным порогам")
    add("")
    checks = result.verdict.get("checks") or []
    if not checks:
        add("Пороги в конфиге не заданы (validation.thresholds) — вердикт информационный.")
    for check in checks:
        status = {True: "ПРОЙДЕН", False: "НЕ ПРОЙДЕН", None: "н/д"}[check["passed"]]
        add(f"- `{check['check']}`: {status} — {check['detail']}")
    add("")
    add(f"**{result.verdict['text']}**")
    add("")
    add("Метрики на обучающей выборке в отчёте приведены ТОЛЬКО как диагностика переобучения")
    add("и оценкой качества не являются.")
    add("")
    add("## 10. Воспроизводимость и версии")
    add("")
    add(f"- seed: {result.seed}; LightGBM: deterministic=true, force_row_wise=true, n_jobs=1;")
    add("- порядок данных не перемешивается; all преобразования/фиты — только внутри train-окон фолдов;")
    add(f"- финальная модель: n_estimators={result.final_n_estimators} (медиана best_iteration фолдов"
        " при включённом early stopping, иначе значение конфига). Финальная модель обучена на всех")
    add("  строках и приведённые метрики ей НЕ присвоены — они получены только на валидационных окнах;")
    add(f"- версии: {result.versions}")
    add("")
    add("## 11. Артефакты")
    add("")
    add(f"- каталог модели: `{result.model_dir}` (model.lgb.txt, metadata.json, feature_manifest.json, oos_predictions.csv)")
    add(f"- отчёт: `{result.report_path}`")
    add("")
    return "\n".join(lines)
