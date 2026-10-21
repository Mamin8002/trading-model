"""Единая точка запуска проекта: `python -m tradebot <команда>`.

Команды:
  test           — полный TEST-путь: CSV -> (обучение ИЛИ загрузка модели) ->
                   сигналы -> риск-контроль -> симулятор -> результаты + отчёт + журнал.
                   Работает офлайн на локальных CSV; mode: LIVE в конфиге отклоняется.
  live-preflight — проверка многофакторного LIVE-гейта (docs/live.md). В крипто-версии
                   интеграция с биржей НЕ реализована (exchange/base.py), поэтому гейт
                   ВСЕГДА отказывает: команда печатает причины и завершается с кодом 2.
                   Сетевые вызовы не выполняются, заявки невозможны.

Коды возврата: 0 — успех; 2 — ошибка конфигурации/гейта;
3 — недостаточно истории; 4 — ошибка данных/модели/симуляции.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import ConfigError, load_config
from ..data.bar import SUPPORTED_TIMEFRAMES


def _cmd_test(args: argparse.Namespace) -> int:
    from ..backtest.engine import run_simulation
    from ..backtest.results import compute_results, render_simulation_report
    from ..data.csv_loader import DataValidationError
    from ..execution.errors import SimConfigError
    from ..execution.journal import JournalWriter
    from ..ml.errors import InsufficientHistoryError, ModelCompatibilityError, TrainingError
    from ..ml.registry import load_model_bundle
    from ..ml.train import load_bars_for_timeframe, run_training
    from ..strategy.signals import ModelSignalSource

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ:\n{exc}", file=sys.stderr)
        return 2
    if config.run.mode != "TEST":
        print(
            f"ОТКАЗ: run.mode={config.run.mode!r}. Команда `test` работает только с mode: TEST. "
            "LIVE-контур — отдельная команда live-preflight за многофакторным гейтом (docs/live.md); "
            "заявки не отправлялись.",
            file=sys.stderr,
        )
        return 2

    # fail-fast: неполная конфигурация риска/симуляции блокирует прогон ДО загрузки
    # данных и обучения (задача: «при неполной конфигурации завершаться без отправки заявок»)
    from ..execution.simulator import validate_simulation_config
    try:
        validate_simulation_config(config.simulation, config.risk, require_risk=True)
    except SimConfigError as exc:
        print(f"НЕПОЛНАЯ КОНФИГУРАЦИЯ (безопасный отказ, заявки не отправлялись):\n{exc}", file=sys.stderr)
        return 2

    try:
        print(f"[1/5] Данные: загрузка {args.timeframe} ...")
        bars, symbol = load_bars_for_timeframe(config, args.timeframe, symbol=args.symbol)
        print(f"      баров: {len(bars)}; символ: {symbol}")

        if args.model_dir:
            print(f"[2/5] Модель: загрузка артефакта {args.model_dir} ...")
            bundle = load_model_bundle(args.model_dir)
            bundle.validate_compatibility(timeframe=args.timeframe)
        elif args.train:
            print("[2/5] Модель: обучение (walk-forward, фаза 3) ...")
            train_result = run_training(
                config, args.timeframe, symbol=symbol, bars=bars,
                model_dir=args.model_output_dir, report_dir=args.report_output_dir,
            )
            print(f"      OOS-метрики: { {k: v for k, v in train_result.oos_metrics.items() if k in ('n', 'roc_auc', 'mae')} }")
            print(f"      вердикт (диагностический): {train_result.verdict['text']}")
            print(f"      отчёт обучения: {train_result.report_path}")
            bundle = load_model_bundle(train_result.model_dir)
        else:
            print(
                "ОШИБКА: укажите --model-dir <каталог артефакта> или --train "
                "(явный выбор источника модели обязателен)",
                file=sys.stderr,
            )
            return 2

        print("[3/5] Сигналы: ModelSignalSource ...")
        source = ModelSignalSource(
            bundle, config.simulation.signal,
            symbol=symbol, timeframe=args.timeframe,
            local_timezone=config.data.source_timezone,
        ) if config.simulation else None
        if source is None:
            print("ОШИБКА: секция simulation не задана (комиссии/слиппедж обязательны явно)", file=sys.stderr)
            return 2

        print("[4/5] Симуляция: риск-контроль + EventSimulator + журнал ...")
        report_dir = Path(args.report_output_dir or config.run.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        journal_path = Path(
            config.simulation.journal_path
            or report_dir / f"journal_{config.run.experiment_id}_{args.timeframe}.jsonl"
        )
        with JournalWriter(journal_path, symbol=symbol, timeframe=args.timeframe,
                           model_version=source.model_version) as journal:
            run = run_simulation(config, bars, source, timeframe=args.timeframe,
                                 symbol=symbol, journal=journal)

        print("[5/5] Результаты и отчёт ...")
        results = compute_results(run, config, bars)
        report_path = report_dir / f"test_run_{config.run.experiment_id}_{args.timeframe}.md"
        report_path.write_text(render_simulation_report(results, config, str(journal_path)), encoding="utf-8")

        print()
        print(f"TEST-прогон завершён: {symbol} {args.timeframe}, баров={results.bars_processed}")
        print(f"Капитал: {results.initial_capital:.2f} -> {results.final_equity:.2f} ({results.net_return * 100:.3f}%)")
        print(f"Комиссии: {results.total_commission:.4f} | слиппедж (оценка): {results.estimated_slippage_cost:.4f}")
        print(f"Входов: {results.n_entries} | завершённых сделок: {results.n_round_trips} | отказов риска: {results.n_rejections}")
        if results.rejections_by_code:
            print("Отказы:", ", ".join(f"{k}={v}" for k, v in sorted(results.rejections_by_code.items())))
        if results.n_halts:
            print(f"ОСТАНОВКИ: {results.halt_reasons}")
        for warning in results.warnings[:5]:
            print(f"warning: {warning}")
        print(f"Журнал: {journal_path}")
        print(f"Отчёт:  {report_path}")
        print("Реальные заявки не выставлялись: это TEST-симуляция (docs/simulation.md).")
        return 0
    except InsufficientHistoryError as exc:
        print(f"\nНЕДОСТАТОЧНО ИСТОРИИ: {exc}", file=sys.stderr)
        return 3
    except SimConfigError as exc:
        print(f"\nНЕПОЛНАЯ КОНФИГУРАЦИЯ (безопасный отказ, заявки не отправлялись):\n{exc}", file=sys.stderr)
        return 2
    except (ModelCompatibilityError, FileNotFoundError) as exc:
        print(f"\nОШИБКА МОДЕЛИ: {exc}", file=sys.stderr)
        return 4
    except (TrainingError, DataValidationError, ValueError) as exc:
        print(f"\nОШИБКА ДАННЫХ/СИМУЛЯЦИИ: {exc}", file=sys.stderr)
        return 4


def _cmd_live_preflight(args: argparse.Namespace) -> int:
    from ..live.gate import evaluate_live_gate

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ:\n{exc}", file=sys.stderr)
        return 2

    print("LIVE-гейт (docs/live.md): проверка условий ДО любых сетевых вызовов ...")
    gate = evaluate_live_gate(config, cli_flag=args.allow_live)
    for failure in gate.failures:
        print(f"  ОТКАЗ: {failure}", file=sys.stderr)
    if not gate.allowed:
        print(
            "\nLIVE-гейт НЕ пройден — сетевые вызовы не выполнялись, заявки невозможны.\n"
            "Подключение к криптобирже не реализовано (exchange/base.py): LIVE заблокирован\n"
            "до отдельной реализации биржевого адаптера (docs/live.md, docs/roadmap.md).",
            file=sys.stderr,
        )
        return 2
    # Недостижимо, пока интеграция с биржей отсутствует (условие №0 гейта).
    for warning in gate.warnings:
        print(f"  ВНИМАНИЕ: {warning}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tradebot", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_test = sub.add_parser("test", help="полный TEST-путь: данные -> модель -> симуляция -> отчёт/журнал")
    p_test.add_argument("--config", required=True)
    p_test.add_argument("--timeframe", required=True, choices=SUPPORTED_TIMEFRAMES)
    p_test.add_argument("--symbol", default=None)
    model_source = p_test.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model-dir", default=None, help="каталог артефакта модели фазы 3")
    model_source.add_argument("--train", action="store_true", help="обучить модель в этом прогоне (walk-forward)")
    p_test.add_argument("--model-output-dir", default=None, help="куда сохранить артефакт при --train")
    p_test.add_argument("--report-output-dir", default=None)
    p_test.set_defaults(func=_cmd_test)

    p_live = sub.add_parser("live-preflight",
                            help="проверка LIVE-гейта (сейчас всегда отказ: биржа не подключена)")
    p_live.add_argument("--config", required=True)
    p_live.add_argument("--allow-live", action="store_true",
                        help="явное подтверждение LIVE-контура (без флага гейт не проходит)")
    p_live.set_defaults(func=_cmd_live_preflight)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
