"""CLI симуляции: `python -m tradebot.cli.simulate`.

Полностью офлайн: CSV-история из конфига + артефакт модели фазы 3 (локальный
каталог). Сеть и секреты не требуются; РЕАЛЬНЫЕ ЗАЯВКИ НЕ ВЫСТАВЛЯЮТСЯ —
это TEST-симулятор (docs/simulation.md).

Пример (PowerShell):
  python -m tradebot.cli.simulate --config configs\\sim_demo.yaml --timeframe 5m --model-dir models\\<артефакт>

Коды возврата: 0 — симуляция выполнена (включая безопасную остановку — она
отражена в отчёте); 2 — ошибка конфигурации (в т.ч. неполные risk/simulation);
4 — ошибка данных/модели/симуляции.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..backtest.engine import run_simulation
from ..backtest.results import compute_results, render_simulation_report
from ..config import ConfigError, load_config
from ..data.bar import SUPPORTED_TIMEFRAMES
from ..data.csv_loader import DataValidationError
from ..execution.errors import SimConfigError
from ..execution.journal import JournalWriter
from ..ml.errors import ModelCompatibilityError, TrainingError
from ..ml.registry import load_model_bundle
from ..ml.train import load_bars_for_timeframe
from ..strategy.signals import ModelSignalSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simulate", description="TEST-симуляция с риск-контролем (без реальных заявок)")
    parser.add_argument("--config", required=True, help="путь к YAML-конфигу (секции risk/instrument/simulation обязательны)")
    parser.add_argument("--timeframe", required=True, choices=SUPPORTED_TIMEFRAMES)
    parser.add_argument("--symbol", default=None, help="символ пары, например BTCUSDT (по умолчанию — единственный в конфиге)")
    parser.add_argument("--model-dir", required=True, help="каталог артефакта модели фазы 3 (models/<...>)")
    parser.add_argument("--journal", default=None, help="путь JSONL-журнала (по умолчанию из конфига/report_dir)")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ:\n{exc}", file=sys.stderr)
        return 2
    if config.risk is None or config.simulation is None:
        print(
            "ОШИБКА КОНФИГУРАЦИИ: для симуляции обязательны секции risk и simulation "
            "(явные лимиты, капитал, комиссии, слиппедж — скрытые значения не подставляются)",
            file=sys.stderr,
        )
        return 2

    try:
        bars, symbol = load_bars_for_timeframe(config, args.timeframe, symbol=args.symbol)
        bundle = load_model_bundle(args.model_dir)
        bundle.validate_compatibility(timeframe=args.timeframe)
        source = ModelSignalSource(
            bundle,
            config.simulation.signal,
            symbol=symbol,
            timeframe=args.timeframe,
            local_timezone=config.data.source_timezone,
        )
        journal_path = Path(
            args.journal
            or config.simulation.journal_path
            or Path(config.run.report_dir) / f"journal_{config.run.experiment_id}_{args.timeframe}.jsonl"
        )
        with JournalWriter(
            journal_path,
            symbol=symbol,
            timeframe=args.timeframe,
            model_version=source.model_version,
        ) as journal:
            run = run_simulation(
                config, bars, source, timeframe=args.timeframe, symbol=symbol, journal=journal
            )
        results = compute_results(run, config, bars)
        report_dir = Path(config.run.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"sim_report_{config.run.experiment_id}_{args.timeframe}.md"
        report_path.write_text(render_simulation_report(results, config, str(journal_path)), encoding="utf-8")
    except SimConfigError as exc:
        print(f"НЕПОЛНАЯ КОНФИГУРАЦИЯ СИМУЛЯЦИИ (безопасный отказ):\n{exc}", file=sys.stderr)
        return 2
    except (ModelCompatibilityError, FileNotFoundError) as exc:
        print(f"ОШИБКА МОДЕЛИ: {exc}", file=sys.stderr)
        return 4
    except (TrainingError, DataValidationError, ValueError) as exc:
        print(f"ОШИБКА ДАННЫХ/СИМУЛЯЦИИ: {exc}", file=sys.stderr)
        return 4

    print(f"Симуляция: {results.symbol} {results.timeframe}, баров={results.bars_processed}, модель={results.model_version}")
    print(f"Капитал: {results.initial_capital:.2f} -> {results.final_equity:.2f} ({results.net_return * 100:.3f}%)")
    print(f"Комиссии: {results.total_commission:.4f} | слиппедж (оценка): {results.estimated_slippage_cost:.4f}")
    print(f"Сделок: входов={results.n_entries}, завершено={results.n_round_trips}, отказов={results.n_rejections}")
    if results.rejections_by_code:
        print("Отказы:", ", ".join(f"{k}={v}" for k, v in sorted(results.rejections_by_code.items())))
    print(f"maxDD={results.max_drawdown:.4f} | hit-rate={results.hit_rate} | конфликтов стоп/цель={results.n_conflicts}")
    if results.n_halts:
        print(f"ОСТАНОВКИ: {results.halt_reasons} (см. отчёт и журнал)")
    for warning in results.warnings[:5]:
        print(f"warning: {warning}")
    print(f"Журнал: {journal_path}")
    print(f"Отчёт:  {report_path}")
    print("Реальные заявки не выставлялись: это TEST-симулятор (docs/simulation.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
