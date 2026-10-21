"""CLI проверки данных: `python -m tradebot.cli.validate_data`.

Полностью офлайн: сеть и секреты не требуются.

Примеры (PowerShell):
  python -m tradebot.cli.validate_data --config configs/demo.yaml
  python -m tradebot.cli.validate_data --csv data/demo/DEMO_M1.csv --timeframe 1m
  python -m tradebot.cli.validate_data --config configs/demo.yaml --resample 10m

Код возврата: 0 — ошибок нет (предупреждения допустимы), 1 — найдены ошибки,
2 — ошибка использования (неверные аргументы/конфиг).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import ConfigError, build_calendar, config_hash, load_config
from ..data.bar import SUPPORTED_TIMEFRAMES, Timeframe
from ..data.csv_loader import DataValidationError, load_csv
from ..data.resampler import resample
from ..data.validator import ValidationReport


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _validate_one(
    path: Path,
    timeframe: Timeframe,
    symbol: str,
    source_timezone: str,
    timestamp_semantics: str,
    calendar,
    warn_zero_volume: bool,
    max_issues: int,
) -> bool:
    _print_header(f"{path} [{timeframe.value}] symbol={symbol or '-'}")
    try:
        result = load_csv(
            path,
            timeframe=timeframe,
            symbol=symbol,
            source_timezone=source_timezone,
            timestamp_semantics=timestamp_semantics,  # type: ignore[arg-type]
            calendar=calendar,
            warn_zero_volume=warn_zero_volume,
        )
    except DataValidationError as exc:
        print(exc.report.summary_text(max_issues=max_issues))
        print("СТАТУС: ОШИБКА — файл не пригоден к использованию")
        return False
    bars = result.bars
    print(f"Баров: {len(bars)}")
    if bars:
        print(f"Период (UTC): {bars[0].ts_open.isoformat()} .. {bars[-1].ts_open.isoformat()}")
        print(f"Семантика метки: {timestamp_semantics} -> нормализовано к bar_open (UTC)")
    warnings = result.report.warnings
    if warnings:
        print(f"ПРЕДУПРЕЖДЕНИЙ: {len(warnings)}")
        for issue in warnings[:max_issues]:
            print("  " + issue.format())
        if len(warnings) > max_issues:
            print(f"  ... и ещё {len(warnings) - max_issues}")
    else:
        print("Предупреждений нет")
    print("СТАТУС: OK")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validate_data", description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="путь к YAML-конфигурации")
    source.add_argument("--csv", type=Path, help="путь к одному CSV-файлу")
    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        help="таймфрейм файла (обязателен с --csv)",
    )
    parser.add_argument("--symbol", default="", help="символ (по умолчанию пустой)")
    parser.add_argument(
        "--source-timezone",
        default="Europe/Moscow",
        help="таймзона naive-меток CSV (по умолчанию Europe/Moscow)",
    )
    parser.add_argument(
        "--timestamp-semantics",
        choices=("bar_open", "bar_close"),
        help="семантика метки (обязательна с --csv; с --config берётся из data.timestamp_semantics)",
    )
    parser.add_argument(
        "--resample",
        choices=SUPPORTED_TIMEFRAMES,
        help="дополнительно: агрегировать загруженные 1m/5m в указанный таймфрейм и напечатать сводку",
    )
    args = parser.parse_args(argv)

    if args.csv is not None:
        if not args.timeframe:
            parser.error("--timeframe обязателен при --csv")
        if not args.timestamp_semantics:
            parser.error(
                "--timestamp-semantics обязателен при --csv: семантика метки должна задаваться явно (docs/data.md)"
            )
        ok = _validate_one(
            args.csv,
            Timeframe.parse(args.timeframe),
            args.symbol,
            args.source_timezone,
            args.timestamp_semantics,
            calendar=None,
            warn_zero_volume=True,
            max_issues=50,
        )
        return 0 if ok else 1

    # режим --config
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ:\n{exc}", file=sys.stderr)
        return 2
    print(f"Конфиг: {args.config} (mode={config.run.mode}, experiment_id={config.run.experiment_id})")
    print(f"Хэш конфига: {config_hash(config)}")
    calendar = build_calendar(config)
    print(f"Календарь сессий: {'настроен (' + config.sessions.timezone + ')' if calendar else 'не задан (пропуски считаются по равномерной сетке)'}")

    all_ok = True
    for symbol_cfg in config.data.symbols:
        for tf_code, file_name in sorted(symbol_cfg.files.items()):
            path = config.data.csv_dir / file_name
            ok = _validate_one(
                path,
                Timeframe.parse(tf_code),
                symbol_cfg.symbol,
                config.data.source_timezone,
                config.data.timestamp_semantics,
                calendar=calendar,
                warn_zero_volume=config.validation.warn_zero_volume,
                max_issues=config.validation.max_reported_issues,
            )
            all_ok = all_ok and ok

            if args.resample and ok:
                target = Timeframe.parse(args.resample)
                source_tf = Timeframe.parse(tf_code)
                if target.minutes <= source_tf.minutes:
                    print(f"(ресэмплинг {source_tf.value}->{target.value} пропущен: целевой ТФ не крупнее)")
                    continue
                if target.minutes % source_tf.minutes:
                    print(f"(ресэмплинг {source_tf.value}->{target.value} невозможен: не кратны)")
                    continue
                try:
                    loaded = load_csv(
                        path,
                        timeframe=source_tf,
                        symbol=symbol_cfg.symbol,
                        source_timezone=config.data.source_timezone,
                        timestamp_semantics=config.data.timestamp_semantics,
                        calendar=calendar,
                        warn_zero_volume=config.validation.warn_zero_volume,
                    )
                except DataValidationError:
                    continue
                result = resample(loaded.bars, source_tf, target, calendar=calendar, symbol=symbol_cfg.symbol)
                _print_header(f"Ресэмплинг {source_tf.value} -> {target.value}: {path.name}")
                print(f"Полных баров: {len(result.bars)}; pending (неполный хвост): {'да' if result.pending else 'нет'}")
                for issue in result.issues[: config.validation.max_reported_issues]:
                    print(f"  [{issue.code.value}] {issue.message}")
                if result.issues:
                    all_ok = False
                    print("СТАТУС: ОШИБКА — есть неполные интервалы (см. issues)")
                else:
                    print("СТАТУС: OK")

    print("\nИТОГ:", "OK" if all_ok else "ЕСТЬ ОШИБКИ")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
