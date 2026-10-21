"""CLI обучения: `python -m tradebot.cli.train`.

Пример (PowerShell):
  python -m tradebot.cli.train --config configs\ml_example.yaml --timeframe 5m

Полностью офлайн: читает CSV из data.csv_dir конфига, обучает LightGBM на
purged walk-forward разбиении, сохраняет артефакт модели и markdown-отчёт.
Секреты и сеть не требуются.

Коды возврата:
  0 — прогон завершён (вердикт по порогам может быть любым — он печатается);
  2 — ошибка конфигурации;
  3 — недостаточно истории для честной валидации;
  4 — ошибка данных/обучения.
"""

from __future__ import annotations

import argparse
import sys

from ..config import ConfigError, load_config
from ..data.bar import SUPPORTED_TIMEFRAMES
from ..data.csv_loader import DataValidationError
from ..ml.errors import InsufficientHistoryError, TrainingError
from ..ml.train import run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train", description="Обучение модели на walk-forward")
    parser.add_argument("--config", required=True, help="путь к YAML-конфигу")
    parser.add_argument("--timeframe", required=True, choices=SUPPORTED_TIMEFRAMES)
    parser.add_argument("--symbol", default=None, help="символ пары, например BTCUSDT (по умолчанию — единственный в конфиге)")
    parser.add_argument("--model-dir", default=None, help="переопределить run.model_dir")
    parser.add_argument("--report-dir", default=None, help="переопределить run.report_dir")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ОШИБКА КОНФИГУРАЦИИ:\n{exc}", file=sys.stderr)
        return 2

    print(
        f"Обучение: experiment={config.run.experiment_id} timeframe={args.timeframe} "
        f"model={config.model.kind} seed={config.model.seed}"
    )
    try:
        result = run_training(
            config,
            args.timeframe,
            symbol=args.symbol,
            model_dir=args.model_dir,
            report_dir=args.report_dir,
        )
    except InsufficientHistoryError as exc:
        print(f"\nНЕДОСТАТОЧНО ИСТОРИИ: {exc}", file=sys.stderr)
        return 3
    except (TrainingError, DataValidationError) as exc:
        print(f"\nОШИБКА ДАННЫХ/ОБУЧЕНИЯ: {exc}", file=sys.stderr)
        return 4

    print(f"\nДатасет: строк={result.dataset_info['n_rows']} "
          f"(warmup={result.dataset_info['warmup_dropped']}, "
          f"нейтраль={result.dataset_info['neutral_dropped']}, "
          f"хвост={result.dataset_info['label_tail_dropped']})")
    print(f"Фолдов: {len(result.folds)}; embargo={result.folds[0].embargo_bars if result.folds else '?'} баров")
    key = "roc_auc" if "roc_auc" in result.oos_metrics else "r2"
    print(f"OOS {key}: {result.oos_metrics.get(key)}")
    print(f"OOS trading: net={result.oos_trading['net_return']:.4f}, сделок={result.oos_trading['n_trades']}, "
          f"maxDD={result.oos_trading['max_drawdown']:.4f}")
    for name, metrics in result.baselines_oos_metrics.items():
        base_key = metrics.get("roc_auc", metrics.get("r2"))
        print(f"  baseline {name}: {key}={base_key}")
    if result.audit and result.audit.get("applicable"):
        print(f"Аудит утечек: delta(random-wf)={result.audit.get('delta')}, flagged={result.audit.get('flagged')}")
    print(f"Вердикт: {result.verdict['text']}")
    print(f"Модель:  {result.model_dir}")
    print(f"Отчёт:   {result.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
