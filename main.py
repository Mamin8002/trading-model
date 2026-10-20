"""
main.py — Точка входа: инициализация логирования, разбор CLI/ENV, запуск бота.

Использование:
    python main.py                          # TEST-режим (бэктест по CSV)
    python main.py --gen-data               # сгенерировать демо-историю и запустить тест
    python main.py --mode LIVE              # боевой режим (нужны ключи Finam в ENV)
    python main.py --timeframe 15m          # переопределить ТФ
    python main.py --symbol GAZP --csv data/gazp_15m.csv
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from config import CONFIG, BotConfig, TIMEFRAME_MINUTES


def setup_logging(log_file: str, level: str = "INFO") -> None:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-10s | %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(sh)
    root.addHandler(fh)
    # приглушить болтливые библиотеки
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finam NN-Trade Robot (tabular ML edition)")
    p.add_argument("--mode", choices=["TEST", "LIVE"], default=None)
    p.add_argument("--timeframe", choices=list(TIMEFRAME_MINUTES), default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--csv", dest="csv_path", default=None)
    p.add_argument("--threshold", type=float, default=None, help="порог уверенности сигнала 0..1")
    p.add_argument("--risk", dest="risk_pct", type=float, default=None, help="риск на сделку, %")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--gen-data", action="store_true",
                   help="сгенерировать синтетическую историю в CONFIG.csv_path и выйти? нет — затем запустить")
    p.add_argument("--train-only", action="store_true",
                   help="только обучить модель и вывести walk-forward метрики (без торговли)")
    return p.parse_args()


def apply_overrides(cfg: BotConfig, a: argparse.Namespace) -> BotConfig:
    if a.mode:
        cfg.mode = a.mode
    if a.timeframe:
        cfg.timeframe = a.timeframe
    if a.symbol:
        cfg.symbol = a.symbol
    if a.csv_path:
        cfg.csv_path = a.csv_path
    if a.threshold is not None:
        cfg.signal_threshold = a.threshold
    if a.risk_pct is not None:
        cfg.risk_per_trade_pct = a.risk_pct
    cfg.validate()
    return cfg


async def run_train_only(cfg: BotConfig) -> None:
    """Обучение + walk-forward отчёт без торгового цикла."""
    from data_loader import CsvDataLoader
    from features import build_dataset
    from model import TradeModel

    loader = CsvDataLoader(cfg)
    df = await loader.fetch_history()
    X, y, feats = build_dataset(df, horizon_bars=cfg.tf.horizon_bars)
    log = logging.getLogger("main")
    log.info("Dataset: %d строк x %d фич. Классы: up=%.1f%% flat=%.1f%% down=%.1f%%",
             len(X), len(feats),
             (y == 1).mean() * 100, (y == 0).mean() * 100, (y == -1).mean() * 100)
    m = TradeModel(cfg.model_type)
    acc, f1, windows = m.walk_forward_cv(X, y, n_splits=cfg.wf_n_splits)
    log.info("Walk-Forward: accuracy=%.4f f1=%.4f", acc, f1)
    for w in windows:
        log.info("  fold %d: test_n=%d acc=%.4f f1=%.4f",
                 w["fold"], w["test_n"], w["accuracy"], w["f1_weighted"])
    m.save(f"{cfg.model_dir}/{cfg.symbol}_{cfg.timeframe}.joblib")


async def amain() -> None:
    args = parse_args()
    cfg = apply_overrides(CONFIG, args)
    setup_logging(cfg.log_file, args.log_level)
    log = logging.getLogger("main")

    log.info("=" * 72)
    log.info("FINAM NN-TRADE ROBOT | mode=%s symbol=%s tf=%s threshold=%.2f risk=%.2f%%",
             cfg.mode, cfg.symbol, cfg.timeframe, cfg.signal_threshold, cfg.risk_per_trade_pct)
    log.info("Config: %s", cfg.to_dict())
    log.info("=" * 72)

    if args.gen_data:
        from data_loader import generate_synthetic_csv
        generate_synthetic_csv(cfg.csv_path, n_bars=max(cfg.tf.min_history_bars + 2000, 6000),
                               tf_minutes=TIMEFRAME_MINUTES[cfg.timeframe],
                               start_price=274.0)

    if args.train_only:
        await run_train_only(cfg)
        return

    from bot import build_runner
    runner = build_runner(cfg)
    try:
        if cfg.mode == "TEST":
            await runner.run()
        else:
            await runner.run()  # бесконечный цикл; Ctrl-C для остановки
    except KeyboardInterrupt:
        log.info("Получен SIGINT — остановка")
    finally:
        if hasattr(runner, "shutdown"):
            await runner.shutdown()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")


if __name__ == "__main__":
    main()
