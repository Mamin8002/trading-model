#!/usr/bin/env python3
r"""Этап 3: запуск бота мониторинга BTC (публичные данные Bybit).

Режимы:
  TEST - офлайн-реплей сохранённых данных (без сети) + демо-сценарий STALE;
  LIVE - публичный WebSocket Bybit в реальном времени (+ seed/backfill истории
         через публичный REST). Только чтение: никаких ордеров и приватных API.

Примеры (PowerShell, из корня проекта):
    python scripts\run_bot.py --mode TEST --replay-bars 40 --speed 4
    python scripts\run_bot.py --mode LIVE                 # до Ctrl+C
    python scripts\run_bot.py --mode LIVE --duration 660  # 11 минут и выход
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from btc_bot.bot import MonitorBot, load_predictor  # noqa: E402
from btc_bot.config import load_config  # noqa: E402
from btc_bot.data_sources.bybit import BybitSource, step_ms_approx  # noqa: E402
from btc_bot.portfolio.advisor import PositionAdvisor  # noqa: E402
from btc_bot.portfolio.journal import Journal, JournalError  # noqa: E402
from btc_bot.ui.console import ConsoleUI  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_bot")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC monitor bot (Bybit public data, read-only).")
    p.add_argument("--config", default=None)
    p.add_argument("--mode", choices=["TEST", "LIVE"], default=None,
                   help="по умолчанию - mode из конфига")
    p.add_argument("--replay-file", default=None,
                   help="TEST: clean CSV (по умолчанию - последний в data/processed/)")
    p.add_argument("--replay-bars", type=int, default=40)
    p.add_argument("--speed", type=float, default=4.0, help="TEST: баров в секунду (0 - мгновенно)")
    p.add_argument("--no-demo-stale", action="store_true", help="TEST: без демо-сценария STALE")
    p.add_argument("--duration", type=float, default=0.0, help="LIVE: секунды (0 - до Ctrl+C)")
    p.add_argument("--line-ui", "--no-ansi", dest="line_ui", action="store_true",
                   help="то же, что --ui line (одна обновляемая строка + события)")
    p.add_argument("--ui", choices=["auto", "dash", "line", "block"], default=None,
                   help="режим интерфейса (по умолчанию ui.mode из конфига = auto)")
    p.add_argument("--model-dir", default=None, help="переопределить каталог артефакта модели")
    p.add_argument("--journal", default=None,
                   help="путь журнала сигналов/позиций (по умолчанию: LIVE - journal.path из конфига, "
                        "TEST - data/journal/journal_test.json, чтобы демо-сигналы не смешивались "
                        "с боевой историей)")
    return p.parse_args()


def find_replay_file(cfg) -> Path | None:
    files = sorted(cfg.resolve(cfg.data.processed_dir).glob("*.clean.csv"))
    return files[-1] if files else None


def main() -> int:
    args = parse_args()
    overrides = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.model_dir:
        overrides.setdefault("model", {})["artifact_dir"] = args.model_dir
    cfg = load_config(args.config, overrides=overrides or None)

    try:
        predictor, model_dir = load_predictor(cfg)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    log.info("Модель: %s (каталог %s)", predictor.model_version, model_dir)
    log.info("Инструмент: %s/%s interval=%s, horizon=%d бар(ов) = %d мин, порог уверенности=%.2f",
             cfg.exchange.category, cfg.exchange.symbol, cfg.exchange.interval,
             cfg.model.horizon_bars, cfg.horizon_minutes, cfg.model.confidence_threshold)

    ui_mode = "line" if args.line_ui else (args.ui or cfg.ui.mode)
    ui = ConsoleUI(mode=ui_mode, refresh_s=cfg.ui.refresh_s,
                   log_events=cfg.ui.log_events, colors=cfg.ui.colors)
    log.info("Интерфейс: режим %s (цвета: %s)", ui.mode, "вкл" if ui.colors else "выкл")
    rest_source = None
    if cfg.mode == "LIVE":
        rest_source = BybitSource(
            base_url=cfg.exchange.rest_base_url,
            request_timeout_s=cfg.exchange.request_timeout_s,
            min_request_interval_s=cfg.exchange.min_request_interval_s,
            max_retries=cfg.exchange.max_retries,
        )

    # журнал сигналов/позиций + советник (Этап 4)
    if args.journal:
        journal_path = Path(args.journal)
    elif cfg.mode == "TEST":
        journal_path = cfg.resolve("data/journal/journal_test.json")
    else:
        journal_path = cfg.resolve(cfg.journal.path)
    try:
        journal = Journal(journal_path)
    except JournalError as e:
        log.error("%s", e)
        return 2
    er = cfg.positions.exit_rules
    default_rules = {
        "take_profit_pct": er.take_profit_pct,
        "stop_loss_pct": er.stop_loss_pct,
        "max_hold_bars": er.max_hold_bars,
    }
    advisor = PositionAdvisor(journal, default_exit_rules=default_rules,
                              interval_step_ms=step_ms_approx(cfg.exchange.interval))
    log.info("Журнал: %s", journal_path)
    if not any(v is not None for v in default_rules.values()):
        log.info("Условия выхода в конфиге НЕ заданы: рекомендации на вход будут воздерживаться "
                 "(задайте positions.exit_rules или --tp/--sl/--max-hold при регистрации позиции).")

    bot = MonitorBot(cfg, predictor, ui, rest_source=rest_source, journal=journal, advisor=advisor)

    if cfg.mode == "TEST":
        replay = Path(args.replay_file) if args.replay_file else find_replay_file(cfg)
        if replay is None or not replay.exists():
            log.error("Файл для реплея не найден. Сначала: python scripts/download_history.py "
                      "(или укажите --replay-file).")
            return 2
        log.info("TEST-реплей: %s (последние %d баров, speed=%.1f бар/с)", replay.name,
                 args.replay_bars, args.speed)
        return bot.run_test_replay(replay, n_bars=args.replay_bars, speed_bps=args.speed,
                                   demo_stale=not args.no_demo_stale)
    else:
        log.info("LIVE: подключаюсь к %s (Ctrl+C для остановки)%s",
                 cfg.exchange.ws_public_url,
                 f", длительность {args.duration:.0f} с" if args.duration else "")
        return bot.run_live(duration_s=args.duration)


if __name__ == "__main__":
    sys.exit(main())
