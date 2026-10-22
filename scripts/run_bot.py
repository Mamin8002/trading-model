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
from btc_bot.model.predictor import ModelMismatchError  # noqa: E402
from btc_bot.portfolio.advisor import PositionAdvisor  # noqa: E402
from btc_bot.portfolio.journal import Journal, JournalError  # noqa: E402
from btc_bot.portfolio.paper import PaperTrader  # noqa: E402
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
    p.add_argument("--threshold", type=float, default=None,
                   help="переопределить порог уверенности (по умолчанию - из model card)")
    p.add_argument("--journal", default=None,
                   help="путь журнала сигналов/позиций (по умолчанию: LIVE - journal.path из конфига, "
                        "TEST - data/journal/journal_test.json, чтобы демо-сигналы не смешивались "
                        "с боевой историей)")
    p.add_argument("--paper", action="store_true",
                   help="TEST: включить paper trading в реплее (в LIVE берётся из paper.enabled)")
    p.add_argument("--no-paper", action="store_true", help="отключить paper trading")
    p.add_argument("--tp", type=float, default=None, help="paper: take-profit %% (переопределяет конфиг)")
    p.add_argument("--sl", type=float, default=None, help="paper: stop-loss %% (переопределяет конфиг)")
    p.add_argument("--max-hold", type=int, default=None, help="paper: макс. баров удержания (переопределяет конфиг)")
    return p.parse_args()


def find_replay_file(cfg) -> Optional[Path]:
    """Самый свежий ПОЛНЫЙ clean-файл ПО ДАТЕ ЗАГРУЗКИ (частичные _PARTIAL пропускаются)."""
    from btc_bot.data.finders import find_latest_clean_csv
    return find_latest_clean_csv(cfg.resolve(cfg.data.processed_dir))


def main() -> int:
    args = parse_args()
    overrides = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.model_dir:
        overrides.setdefault("model", {})["artifact_dir"] = args.model_dir
    if args.threshold is not None:
        overrides.setdefault("model", {})["confidence_threshold"] = args.threshold
    cfg = load_config(args.config, overrides=overrides or None)

    try:
        predictor, model_dir = load_predictor(cfg)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    except ModelMismatchError as e:
        log.error("%s", e)
        log.error("Конфиг изменён (например, horizon_bars) - выполните: python scripts\\train_model.py")
        return 2
    log.info("Модель: %s (каталог %s)", predictor.model_version, model_dir)
    log.info("Инструмент: %s/%s interval=%s, horizon=%d бар(ов) = %d мин, порог уверенности=%.3f (%s)",
             cfg.exchange.category, cfg.exchange.symbol, cfg.exchange.interval,
             cfg.model.horizon_bars, cfg.horizon_minutes, predictor.confidence_threshold,
             "из конфига" if cfg.model.confidence_threshold is not None else "из model card")

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

    # paper trading (Этап 5b): виртуальные позиции, без ордеров и приватного API
    paper = None
    paper_enabled = cfg.paper.enabled and not args.no_paper
    if cfg.mode == "TEST" and not args.paper:
        paper_enabled = False  # в TEST-реплее paper включается явным флагом --paper
    if paper_enabled:
        pr = dict(default_rules)  # копия, чтобы CLI-переопределения не меняли конфиг молча
        if args.tp is not None:
            pr["take_profit_pct"] = args.tp
        if args.sl is not None:
            pr["stop_loss_pct"] = args.sl
        if args.max_hold is not None:
            pr["max_hold_bars"] = args.max_hold
        paper = PaperTrader(
            journal,
            symbol=cfg.exchange.symbol, category=cfg.exchange.category,
            fee_pct=cfg.trading.fee_pct, slippage_pct=cfg.trading.slippage_pct,
            order_size_quote=cfg.trading.order_size_quote,
            exit_rules=pr,
            entry_min_confidence=cfg.paper.entry_min_confidence,
            cooldown_bars=cfg.paper.cooldown_bars,
            max_open_positions=cfg.paper.max_open_positions,
            intrabar_exits=cfg.paper.intrabar_exits,
            exit_at_horizon=cfg.paper.exit_at_horizon,
            partial_tp={"enabled": cfg.paper.partial_tp.enabled,
                        "at_pct": cfg.paper.partial_tp.at_pct,
                        "fraction": cfg.paper.partial_tp.fraction},
            interval_step_ms=step_ms_approx(cfg.exchange.interval),
            horizon_bars=cfg.model.horizon_bars,
        )
        log.info("Paper trading ВКЛЮЧЕН (виртуально, без ордеров): баланс %.2f USDT, "
                 "сделка %.2f USDT, fee %.2f%%, slip %.3f%%, правила выхода: TP=%s SL=%s max_hold=%s, "
                 "exit_at_horizon=%s (H=%d)",
                 cfg.paper.initial_balance_quote, cfg.trading.order_size_quote,
                 cfg.trading.fee_pct, cfg.trading.slippage_pct,
                 pr.get("take_profit_pct"), pr.get("stop_loss_pct"), pr.get("max_hold_bars"),
                 cfg.paper.exit_at_horizon, cfg.model.horizon_bars)
        if not paper._rules_defined():
            log.info("Paper: правила выхода не заданы и exit_at_horizon=false -> входы будут воздерживаться.")

    bot = MonitorBot(cfg, predictor, ui, rest_source=rest_source, journal=journal,
                     advisor=advisor, paper=paper)

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
