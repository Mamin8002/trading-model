#!/usr/bin/env python3
r"""Этап 4: ручной учёт фактических позиций (полностью офлайн, без биржевых вызовов).

Бот НЕ имеет приватного API и не знает, исполнили ли вы сигнал. Позиции
регистрируются вручную. Для spot BTCUSDT поддерживается только LONG:
SELL-сигнал означает "закрыть/уменьшить длинную позицию", шорт не открывается.

Команды (PowerShell, из корня проекта):
  python scripts\position_cli.py register-entry --price 84500 --quantity 0.01 \
      [--time "2026-10-22 10:15" | --time-ms 1792000000000] [--signal-id sig-000123] \
      [--direction LONG] [--tp 1.5] [--sl 1.0] [--max-hold 12] [--note "..."]
  python scripts\position_cli.py close --position-id pos-000001 --price 85000 \
      [--quantity 0.004] [--time "..."] [--note "..."]          # без --quantity = полное закрытие
  python scripts\position_cli.py list [--all]
  python scripts\position_cli.py show --position-id pos-000001
  python scripts\position_cli.py signals [--last 20]
  python scripts\position_cli.py stats
  python scripts\position_cli.py events [--last 20]

Общий флаг: --journal PATH (по умолчанию journal.path из конфига), --config PATH.
Время --time трактуется как UTC ("YYYY-MM-DD[ HH:MM[:SS]]").
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from btc_bot.config import load_config  # noqa: E402
from btc_bot.portfolio.journal import Journal, JournalError  # noqa: E402


def parse_time_ms(args) -> int | None:
    if getattr(args, "time_ms", None) is not None:
        return int(args.time_ms)
    t = getattr(args, "time", None)
    if not t:
        return None
    try:
        ts = pd.Timestamp(t, tz="UTC") if " " in t or "T" in t else pd.Timestamp(t + " 00:00:00", tz="UTC")
        return int(ts.timestamp() * 1000)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Не удалось разобрать время {t!r} (ожидалось UTC 'YYYY-MM-DD[ HH:MM[:SS]]'): {e}")


def utc(ms) -> str:
    if ms is None:
        return "-"
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ручной учёт позиций (офлайн).")
    p.add_argument("--config", default=None)
    p.add_argument("--journal", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register-entry", help="зарегистрировать фактический вход")
    r.add_argument("--price", type=float, required=True)
    r.add_argument("--quantity", type=float, required=True)
    r.add_argument("--time", default=None, help="UTC 'YYYY-MM-DD[ HH:MM[:SS]]' (по умолчанию сейчас)")
    r.add_argument("--time-ms", type=int, default=None)
    r.add_argument("--direction", default="LONG", choices=["LONG"])
    r.add_argument("--signal-id", default=None, help="связать с сигналом из журнала")
    r.add_argument("--tp", type=float, default=None, help="take-profit, %% от цены входа")
    r.add_argument("--sl", type=float, default=None, help="stop-loss, %% от цены входа")
    r.add_argument("--max-hold", type=int, default=None, help="макс. число баров удержания")
    r.add_argument("--note", default="")

    c = sub.add_parser("close", help="полное или частичное закрытие")
    c.add_argument("--position-id", required=True)
    c.add_argument("--price", type=float, required=True)
    c.add_argument("--quantity", type=float, default=None, help="по умолчанию - весь остаток")
    c.add_argument("--time", default=None)
    c.add_argument("--time-ms", type=int, default=None)
    c.add_argument("--note", default="")

    l = sub.add_parser("list", help="список позиций")
    l.add_argument("--all", action="store_true", help="включая закрытые")

    s = sub.add_parser("show", help="позиция полностью")
    s.add_argument("--position-id", required=True)

    g = sub.add_parser("signals", help="история сигналов")
    g.add_argument("--last", type=int, default=20)

    sub.add_parser("stats", help="статистика журнала")

    e = sub.add_parser("events", help="журнал событий")
    e.add_argument("--last", type=int, default=20)

    sub.add_parser("paper-stats", help="статистика paper trading (виртуальные сделки)")

    pl = sub.add_parser("paper-list", help="виртуальные позиции")
    pl.add_argument("--all", action="store_true", help="включая закрытые")

    ph = sub.add_parser("paper-history", help="лог действий paper trading")
    ph.add_argument("--last", type=int, default=30)
    return p


def cmd_register_entry(journal: Journal, cfg, args) -> int:
    rules = {}
    if args.tp is not None:
        rules["take_profit_pct"] = args.tp
    if args.sl is not None:
        rules["stop_loss_pct"] = args.sl
    if args.max_hold is not None:
        rules["max_hold_bars"] = args.max_hold
    pos = journal.register_entry(
        symbol=cfg.exchange.symbol, category=cfg.exchange.category,
        direction=args.direction, price=args.price, quantity=args.quantity,
        opened_at_ms=parse_time_ms(args), signal_id=args.signal_id,
        exit_rules=rules or None, note=args.note,
        allow_short=cfg.positions.allow_short,
    )
    src = pos.data.get("source_signal")
    print(f"Зарегистрирован вход: {pos.position_id} {pos.data['direction']} {pos.data['symbol']} "
          f"{pos.data['quantity_initial']:g} @ {pos.entry_price:,.2f} ({utc(pos.data['opened_at_ms'])} UTC)")
    if src:
        print(f"Связана с сигналом: {src['signal_id']} ({src['action']} conf {src['confidence']:.3f}, "
              f"выдан {utc(src['signal_created_at_ms'])}, горизонт {src['horizon_minutes']} мин, "
              f"модель {src['model_version']})")
        print("Сигнал отмечен как ПОДТВЕРЖДЁННЫЙ входом (до этого он исполненным не считался).")
    else:
        print("Без привязки к сигналу (ручная регистрация).")
    print(f"Условия выхода: {rules if rules else 'не заданы для позиции (будут взяты из конфига, если есть)'}")
    return 0


def cmd_close(journal: Journal, args) -> int:
    pos = journal.close_position(args.position_id, price=args.price, quantity=args.quantity,
                                 closed_at_ms=parse_time_ms(args), note=args.note)
    d = pos.data
    if d["status"] == "CLOSED":
        print(f"Позиция {pos.position_id} ЗАКРЫТА ПОЛНОСТЬЮ ({utc(d['closed_at_ms'])} UTC)")
        print(f"Средняя цена выхода: {d['avg_exit_price']:,.2f}")
        print(f"Реализованный результат: {d['realized_pnl_quote']:+.4f} USDT "
              f"({d['realized_pnl_pct']:+.3f}%) -> {d['outcome']}")
    else:
        print(f"Частичное закрытие {pos.position_id}: {args.quantity:g} @ {args.price:,.2f}; "
              f"остаток {pos.quantity_open:g} из {d['quantity_initial']:g}; статус OPEN")
    return 0


def cmd_list(journal: Journal, args) -> int:
    positions = journal.positions() if args.all else journal.positions("OPEN")
    if not positions:
        print("Позиций нет." if args.all else "Открытых позиций нет (фактическое состояние портфеля неизвестно).")
        return 0
    print(f"{'ID':<12} {'СТАТУС':<8} {'НАПР':<6} {'СИМВОЛ':<9} {'ВХОД':>12} {'ОТКРЫТО/ВСЕГО':>18} "
          f"{'ВЫХОД_СР':>12} {'PNL USDT':>10}  ОТКРЫТА (UTC)")
    for p in positions:
        d = p.data
        avg = f"{d['avg_exit_price']:,.2f}" if d.get("avg_exit_price") is not None else "-"
        pnl = f"{d['realized_pnl_quote']:+.4f}" if d.get("realized_pnl_quote") is not None else "-"
        print(f"{p.position_id:<12} {d['status']:<8} {d['direction']:<6} {d['symbol']:<9} "
              f"{p.entry_price:>12,.2f} {p.quantity_open:>8g}/{d['quantity_initial']:<9g} "
              f"{avg:>12} {pnl:>10}  {utc(d['opened_at_ms'])}")
    return 0


def cmd_show(journal: Journal, args) -> int:
    pos = journal.get_position(args.position_id)
    if pos is None:
        print(f"Позиция {args.position_id!r} не найдена")
        return 2
    print(json.dumps(pos.data, indent=2, ensure_ascii=False))
    return 0


def cmd_signals(journal: Journal, args) -> int:
    sigs = journal.last_signals(args.last)
    if not sigs:
        print("Сигналов пока нет.")
        return 0
    print(f"{'ID':<12} {'ВЫДАН (UTC)':<20} {'ACT':<5} {'CONF':>6} {'ЦЕНА':>12} {'FRESH':<6} "
          f"{'ВХОД ПОДТВ.':<12} ПОЗИЦИЯ")
    for s in sigs:
        print(f"{s.get('signal_id', '?'):<12} {utc(s.get('created_at_ms')):<20} "
              f"{s.get('action', '?'):<5} {s.get('confidence', 0):>6.3f} "
              f"{s.get('price_at_signal', 0):>12,.2f} {str(s.get('data_fresh')):<6} "
              f"{str(s.get('entry_confirmed')):<12} {s.get('position_id') or '-'}")
    return 0


def cmd_stats(journal: Journal) -> int:
    st = journal.stats()
    print("Статистика журнала:")
    for k, v in st.items():
        print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Напоминание: PnL считается только по РУЧНОЙ регистрации сделок; результат past-позиций "
          "не используется для автопереобучения модели.")
    return 0


def cmd_events(journal: Journal, args) -> int:
    for ev in journal.events(args.last):
        details = {k: v for k, v in ev.items() if k not in ("ts_ms", "ts_utc", "kind")}
        brief = json.dumps(details, ensure_ascii=False, default=str)
        if len(brief) > 160:
            brief = brief[:157] + "..."
        print(f"{ev['ts_utc']}  {ev['kind']:<26} {brief}")
    return 0


def cmd_paper_stats(journal: Journal, cfg) -> int:
    st = journal.paper_stats(cfg.paper.initial_balance_quote)
    print("Paper trading (ВИРТУАЛЬНЫЕ сделки; ордера не отправлялись):")
    for k, v in st.items():
        print(f"  {k}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Издержки: fee "
          f"{cfg.trading.fee_pct}%/сторона, slippage {cfg.trading.slippage_pct}%/сторона, "
          f"объём {cfg.trading.order_size_quote} USDT/сделку")
    return 0


def cmd_paper_list(journal: Journal, args) -> int:
    poss = journal.paper_positions() if args.all else journal.paper_positions("OPEN")
    if not poss:
        print("Виртуальных позиций нет." if args.all else "Открытых виртуальных позиций нет.")
        return 0
    for p in poss:
        rs = p.get("rules_snapshot") or {}
        src = p.get("source_signal") or {}
        print(f"{p['position_id']} [{p['status']}] LONG {p['symbol']}: вход {p['entry_price']:,.2f} "
              f"({p.get('opened_at_utc')}), qty {p['quantity_open']:.8f}/{p['quantity_initial']:.8f}, "
              f"realized {p.get('realized_pnl_quote') or 0:+.4f} USDT, outcome {p.get('outcome') or '-'}")
        print(f"  правила(снимок): TP={rs.get('take_profit_pct')} SL={rs.get('stop_loss_pct')} "
              f"max_hold={rs.get('max_hold_bars')} exit_at_horizon={rs.get('exit_at_horizon')} "
              f"(H={rs.get('horizon_bars')}) fee={rs.get('fee_pct')}% slip={rs.get('slippage_pct')}%")
        if src:
            print(f"  источник: сигнал {src.get('signal_id')} ({src.get('action')} "
                  f"conf {src.get('confidence'):.3f}), модель {src.get('model_version')}")
    return 0


def cmd_paper_history(journal: Journal, args) -> int:
    acts = journal.paper_actions(args.last)
    if not acts:
        print("Действий paper trading пока нет.")
        return 0
    for a in acts:
        print(f"{a['ts_utc']}  {a['kind']:<14} pos={a.get('position_id') or '-':<14} "
              f"px={a.get('price') if a.get('price') is None else format(a['price'], ',.2f')} "
              f"pnl={a.get('pnl_quote') if a.get('pnl_quote') is None else format(a['pnl_quote'], '+.4f')} "
              f"| {a.get('reason', '')[:120]}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    journal_path = Path(args.journal) if args.journal else cfg.resolve(cfg.journal.path)
    try:
        journal = Journal(journal_path)
    except JournalError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    try:
        if args.cmd == "register-entry":
            return cmd_register_entry(journal, cfg, args)
        if args.cmd == "close":
            return cmd_close(journal, args)
        if args.cmd == "list":
            return cmd_list(journal, args)
        if args.cmd == "show":
            return cmd_show(journal, args)
        if args.cmd == "signals":
            return cmd_signals(journal, args)
        if args.cmd == "stats":
            return cmd_stats(journal)
        if args.cmd == "events":
            return cmd_events(journal, args)
        if args.cmd == "paper-stats":
            return cmd_paper_stats(journal, cfg)
        if args.cmd == "paper-list":
            return cmd_paper_list(journal, args)
        if args.cmd == "paper-history":
            return cmd_paper_history(journal, args)
    except JournalError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
