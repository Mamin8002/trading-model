"""
portfolio.py — Учёт капитала, позиций, журнал сделок и метрики эффективности.

Отслеживает:
  * initial_balance / current balance (cash) / equity (cash + open PnL);
  * открытую позицию (сторона, объём, цены входа/SL/TP, время входа);
  * закрытые сделки (Trade Journal в CSV: время входа/выхода, цены, PnL RUB/%, причина выхода);
  * агрегированные метрики: total PnL, win rate, profit factor, max drawdown, avg R.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional

log = logging.getLogger("portfolio")


@dataclass
class OpenPosition:
    trade_id: int
    side: str                # LONG | SHORT
    qty: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    entry_bar_index: int     # для таймаута по барам
    risk_amount: float


@dataclass
class ClosedTrade:
    trade_id: int
    symbol: str
    side: str
    qty: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl_rub: float
    pnl_pct: float           # % от депозита на момент входа
    r_multiple: float        # PnL / плановый риск
    commission_rub: float
    exit_reason: str         # TP | SL | TIMEOUT | MANUAL | EOD
    confidence: float = 0.0
    mode: str = ""


JOURNAL_FIELDS = list(asdict(ClosedTrade(0, "", "", 0, "", "", 0, 0, 0, 0, 0, 0, "", 0, "")).keys())


class Portfolio:
    def __init__(self, initial_balance: float, symbol: str, journal_csv: str,
                 commission_pct: float = 0.05):
        self.initial_balance = float(initial_balance)
        self.cash = float(initial_balance)
        self.symbol = symbol
        self.journal_csv = journal_csv
        # комиссия за всю сделку (вход+выход), в долях (0.05% -> 0.0005)
        self.commission_frac = commission_pct / 100.0
        self.position: Optional[OpenPosition] = None
        self.closed: list[ClosedTrade] = []
        self._next_id = 1
        self._peak_equity = self.cash
        self.max_drawdown_pct = 0.0
        self._ensure_journal_file()

    # ----------------------------------------------------------- helpers
    def _ensure_journal_file(self):
        os.makedirs(os.path.dirname(self.journal_csv) or ".", exist_ok=True)
        if not os.path.exists(self.journal_csv):
            with open(self.journal_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(JOURNAL_FIELDS)

    @property
    def has_position(self) -> bool:
        return self.position is not None

    def unrealized_pnl(self, mark_price: float) -> float:
        if not self.position:
            return 0.0
        p = self.position
        sign = 1.0 if p.side == "LONG" else -1.0
        return sign * (mark_price - p.entry_price) * p.qty

    def equity(self, mark_price: float) -> float:
        return self.cash + self.unrealized_pnl(mark_price)

    # ------------------------------------------------------------ actions
    def open_position(self, side: str, qty: int, entry_price: float,
                      stop: float, target: float, now: datetime,
                      bar_index: int, risk_amount: float, ) -> OpenPosition:
        assert not self.has_position, "Позиция уже открыта"
        cost = qty * entry_price
        required = cost  # LONG и SHORT (маржа = notional, упрощённая модель быка)
        if required > self.cash:
            log.warning("Недостаточно кэша (%.2f < %.2f) — позиция урезается", self.cash, required)
            qty = int(self.cash // entry_price)
            if qty <= 0:
                raise RuntimeError("Кэш исчерпан, открыть позицию невозможно")
        self.position = OpenPosition(
            trade_id=self._next_id, side=side, qty=qty, entry_price=entry_price,
            stop_loss=stop, take_profit=target, entry_time=now.isoformat(sep=" "),
            entry_bar_index=bar_index, risk_amount=risk_amount,
        )
        self._next_id += 1
        log.info("OPEN %s %s x%d @ %.4f | SL %.4f | TP %.4f | риск %.2f RUB",
                 side, self.symbol, qty, entry_price, stop, target, risk_amount)
        return self.position

    def close_position(self, exit_price: float, now: datetime, reason: str,
                        confidence: float = 0.0, mode: str = "") -> ClosedTrade:
        assert self.has_position, "Нет открытой позиции"
        p = self.position
        sign = 1.0 if p.side == "LONG" else -1.0
        gross = sign * (exit_price - p.entry_price) * p.qty
        notional_in = p.qty * p.entry_price
        notional_out = p.qty * exit_price
        commission = (notional_in + notional_out) * self.commission_frac
        pnl_net = gross - commission
        equity_before = self.cash
        self.cash += pnl_net
        pnl_pct = pnl_net / equity_before * 100.0 if equity_before > 0 else 0.0
        r_mult = pnl_net / p.risk_amount if p.risk_amount > 0 else 0.0
        trade = ClosedTrade(
            trade_id=p.trade_id, symbol=self.symbol, side=p.side, qty=p.qty,
            entry_time=p.entry_time, exit_time=now.isoformat(sep=" "),
            entry_price=round(p.entry_price, 6), exit_price=round(exit_price, 6),
            pnl_rub=round(pnl_net, 4), pnl_pct=round(pnl_pct, 4),
            r_multiple=round(r_mult, 3), commission_rub=round(commission, 4),
            exit_reason=reason, confidence=round(confidence, 4), mode=mode,
        )
        self.closed.append(trade)
        self._append_journal(trade)
        self.position = None
        log.info("CLOSE %s %s x%d @ %.4f -> %.4f | PnL %.2f RUB (%.3f%%) | R=%.2f | %s",
                 trade.side, trade.symbol, trade.qty, trade.entry_price, exit_price,
                 pnl_net, pnl_pct, r_mult, reason)
        # drawdown tracking
        eq = self.cash
        self._peak_equity = max(self._peak_equity, eq)
        dd = (self._peak_equity - eq) / self._peak_equity * 100 if self._peak_equity > 0 else 0
        self.max_drawdown_pct = max(self.max_drawdown_pct, dd)
        return trade

    def _append_journal(self, t: ClosedTrade):
        with open(self.journal_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            w.writerow(asdict(t))

    # ------------------------------------------------------------ metrics
    def metrics(self) -> dict:
        n = len(self.closed)
        if n == 0:
            return {"trades": 0, "total_pnl": round(self.cash - self.initial_balance, 2),
                    "win_rate": 0.0, "profit_factor": 0.0, "avg_r": 0.0,
                    "max_drawdown_pct": round(self.max_drawdown_pct, 2),
                    "return_pct": 0.0}
        pnls = [t.pnl_rub for t in self.closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
        return {
            "trades": n,
            "total_pnl": round(sum(pnls), 2),
            "return_pct": round((self.cash / self.initial_balance - 1) * 100, 2),
            "win_rate": round(len(wins) / n * 100, 2),
            "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
            "avg_r": round(sum(t.r_multiple for t in self.closed) / n, 3),
            "best_trade": round(max(pnls), 2),
            "worst_trade": round(min(pnls), 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "balance": round(self.cash, 2),
        }

    def summary_line(self, mark_price: Optional[float] = None) -> str:
        m = self.metrics()
        upnl = self.unrealized_pnl(mark_price) if (mark_price and self.position) else 0.0
        return (f"Balance={self.cash:,.2f} RUB | Equity={self.cash + upnl:,.2f} | "
                f"Trades={m['trades']} | WinRate={m['win_rate']}% | "
                f"PnL={m['total_pnl']:,.2f} ({m['return_pct']}%) | MaxDD={m['max_drawdown_pct']}%")
