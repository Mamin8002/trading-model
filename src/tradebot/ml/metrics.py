"""Метрики ML и упрощённая торговая оценка (docs/ml.md §5).

Торговая оценка фазы 3 — ДИАГНОСТИЧЕСКАЯ (полный симулятор — этап B1 roadmap):
- позиция определяется сигналом на закрытии бара t и действует один бар:
  вход по open[t+1], выход по open[t+2] (architecture §6.6, decisions D5:
  исполнение не раньше открытия следующего бара);
- издержки: (commission_bps + slippage_bps) на единицу оборота (|Δпозиции|),
  включая ликвидацию позиции в конце сегмента;
- внутри бара (high/low) исполнение НЕ моделируется — это заведомо упрощение,
  оно зафиксировано в допущениях отчёта;
- сегменты валидационных фолдов симулируются независимо (между фолдами — разрывы
  данных, позиция через разрыв не переносится; стартовая позиция сегмента = 0).

Годовой пересчёт: bars_per_year берётся из конфига (evaluation.bars_per_year) либо
выводится из данных как медиана баров на торговый день × trading_days_per_year;
источник и значения всегда печатаются в допущениях отчёта.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "TradingAssumptions",
    "TradingStats",
    "aggregate_trading",
    "annualized_cagr",
    "classification_metrics",
    "derive_bars_per_year",
    "positions_from_proba",
    "positions_from_regression",
    "regression_metrics",
    "sharpe_ratio",
    "simulate_positions",
]


@dataclass(frozen=True)
class ClassificationMetrics:
    n: int
    accuracy: float
    balanced_accuracy: float | None
    roc_auc: float | None  # None, если в окне один класс
    log_loss: float | None
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    single_class: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "roc_auc": self.roc_auc,
            "log_loss": self.log_loss,
            "threshold": self.threshold,
            "confusion": {"tn": self.tn, "fp": self.fp, "fn": self.fn, "tp": self.tp},
            "single_class": self.single_class,
            "note": self.note,
        }


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    mae: float
    rmse: float
    r2: float | None  # None при нулевой дисперсии y

    def as_dict(self) -> dict:
        return {"n": self.n, "mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> ClassificationMetrics:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int64)
    proba = np.asarray(proba, dtype=np.float64)
    pred = (proba >= threshold).astype(np.int64)
    classes = np.unique(y_true)
    single_class = len(classes) <= 1
    auc = None if single_class else float(roc_auc_score(y_true, proba))
    ll = None if single_class else float(log_loss(y_true, proba, labels=[0, 1]))
    bal = None if single_class else float(balanced_accuracy_score(y_true, pred))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    note = "один класс в окне: AUC/log-loss/balanced не определены" if single_class else ""
    return ClassificationMetrics(
        n=len(y_true),
        accuracy=float(accuracy_score(y_true, pred)),
        balanced_accuracy=bal,
        roc_auc=auc,
        log_loss=ll,
        threshold=threshold,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        single_class=single_class,
        note=note,
    )


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err**2)))
    var = float(np.var(y_true))
    r2 = float(1.0 - np.mean(err**2) / var) if var > 0 else None
    return RegressionMetrics(n=len(y_true), mae=mae, rmse=rmse, r2=r2)


def positions_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Длинная позиция (1), если P(up) >= threshold; иначе флэт (0). Long-only."""
    return (np.asarray(proba, dtype=np.float64) >= threshold).astype(np.float64)


def positions_from_regression(pred: np.ndarray) -> np.ndarray:
    """Для регрессии: позиция = знак прогноза (long при pred > 0, иначе флэт)."""
    return (np.asarray(pred, dtype=np.float64) > 0.0).astype(np.float64)


@dataclass(frozen=True)
class TradingAssumptions:
    commission_bps: float
    slippage_bps: float
    bars_per_year: float
    bars_per_day_median: float | None  # None, если bars_per_year задан конфигом
    trading_days_per_year: int
    risk_free_annual: float
    execution: str = (
        "сигнал на закрытии бара t; вход по open[t+1]; выход по open[t+2]; "
        "внутрибарное исполнение не моделируется; издержки на единицу оборота; "
        "сегменты фолдов независимы (позиция через разрыв не переносится)"
    )
    bars_per_year_source: str = "derived"  # derived | config

    def as_dict(self) -> dict:
        return {
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "bars_per_year": self.bars_per_year,
            "bars_per_day_median": self.bars_per_day_median,
            "trading_days_per_year": self.trading_days_per_year,
            "risk_free_annual": self.risk_free_annual,
            "execution": self.execution,
            "bars_per_year_source": self.bars_per_year_source,
        }


@dataclass
class TradingStats:
    n_bars: int = 0
    n_trades: int = 0
    bars_in_market: int = 0
    exposure: float = 0.0
    gross_return: float = 0.0        # суммарная валовая доходность (сумма по барам)
    cost_return: float = 0.0         # суммарные издержки (положительное число)
    net_return: float = 0.0          # compounded: prod(1+net_bar) - 1
    max_drawdown: float = 0.0        # >= 0
    turnover: float = 0.0            # суммарный |Δпозиции| (включая ликвидацию)
    per_bar_net: np.ndarray = field(default_factory=lambda: np.empty(0))
    trade_returns: tuple[float, ...] = ()

    @property
    def hit_rate(self) -> float | None:
        if not self.trade_returns:
            return None
        return float(np.mean([r > 0 for r in self.trade_returns]))

    @property
    def profit_factor(self) -> float | None:
        gains = [r for r in self.trade_returns if r > 0]
        losses = [r for r in self.trade_returns if r < 0]
        if not gains or not losses:
            return None
        return float(sum(gains) / abs(sum(losses)))

    def as_dict(self) -> dict:
        return {
            "n_bars": self.n_bars,
            "n_trades": self.n_trades,
            "bars_in_market": self.bars_in_market,
            "exposure": self.exposure,
            "gross_return": self.gross_return,
            "cost_return": self.cost_return,
            "net_return": self.net_return,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
        }


def simulate_positions(
    open_t1: np.ndarray,
    open_t2: np.ndarray,
    positions: np.ndarray,
    *,
    commission_bps: float,
    slippage_bps: float,
) -> TradingStats:
    """Оценочная симуляция одного непрерывного сегмента (см. допущения модуля)."""
    open_t1 = np.asarray(open_t1, dtype=np.float64)
    open_t2 = np.asarray(open_t2, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    if not (len(open_t1) == len(open_t2) == n):
        raise ValueError("simulate_positions: длины массивов должны совпадать")
    if n == 0:
        return TradingStats()

    cost_rate = (commission_bps + slippage_bps) / 10_000.0
    gross = positions * (open_t2 / open_t1 - 1.0)
    prev_pos = np.concatenate([[0.0], positions[:-1]])
    delta = np.abs(positions - prev_pos)
    # ликвидация позиции в конце сегмента (консервативно: всегда закрываемся)
    delta = delta.copy()
    delta[-1] += positions[-1]
    costs = delta * cost_rate
    net = gross - costs

    equity = np.cumprod(1.0 + net)
    running_max = np.maximum.accumulate(equity)
    drawdown = float(np.max(1.0 - equity / running_max)) if n else 0.0

    # сделки: непрерывные отрезки позиций == 1
    trade_returns: list[float] = []
    in_trade = False
    trade_equity = 1.0
    for t in range(n):
        if positions[t] == 1.0 and not in_trade:
            in_trade = True
            trade_equity = 1.0 + net[t]
        elif positions[t] == 1.0 and in_trade:
            trade_equity *= 1.0 + net[t]
        elif positions[t] == 0.0 and in_trade:
            trade_equity *= 1.0 - costs[t]  # издержки выхода на баре закрытия
            trade_returns.append(trade_equity - 1.0)
            in_trade = False
    if in_trade:  # позиция не закрыта до конца сегмента — учитываем принудительный выход
        trade_equity *= 1.0 - cost_rate
        trade_returns.append(trade_equity - 1.0)

    return TradingStats(
        n_bars=n,
        n_trades=len(trade_returns),
        bars_in_market=int(np.sum(positions == 1.0)),
        exposure=float(np.mean(positions == 1.0)),
        gross_return=float(np.sum(gross)),
        cost_return=float(np.sum(costs)),
        net_return=float(equity[-1] - 1.0),
        max_drawdown=drawdown,
        turnover=float(np.sum(delta)),
        per_bar_net=net,
        trade_returns=tuple(trade_returns),
    )


def aggregate_trading(segments: list[TradingStats]) -> TradingStats:
    """Объединяет сегменты фолдов: equity сцепляется мультипликативно."""
    if not segments:
        return TradingStats()
    per_bar = np.concatenate([s.per_bar_net for s in segments if s.n_bars])
    equity_level = 1.0
    peak = 1.0
    max_dd = 0.0
    for s in segments:
        if s.n_bars == 0:
            continue
        curve = equity_level * np.cumprod(1.0 + s.per_bar_net)
        peak = max(peak, equity_level)
        running_max = np.maximum.accumulate(np.concatenate([[equity_level], curve]))
        max_dd = max(max_dd, float(np.max(1.0 - curve / running_max[1:])))
        equity_level = float(curve[-1])
        peak = max(peak, equity_level)
    trades = [r for s in segments for r in s.trade_returns]
    return TradingStats(
        n_bars=int(sum(s.n_bars for s in segments)),
        n_trades=len(trades),
        bars_in_market=int(sum(s.bars_in_market for s in segments)),
        exposure=float(sum(s.bars_in_market for s in segments) / max(1, sum(s.n_bars for s in segments))),
        gross_return=float(sum(s.gross_return for s in segments)),
        cost_return=float(sum(s.cost_return for s in segments)),
        net_return=equity_level - 1.0,
        max_drawdown=max_dd,
        turnover=float(sum(s.turnover for s in segments)),
        per_bar_net=per_bar,
        trade_returns=tuple(trades),
    )


def derive_bars_per_year(
    ts: np.ndarray,
    *,
    trading_days_per_year: int = 252,
    local_timezone: str = "Europe/Moscow",
    config_override: int | None = None,
) -> tuple[float, float | None, str]:
    """(bars_per_year, bars_per_day_median, source).

    Если задан config_override — используется он (source='config'); иначе медиана
    баров на локальную календарную дату × trading_days_per_year (source='derived').
    Медиана устойчива к неполным дням в начале/конце выборки.
    """
    if config_override is not None:
        return float(config_override), None, "config"
    # ts — datetime64 без таймзоны (значения по контракту UTC) либо tz-aware массив
    # (pandas 3.x может вернуть datetime64[ns, UTC]); поддерживаем оба случая
    index = pd.DatetimeIndex(pd.to_datetime(ts))
    if index.tz is None:
        index = index.tz_localize("UTC")
    index = index.tz_convert(local_timezone)
    per_day = pd.Series(np.ones(len(index)), index=index.date).groupby(level=0).sum()
    bars_per_day = float(np.median(per_day.to_numpy())) if len(per_day) else 0.0
    return bars_per_day * trading_days_per_year, bars_per_day, "derived"


def annualized_cagr(total_net_return: float, n_bars: int, bars_per_year: float) -> float | None:
    if n_bars <= 0 or bars_per_year <= 0 or total_net_return <= -1.0:
        return None
    years = n_bars / bars_per_year
    if years <= 0:
        return None
    return float((1.0 + total_net_return) ** (1.0 / years) - 1.0)


def sharpe_ratio(per_bar_net: np.ndarray, bars_per_year: float, risk_free_annual: float) -> float | None:
    """Sharpe по баровым доходностям; rf пересчитывается на бар из годовой ставки."""
    if len(per_bar_net) < 2 or bars_per_year <= 0:
        return None
    std = float(np.std(per_bar_net, ddof=1))
    if std == 0:
        return None
    rf_per_bar = (1.0 + risk_free_annual) ** (1.0 / bars_per_year) - 1.0
    excess = np.asarray(per_bar_net, dtype=np.float64) - rf_per_bar
    return float(np.mean(excess) / std * math.sqrt(bars_per_year))
