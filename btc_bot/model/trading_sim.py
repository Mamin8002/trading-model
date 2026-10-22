"""Симулятор сделок для walk-forward-оценки и paper trading (Этап 5).

Единые правила для всех сравниваемых вариантов (модели и baseline'ы):
  * вход по close сигнального бара с добавлением слиппеджа В СТОРОНУ входа;
  * выход по истечении горизонта H баров (close) либо досрочно по настроенным
    правилам SL/TP/max_hold (проверка по high/low/close закрытых баров);
  * если в одном баре задеты и TP, и SL - считается SL (консервативно);
  * гэп через уровень исполняется по open (реалистично);
  * комиссии fee_pct берутся с обеих сторон; слиппедж - с обеих сторон;
  * сделки, чей горизонт выходит за последнее доступное окно (последние H баров),
    НЕ открываются (нет будущих данных для оценки - без утечки за границу окна);
  * одна позиция в момент времени (без пирамидинга); следующий вход - не раньше
    бара выхода.

Уровни SL/TP считаются от close сигнального бара (до слиппеджа) - как задаёт их
пользователь в конфигурации.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..data_sources.bybit import step_ms_approx


@dataclass(frozen=True)
class TradingCosts:
    fee_pct: float = 0.1        # комиссия за сторону, %
    slippage_pct: float = 0.02  # проскальзывание за сторону, %
    order_size_quote: float = 100.0


@dataclass(frozen=True)
class ExitRulesSim:
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    max_hold_bars: Optional[int] = None


def _max_drawdown_fraction(net_rets: list[float]) -> float:
    """Максимальная просадка накопленной кривой Σ net_ret (в долях единицы сделки)."""
    peak = 0.0
    cum = 0.0
    dd = 0.0
    for r in net_rets:
        cum += r
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def simulate_trades(
    ohlc: pd.DataFrame,
    prob_up: np.ndarray,
    threshold: float,
    costs: TradingCosts,
    exit_rules: ExitRulesSim,
    horizon_bars: int,
    interval: str,
) -> dict:
    """Симуляция сделок на окне. ohlc: start_ms/open/high/low/close (хронология).

    prob_up: вероятность класса +1 для каждой строки окна (для baseline'ов -
    вырожденные массивы 0/1, тогда вход на каждом баре).
    Возвращает метрики: n_trades, avg_net_pct, total_net_usd, win_rate,
    max_dd_fraction, avg_gross_pct, exit_reasons, skipped_entries.
    """
    n = len(ohlc)
    if n == 0:
        return {"n_trades": 0, "avg_net_pct": 0.0, "total_net_usd": 0.0, "win_rate": 0.0,
                "max_dd_fraction": 0.0, "avg_gross_pct": 0.0, "exit_reasons": {},
                "skipped_entries": 0}
    step = step_ms_approx(interval)
    t = ohlc["start_ms"].to_numpy(dtype="int64")
    o = ohlc["open"].to_numpy(dtype="float64")
    h = ohlc["high"].to_numpy(dtype="float64")
    l = ohlc["low"].to_numpy(dtype="float64")
    c = ohlc["close"].to_numpy(dtype="float64")
    fee = costs.fee_pct / 100.0
    slip = costs.slippage_pct / 100.0
    horizon_ms = horizon_bars * step

    net_rets: list[float] = []
    gross_rets: list[float] = []
    reasons: dict[str, int] = {}
    skipped = 0
    i = 0
    while i < n:
        conf = max(prob_up[i], 1.0 - prob_up[i])
        if conf < threshold:
            i += 1
            continue
        if t[i] + horizon_ms > t[-1]:
            # горизонт сделки выходит за окно - вход запрещён (нет данных для честного выхода)
            skipped += 1
            i += 1
            continue
        direction = 1 if prob_up[i] > 0.5 else -1
        entry_px = c[i] * (1.0 + direction * slip)
        tp = exit_rules.take_profit_pct
        sl = exit_rules.stop_loss_pct
        mh = exit_rules.max_hold_bars
        tp_level = c[i] * (1 + direction * tp / 100.0) if tp is not None else None
        sl_level = c[i] * (1 - direction * sl / 100.0) if sl is not None else None
        target_time = t[i] + horizon_ms

        exit_px: Optional[float] = None
        reason = "horizon"
        j = i + 1
        while j < n:
            # порядок проверок в баре: SL -> TP (консервативно) -> max_hold -> горизонт
            if sl_level is not None:
                hit = l[j] <= sl_level if direction == 1 else h[j] >= sl_level
                if hit:
                    exit_px = min(o[j], sl_level) if direction == 1 else max(o[j], sl_level)
                    reason = "stop_loss"
                    break
            if tp_level is not None:
                hit = h[j] >= tp_level if direction == 1 else l[j] <= tp_level
                if hit:
                    exit_px = max(o[j], tp_level) if direction == 1 else min(o[j], tp_level)
                    reason = "take_profit"
                    break
            bars_held = (t[j] - t[i]) // step
            if mh is not None and bars_held >= mh:
                exit_px = c[j]
                reason = "max_hold"
                break
            if t[j] >= target_time:
                exit_px = c[j]
                reason = "horizon"
                break
            j += 1
        if exit_px is None:
            skipped += 1  # не хватило окна (не должно случаться из-за entry-skip)
            i += 1
            continue
        exit_eff = exit_px * (1.0 - direction * slip)
        gross = direction * (exit_px / c[i] - 1.0)
        net = direction * (exit_eff / entry_px - 1.0) - 2.0 * fee
        net_rets.append(net)
        gross_rets.append(gross)
        reasons[reason] = reasons.get(reason, 0) + 1
        i = j + 1  # следующий вход не раньше следующего бара после выхода

    size = costs.order_size_quote
    total_net_usd = float(sum(net_rets) * size)
    return {
        "n_trades": len(net_rets),
        "avg_net_pct": float(np.mean(net_rets) * 100.0) if net_rets else 0.0,
        "total_net_usd": total_net_usd,
        "win_rate": float(np.mean([1.0 if r > 0 else 0.0 for r in net_rets])) if net_rets else 0.0,
        "max_dd_fraction": _max_drawdown_fraction(net_rets),
        "avg_gross_pct": float(np.mean(gross_rets) * 100.0) if gross_rets else 0.0,
        "exit_reasons": reasons,
        "skipped_entries": skipped,
        "threshold": float(threshold),
    }


DEFAULT_THR_GRID = tuple(round(x, 3) for x in np.arange(0.50, 0.7501, 0.01))


def select_threshold(
    ohlc_val: pd.DataFrame,
    prob_up_val: np.ndarray,
    costs: TradingCosts,
    exit_rules: ExitRulesSim,
    horizon_bars: int,
    interval: str,
    grid: tuple = DEFAULT_THR_GRID,
    min_trades: int = 5,
) -> tuple[float, dict]:
    """Выбор порога входа ТОЛЬКО на переданном (обучающем/валидационном) срезе.

    Критерий (предрегистрирован): средний net return на сделку; тай-брейк - более
    высокий порог (реже торгуем). Если ни один порог не дал >= min_trades сделок -
    возвращается максимальный порог сетки (консервативно).
    Вызывающий код обязан передавать сюда данные, НЕ входящие в тестовое окно.
    """
    best: Optional[tuple[float, float, dict]] = None
    for thr in grid:
        m = simulate_trades(ohlc_val, prob_up_val, thr, costs, exit_rules, horizon_bars, interval)
        if m["n_trades"] >= min_trades:
            score = m["avg_net_pct"]
            if best is None or score > best[0] + 1e-12 or (abs(score - best[0]) <= 1e-12 and thr > best[1]):
                best = (score, float(thr), m)
    if best is None:
        return float(grid[-1]), {"reason": f"нет порога с >= {min_trades} сделками - взят максимальный",
                                 "n_trades_evaluated": 0}
    return best[1], {"reason": "best avg net per trade", "val_avg_net_pct": best[0],
                     "val_n_trades": best[2]["n_trades"]}


def decide_adoption(primary_new: float, primary_current: float, primary_baseline: float) -> bool:
    """Pre-registered adopt-gate: новая версия заменяет текущую ТОЛЬКО если её
    первичная метрика (net return per trade на отложенных окнах) строго лучше
    И текущей модели, И majority-baseline."""
    if np.isnan(primary_new):
        return False
    cur = primary_current if not np.isnan(primary_current) else -np.inf
    base = primary_baseline if not np.isnan(primary_baseline) else -np.inf
    return primary_new > cur and primary_new > base
