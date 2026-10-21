"""Расчёт результатов симуляции и отчёт (docs/simulation.md §6).

Все торговые цифры — результаты ДИАГНОСТИЧЕСКОЙ симуляции с ЯВНЫМИ допущениями
(комиссии, слиппедж, модель исполнения, годовой пересчёт). Отчёт обязан показывать
допущения и ограничения; заявлений о прибыльности нет и быть не должно (D16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from ..config.schema import AppConfig
from ..execution.interfaces import Fill
from ..ml.metrics import annualized_cagr, derive_bars_per_year, sharpe_ratio
from .engine import PositionEpisode, SimulationRun

__all__ = ["SimulationResults", "compute_results", "render_simulation_report"]


@dataclass
class TradeSummary:
    entry_ts: str
    exit_ts: str | None
    units: int
    entry_price: float
    exit_price: float | None
    cause: str | None
    commission: float
    net_pnl: float | None      # None для незакрытого эпизода
    open_at_end: bool


@dataclass
class SimulationResults:
    symbol: str
    timeframe: str
    model_version: str
    period: tuple[str, str]
    bars_processed: int
    initial_capital: float
    final_equity: float
    net_return: float
    total_commission: float
    estimated_slippage_cost: float
    turnover_quote: float
    n_entries: int
    n_round_trips: int
    n_rejections: int
    rejections_by_code: dict[str, int]
    n_halts: int
    halt_reasons: list[str]
    n_conflicts: int
    max_drawdown: float
    exposure: float
    hit_rate: float | None
    profit_factor: float | None
    trades: list[TradeSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    annualized: dict[str, Any] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)
    risk_limits: dict[str, Any] = field(default_factory=dict)
    completed: bool = True


def _episode_pnl(episode: PositionEpisode) -> tuple[float | None, float]:
    """(чистый PnL эпизода, комиссия). Для открытого эпизода PnL=None."""
    commission = episode.entry.commission_quote + (episode.exit.commission_quote if episode.exit else 0.0)
    if episode.exit is None:
        return None, commission
    gross = (episode.exit.price - episode.entry.price) * episode.entry.units
    return gross - commission, commission


def compute_results(
    run: SimulationRun,
    config: AppConfig,
    bars: Sequence[Any],
    *,
    local_timezone: str | None = None,
) -> SimulationResults:
    risk_cfg = config.risk
    sim_cfg = config.simulation
    assert risk_cfg is not None and sim_cfg is not None  # проверено стартом симулятора
    local_tz = local_timezone or config.data.source_timezone

    portfolio = run.portfolio
    if portfolio is None:
        raise ValueError("SimulationRun без portfolio — симуляция не выполнена")

    initial = float(risk_cfg.initial_capital_quote)
    final_equity = run.equity_curve[-1][1] if run.equity_curve else initial
    net_return = final_equity / initial - 1.0 if initial > 0 else 0.0

    total_commission = sum(f.commission_quote for f in run.fills)
    slippage_cost = sum(abs(f.price - f.raw_price) * f.units for f in run.fills)
    turnover = sum(f.price * f.units for f in run.fills)

    # просадка по кривой капитала (пик отсчитывается не ниже стартового капитала)
    max_dd = 0.0
    if run.equity_curve:
        equities = np.array([e for _, e in run.equity_curve], dtype="float64")
        peaks = np.maximum.accumulate(np.maximum(equities, initial))
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peaks > 0, 1.0 - equities / peaks, 0.0)
        max_dd = float(np.max(dd)) if len(dd) else 0.0

    bars_in_market = 0
    for ts, _equity in run.equity_curve:
        if _position_at(run, ts):
            bars_in_market += 1

    trades: list[TradeSummary] = []
    pnls: list[float] = []
    for episode in run.episodes:
        pnl, commission = _episode_pnl(episode)
        trades.append(
            TradeSummary(
                entry_ts=episode.entry.ts.isoformat(),
                exit_ts=episode.exit.ts.isoformat() if episode.exit else None,
                units=episode.entry.units,
                entry_price=episode.entry.price,
                exit_price=episode.exit.price if episode.exit else None,
                cause=episode.exit_cause,
                commission=commission,
                net_pnl=pnl,
                open_at_end=episode.exit is None,
            )
        )
        if pnl is not None:
            pnls.append(pnl)

    hit_rate = float(np.mean([p > 0 for p in pnls])) if pnls else None
    gains = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    profit_factor = float(sum(gains) / abs(sum(losses))) if gains and losses else None

    rejections_by_code: dict[str, int] = {}
    for rejection in run.rejections:
        rejections_by_code[rejection["code"]] = rejections_by_code.get(rejection["code"], 0) + 1

    # годовой пересчёт (допущения — из конфига/данных, источник фиксируется)
    from ..data.bar import UTC

    ts_array = (
        np.array([b.ts_open.astimezone(UTC).replace(tzinfo=None) for b in bars], dtype="datetime64[us]")
        if len(bars)
        else np.empty(0, dtype="datetime64[us]")
    )
    bars_per_year, bars_per_day, source = derive_bars_per_year(
        ts_array,
        trading_days_per_year=config.evaluation.trading_days_per_year,
        local_timezone=local_tz,
        config_override=config.evaluation.bars_per_year,
    )
    per_bar_returns = _per_bar_returns(run.equity_curve)
    annualized = {
        "cagr": annualized_cagr(net_return, len(run.equity_curve), bars_per_year),
        "sharpe": sharpe_ratio(per_bar_returns, bars_per_year, config.evaluation.risk_free_annual),
        "bars_per_year": bars_per_year,
        "bars_per_day_median": bars_per_day,
        "bars_per_year_source": source,
    }

    assumptions = {
        "commission_bps": sim_cfg.commission_bps,
        "commission_min_quote": sim_cfg.commission_min_quote,
        "slippage_bps": sim_cfg.slippage_bps,
        "fill_policy": "сигнал после закрытия бара t -> исполнение по open бара t+1 +/- слиппедж (невыгодной стороной), кламп [low, high] бара",
        "stop_target_conflict": sim_cfg.conflict_policy,
        "intrabar": "внутрибарная траектория не моделируется; стоп/цель — по low/high бара с консервативным правилом конфликта",
        "liquidity": "полное исполнение заявленного объёма; частичные филлы и очередь не моделируются",
        "open_position_at_end": "позиция на конец данных НЕ закрывается фиктивной сделкой; оценка mark-to-market",
        "entry_price_estimate_for_sizing": "close сигнального бара * (1 + slippage) — только для сайзинга/проверки капитала",
        "annualization": annualized,
        "quote_currency": config.instrument.quote_currency if config.instrument else None,
        "disclaimer": "значения комиссий/слиппеджа — ЯВНЫЕ допущения пользователя, а не фактические тарифы биржи",
    }
    risk_limits = {
        "initial_capital_quote": risk_cfg.initial_capital_quote,
        "sizing_mode": (
            "fixed(position_units)" if risk_cfg.position_units is not None
            else "risk_per_trade_quote" if risk_cfg.risk_per_trade_quote is not None
            else "risk_per_trade_pct" if risk_cfg.risk_per_trade_pct is not None
            else "НЕ ЗАДАН"
        ),
        "position_units": risk_cfg.position_units,
        "risk_per_trade_quote": risk_cfg.risk_per_trade_quote,
        "risk_per_trade_pct": risk_cfg.risk_per_trade_pct,
        "max_position_units": risk_cfg.max_position_units,
        "max_orders_per_day": risk_cfg.max_orders_per_day,
        "daily_loss_limit_quote": risk_cfg.daily_loss_limit_quote,
        "max_drawdown_pct": risk_cfg.max_drawdown_pct,
        "max_data_age_seconds": risk_cfg.max_data_age_seconds,
        "allow_short": risk_cfg.allow_short,
        "halt_on_limit_breach": risk_cfg.halt_on_limit_breach,
        "price_step": config.instrument.price_step if config.instrument else None,
        "lot_size_units": config.instrument.lot_size_units if config.instrument else None,
    }

    exposure = bars_in_market / len(run.equity_curve) if run.equity_curve else 0.0

    return SimulationResults(
        symbol=run.symbol,
        timeframe=run.timeframe,
        model_version=run.model_version,
        period=(run.equity_curve[0][0].isoformat(), run.equity_curve[-1][0].isoformat()) if run.equity_curve else ("", ""),
        bars_processed=run.bars_processed,
        initial_capital=initial,
        final_equity=final_equity,
        net_return=net_return,
        total_commission=total_commission,
        estimated_slippage_cost=slippage_cost,
        turnover_quote=turnover,
        n_entries=sum(1 for f in run.fills if f.cause == "entry_signal"),
        n_round_trips=len([e for e in run.episodes if e.exit is not None]),
        n_rejections=len(run.rejections),
        rejections_by_code=rejections_by_code,
        n_halts=len(run.halts),
        halt_reasons=[h.reason for h in run.halts],
        n_conflicts=len(run.conflict_bars),
        max_drawdown=max_dd,
        exposure=exposure,
        hit_rate=hit_rate,
        profit_factor=profit_factor,
        trades=trades,
        warnings=list(run.warnings),
        annualized=annualized,
        assumptions=assumptions,
        risk_limits=risk_limits,
        completed=run.completed,
    )


def _position_at(run: SimulationRun, ts: datetime) -> bool:
    """Была ли позиция открыта в момент close бара ts (по филлам)."""
    units = 0
    for fill in run.fills:
        if fill.ts > ts:
            break
        units += fill.units if fill.side.value == "buy" else -fill.units
    return units > 0


def _per_bar_returns(equity_curve: list[tuple[datetime, float]]) -> np.ndarray:
    if len(equity_curve) < 2:
        return np.empty(0)
    equities = np.array([e for _, e in equity_curve], dtype="float64")
    return equities[1:] / equities[:-1] - 1.0


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "н/д"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_simulation_report(results: SimulationResults, config: AppConfig, journal_path: str | None) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Отчёт симуляции: {config.run.experiment_id} ({results.symbol}, {results.timeframe})")
    add("")
    add("> **Дисклеймер.** Результаты — диагностическая симуляция на истории с ЯВНЫМИ")
    add("> допущениями. Не являются обещанием доходности; система не production-ready (D16).")
    add("> Реальные заявки не выставлялись и не имитировались подтверждения брокера.")
    add("")
    add("## 1. Прогон")
    add("")
    add(f"- модель/источник сигналов: `{results.model_version}`")
    add(f"- период: {results.period[0]} .. {results.period[1]} ({results.bars_processed} баров)")
    add(f"- завершён штатно: {'да' if results.completed else 'НЕТ (безопасная остановка: ' + ', '.join(results.halt_reasons) + ')'}")
    add(f"- журнал решений/сделок: `{journal_path or 'не вёлся'}`")
    add("")
    add("## 2. Рисковые лимиты (все — ЯВНЫЕ значения конфига)")
    add("")
    add("| параметр | значение |")
    add("|---|---|")
    for key, value in results.risk_limits.items():
        add(f"| {key} | {'НЕ ЗАДАН' if value is None else value} |")
    add("")
    add("## 3. Допущения симуляции (обязательный раздел)")
    add("")
    for key, value in results.assumptions.items():
        if key == "annualization":
            continue
        add(f"- **{key}**: {value}")
    ann = results.annualized
    add(
        f"- **годовой пересчёт**: bars_per_year={_fmt(ann.get('bars_per_year'), 1)} "
        f"(источник: {ann.get('bars_per_year_source')}"
        + (f", медиана {ann.get('bars_per_day_median')} баров/день" if ann.get("bars_per_day_median") else "")
        + f", торговых дней/год: {config.evaluation.trading_days_per_year})"
    )
    add("")
    add("## 4. Результаты")
    add("")
    add(f"- капитал: {results.initial_capital:.2f} -> {results.final_equity:.2f} "
        f"({_fmt(results.net_return * 100, 2)}% за период)")
    add(f"- комиссии всего: {results.total_commission:.4f}; оценочные издержки слиппеджа: {results.estimated_slippage_cost:.4f}")
    add(f"- оборот: {results.turnover_quote:.2f}; входов: {results.n_entries}; завершённых сделок: {results.n_round_trips}")
    add(f"- max drawdown: {_fmt(results.max_drawdown)}; exposure: {_fmt(results.exposure, 3)}")
    add(f"- hit-rate (сделки): {_fmt(results.hit_rate, 3)}; profit factor: {_fmt(results.profit_factor, 2)}")
    add(f"- CAGR: {_fmt(ann.get('cagr'))}; Sharpe: {_fmt(ann.get('sharpe'), 2)} (допущения выше)")
    add("")
    add("### Отказы риск-контроля")
    add("")
    if results.rejections_by_code:
        for code, count in sorted(results.rejections_by_code.items()):
            add(f"- `{code}`: {count}")
    else:
        add("- отказов не было")
    add("")
    if results.n_conflicts:
        add(f"### Конфликты «стоп и цель в одном баре»: {results.n_conflicts}")
        add("")
        add(f"Применено правило: `{results.assumptions['stop_target_conflict']}` (см. раздел 3 и docs/simulation.md §4).")
        add("")
    if results.n_halts:
        add("### Остановки (halt)")
        add("")
        for reason in results.halt_reasons:
            add(f"- {reason}")
        add("")
    add("### Сделки")
    add("")
    add("| вход | выход | шт | цена входа | цена выхода | причина выхода | комиссия | чистый PnL |")
    add("|---|---|---|---|---|---|---|---|")
    for trade in results.trades[:200]:
        add(
            f"| {trade.entry_ts} | {trade.exit_ts or 'открыта'} | {trade.units} | {trade.entry_price:.4f} "
            f"| {_fmt(trade.exit_price)} | {trade.cause or '—'} | {trade.commission:.4f} "
            f"| {_fmt(trade.net_pnl)}{' (открыта)' if trade.open_at_end else ''} |"
        )
    if len(results.trades) > 200:
        add(f"| ... ещё {len(results.trades) - 200} сделок в журнале | | | | | | | |")
    add("")
    if results.warnings:
        add("### Предупреждения")
        add("")
        for warning in results.warnings[:50]:
            add(f"- {warning}")
        add("")
    add("## 5. Ограничения симулятора")
    add("")
    add("- частичные исполнения, ликвидность стакана и очередь не моделируются;")
    add("- внутрибарная траектория неизвестна: стоп/цель проверяются по low/high,")
    add("  конфликт разрешается консервативным правилом (или остановкой — по конфигу);")
    add("- шорты, маржа/плечо, налоги не моделируются; long-only, один символ;")
    add("- комиссии/слиппедж — пользовательские допущения, а не фактические тарифы;")
    add("- результаты на истории не гарантируют будущие (docs/decisions.md D16).")
    add("")
    return "\n".join(lines)
