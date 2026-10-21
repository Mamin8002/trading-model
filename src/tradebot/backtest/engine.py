"""EventSimulator — событийный TEST-симулятор (docs/simulation.md, architecture §2 backtest/).

Хронология обработки каждого бара (строго причинная):
1. валидность бара (порядок времени, OHLC-инварианты) — иначе БЕЗОПАСНАЯ ОСТАНОВКА;
2. смена локального торгового дня (сброс дневных счётчиков);
3. исполнение ОТЛОЖЕННЫХ намерений предыдущего бара по open этого бара
   (вход/выход по сигналу — «не раньше следующей доступной цены», ТЗ фазы 4);
4. проверка стопа/цели открытой позиции по диапазону этого бара
   (консервативное правило конфликта — docs/simulation.md §4);
5. отметка капитала по close, пик/просадка, проверка лимита просадки;
6. принятие решения ПОСЛЕ закрытия бара: сигнал -> сайзинг -> риск-контроль ->
   отложенное намерение на следующий бар (или мотивированный отказ в журнал).

Гарантии:
- цена закрытия сигнального бара используется только как reference/оценка для
  сайзинга и НИКОГДА как цена исполнения;
- если следующего бара нет (конец данных), намерение не исполняется —
  отказ NO_NEXT_BAR в журнале;
- позиция, открытая на конец данных, НЕ закрывается фиктивной сделкой —
  warning «position_open_at_end», оценка mark-to-market по последнему close;
- halt блокирует новые входы; защитные выходы (стопы) продолжают работать;
- после каждого исполнения — сверка позиции (независимый журнал филлов vs
  портфель); расхождение -> остановка POSITION_DESYNC.

Не моделируется (ограничения, docs/simulation.md §7): частичные исполнения,
ликвидность/очередь, внутрибарная траектория (кроме правил стоп/цель), шорты,
маржа, налоги.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

from ..config.schema import AppConfig
from ..data.bar import Bar, Timeframe
from ..execution.errors import IntrabarDataRequired
from ..execution.interfaces import (
    ExitCause,
    Fill,
    HaltEvent,
    OrderIntent,
    PortfolioState,
    RejectionCode,
    RiskContext,
    Side,
    Signal,
)
from ..execution.journal import JournalWriter
from ..execution.risk import RiskGuards
from ..execution.simulator import SimulatedBroker, validate_simulation_config
from ..execution.sizing import SizingError, compute_position_size

__all__ = ["EventSimulator", "PositionEpisode", "SimulationRun"]


@dataclass
class PositionEpisode:
    entry: Fill
    exit: Fill | None = None
    stop_price: float | None = None
    target_price: float | None = None
    conflict: bool = False
    exit_cause: str | None = None


@dataclass
class SimulationRun:
    symbol: str
    timeframe: str
    model_version: str
    bars_processed: int
    fills: list[Fill] = field(default_factory=list)
    episodes: list[PositionEpisode] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    halts: list[HaltEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conflict_bars: list[datetime] = field(default_factory=list)
    portfolio: PortfolioState | None = None
    journal_path: str | None = None
    completed: bool = True  # False, если цикл прерван безопасной остановкой

    @property
    def halted(self) -> bool:
        return bool(self.halts)


def _intent_id(ts: datetime, side: Side, units: int) -> str:
    """Детерминированный id намерения (<= 20 символов, как client_order_id LIVE)."""
    digest = hashlib.sha1(f"{ts.isoformat()}|{side.value}|{units}".encode()).hexdigest()
    return digest[:20]


def _round_step_up(price: float, step: float) -> float:
    return math.ceil(price / step - 1e-12) * step


def _round_step_down(price: float, step: float) -> float:
    return math.floor(price / step + 1e-12) * step


class EventSimulator:
    def __init__(
        self,
        *,
        config: AppConfig,
        bars: Sequence[Bar],
        signal_source,
        timeframe: "Timeframe | str",
        symbol: str,
        journal: JournalWriter | None = None,
        bar_hook: Callable[[int, Bar, "EventSimulator"], None] | None = None,
    ) -> None:
        validate_simulation_config(config.simulation, config.risk)
        if config.risk is None or config.simulation is None:  # для mypy/ясности
            raise ValueError("risk и simulation обязательны (проверено validate_simulation_config)")
        if not bars:
            raise ValueError("EventSimulator: пустая история баров")
        self.config = config
        self.bars = list(bars)
        self.signal_source = signal_source
        self.timeframe = Timeframe.parse(timeframe)
        self.symbol = symbol
        self.journal = journal
        self.bar_hook = bar_hook  # тестовый хук (инъекция сбоев), в бою не используется

        self.guards = RiskGuards(
            config.risk,
            config.instrument,
            config.simulation,
            local_timezone=config.data.source_timezone,
            journal=journal,
        )
        self.broker = SimulatedBroker(config.simulation, config.instrument)
        self.portfolio = PortfolioState(
            cash=float(config.risk.initial_capital_quote),
            peak_equity=float(config.risk.initial_capital_quote),
        )
        self._signals = signal_source.signals_for(self.bars)
        self._episode: PositionEpisode | None = None
        self._pending: OrderIntent | None = None

    # ------------------------------------------------------------------ запуск

    def run(self) -> SimulationRun:
        run = SimulationRun(
            symbol=self.symbol,
            timeframe=self.timeframe.value,
            model_version=getattr(self.signal_source, "model_version", "unknown"),
            bars_processed=0,
            journal_path=str(self.journal.path) if self.journal else None,
        )
        self._run = run
        prev_bar: Bar | None = None
        for index, bar in enumerate(self.bars):
            # 1. валидность бара и порядка времени
            valid = self._bar_is_valid(bar, prev_bar)
            if not valid:
                self._halt(run, bar.ts_open, "invalid_data",
                           f"бар {bar.ts_open.isoformat()} невалиден (OHLC/порядок) — симуляция остановлена")
                break
            self.guards.rollover_day(self.portfolio, bar.ts_open)

            # 2. исполнение отложенных намерений по open
            self._process_pending(run, bar)

            # 3. стоп/цель по диапазону бара (после возможного входа на этом же баре —
            #    консервативно: стоп может сработать в баре входа)
            stopped = self._check_stop_target(run, bar)
            if stopped:
                break

            # 4. отметка капитала по close + просадка
            equity = self.portfolio.equity(bar.close)
            if equity > self.portfolio.peak_equity:
                self.portfolio.peak_equity = equity
            run.equity_curve.append((bar.ts_open, equity))
            dd_cfg = self.config.risk.max_drawdown_pct
            if dd_cfg is not None and self.portfolio.peak_equity > 0 and not self.guards.is_halted:
                dd_pct = (self.portfolio.peak_equity - equity) / self.portfolio.peak_equity * 100.0
                if dd_pct > dd_cfg:
                    self.guards.halt("drawdown_limit", f"просадка {dd_pct:.2f}% > {dd_cfg:.2f}% на баре {bar.ts_open.isoformat()}")
                    run.halts.append(HaltEvent(bar.ts_open, "drawdown_limit", f"просадка {dd_pct:.4f}%"))

            # 5. решение после закрытия бара
            self._decide(run, index, bar)

            run.bars_processed = index + 1
            prev_bar = bar
            if self.bar_hook is not None:
                self.bar_hook(index, bar, self)
            if self.guards.is_halted and self._halt_breaks(run):
                break

        # конец данных: открытая позиция НЕ закрывается фиктивной сделкой
        if self.portfolio.units > 0:
            message = (
                f"position_open_at_end: {self.portfolio.units} шт остаются открытыми на конец данных "
                f"(оценка mark-to-market по последнему close; фиктивная продажа не создаётся)"
            )
            run.warnings.append(message)
            self._journal("warning", message=message, event_ts=self.bars[-1].ts_open.isoformat())
        if self._episode is not None and self._episode.exit is None:
            run.episodes.append(self._episode)  # незакрытый эпизод — для отчёта как есть
        if self._pending is not None:
            # намерение без следующего бара — фиксируем отказ
            self._record_rejection(run, self.bars[-1].ts_open, RejectionCode.NO_NEXT_BAR,
                                   "Конец данных: следующий бар для исполнения недоступен")
            self._pending = None
        self._journal(
            "state",
            event_ts=self.bars[-1].ts_open.isoformat(),
            cash=self.portfolio.cash,
            units=self.portfolio.units,
            realized_pnl=self.portfolio.realized_pnl,
            total_commission=self.portfolio.total_commission,
            peak_equity=self.portfolio.peak_equity,
            halted=self.guards.is_halted,
            halt_reason=self.guards.halt_reason,
        )
        run.portfolio = self.portfolio
        run.completed = not run.halted
        return run

    # ------------------------------------------------------------------ шаги

    @staticmethod
    def _bar_is_valid(bar: Bar, prev_bar: Bar | None) -> bool:
        if prev_bar is not None and bar.ts_open <= prev_bar.ts_open:
            return False
        if not (bar.high >= bar.low and bar.high >= bar.open and bar.high >= bar.close
                and bar.low <= bar.open and bar.low <= bar.close):
            return False
        if bar.low <= 0 or not bar.is_final:
            return False
        return True

    def _process_pending(self, run: SimulationRun, bar: Bar) -> None:
        intent = self._pending
        if intent is None:
            return
        self._pending = None
        if intent.side is Side.BUY:
            fill = self.broker.execute_at_bar_open(
                bar, Side.BUY, intent.quantity_units, "entry_signal",
                symbol=intent.symbol, timeframe=intent.timeframe, intent_id=intent.client_intent_id,
            )
            self._apply_fill(run, fill)
            self._open_episode(run, intent, fill, bar)
        else:
            if self.portfolio.units <= 0:
                run.warnings.append(f"exit-intent на баре {bar.ts_open.isoformat()} без позиции — пропущен")
                return
            fill = self.broker.execute_at_bar_open(
                bar, Side.SELL, self.portfolio.units, "exit_signal",
                symbol=intent.symbol, timeframe=intent.timeframe, intent_id=intent.client_intent_id,
            )
            self._apply_fill(run, fill)
            self._close_episode(run, fill, ExitCause.SIGNAL.value)

    def _open_episode(self, run: SimulationRun, intent: OrderIntent, fill: Fill, bar: Bar) -> None:
        """Стоп/цель: явные из сигнала либо из simulation.signal.*_bps от ЦЕНЫ ФИЛЛА.

        Округление до шага цены — консервативное (стоп для длинной позиции
        округляется ВВЕРХ — сработает не позже заданного; цель — ВНИЗ),
        см. docs/simulation.md §4.
        """
        step = self.config.instrument.price_step if self.config.instrument else None
        stop = intent.stop_price
        target = intent.target_price
        sig_cfg = self.config.simulation.signal
        if stop is None and sig_cfg.stop_bps is not None:
            stop = fill.price * (1.0 - sig_cfg.stop_bps / 10_000.0)
        if target is None and sig_cfg.target_bps is not None:
            target = fill.price * (1.0 + sig_cfg.target_bps / 10_000.0)
        if step is not None:
            if stop is not None:
                stop = _round_step_up(stop, step)    # консервативно: сработает не позже заданного
            if target is not None:
                target = _round_step_down(target, step)  # консервативно: не выше заданного
        else:
            if stop is not None or target is not None:
                run.warnings.append(
                    "instrument.price_step не подтверждён — стоп/цель не округляются до шага цены"
                )
        if stop is not None and stop >= fill.price:
            run.warnings.append(
                f"стоп {stop} не ниже цены входа {fill.price} после округления — стоп отключён для позиции"
            )
            stop = None
        if target is not None and target <= fill.price:
            run.warnings.append(
                f"цель {target} не выше цены входа {fill.price} после округления — цель отключена для позиции"
            )
            target = None
        self._episode = PositionEpisode(entry=fill, stop_price=stop, target_price=target)
        self._journal(
            "state",
            event_ts=bar.ts_open.isoformat(),
            position_opened=True,
            units=fill.units,
            entry_price=fill.price,
            stop_price=stop,
            target_price=target,
        )

    def _check_stop_target(self, run: SimulationRun, bar: Bar) -> bool:
        """Возвращает True, если симуляция прервана (требуются внутрибарные данные)."""
        episode = self._episode
        if episode is None or episode.exit is not None:
            return False
        if episode.stop_price is None and episode.target_price is None:
            return False
        try:
            plan = self.broker.evaluate_exit_on_bar(
                bar,
                entry_price=episode.entry.price,
                stop_price=episode.stop_price,
                target_price=episode.target_price,
            )
        except IntrabarDataRequired as exc:
            self._halt(run, bar.ts_open, "intrabar_data_required", str(exc))
            return True
        if plan is None:
            return False
        fill = self.broker.execute_exit_in_bar(
            bar, plan, self.portfolio.units,
            symbol=self.symbol, timeframe=self.timeframe.value,
            intent_id=episode.entry.intent_id,
        )
        if plan.conflict:
            run.conflict_bars.append(bar.ts_open)
            message = (
                f"stop_target_conflict на баре {bar.ts_open.isoformat()}: low<=стоп и high>=цель; "
                f"внутрибарных данных нет — применено консервативное правило "
                f"{self.config.simulation.conflict_policy} (исполнен стоп)"
            )
            run.warnings.append(message)
            self._journal(
                "warning",
                event_ts=bar.ts_open.isoformat(),
                message=message,
                stop_price=episode.stop_price,
                target_price=episode.target_price,
            )
        self._apply_fill(run, fill)
        self._close_episode(run, fill, plan.cause.value, conflict=plan.conflict)
        return False

    def _decide(self, run: SimulationRun, index: int, bar: Bar) -> None:
        signal: Signal | None = self._signals.get(bar.ts_open)
        if signal is None:
            return
        self._journal(
            "signal",
            event_ts=bar.ts_open.isoformat(),
            side=signal.side.value if signal.side else "exit",
            proba=signal.proba,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            meta=signal.meta,
        )
        in_position = self.portfolio.units > 0
        if signal.side is Side.BUY:
            if in_position:
                run.warnings.append(
                    f"вход-сигнал на баре {bar.ts_open.isoformat()} при открытой позиции — "
                    "игнорируется (реверсы в фазе 4 не поддержаны)"
                )
                return
            self._prepare_entry(run, index, bar, signal)
        else:
            # side is None или SELL = выход (источник сигналов кладёт в карту только
            # осмысленные решения; «нет сигнала» = отсутствие ключа)
            if not in_position:
                return  # выход без позиции — не событие (журнал уже содержит signal)
            self._prepare_exit(run, index, bar, signal)

    def _prepare_entry(self, run: SimulationRun, index: int, bar: Bar, signal: Signal) -> None:
        sim_cfg = self.config.simulation
        assert self.config.risk is not None
        # оценка цены исполнения для сайзинга/капитала: следующий open неизвестен на момент
        # решения — используем close*(1+slippage) как ЯВНУЮ оценку (допущение в отчёте)
        est_entry = bar.close * (1.0 + (sim_cfg.slippage_bps or 0.0) / 10_000.0)
        ctx = RiskContext(
            decision_ts=bar.ts_close(self.timeframe),
            last_bar_ts_open=bar.ts_open,
            last_close=bar.close,
            equity=self.portfolio.equity(bar.close),
            cash=self.portfolio.cash,
            position_units=self.portfolio.units,
            data_age_seconds=0.0,  # TEST-replay: решение на закрытии последнего бара
        )
        try:
            sizing = compute_position_size(
                risk_cfg=self.config.risk,
                instrument_cfg=self.config.instrument,
                sim_cfg=sim_cfg,
                entry_price=est_entry,
                stop_price=signal.stop_price if signal.stop_price is not None else self._planned_stop(est_entry),
                cash=self.portfolio.cash,
                equity=ctx.equity,
                current_units=self.portfolio.units,
            )
        except SizingError as exc:
            self._record_rejection(run, bar.ts_open, exc.rejection.code, exc.rejection.message,
                                   limit=exc.rejection.limit)
            return
        for note in sizing.notes:
            run.warnings.append(note)
        intent = OrderIntent(
            created_at=bar.ts_open,
            symbol=self.symbol,
            timeframe=self.timeframe.value,
            side=Side.BUY,
            quantity_units=sizing.units,
            signal=signal,
            reference_price=bar.close,  # СПРАВОЧНО; не цена исполнения
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            client_intent_id=_intent_id(bar.ts_open, Side.BUY, sizing.units),
        )
        decision = self.guards.evaluate(intent, self.portfolio, ctx)
        self._sync_halts(run, bar.ts_open)
        if not decision.approved:
            for rejection in decision.rejections:
                self._record_rejection(run, bar.ts_open, rejection.code, rejection.message,
                                       limit=rejection.limit)
            return
        if index + 1 >= len(self.bars):
            self._record_rejection(run, bar.ts_open, RejectionCode.NO_NEXT_BAR,
                                   "Сигнал на последнем баре: следующая доступная цена отсутствует")
            return
        self._pending = intent
        self._journal(
            "intent",
            event_ts=bar.ts_open.isoformat(),
            intent_id=intent.client_intent_id,
            side=intent.side.value,
            quantity_units=intent.quantity_units,
            reference_price=intent.reference_price,
            reference_price_note="close сигнального бара — справочно, НЕ цена исполнения",
            sizing_mode=sizing.mode,
            sizing_raw_units=sizing.raw_units,
            capped_by=sizing.capped_by,
            stop_price=intent.stop_price,
            target_price=intent.target_price,
            planned_execution="open следующего бара +/- slippage_bps",
        )

    def _planned_stop(self, est_entry: float) -> float | None:
        bps = self.config.simulation.signal.stop_bps
        if bps is None:
            return None
        return est_entry * (1.0 - bps / 10_000.0)

    def _prepare_exit(self, run: SimulationRun, index: int, bar: Bar, signal: Signal) -> None:
        intent = OrderIntent(
            created_at=bar.ts_open,
            symbol=self.symbol,
            timeframe=self.timeframe.value,
            side=Side.SELL,
            quantity_units=self.portfolio.units,
            signal=signal,
            reference_price=bar.close,
            client_intent_id=_intent_id(bar.ts_open, Side.SELL, self.portfolio.units),
        )
        ctx = RiskContext(
            decision_ts=bar.ts_close(self.timeframe),
            last_bar_ts_open=bar.ts_open,
            last_close=bar.close,
            equity=self.portfolio.equity(bar.close),
            cash=self.portfolio.cash,
            position_units=self.portfolio.units,
            data_age_seconds=0.0,
        )
        decision = self.guards.evaluate(intent, self.portfolio, ctx)
        self._sync_halts(run, bar.ts_open)
        if not decision.approved:
            for rejection in decision.rejections:
                self._record_rejection(run, bar.ts_open, rejection.code, rejection.message, limit=rejection.limit)
            return
        if index + 1 >= len(self.bars):
            self._record_rejection(run, bar.ts_open, RejectionCode.NO_NEXT_BAR,
                                   "Сигнал выхода на последнем баре: следующая доступная цена отсутствует")
            return
        self._pending = intent
        self._journal(
            "intent",
            event_ts=bar.ts_open.isoformat(),
            intent_id=intent.client_intent_id,
            side=intent.side.value,
            quantity_units=intent.quantity_units,
            reference_price=intent.reference_price,
            reference_price_note="close сигнального бара — справочно, НЕ цена исполнения",
            planned_execution="open следующего бара -/ + slippage_bps",
        )

    # ----------------------------------------------------------------- учёт

    def _apply_fill(self, run: SimulationRun, fill: Fill) -> None:
        portfolio = self.portfolio
        if fill.side is Side.BUY:
            portfolio.cash -= fill.price * fill.units + fill.commission_quote
            portfolio.avg_entry_price = fill.price
            portfolio.units += fill.units
            self.guards.on_fill(portfolio, fill.price, fill.side, fill.units, fill.commission_quote)
        else:
            # on_fill ДО сброса avg_entry: реализованный PnL считается от средней цены входа
            self.guards.on_fill(portfolio, fill.price, fill.side, fill.units, fill.commission_quote)
            portfolio.cash += fill.price * fill.units - fill.commission_quote
            portfolio.units -= fill.units
            if portfolio.units == 0:
                portfolio.avg_entry_price = 0.0
        run.fills.append(fill)
        self._journal(
            "fill",
            event_ts=fill.ts.isoformat(),
            intent_id=fill.intent_id,
            side=fill.side.value,
            units=fill.units,
            raw_price=fill.raw_price,
            price=fill.price,
            slippage_bps=fill.slippage_bps,
            commission_quote=fill.commission_quote,
            cause=fill.cause,
            bar_ts_open=fill.bar_ts_open.isoformat(),
            cash_after=portfolio.cash,
            units_after=portfolio.units,
        )
        self._sync_halts(run, fill.ts)
        # сверка согласованности позиции (независимый журнал vs портфель)
        if not self.guards.verify_consistency(portfolio.units):
            run.halts.append(HaltEvent(fill.ts, "position_desync", "позиция портфеля не совпала с журналом исполнений"))
            raise _DesyncHalt()

    def _sync_halts(self, run: SimulationRun, ts: datetime) -> None:
        """Фиксирует в run.halts остановку, установленную риск-контролем
        (дневной лимит в on_fill, stale/invalid в evaluate и т.п.)."""
        if self.guards.is_halted:
            reason = self.guards.halt_reason or "unknown"
            if not any(h.reason == reason for h in run.halts):
                run.halts.append(HaltEvent(ts, reason, "установлено риск-контролем"))
                self._journal("halt", event_ts=ts.isoformat(), reason=reason,
                              detail="установлено риск-контролем")

    def _close_episode(self, run: SimulationRun, fill: Fill, cause: str, conflict: bool = False) -> None:
        if self._episode is not None:
            self._episode.exit = fill
            self._episode.exit_cause = cause
            self._episode.conflict = conflict
            run.episodes.append(self._episode)
            self._episode = None

    def _halt(self, run: SimulationRun, ts: datetime, reason: str, detail: str) -> None:
        self.guards.halt(reason, detail)
        run.halts.append(HaltEvent(ts, reason, detail))
        self._journal("halt", event_ts=ts.isoformat(), reason=reason, detail=detail)

    def _halt_breaks(self, run: SimulationRun) -> bool:
        """Остановка прерывает цикл при invalid/desync/intrabar; лимитные halt
        (drawdown/daily_loss) продолжают обработку баров, чтобы защитные стопы
        могли исполниться (консервативный порядок, docs/simulation.md §5)."""
        if not run.halts:
            return False
        return run.halts[-1].reason in ("invalid_data", "position_desync", "intrabar_data_required", "data_source_error", "stale_data")

    def _record_rejection(self, run: SimulationRun, ts: datetime, code: RejectionCode, message: str, limit: str | None = None) -> None:
        run.rejections.append(
            {"ts": ts.isoformat(), "code": code.value, "message": message, "limit": limit}
        )

    def _journal(self, kind: str, **fields: object) -> None:
        if self.journal is not None:
            self.journal.record(kind, **fields)


class _DesyncHalt(Exception):
    """Внутренний сигнал прерывания цикла при рассинхроне позиции."""


def run_simulation(
    config: AppConfig,
    bars: Sequence[Bar],
    signal_source,
    *,
    timeframe: "Timeframe | str",
    symbol: str,
    journal: JournalWriter | None = None,
    bar_hook: Callable[[int, Bar, EventSimulator], None] | None = None,
) -> SimulationRun:
    """Удобная обёртка: создаёт EventSimulator и выполняет прогон.

    Прерывание по рассинхрону позиции перехватывается и возвращается как
    обычный SimulationRun с halt-событием (безопасная остановка, не авария).
    """
    engine = EventSimulator(
        config=config,
        bars=bars,
        signal_source=signal_source,
        timeframe=timeframe,
        symbol=symbol,
        journal=journal,
        bar_hook=bar_hook,
    )
    try:
        return engine.run()
    except _DesyncHalt:
        run = engine._run
        run.completed = False
        if engine.portfolio.units > 0:
            run.warnings.append("position_open_at_end: остановка по рассинхрону, позиция не ликвидирована автоматически")
        run.portfolio = engine.portfolio
        return run
