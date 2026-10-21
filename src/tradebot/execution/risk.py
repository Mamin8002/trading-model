"""RiskGuards — проверки риск-контроля (docs/simulation.md §3, architecture §7).

Принципы (ТЗ фазы 4):
- каждый активный гейт опирается на ЯВНОЕ значение конфига; недостающий параметр —
  безопасный отказ RISK_PARAM_NOT_SET, а не «пропуск проверки»;
- отказ всегда машиночитаем (RejectionCode) и попадает в журнал с причиной;
- нарушение дневного лимита убытка / просадки / рассинхрон позиции / устаревшие
  или невалидные данные → остановка стратегии (halt): новые входы блокируются,
  защитные выходы (стопы) продолжают исполняться — это консервативный порядок,
  задокументированный в docs/simulation.md §5;
- секретов в решениях и журнале нет (тексты проходят через SecretMasker).

Финансовые величины (лимиты, капитал, риск) берутся только из конфига;
RiskGuards ничего не додумывает и не «округляет в пользу сделки».
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol
from zoneinfo import ZoneInfo

from ..config.schema import InstrumentConfig, RiskConfig, SimulationConfig
from .errors import RiskHaltError
from .interfaces import (
    OrderIntent,
    PortfolioState,
    RejectionCode,
    RiskContext,
    RiskDecision,
    RiskRejection,
    Side,
)
from .sizing import estimate_commission

__all__ = ["JournalSink", "RiskGuards"]


class JournalSink(Protocol):
    def record(self, kind: str, **fields: object) -> dict:  # pragma: no cover - protocol
        ...


def _is_step_multiple(price: float, step: float, tolerance: float = 1e-9) -> bool:
    ratio = price / step
    return abs(ratio - round(ratio)) <= tolerance * max(1.0, abs(ratio))


class RiskGuards:
    """Состояние: только флаг остановки; учёт дневных лимитов — в PortfolioState."""

    def __init__(
        self,
        risk_cfg: RiskConfig,
        instrument_cfg: InstrumentConfig | None,
        sim_cfg: SimulationConfig,
        *,
        local_timezone: str = "UTC",
        journal: JournalSink | None = None,
    ) -> None:
        self.risk_cfg = risk_cfg
        self.instrument_cfg = instrument_cfg
        self.sim_cfg = sim_cfg
        self.journal = journal
        self._tz = ZoneInfo(local_timezone)
        self.halted = False
        self.halt_reason: str | None = None
        self._expected_units_ledger = 0  # независимый учёт для сверки позиции

    # ------------------------------------------------------------------ halt

    def halt(self, reason: str, detail: str = "") -> None:
        if not self.halted:
            self.halted = True
            self.halt_reason = reason
            self._journal(
                "halt",
                reason=reason,
                detail=detail,
            )

    @property
    def is_halted(self) -> bool:
        return self.halted

    # ------------------------------------------------------------- дневные окна

    def rollover_day(self, portfolio: PortfolioState, ts: datetime) -> None:
        local_day: date = ts.astimezone(self._tz).date()
        if portfolio.day_key != local_day:
            portfolio.day_key = local_day
            portfolio.day_realized_pnl = 0.0
            portfolio.day_orders = 0

    # ---------------------------------------------------------------- оценка

    def evaluate(
        self,
        intent: OrderIntent,
        portfolio: PortfolioState,
        ctx: RiskContext,
    ) -> RiskDecision:
        """Полный цикл проверок намерения. Порядок фиксирован и логируется."""
        rejections: list[RiskRejection] = []
        warnings: list[str] = []
        checks: list[str] = []
        cfg = self.risk_cfg

        def reject(code: RejectionCode, message: str, **kw) -> None:
            rejections.append(RiskRejection(code, message, **kw))

        is_entry = intent.side is Side.BUY

        # Для ЗАЩИТНЫХ ВЫХОДОВ (SELL при открытой позиции) halt и устаревшие данные
        # не являются блокировкой: остановка запрещает НОВЫЕ ВХОДЫ, но выход из
        # риска разрешён всегда (консервативный порядок, docs/simulation.md §5).
        # 1. остановка стратегии
        checks.append("halt_flag")
        if self.halted and is_entry:
            reject(
                RejectionCode.HALTED,
                f"Стратегия остановлена ({self.halt_reason}); новые входы запрещены",
                limit="halt",
            )

        # 2. состояние источника данных и валидность бара
        checks.append("data_source")
        if not ctx.data_source_ok:
            if is_entry:
                reject(RejectionCode.INVALID_DATA, "Источник данных сообщил об ошибке — безопасный отказ")
            self.halt("data_source_error", "ошибка источника данных на момент решения")
        checks.append("bar_validity")
        if not ctx.bar_is_valid:
            if is_entry:
                reject(RejectionCode.INVALID_DATA, "Текущий бар невалиден (OHLC/порядок времени)")
            self.halt("invalid_data", f"невалидный бар {ctx.last_bar_ts_open.isoformat()}")

        # 3. устаревшие данные
        checks.append("data_age")
        if cfg.max_data_age_seconds is None:
            reject(
                RejectionCode.RISK_PARAM_NOT_SET,
                "risk.max_data_age_seconds не задан — проверка свежести данных невозможна",
                limit="risk.max_data_age_seconds",
            )
        else:
            age = ctx.data_age_seconds
            if age is None:
                # TEST-replay: решение принимается на закрытии последнего бара
                age = max(0.0, (ctx.decision_ts - ctx.last_bar_ts_open).total_seconds())
            if age > cfg.max_data_age_seconds:
                if is_entry:
                    reject(
                        RejectionCode.STALE_DATA,
                        f"Данные устарели: возраст {age:.0f} c > лимита {cfg.max_data_age_seconds:.0f} c",
                        limit="risk.max_data_age_seconds",
                        limit_value=cfg.max_data_age_seconds,
                        observed_value=age,
                    )
                self.halt("stale_data", f"возраст данных {age:.0f} c")

        # 4. просадка от пика капитала
        checks.append("drawdown")
        if cfg.max_drawdown_pct is None:
            reject(
                RejectionCode.RISK_PARAM_NOT_SET,
                "risk.max_drawdown_pct не задан",
                limit="risk.max_drawdown_pct",
            )
        elif portfolio.peak_equity > 0:
            dd_pct = (portfolio.peak_equity - ctx.equity) / portfolio.peak_equity * 100.0
            if dd_pct > cfg.max_drawdown_pct:
                if is_entry:
                    reject(
                        RejectionCode.DRAWDOWN,
                        f"Просадка {dd_pct:.2f}% превысила лимит {cfg.max_drawdown_pct:.2f}%",
                        limit="risk.max_drawdown_pct",
                        limit_value=cfg.max_drawdown_pct,
                        observed_value=dd_pct,
                    )
                if cfg.halt_on_limit_breach:
                    self.halt("drawdown_limit", f"просадка {dd_pct:.2f}%")

        # 5. дневной лимит убытка
        checks.append("daily_loss")
        if cfg.daily_loss_limit_quote is None:
            reject(
                RejectionCode.RISK_PARAM_NOT_SET,
                "risk.daily_loss_limit_quote не задан",
                limit="risk.daily_loss_limit_quote",
            )
        elif portfolio.day_realized_pnl <= -cfg.daily_loss_limit_quote:
            if is_entry:
                reject(
                    RejectionCode.DAILY_LOSS,
                    f"Дневной убыток {portfolio.day_realized_pnl:.2f} достиг лимита "
                    f"-{cfg.daily_loss_limit_quote:.2f}",
                    limit="risk.daily_loss_limit_quote",
                    limit_value=cfg.daily_loss_limit_quote,
                    observed_value=portfolio.day_realized_pnl,
                )
            if cfg.halt_on_limit_breach:
                self.halt("daily_loss_limit", f"дневной убыток {portfolio.day_realized_pnl:.2f}")

        if rejections:
            return self._finish(intent, RiskDecision(False, tuple(rejections), tuple(warnings), tuple(checks)))

        # 6. проверки стороны операции
        checks.append("side_policy")
        if intent.side is Side.SELL:
            if portfolio.units <= 0:
                if cfg.allow_short:
                    reject(
                        RejectionCode.SHORT_NOT_ALLOWED,
                        "Короткие продажи декларированы, но не реализованы в фазе 4 (long-only симулятор)",
                    )
                else:
                    reject(
                        RejectionCode.NO_POSITION_TO_CLOSE,
                        "Нет длинной позиции для закрытия; шорты запрещены (risk.allow_short=false)",
                    )
            elif intent.quantity_units > portfolio.units:
                reject(
                    RejectionCode.SHORT_NOT_ALLOWED,
                    f"Продажа {intent.quantity_units} при позиции {portfolio.units} "
                    "привела бы к шорту — запрещено",
                    observed_value=float(intent.quantity_units),
                    limit_value=float(portfolio.units),
                )
        else:  # BUY
            # 7. режим размера задан (сайзинг уже выполнен до evaluate — контролируем)
            checks.append("sizing_mode")
            if (
                cfg.position_units is None
                and cfg.risk_per_trade_quote is None
                and cfg.risk_per_trade_pct is None
            ):
                reject(
                    RejectionCode.RISK_PARAM_NOT_SET,
                    "Режим размера позиции не задан в risk-конфиге",
                    limit="risk.position_units|risk.risk_per_trade_*",
                )
            # 8. лимит заявок в день
            checks.append("orders_per_day")
            if cfg.max_orders_per_day is None:
                reject(
                    RejectionCode.RISK_PARAM_NOT_SET,
                    "risk.max_orders_per_day не задан",
                    limit="risk.max_orders_per_day",
                )
            elif portfolio.day_orders >= cfg.max_orders_per_day:
                reject(
                    RejectionCode.ORDER_LIMIT,
                    f"Лимит заявок на день исчерпан: {portfolio.day_orders}/{cfg.max_orders_per_day}",
                    limit="risk.max_orders_per_day",
                    limit_value=float(cfg.max_orders_per_day),
                    observed_value=float(portfolio.day_orders),
                )
            # 9. максимальная позиция
            checks.append("max_position")
            if cfg.max_position_units is None:
                reject(
                    RejectionCode.RISK_PARAM_NOT_SET,
                    "risk.max_position_units не задан",
                    limit="risk.max_position_units",
                )
            elif portfolio.units + intent.quantity_units > cfg.max_position_units:
                reject(
                    RejectionCode.MAX_POSITION,
                    f"Позиция после сделки ({portfolio.units + intent.quantity_units}) превысила бы "
                    f"лимит {cfg.max_position_units}",
                    limit="risk.max_position_units",
                    limit_value=float(cfg.max_position_units),
                    observed_value=float(portfolio.units + intent.quantity_units),
                )
            # 10. достаточность капитала (стоимость + комиссия по явным настройкам)
            checks.append("capital")
            notional = intent.quantity_units * intent.reference_price
            commission = estimate_commission(notional, self.sim_cfg)
            if notional + commission > ctx.cash + 1e-9:
                reject(
                    RejectionCode.INSUFFICIENT_CAPITAL,
                    f"Недостаточно капитала: требуется {notional + commission:.2f} "
                    f"(стоимость {notional:.2f} + комиссия {commission:.2f}), доступно {ctx.cash:.2f}",
                    limit="cash",
                    limit_value=ctx.cash,
                    observed_value=notional + commission,
                )

        # 11. параметры инструмента: шаг цены для стоп/цели и корректность границ
        checks.append("instrument_params")
        step = self.instrument_cfg.price_step if self.instrument_cfg else None
        for name, price in (("stop_price", intent.stop_price), ("target_price", intent.target_price)):
            if price is None:
                continue
            if step is None:
                warnings.append(
                    f"instrument.price_step не подтверждён — кратность {name} не проверяется (не подставляется)"
                )
            elif not _is_step_multiple(price, step):
                reject(
                    RejectionCode.INVALID_PRICE_STEP,
                    f"{name}={price} не кратна шагу цены {step}",
                    limit="instrument.price_step",
                    limit_value=step,
                    observed_value=price,
                )
        if intent.side is Side.BUY:
            if intent.stop_price is not None and intent.stop_price >= intent.reference_price:
                reject(
                    RejectionCode.INVALID_STOP_TARGET,
                    f"Для длинной позиции стоп ({intent.stop_price}) должен быть ниже цены входа "
                    f"({intent.reference_price})",
                )
            if intent.target_price is not None and intent.target_price <= intent.reference_price:
                reject(
                    RejectionCode.INVALID_STOP_TARGET,
                    f"Для длинной позиции цель ({intent.target_price}) должна быть выше цены входа "
                    f"({intent.reference_price})",
                )

        approved = not rejections
        return self._finish(intent, RiskDecision(approved, tuple(rejections), tuple(warnings), tuple(checks)))

    def _finish(self, intent: OrderIntent, decision: RiskDecision) -> RiskDecision:
        self._journal(
            "risk_decision",
            created_at=intent.created_at.isoformat(),
            symbol=intent.symbol,
            timeframe=intent.timeframe,
            side=intent.side.value,
            quantity_units=intent.quantity_units,
            model_version=intent.signal.model_version,
            approved=decision.approved,
            rejections=[r.__dict__ for r in decision.rejections],
            warnings=list(decision.warnings),
            checks=list(decision.checks_performed),
        )
        return decision

    # ------------------------------------------------------------- учёт филлов

    def on_fill(self, portfolio: PortfolioState, fill_price: float, side: Side, units: int, commission: float) -> None:
        """Обновляет риск-учёт после исполнения (вызывается симулятором)."""
        if side is Side.BUY:
            portfolio.day_orders += 1
            portfolio.realized_pnl -= commission
            portfolio.day_realized_pnl -= commission
            self._expected_units_ledger += units
        else:
            gross = (fill_price - portfolio.avg_entry_price) * units
            portfolio.realized_pnl += gross - commission
            portfolio.day_realized_pnl += gross - commission
            self._expected_units_ledger -= units
        portfolio.total_commission += commission
        # нарушение дневного лимита фиксируется сразу (остановка — при halt_on_limit_breach)
        limit = self.risk_cfg.daily_loss_limit_quote
        if limit is not None and portfolio.day_realized_pnl <= -limit and not self.halted:
            if self.risk_cfg.halt_on_limit_breach:
                self.halt("daily_loss_limit", f"дневной убыток {portfolio.day_realized_pnl:.2f} после исполнения")

    def verify_consistency(self, portfolio_units: int) -> bool:
        """Сверка позиции: независимый журнал филлов vs состояние портфеля.

        Расхождение = несогласованное состояние позиции → немедленная остановка
        (ТЗ фазы 4, п.6). Возвращает True при согласованности.
        """
        if portfolio_units != self._expected_units_ledger:
            self.halt(
                "position_desync",
                f"позиция портфеля {portfolio_units} != журнал исполнений {self._expected_units_ledger}",
            )
            self._journal(
                "risk_decision",
                approved=False,
                rejections=[
                    RiskRejection(
                        RejectionCode.POSITION_DESYNC,
                        "Несогласованное состояние позиции — торговля остановлена до сверки",
                        observed_value=float(portfolio_units),
                        limit_value=float(self._expected_units_ledger),
                    ).__dict__
                ],
            )
            return False
        return True

    def reset_expected_units(self, units: int) -> None:
        """Инициализация сверки стартовой позицией (по умолчанию 0)."""
        self._expected_units_ledger = units

    # ---------------------------------------------------------------- journal

    def _journal(self, kind: str, **fields: object) -> None:
        if self.journal is not None:
            self.journal.record(kind, **fields)
