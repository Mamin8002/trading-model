"""Расчёт размера позиции (docs/simulation.md §3.2).

Два ЯВНЫХ режима (ровно один задаётся конфигом risk):
1. фиксированный: ``risk.position_units`` — используется как есть; кратность лоту
   проверяется (НЕ подгоняется молча — при нарушении отказ INVALID_LOT);
2. рисковый: ``risk.risk_per_trade_quote`` или ``risk.risk_per_trade_pct`` —
   units = floor(risk_amount / loss_per_unit) с округлением ВНИЗ до лота,
   где loss_per_unit = |entry_price − stop_price| (стоп обязателен; без стопа —
   отказ SIZING_STOP_REQUIRED, а не «подбор» расстояния).

Если режим не задан — SizingError(RISK_PARAM_NOT_SET): безопасный отказ,
скрытые финансовые значения за пользователя не выбираются (ТЗ фазы 4).

Дополнительные верхние границы (применяются после расчёта и фиксируются в
журнале): доступный капитал (стоимость покупки + оценка комиссии) и
``risk.max_position_units``. Комиссия для оценки капитала берётся из simulation
(bps + минимум) — единственные доступные явные значения; «запас» не домножается.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import InstrumentConfig, RiskConfig, SimulationConfig
from .interfaces import RejectionCode, RiskRejection

__all__ = ["SizingError", "SizingResult", "compute_position_size", "estimate_commission"]


class SizingError(Exception):
    """Отказ расчёта размера с машиночитаемым кодом (пишется в журнал)."""

    def __init__(self, rejection: RiskRejection) -> None:
        self.rejection = rejection
        super().__init__(rejection.message)


@dataclass(frozen=True)
class SizingResult:
    units: int
    mode: str                     # fixed | risk_quote | risk_pct
    raw_units: int | None         # до округления до лота (рисковый режим)
    capped_by: str | None         # capital | max_position | None
    notes: tuple[str, ...] = ()


def estimate_commission(notional: float, sim_cfg: SimulationConfig) -> float:
    """Комиссия по явным настройкам: max(bps·notional, min_quote)."""
    if sim_cfg.commission_bps is None:
        raise SizingError(
            RiskRejection(
                RejectionCode.RISK_PARAM_NOT_SET,
                "simulation.commission_bps не задан — комиссия не выдумывается (ТЗ фазы 4)",
                limit="simulation.commission_bps",
            )
        )
    commission = notional * sim_cfg.commission_bps / 10_000.0
    if sim_cfg.commission_min_quote is not None:
        commission = max(commission, sim_cfg.commission_min_quote)
    return commission


def compute_position_size(
    *,
    risk_cfg: RiskConfig,
    instrument_cfg: InstrumentConfig | None,
    sim_cfg: SimulationConfig,
    entry_price: float,
    stop_price: float | None,
    cash: float,
    equity: float,
    current_units: int,
) -> SizingResult:
    """Возвращает размер в штуках или бросает SizingError с кодом отказа.

    entry_price — оценочная цена исполнения (open следующего бара в симуляторе
    передаёт вызывающий; на момент решения это последняя доступная цена + slip).
    """
    if entry_price <= 0:
        raise SizingError(
            RiskRejection(RejectionCode.INVALID_DATA, f"Некорректная цена входа: {entry_price}")
        )
    lot = instrument_cfg.lot_size_units if instrument_cfg else None
    notes: list[str] = []
    if lot is None:
        notes.append("lot_size_units не подтверждён источником — округление до лота пропущено (лот=1 не предполагается)")

    # --- режим размера ---------------------------------------------------
    if risk_cfg.position_units is not None:
        units = risk_cfg.position_units
        mode = "fixed"
        raw_units = None
        if lot is not None and units % lot != 0:
            raise SizingError(
                RiskRejection(
                    RejectionCode.INVALID_LOT,
                    f"position_units={units} не кратно lot_size_units={lot}; "
                    "исправьте конфиг — молчаливое округление пользовательского значения запрещено",
                    limit="instrument.lot_size_units",
                    limit_value=float(lot),
                    observed_value=float(units),
                )
            )
    elif risk_cfg.risk_per_trade_quote is not None or risk_cfg.risk_per_trade_pct is not None:
        if stop_price is None:
            raise SizingError(
                RiskRejection(
                    RejectionCode.SIZING_STOP_REQUIRED,
                    "Рисковый сайзинг требует стоп-цену сигнала (simulation.signal.stop_bps); "
                    "расстояние до стопа не подбирается автоматически",
                    limit="simulation.signal.stop_bps",
                )
            )
        loss_per_unit = abs(entry_price - stop_price)
        if loss_per_unit <= 0:
            raise SizingError(
                RiskRejection(
                    RejectionCode.INVALID_STOP_TARGET,
                    f"Стоп совпадает с ценой входа ({stop_price}); риск на единицу не определён",
                )
            )
        if risk_cfg.risk_per_trade_quote is not None:
            risk_amount = float(risk_cfg.risk_per_trade_quote)
            mode = "risk_quote"
        else:
            if equity <= 0:
                raise SizingError(
                    RiskRejection(RejectionCode.INSUFFICIENT_CAPITAL, "Капитал <= 0 — процентный риск не определён")
                )
            risk_amount = equity * float(risk_cfg.risk_per_trade_pct) / 100.0
            mode = "risk_pct"
        raw_units = int(risk_amount // loss_per_unit)
        units = raw_units
        if lot is not None:
            units = (units // lot) * lot  # округление ВНИЗ до целых лотов (консервативно)
    else:
        raise SizingError(
            RiskRejection(
                RejectionCode.RISK_PARAM_NOT_SET,
                "Режим размера позиции не задан: установите risk.position_units ЛИБО "
                "risk.risk_per_trade_quote/risk_per_trade_pct. Скрытые значения не подставляются",
                limit="risk.position_units|risk.risk_per_trade_*",
            )
        )

    if units <= 0:
        raise SizingError(
            RiskRejection(
                RejectionCode.RISK_PARAM_NOT_SET,
                f"Расчёт размера дал 0 единиц (режим {mode}) — проверьте риск/стоп/лот",
                limit="risk",
            )
        )

    # --- верхние границы --------------------------------------------------
    # В ФИКСИРОВАННОМ режиме размер НЕ урезается молча: превышение лимитов
    # (max_position, капитал) отклоняется риск-контролем с явным кодом отказа.
    # В рисковых режимах уменьшение размера — часть расчёта (с фиксацией в журнале).
    capped_by: str | None = None
    if mode != "fixed":
        if risk_cfg.max_position_units is not None:
            room = risk_cfg.max_position_units - current_units
            if room < 0:
                room = 0
            if units > room:
                units = room
                capped_by = "max_position"
                notes.append(f"урезано до max_position_units={risk_cfg.max_position_units}")
            if lot is not None:
                units = (units // lot) * lot
        # капитал: стоимость + комиссия должны покрываться свободными средствами
        while units > 0:
            notional = units * entry_price
            commission = estimate_commission(notional, sim_cfg)
            if notional + commission <= cash + 1e-9:
                break
            step = lot if lot is not None else 1
            units -= step
            capped_by = "capital"
        if units <= 0:
            raise SizingError(
                RiskRejection(
                    RejectionCode.INSUFFICIENT_CAPITAL,
                    f"Капитала не хватает даже на 1 шаг позиции (cash={cash:.2f}, entry={entry_price:.4f}, "
                    f"commission_bps={sim_cfg.commission_bps})",
                    limit="cash",
                    limit_value=cash,
                    observed_value=entry_price,
                )
            )
        if capped_by == "capital":
            notes.append("размер ограничен доступным капиталом (стоимость + комиссия)")

    return SizingResult(units=int(units), mode=mode, raw_units=raw_units, capped_by=capped_by, notes=tuple(notes))
