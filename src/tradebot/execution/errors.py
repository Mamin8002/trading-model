"""Ошибки исполнительного контура фазы 4."""

from __future__ import annotations

__all__ = ["IntrabarDataRequired", "RiskHaltError", "SimConfigError"]


class SimConfigError(Exception):
    """Конфигурация симулятора/рисков неполна или противоречива.

    Симуляция НЕ запускается с недостающими явными параметрами (ТЗ фазы 4:
    «если значения не заданы, требуй явную конфигурацию или безопасно откажись»).
    Сообщение перечисляет все недостающие поля.
    """

    def __init__(self, missing: list[str], message: str | None = None) -> None:
        self.missing = missing
        text = message or (
            "Не заданы обязательные параметры симуляции/риска (скрытые значения не подставляются):\n  - "
            + "\n  - ".join(missing)
        )
        super().__init__(text)


class RiskHaltError(Exception):
    """Стратегия остановлена риск-контролем (лимит, рассинхрон, kill switch)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"HALT[{reason}]: {detail}" if detail else f"HALT[{reason}]")


class IntrabarDataRequired(Exception):
    """Конфликт «стоп и цель в одном баре» при политике require_intrabar_data.

    Симулятор отказывается угадывать выгодный сценарий и требует внутрибарные
    данные (тики/L2) — см. docs/simulation.md §4.
    """

    def __init__(self, bar_ts: str, stop: float, target: float) -> None:
        self.bar_ts = bar_ts
        self.stop = stop
        self.target = target
        super().__init__(
            f"Бар {bar_ts}: достигнуты и стоп ({stop}), и цель ({target}); "
            "внутрибарных данных нет — порядок исполнения неоднозначен. "
            "Политика require_intrabar_data: симуляция остановлена, предоставьте тиковые данные "
            "или выберите консервативную политику stop_first явно в конфиге"
        )
