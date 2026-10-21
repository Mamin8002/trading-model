"""Многофакторный гейт LIVE (docs/live.md). Статус: LIVE ЗАБЛОКИРОВАН.

LIVE-контур по умолчанию ЗАКРЫТ. Ключевое отличие крипто-версии: интеграция с
биржей НЕ реализована (exchange/base.py), поэтому условие №0 не выполнено НИКОГДА
и гейт не проходит ни при каких комбинациях флагов/конфигов/переменных окружения.
Это явная безопасная блокировка, а не скрытый fallback: факт availability читается
из единственной константы EXCHANGE_INTEGRATION_AVAILABLE.

Для разрешения ордерных операций (ПОСЛЕ будущей реализации адаптера биржи)
потребуется одновременное выполнение ВСЕХ условий (скрытых обходов нет):

0. реализованная интеграция с биржей (exchange.base.EXCHANGE_INTEGRATION_AVAILABLE);
1. ``run.mode: LIVE`` в конфигурации;
2. явный флаг CLI ``--allow-live`` (отдельный запуск, не по умолчанию);
3. переменная окружения ``TRADEBOT_LIVE_ACK`` == значению из ``live.ack_expected_value``;
4. отдельная секция ``live`` с ``allow_orders: true``;
5. ``live.max_order_quantity_units`` задан явно;
6. параметры инструмента подтверждены: ``instrument.price_step`` и
   ``instrument.lot_size_units`` (без них нельзя валидировать цену/количество);
7. полный risk-конфиг (те же требования, что для симулятора: капитал, режим
   размера, все лимиты — validate_simulation_config);
8. файл kill-switch отсутствует.

Любое невыполненное условие -> LiveGateError со СПИСКОМ всех нарушений; вызов
происходит ДО любых сетевых вызовов — заявки физически не могут быть отправлены
при непройденном гейте.

ВАЖНО: даже после реализации адаптера прохождение гейта НЕ включает торговый
цикл автоматически — LIVE-оркестратор (потоки, BarFinalizer, reconciliation-цикл)
также требует отдельной реализации (docs/roadmap.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..config.schema import AppConfig
from ..exchange.base import EXCHANGE_INTEGRATION_AVAILABLE
from ..execution.errors import SimConfigError
from ..execution.simulator import validate_simulation_config

__all__ = ["LiveGateError", "LiveGateResult", "evaluate_live_gate"]


class LiveGateError(Exception):
    """Гейт LIVE не пройден. Содержит полный список нарушений."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = failures
        super().__init__(
            "LIVE-гейт НЕ пройден — любые сетевые вызовы и заявки запрещены:\n  - "
            + "\n  - ".join(failures)
        )


@dataclass(frozen=True)
class LiveGateResult:
    allowed: bool
    failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks_performed: tuple[str, ...] = field(default=())

    def raise_if_denied(self) -> None:
        if not self.allowed:
            raise LiveGateError(list(self.failures))


def evaluate_live_gate(
    config: AppConfig,
    *,
    cli_flag: bool,
    env: Mapping[str, str] | None = None,
    kill_switch_exists: bool | None = None,
) -> LiveGateResult:
    """Чистая функция-гейт: не делает сетевых вызовов и не читает файлы сама
    (kill_switch_exists передаёт вызывающий; None -> проверка файла по конфигу)."""
    import os

    environment: Mapping[str, str] = os.environ if env is None else env
    failures: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    # Условие №0: интеграция с биржей. Сейчас False всегда -> LIVE заблокирован.
    checks.append("exchange_integration")
    if not EXCHANGE_INTEGRATION_AVAILABLE:
        failures.append(
            "интеграция с криптобиржей НЕ реализована (exchange/base.py): ордерный контур "
            "невозможен; LIVE заблокирован до отдельной реализации подключения к бирже"
        )

    checks.append("run.mode")
    if config.run.mode != "LIVE":
        failures.append(
            f"run.mode={config.run.mode!r}: LIVE требует явного mode: LIVE в отдельной конфигурации"
        )

    checks.append("cli_flag")
    if not cli_flag:
        failures.append("флаг CLI --allow-live не передан (LIVE включается только явным флагом запуска)")

    live = config.live
    checks.append("live_section")
    if live is None:
        failures.append("секция live отсутствует: LIVE-контур не сконфигурирован (allow_orders, kill-switch, лимит заявки)")
    else:
        checks.append("live.allow_orders")
        if not live.allow_orders:
            failures.append("live.allow_orders: false — ордерный контур выключен конфигурацией")
        checks.append("live.max_order_quantity_units")
        if live.max_order_quantity_units is None:
            failures.append("live.max_order_quantity_units не задан — лимит размера заявки обязателен")
        checks.append("ack_env")
        ack_value = str(environment.get(live.ack_env_var, ""))
        if ack_value != live.ack_expected_value:
            failures.append(
                f"переменная окружения {live.ack_env_var} не установлена или не равна "
                f"ожидаемому подтверждению (см. docs/live.md)"
            )
        checks.append("kill_switch")
        exists = kill_switch_exists
        if exists is None:
            exists = Path(live.kill_switch_file).exists()
        if exists:
            failures.append(
                f"kill-switch активен ({live.kill_switch_file}): новые заявки запрещены до удаления файла"
            )

    checks.append("instrument_params")
    instrument = config.instrument
    if instrument is None or instrument.price_step is None:
        failures.append("instrument.price_step не подтверждён — валидация цены заявки невозможна")
    if instrument is None or instrument.lot_size_units is None:
        failures.append("instrument.lot_size_units не подтверждён — расчёт количества невозможен")

    checks.append("risk_completeness")
    try:
        validate_simulation_config(config.simulation, config.risk, require_risk=True)
    except SimConfigError as exc:
        failures.extend(exc.missing)

    allowed = not failures
    if allowed:
        # Недостижимо, пока EXCHANGE_INTEGRATION_AVAILABLE = False; ветвь сохранена
        # для будущей реализации адаптера — и там предупреждение останется честным.
        warnings.append(
            "гейт пройден, но LIVE-оркестратор (потоки/сверка) требует отдельной реализации "
            "(docs/roadmap.md); адаптер биржи не проверен в реальной среде (docs/live.md)"
        )
    return LiveGateResult(
        allowed=allowed,
        failures=tuple(failures),
        warnings=tuple(warnings),
        checks_performed=tuple(checks),
    )
