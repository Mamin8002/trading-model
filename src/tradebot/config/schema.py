"""Схема конфигурации (docs/architecture.md §3.4, docs/decisions.md D10).

Принципы:
- декларативный YAML + строгая pydantic-схема (extra="forbid"): опечатка в конфиге —
  ошибка загрузки, а не молчаливое поведение по умолчанию;
- ``lookback_bars`` и ``horizon_bars`` для каждого включённого таймфрейма —
  ОБЯЗАТЕЛЬНЫЕ поля без значений по умолчанию (требование ТЗ: значения явные,
  ничего не подбирается молча при запуске);
- секреты (API-ключи будущей биржи) в конфиге запрещены — только имена
  env-переменных, значения читаются из окружения (docs/live.md);
- крипторежим: метки времени CSV по умолчанию трактуются как UTC, рынок 24/7
  (docs/data.md §2-3); секция биржевого API отсутствует — подключение биржи
  НЕ реализовано (exchange/base.py, docs/live.md).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..data.bar import SUPPORTED_TIMEFRAMES

__all__ = [
    "AppConfig",
    "DataConfig",
    "DaySessionsConfig",
    "LabelConfig",
    "RunConfig",
    "SessionsConfig",
    "SessionWindowConfig",
    "SymbolConfig",
    "TimeframeConfig",
    "ValidationConfig",
]

_STRICT = ConfigDict(extra="forbid")
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_HHMM_END = re.compile(r"^(?:([01]?\d|2[0-3]):([0-5]\d)|24:00)$")


def _validate_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Неизвестная таймзона {value!r}: {exc}") from exc
    return value


class RunConfig(BaseModel):
    model_config = _STRICT

    mode: Literal["TEST", "LIVE"] = "TEST"
    experiment_id: str = Field(min_length=1, max_length=120)
    #: каталоги артефактов обучения (модели/отчёты); оба игнорируются git-ом
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")


class SymbolConfig(BaseModel):
    """Торговая пара криптобиржи (например BTCUSDT или ETH/USDT — формат биржи).

    Тикер — строковый идентификатор пары; сопоставление с реальными инструментами
    биржи появится вместе с биржевым адаптером (НЕ реализован — exchange/base.py).
    Прежний формат символа ticker@mic удалён вместе с брокерской интеграцией (docs/decisions.md D12).
    """

    model_config = _STRICT

    ticker: str = Field(min_length=1, max_length=32)
    #: файлы истории по таймфреймам: {"1m": "BTCUSDT_M1.csv", ...}
    files: dict[str, str] = Field(default_factory=dict)

    @field_validator("files")
    @classmethod
    def _files_keys_supported(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value:
            if key not in SUPPORTED_TIMEFRAMES:
                raise ValueError(
                    f"Неподдерживаемый таймфрейм файла {key!r}; допустимы: {', '.join(SUPPORTED_TIMEFRAMES)}"
                )
        return value

    @property
    def symbol(self) -> str:
        """Символ инструмента (для криптопары совпадает с тикером)."""
        return self.ticker


class DataConfig(BaseModel):
    model_config = _STRICT

    csv_dir: Path = Path("data/raw")
    symbols: list[SymbolConfig] = Field(min_length=1)
    #: ЯВНАЯ семантика метки времени в CSV; значение по умолчанию отсутствует —
    #: её необходимо подтвердить в конфиге (docs/data.md §2, decisions D1).
    timestamp_semantics: Literal["bar_open", "bar_close"]
    #: криптобиржи отдают свечи в UTC — соглашение проекта по умолчанию (docs/data.md §2)
    source_timezone: str = "UTC"
    #: в каких единицах колонка volume вашей выгрузки: базовый актив (BTC для
    #: BTCUSDT — стандарт крипто-свечей) или квотирующий (USDT)
    volume_units: Literal["base_asset", "quote_asset"] = "base_asset"

    @field_validator("source_timezone")
    @classmethod
    def _tz_known(cls, value: str) -> str:
        return _validate_timezone(value)


class SessionWindowConfig(BaseModel):
    model_config = _STRICT

    start: str
    end: str

    @field_validator("start")
    @classmethod
    def _hhmm_start(cls, value: str) -> str:
        if not _HHMM.match(value):
            raise ValueError(f"Ожидается формат HH:MM, получено {value!r}")
        return value

    @field_validator("end")
    @classmethod
    def _hhmm_end(cls, value: str) -> str:
        if not _HHMM_END.match(value):
            raise ValueError(
                f'Ожидается формат HH:MM (для конца окна допустимо "24:00"), получено {value!r}'
            )
        return value

    @model_validator(mode="after")
    def _start_before_end(self) -> "SessionWindowConfig":
        def minutes(v: str) -> int:
            h, m = v.split(":")
            return int(h) * 60 + int(m)

        if minutes(self.start) >= minutes(self.end):
            raise ValueError(
                f"Окно сессии start={self.start} должно быть раньше end={self.end} "
                '(окна через полночь не поддерживаются; для рынка 24/7 используйте 00:00-24:00)'
            )
        return self


class DaySessionsConfig(BaseModel):
    model_config = _STRICT

    #: 0 = понедельник ... 6 = воскресенье
    days: list[int] = Field(min_length=1)
    sessions: list[SessionWindowConfig] = Field(min_length=1)

    @field_validator("days")
    @classmethod
    def _days_range(cls, value: list[int]) -> list[int]:
        for day in value:
            if not 0 <= day <= 6:
                raise ValueError(f"День недели вне диапазона 0..6: {day}")
        if len(set(value)) != len(value):
            raise ValueError("Дублирующиеся дни недели в одном правиле")
        return value


class SessionsConfig(BaseModel):
    model_config = _STRICT

    #: время окон сессий; для крипторынка — UTC с окном 00:00-24:00 (рынок 24/7)
    timezone: str = "UTC"
    days: list[DaySessionsConfig] = Field(min_length=1)

    @field_validator("timezone")
    @classmethod
    def _tz_known(cls, value: str) -> str:
        return _validate_timezone(value)


class LabelConfig(BaseModel):
    """Целевая переменная (decisions D5, docs/ml.md §3).

    kind:
    - fwd_return_class: классификация направления forward-return через horizon_bars
      с нейтральной полосой neutral_band_bps (движение в полосе -> нейтрально и
      исключается из обучения);
    - fwd_return_reg: регрессия forward-return;
    - triple_barrier: НЕ реализовано в фазе 3 (отложено до этапа B1+, decisions D5),
      запрос этого вида завершается явной ошибкой, а не молчаливой подменой.
    """

    model_config = _STRICT

    kind: Literal["fwd_return_class", "fwd_return_reg", "triple_barrier"] = "fwd_return_class"
    neutral_band_bps: float = Field(default=0.0, ge=0.0)


class TimeframeConfig(BaseModel):
    model_config = _STRICT

    enabled: bool = True
    #: ОБЯЗАТЕЛЬНЫЕ явные значения (требование ТЗ) — без defaults.
    lookback_bars: int = Field(ge=1)
    horizon_bars: int = Field(ge=1)
    #: Базовый таймфрейм для локального ресэмплинга (обязателен для "10m":
    #: нативных 10-минутных свечей криптобиржи обычно не отдают — 10m строится
    #: локально из 5m/1m, docs/decisions.md D3).
    base_timeframe: str | None = None
    label: LabelConfig | None = None

    @field_validator("base_timeframe")
    @classmethod
    def _base_supported(cls, value: str | None) -> str | None:
        if value is not None and value not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Неподдерживаемый base_timeframe {value!r}; допустимы: {', '.join(SUPPORTED_TIMEFRAMES)}"
            )
        return value


class FeatureParamsConfig(BaseModel):
    """Периоды индикаторов набора признаков v1 (docs/ml.md §2). Все значения явные,
    меняются только через конфиг; документированные дефолты — общепринятые периоды."""

    model_config = _STRICT

    rsi_period: int = Field(default=14, ge=2, le=500)
    macd_fast: int = Field(default=12, ge=2, le=500)
    macd_slow: int = Field(default=26, ge=3, le=1000)
    macd_signal: int = Field(default=9, ge=2, le=500)
    atr_period: int = Field(default=14, ge=2, le=500)
    ema_periods: list[int] = Field(default_factory=lambda: [8, 21, 50], min_length=1, max_length=6)
    return_lags: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 21], min_length=1, max_length=12)
    vol_windows: list[int] = Field(default_factory=lambda: [12, 48], min_length=1, max_length=6)
    volume_window: int = Field(default=20, ge=2, le=2000)
    volume_trend_window: int = Field(default=60, ge=2, le=5000)

    @model_validator(mode="after")
    def _consistent(self) -> "FeatureParamsConfig":
        if self.macd_fast >= self.macd_slow:
            raise ValueError(f"macd_fast ({self.macd_fast}) должен быть меньше macd_slow ({self.macd_slow})")
        for name, values in (("ema_periods", self.ema_periods), ("return_lags", self.return_lags), ("vol_windows", self.vol_windows)):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} содержит дубликаты: {values}")
            if any(v < 1 for v in values):
                raise ValueError(f"{name} содержит значения < 1: {values}")
        if sorted(self.vol_windows) != list(self.vol_windows):
            raise ValueError(f"vol_windows должен быть по возрастанию: {self.vol_windows}")
        if self.volume_trend_window <= self.volume_window:
            raise ValueError(
                f"volume_trend_window ({self.volume_trend_window}) должен быть больше "
                f"volume_window ({self.volume_window})"
            )
        return self


class FeaturesConfig(BaseModel):
    model_config = _STRICT

    #: версия набора признаков: манифест колонок сохраняется вместе с моделью
    set: Literal["v1"] = "v1"
    params: FeatureParamsConfig = Field(default_factory=FeatureParamsConfig)


class ModelParamsConfig(BaseModel):
    """Параметры LightGBM. Фиксируются ДО обучения (decisions D6): подбор на
    валидационных фолдах запрещён; early stopping использует holdout внутри
    обучающего окна (см. early_stopping_holdout_bars)."""

    model_config = _STRICT

    n_estimators: int = Field(default=300, ge=10, le=5000)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    num_leaves: int = Field(default=31, ge=2, le=512)
    max_depth: int = Field(default=-1)
    min_child_samples: int = Field(default=50, ge=1)
    subsample: float = Field(default=1.0, gt=0.0, le=1.0)
    subsample_freq: int = Field(default=0, ge=0)
    colsample_bytree: float = Field(default=0.9, gt=0.0, le=1.0)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=0.0, ge=0.0)
    #: баров в конце ОБУЧАЮЩЕГО окна под holdout для early stopping (0 = выключено,
    #: фиксированное n_estimators). Валидационный фолд для подбора НЕ используется.
    early_stopping_holdout_bars: int = Field(default=0, ge=0)
    early_stopping_rounds: int = Field(default=30, ge=5)

    @model_validator(mode="after")
    def _depth(self) -> "ModelParamsConfig":
        if self.max_depth != -1 and self.max_depth < 1:
            raise ValueError("max_depth должен быть -1 (без ограничения) или >= 1")
        if self.subsample < 1.0 and self.subsample_freq == 0:
            raise ValueError("subsample < 1.0 требует subsample_freq >= 1")
        return self


class ModelConfig(BaseModel):
    model_config = _STRICT

    #: lightgbm — основной кандидат (decisions D4); catboost — контрольный,
    #: в фазе 3 не подключён (явная ошибка при запуске, не молчаливая замена)
    kind: Literal["lightgbm", "catboost"] = "lightgbm"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)


class WalkForwardConfig(BaseModel):
    """Хронологическая purged walk-forward схема (decisions D6, docs/ml.md §4)."""

    model_config = _STRICT

    n_splits: int = Field(default=4, ge=2, le=20)
    #: зазор между обучающим и проверочным окнами, баров. None = автоматически
    #: равно horizon_bars таймфрейма (минимальное честное значение); явное значение
    #: должно быть >= horizon_bars (проверяется в AppConfig).
    embargo_bars: int | None = Field(default=None, ge=0)
    min_train_bars: int = Field(default=500, ge=50)
    min_val_bars: int = Field(default=20, ge=5)
    #: диагностический аудит «random-split vs walk-forward» (architecture §6.8):
    #: большая разница AUC трактуется как признак утечки/нестационарности.
    #: Для самого разбиения аудита результат НЕ используется при выборе модели.
    audit_random_split: bool = True


class ThresholdsConfig(BaseModel):
    """Диагностические пороги приёмки, зафиксированные ДО обучения (decisions D6).

    Это НЕ гарантия прибыльности: пороги отвечают только на вопрос «прошёл ли
    прогон заранее заданный диагностический минимум»."""

    model_config = _STRICT

    min_oos_auc: float | None = Field(default=None, ge=0.5, le=1.0)
    min_oos_trades: int | None = Field(default=None, ge=0)


class ValidationConfig(BaseModel):
    model_config = _STRICT

    max_reported_issues: int = Field(default=50, ge=1)
    warn_zero_volume: bool = True
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)


class EvaluationConfig(BaseModel):
    """Параметры оценочной симуляции (упрощённая модель исполнения; полный
    симулятор — execution/simulator.py). Значения комиссий/слиппеджа — ОЦЕНОЧНЫЕ
    допущения, а не фактические комиссии биржи: уточняйте тарифы своей биржи."""

    model_config = _STRICT

    commission_bps: float = Field(default=3.0, ge=0.0, le=500.0)
    slippage_bps: float = Field(default=2.0, ge=0.0, le=500.0)
    #: порог вероятности класса 1 для длинной позиции (классификация);
    #: для регрессии позиция определяется знаком прогноза (docs/ml.md §5)
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    #: допущения годового пересчёта (docs/ml.md §5.4). Крипторынок торгуется
    #: без выходных — 365 дней в году (календарный факт, не финансовое допущение)
    trading_days_per_year: int = Field(default=365, ge=100, le=366)
    #: None = вывести из данных (медиана баров на торговый день × trading_days_per_year)
    bars_per_year: int | None = Field(default=None, ge=1)
    risk_free_annual: float = Field(default=0.0, ge=0.0, le=0.5)


class RiskConfig(BaseModel):
    """Правила риск-контроля фазы 4 (docs/simulation.md §3).

    НИКАКИХ скрытых финансовых значений по умолчанию: все лимиты — None.
    Поведение при неза заданном значении (требование ТЗ фазы 4):
    - параметры сайзинга (position_units ИЛИ risk_per_trade_*): расчёт размера
      позиции БЕЗОПАСНО ОТКЛАДЫВАЕТСЯ (отказ с кодом RISK_PARAM_NOT_SET);
    - лимиты (max_position_units, daily_loss_limit_quote, max_drawdown_pct,
      max_orders_per_day, max_data_age_seconds): проверка, которой не хватает
      параметра, отказывает во входе с тем же кодом — лимит нельзя «молча
      пропустить».
    initial_capital_quote обязателен для запуска симулятора (проверяется при старте).
    """

    model_config = _STRICT

    #: стартовый капитал в валюте котировки — ЯВНОЕ значение, дефолта нет
    initial_capital_quote: float | None = Field(default=None, gt=0.0)
    #: режим фиксированного размера: единиц инструмента на сделку (в штуках)
    position_units: int | None = Field(default=None, ge=1)
    #: рисковый режим: максимальный убыток на сделку в валюте котировки
    #: (требует стоп-цену в сигнале; размер = risk / (entry - stop) с округлением до лота)
    risk_per_trade_quote: float | None = Field(default=None, gt=0.0)
    #: альтернатива: процент от текущего капитала (взаимоисключительно с _quote)
    risk_per_trade_pct: float | None = Field(default=None, gt=0.0, le=100.0)
    #: максимальная позиция в штуках
    max_position_units: int | None = Field(default=None, ge=0)
    #: максимум заявок (входов) на локальный торговый день
    max_orders_per_day: int | None = Field(default=None, ge=1)
    #: дневной лимит убытка в валюте котировки (положительное число = размер лимита)
    daily_loss_limit_quote: float | None = Field(default=None, gt=0.0)
    #: предел просадки капитала от пика, %
    max_drawdown_pct: float | None = Field(default=None, gt=0.0, le=100.0)
    #: максимальный возраст данных на момент решения, секунды
    max_data_age_seconds: float | None = Field(default=None, gt=0.0)
    #: короткие позиции запрещены по умолчанию (явный флаг)
    allow_short: bool = False
    #: при нарушении дневного лимита/просадки — полная остановка стратегии (True),
    #: а не только отказ очередной заявки
    halt_on_limit_breach: bool = True

    @model_validator(mode="after")
    def _sizing_unambiguous(self) -> "RiskConfig":
        modes = [
            self.position_units is not None,
            self.risk_per_trade_quote is not None,
            self.risk_per_trade_pct is not None,
        ]
        if sum(modes) > 1:
            raise ValueError(
                "Задайте РОВНО ОДИН режим размера позиции: position_units (фиксированный) "
                "или risk_per_trade_quote/risk_per_trade_pct (рисковый). Двусмысленность запрещена"
            )
        return self


class InstrumentConfig(BaseModel):
    """Параметры инструмента для проверок шага цены и лота (фаза 4).

    Значения должны происходить ИЗ ПОДТВЕРЖДЁННОГО источника: спецификация
    инструмента криптобиржи (например tickSize/stepSize в exchangeInfo — когда
    биржевой адаптер будет реализован). Выдумывать параметры запрещено;
    None = параметр не подтверждён — зависимая проверка безопасно откладывается
    и это фиксируется в журнале/отчёте.

    Количество криптопар обычно дробное (stepSize вроде 0.00001 BTC);
    целочисленный lot_size_units — консервативное ограничение текущей схемы,
    вопрос дробных количеств отложен до подключения биржи (docs/decisions.md D14).
    """

    model_config = _STRICT

    #: минимальный шаг цены в квотирующей валюте (tickSize биржи)
    price_step: float | None = Field(default=None, gt=0.0)
    #: минимальный шаг количества в единицах базового актива (stepSize/lotSize
    #: биржи), целым числом; 1 = без дробления (дробные количества — D14, будущее)
    lot_size_units: int | None = Field(default=None, ge=1)
    #: валюта котировки (например USDT) — только для отчётов/журнала
    quote_currency: str | None = Field(default=None, min_length=3, max_length=8)


class SignalConfig(BaseModel):
    """Параметры преобразования модельного прогноза в сигналы (docs/simulation.md §2)."""

    model_config = _STRICT

    #: вход при proba >= entry_threshold (нейтральный дефолт 0.5 — не финансовая величина)
    entry_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    #: выход при proba <= exit_threshold; None = выход только по стопу/цели/концу данных
    exit_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)
    #: стоп/цель в базисных пунктах от цены входа; None = без стопа/цели.
    #: Это ЯВНЫЕ стратегические параметры пользователя, не скрытые допущения.
    stop_bps: float | None = Field(default=None, gt=0.0, le=5000.0)
    target_bps: float | None = Field(default=None, gt=0.0, le=20000.0)

    @model_validator(mode="after")
    def _thresholds_consistent(self) -> "SignalConfig":
        if self.exit_threshold is not None and self.exit_threshold >= self.entry_threshold:
            raise ValueError(
                f"exit_threshold ({self.exit_threshold}) должен быть строго меньше "
                f"entry_threshold ({self.entry_threshold})"
            )
        return self


class LiveConfig(BaseModel):
    """Граница LIVE (docs/live.md). По умолчанию всё выключено.

    ВАЖНО: в крипто-версии проекта биржевой адаптер НЕ реализован
    (exchange/base.py), поэтому LIVE-гейт дополнительно требует факта
    «интеграция с биржей доступна» и сейчас НЕ ПРОХОДИТ НИКОГДА — LIVE
    безопасно заблокирован до отдельной реализации подключения к бирже.

    Для разрешения ордерного контура (после реализации адаптера) потребуется
    ОДНОВРЕМЕННО (многофакторный гейт, без скрытых обходов и fallback):
    1. реализованный и включённый биржевой адаптер (сейчас отсутствует);
    2. run.mode: LIVE в конфиге;
    3. явный флаг CLI --allow-live у команды live-preflight/live;
    4. переменная окружения TRADEBOT_LIVE_ACK со значением-подтверждением;
    5. live.allow_orders: true в отдельной секции;
    6. подтверждённые параметры инструмента (price_step, lot_size_units),
       полный risk-конфиг, live.max_order_quantity_units;
    7. отсутствие файла kill-switch.
    Любое невыполненное условие -> безопасный отказ ДО сетевых вызовов.
    """

    model_config = _STRICT

    #: мастер-переключатель ордерного контура (по умолчанию false)
    allow_orders: bool = False
    #: файл аварийного выключателя: его наличие запрещает новые заявки (architecture §7)
    kill_switch_file: str = ".kill"
    #: максимальный размер одной заявки в штуках — ОБЯЗАТЕЛЬНО для LIVE-заявок,
    #: дефолта нет: без явного значения гейт не проходит
    max_order_quantity_units: int | None = Field(default=None, ge=1)
    #: значение env-переменной-подтверждения (фиксировано, не финансовый параметр)
    ack_env_var: str = "TRADEBOT_LIVE_ACK"
    ack_expected_value: str = "ENABLE-LIVE"


class SimulationConfig(BaseModel):
    """Параметры TEST-симулятора исполнения (docs/simulation.md).

    commission_bps и slippage_bps ОБЯЗАТЕЛЬНЫ для запуска симуляции (проверяется
    при старте движка): выдуманные издержки запрещены (ТЗ фазы 4) — значения
    задаёт пользователь и они печатаются в допущениях отчёта.
    """

    model_config = _STRICT

    #: комиссия, б.п. от оборота сделки (None = симуляция не запускается)
    commission_bps: float | None = Field(default=None, ge=0.0, le=1000.0)
    #: минимальная комиссия на сделку в валюте котировки (None = без минимума)
    commission_min_quote: float | None = Field(default=None, ge=0.0)
    #: проскальзывание, б.п. — применяется НЕВЫГОДНОЙ стороной к каждой симулируемой
    #: цене исполнения (None = симуляция не запускается)
    slippage_bps: float | None = Field(default=None, ge=0.0, le=1000.0)
    #: правило конфликта «стоп и цель в одной свече» (docs/simulation.md §4):
    #: stop_first — консервативное (исполняем стоп); require_intrabar_data —
    #: остановка симуляции с требованием внутрибарных данных. Выгодный сценарий
    #: автоматически НЕ выбирается никогда.
    conflict_policy: Literal["stop_first", "require_intrabar_data"] = "stop_first"
    signal: SignalConfig = Field(default_factory=SignalConfig)
    #: путь JSONL-журнала решений/сделок (None = report_dir/journal_<experiment>_<tf>.jsonl)
    journal_path: str | None = None


class AppConfig(BaseModel):
    model_config = _STRICT

    run: RunConfig
    data: DataConfig
    timeframes: dict[str, TimeframeConfig] = Field(min_length=1)
    sessions: SessionsConfig | None = None
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    #: фаза 4: риск-контроль, параметры инструмента, TEST-симулятор (docs/simulation.md).
    #: Все три секции опциональны для совместимости с конфигами фаз 0-3;
    #: симулятор при их отсутствии не запускается и требует явной конфигурации.
    risk: RiskConfig | None = None
    instrument: InstrumentConfig | None = None
    simulation: SimulationConfig | None = None
    #: граница LIVE (docs/live.md). По умолчанию None — LIVE-контур закрыт;
    #: дополнительно заблокирован отсутствием биржевого адаптера (exchange/base.py).
    live: LiveConfig | None = None

    @field_validator("timeframes")
    @classmethod
    def _timeframes_supported(cls, value: dict[str, TimeframeConfig]) -> dict[str, TimeframeConfig]:
        for key in value:
            if key not in SUPPORTED_TIMEFRAMES:
                raise ValueError(
                    f"Неподдерживаемый таймфрейм {key!r} в секции timeframes; "
                    f"допустимы: {', '.join(SUPPORTED_TIMEFRAMES)}"
                )
        return value

    @model_validator(mode="after")
    def _derived_timeframes_have_base(self) -> "AppConfig":
        """10m обязан иметь base_timeframe (decisions D3);
        для остальных base_timeframe, если задан, должен быть мельче и кратен."""
        tf_minutes = {"1m": 1, "5m": 5, "10m": 10, "15m": 15, "1h": 60}
        for name, cfg in self.timeframes.items():
            if name == "10m" and cfg.base_timeframe is None:
                raise ValueError(
                    "timeframes.'10m': base_timeframe обязателен — нативных 10m-свечей "
                    "источник обычно не предоставляет (docs/decisions.md D3); укажите '5m' или '1m'"
                )
            if cfg.base_timeframe is not None:
                base = tf_minutes[cfg.base_timeframe]
                target = tf_minutes[name]
                if base >= target or target % base != 0:
                    raise ValueError(
                        f"timeframes.{name!r}: base_timeframe {cfg.base_timeframe!r} должен быть "
                        f"мельче и кратен {name!r}"
                    )
        return self

    @model_validator(mode="after")
    def _embargo_covers_horizon(self) -> "AppConfig":
        """Явный embargo_bars должен покрывать горизонт метки каждого включённого ТФ,
        иначе перекрывающиеся целевые окна протекут из train в val (decisions D6).
        embargo_bars=None означает «автоматически = horizon_bars» и всегда корректен."""
        embargo = self.validation.walk_forward.embargo_bars
        if embargo is None:
            return self
        for name, cfg in self.timeframes.items():
            if cfg.enabled and embargo < cfg.horizon_bars:
                raise ValueError(
                    f"validation.walk_forward.embargo_bars={embargo} меньше horizon_bars="
                    f"{cfg.horizon_bars} таймфрейма {name!r}: перекрывающиеся целевые горизонты "
                    "невозможно исключить. Задайте embargo_bars >= horizon_bars или None (авто)"
                )
        return self
