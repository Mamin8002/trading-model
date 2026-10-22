"""Application configuration: dataclasses + YAML loading + validation.

Конфигурация не содержит секретов: бот использует только публичные
эндпоинты Bybit (REST /v5/market/*, WS public spot).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

VALID_MODES = ("TEST", "LIVE")
VALID_CATEGORIES = ("spot", "linear", "inverse")
VALID_INTERVALS = ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")

_INTERVAL_MINUTES = {
    "1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60,
    "120": 120, "240": 240, "360": 360, "720": 720,
}


class ConfigError(ValueError):
    """Некорректная конфигурация."""


@dataclass
class ExchangeCfg:
    source: str = "bybit"
    category: str = "spot"
    symbol: str = "BTCUSDT"
    interval: str = "5"
    rest_base_url: str = "https://api.bybit.com"
    ws_public_url: str = "wss://stream.bybit.com/v5/public/spot"
    request_timeout_s: float = 15.0
    min_request_interval_s: float = 0.35
    max_retries: int = 5

    @property
    def interval_minutes(self) -> int:
        """Длительность интервала в минутах (для D/W/M - приблизительная, в днях*1440)."""
        if self.interval in _INTERVAL_MINUTES:
            return _INTERVAL_MINUTES[self.interval]
        if self.interval == "D":
            return 1440
        if self.interval == "W":
            return 7 * 1440
        if self.interval == "M":
            return 30 * 1440
        raise ConfigError(f"Неизвестный interval: {self.interval!r}")

    @property
    def interval_ms(self) -> int:
        return self.interval_minutes * 60_000


@dataclass
class ModelCfg:
    algorithm: str = "xgboost"
    horizon_bars: int = 3          # 3 бара x 5 мин = 15 мин (диапазон задачи: 2-3 = 10-15 мин)
    confidence_threshold: Optional[float] = None  # None = порог из model card (выбранный без утечки);
                                                  # число = ручное переопределение
    random_seed: int = 42
    # "auto"  - сравнить ratio и v2 в walk-forward и применить pre-registered adopt-gate;
    # "raw"   - признаки как в исходном ноутбуке (сырые OHLCV+turnover);
    # "ratio" - v1: цены относительно close бара (стационарнее);
    # "v2"    - лаг-доходности, форма бара, объём, время суток (только закрытые бары <= t).
    feature_set: str = "auto"
    # гиперпараметры XGBoost (дефолты как в исходном ноутбуке)
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    # каталог артефакта модели; None - брать последний ПРИНЯТЫЙ (adopted) из data.models_dir
    artifact_dir: Optional[str] = None

    @property
    def horizon_minutes(self) -> int:
        return self.horizon_bars * 5  # уточняется AppConfig.horizon_minutes с учётом interval


@dataclass
class TradingCfg:
    """Издержки и размер для симуляции/paper trading. Задаются ЯВНО (спецификация Этапа 5)."""
    fee_pct: float = 0.1          # комиссия за сторону, % от цены (Bybit spot taker)
    slippage_pct: float = 0.02    # проскальзывание за сторону, %
    order_size_quote: float = 100.0  # виртуальный объём сделки в котировочной валюте (USDT)


@dataclass
class DataCfg:
    history_start: str = "2024-01-01"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "data/models"


@dataclass
class SignalsCfg:
    stale_after_intervals: int = 3
    history_display_bars: int = 10


@dataclass
class StreamCfg:
    ping_interval_s: int = 20          # рекомендация Bybit: app-level ping каждые 20 с
    connect_timeout_s: float = 10.0
    recv_timeout_s: float = 1.0
    reconnect_base_s: float = 1.0      # backoff: 1, 2, 4, ... <= max
    reconnect_max_s: float = 60.0      # <= 500 соединений/5мин - не частим
    backfill_on_reconnect: bool = True # догрузка пропущенных свечей через публичный REST
    history_bars: int = 300            # сколько закрытых баров держать в памяти


@dataclass
class ExitRulesCfg:
    """Явно задаваемые условия выхода. None = правило НЕ задано (бот его не изобретает)."""
    take_profit_pct: Optional[float] = None   # % роста от цены входа -> рекомендация закрыть
    stop_loss_pct: Optional[float] = None     # % снижения от цены входа -> рекомендация закрыть
    max_hold_bars: Optional[int] = None       # макс. число баров удержания -> рекомендация закрыть


@dataclass
class PositionsCfg:
    exit_rules: ExitRulesCfg = field(default_factory=ExitRulesCfg)
    allow_short: bool = False   # spot: SELL НЕ открывает шорт; регистрация SHORT запрещена


@dataclass
class UiCfg:
    mode: str = "auto"       # auto | dash | line | block
    log_events: bool = True  # line/block: печатать значимые события отдельными строками
    refresh_s: float = 1.0   # частота обновления статусной строки/дашборда
    colors: str = "auto"     # auto | on | off


@dataclass
class IntrabarCfg:
    """Внутрисвечной анализ по потоку сделок (publicTrade). Spike = АЛЕРТ, не сигнал входа."""
    subscribe_trades: bool = True   # подписка на publicTrade.{symbol} в LIVE
    window_s: int = 60              # скользящее окно для оценки быстрого движения
    spike_pct: float = 0.15         # движение >= X% за окно -> алерт "быстрое движение"
    cooldown_s: int = 300           # не повторять алерт чаще, чем раз в N с


@dataclass
class PartialTpCfg:
    enabled: bool = False
    at_pct: Optional[float] = None   # уровень прибыли % для частичного закрытия
    fraction: float = 0.5            # доля позиции для частичного закрытия


@dataclass
class PaperCfg:
    """Paper trading: виртуальные позиции БЕЗ отправки ордеров и без приватного API."""
    enabled: bool = True             # в LIVE; в TEST-реплее включается флагом --paper
    initial_balance_quote: float = 10_000.0
    entry_min_confidence: Optional[float] = None  # None = порог модели (из card/конфига)
    cooldown_bars: int = 1           # пауза после закрытия (в барах)
    max_open_positions: int = 1      # конфликтующие позиции не открываются
    intrabar_exits: bool = True      # SL/TP по тикам в LIVE (в TEST/реплее - по барам)
    exit_at_horizon: bool = True     # закрывать по истечении горизонта прогноза (H баров)
    partial_tp: PartialTpCfg = field(default_factory=PartialTpCfg)


@dataclass
class JournalCfg:
    path: str = "data/journal/journal.json"


@dataclass
class AppConfig:
    mode: str = "TEST"
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    signals: SignalsCfg = field(default_factory=SignalsCfg)
    stream: StreamCfg = field(default_factory=StreamCfg)
    positions: PositionsCfg = field(default_factory=PositionsCfg)
    trading: TradingCfg = field(default_factory=TradingCfg)
    intrabar: IntrabarCfg = field(default_factory=IntrabarCfg)
    paper: PaperCfg = field(default_factory=PaperCfg)
    ui: UiCfg = field(default_factory=UiCfg)
    journal: JournalCfg = field(default_factory=JournalCfg)
    project_root: Path = PROJECT_ROOT
    config_path: Optional[Path] = None

    # -- helpers --------------------------------------------------------
    def resolve(self, relative: str | Path) -> Path:
        """Разрешить относительный путь относительно корня проекта."""
        p = Path(relative)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def horizon_minutes(self) -> int:
        """Горизонт прогноза в минутах = horizon_bars * длительность интервала."""
        return self.model.horizon_bars * self.exchange.interval_minutes

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ConfigError(f"mode должен быть одним из {VALID_MODES}, получено {self.mode!r}")
        if self.exchange.category not in VALID_CATEGORIES:
            raise ConfigError(f"category должен быть одним из {VALID_CATEGORIES}")
        if self.exchange.interval not in VALID_INTERVALS:
            raise ConfigError(f"interval должен быть одним из {VALID_INTERVALS}")
        if not self.exchange.symbol or not self.exchange.symbol.isupper():
            raise ConfigError("symbol должен быть в верхнем регистре (например BTCUSDT)")
        if self.model.horizon_bars < 1 or not isinstance(self.model.horizon_bars, int):
            raise ConfigError("model.horizon_bars должен быть целым >= 1")
        if not (self.model.confidence_threshold is None
                or (0.5 < self.model.confidence_threshold <= 1.0)):
            raise ConfigError("model.confidence_threshold должен быть null или в (0.5, 1.0]")
        if self.model.feature_set not in ("auto", "raw", "ratio", "v2"):
            raise ConfigError("model.feature_set должен быть auto|raw|ratio|v2")
        if self.model.n_estimators < 1 or self.model.max_depth < 1 or self.model.learning_rate <= 0:
            raise ConfigError("model.n_estimators/max_depth/learning_rate должны быть положительными")
        if not (0 <= self.trading.fee_pct < 5 and 0 <= self.trading.slippage_pct < 5
                and self.trading.order_size_quote > 0):
            raise ConfigError("trading: fee_pct/slippage_pct в [0,5), order_size_quote > 0")
        ib = self.intrabar
        if ib.window_s < 1 or ib.spike_pct <= 0 or ib.cooldown_s < 0:
            raise ConfigError("intrabar: window_s >= 1, spike_pct > 0, cooldown_s >= 0")
        pp = self.paper
        if pp.initial_balance_quote <= 0 or pp.cooldown_bars < 0 or pp.max_open_positions < 1:
            raise ConfigError("paper: initial_balance_quote > 0, cooldown_bars >= 0, max_open_positions >= 1")
        if pp.entry_min_confidence is not None and not (0.5 < pp.entry_min_confidence <= 1.0):
            raise ConfigError("paper.entry_min_confidence должен быть null или в (0.5, 1.0]")
        if pp.partial_tp.enabled:
            if pp.partial_tp.at_pct is None or pp.partial_tp.at_pct <= 0:
                raise ConfigError("paper.partial_tp.at_pct обязателен и > 0 при enabled=true")
            if not (0.0 < pp.partial_tp.fraction < 1.0):
                raise ConfigError("paper.partial_tp.fraction должен быть в (0, 1)")
        if self.signals.stale_after_intervals < 1:
            raise ConfigError("signals.stale_after_intervals должен быть >= 1")
        if self.stream.ping_interval_s < 1 or self.stream.reconnect_base_s <= 0 or self.stream.reconnect_max_s < self.stream.reconnect_base_s:
            raise ConfigError("stream: некорректные параметры ping/reconnect")
        if self.ui.mode not in ("auto", "dash", "line", "block"):
            raise ConfigError("ui.mode должен быть auto|dash|line|block")
        if self.ui.colors not in ("auto", "on", "off"):
            raise ConfigError("ui.colors должен быть auto|on|off")
        if self.ui.refresh_s <= 0:
            raise ConfigError("ui.refresh_s должен быть > 0")
        er = self.positions.exit_rules
        for name, val in (("take_profit_pct", er.take_profit_pct), ("stop_loss_pct", er.stop_loss_pct)):
            if val is not None and (not isinstance(val, (int, float)) or val <= 0):
                raise ConfigError(f"positions.exit_rules.{name} должен быть null или > 0")
        if er.max_hold_bars is not None and (not isinstance(er.max_hold_bars, int) or er.max_hold_bars < 1):
            raise ConfigError("positions.exit_rules.max_hold_bars должен быть null или целым >= 1")
        if self.positions.allow_short:
            raise ConfigError(
                "positions.allow_short=true не поддерживается: бот работает со spot без shorts "
                "(SELL = закрыть/уменьшить длинную позицию)"
            )
        if self.model.algorithm not in ("xgboost",):
            # на этапе 2 поддерживается только xgboost (выбор исходного репозитория)
            raise ConfigError(f"algorithm {self.model.algorithm!r} пока не поддерживается")


_SUBCFG = {
    "exchange": ExchangeCfg,
    "model": ModelCfg,
    "data": DataCfg,
    "signals": SignalsCfg,
    "stream": StreamCfg,
    "positions": PositionsCfg,
    "trading": TradingCfg,
    "intrabar": IntrabarCfg,
    "paper": PaperCfg,
    "ui": UiCfg,
    "journal": JournalCfg,
}

_NESTED_CFG = {
    "positions": {"exit_rules": ExitRulesCfg},
    "paper": {"partial_tp": PartialTpCfg},
}


def _cfg_from_dict(cls, d: dict[str, Any], nested: Optional[dict] = None):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"Неизвестные ключи в секции {cls.__name__}: {sorted(unknown)}")
    kwargs = dict(d)
    for key, subcls in (nested or {}).items():
        if key in kwargs:
            if not isinstance(kwargs[key], dict):
                raise ConfigError(f"{cls.__name__}.{key} должен быть словарём")
            kwargs[key] = _cfg_from_dict(subcls, kwargs[key])
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: Optional[dict] = None) -> AppConfig:
    """Загрузить конфигурацию из YAML (по умолчанию config/default.yaml).

    overrides - словарь в той же структуре, что и YAML (слияние поверх файла).
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Конфиг {cfg_path} должен быть YAML-словарём")
        raw = loaded
    else:
        if path is not None:
            raise ConfigError(f"Конфиг не найден: {cfg_path}")
    if overrides:
        raw = _deep_merge(raw, overrides)

    top_known = {f.name for f in fields(AppConfig)} - {"project_root", "config_path"}
    unknown_top = set(raw) - top_known
    if unknown_top:
        raise ConfigError(f"Неизвестные секции в конфиге: {sorted(unknown_top)}")

    kwargs: dict[str, Any] = {}
    for key, cls in _SUBCFG.items():
        section = raw.get(key, {}) or {}
        if not isinstance(section, dict):
            raise ConfigError(f"Секция {key} должна быть словарём")
        kwargs[key] = _cfg_from_dict(cls, section, nested=_NESTED_CFG.get(key))
    if "mode" in raw:
        kwargs["mode"] = str(raw["mode"]).upper()

    cfg = AppConfig(config_path=cfg_path, **kwargs)
    cfg.validate()
    return cfg
