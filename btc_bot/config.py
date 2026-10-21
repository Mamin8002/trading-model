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
    horizon_bars: int = 1
    confidence_threshold: float = 0.55
    random_seed: int = 42
    # "auto" - сравнить raw и ratio в walk-forward и выбрать лучший (фиксируется в model_card);
    # "raw"  - признаки как в исходном ноутбуке (сырые OHLCV+turnover);
    # "ratio"- минимально необходимая модификация: цены относительно close бара (стационарнее).
    feature_set: str = "auto"
    # гиперпараметры XGBoost (дефолты как в исходном ноутбуке)
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.3
    # каталог артефакта модели; None - брать последний по дате из data.models_dir
    artifact_dir: Optional[str] = None

    @property
    def horizon_minutes(self) -> int:
        return self.horizon_bars * 5  # уточняется AppConfig.horizon_minutes с учётом interval


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
        if not (0.5 < self.model.confidence_threshold <= 1.0):
            raise ConfigError("model.confidence_threshold должен быть в (0.5, 1.0]")
        if self.model.feature_set not in ("auto", "raw", "ratio"):
            raise ConfigError("model.feature_set должен быть auto|raw|ratio")
        if self.model.n_estimators < 1 or self.model.max_depth < 1 or self.model.learning_rate <= 0:
            raise ConfigError("model.n_estimators/max_depth/learning_rate должны быть положительными")
        if self.signals.stale_after_intervals < 1:
            raise ConfigError("signals.stale_after_intervals должен быть >= 1")
        if self.stream.ping_interval_s < 1 or self.stream.reconnect_base_s <= 0 or self.stream.reconnect_max_s < self.stream.reconnect_base_s:
            raise ConfigError("stream: некорректные параметры ping/reconnect")
        if self.model.algorithm not in ("xgboost",):
            # на этапе 2 поддерживается только xgboost (выбор исходного репозитория)
            raise ConfigError(f"algorithm {self.model.algorithm!r} пока не поддерживается")


_SUBCFG = {
    "exchange": ExchangeCfg,
    "model": ModelCfg,
    "data": DataCfg,
    "signals": SignalsCfg,
    "stream": StreamCfg,
    "journal": JournalCfg,
}


def _cfg_from_dict(cls, d: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"Неизвестные ключи в секции {cls.__name__}: {sorted(unknown)}")
    return cls(**d)


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
        kwargs[key] = _cfg_from_dict(cls, section)
    if "mode" in raw:
        kwargs["mode"] = str(raw["mode"]).upper()

    cfg = AppConfig(config_path=cfg_path, **kwargs)
    cfg.validate()
    return cfg
