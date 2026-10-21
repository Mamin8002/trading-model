"""Загрузка конфигурации из YAML и служебные операции (хэш, календарь сессий)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ..data.sessions import DayRule, SessionCalendar, SessionWindow
from .schema import AppConfig

__all__ = ["ConfigError", "config_hash", "load_config", "build_calendar"]


class ConfigError(Exception):
    """Конфигурация не прошла загрузку/валидацию. Сообщение безопасно для логов."""


def load_config(path: "str | Path") -> AppConfig:
    """Читает YAML и валидирует строгой схемой. Любая ошибка — ConfigError с причиной."""
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigError(f"Файл конфигурации не найден: {file_path}")
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Не удалось разобрать YAML {file_path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"Конфигурация пуста: {file_path}")
    if not isinstance(raw, dict):
        raise ConfigError(f"Конфигурация должна быть словарём верхнего уровня, получено {type(raw).__name__}")
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError и производные
        raise ConfigError(f"Конфигурация {file_path} не прошла валидацию:\n{_format_validation(exc)}") from exc


def _format_validation(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        lines = []
        for err in errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            lines.append(f"  - {loc or '<root>'}: {err.get('msg', err)}")
        return "\n".join(lines)
    return str(exc)


def config_hash(config: AppConfig) -> str:
    """Детерминированный sha256-хэш нормализованного конфига.

    Используется в именах артефактов (architecture §3.2): одинаковый конфиг ->
    одинаковый хэш независимо от порядка ключей в YAML.
    """
    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_calendar(config: AppConfig) -> SessionCalendar | None:
    """Строит SessionCalendar из конфига; None, если секция sessions не задана."""
    if config.sessions is None:
        return None
    day_rules = []
    for day_cfg in config.sessions.days:
        windows = tuple(
            SessionWindow.parse(window.start, window.end) for window in day_cfg.sessions
        )
        day_rules.append(DayRule(frozenset(day_cfg.days), windows))
    return SessionCalendar(config.sessions.timezone, day_rules)
