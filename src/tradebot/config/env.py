"""Безопасная работа с секретами: окружение, опциональный .env, маскирование.

Правила (docs/research.md S3b, S3j Рецепт 8, S4 «Чего не делать»):
- secret_key и account_id читаются ТОЛЬКО из переменных окружения (или из
  игнорируемого git-ом файла .env, который никогда не коммитится);
- значения секретов не попадают в сообщения об ошибках и в логи: любое сообщение
  перед выводом проходит через SecretMasker;
- .env.example содержит только имена переменных и фиктивные заглушки.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

__all__ = [
    "SecretMasker",
    "MaskingFilter",
    "SecretNotFound",
    "load_dotenv_file",
    "resolve_secret",
    "resolve_optional",
]


class SecretNotFound(RuntimeError):
    """Секрет не задан в окружении. Сообщение не содержит значения."""


class SecretMasker:
    """Реестр секретных значений для маскирования в тексте/логах."""

    #: значения короче этой длины не регистрируются: одно-двухсимвольные «секреты»
    #: маскировали бы обычный текст (логи/журналы) до нечитаемости. Реальные
    #: secret_key/access_key значительно длиннее.
    MIN_REGISTERED_LENGTH = 4

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def register(self, value: str, label: str = "SECRET") -> None:
        if value and len(value) >= self.MIN_REGISTERED_LENGTH:
            self._values[value] = f"***{label}***"

    def mask(self, text: str) -> str:
        for value, replacement in self._values.items():
            if value and value in text:
                text = text.replace(value, replacement)
        return text


class MaskingFilter(logging.Filter):
    """Фильтр логов: маскирует зарегистрированные секреты в итоговом сообщении."""

    def __init__(self, masker: SecretMasker) -> None:
        super().__init__()
        self.masker = masker

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.masker.mask(record.getMessage())
        record.args = None
        return True


#: Процессный маскировщик: адаптеры регистрируют в нём secret/access_key,
#: а логгер "tradebot" фильтруется через него (подключение — в finam/errors.py).
GLOBAL_MASKER = SecretMasker()


def load_dotenv_file(path: "str | Path" = ".env", *, override: bool = False) -> int:
    """Минимальный парсер KEY=VALUE (без сторонних зависимостей).

    - строки начинающиеся с '#' и пустые — игнорируются;
    - кавычки (одинарные/двойные) снимаются;
    - существующие переменные окружения НЕ перезаписываются (override=False),
      т.е. реальное окружение всегда приоритетнее файла;
    - возвращает число загруженных переменных.

    Файл .env должен быть исключён из git (.gitignore). Никогда не указывайте
    здесь секретов, которые попадут в репозиторий.
    """
    env_path = Path(path)
    if not env_path.exists():
        return 0
    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
            loaded += 1
    return loaded


def resolve_secret(env_var: str, masker: SecretMasker | None = None, label: str = "SECRET") -> str:
    """Читает обязательный секрет из окружения и регистрирует его в маскировщике.

    Ошибка при отсутствии переменной НЕ содержит значения (его и нет) и печатает
    только имя переменной и инструкцию.
    """
    value = os.environ.get(env_var, "").strip()
    if not value:
        raise SecretNotFound(
            f"Переменная окружения {env_var} не задана или пуста. "
            "Скопируйте .env.example в .env (файл .env игнорируется git-ом) или "
            f"установите переменную в сессии: $env:{env_var}='...'"
        )
    target = masker if masker is not None else GLOBAL_MASKER
    target.register(value, label=label)
    return value


def resolve_optional(env_var: str, masker: SecretMasker | None = None, label: str = "SECRET") -> str | None:
    """Как resolve_secret, но отсутствие переменной — не ошибка (возвращает None)."""
    value = os.environ.get(env_var, "").strip()
    if not value:
        return None
    target = masker if masker is not None else GLOBAL_MASKER
    target.register(value, label=label)
    return value
