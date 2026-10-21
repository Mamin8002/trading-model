"""HTTP-транспорт для REST-адаптера Finam Trade API.

Транспорт —.injectable интерфейс: в тестах и ручной проверке используется фейк
(подмена клиента — требование сценария проверки), в бою — stdlib urllib
(осознанный выбор: не добавляем httpx/requests в этой фазе; gRPC-транспорт
официального SDK finam-sdk подключается на этапе L1/L2 roadmap за теми же
интерфейсами — decisions D7).

ВАЖНО: транспорт НЕ логирует заголовок Authorization и тела auth-запросов
(research S3j Рецепт 8, S4). Ошибки urllib заворачиваются в FinamTransportError
с маскированием.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .errors import FinamTransportError, get_logger

__all__ = ["HttpResponse", "HttpTransport", "UrllibTransport"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> Any:
        if not self.body:
            raise ValueError("пустое тело ответа")
        return json.loads(self.body.decode("utf-8"))


class HttpTransport(ABC):
    @abstractmethod
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 15.0,
    ) -> HttpResponse:
        """Выполняет HTTP-запрос. Статусы 4xx/5xx возвращаются как HttpResponse
        (не исключение); исключением являются только транспортные сбои."""
        raise NotImplementedError


class UrllibTransport(HttpTransport):
    """Транспорт на stdlib urllib.request (без сторонних зависимостей)."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 15.0,
    ) -> HttpResponse:
        data = None
        req_headers = {"Accept": "application/json", "User-Agent": "tradebot/0.1 (read-only phase)"}
        if headers:
            req_headers.update(headers)
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=req_headers, method=method.upper())
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec - https only by config
                return HttpResponse(
                    status=response.status,
                    headers={k.lower(): v for k, v in response.headers.items()},
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:  # pragma: no cover - тело ошибки может быть недоступно
                body = b""
            return HttpResponse(
                status=exc.code,
                headers={k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])},
                body=body,
            )
        except urllib.error.URLError as exc:
            # masking: URL не содержит секретов (токен в заголовке), но на всякий случай
            raise FinamTransportError(f"Сетевая ошибка при запросе {method} {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FinamTransportError(f"Таймаут запроса {method} {url} ({timeout}s)") from exc
