"""Клиентский rate-limiter: 200 запросов/минуту НА МЕТОД (research §2.7, S4).

Скользящее окно; при исчерпании лимита — ожидание до освобождения окна
(не немедленный повтор: документация прямо запрещает «атаковать снова» на 429).
Функции времени/сна injectable для детерминированных тестов.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict, deque
from typing import Callable

__all__ = ["SlidingWindowLimiter"]


class SlidingWindowLimiter:
    def __init__(
        self,
        limit_per_minute: int = 200,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        jitter_seconds: float = 0.25,
        rng: random.Random | None = None,
    ) -> None:
        if limit_per_minute < 1:
            raise ValueError("limit_per_minute должен быть >= 1")
        if limit_per_minute > 200:
            # Документированный лимит API — 200/мин на метод; выше не поднимаем.
            raise ValueError(
                "limit_per_minute не может превышать документированный лимит API (200/мин на метод)"
            )
        self.limit = limit_per_minute
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter_seconds
        self._rng = rng or random.Random()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def acquire(self, method_key: str) -> float:
        """Блокирующее приобретение слота. Возвращает суммарное время ожидания (с)."""
        window = self._hits[method_key]
        now = self._clock()
        total_waited = 0.0
        iterations = 0
        while True:
            cutoff = now - 60.0
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) < self.limit:
                window.append(now)
                return total_waited
            iterations += 1
            if iterations > 1000:
                # Защита от патологического случая: sleep не продвигает часы
                # (например, тестовая заглушка). Бесконечное ожидание недопустимо.
                raise RuntimeError(
                    "SlidingWindowLimiter: часы не продвигаются после 1000 итераций ожидания "
                    f"(method_key={method_key}, limit={self.limit})"
                )
            wait = window[0] + 60.0 - now + self._rng.uniform(0.0, self._jitter)
            wait = max(wait, 0.001)
            self._sleep(wait)
            total_waited += wait
            now = self._clock()

    def note_external(self, method_key: str) -> None:
        """Учитывает запрос, выполненный вне limiter-а (например, ретрай после 429)."""
        self._hits[method_key].append(self._clock())
