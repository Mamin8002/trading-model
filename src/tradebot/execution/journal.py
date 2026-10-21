"""Журнал решений и сделок (JSONL) — аудит фазы 4 (docs/simulation.md §6).

Каждая строка — одно событие: signal | intent | risk_decision | fill | halt |
warning | state. Обязательные поля каждого события: schema_version, written_at
(UTC), kind, symbol, timeframe, model_version; далее — поля события.

Гарантии:
- журнал позволяет восстановить, ПОЧЕМУ сигнал был исполнен или отклонён
  (risk_decision хранит rejections с кодами/лимитами/наблюдаемыми значениями,
  fill хранит план vs факт: raw_price/price/commission/cause);
- секретов в журнале нет: сериализованная строка проходит через SecretMasker
  (config/env.py), плюс структурное правило — ключи/счета в поля не кладутся
  вообще;
- файл — JSON Lines: потокобезопасное добавление, построчное чтение при аудите.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config.env import GLOBAL_MASKER, SecretMasker

__all__ = ["JOURNAL_SCHEMA_VERSION", "JournalWriter", "audit_line", "read_journal"]

JOURNAL_SCHEMA_VERSION = 1

_KINDS = {"signal", "intent", "risk_decision", "fill", "halt", "warning", "state", "order_event", "reconciliation"}


class JournalWriter:
    def __init__(
        self,
        path: "str | Path",
        *,
        symbol: str,
        timeframe: str,
        model_version: str,
        masker: SecretMasker | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_version = model_version
        self._masker = masker or GLOBAL_MASKER
        self._fh = self.path.open("a", encoding="utf-8")
        self.count = 0

    def record(self, kind: str, **fields: Any) -> dict:
        """Добавляет событие; возвращает записанный словарь (для тестов/отладки)."""
        if kind not in _KINDS:
            raise ValueError(f"Неизвестный вид события журнала: {kind!r} (допустимы: {sorted(_KINDS)})")
        payload: dict[str, Any] = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_version": self.model_version,
        }
        payload.update(fields)
        line = self._masker.mask(json.dumps(payload, ensure_ascii=False, default=str))
        self._fh.write(line + "\n")
        self._fh.flush()
        self.count += 1
        return json.loads(line)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JournalWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_journal(path: "str | Path") -> list[dict]:
    """Читает JSONL-журнал (для аудита и проверок)."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Журнал повреждён в строке {line_no}: {exc}") from exc
    return records


def audit_line(record: dict) -> str:
    """Человекочитаемая строка аудита одного события."""
    kind = record.get("kind", "?")
    ts = record.get("event_ts") or record.get("written_at", "")
    symbol = record.get("symbol", "")
    if kind == "risk_decision":
        status = "APPROVED" if record.get("approved") else "REJECTED"
        reasons = ""
        if not record.get("approved"):
            reasons = " | " + "; ".join(
                f"{r.get('code')}: {r.get('message')}" for r in record.get("rejections", [])
            )
        return (
            f"[{ts}] {symbol} {kind.upper()} {status} side={record.get('side')} "
            f"qty={record.get('quantity_units')} model={record.get('model_version')}{reasons}"
        )
    if kind == "fill":
        return (
            f"[{ts}] {symbol} FILL {record.get('side')} {record.get('units')} шт "
            f"по {record.get('price')} (raw {record.get('raw_price')}, slip {record.get('slippage_bps')} б.п.) "
            f"комиссия {record.get('commission_quote')} причина {record.get('cause')}"
        )
    if kind == "halt":
        return f"[{ts}] {symbol} HALT reason={record.get('reason')} detail={record.get('detail', '')}"
    if kind == "warning":
        return f"[{ts}] {symbol} WARNING {record.get('message', '')}"
    if kind == "intent":
        return (
            f"[{ts}] {symbol} INTENT {record.get('side')} qty={record.get('quantity_units')} "
            f"ref_price={record.get('reference_price')} (справочно, не цена исполнения)"
        )
    if kind == "signal":
        return (
            f"[{ts}] {symbol} SIGNAL side={record.get('side')} proba={record.get('proba')} "
            f"model={record.get('model_version')}"
        )
    return f"[{ts}] {symbol} {kind.upper()} " + json.dumps(
        {k: v for k, v in record.items() if k not in ("schema_version", "written_at", "kind", "symbol", "timeframe", "model_version")},
        ensure_ascii=False,
        default=str,
    )


def iter_audit(path: "str | Path") -> Iterator[str]:
    for record in read_journal(path):
        yield audit_line(record)
