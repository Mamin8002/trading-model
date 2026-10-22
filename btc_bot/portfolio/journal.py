"""Журнал: история сигналов модели + ручные позиции пользователя (JSON, атомарная запись).

Принципы (по спецификации):
  * сигналы модели и фактические позиции РАЗДЕЛЕНЫ: без приватного API бот не знает,
    исполнил ли пользователь сигнал и по какой цене;
  * сигнал НЕ считается исполненным, пока пользователь вручную не зарегистрировал вход
    (register_entry с указанием signal_id связывает их и помечает подтверждение);
  * каждая мутация: reload -> change -> atomic save (журнал переживает перезапуск и
    не боится параллельного CLI);
  * фиксируются: сигналы с ценой на момент выдачи, подтверждения входа/выхода,
    причины смены статуса, результат позиции;
  * для spot BTCUSDT direction может быть только LONG: SELL - это рекомендация
    закрыть/уменьшить длинную позицию, а НЕ открытие короткой.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1
QTY_EPS = 1e-12


class JournalError(RuntimeError):
    pass


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _utc_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class PositionView:
    """Позиция "как есть" (словарь журнала) + удобные вычисления."""
    data: dict

    @property
    def position_id(self) -> str:
        return self.data["position_id"]

    @property
    def status(self) -> str:
        return self.data["status"]

    @property
    def direction(self) -> str:
        return self.data["direction"]

    @property
    def symbol(self) -> str:
        return self.data["symbol"]

    @property
    def opened_at_ms(self) -> int:
        return int(self.data["opened_at_ms"])

    @property
    def quantity_open(self) -> float:
        return float(self.data["quantity_open"])

    @property
    def entry_price(self) -> float:
        return float(self.data["entry_price"])

    @property
    def exit_rules(self) -> Optional[dict]:
        return self.data.get("exit_rules")

    def unrealized_pct(self, price_now: float) -> float:
        return (price_now - self.entry_price) / self.entry_price * 100.0


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save(self._empty())

    # -- persistence -----------------------------------------------------
    def _empty(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_iso(_now_ms()),
            "signals": [],
            "positions": [],
            "paper_positions": [],   # виртуальные позиции бота (НЕ смешиваются с ручными)
            "paper_actions": [],     # лог действий paper trading
            "events": [],
            "counters": {"signal": 0, "position": 0, "paper_position": 0},
        }

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            # НЕ теряем и НЕ перезаписываем данные молча: файл остаётся как есть
            raise JournalError(
                f"Журнал {self.path} повреждён ({e}). Файл не изменён - восстановите его "
                f"вручную или переместите и создайте новый (потеря истории недопустима молча)."
            )
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            raise JournalError(f"Несовместимая схема журнала: {data.get('schema_version')}")
        # обратная совместимость со старыми журналами (до Этапа 5b)
        data.setdefault("paper_positions", [])
        data.setdefault("paper_actions", [])
        data.setdefault("counters", {}).setdefault("paper_position", 0)
        return data

    def _save(self, data: dict) -> None:
        data["updated_at_utc"] = _utc_iso(_now_ms())
        text = json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _append_event(self, data: dict, ts_ms: int, kind: str, details: dict) -> None:
        data["events"].append({"ts_ms": ts_ms, "ts_utc": _utc_iso(ts_ms), "kind": kind, **details})

    def record_advice(self, advice: dict, signal_id: Optional[str] = None,
                      ts_ms: Optional[int] = None) -> None:
        """Зафиксировать выданную рекомендацию советника (event log)."""
        data = self._load()
        self._append_event(data, int(ts_ms or advice.get("ts_ms") or _now_ms()),
                           "advice_issued", {"signal_id": signal_id, "advice": advice})
        self._save(data)

    # -- signals -------------------------------------------------------------
    def record_signal(self, signal) -> dict:
        """Записать сигнал модели. signal: Signal dataclass или dict."""
        sd = signal.to_dict() if hasattr(signal, "to_dict") else dict(signal)
        ts = int(sd.get("created_at_ms") or _now_ms())
        data = self._load()
        entry = {
            **sd,
            "entry_confirmed": False,   # станет True только после ручной регистрации входа
            "position_id": None,
            "recorded_at_utc": _utc_iso(_now_ms()),
        }
        data["counters"]["signal"] += 1
        entry["journal_seq"] = data["counters"]["signal"]
        data["signals"].append(entry)
        self._append_event(data, ts, "signal_recorded",
                           {"signal_id": entry.get("signal_id"), "action": entry.get("action"),
                            "confidence": entry.get("confidence"),
                            "price_at_signal": entry.get("price_at_signal"),
                            "data_fresh": entry.get("data_fresh"),
                            "model_version": entry.get("model_version")})
        self._save(data)
        return entry

    def last_signals(self, n: int = 20) -> list[dict]:
        return self._load()["signals"][-n:]

    def find_signal(self, signal_id: str) -> Optional[dict]:
        for s in self._load()["signals"]:
            if s.get("signal_id") == signal_id:
                return s
        return None

    def consecutive_same_action(self, action: str) -> int:
        """Сколько последних сигналов подряд имеют такое же действие (учёт истории)."""
        n = 0
        for s in reversed(self._load()["signals"]):
            if s.get("action") == action:
                n += 1
            else:
                break
        return n

    # -- positions -------------------------------------------------------------
    def register_entry(
        self,
        *,
        symbol: str,
        category: str,
        direction: str,
        price: float,
        quantity: float,
        opened_at_ms: Optional[int] = None,
        signal_id: Optional[str] = None,
        exit_rules: Optional[dict] = None,
        note: str = "",
        allow_short: bool = False,
    ) -> PositionView:
        """Ручная регистрация фактического входа пользователя."""
        direction = direction.upper()
        if direction not in ("LONG", "SHORT"):
            raise JournalError(f"direction должен быть LONG или SHORT, получено {direction!r}")
        if direction == "SHORT" and not allow_short:
            raise JournalError(
                "Регистрация SHORT запрещена: для spot SELL означает закрыть/уменьшить длинную "
                "позицию, короткие позиции ботом не поддерживаются."
            )
        if price <= 0 or quantity <= 0:
            raise JournalError("price и quantity должны быть > 0")

        ts = _now_ms()
        opened_at = int(opened_at_ms if opened_at_ms is not None else ts)
        data = self._load()

        source_signal = None
        if signal_id:
            sig = next((s for s in data["signals"] if s.get("signal_id") == signal_id), None)
            if sig is None:
                raise JournalError(f"Сигнал {signal_id!r} не найден в журнале")
            if sig.get("entry_confirmed"):
                raise JournalError(f"Сигнал {signal_id!r} уже подтверждён входом позиции {sig.get('position_id')}")
            source_signal = {
                "signal_id": sig.get("signal_id"),
                "action": sig.get("action"),
                "confidence": sig.get("confidence"),
                "signal_created_at_ms": sig.get("created_at_ms"),
                "bar_end_ms": sig.get("bar_end_ms"),
                "price_at_signal": sig.get("price_at_signal"),
                "horizon_bars": sig.get("horizon_bars"),
                "horizon_minutes": sig.get("horizon_minutes"),
                "model_version": sig.get("model_version"),
            }

        data["counters"]["position"] += 1
        pid = f"pos-{data['counters']['position']:06d}"
        pos = {
            "position_id": pid,
            "symbol": symbol,
            "category": category,
            "direction": direction,
            "status": "OPEN",
            "opened_at_ms": opened_at,
            "opened_at_utc": _utc_iso(opened_at),
            "entry_price": float(price),
            "quantity_initial": float(quantity),
            "quantity_open": float(quantity),
            "source_signal": source_signal,
            "exit_rules": exit_rules,
            "note": note,
            "fills": [{
                "ts_ms": opened_at, "ts_utc": _utc_iso(opened_at), "type": "entry",
                "price": float(price), "quantity": float(quantity), "note": note,
            }],
            "status_history": [{
                "ts_ms": ts, "ts_utc": _utc_iso(ts), "event": "OPENED",
                "reason": f"Ручная регистрация входа: {quantity} @ {price}"
                          + (f" (подтверждение сигнала {signal_id})" if signal_id else " (без привязки к сигналу)"),
            }],
            "closed_at_ms": None,
            "avg_exit_price": None,
            "realized_pnl_quote": None,
            "realized_pnl_pct": None,
            "outcome": None,
        }
        data["positions"].append(pos)
        if signal_id:
            for s in data["signals"]:
                if s.get("signal_id") == signal_id:
                    s["entry_confirmed"] = True   # сигнал исполнен ТОЛЬКО после ручной регистрации
                    s["position_id"] = pid
                    break
        self._append_event(data, ts, "entry_registered",
                           {"position_id": pid, "symbol": symbol, "direction": direction,
                            "price": float(price), "quantity": float(quantity),
                            "signal_id": signal_id, "exit_rules": exit_rules})
        self._save(data)
        return PositionView(pos)

    def close_position(
        self,
        position_id: str,
        *,
        price: float,
        quantity: Optional[float] = None,
        closed_at_ms: Optional[int] = None,
        note: str = "",
    ) -> PositionView:
        """Полное или частичное закрытие позиции."""
        if price <= 0:
            raise JournalError("price должен быть > 0")
        ts = _now_ms()
        closed_at = int(closed_at_ms if closed_at_ms is not None else ts)
        data = self._load()
        pos = next((p for p in data["positions"] if p["position_id"] == position_id), None)
        if pos is None:
            raise JournalError(f"Позиция {position_id!r} не найдена")
        if pos["status"] != "OPEN":
            raise JournalError(f"Позиция {position_id} уже закрыта (статус {pos['status']})")

        qty_open = float(pos["quantity_open"])
        qty = float(quantity) if quantity is not None else qty_open
        if qty <= 0:
            raise JournalError("quantity должен быть > 0")
        if qty > qty_open + QTY_EPS:
            raise JournalError(f"quantity {qty} больше остатка позиции {qty_open}")
        qty = min(qty, qty_open)

        pos["fills"].append({
            "ts_ms": closed_at, "ts_utc": _utc_iso(closed_at),
            "type": "exit_partial" if qty < qty_open - QTY_EPS else "exit_full",
            "price": float(price), "quantity": qty, "note": note,
        })
        pos["quantity_open"] = qty_open - qty

        exits = [f for f in pos["fills"] if f["type"].startswith("exit")]
        total_exit_qty = sum(f["quantity"] for f in exits)
        avg_exit = sum(f["price"] * f["quantity"] for f in exits) / total_exit_qty if total_exit_qty else None
        realized = sum((f["price"] - pos["entry_price"]) * f["quantity"] for f in exits)
        pos["avg_exit_price"] = avg_exit
        pos["realized_pnl_quote"] = realized
        pos["realized_pnl_pct"] = (realized / (pos["entry_price"] * total_exit_qty) * 100.0
                                   if total_exit_qty else None)

        fully_closed = pos["quantity_open"] <= QTY_EPS
        if fully_closed:
            pos["status"] = "CLOSED"
            pos["quantity_open"] = 0.0
            pos["closed_at_ms"] = closed_at
            pos["closed_at_utc"] = _utc_iso(closed_at)
            pos["outcome"] = ("WIN" if realized > 0 else ("LOSS" if realized < 0 else "FLAT"))
            reason = (f"Полное закрытие: {qty} @ {price}. Результат: {realized:+.2f} USDT "
                      f"({pos['realized_pnl_pct']:+.3f}%) -> {pos['outcome']}")
        else:
            reason = f"Частичное закрытие: {qty} @ {price}; остаток {pos['quantity_open']}"
        pos["status_history"].append({
            "ts_ms": ts, "ts_utc": _utc_iso(ts),
            "event": "CLOSED" if fully_closed else "PARTIAL_CLOSE",
            "reason": reason + (f" ({note})" if note else ""),
        })
        self._append_event(data, ts, "position_closed" if fully_closed else "position_partial_close",
                           {"position_id": position_id, "price": float(price), "quantity": qty,
                            "remaining": pos["quantity_open"], "realized_pnl_quote": realized,
                            "status": pos["status"], "reason": reason})
        self._save(data)
        return PositionView(pos)

    # -- reads ------------------------------------------------------------------
    def positions(self, status: Optional[str] = None) -> list[PositionView]:
        data = self._load()
        out = [PositionView(p) for p in data["positions"]]
        if status:
            out = [p for p in out if p.status == status.upper()]
        return out

    def open_positions(self, symbol: Optional[str] = None) -> list[PositionView]:
        return [p for p in self.positions("OPEN")
                if symbol is None or p.data["symbol"] == symbol]

    def get_position(self, position_id: str) -> Optional[PositionView]:
        for p in self.positions():
            if p.position_id == position_id:
                return p
        return None

    def events(self, n: int = 50) -> list[dict]:
        return self._load()["events"][-n:]

    # -- paper trading (виртуальные позиции бота) --------------------------------
    def next_paper_id(self, data: dict) -> str:
        data["counters"]["paper_position"] = int(data["counters"].get("paper_position", 0)) + 1
        return f"paper-{data['counters']['paper_position']:06d}"

    def save_paper_position(self, pos: dict) -> None:
        """Создать или обновить виртуальную позицию (по position_id). Атомарно."""
        data = self._load()
        for i, p in enumerate(data["paper_positions"]):
            if p["position_id"] == pos["position_id"]:
                data["paper_positions"][i] = pos
                break
        else:
            data["paper_positions"].append(pos)
        self._save(data)

    def log_paper_action(self, kind: str, details: dict) -> None:
        data = self._load()
        ts = _now_ms()
        data["paper_actions"].append({"ts_ms": ts, "ts_utc": _utc_iso(ts), "kind": kind, **details})
        self._save(data)

    def paper_positions(self, status: Optional[str] = None) -> list[dict]:
        out = self._load()["paper_positions"]
        if status:
            out = [p for p in out if p.get("status") == status.upper()]
        return out

    def paper_actions(self, n: int = 30) -> list[dict]:
        return self._load()["paper_actions"][-n:]

    def paper_stats(self, initial_balance: float) -> dict:
        data = self._load()
        poss = data["paper_positions"]
        realized = sum(float(p.get("realized_pnl_quote") or 0.0) for p in poss)
        open_p = [p for p in poss if p.get("status") == "OPEN"]
        closed = [p for p in poss if p.get("status") == "CLOSED"]
        return {
            "initial_balance_quote": initial_balance,
            "balance_quote": initial_balance + realized,
            "realized_pnl_quote": realized,
            "n_open": len(open_p),
            "n_closed": len(closed),
            "wins": sum(1 for p in closed if float(p.get("realized_pnl_quote") or 0) > 0),
            "losses": sum(1 for p in closed if float(p.get("realized_pnl_quote") or 0) < 0),
            "n_actions": len(data["paper_actions"]),
        }

    def stats(self) -> dict:
        data = self._load()
        closed = [p for p in data["positions"] if p["status"] == "CLOSED"]
        pnl = [p["realized_pnl_quote"] for p in closed if p.get("realized_pnl_quote") is not None]
        return {
            "n_signals": len(data["signals"]),
            "n_positions": len(data["positions"]),
            "n_open": sum(1 for p in data["positions"] if p["status"] == "OPEN"),
            "n_closed": len(closed),
            "total_realized_pnl_quote": sum(pnl) if pnl else 0.0,
            "wins": sum(1 for p in closed if p.get("outcome") == "WIN"),
            "losses": sum(1 for p in closed if p.get("outcome") == "LOSS"),
        }
