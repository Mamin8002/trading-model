"""Консольный UI: три режима отображения.

  * line  - ОДНА строка статуса, обновляемая НА МЕСТЕ (\r); значимые события
            (сигнал, смена соединения/свежести, рекомендация) печатаются
            отдельными строками выше - раз в несколько минут, а не каждую секунду.
            Работает в любом терминале (PowerShell, PyCharm, cmd).
  * dash  - полноэкранный дашборд на alternate screen (\x1b[?1049h): экран
            перерисовывается на месте, scrollback не засоряется. Требует ANSI/VT.
  * block - полный снимок печатается только при изменении (для логов/pipe).

format_snapshot_text() - чистая функция (детерминированный текст для тестов).
Цвета и Unicode-символы включаются только когда вывод их поддерживает.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, TextIO

LINE = "=" * 78
THIN = "-" * 78

DISCLAIMER = ("Сигналы - вывод модели, не гарантия дохода и не персональная финансовая "
              "рекомендация.")

# ANSI
RESET, BOLD = "\x1b[0m", "\x1b[1m"
RED, GREEN, YELLOW, CYAN, GREY = "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[36m", "\x1b[90m"

GLYPHS_UNI = {"ok": "●", "warn": "◐", "bad": "○", "buy": "▲", "sell": "▼",
              "hold": "◆", "fresh": "✓", "stale": "✗", "arrow": "→", "alarm": "⚠"}
GLYPHS_ASCII = {"ok": "*", "warn": "o", "bad": "x", "buy": "^", "sell": "v",
                "hold": "+", "fresh": "OK", "stale": "!", "arrow": "->", "alarm": "!!"}


def fmt_utc(ms: Optional[int], with_date: bool = True) -> str:
    if ms is None:
        return "н/д"
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S") + " UTC"


def _fmt_bar_line(bar, show_date: bool = False) -> str:
    return (f" {fmt_utc(bar.start_ms, show_date):>22}  "
            f"O {bar.open:>10.1f}  H {bar.high:>10.1f}  L {bar.low:>10.1f}  "
            f"C {bar.close:>10.1f}  V {bar.volume:>9.3f}")


def format_snapshot_text(view: dict) -> str:
    """Детерминированный текст полного дашборда из среза состояния (view)."""
    lines: list[str] = []
    mode = view.get("mode", "?")
    ex = view.get("exchange", {})
    model = view.get("model")
    lines.append(LINE)
    lines.append(f" BTC MONITOR | Bybit public data | режим {mode}")
    lines.append(f" {ex.get('category', '?')}/{ex.get('symbol', '?')}  interval {ex.get('interval', '?')}m"
                 + (f"  |  модель {model['version']} (горизонт {model['horizon_minutes']} мин, порог {model['threshold']:.2f})"
                    if model else "  |  модель НЕ ЗАГРУЖЕНА"))
    lines.append(LINE)

    conn = view.get("conn_state", "DISCONNECTED")
    reconn = view.get("reconnect_count", 0)
    lines.append(f" Соединение: {conn} ({view.get('conn_detail', '')})  | переподключений: {reconn}")

    fresh = view.get("fresh", False)
    age_s = view.get("age_s")
    stale_thr_s = view.get("stale_threshold_s")
    age_txt = f"{age_s:.0f} с назад" if age_s is not None else "нет данных"
    fresh_txt = "FRESH" if fresh else "STALE/НЕТ ДАННЫХ"
    lines.append(f" Данные: {fresh_txt} (последнее обновление {age_txt}, порог {stale_thr_s:.0f} с)"
                 f" | серверное время: {fmt_utc(view.get('server_ts_ms'), with_date=False)}")

    price = view.get("price")
    if price is not None:
        p_age = view.get("price_age_s")
        lines.append(f" Цена: {price:,.2f} USDT (ticker{f', {p_age:.0f} с назад' if p_age is not None else ''})")
    else:
        lines.append(" Цена: нет данных")
    lines.append(THIN)

    live = view.get("live_candle")
    if live is not None:
        lines.append(f" Формирующаяся свеча {fmt_utc(live.start_ms, with_date=False)}:"
                     f" O {live.open:,.1f}  H {live.high:,.1f}  L {live.low:,.1f}"
                     f"  C {live.close:,.1f}  V {live.volume:.3f}  (сигнал ТОЛЬКО после закрытия)")
    bars = view.get("bars") or []
    n_disp = int(view.get("display_bars", 10))
    if bars:
        lines.append(f" Последние закрытые свечи (UTC, всего в памяти {len(bars)}):")
        for b in bars[-n_disp:]:
            lines.append(_fmt_bar_line(b))
    else:
        lines.append(" Закрытых свечей пока нет (ожидание данных...)")
    lines.append(THIN)

    sig = view.get("last_signal")
    if sig is not None:
        fresh_mark = "данные FRESH" if sig.data_fresh else "данные НЕ АКТУАЛЬНЫ (сигнал помечен как устаревший)"
        lines.append(f" ТЕКУЩИЙ СИГНАЛ (бар {fmt_utc(sig.bar_start_ms, with_date=False)} -> "
                     f"{fmt_utc(sig.bar_end_ms, with_date=False)}):")
        lines.append(f"   >>> {sig.action} <<<  уверенность {sig.confidence:.3f}  "
                     f"выдан {fmt_utc(sig.created_at_ms)}")
        lines.append(f"   инструмент {sig.category}/{sig.symbol}  цена на момент выдачи "
                     f"{sig.price_at_signal:,.2f} ({sig.price_source})  горизонт {sig.horizon_minutes} мин")
        lines.append(f"   версия модели {sig.model_version}  | {fresh_mark}")
        if sig.reasons:
            for r in sig.reasons:
                lines.append(f"   причина: {r}")
    else:
        lines.append(" ТЕКУЩИЙ СИГНАЛ: нет (ожидание первого закрытого бара)")
    lines.append(THIN)

    hist = view.get("signal_history") or []
    if hist:
        lines.append(" История сигналов (последние):")
        for s in hist[-5:]:
            mark = "fresh" if s.data_fresh else "STALE"
            data_reasons = [r for r in s.reasons if "воздержание" not in r]
            reason = f"  ({data_reasons[0] if data_reasons else s.reasons[-1]})" if s.reasons else ""
            lines.append(f"   {fmt_utc(s.created_at_ms, with_date=False)}  {s.action:<4} "
                         f"conf {s.confidence:.3f}  bar {fmt_utc(s.bar_start_ms, with_date=False)}  "
                         f"{mark}{reason}")
    else:
        lines.append(" История сигналов: пуста")

    lines.append(THIN)
    journal_path = view.get("journal_path")
    stats = view.get("portfolio_stats")
    if journal_path:
        stats_txt = (f" | сигналов: {stats['n_signals']}, позиций: {stats['n_positions']} "
                     f"(открыто {stats['n_open']}, закрыто {stats['n_closed']},"
                     f" PnL {stats['total_realized_pnl_quote']:+.2f} USDT)" if stats else "")
        lines.append(f" Журнал: {journal_path}{stats_txt}")
    positions = view.get("positions") or []
    if positions:
        lines.append(" Открытые позиции (зарегистрированы вручную):")
        for p in positions:
            unreal = (f"{p['unrealized_pct']:+.3f}%"
                      if p.get("unrealized_pct") is not None and p.get("price_now") else "н/д")
            price_now_txt = f"{p['price_now']:,.2f}" if p.get("price_now") else "н/д"
            lines.append(f"   {p['position_id']} {p['direction']} {p['symbol']}: вход {p['entry_price']:,.2f} "
                         f"({p.get('opened_at_utc', '?')}), объём {p['quantity_open']:g} из "
                         f"{p['quantity_initial']:g}, текущая {price_now_txt} ({unreal})")
            if p.get("source_signal_id"):
                lines.append(f"     исходный сигнал: {p['source_signal_id']} "
                             f"({p.get('source_signal_action')} conf {p.get('source_signal_conf'):.3f}, "
                             f"горизонт {p.get('horizon_minutes')} мин, модель {p.get('model_version')})")
            else:
                lines.append("     исходный сигнал: не привязана (ручная регистрация)")
            lines.append(f"     условия выхода: {p.get('rules_text') or 'не заданы'}")
    else:
        lines.append(" Позиции: зарегистрированных открытых позиций нет - фактическое состояние "
                     "портфеля неизвестно (бот не может знать его без ручной регистрации)")

    advice = view.get("last_advice")
    if advice:
        lines.append(f" Рекомендация [{advice['action']}] (по сигналу {advice.get('signal_id')}):")
        lines.append(f"   {advice['text']}")
        for r in (advice.get("reasons") or [])[:4]:
            lines.append(f"   - {r}")

    warnings = view.get("warnings") or []
    lines.append(f" Предупреждения: {'; '.join(warnings) if warnings else 'нет'}")
    lines.append(LINE)
    lines.append(f" {DISCLAIMER}")
    lines.append(f" Обновлено: {fmt_utc(view.get('rendered_at_ms'))}")
    return "\n".join(lines)


def _action_glyph(action: str, g: dict) -> str:
    return {"BUY": g["buy"], "SELL": g["sell"], "HOLD": g["hold"]}.get(action, "?")


def format_status_line(view: dict, g: Optional[dict] = None) -> str:
    """Однострочный статус (режим line): всё главное в одну обновляемую строку."""
    g = g or GLYPHS_UNI
    conn = str(view.get("conn_state", "?"))
    fresh = view.get("fresh", False)
    age = view.get("age_s")
    dot = g["ok"] if (conn == "CONNECTED" and fresh) else (g["warn"] if conn in ("CONNECTED", "CONNECTING") else g["bad"])
    parts = [
        f"{dot} {fmt_utc(view.get('rendered_at_ms'), with_date=False)}",
        conn,
        (f"{g['fresh']} {age:.0f}с" if fresh else f"{g['stale']} STALE" + (f" {age:.0f}с" if age is not None else "")),
    ]
    price = view.get("price")
    parts.append(f"{price:,.1f}" if price is not None else "цена: -")
    live = view.get("live_candle")
    if live is not None:
        parts.append(f"свеча {fmt_utc(live.start_ms, with_date=False)} C {live.close:,.1f}")
    sig = view.get("last_signal")
    if sig is not None:
        mark = "" if sig.data_fresh else " [НЕ АКТУАЛЕН]"
        parts.append(f"{_action_glyph(sig.action, g)} {sig.action} {sig.confidence:.3f} "
                     f"@{fmt_utc(sig.created_at_ms, with_date=False)}{mark}")
    else:
        parts.append("сигнал: ожидание закрытия бара")
    positions = view.get("positions") or []
    advice = view.get("last_advice")
    if positions:
        p = positions[-1]
        unreal = f" {p['unrealized_pct']:+.2f}%" if p.get("unrealized_pct") is not None else ""
        parts.append(f"{p['position_id']}{unreal}")
    if advice:
        parts.append(f"[{advice['action']}]")
    return " | ".join(parts)


# совместимость со старым именем
def compact_line(view: dict) -> str:
    return format_status_line(view)


# ------------------------------------------------------------------ события
def event_signal_text(sig, g: Optional[dict] = None) -> str:
    g = g or GLYPHS_UNI
    tag = "" if sig.data_fresh else " [НЕ АКТУАЛЕН: " + "; ".join(sig.reasons[:2]) + "]"
    reason = ""
    if sig.action == "HOLD" and sig.reasons:
        reason = f" ({sig.reasons[-1]})"
    return (f"[{fmt_utc(sig.created_at_ms, with_date=False)}] "
            f"{_action_glyph(sig.action, g)} СИГНАЛ {sig.action} conf {sig.confidence:.3f} "
            f"| цена {sig.price_at_signal:,.2f} | бар {fmt_utc(sig.bar_start_ms, with_date=False)}"
            f"{g['arrow']}{fmt_utc(sig.bar_end_ms, with_date=False)} | горизонт {sig.horizon_minutes} мин "
            f"| модель {sig.model_version}{reason}{tag}")


def event_advice_text(advice, g: Optional[dict] = None) -> str:
    g = g or GLYPHS_UNI
    text = advice.text if isinstance(advice.text, str) else str(advice)
    if len(text) > 150:
        text = text[:147] + "..."
    return f"    {g['arrow']} Рекомендация [{advice.action}]: {text}"


def event_state_text(ev, g: Optional[dict] = None) -> str:
    g = g or GLYPHS_UNI
    dot = g["ok"] if ev.state.value == "CONNECTED" else (g["bad"] if ev.state.value in ("RECONNECTING", "DISCONNECTED") else g["warn"])
    return f"[{fmt_utc(ev.received_ms, with_date=False)}] {dot} Соединение: {ev.state.value}" + (f" ({ev.detail})" if ev.detail else "")


def event_fresh_text(ev, g: Optional[dict] = None) -> str:
    g = g or GLYPHS_UNI
    mark = g["fresh"] if ev.fresh else g["alarm"]
    return f"[{fmt_utc(ev.received_ms, with_date=False)}] {mark} Данные: {'FRESH' if ev.fresh else 'STALE'} ({ev.detail})"


def event_gap_text(ev, g: Optional[dict] = None) -> str:
    g = g or GLYPHS_UNI
    return (f"[{fmt_utc(ev.received_ms, with_date=False)}] {g['alarm']} Разрыв истории свечей: "
            f"ожидался бар {fmt_utc(ev.expected_start_ms)}, получен {fmt_utc(ev.got_start_ms)}")


def event_info(text: str) -> str:
    return f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {text}"


class ConsoleUI:
    """Режимы: auto | dash | line | block (см. модульный docstring)."""

    def __init__(self, stream: Optional[TextIO] = None, refresh_s: float = 1.0,
                 force_line_mode: bool = False, mode: Optional[str] = None,
                 log_events: bool = True, colors: str = "auto"):
        self.stream = stream or sys.stdout
        self.refresh_s = refresh_s
        self.log_events_enabled = log_events
        self._last_render = 0.0
        self._last_status: Optional[str] = None
        self._last_block_text = ""
        self._dash_lines = 0
        self._in_alt_screen = False

        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        env_no_ansi = os.environ.get("NO_ANSI") == "1"
        self._vt_ok = False
        if tty and not env_no_ansi:
            self._vt_ok = self._enable_vt()

        if mode is None:
            mode = "line" if force_line_mode else "auto"
        if mode == "auto":
            if not tty or env_no_ansi:
                mode = "block"
            else:
                mode = "dash" if self._vt_ok else "line"
        if mode not in ("dash", "line", "block"):
            mode = "block"
        self.mode = mode
        self.ansi = (mode == "dash")

        # Unicode-глифы, только если поток их переварит
        try:
            enc = (getattr(self.stream, "encoding", None) or "utf-8").lower()
            if "utf" not in enc and hasattr(self.stream, "reconfigure"):
                self.stream.reconfigure(encoding="utf-8", errors="replace")
                enc = "utf-8"
        except Exception:  # noqa: BLE001
            enc = "ascii"
        self.glyphs = GLYPHS_UNI if "utf" in enc else GLYPHS_ASCII
        self.colors = (self._vt_ok and colors != "off") if colors == "auto" else (colors == "on")

    # -- ANSI helpers ---------------------------------------------------------
    def _enable_vt(self) -> bool:
        if os.name != "nt":
            return True  # предполагаем xterm-совместимый терминал
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False  # не настоящая консоль (например, пайп или pty IDE)
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            check = ctypes.c_uint32()
            kernel32.GetConsoleMode(handle, ctypes.byref(check))
            return bool(check.value & 0x0004)
        except Exception:  # noqa: BLE001
            return False

    def _c(self, text: str, color: str) -> str:
        return f"{color}{text}{RESET}" if self.colors else text

    def begin(self) -> None:
        if self.mode == "dash" and not self._in_alt_screen:
            self.stream.write("\x1b[?1049h\x1b[?25l")  # alternate screen, скрыть курсор
            self.stream.flush()
            self._in_alt_screen = True

    def end(self) -> None:
        if self._in_alt_screen:
            self.stream.write("\x1b[?25h\x1b[?1049l")  # показать курсор, выйти из alt screen
            self.stream.flush()
            self._in_alt_screen = False

    # -- rendering -------------------------------------------------------------
    def should_render(self) -> bool:
        """Нужно ли рисовать (по частоте refresh_s) - для дешёвого опроса в цикле бота."""
        return (time.monotonic() - self._last_render) >= self.refresh_s

    def render(self, view: dict, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_render) < self.refresh_s:
            return
        self._last_render = now
        if self.mode == "dash":
            text = self._paint(format_snapshot_text(view))
            self.stream.write("\x1b[H\x1b[2J" + text + "\n")
            self.stream.flush()
            self._dash_lines = text.count("\n") + 1
        elif self.mode == "line":
            status = format_status_line(view, self.glyphs)
            if self.colors:
                conn = str(view.get("conn_state", ""))
                if conn == "CONNECTED" and view.get("fresh"):
                    status = self._c(status, GREEN)
                elif conn in ("RECONNECTING", "DISCONNECTED") or not view.get("fresh"):
                    status = self._c(status, YELLOW)
            clean = "\r\x1b[K" if self._vt_ok else "\r"
            self.stream.write(clean + status)
            self.stream.flush()
            self._last_status = status
        else:  # block
            text = format_snapshot_text(view)
            if text != self._last_block_text or force:
                self._last_block_text = text
                self.stream.write(text + "\n")
                self.stream.flush()

    def _paint(self, text: str) -> str:
        if not self.colors:
            return text
        for token, color in ((">>> BUY <<<", GREEN), (">>> SELL <<<", RED), (">>> HOLD <<<", YELLOW),
                             ("FRESH", GREEN), ("STALE", RED), ("CONNECTED", GREEN),
                             ("RECONNECTING", YELLOW), ("DISCONNECTED", RED)):
            text = text.replace(token, f"{color}{token}{RESET}")
        return text

    # -- events ------------------------------------------------------------------
    def log_event(self, text: str) -> None:
        """Строка значимого события (только line/block; в dash всё видно на панели)."""
        if not self.log_events_enabled or self.mode == "dash":
            return
        if self.glyphs is GLYPHS_ASCII:
            for key, uni_glyph in GLYPHS_UNI.items():
                text = text.replace(uni_glyph, GLYPHS_ASCII[key])
        self.stream.write(text + "\n")
        self.stream.flush()
        if self.mode == "line" and self._last_status is not None:
            clean = "\r\x1b[K" if self._vt_ok else "\r"
            self.stream.write(clean + self._last_status)
            self.stream.flush()

    def render_full_once(self, view: dict) -> None:
        self.stream.write(format_snapshot_text(view) + "\n")
        self.stream.flush()

    def print_event(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()
