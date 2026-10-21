"""Консольный дашборд: цена, история свечей, соединение, свежесть, сигналы.

format_snapshot_text() - чистая функция (детерминированный текст для тестов).
ConsoleUI - обёртка: ANSI-перерисовка в TTY, компактный построчный режим иначе
(например, при перенаправлении вывода или в проверках).
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
    """Детерминированный текст дашборда из среза состояния (view)."""
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
                     f"  O {live.open:,.1f}  H {live.high:,.1f}  L {live.low:,.1f}"
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
            # приоритет причине, связанной с данными (устарел/разрыв), затем остальным
            data_reasons = [r for r in s.reasons if "воздержание" not in r]
            reason = f"  ({data_reasons[0] if data_reasons else s.reasons[-1]})" if s.reasons else ""
            lines.append(f"   {fmt_utc(s.created_at_ms, with_date=False)}  {s.action:<4} "
                         f"conf {s.confidence:.3f}  bar {fmt_utc(s.bar_start_ms, with_date=False)}  "
                         f"{mark}{reason}")
    else:
        lines.append(" История сигналов: пуста")

    pos_txt = view.get("positions_text")
    lines.append(THIN)
    lines.append(f" Позиции: {pos_txt if pos_txt else '(учёт позиций - Этап 4; фактическое состояние портфеля неизвестно)'}")

    warnings = view.get("warnings") or []
    lines.append(f" Предупреждения: {'; '.join(warnings) if warnings else 'нет'}")
    lines.append(LINE)
    lines.append(f" {DISCLAIMER}")
    lines.append(f" Обновлено: {fmt_utc(view.get('rendered_at_ms'))}")
    return "\n".join(lines)


def compact_line(view: dict) -> str:
    price = view.get("price")
    sig = view.get("last_signal")
    sig_txt = (f"{sig.action} conf={sig.confidence:.3f} @{fmt_utc(sig.bar_end_ms, with_date=False)}"
               f"{'' if sig.data_fresh else ' [STALE]'}") if sig else "нет"
    return (f"{fmt_utc(view.get('rendered_at_ms'), with_date=False)} | {view.get('conn_state', '?')} | "
            f"{'FRESH' if view.get('fresh') else 'STALE'} "
            f"({view.get('age_s') if view.get('age_s') is None else round(view['age_s'])}s) | "
            f"price={price if price is None else format(price, ',.1f')} | сигнал: {sig_txt}")


class ConsoleUI:
    """ANSI-перерисовка в TTY; построчный компактный вывод иначе."""

    def __init__(self, stream: Optional[TextIO] = None, refresh_s: float = 1.0,
                 force_line_mode: bool = False):
        self.stream = stream or sys.stdout
        self.refresh_s = refresh_s
        self._last_render = 0.0
        self._last_compact = ""
        self._last_text = ""
        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.ansi = tty and not force_line_mode and os.environ.get("NO_ANSI") != "1"
        if self.ansi:
            self._enable_vt()

    def _enable_vt(self) -> None:
        """Включить обработку ANSI-escape в консоли Windows (ENABLE_VIRTUAL_TERMINAL_PROCESSING)."""
        if os.name != "nt":
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:  # noqa: BLE001 - fallback ниже
            pass
        try:
            os.system("")  # дополнительный известный триггер VT на старых сборках
        except Exception:  # noqa: BLE001
            pass

    def render(self, view: dict, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_render) < self.refresh_s:
            return
        self._last_render = now
        if self.ansi:
            text = format_snapshot_text(view)
            if text == self._last_text and not force:
                return  # ничего не изменилось - не перерисовываем (меньше шума в скроллбэке)
            self._last_text = text
            self.stream.write("\x1b[2J\x1b[H" + text + "\n")
            self.stream.flush()
        else:
            line = compact_line(view)
            if line != self._last_compact:
                self._last_compact = line
                self.stream.write(line + "\n")
                self.stream.flush()

    def render_full_once(self, view: dict) -> None:
        """Полный текст один раз (для не-TTY режимов и логов)."""
        self.stream.write(format_snapshot_text(view) + "\n")
        self.stream.flush()

    def print_event(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()
