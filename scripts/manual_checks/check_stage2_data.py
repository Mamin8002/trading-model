#!/usr/bin/env python3
"""Ручная проверка Этапа 2a: сбор, очистка и сохранение исторических данных.

Запуск (PowerShell, из корня проекта btc_bybit_bot):
    python scripts/manual_checks/check_stage2_data.py
Опции:
    --no-network   пропустить сетевую часть (B) - только офлайн-проверки на синтетике

Секреты и торговые операции не используются: сетевая часть обращается только
к публичным read-only эндпоинтам Bybit (market/time, market/kline).

Создаваемые файлы (сетевая часть):
    data/manual_checks/stage2a_sample_<UTC>.csv (+ .metadata.json)
    data/manual_checks/stage2a_sample_<UTC>.clean.csv (+ .clean.report.json)

Ожидаемый вывод: строки [PASS]/[FAIL] по каждой проверке и итог "RESULT: PASS".
Код возврата: 0 - PASS, 1 - FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from btc_bot.data.prep import (  # noqa: E402
    FEATURE_COLUMNS,
    build_dataset,
    clean_klines,
    validate_ohlc,
)
from btc_bot.data.store import (  # noqa: E402
    load_klines_csv,
    load_metadata,
    save_klines_csv,
)
from btc_bot.data_sources.base import DataSourceError, Kline  # noqa: E402
from btc_bot.data_sources.bybit import BybitSource, period_end_ms  # noqa: E402

INTERVAL = "5"
STEP_MS = 300_000
BASE = 1_700_000_100_000  # выровнено по 5 минутам

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f" :: {detail}"
    print(line)


def record_skip(name: str, detail: str) -> None:
    print(f"[SKIP] {name} :: {detail}")


def mk(start_ms: int, o: float, h: float, l: float, c: float,
       v: float = 1.0, t: float = 100.0, closed: bool = True) -> dict:
    return {"start_ms": start_ms, "open": o, "high": h, "low": l, "close": c,
            "volume": v, "turnover": t, "is_closed": closed}


def ts(i: int) -> int:
    return BASE + i * STEP_MS


# ---------------------------------------------------------------- offline A
def check_a1_dedup_sort() -> None:
    rows = [mk(ts(2), 10, 11, 9, 10.5), mk(ts(0), 9, 10, 8, 9.5),
            mk(ts(2), 10, 11, 9, 10.5), mk(ts(1), 9.5, 10.5, 9, 10)]
    df, rep = clean_klines(pd.DataFrame(rows), INTERVAL)
    ok = (len(df) == 3 and rep.n_duplicates_removed == 1
          and list(df["start_ms"]) == [ts(0), ts(1), ts(2)])
    record("A1 дедупликация и сортировка по возрастанию", ok,
           f"n_output={rep.n_output}, dupes={rep.n_duplicates_removed}")


def check_a2_gaps() -> None:
    rows = [mk(ts(0), 10, 11, 9, 10.5), mk(ts(1), 10, 11, 9, 10.5),
            mk(ts(4), 10, 11, 9, 10.5)]  # пропущены бары 2 и 3
    df, rep = clean_klines(pd.DataFrame(rows), INTERVAL)
    ok = (rep.total_missing_bars == 2 and len(rep.gaps) == 1
          and rep.gaps[0].after_start_ms == ts(1) and rep.gaps[0].until_start_ms == ts(4)
          and len(df) == 3)
    record("A2 детекция пропусков (gaps) без заполнения", ok,
           f"missing={rep.total_missing_bars}, ranges={len(rep.gaps)}")


def check_a3_unconfirmed() -> None:
    rows = [mk(ts(0), 10, 11, 9, 10.5), mk(ts(1), 10, 11, 9, 10.5),
            mk(ts(2), 10, 11, 9, 10.7, closed=False)]
    df, rep = clean_klines(pd.DataFrame(rows), INTERVAL)
    ok = (rep.n_unconfirmed_excluded == 1 and len(df) == 2
          and df["start_ms"].max() == ts(1))
    record("A3 исключение незакрытой свечи", ok,
           f"excluded={rep.n_unconfirmed_excluded}, n_output={rep.n_output}")


def check_a4_invalid_ohlc() -> None:
    rows = [mk(ts(0), 10, 11, 9, 10.5),
            mk(ts(1), 10, 9, 11, 10.5),      # high < low - некорректно
            mk(ts(2), 10, 11, 9, -5)]         # close <= 0 - некорректно
    df, rep = clean_klines(pd.DataFrame(rows), INTERVAL)
    ok = rep.n_invalid_removed == 2 and len(df) == 1 and len(rep.invalid_examples) == 2
    record("A4 удаление некорректных OHLC-строк с примерами в отчёте", ok,
           f"invalid={rep.n_invalid_removed}, examples={len(rep.invalid_examples)}")


def check_a5_targets() -> None:
    # closes: 10, 11, 10.5, 10.5, 12  (бар 3 -> ничья со баром... см. расчёты)
    closes = [10.0, 11.0, 10.5, 10.5, 12.0]
    rows = []
    for i, c in enumerate(closes):
        rows.append(mk(ts(i), c, c + 0.5, c - 0.5, c))
    df, _ = clean_klines(pd.DataFrame(rows), INTERVAL)

    # H=1: y = [+1(11>10), -1(10.5<11), -1(ничья 10.5==10.5 -> -1), +1(12>10.5)], бар4 без будущего
    X, y, times, rep = build_dataset(df, horizon_bars=1, interval=INTERVAL)
    ok1 = (list(y) == [1, -1, -1, 1] and rep.n_rows == 4
           and rep.n_dropped_no_future_bar == 1
           and list(X.columns) == FEATURE_COLUMNS)

    # H=2: будущие close через 2 бара: y = [+1(10.5>10), -1(10.5<11), +1(12>10.5)]
    X2, y2, _, rep2 = build_dataset(df, horizon_bars=2, interval=INTERVAL)
    ok2 = list(y2) == [1, -1, 1] and rep2.n_rows == 3 and rep2.n_dropped_no_future_bar == 2

    # пропуск в истории -> строка без точного будущего бара исключается, не фабрикуется
    # бары t0..t3 и t5 (бар t4 отсутствует): close = 10, 11, 10.5, 10.5, 12
    rows_gap = [mk(ts(0), 10.0, 10.5, 9.5, 10.0), mk(ts(1), 11.0, 11.5, 10.5, 11.0),
                mk(ts(2), 10.5, 11.0, 10.0, 10.5), mk(ts(3), 10.5, 11.0, 10.0, 10.5),
                mk(ts(5), 12.0, 12.5, 11.5, 12.0)]
    dfg, _ = clean_klines(pd.DataFrame(rows_gap), INTERVAL)
    X3, y3, _, rep3 = build_dataset(dfg, horizon_bars=1, interval=INTERVAL)
    ok3 = (rep3.n_dropped_no_future_bar == 2 and list(y3) == [1, -1, -1])

    record("A5 цель y=±1 для H=1/H=2, ничья->-1, пропуск->исключение строки",
           ok1 and ok2 and ok3,
           f"H1={[int(v) for v in y]}, H2={[int(v) for v in y2]}, gap_dropped={rep3.n_dropped_no_future_bar}")


def check_a6_store_roundtrip() -> None:
    kl = [Kline(ts(0), 10, 11, 9, 10.5, 1.0, 100.0, True),
          Kline(ts(1), 10.5, 11.5, 10, 11.0, 2.0, 200.0, True),
          Kline(ts(2), 11.0, 11.2, 10.8, 10.9, 1.5, 150.0, False)]
    with tempfile.TemporaryDirectory() as td:
        csv_path, meta_path = save_klines_csv(
            kl, Path(td) / "sample.csv", source="bybit",
            category="spot", symbol="BTCUSDT", interval=INTERVAL,
        )
        df = load_klines_csv(csv_path)
        meta = load_metadata(meta_path)
        ok_df = (len(df) == 3
                 and df["start_ms"].is_monotonic_increasing
                 and bool(df["is_closed"].iloc[-1]) is False
                 and abs(float(df["close"].iloc[0]) - 10.5) < 1e-9)
        need = {"source", "category", "symbol", "interval", "n_rows",
                "fetched_at_utc", "period_start_ms", "period_end_ms", "api_endpoint"}
        ok_meta = need.issubset(meta.keys()) and meta["n_rows"] == 3 and meta["symbol"] == "BTCUSDT"
        ok = ok_df and ok_meta
    record("A6 сохранение/загрузка CSV + полнота metadata.json", ok,
           f"df_ok={ok_df}, meta_ok={ok_meta}, meta_keys={sorted(meta.keys())[:5]}...")


# ---------------------------------------------------------------- network B
def _fetch_recent_days(source: BybitSource, symbol: str, days: int = 7):
    now_ms = source.get_server_time_ms()
    start_ms = now_ms - days * 86_400_000
    klines = source.fetch_klines(symbol=symbol, category="spot", interval=INTERVAL,
                                 start_ms=start_ms, end_ms=now_ms)
    return klines, now_ms


def check_b1_server_time(source: BybitSource) -> bool:
    try:
        srv = source.get_server_time_ms()
    except DataSourceError as e:
        record("B1 серверное время Bybit", False, f"ошибка: {e}")
        return False
    local = int(datetime.now(timezone.utc).timestamp() * 1000)
    drift = abs(srv - local)
    ok = drift < 120_000  # до 2 минут расхождения часов
    record("B1 серверное время Bybit (/v5/market/time)", ok,
           f"server={srv}, drift={drift} ms")
    return ok


def check_b2_fetch(source: BybitSource, symbol: str) -> list[Kline] | None:
    try:
        klines, now_ms = _fetch_recent_days(source, symbol, days=7)
    except DataSourceError as e:
        record("B2 загрузка 7 дней 5m-свечей (public REST)", False, f"ошибка: {e}")
        return None
    problems = []
    if len(klines) < 1900:
        problems.append(f"мало свечей: {len(klines)} (ожидали ~2016)")
    starts = [k.start_ms for k in klines]
    if starts != sorted(starts):
        problems.append("порядок не по возрастанию")
    if len(set(starts)) != len(starts):
        problems.append("есть дубликаты start_ms")
    unclosed_idx = [i for i, k in enumerate(klines) if not k.is_closed]
    if unclosed_idx and unclosed_idx != [len(klines) - 1]:
        problems.append(f"незакрытые свечи не в конце: {unclosed_idx[:5]}")
    for k in klines:
        if not (k.high >= k.low and k.high >= max(k.open, k.close)
                and k.low <= min(k.open, k.close) and k.low > 0 and k.volume >= 0):
            problems.append(f"некорректная OHLC у {k.start_ms}")
            break
    closes = [k.close for k in klines]
    if not (1_000 < min(closes) and max(closes) < 10_000_000):
        problems.append(f"цены вне разумного диапазона BTC: {min(closes)}..{max(closes)}")
    # шаги сетки: почти все 300000 ms; редкие пропуски допустимы, но должны быть редки
    bad_steps = sum(1 for a, b in zip(starts, starts[1:]) if (b - a) % STEP_MS != 0 or b - a <= 0)
    if bad_steps:
        problems.append(f"{bad_steps} нестандартных шагов сетки")
    last_closed = [k for k in klines if k.is_closed][-1]
    if period_end_ms(last_closed.start_ms, INTERVAL) > now_ms:
        problems.append("последняя 'закрытая' свеча закрыта в будущем - ошибка флага")
    record("B2 загрузка 7 дней 5m-свечей (public REST, пагинация)", not problems,
           "; ".join(problems) if problems else
           f"n={len(klines)}, первая={datetime.fromtimestamp(starts[0]/1000, tz=timezone.utc):%Y-%m-%d %H:%M}, "
           f"последняя={datetime.fromtimestamp(starts[-1]/1000, tz=timezone.utc):%Y-%m-%d %H:%M} UTC, "
           f"незакрытых={len(unclosed_idx)}, последняя цена={closes[-1]}")
    return klines if not problems else None


def check_b3_persist(klines: list[Kline], symbol: str) -> None:
    out_dir = PROJECT_ROOT / "data" / "manual_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = out_dir / f"stage2a_sample_{stamp}.csv"
    try:
        csv_path, meta_path = save_klines_csv(
            klines, raw_path, source="bybit", category="spot", symbol=symbol, interval=INTERVAL,
        )
        df_raw = pd.read_csv(csv_path)
        df_clean, rep = clean_klines(df_raw, INTERVAL)
        clean_path = out_dir / f"stage2a_sample_{stamp}.clean.csv"
        report_path = out_dir / f"stage2a_sample_{stamp}.clean.report.json"
        df_clean.to_csv(clean_path, index=False)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(rep.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        meta = load_metadata(meta_path)

        problems = []
        if rep.n_input != len(klines):
            problems.append(f"n_input={rep.n_input} != {len(klines)}")
        if (rep.n_output + rep.n_duplicates_removed + rep.n_invalid_removed
                + rep.n_unconfirmed_excluded) != rep.n_input:
            problems.append("баланс очистки не сходится")
        if not df_clean["is_closed"].all():
            problems.append("в clean остались незакрытые")
        if not df_clean["start_ms"].is_monotonic_increasing or df_clean["start_ms"].duplicated().any():
            problems.append("clean не монотонен/с дублями")
        if not validate_ohlc(df_clean).all():
            problems.append("в clean есть некорректные OHLC")
        if meta.get("symbol") != symbol or meta.get("interval") != INTERVAL or meta.get("category") != "spot":
            problems.append("metadata не соответствует запросу")
        if rep.total_missing_bars > 10:
            problems.append(f"подозрительно много пропусков за 7 дней: {rep.total_missing_bars}")
        record("B3 сохранение raw+metadata, очистка, clean-файл и отчёт", not problems,
               "; ".join(problems) if problems else
               f"input={rep.n_input}, clean={rep.n_output}, unclosed_excluded={rep.n_unconfirmed_excluded}, "
               f"gaps={rep.total_missing_bars}; файлы: {csv_path.name}, {clean_path.name}")
    except Exception as e:  # noqa: BLE001 - проверка должна печатать FAIL, а не traceback
        record("B3 сохранение raw+metadata, очистка, clean-файл и отчёт", False, f"исключение: {e}")


# ------------------------------------------------------------------ runner
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-network", action="store_true",
                    help="пропустить сетевую часть B (только офлайн-проверки)")
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()

    print("== Этап 2a: сбор, очистка и сохранение исторических данных ==")
    print(f"Время: {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC, символ: {args.symbol}")
    print("-- Часть A: офлайн на синтетических данных (детерминированно) --")
    check_a1_dedup_sort()
    check_a2_gaps()
    check_a3_unconfirmed()
    check_a4_invalid_ohlc()
    check_a5_targets()
    check_a6_store_roundtrip()

    if args.no_network:
        print("-- Часть B: ПРОПУЩЕНА (--no-network). Сетевые проверки не выполнялись. --")
        record_skip("B1/B2/B3 сетевые проверки", "запустите без --no-network для полной проверки")
    else:
        print("-- Часть B: read-only публичные данные Bybit (без секретов, без торговли) --")
        source = BybitSource()
        try:
            if check_b1_server_time(source):
                klines = check_b2_fetch(source, args.symbol)
                if klines:
                    check_b3_persist(klines, args.symbol)
                else:
                    record_skip("B3 сохранение и очистка", "B2 не прошла - нет данных для B3")
            else:
                record_skip("B2/B3", "B1 не прошла (сеть?) - попробуйте позже или --no-network")
        except DataSourceError as e:
            record("B сетевая часть", False, f"ошибка источника данных: {e}")
        except Exception as e:  # noqa: BLE001 - проверка обязана печатать FAIL, а не traceback
            record("B сетевая часть", False, f"неожиданное исключение: {type(e).__name__}: {e}")

    failed = [name for name, ok, _ in _results if not ok]
    print()
    if failed:
        print(f"RESULT: FAIL ({len(failed)} проваленных: {', '.join(failed)})")
        print("Совет: при сетевых сбоях повторите позже; офлайн-часть: --no-network")
        return 1
    print(f"RESULT: PASS (проверок пройдено: {len(_results)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
