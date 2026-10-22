#!/usr/bin/env python3
"""Этап 2a: загрузка исторических свечей BTCUSDT с Bybit (public REST) и сохранение.

Использование (из корня проекта btc_bybit_bot):
    python scripts/download_history.py
    python scripts/download_history.py --start 2025-01-01 --end 2025-06-30

    # если загрузка прервалась (сеть / ошибка Bybit) - часть баров сохранена
    # в файл *_PARTIAL_*.csv; продолжение БЕЗ повторного скачивания начала:
    python scripts/download_history.py --resume data/raw/bybit_..._PARTIAL_fetched....csv

Создаваемые файлы (полная загрузка):
    data/raw/bybit_<category>_<symbol>_<interval>_<start>_<end>_fetched<UTC>.csv
    data/raw/....csv.metadata.json      (инструмент, категория, интервал, период, дата загрузки)
    data/processed/<то же имя>.clean.csv            (очищенные закрытые бары)
    data/processed/<то же имя>.clean.report.json    (отчёт очистки: дубли, gaps, незакрытые)

При прерывании загрузки сохраняется частичный набор с маркером _PARTIAL
и metadata.complete=false; такие файлы НЕ выбираются автоматически для
обучения/реплея - только для --resume.

Надёжность: транзиентные ошибки Bybit (retCode 10006/10016/10018, HTTP 429/5xx,
сетевые сбои) ретраятся клиентом с backoff; если запросы всё же исчерпаны,
уже загруженные бары прикрепляются к ошибке и сохраняются как частичные.

Только публичные данные; ключи API не используются и не требуются.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import clean_klines  # noqa: E402
from btc_bot.data.store import (  # noqa: E402
    dedupe_klines,
    klines_to_dataframe,
    load_klines_csv,
    load_metadata,
    save_klines_csv,
)
from btc_bot.data_sources.base import DataSourceError, Kline  # noqa: E402
from btc_bot.data_sources.bybit import BybitSource, period_end_ms, step_ms_approx  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_history")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Загрузка исторических свечей Bybit (public).")
    p.add_argument("--config", default=None, help="путь к YAML-конфигу (по умолчанию config/default.yaml)")
    p.add_argument("--start", default=None, help="начало периода UTC, YYYY-MM-DD (по умолчанию data.history_start из конфига)")
    p.add_argument("--end", default=None, help="конец периода UTC, YYYY-MM-DD (по умолчанию из partial-метаданных или сейчас)")
    p.add_argument("--max-pages", type=int, default=None, help="ограничение числа REST-запросов (для отладки)")
    p.add_argument("--resume", default=None,
                   help="продолжить прерванную загрузку: путь к частичному raw-файлу "
                        "data/raw/*_PARTIAL_*.csv (или его .metadata.json)")
    return p.parse_args()


def interval_token(interval: str) -> str:
    return interval if interval in ("D", "W", "M") else f"{interval}m"


def make_base_name(category: str, symbol: str, interval: str,
                   start_ms: int, end_ms: int, partial: bool = False) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mid = "_PARTIAL" if partial else ""
    return (
        f"bybit_{category}_{symbol}_{interval_token(interval)}_"
        f"{pd.Timestamp(start_ms, unit='ms', tz='UTC'):%Y%m%d}_"
        f"{pd.Timestamp(end_ms, unit='ms', tz='UTC'):%Y%m%d}{mid}_fetched{stamp}"
    )


def klines_from_df(df: pd.DataFrame) -> list[Kline]:
    """DataFrame raw-формата -> список Kline (для слияния при --resume)."""
    return [
        Kline(
            start_ms=int(r.start_ms), open=float(r.open), high=float(r.high),
            low=float(r.low), close=float(r.close), volume=float(r.volume),
            turnover=float(r.turnover), is_closed=bool(r.is_closed),
        )
        for r in df.itertuples(index=False)
    ]


def persist_outputs(
    klines: list[Kline],
    base_name: str,
    raw_dir: Path,
    proc_dir: Path,
    *,
    category: str,
    symbol: str,
    interval: str,
    extra_meta: Optional[dict] = None,
    complete: bool = True,
):
    """Сохранить raw CSV + metadata + clean CSV + отчёт очистки. Возвращает dict путей и отчёт."""
    raw_path = raw_dir / f"{base_name}.csv"
    clean_path = proc_dir / f"{base_name}.clean.csv"
    report_path = proc_dir / f"{base_name}.clean.report.json"

    meta_extra = dict(extra_meta or {})
    meta_extra["complete"] = bool(complete)

    csv_path, meta_path = save_klines_csv(
        klines, raw_path, source="bybit", category=category, symbol=symbol,
        interval=interval, extra_meta=meta_extra,
    )
    df_raw = klines_to_dataframe(klines)
    df_clean, report = clean_klines(df_raw, interval)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    return {
        "csv_path": csv_path, "meta_path": meta_path,
        "clean_path": clean_path, "report_path": report_path, "report": report,
    }


def load_partial_for_resume(path: Path) -> tuple[list[Kline], dict]:
    """Загрузить частичный raw-файл (+metadata). Возвращает (klines, meta)."""
    if path.suffix == ".json":  # передали metadata.json -> берём соседний CSV
        path = path.with_name(path.name.replace(".metadata.json", ".csv"))
    if not path.exists():
        raise FileNotFoundError(f"Файл для --resume не найден: {path}")
    df = load_klines_csv(path)
    if df.empty:
        raise ValueError(f"Частичный файл пуст: {path}")
    meta_path = path.with_suffix(path.suffix + ".metadata.json")
    meta = load_metadata(meta_path) if meta_path.exists() else {}
    return klines_from_df(df), meta


def save_partial_and_report_error(
    err: Exception, klines_before: list[Kline], base_name: str,
    raw_dir: Path, proc_dir: Path, *, category: str, symbol: str, interval: str,
    requested_start_ms: int, requested_end_ms: int,
) -> Optional[Path]:
    """Сохранить то, что успели скачать (если есть), и напечатать инструкцию --resume."""
    partial = list(klines_before) + list(getattr(err, "partial_klines", []) or [])
    if not partial:
        return None
    # дедупликация (закрытая свеча предпочтительнее) на случай --resume
    klines = dedupe_klines(partial)

    partial_base = f"{base_name}_PARTIAL" if "_PARTIAL" not in base_name else base_name
    try:
        out = persist_outputs(
            klines, partial_base, raw_dir, proc_dir,
            category=category, symbol=symbol, interval=interval,
            complete=False,
            extra_meta={
                "requested_start_ms": requested_start_ms,
                "requested_end_ms": requested_end_ms,
                "interrupted_by_error": str(err),
                "pages_fetched": getattr(err, "pages_fetched", None),
            },
        )
    except Exception as save_err:  # noqa: BLE001 - важнее сообщить основную ошибку
        log.error("Не удалось сохранить частичные данные: %s", save_err)
        return None

    print()
    print("=== ЗАГРУЗКА ПРЕРВАНА, ЧАСТЬ ДАННЫХ СОХРАНЕНА ===")
    print(f"Ошибка:                {err}")
    print(f"Сохранено свечей:      {len(klines)} (последняя: "
          f"{pd.Timestamp(klines[-1].start_ms, unit='ms', tz='UTC')})")
    print(f"Частичный файл:        {out['csv_path']}")
    print("Продолжить загрузку (без повторного скачивания начала):")
    print(f"    python scripts/download_history.py --resume \"{out['csv_path']}\"")
    print("Частичные файлы (_PARTIAL) НЕ используются для обучения автоматически.")
    return out["csv_path"]


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    ex = cfg.exchange

    raw_dir = cfg.resolve(cfg.data.raw_dir)
    proc_dir = cfg.resolve(cfg.data.processed_dir)

    source = BybitSource(
        base_url=ex.rest_base_url,
        request_timeout_s=ex.request_timeout_s,
        min_request_interval_s=ex.min_request_interval_s,
        max_retries=ex.max_retries,
    )

    resumed_klines: list[Kline] = []
    resumed_meta: dict = {}

    if args.resume:
        # -- режим продолжения прерванной загрузки --------------------------
        try:
            resumed_klines, resumed_meta = load_partial_for_resume(Path(args.resume))
        except (FileNotFoundError, ValueError, KeyError) as e:
            log.error("--resume: %s", e)
            return 2
        for field in ("category", "symbol", "interval"):
            if resumed_meta.get(field) and resumed_meta[field] != getattr(ex, field):
                log.error("--resume: файл от %s/%s/%s, а конфиг - %s/%s/%s",
                          resumed_meta.get("category"), resumed_meta.get("symbol"),
                          resumed_meta.get("interval"), ex.category, ex.symbol, ex.interval)
                return 2
        start_ms = int(resumed_meta.get("requested_start_ms", resumed_klines[0].start_ms))
        fetch_from_ms = period_end_ms(resumed_klines[-1].start_ms, ex.interval)
        if args.end:
            end_ms = int((pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000) - 1
        elif resumed_meta.get("requested_end_ms"):
            end_ms = int(resumed_meta["requested_end_ms"])
        else:
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_str = pd.Timestamp(start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        log.info("ПРОДОЛЖЕНИЕ загрузки: %d баров уже есть (последний %s), докачиваю с %s",
                 len(resumed_klines),
                 pd.Timestamp(resumed_klines[-1].start_ms, unit="ms", tz="UTC"),
                 pd.Timestamp(fetch_from_ms, unit="ms", tz="UTC"))
        if args.start:
            log.warning("--start игнорируется в режиме --resume (берётся из метаданных частичного файла)")
    else:
        start_str = args.start or cfg.data.history_start
        start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
        fetch_from_ms = start_ms
        if args.end:
            end_ms = int((pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000) - 1
        else:
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if end_ms <= start_ms:
        log.error("end должен быть позже start")
        return 2
    if end_ms < fetch_from_ms:
        log.error("конец периода раньше последнего скачанного бара - загружать нечего")
        return 0 if args.resume else 2

    base_name = make_base_name(ex.category, ex.symbol, ex.interval, start_ms, end_ms)

    def progress(pages: int, rows: int, last_start_ms: int) -> None:
        if pages % 25 == 0 or pages == 1:
            log.info("страниц: %d, свечей: %d, последняя: %s",
                     pages, rows, pd.Timestamp(last_start_ms, unit="ms", tz="UTC"))

    log.info("Загрузка %s/%s interval=%s с %s по %s ...",
             ex.category, ex.symbol, ex.interval,
             pd.Timestamp(fetch_from_ms, unit="ms", tz="UTC"), pd.Timestamp(end_ms, unit="ms", tz="UTC"))
    try:
        new_klines = source.fetch_klines(
            symbol=ex.symbol,
            category=ex.category,
            interval=ex.interval,
            start_ms=fetch_from_ms,
            end_ms=end_ms,
            on_progress=progress,
            max_pages=args.max_pages,
        )
    except DataSourceError as e:
        log.error("Ошибка загрузки: %s", e)
        saved = save_partial_and_report_error(
            e, resumed_klines, base_name, raw_dir, proc_dir,
            category=ex.category, symbol=ex.symbol, interval=ex.interval,
            requested_start_ms=start_ms, requested_end_ms=end_ms,
        )
        return 3 if saved else 2

    # склейка: закрытая версия бара предпочтительнее незакрытой из частичного файла
    klines = dedupe_klines(resumed_klines + new_klines) if resumed_klines else new_klines
    if not klines:
        log.error("Сервер не вернул ни одной свечи - данные НЕ сохранены (ничего не фабрикуется).")
        return 2

    n_unclosed = sum(1 for k in klines if not k.is_closed)

    extra_meta = {
        "requested_start_ms": start_ms, "requested_end_ms": end_ms,
        "archive_gaps_note": (
            "coverage<100% означает, что в архиве биржи отсутствуют бары "
            "(границы архивных сегментов/технические пропуски). Данные НЕ заполнялись."),
    }
    if args.resume:
        extra_meta.update({
            "resumed_from": Path(args.resume).name,
            "resume_bars_before": len(resumed_klines),
            "resume_fetch_start_ms": fetch_from_ms,
        })

    out = persist_outputs(
        klines, base_name, raw_dir, proc_dir,
        category=ex.category, symbol=ex.symbol, interval=ex.interval,
        extra_meta=extra_meta, complete=True,
    )
    report = out["report"]
    csv_path, clean_path, report_path = out["csv_path"], out["clean_path"], out["report_path"]

    # покрытие/гэпы считаем после дедупликации (clean), метаданные дополняем
    step = step_ms_approx(ex.interval)
    span_expected = (report.period_end_ms - report.period_start_ms) // step + 1 if report.n_output else 0
    coverage_pct = (report.n_output / span_expected * 100.0) if span_expected else 0.0
    requested_expected = max(1, (end_ms - start_ms) // step)
    requested_coverage_pct = report.n_output / requested_expected * 100.0
    tail_lag_ms = end_ms - (report.period_end_ms or end_ms)

    meta = load_metadata(out["meta_path"])
    meta.update({
        "coverage_pct_of_fetched_span": round(coverage_pct, 4),
        "coverage_pct_of_requested_span": round(requested_coverage_pct, 4),
        "archive_gaps": [g.to_dict() for g in report.gaps],
        "archive_gaps_missing_bars": report.total_missing_bars,
    })
    with open(out["meta_path"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    log.info("RAW сохранён: %s (%d строк, незакрытых на момент загрузки: %d)", csv_path, len(klines), n_unclosed)
    log.info("METADATA: %s", out["meta_path"])
    log.info("CLEAN сохранён: %s", clean_path)
    log.info("ОТЧЁТ ОЧИСТКИ: %s", report_path)
    print()
    print("=== ИТОГ ЗАГРУЗКИ ===")
    print(f"Инструмент:            {ex.category}/{ex.symbol}, interval={ex.interval}")
    end_str = args.end or pd.Timestamp(end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    print(f"Период запроса:        {start_str} .. {end_str} (UTC)")
    if args.resume:
        print(f"Продолжение файла:     {Path(args.resume).name} (+{len(new_klines)} новых баров)")
    print(f"Получено свечей:       {len(klines)}")
    print(f"Дубликаты удалены:     {report.n_duplicates_removed}")
    print(f"Некорректные удалены:  {report.n_invalid_removed}")
    print(f"Незакрытые исключены:  {report.n_unconfirmed_excluded}")
    print(f"Диапазоны пропусков:   {len(report.gaps)} (всего отсутствующих баров: {report.total_missing_bars})")
    print(f"Чистых баров:          {report.n_output}")
    print(f"Покрытие:              {coverage_pct:.2f}% загруженного диапазона "
          f"({requested_coverage_pct:.2f}% запрошенного периода)")
    if report.gaps:
        print("ОТСУТСТВУЮЩИЕ ДИАПАЗОНЫ (архив биржи неполон; бары НЕ заполнялись):")
        for g in report.gaps[:10]:
            a = pd.Timestamp(g.after_start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
            b = pd.Timestamp(g.until_start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
            print(f"  {a} -> {b}  ({g.missing_bars} баров)")
        if len(report.gaps) > 10:
            print(f"  ... и ещё {len(report.gaps) - 10} диапазон(ов)")
    if tail_lag_ms > max(12 * step, 3_600_000):
        print(f"ВНИМАНИЕ: последняя свеча на {tail_lag_ms / 3_600_000:.1f} ч раньше конца запроса - "
              f"проверьте, не оборвалась ли загрузка")
    print(f"Файлы:                 {csv_path.name}, {clean_path.name}, {report_path.name}")
    print("Пропуски НЕ заполнялись - данные сохранены как получены от биржи.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
