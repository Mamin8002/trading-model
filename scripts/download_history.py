#!/usr/bin/env python3
"""Этап 2a: загрузка исторических свечей BTCUSDT с Bybit (public REST) и сохранение.

Использование (из корня проекта btc_bybit_bot):
    python scripts/download_history.py
    python scripts/download_history.py --start 2025-01-01 --end 2025-06-30

Создаваемые файлы:
    data/raw/bybit_<category>_<symbol>_<interval>_<start>_<end>_fetched<UTC>.csv
    data/raw/....csv.metadata.json      (инструмент, категория, интервал, период, дата загрузки)
    data/processed/<то же имя>.clean.csv            (очищенные закрытые бары)
    data/processed/<то же имя>.clean.report.json    (отчёт очистки: дубли, gaps, незакрытые)

Только публичные данные; ключи API не используются и не требуются.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from btc_bot.config import load_config  # noqa: E402
from btc_bot.data.prep import clean_klines  # noqa: E402
from btc_bot.data.store import save_klines_csv  # noqa: E402
from btc_bot.data_sources.base import DataSourceError  # noqa: E402
from btc_bot.data_sources.bybit import BybitSource  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_history")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Загрузка исторических свечей Bybit (public).")
    p.add_argument("--config", default=None, help="путь к YAML-конфигу (по умолчанию config/default.yaml)")
    p.add_argument("--start", default=None, help="начало периода UTC, YYYY-MM-DD (по умолчанию data.history_start из конфига)")
    p.add_argument("--end", default=None, help="конец периода UTC, YYYY-MM-DD (по умолчанию сейчас)")
    p.add_argument("--max-pages", type=int, default=None, help="ограничение числа REST-запросов (для отладки)")
    return p.parse_args()


def interval_token(interval: str) -> str:
    return interval if interval in ("D", "W", "M") else f"{interval}m"


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    ex = cfg.exchange

    start_str = args.start or cfg.data.history_start
    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    if args.end:
        end_ms = int((pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if end_ms <= start_ms:
        log.error("end должен быть позже start")
        return 2

    source = BybitSource(
        base_url=ex.rest_base_url,
        request_timeout_s=ex.request_timeout_s,
        min_request_interval_s=ex.min_request_interval_s,
        max_retries=ex.max_retries,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = (
        f"bybit_{ex.category}_{ex.symbol}_{interval_token(ex.interval)}_"
        f"{pd.Timestamp(start_ms, unit='ms', tz='UTC'):%Y%m%d}_"
        f"{pd.Timestamp(end_ms, unit='ms', tz='UTC'):%Y%m%d}_fetched{stamp}"
    )
    raw_dir = cfg.resolve(cfg.data.raw_dir)
    proc_dir = cfg.resolve(cfg.data.processed_dir)
    raw_path = raw_dir / f"{base_name}.csv"
    clean_path = proc_dir / f"{base_name}.clean.csv"
    report_path = proc_dir / f"{base_name}.clean.report.json"

    def progress(pages: int, rows: int, last_start_ms: int) -> None:
        if pages % 25 == 0 or pages == 1:
            log.info("страниц: %d, свечей: %d, последняя: %s",
                     pages, rows, pd.Timestamp(last_start_ms, unit="ms", tz="UTC"))

    log.info("Загрузка %s/%s interval=%s с %s по %s ...",
             ex.category, ex.symbol, ex.interval,
             pd.Timestamp(start_ms, unit="ms", tz="UTC"), pd.Timestamp(end_ms, unit="ms", tz="UTC"))
    try:
        klines = source.fetch_klines(
            symbol=ex.symbol,
            category=ex.category,
            interval=ex.interval,
            start_ms=start_ms,
            end_ms=end_ms,
            on_progress=progress,
            max_pages=args.max_pages,
        )
    except DataSourceError as e:
        log.error("Ошибка загрузки: %s", e)
        return 2

    if not klines:
        log.error("Сервер не вернул ни одной свечи - данные НЕ сохранены (ничего не фабрикуется).")
        return 2

    n_unclosed = sum(1 for k in klines if not k.is_closed)
    csv_path, meta_path = save_klines_csv(
        klines,
        raw_path,
        source="bybit",
        category=ex.category,
        symbol=ex.symbol,
        interval=ex.interval,
        extra_meta={"requested_start_ms": start_ms, "requested_end_ms": end_ms},
    )
    log.info("RAW сохранён: %s (%d строк, незакрытых на момент загрузки: %d)", csv_path, len(klines), n_unclosed)
    log.info("METADATA: %s", meta_path)

    df_raw = pd.read_csv(csv_path)
    df_clean, report = clean_klines(df_raw, ex.interval)
    df_clean.to_csv(clean_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    log.info("CLEAN сохранён: %s", clean_path)
    log.info("ОТЧЁТ ОЧИСТКИ: %s", report_path)
    print()
    print("=== ИТОГ ЗАГРУЗКИ ===")
    print(f"Инструмент:            {ex.category}/{ex.symbol}, interval={ex.interval}")
    print(f"Период запроса:        {start_str} .. {args.end or 'now'} (UTC)")
    print(f"Получено свечей:       {len(klines)}")
    print(f"Дубликаты удалены:     {report.n_duplicates_removed}")
    print(f"Некорректные удалены:  {report.n_invalid_removed}")
    print(f"Незакрытые исключены:  {report.n_unconfirmed_excluded}")
    print(f"Диапазоны пропусков:   {len(report.gaps)} (всего отсутствующих баров: {report.total_missing_bars})")
    print(f"Чистых баров:          {report.n_output}")
    print(f"Файлы:                 {csv_path.name}, {clean_path.name}, {report_path.name}")
    print("Пропуски НЕ заполнялись - данные сохранены как получены от биржи.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
