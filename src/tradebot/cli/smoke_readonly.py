"""НЕОБЯЗАТЕЛЬНАЯ read-only проверка подключения к Finam Trade API.

ВНИМАНИЕ:
- эта команда обращается к реальному API и ТРЕБУЕТ ВАШИ учётные данные:
  переменную окружения TRADE_API_SECRET (secret_key выпускается на портале
  в разделе "Токены"; docs/research.md S3b). Секрет никогда не печатается;
- команда READ-ONLY: выполняются только документированные методы чтения
  (аутентификация, детали токена, метаданные инструмента, история свечей,
  снимок котировки, снимок стакана). Выставление заявок не реализовано в проекте
  и не может быть вызвано этой командой;
- идентификаторы счетов НЕ печатаются (чувствительные данные) — выводится только
  количество счетов и признак readonly;
- автоматически (тестами, ручной проверкой, CI) команда НЕ запускается.

Запуск (PowerShell), только когда готовы:
  $env:TRADE_API_SECRET = "<ваш secret_key>"
  python -m tradebot.cli.smoke_readonly --symbol SBER@MISX --timeframe 5m --hours 24 --bar-timestamp-semantics bar_open

Флаг --bar-timestamp-semantics ОБЯЗАТЕЛЕН: документация не фиксирует, является ли
timestamp свечи началом или концом интервала (docs/research.md Q3). Сравните вывод
с фактическим расписанием торгов, чтобы выбрать значение, и зафиксируйте его
в конфиге (finam.bar_timestamp_semantics).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from ..config import SecretNotFound, resolve_optional, resolve_secret
from ..config.schema import FinamConfig
from ..data.bar import SUPPORTED_TIMEFRAMES
from ..finam.errors import FinamError
from ..finam.rest_client import FinamRestClient
from ..finam.source import FinamHistorySource, FinamSnapshotSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smoke_readonly", description=__doc__.split("\n")[0])
    parser.add_argument("--symbol", required=True, help="символ в формате ticker@mic, например SBER@MISX")
    parser.add_argument("--timeframe", choices=SUPPORTED_TIMEFRAMES, default="5m")
    parser.add_argument("--hours", type=int, default=24, help="глубина запроса истории в часах (по умолчанию 24)")
    parser.add_argument(
        "--bar-timestamp-semantics",
        choices=("bar_open", "bar_close"),
        required=True,
        help="ЯВНАЯ семантика timestamp свечей API (research Q3) — см. docstring",
    )
    parser.add_argument("--base-url", default="https://api.finam.ru")
    parser.add_argument(
        "--env-file",
        default=None,
        help="необязательный путь к .env (игнорируется git-ом); существующие переменные окружения имеют приоритет",
    )
    args = parser.parse_args(argv)

    if args.env_file:
        from ..config.env import load_dotenv_file

        loaded = load_dotenv_file(args.env_file)
        print(f"Из .env загружено переменных: {loaded} (значения не печатаются)")

    print("READ-ONLY проверка Finam Trade API. Заявки не выставляются (не реализовано).")
    if args.timeframe == "10m":
        print(
            "Таймфрейм 10m не поддерживается API (нет TIME_FRAME_M10, research §2.3). "
            "Используйте 5m и локальный ресэмплинг."
        )
        return 2

    try:
        secret = resolve_secret("TRADE_API_SECRET", label="SECRET_KEY")
    except SecretNotFound as exc:
        print(f"НЕТ УЧЁТНЫХ ДАННЫХ: {exc}")
        return 2

    client = FinamRestClient(secret, base_url=args.base_url)
    try:
        details = client.authenticate()
        print(
            f"[1/5] Аутентификация: OK (readonly={details.readonly}, "
            f"счетов доступно={len(details.account_ids)}, expires_at={details.expires_at.isoformat()})"
        )
        if not details.expires_at_parsed:
            print("       ВНИМАНИЕ: expires_at не распознан — токен будет перевыпускаться каждые 60 с")
        for perm in details.md_permissions:
            print(
                f"       md_permission: mic={perm.get('mic')}, quote_level={perm.get('quote_level')}, "
                f"delay_minutes={perm.get('delay_minutes')}"
            )

        asset = client.get_asset(args.symbol)
        print(
            f"[2/5] Инструмент {args.symbol}: name={asset.name}, quote_currency={asset.quote_currency}, "
            f"lot_size={asset.lot_size}, min_step={asset.min_step}, decimals={asset.decimals}"
        )

        account_id = resolve_optional("FINAM_ACCOUNT_ID", label="ACCOUNT_ID")
        if account_id:
            params = client.get_asset_params(args.symbol, account_id=account_id)
            print(
                f"[3/5] Торговые параметры: trade_lot_size={params.trade_lot_size}, is_tradable={params.is_tradable}"
            )
            if params.trade_lot_size == 0:
                print("       trade_lot_size=0 означает 'недоступно' (research §2.7) — торговые операции запрещены")
        else:
            print("[3/5] Торговые параметры: пропущено (FINAM_ACCOUNT_ID не задан; read-only проверка продолжается)")

        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=max(1, args.hours))
        source = FinamHistorySource(client, bar_timestamp_semantics=args.bar_timestamp_semantics)
        bars = source.fetch_history(args.symbol, args.timeframe, start, end)
        print(f"[4/5] История {args.timeframe} за {args.hours} ч: получено баров={len(bars)}")
        if bars:
            first, last = bars[0], bars[-1]
            print(
                f"       первый: {first.ts_open.isoformat()} close={first.close}; "
                f"последний: {last.ts_open.isoformat()} close={last.close}"
            )
            print(
                "       СЕМАНТИКА: метки нормализованы как "
                f"{args.bar_timestamp_semantics}. Сверьте время первого/последнего бара с фактическим "
                "расписанием торгов — при расхождении повторите с противоположным значением флага"
            )

        snapshots = FinamSnapshotSource(client)
        quote = snapshots.get_latest_quote(args.symbol)
        print(
            f"[5/5] Котировка: last={quote.last}, bid={quote.bid}, ask={quote.ask}, ts={quote.timestamp}"
        )
        book = snapshots.get_order_book(args.symbol)
        print(
            f"       Стакан (снимок): уровней={book.depth} (глубина документацией не гарантируется — research Q4)"
        )
        print("\nRESULT: PASS — read-only проверка выполнена, заявок не отправлялось")
        return 0
    except FinamError as exc:
        print(f"\nRESULT: FAIL — ошибка Finam API: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
