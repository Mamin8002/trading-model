# Research: Finam Trade API и референсный проект WISEPLAT

Дата изучения: **2026-10-21** (все ссылки проверены в этот день).
Статус: черновик исследования перед проектированием. Код не изменялся.

Обозначения:
- **[ПОДТВЕРЖДЕНО]** — факт прочитан в официальной документации / proto / спецификации.
- **[НЕИЗВЕСТНО]** — официальные источники не дают ответа; см. раздел «Открытые вопросы».
- **[РАСХОЖДЕНИЕ]** — источники противоречат друг другу или требованиям задачи.

---

## 1. Источники

### 1.1 Официальная документация Finam Trade API

Важно: домен из ТЗ `https://tradeapi.finam.ru/` теперь **редиректит на `https://api.finam.ru/`** (проверено: HTTP 200, final URL `https://api.finam.ru/`; старый путь `https://tradeapi.finam.ru/docs/` отдаёт 404). Актуальная документация живёт на `api.finam.ru`.

| # | Источник | URL | Что использовано |
|---|---|---|---|
| S1 | Лендинг Trade API | https://api.finam.ru/ (редирект с https://tradeapi.finam.ru/) | состав продуктов, playground, Postman-коллекция |
| S2 | База знаний, каталог курсов | https://api.finam.ru/getting-started/ | структура обучения, FAQ |
| S3 | Курс «Finam Trade API: от первого запроса до надёжного клиента» (13 лонгридов, обновлён 28.09.2026) | https://api.finam.ru/getting-started/bystryy-start/ | архитектура API, аутентификация, данные, ордера, надёжность |
| S3a | — «Как устроен Trade API» | https://api.finam.ru/getting-started/bystryy-start/kak-ustroen-trade-api/ | REST/WS/gRPC, secret_key vs access_key, лимит 200 зап/мин |
| S3b | — «Подготовить безопасное окружение» | https://api.finam.ru/getting-started/bystryy-start/podgotovit-bezopasnoe-okruzhenie/ | демо-счёт, выпуск токенов, readonly/full доступ |
| S3c | — «Получить access_key и увидеть счёт» | https://api.finam.ru/getting-started/bystryy-start/poluchit-access-key-i-uvidet-schet/ | POST /v1/sessions, /v1/sessions/details, expires_at (~15 мин), decimal-объекты, типичные ошибки |
| S3d | — «Найти инструмент и проверить торговые параметры» | https://api.finam.ru/getting-started/bystryy-start/nayti-instrument-i-proverit-torgovye-parametry/ | формат ticker@mic, lot_size vs trade_lot_size, min_step/decimals, справочники |
| S3e | — «Получить котировку и историю» | https://api.finam.ru/getting-started/bystryy-start/poluchit-kotirovku-i-istoriyu/ | GET bars/quotes latest, глубина истории по таймфреймам, «текущая свеча ещё меняется» |
| S3f | — «Получать данные через WebSocket» | https://api.finam.ru/getting-started/bystryy-start/poluchat-dannye-cherez-websocket/ | wss://api.finam.ru/ws, типы подписок, HANDSHAKE_SUCCESS, конверты DATA/ERROR/EVENT, decimal в WS |
| S3g | — «Безопасно выставить и отменить заявку» | https://api.finam.ru/getting-started/bystryy-start/bezopasno-vystavit-i-otmenit-zayavku/ | PlaceOrder/CancelOrder, client_order_id ≤ 20 симв., статусы заявок |
| S3h | — «Следить за собственными заявками и сделками» | https://api.finam.ru/getting-started/bystryy-start/sledit-za-sobstvennymi-zayavkami-i-sdelkami/ | потоки ORDERS/TRADES, reconciliation |
| S3i | — «Собрать учебного торгового клиента» | https://api.finam.ru/getting-started/bystryy-start/sobrat-uchebnogo-torgovogo-klienta/ | dry-run по умолчанию, FINAM_ENABLE_ORDERS, чего нет в учебном клиенте |
| S3j | — «Практические рецепты надёжного клиента» | https://api.finam.ru/getting-started/bystryy-start/prakticheskie-recepty-nadezhnogo-klienta/ | backoff 1→2→4→8→16→30 с, таблица безопасных повторов, лоты→штуки, паттерн завершённой свечи, предохранители |
| S3k | — «Перейти на gRPC и официальные SDK» | https://api.finam.ru/getting-started/bystryy-start/pereyti-na-grpc-i-oficialnye-sdk/ | finam-sdk (Python), SubscribeJwtRenewal, is_data_snapshot |
| S3l | — «Диагностика ошибок и дальнейшие шаги» | https://api.finam.ru/getting-started/bystryy-start/diagnostika-oshibok-i-dalneyshie-shagi/ | карта HTTP-ошибок, чек-листы WS и отклонённых заявок, «потерянный ответ PlaceOrder» |
| S4 | FAQ «Что делать, когда API отвечает не так» (5 тем, обновлён 28.09.2026) | https://api.finam.ru/getting-started/chto-delat-kogda-api-otvechaet-ne-tak/ + подстраницы limity-zaprosov, podpiski-i-vosstanovlenie, chego-ne-delat, pogranichnye-sluchai | 200 зап/мин **на метод**, 429-политика, восстановление потока, trade_lot_size=0, дубли свечей |
| S5 | REST-справочник (OpenAPI) | https://api.finam.ru/docs/rest/ (+ markdown-версии методов, напр. /docs/rest/marketdataservice_bars.md, /docs/rest/marketdataservice_orderbook.md, /docs/rest/ordersservice_placeorder.md) | точные пути, параметры, enum TimeFrame, коды ответов |
| S6 | Changelog API | https://api.finam.ru/changelog/ | текущая версия API **v2.23.0 от 01.10.2026**; GuardTime для SL/TP; commission в AccountTrade |
| S7 | Токены (портал) | https://api.finam.ru/tokens/ | место выпуска secret_key и создания демо-счёта (страница за логином) |

### 1.2 Официальный репозиторий SDK

| # | Источник | URL | Версия/состояние на 2026-10-21 |
|---|---|---|---|
| S8 | Репозиторий FinamWeb/finam-trade-api | https://github.com/FinamWeb/finam-trade-api | ветка `main`, последний push 01.10.2026 (merge PR #85 `release_2.23.0`, commit `52c95c8`), 140 коммитов, создан 03.04.2025 |
| S8a | README SDK | https://github.com/FinamWeb/finam-trade-api/blob/main/README.md | Python SDK `finam-sdk`, Node `@finam/trade-api`, Kotlin, Go, JS-биндинги; примеры стратегий |
| S8b | Python SDK README | https://github.com/FinamWeb/finam-trade-api/blob/main/sdk/python/README.md | FinamClient/AsyncFinamClient, сервисы, ошибки, retry-политика, for_testing() |
| S8c | Python SDK CHANGELOG | https://github.com/FinamWeb/finam-trade-api/blob/main/sdk/python/CHANGELOG.md | история релизов; 2.19.1 — 02.09.2026 |
| S8d | PyPI `finam-sdk` | https://pypi.org/pypi/finam-sdk/json | **версия 2.23.0**, requires_python `>=3.10`, зависимости: `grpcio>=1.83.0`, `protobuf>=7.35.1`, `googleapis-common-protos>=1.62` |
| S8e | Migration guide | https://github.com/FinamWeb/finam-trade-api/blob/main/MIGRATION_GUIDE.md | соответствие старый API → новый API |
| S8f | Proto-контракты | https://github.com/FinamWeb/finam-trade-api/tree/main/proto/grpc/tradeapi/v1 (marketdata_service.proto, orders_service.proto, trade.proto, assets, accounts, auth, metrics) | enum TimeFrame, OrderType, TimeInForce, OrderStatus, сообщения Bar/Quote/OrderBook/Order |
| S8g | AsyncAPI-спецификация WebSocket | https://github.com/FinamWeb/finam-trade-api/blob/main/specs/asyncapi/asyncapi-v1.0.0.yaml | wss api.finam.ru:443 `/ws`, команды SUBSCRIBE/UNSUBSCRIBE/UNSUBSCRIBE_ALL, конверты, SubscriptionType |
| S8h | Swagger (REST) | https://github.com/FinamWeb/finam-trade-api/blob/main/docs/swagger/api.swagger.json | полный список REST-путей (32 операции) |
| S8i | Пример стратегии SMA 9/30 (эталонный паттерн) | https://github.com/FinamWeb/finam-trade-api/blob/main/examples/strategies/sma_crossover/README.md | обработка завершённых свечей, dry-run, защитные проверки, «ровно один счёт» |

### 1.3 Референсный проект WISEPLAT

| # | Источник | URL | Состояние на 2026-10-21 |
|---|---|---|---|
| S9 | Репозиторий WISEPLAT/Hackathon-Finam-NN-Trade-Robot | https://github.com/WISEPLAT/Hackathon-Finam-NN-Trade-Robot | создан 05.05.2023, **последний push 21.11.2023** (не обновлялся ~3 года), 68 звёзд, 42 форка, ветка `master`; лицензия в метаданных GitHub не указана, в README заявлена MIT |
| S9a | README | .../blob/master/README.md | назначение, пайплайн 1–7, признание автора: «стратегия не даст заработать» |
| S9b | requirements.txt | .../blob/master/requirements.txt | aiohttp, aiomoex, pandas, matplotlib, tensorflow, finam-trade-api |
| S9c | my_config/trade_config.py | .../blob/master/my_config/trade_config.py | таймфреймы M1(вход)/M10(выход), тикеры SBER/VTBR, TQBR, часы 10:00–23:50, параметры картинок |
| S9d | my_config/Config_example.py | .../blob/master/my_config/Config_example.py | ClientIds + AccessToken **через comon.ru (старый API)** |
| S9e | 1_get_historical_data_for_strategy_from_moex.py | .../blob/master/1_get_historical_data_for_strategy_from_moex.py | загрузка свечей с MOEX через aiomoex → CSV `datetime,open,high,low,close,volume` |
| S9f | functions.py | .../blob/master/functions.py | маппинг таймфреймов MOEX (M1/M10/H1/…), расчёт future_key, классификация по проценту |
| S9g | 7_live_strategy.py | .../blob/master/7_live_strategy.py | live-цикл: поллинг MOEX каждые 10 с, CNN-предсказание, рыночные ордера через FinamPy |
| S9h | csv/SBER_M10.csv (выборка) | .../blob/master/csv/SBER_M10.csv | 45 004 строки, период 2021-01-04…2023-05-03, формат совпадает с ТЗ |
| S9i | Сторонние библиотеки проекта | https://github.com/cia76/FinamPy, https://github.com/DBoyara/FinamTradeApiPy | неофициальные клиенты **старого** Trade API |

Локальные копии загруженных источников (raw-файлы) лежат в служебном каталоге `.research/` рабочей области — это сырьё для проверки цитат, не часть поставки.

---

## 2. Подтверждённые факты о Finam Trade API

### 2.1 Общая архитектура **[ПОДТВЕРЖДЕНО]** (S3a)

- Три интерфейса к одному бэкенду: **REST** (запрос-ответ, снимки состояния), **WebSocket** (`wss://api.finam.ru/ws`, JSON-события), **gRPC/ConnectRPC** (типизированные unary и streaming-методы; основа REST — grpc-gateway).
- Базовый URL: `https://api.finam.ru`; WS: `wss://api.finam.ru/ws`; gRPC endpoint по умолчанию в SDK — тот же хост.
- Официальное правило: «состояние — разовым методом, изменения — потоком; после обрыва потока снова запросить снимок и сверить».

### 2.2 Аутентификация **[ПОДТВЕРЖДЕНО]** (S3a, S3b, S3c, S5)

- Два разных ключа:
  - **secret_key** — долгоживущий, выпускается в разделе «Токены» портала (https://api.finam.ru/tokens/), показывается один раз; хранится как пароль, только в env/секретах.
  - **access_key** — короткоживущий (~**15 минут**, точный срок в ответе `expires_at`), получается обменом secret_key.
- REST: `POST /v1/sessions` с телом `{"secret": ...}` → `{"token": access_key}`. Рабочие запросы: заголовок `Authorization: Bearer <access_key>` (допустим и голый токен без `Bearer`).
- `POST /v1/sessions/details` (тело `{"token": ...}`) → `account_ids[]`, `readonly`, `created_at`, `expires_at`, `md_permissions[]` (quote_level, **delay_minutes**, mic/country) — т.е. права на рыночные данные (включая задержку) привязаны к токену.
- gRPC: `AuthService.Auth`, `AuthService.TokenDetails`, поток `AuthService.SubscribeJwtRenewal` — Python SDK обновляет access_key автоматически в фоне (S8b, S3k).
- Обновление access_key рекомендуется по `expires_at`/потоку renewal, а НЕ по таймеру и НЕ по ловле 401 (S4 pogranichnye-sluchai).
- Ошибки: 401 — истёк/неверен access_key; 403 — нет прав; `account_ids` пуст — при выпуске токена не выбран счёт.

### 2.3 Исторические данные (свечи) **[ПОДТВЕРЖДЕНО]** (S5 marketdataservice_bars.md, S8f, S3e)

- REST: `GET /v1/instruments/{symbol}/bars?timeframe=TIME_FRAME_*&interval.start_time=...&interval.end_time=...`
  - `start_time` включительно, `end_time` исключительно;
  - symbol в формате `ticker@mic` (например `SBER@MISX`);
  - ответ: `bars[] {timestamp, open/high/low/close/volume (decimal-объекты {"value":"..."}), is_data_snapshot}`.
- gRPC: `MarketDataService.Bars` (тот же контракт).
- **Таймфреймы и глубина истории** (enum `TimeFrame`, идентичен в REST/gRPC/WS):

| Enum | Значение | Глубина |
|---|---|---|
| TIME_FRAME_M1 | 1 минута | **7 дней** |
| TIME_FRAME_M5 | 5 минут | 30 дней |
| TIME_FRAME_M15 | 15 минут | 30 дней |
| TIME_FRAME_M30 | 30 минут | 30 дней |
| TIME_FRAME_H1 | 1 час | 30 дней |
| TIME_FRAME_H2/H4/H8 | 2/4/8 часов | 30 дней |
| TIME_FRAME_D | день | 365 дней |
| TIME_FRAME_W/MN/QR | неделя/месяц/квартал | 5 лет |

- **[РАСХОЖДЕНИЕ] Таймфрейма M10 в API нет** — ни в REST, ни в gRPC, ни в WS-подписке BARS (проверено по enum в `marketdata_service.proto` и по `marketdataservice_bars.md`). Требуемый ТЗ таймфрейм 10m недоступен нативно → должен строиться ресэмплингом локально (см. architecture.md).
- Официальная рекомендация: закрытые свечи кэшировать локально, дозапрашивать только новое окно; решения принимать только по завершённым свечам (последняя свеча мутирует).
- **[НЕИЗВЕСТНО]** Семантика `Bar.timestamp` — начало или конец интервала свечи — в документации не stated. Требует эмпирической проверки (см. открытые вопросы Q3).

### 2.4 Потоковые данные **[ПОДТВЕРЖДЕНО]** (S3f, S8g, S8f, S3h)

- WebSocket `wss://api.finam.ru/ws`, AsyncAPI 3.0 (`specs/asyncapi/asyncapi-v1.0.0.yaml`):
  - команды клиента: `{action: SUBSCRIBE|UNSUBSCRIBE|UNSUBSCRIBE_ALL, type, data, token}`; токен можно в заголовке подключения или в поле `token` команды;
  - конверты сервера: `DATA` (subscription_type + payload), `ERROR` (error_info: code/type/message), `EVENT` (event_info: HANDSHAKE_SUCCESS, CONNECTION_CLOSED…);
  - подписываться только после `HANDSHAKE_SUCCESS`;
  - ошибка может лежать и внутри `payload.error` — проверять оба уровня;
  - decimal в WS — обычные строки (`"last":"313.58"`), в REST — объекты `{"value":"313.58"}`, в gRPC — `google.type.Decimal`. Модели между протоколами не переносимы без преобразования.
- Типы подписок (S3f): `QUOTES`, `ORDER_BOOK`, `BARS`, `INSTRUMENT_TRADES`, `ORDERS` (свои заявки), `TRADES` (свои сделки), `ACCOUNT` (состояние счёта).
  - Примечание: в enum `SubscriptionType` asyncapi-v1.0.0 перечислены ORDERS, TRADES, QUOTES, ORDER_BOOK, BARS, INSTRUMENT_TRADES; тип ACCOUNT присутствует в курсе и в схемах `SubscribeAccountResponse`, но отсутствует в enum AsyncAPI — **[РАСХОЖДЕНИЕ]** внутри документации (Q7).
- BARS-подписка: `{symbol, timeframe}` — тот же enum TimeFrame (т.е. M10 тоже недоступен в потоке). Поток может повторно присылать свечу с той же меткой (обновление незакрытой свечи); завершённой свеча считается только после появления свечи с более поздним timestamp (S3j Рецепт 6, S4 pogranichnye-sluchai).
- gRPC-стримы (MarketDataService): `SubscribeQuote` (список символов), `SubscribeOrderBook` (один символ), `SubscribeLatestTrades`, `SubscribeBars`; (OrdersService): `SubscribeOrders`, `SubscribeTrades`, `SubscribeOrderTrade` (двунаправленный). Признак `is_data_snapshot` отличает начальный снимок от инкрементальных обновлений.
- Обрыв потока: пропущенные события НЕ доставляются задним числом; восстановление = новый access_key → подключение → handshake → переподписки → REST-снимок → сверка (reconciliation) с локальным состоянием (S3j Рецепт 1, S4 podpiski-i-vosstanovlenie).

### 2.5 Стакан **[ПОДТВЕРЖДЕНО]** (S5 marketdataservice_orderbook.md, S8f)

- REST-снимок: `GET /v1/instruments/{symbol}/orderbook` → `orderbook.rows[] {price, buy_size|sell_size (oneof), action (ACTION_ADD/REMOVE/UPDATE), mpid, timestamp}`.
- Поток: WS `ORDER_BOOK` / gRPC `SubscribeOrderBook` — `StreamOrderBook {symbol, rows[] (с action-инкрементами), is_data_snapshot}`.
- Маркетинг лендинга заявляет «прямой доступ к биржевым стаканам (L2)» (S1).
- **[НЕИЗВЕСТНО]** Гарантированная глубина стакана (число уровней) и частота обновлений в документации не указаны (Q4).

### 2.6 Операции с ордерами **[ПОДТВЕРЖДЕНО]** (S5 ordersservice_placeorder.md, S8f orders_service.proto, S3g, S6)

REST-поверхность OrdersService/AccountsService (полный список из swagger, S8h):

| Операция | Метод |
|---|---|
| Выставить заявку | `POST /v1/accounts/{account_id}/orders` (PlaceOrder) |
| Выставить SL/TP-заявку | `POST /v1/accounts/{account_id}/sltp-orders` (PlaceSLTPOrder; с v2.23.0 — `slGuardTime`/`tpGuardTime`) |
| Отменить заявку | `DELETE /v1/accounts/{account_id}/orders/{order_id}` (CancelOrder) |
| Активные заявки | `GET /v1/accounts/{account_id}/orders` (GetOrders) |
| Заявка по ID | `GET /v1/accounts/{account_id}/orders/{order_id}` (GetOrder) |
| Сделки счёта | `GET /v1/accounts/{account_id}/trades` (с interval/limit; commission — для исторических) |
| Транзакции | `GET /v1/accounts/{account_id}/transactions` |
| Счёт/позиции/деньги | `GET /v1/accounts/{account_id}` (GetAccount) |
| Потоки своих заявок/сделок | WS ORDERS/TRADES или gRPC SubscribeOrders/SubscribeTrades/SubscribeOrderTrade |

Тело PlaceOrder (proto `Order`):
- `symbol` (`ticker@mic`), `quantity` — **в штуках** (decimal), не в лотах;
- `side`: `SIDE_BUY` / `SIDE_SELL`;
- `type`: `ORDER_TYPE_MARKET`, `ORDER_TYPE_LIMIT`, `ORDER_TYPE_STOP`, `ORDER_TYPE_STOP_LIMIT`, `ORDER_TYPE_MULTI_LEG` (+ SL/TP отдельным методом);
- `time_in_force`: `DAY`, `GOOD_TILL_CANCEL`, `GOOD_TILL_CROSSING`, `EXT`, `ON_OPEN`, `ON_CLOSE`, `IOC`, `FOK`;
- `limit_price` (для LIMIT/STOP_LIMIT), `stop_price` + `stop_condition` (`STOP_CONDITION_LAST_UP/LAST_DOWN`) для стоп-заявок, `valid_before` для условных;
- `client_order_id` — **≤ 20 символов** (UUID не помещается; в примерах — 10 случайных байт в hex), автогенерируется сервером, если не передан; `comment` ≤ 128 символов;
- Статусы (`OrderStatus`): NEW, PARTIALLY_FILLED, FILLED, DONE_FOR_DAY, CANCELED, REPLACED, PENDING_CANCEL, REJECTED, SUSPENDED, PENDING_NEW, EXPIRED, FAILED, FORWARDING, WAIT, DENIED_BY_BROKER, REJECTED_BY_EXCHANGE, WATCHING, EXECUTED, DISABLED, LINK_WAIT, SL_GUARD_TIME, SL_EXECUTED, SL_FORWARDING, TP_GUARD_TIME и др. Логику завершения строить по enum + `executed_quantity`/`remaining_quantity`, а не по одному статусу (S3g).

Правила, зафиксированные документацией:
- цена лимитной заявки должна быть кратна шагу: `шаг = min_step / 10^decimals` (S3d);
- лоты → штуки: `quantity_pieces = lots × trade_lot_size`; если `trade_lot_size == 0` — значение недоступно, торговую операцию нужно остановить, не угадывая (S3j Рецепт 4, S4);
- ответ PlaceOrder не получен = «неизвестный результат»: НЕ повторять вслепую, сначала сверка активных заявок по `client_order_id`, при неоднозначности — остановка и передача оператору (S3l, S4 chego-ne-delat);
- заявка ≠ сделка: одна заявка порождает несколько сделок; позицию/среднюю цену нельзя восстанавливать только по статусу заявки (S3h).

### 2.7 Инструменты, счета, ограничения **[ПОДТВЕРЖДЕНО]** (S3d, S3c, S8h, S4)

- Символ: `ticker@mic` (один тикер может торговаться на нескольких площадках; хранить полный symbol). Примеры: `SBER@MISX` (акции MOEX), `SBERF@RTSX` (фьючерсы).
- Справочники: `GET /v1/assets` (торгуемые), `GET /v1/assets/all` (полный каталог, курсорная пагинация, `next_cursor='0'` — конец), `GET /v1/assets/{symbol}` (min_step, decimals, lot_size, quote_currency…), `GET /v1/assets/{symbol}/params` (trade_lot_size, is_tradable — в привязке к счёту), `GET /v1/exchanges`, `GET /v1/assets/{symbol}/schedule` (расписание сессий), `GET /v1/assets/clock` (серверное время), опционы, состав индексов. Каталог рекомендуется кэшировать локально и обновлять по расписанию.
- Счёт: `GetAccount` → equity, позиции (symbol, quantity, average_price, current_price, margins, pnl), cash, portfolio_mc/portfolio_forts, даты. Тип счёта и статус (`type`, `status`) в ответе присутствуют.
- Доступность инструмента зависит от типа счёта, торгового режима, тестов и статуса квал-инвестора — API не расширяет права счёта (S3a).
- **Лимит REST: 200 запросов в минуту на метод** (не общий!), превышение → `429`. `GET /v1/usage` (GetUsageMetrics) — мониторинг потребления квот; лимиты можно запросить увеличить через поддержку (ссылка «Увеличить лимиты API» в подвале портала).
- Денежные/количественные значения — строковые decimal; для расчётов использовать Decimal-типы или целые минимальные единицы, бинарный float для цены заявки запрещён (S3c, S4 chego-ne-delat).

### 2.8 Обработка ошибок и повторов **[ПОДТВЕРЖДЕНО]** (S3l, S4, S8b)

Карта HTTP-ошибок (официальная, S3l):

| Код | Причина | Действие клиента |
|---|---|---|
| 400 | неверный symbol/интервал/цена/поля | не повторять; исправить запрос |
| 401 | access_key неверен/истёк | получить новый access_key, повторить безопасный запрос |
| 403 | недостаточно прав | проверить тип токена, счёт, разрешения |
| 404 | не найден счёт/инструмент/заявка | сверить идентификатор и доступ токена |
| 429 | исчерпана квота | остановить повторы, снизить частоту, дождаться окна |
| 500 | внутренняя ошибка | ограниченный повтор безопасной операции, затем алерт |
| 503 | сервис недоступен | backoff + jitter, контроль общего времени ожидания |
| 504 | дедлайн истёк | для чтения — повтор; для торговли — сначала сверка |

- Тело ошибки: `code`, `message`, иногда `details[]`; логировать со статусом и id операции, без токенов.
- Backoff: растущие паузы `1 → 2 → 4 → 8 → 16 → 30` секунд + случайная добавка (jitter).
- Таблица безопасных повторов (S3j Рецепт 3 / S4): GET до ответа — повтор с backoff; GET 503/504 — ограниченный повтор; 429 — не повторять немедленно; WS обрыв — переподключение + сверка; POST PlaceOrder 400 — не повторять; ответ PlaceOrder потерян — только сверка; DELETE CancelOrder потерян — сначала GetOrder.
- Python SDK (S8b): типизированные ошибки `FinamError` ← `AuthError(401)`, `PermissionDeniedError(403)`, `InvalidArgumentError(400)`, `NotFoundError(404)`, `RateLimitError(429)`, `InternalError(500)`, `ServiceUnavailableError(503)`, `DeadlineExceededError(504)`; unary-повторы при `UNAVAILABLE` (экспоненциально + jitter); `RESOURCE_EXHAUSTED` повторяется только при серверном заголовке `grpc-retry-pushback-ms`; **стримы не ретраятся автоматически** — переподключение на стороне приложения; настраиваемая `RetryPolicy(max_attempts, initial_backoff, max_backoff)`.

### 2.9 Тестовая среда / демо-счёт **[ЧАСТИЧНО ПОДТВЕРЖДЕНО]**

- **[ПОДТВЕРЖДЕНО]**: демо-счёт создаётся на портале в разделе «Токены» (https://api.finam.ru/tokens/) при выпуске токена (S3b: «Если подходящего счёта нет, создайте демо-счёт»); токен выпускается с доступом к выбранным счетам, readonly или full; торговые примеры курса требуют демо-счёта и отдельного токена с полным доступом (S3g, S3i).
- **[ПОДТВЕРЖДЕНО]**: отдельного «sandbox-хоста» нет — демонстрационный и боевой контуры различаются **счётом**, а не endpoint; recommended-паттерн безопасности — dry-run по умолчанию + явный флаг разрешения заявок + разрешённый account_id allowlist + предохранители (S3i, S3j Рецепт 7).
- **[ПОДТВЕРЖДЕНО]**: SDK предоставляет `FinamClient.for_testing(secret, endpoint="localhost:50051")` — plaintext-канал для локального фейкового сервера (никогда не направлять на api.finam.ru) — удобная точка для контрактных тестов (S8b).
- **[НЕИЗВЕСТНО]** (Q1): модель исполнения заявок на демо-счёте (имитация биржи/маркет-мейкер? реальная маршрутизация с виртуальными деньгами?), стартовый капитал, ограничения, срок жизни, сброс состояния, доступность рыночных данных на демо (полные/задержанные) — в открытых разделах документации не описано. Курс упоминает лишь, что «демо-сервис может отклонять слишком далёкую от рынка цену» (S3g).

---

## 3. Референсный проект WISEPLAT — факты и сопоставление

### 3.1 Что это **[ПОДТВЕРЖДЕНО]** (S9)

Хакатонный проект «Финам Trade API» (2023), автор Олег Шпагин. Пайплайн из пронумерованных скриптов:

1. `1_get_historical_data_for_strategy_from_moex.py` — история **с MOEX через aiomoex** (бесплатный канал, задержка 15 минут), сохранение в `csv/{ticker}_{tf}.csv`;
2. `2_prepare_dataset_images_from_historical_data.py` — генерация **картинок** (close + SMA16/SMA64, окно 128 свечей, шаг 16, 128×128 px) для младшего ТФ; класс = 1, если закрытие старшего ТФ выше предыдущего закрытия;
3. `3_train_neural_network.py` — обучение **CNN** (TensorFlow/Keras), 40 чекпоинтов `.hdf5`; лучшая модель вручную копируется в `NN_winner/cnn_Open.hdf5`;
4. `4_check_predictions_by_neural_network.py` — контроль предсказаний;
5–6. `5_test_api_finam_v1.py` / `6_test_api_finam_v2.py` — проверка подключений через **FinamPy** (неофициальный WS-клиент старого API) и **FinamTradeApiPy** (неофициальный REST-клиент старого API);
7. `7_live_strategy.py` — live-цикл: поллинг MOEX раз в 10 с → локальная сборка свечей (последняя незакрытая отбрасывается) → картинка → CNN → при классе 1 **рыночная покупка 1 лота** через `FinamPy.new_order(..., use_credit=True)`; выход — рыночная продажа через +1 бар старшего ТФ; стоп-лосса нет.

Конфигурация: `my_config/trade_config.py` — `timeframe_0="M1"` (вход/торговля), `timeframe_1="M10"` (метка/выход), тикеры `{"SBER","VTBR"}`, board `TQBR`, часы 10:00–23:50, `start="2021-01-01"`; `my_config/Config.py` — `ClientIds`, `AccessToken` (токен **comon.ru**, т.е. старый API).

Данные в репозитории: 4 CSV (SBER/VTBR × M1/M10; SBER_M10.csv — 45 004 строки, 2021-01-04…2023-05-03), датасет 5602 картинки (класс 0: 2879, класс 1: 2723), 40 моделей.

Формат CSV (S9e, S9h): `datetime,open,high,low,close,volume` — **точно совпадает с контрактом из ТЗ** (вероятный источник требования). Нюанс: текущий скрипт сдвигает `datetime = begin + timeframe` (т.е. метка конца свечи), но опубликованные CSV начинаются с 10:00 и заканчиваются 23:50 — **[РАСХОЖДЕНИЕ]** между скриптом и данными в репозитории; однозначной семантики метки это не даёт (см. Q3).

### 3.2 Самостоятельные признания автора (важно для статуса референса)

- «Торговая стратегия, заложенная в этом роботе, **не даст ему заработать**» (вход по подтверждённой формации, выход через +1 бар, без ожидания движения) (S9a).
- «В коде стратегии не реализована проверка доступности денежных средств».
- Скрипты защищены `exit(777)` от случайного перезапуска/перезаписи данных.
- Данные live-торговли — бесплатный канал MOEX с задержкой 15 минут (торговые решения на задержанных данных).

Проект используется в нашем плане **только как объект анализа** (структура пайплайна, формат CSV, двух-таймфреймовая схема «вход на младшем ТФ / метка на старшем»), **не как подтверждение качества или прибыльности**.

### 3.3 Сопоставление с текущими требованиями

| Аспект | WISEPLAT (2023) | Наше требование / текущий API (2026) | Вывод |
|---|---|---|---|
| Модель | CNN по картинкам графиков | Табличные признаки, LightGBM/CatBoost | Подход WISEPLAT **не переносим** (и запрещён ТЗ: никаких картинок). Заимствуем только идею разделения «ТФ признаков / ТФ метки» |
| История | MOEX aiomoex (задержка 15 мин), т.к. истории в Finam API тогда не было | Finam API `Bars` существует, но глубина M1=7д, M5..H1=30д | Для длинной истории обучения внешний CSV (MOEX) остаётся основным источником — подход WISEPLAT здесь по-прежнему актуален; API — для LIVE-догрева и короткого бэктеста |
| Таймфреймы | M1, M10, H1 (MOEX нативно даёт M10) | В Finam API **M10 отсутствует** | 10m строим ресэмплингом из 5m (или из 1m) — единый модуль для TEST и LIVE |
| LIVE-данные | поллинг MOEX раз в 10 с, задержка 15 мин | WS-подписки BARS/QUOTES/ORDER_BOOK + REST-снимки, паттерн незакрытой свечи | Поллинг заменяем на подписки; паттерн «последняя свеча неполная» у WISEPLAT совпадает с официальным Рецептом 6 |
| Торговый транспорт | FinamPy / FinamTradeApiPy — неофициальные клиенты **старого** API (client_id + security_board + security_code, BUY_SELL_*, токен comon.ru) | Новый API: ticker@mic, PlaceOrder, secret→access_key | **[РАСХОЖДЕНИЕ]** Код интеграции WISEPLAT **несовместим** с текущим API; маппинг старый→новый — MIGRATION_GUIDE (S8e). Повторное использование кода невозможно, только архитектурные выводы |
| Исполнение | рыночные ордера с плечом, без SL, без проверки денег, состояние в памяти | Официальные рекомендации: dry-run по умолчанию, предохранители, лоты→штуки, шаг цены, reconciliation | У WISEPLAT отсутствуют все guardrails из S3i/S3j — в нашем плане они обязательны |
| Конфиг | Python-классы в `my_config/*.py` | Declarative config (YAML) + валидация + версионирование | Заменяем на схемный конфиг (см. architecture.md) |
| Тесты/CI | нет | unit + property-based (причинность), контрактные тесты API | Новый объём работы |

---

## 4. Версии, зафиксированные на дату исследования (2026-10-21)

| Компонент | Версия/состояние | Источник |
|---|---|---|
| Finam Trade API (сервис) | v2.23.0 от 01.10.2026 (changelog, кварталы с 2025 Q2) | S6 |
| Python SDK `finam-sdk` (PyPI) | 2.23.0; requires_python ≥3.10; grpcio≥1.83.0, protobuf≥7.35.1, googleapis-common-protos≥1.62 | S8d |
| Репозиторий FinamWeb/finam-trade-api | main @ 52c95c8 (01.10.2026) | S8 |
| AsyncAPI WS-спецификация | asyncapi-v1.0.0.yaml | S8g |
| Курс/FAQ api.finam.ru | «Обновлено 28 сентября 2026» | S3, S4 |
| WISEPLAT | master, последний push 21.11.2023; Python 3.10+/3.11+; tensorflow, aiomoex, FinamPy | S9 |

---

## 5. Открытые вопросы (не угадываем — изолируем интерфейсами)

| # | Вопрос | Почему не решаем сейчас | Мера в архитектуре |
|---|---|---|---|
| Q1 | Демо-счёт: модель исполнения, виртуальный капитал, лимиты, сброс, доступность market data (задержки по `md_permissions.delay_minutes`) | Публичная документация описывает только создание демо-счёта и токенов; деталей исполнения нет | Весь торговый путь — за интерфейсом `BrokerGateway` + режим dry-run по умолчанию; до включения заявок — проверочный сценарий на демо с журналом (этап L3 в roadmap) |
| Q2 | Права/тарифы токена: какие рыночные данные входят (MOEX L1/L2, задержки), стоимость, «Увеличить лимиты API» | Определяется договором с брокером и настройкой токена, вне документации | `TokenDetails.md_permissions` читаем в рантайге и логируем; архитектура не рассчитывает на L2-стакан как обязательный источник |
| Q3 | Семантика `Bar.timestamp` (начало/конец интервала) и таймзона (UTC?) — и аналогично для пользовательских CSV | Документация не фиксирует; у WISEPLAT скрипт и данные противоречат друг другу | Контракт данных требует явного `timestamp_semantics` (bar_open|bar_close) и таймзоны; этап D1 включает валидатор-детектор (сверка с расписанием сессий и аукционами); внутреннее представление — только bar_open, tz-aware UTC |
| Q4 | Глубина стакана (число уровней) и частота обновлений ORDER_BOOK | Не documented | Стакан — опциональный источник признаков за интерфейсом; фичи из стакана включаются только после эмпирической проверки на демо |
| Q5 | Поведение BARS-потока при клиринге/аукционах (пропуски, внеочередные свечи, вечерняя/утренняя сессии MOEX) | Не documented | Bar-finalizer + сверка с `Schedule`; пропущенные бары дополняются REST `Bars` при реконнекте |
| Q6 | Ограничения PlaceOrder на демо (отклонение далёких цен и пр.) — упомянуто вскользь | Формальных правил нет | Pre-trade валидация (шаг цены, лот, is_tradable, расписание) + журнал отказов |
| Q7 | Тип подписки ACCOUNT отсутствует в enum AsyncAPI v1.0.0, но есть в курсе и схемах ответов | Рассинхрон спецификаций | Для состояния счёта используем REST-снимки + ORDERS/TRADES; ACCOUNT — опционально, с runtime-проверкой поддержки |
| Q8 | Точная ёмкость ответа `Bars` (макс. число свечей за запрос, пагинация) | Не documented | Загрузчик истории работает окнами с запасом и проверяет полноту (границы inclusivity уже documented) |
| Q9 | Условия реального счёта (комиссии, маржа, режимы T+, правила шортов) для корректной симуляции издержек | Вне API-документации (брокерские тарифы) | Комиссии/слиппедж — параметры конфига симулятора с явной пометкой «оценочные»; решение о реальных ордерах на этом этапе не принимается |

---

## 6. Сводка расхождений «документация ↔ требования/референс»

1. **M10 отсутствует в API** (ТЗ требует 10m) → локальный ресэмплинг 5m→10m единым модулем для TEST и LIVE; для истории — из CSV. Последствие: LIVE-свечи 10m появляются с задержкой на закрытие пары 5m-баров; альтернативы (M1→10m, отказ от 10m, внешний MOEX-поток) зафиксированы в decisions.md (D3).
2. **Глубина истории API мала** (M1 — 7 дней, M5/M15/H1 — 30 дней) при необходимости длинной истории для обучения → основной источник обучения — CSV-архив (как у WISEPLAT), API — только LIVE-догрев/короткая проверка. Последствие: распределения CSV-истории и API-данных нужно сверять (этап D2).
3. **WISEPLAT написан под старый API** (FinamPy/comon.ru/security_board) и несовместим с текущим (ticker@mic, PlaceOrder, secret→access_key). Последствие: референс даёт только структурные идеи; интеграционный код пишем заново под `finam-sdk`.
4. **Семантика timestamp свечей неоднозначна** и в CSV WISEPLAT (скрипт vs данные), и в документации API → явный контракт + валидатор (Q3).
5. **ACCOUNT-подписка**: курс vs AsyncAPI enum (Q7) — изолируем, не полагаемся.
