"""Read-only адаптер Finam Trade API (этап 0-2 roadmap; торговые операции запрещены)."""

from .errors import (
    FinamAuthError,
    FinamError,
    FinamInvalidRequestError,
    FinamInvalidResponseError,
    FinamNotFoundError,
    FinamPermissionDeniedError,
    FinamRateLimitError,
    FinamServerError,
    FinamTransportError,
    HistoryDepthError,
    UnsupportedTimeframeError,
    get_logger,
)
from .ratelimit import SlidingWindowLimiter
from .rest_client import (
    API_TIMEFRAME_BY_CODE,
    HISTORY_DEPTH_LIMITS,
    AssetInfo,
    AssetParams,
    FinamBarRecord,
    FinamRestClient,
    TokenDetails,
)
from .source import FinamHistorySource, FinamSnapshotSource, client_from_config
from .transport import HttpResponse, HttpTransport, UrllibTransport

try:  # ордерный адаптер фазы 5 (за LIVE-гейтом); импорт не должен ломать read-only контур
    from .broker import FinamBrokerGateway
except ImportError:  # pragma: no cover - защитный путь при частичной установке
    FinamBrokerGateway = None  # type: ignore[assignment]

__all__ = [
    "API_TIMEFRAME_BY_CODE",
    "AssetInfo",
    "AssetParams",
    "FinamAuthError",
    "FinamBarRecord",
    "FinamBrokerGateway",
    "FinamError",
    "FinamHistorySource",
    "FinamInvalidRequestError",
    "FinamInvalidResponseError",
    "FinamNotFoundError",
    "FinamPermissionDeniedError",
    "FinamRateLimitError",
    "FinamRestClient",
    "FinamServerError",
    "FinamSnapshotSource",
    "FinamTransportError",
    "HISTORY_DEPTH_LIMITS",
    "HttpResponse",
    "HttpTransport",
    "HistoryDepthError",
    "SlidingWindowLimiter",
    "TokenDetails",
    "UnsupportedTimeframeError",
    "UrllibTransport",
    "client_from_config",
    "get_logger",
]
