"""Слой данных: контракт Bar, загрузка/валидация CSV, сессии, ресэмплинг таймфреймов."""

from .bar import SUPPORTED_TIMEFRAMES, Bar, BarSource, Timeframe, floor_to_grid
from .csv_loader import (
    REQUIRED_COLUMNS,
    CsvLoadResult,
    DataValidationError,
    load_csv,
)
from .resampler import ResampleIssue, ResampleIssueCode, ResampleResult, resample
from .sessions import DayRule, SessionCalendar, SessionWindow
from .validator import (
    IssueCode,
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_bars,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "SUPPORTED_TIMEFRAMES",
    "Bar",
    "BarSource",
    "CsvLoadResult",
    "DataValidationError",
    "DayRule",
    "IssueCode",
    "ResampleIssue",
    "ResampleIssueCode",
    "ResampleResult",
    "SessionCalendar",
    "SessionWindow",
    "Severity",
    "Timeframe",
    "ValidationIssue",
    "ValidationReport",
    "floor_to_grid",
    "load_csv",
    "resample",
    "validate_bars",
]
