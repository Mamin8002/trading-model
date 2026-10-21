"""TEST-симулятор: событийный движок и расчёт результатов (фаза 4)."""

from .engine import EventSimulator, PositionEpisode, SimulationRun, run_simulation
from .results import SimulationResults, TradeSummary, compute_results, render_simulation_report

__all__ = [
    "EventSimulator",
    "PositionEpisode",
    "SimulationResults",
    "SimulationRun",
    "TradeSummary",
    "compute_results",
    "render_simulation_report",
    "run_simulation",
]
