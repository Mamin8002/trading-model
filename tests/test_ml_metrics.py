"""Тесты метрик: торговая симуляция на вручную посчитанном примере, годовой
пересчёт, агрегация сегментов, метрики классификации."""

from __future__ import annotations

import numpy as np
import pytest

from tradebot.ml.metrics import (
    aggregate_trading,
    annualized_cagr,
    classification_metrics,
    derive_bars_per_year,
    positions_from_proba,
    positions_from_regression,
    regression_metrics,
    sharpe_ratio,
    simulate_positions,
)


class TestSimulateKnownValues:
    """Ручной расчёт:
    open_t1 = [100, 101, 102, 103], open_t2 = [101, 102, 103, 104],
    позиции = [1, 1, 0, 0], комиссия 10 б.п., слиппедж 0 -> cost_rate = 0.001.

    gross: [0.01, 102/101-1, 0, 0]
    издержки: вход на баре 0 (|1-0|), выход на баре 2 (|0-1|) -> [0.001, 0, 0.001, 0]
    net: [0.009, 0.00990099, -0.001, 0]
    compounded net_return = 1.009 * 1.00990099 * 0.999 - 1
    одна сделка (сегмент бары 0..1), turnover = 2, bars_in_market = 2.
    """

    open_t1 = np.array([100.0, 101.0, 102.0, 103.0])
    open_t2 = np.array([101.0, 102.0, 103.0, 104.0])
    positions = np.array([1.0, 1.0, 0.0, 0.0])

    def test_known_values(self):
        stats = simulate_positions(
            self.open_t1, self.open_t2, self.positions, commission_bps=10.0, slippage_bps=0.0
        )
        expected_net = 1.009 * (1.0 + (102.0 / 101.0 - 1.0)) * 0.999 - 1.0
        assert stats.net_return == pytest.approx(expected_net, rel=1e-9)
        assert stats.n_trades == 1
        assert stats.turnover == pytest.approx(2.0)
        assert stats.bars_in_market == 2
        assert stats.exposure == pytest.approx(0.5)
        assert stats.gross_return == pytest.approx(0.01 + (102.0 / 101.0 - 1.0))
        assert stats.cost_return == pytest.approx(0.002)
        assert stats.hit_rate == pytest.approx(1.0)
        assert stats.profit_factor is None  # убыточных сделок нет
        # maxDD: просадка на баре выхода ~0.1%
        assert stats.max_drawdown == pytest.approx(1.0 - 0.999 * 1.0, abs=2e-3)

    def test_liquidation_cost_at_segment_end(self):
        # позиция до конца сегмента -> ликвидация добавляет издержки в конце
        stats = simulate_positions(
            np.array([100.0, 101.0]),
            np.array([101.0, 102.0]),
            np.array([1.0, 1.0]),
            commission_bps=100.0,
            slippage_bps=0.0,
        )
        # cost_rate = 0.01: вход (0.01) + выход-ликвидация (0.01)
        assert stats.turnover == pytest.approx(2.0)
        assert stats.cost_return == pytest.approx(0.02)

    def test_zero_position_no_costs(self):
        stats = simulate_positions(
            self.open_t1, self.open_t2, np.zeros(4), commission_bps=10.0, slippage_bps=5.0
        )
        assert stats.net_return == 0.0
        assert stats.n_trades == 0 and stats.turnover == 0.0

    def test_costs_reduce_return(self):
        cheap = simulate_positions(self.open_t1, self.open_t2, self.positions, commission_bps=0.0, slippage_bps=0.0)
        expensive = simulate_positions(self.open_t1, self.open_t2, self.positions, commission_bps=50.0, slippage_bps=50.0)
        assert expensive.net_return < cheap.net_return

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            simulate_positions(self.open_t1, self.open_t2[:3], self.positions, commission_bps=1.0, slippage_bps=1.0)


class TestAggregate:
    def test_chaining_two_segments(self):
        s1 = simulate_positions(
            np.array([100.0]), np.array([110.0]), np.array([1.0]), commission_bps=0.0, slippage_bps=0.0
        )  # +10%
        s2 = simulate_positions(
            np.array([100.0]), np.array([95.0]), np.array([1.0]), commission_bps=0.0, slippage_bps=0.0
        )  # -5%
        total = aggregate_trading([s1, s2])
        assert total.net_return == pytest.approx(1.10 * 0.95 - 1.0)
        assert total.n_bars == 2 and total.n_trades == 2
        # просадка считается по сцепленной кривой
        assert total.max_drawdown == pytest.approx(0.05, abs=1e-9)


class TestAnnualization:
    def test_cagr_formula(self):
        # total=+10% за 100 баров, 1000 баров/год -> 0.1 года -> (1.1)^10 - 1
        assert annualized_cagr(0.10, 100, 1000) == pytest.approx(1.1**10 - 1.0, rel=1e-9)

    def test_cagr_none_on_degenerate(self):
        assert annualized_cagr(-1.0, 100, 1000) is None
        assert annualized_cagr(0.1, 0, 1000) is None

    def test_derive_bars_per_year_from_data(self):
        import pandas as pd

        idx = pd.date_range("2024-01-01 08:00", periods=20, freq="5min", tz="UTC").append(
            pd.date_range("2024-01-02 08:00", periods=10, freq="5min", tz="UTC")
        )
        ts = idx.to_numpy()  # datetime64 (UTC)
        bpy, bpd, source = derive_bars_per_year(ts, trading_days_per_year=252, local_timezone="UTC")
        assert bpd == pytest.approx(15.0)  # медиана(20, 10)
        assert bpy == pytest.approx(15.0 * 252)
        assert source == "derived"

    def test_config_override(self):
        ts = np.array([np.datetime64("2024-01-01T08:00", "us")])
        bpy, bpd, source = derive_bars_per_year(ts, config_override=5000)
        assert bpy == 5000.0 and bpd is None and source == "config"

    def test_sharpe_constant_returns_none(self):
        assert sharpe_ratio(np.ones(50) * 0.001, 1000, 0.0) is None

    def test_sharpe_scaling(self):
        rng = np.random.RandomState(0)
        rets = rng.normal(0.001, 0.01, size=500)
        s1 = sharpe_ratio(rets, 252, 0.0)
        s2 = sharpe_ratio(rets, 252 * 4, 0.0)
        assert s2 == pytest.approx(s1 * 2.0, rel=1e-9)  # sqrt-масштабирование


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        m = classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), threshold=0.5)
        assert m.accuracy == 1.0 and m.roc_auc == 1.0
        assert (m.tn, m.fp, m.fn, m.tp) == (2, 0, 0, 2)
        assert not m.single_class

    def test_single_class_window(self):
        m = classification_metrics(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.55]), threshold=0.5)
        assert m.single_class and m.roc_auc is None and m.log_loss is None
        assert "один класс" in m.note

    def test_regression_metrics_known(self):
        rm = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert rm.mae == 0.0 and rm.rmse == 0.0 and rm.r2 == pytest.approx(1.0)
        rm2 = regression_metrics(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        assert rm2.r2 is None  # нулевая дисперсия


class TestPositions:
    def test_positions_from_proba_threshold(self):
        pos = positions_from_proba(np.array([0.49, 0.5, 0.51, 0.9]), 0.5)
        assert np.array_equal(pos, [0.0, 1.0, 1.0, 1.0])

    def test_positions_from_regression_sign(self):
        pos = positions_from_regression(np.array([-0.01, 0.0, 0.02]))
        assert np.array_equal(pos, [0.0, 0.0, 1.0])
