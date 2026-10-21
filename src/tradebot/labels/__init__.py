"""Целевая переменная. Импортируется только из ml/dataset и тестов (architecture §6.2)."""

from .target import TargetInfo, TargetSpec, build_target, compute_forward_return

__all__ = ["TargetInfo", "TargetSpec", "build_target", "compute_forward_return"]
