"""摄影测量外方位元素解算系统。

仅依赖 NumPy 的解算子包；图形界面位于 :mod:`photogrammetry.gui`，
因此可以在无 Tk 的环境中直接调用 :func:`photogrammetry.solve`。
"""

from __future__ import annotations

from .model import design_matrix, project, residuals, rotation_matrix
from .solver import MIN_POINTS, Solution, SolveError, solve_exterior_orientation

__all__ = [
    "MIN_POINTS",
    "Solution",
    "SolveError",
    "design_matrix",
    "project",
    "residuals",
    "rotation_matrix",
    "solve_exterior_orientation",
]

__version__ = "2.0.0"
