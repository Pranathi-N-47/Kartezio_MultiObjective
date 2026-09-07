# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""
Multi-Objective Optimization (MOO) extension for Kartezio.

Provides objectives, complexity metrics, NSGA-II evolutionary search,
and post-hoc robustness evaluation for explainable computer vision pipelines.
"""

from .callbacks import FileCallback
from .complexity import (
    ActiveNodeCount,
    ProcessTime,
)
from .objectives import (
    ComplexityMetric,
    PerformanceObjective,
)
from .pareto import ParetoFront
from .trainer import NSGA2Trainer

__all__ = [
    "ComplexityMetric",
    "PerformanceObjective",
    "ProcessTime",
    "ActiveNodeCount",
    "NSGA2Trainer",
    "ParetoFront",
    "FileCallback",
]
