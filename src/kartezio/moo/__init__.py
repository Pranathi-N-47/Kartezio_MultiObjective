# -------------------------------------------------------------------------
# Kartezio – Multi‑Objective Extension (NSGA‑II)
# Copyright (c) 2024‑2026 Inserm Transfert SA and co‑owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non‑commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""
Multi-Objective Optimization (MOO) extension for Kartezio.

Provides objectives, complexity metrics, NSGA-II evolutionary search,
and post-hoc robustness evaluation for explainable computer vision pipelines.
"""

from kartezio.moo.complexity import (
    ActiveNodeCount,
    ProcessTime,
)
from kartezio.moo.objectives import (
    ComplexityMetric,
    PerformanceObjective,
)

__all__ = [
    "ComplexityMetric",
    "PerformanceObjective",
    "ProcessTime",
    "ActiveNodeCount",
]
