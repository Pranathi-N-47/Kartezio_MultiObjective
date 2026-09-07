# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""Public exports for the Kartezio MOO (NSGA-II) extension.

Provides:
  - NSGA2Trainer  - high-level facade to run the NSGA-II algorithm.
  - ParetoFront   - container for the final non-dominated set.
  - FileCallback  - simple text logger written after each generation.
"""

from .trainer import NSGA2Trainer
from .pareto import ParetoFront
from .callbacks import FileCallback

__all__ = ["NSGA2Trainer", "ParetoFront", "FileCallback"]
