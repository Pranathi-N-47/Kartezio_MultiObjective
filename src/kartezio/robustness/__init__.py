# -------------------------------------------------------------------------
# Kartezio - Robustness Module
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""Public exports for the Kartezio robustness module.

Provides:
  - PerturbationType  - enum of the four supported image perturbations.
  - ImagePerturbation - applies an Albumentations-based perturbation to an
                        image in Kartezio DataList format.
  - ParetoEvaluator   - passes a (perturbed) image through a Pareto-front
                        Kartezio pipeline and returns the raw pipeline output.
"""

from .perturbations import ImagePerturbation, PerturbationType
from .evaluator import ParetoEvaluator

__all__ = [
    "PerturbationType",
    "ImagePerturbation",
    "ParetoEvaluator",
]
