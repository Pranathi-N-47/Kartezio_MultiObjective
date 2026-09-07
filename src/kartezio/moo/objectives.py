# -------------------------------------------------------------------------
# Kartezio – Multi‑Objective Extension (NSGA‑II)
# Copyright (c) 2024‑Present Inserm Transfert SA and co‑owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non‑commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""
Objectives and adapter interfaces for Multi-Objective Optimization (MOO).

Provides the abstract ComplexityMetric base class for evaluating resource usage
and the PerformanceObjective adapter to wrap existing Kartezio Fitness functions.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from kartezio.core.components import (
    Components,
    Fitness,
    KartezioComponent,
    fundamental,
)
from kartezio.types import DataBatch, DataPopulation


@fundamental()
class ComplexityMetric(KartezioComponent, ABC):
    """
    Abstract base class for computational and resource metrics in MOO.

    Inherits from KartezioComponent and registers as a fundamental component.
    All complexity metrics define a scalar evaluation score and are minimized by default.
    """

    def __init__(self, name: Optional[str] = None, minimize: bool = True):
        """Initialize the complexity metric with an optional display name and optimization direction."""
        super().__init__()
        if name is not None:
            self.name = name
        self.minimize = minimize

    @abstractmethod
    def evaluate(self, individual: Any, context: Any = None) -> float:
        """
        Evaluate and return the scalar complexity score for a given individual.

        Concrete metrics implement this method using the individual's genotype
        or execution metadata provided in the context.
        """
        pass

    def __to_dict__(self) -> dict:
        """Serialize metric configuration and parameters to a dictionary for reproducibility."""
        return {
            "name": self.name,
            "args": {
                "minimize": self.minimize,
            },
        }

    @classmethod
    def __from_dict__(cls, dict_infos: dict) -> "ComplexityMetric":
        """Instantiate a ComplexityMetric component from its serialized dictionary representation."""
        return Components.instantiate(
            "ComplexityMetric",
            dict_infos["name"],
            **dict_infos.get("args", {}),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', minimize={self.minimize})"


class PerformanceObjective:
    """
    Adapter wrapping existing Kartezio Fitness classes for multi-objective optimization.

    Standardizes task performance evaluation into the uniform evaluate(individual, context)
    interface required by NSGA-II, while keeping original Fitness implementations untouched.
    """

    def __init__(
        self,
        fitness: Fitness,
        name: Optional[str] = None,
        minimize: bool = True,
    ):
        """Wrap an instantiated Kartezio Fitness object as an NSGA-II objective."""
        assert isinstance(
            fitness, Fitness
        ), f"Expected an instance of Fitness, got {type(fitness)}."
        self.fitness = fitness
        self.name = name if name is not None else fitness.name
        self.minimize = minimize

    def evaluate(self, individual: Any, context: Any = None) -> float:
        """
        Evaluate task performance loss for an individual across supported context types.

        Retrieves pre-computed scores from population cache when available, evaluates
        against provided ground truth batches, or runs the decoder on raw inputs as fallback.
        """
        # 1. Check for pre-computed population fitness
        if hasattr(context, "score") and hasattr(context.score, "fitness"):
            if isinstance(individual, int):
                return float(context.score.fitness[individual])
            elif hasattr(context, "individuals") and individual in context.individuals:
                idx = context.individuals.index(individual)
                return float(context.score.fitness[idx])

        if hasattr(context, "population") and hasattr(context.population, "score"):
            pop = context.population
            idx = getattr(context, "index", individual)
            if isinstance(idx, int):
                return float(pop.score.fitness[idx])

        # 2. Check for explicit ground truth and predictions
        if isinstance(context, dict):
            if "fitness" in context and isinstance(individual, int):
                return float(context["fitness"][individual])
            if "y_true" in context and "y_pred" in context:
                y_true = context["y_true"]
                y_pred = context["y_pred"]
                if isinstance(individual, int) and len(y_pred) > individual:
                    y_pred_ind = y_pred[individual]
                else:
                    y_pred_ind = y_pred
                return float(np.mean(self.fitness.evaluate(y_true, y_pred_ind)))

        # 3. Fallback: evaluate on-the-fly with decoder and data
        decoder = getattr(context, "decoder", None)
        x_data = getattr(context, "x", None)
        y_data = getattr(context, "y", None)

        if isinstance(context, dict):
            decoder = decoder or context.get("decoder")
            x_data = x_data or context.get("x")
            y_data = y_data or context.get("y")

        if decoder is not None and x_data is not None and y_data is not None:
            genotype = individual
            if isinstance(individual, int) and hasattr(context, "population"):
                genotype = context.population.individuals[individual]
            y_pred, _ = decoder.decode(genotype, x_data)
            scores = self.fitness.evaluate(y_data, y_pred)
            return float(np.mean(scores))

        raise ValueError(
            f"PerformanceObjective '{self.name}' could not evaluate individual {individual}. "
            "Context must provide either pre-computed population scores, "
            "(y_true, y_pred) batches, or (decoder, x, y) datasets."
        )

    def batch(
        self,
        y_true: DataBatch,
        y_pred: DataPopulation,
        reduction: Optional[str] = None,
    ):
        """Delegate batch prediction evaluation directly to the wrapped Fitness metric."""
        return self.fitness.batch(y_true, y_pred, reduction=reduction)

    def __repr__(self) -> str:
        return (
            f"PerformanceObjective(name='{self.name}', "
            f"fitness={self.fitness.__class__.__name__}, minimize={self.minimize})"
        )
