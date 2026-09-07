# -------------------------------------------------------------------------
# Kartezio – Multi‑Objective Extension (NSGA‑II)
# Copyright (c) 2024‑2026 Inserm Transfert SA and co‑owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non‑commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""
Objectives and adapter interfaces for Multi-Objective Optimization (MOO).

This module defines:
1. ComplexityMetric: The fundamental abstract base class for all computational
   and resource metrics (e.g., latency, active node count, operation count).
2. PerformanceObjective: An adapter that wraps existing Kartezio Fitness classes
   to provide a uniform evaluation interface within MOO evolutionary loops.
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
    Abstract base class for all complexity and resource metrics in Kartezio MOO.

    Inherits from KartezioComponent and ABC, and registers as a fundamental
    component under Kartezio's registry system via the `@fundamental()` decorator.

    Attributes
    ----------
    name : str
        Human-readable name of the metric used in logs, reports, and summary tables.
    minimize : bool
        Flag indicating whether this metric should be minimized (True) or
        maximized (False) during multi-objective Pareto optimization. Default is True.
    """

    def __init__(self, name: Optional[str] = None, minimize: bool = True):
        """
        Initialize the complexity metric.

        Parameters
        ----------
        name : str, optional
            Custom name for the metric. If None, uses the class name from registry.
        minimize : bool, default=True
            Optimization direction. Most complexity metrics (latency, node count)
            are minimized.
        """
        super().__init__()
        if name is not None:
            self.name = name
        self.minimize = minimize

    @abstractmethod
    def evaluate(self, individual: Any, context: Any = None) -> float:
        """
        Evaluate the complexity metric for a given individual.

        Parameters
        ----------
        individual : Any
            The individual to evaluate. Can be a Genotype instance, an index in
            the population, or an individual object depending on the caller.
        context : Any, optional
            Evaluation context providing access to the population, decoder,
            input data, or pre-computed metrics.

        Returns
        -------
        float
            The evaluated scalar complexity metric score.
        """
        pass

    def __to_dict__(self) -> dict:
        """
        Serialize metric metadata to a dictionary for reproducibility.

        Returns
        -------
        dict
            Dictionary containing component parameters.
        """
        return {
            "name": self.name,
            "args": {
                "minimize": self.minimize,
            },
        }

    @classmethod
    def __from_dict__(cls, dict_infos: dict) -> "ComplexityMetric":
        """
        Instantiate a ComplexityMetric from its dictionary representation.

        Parameters
        ----------
        dict_infos : dict
            Dictionary produced by `__to_dict__`.

        Returns
        -------
        ComplexityMetric
            Instantiated complexity metric.
        """
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

    Kartezio's existing single-objective framework relies on subclasses of `Fitness`
    (e.g., IoU, AveragePrecision), which evaluate loss between ground truth masks and
    predictions (where lower is better, 0.0 representing perfect score).

    This adapter standardizes the evaluation contract into `evaluate(individual, context)`
    so that task performance can be handled alongside `ComplexityMetric` instances
    in NSGA-II without altering any existing Fitness implementations.

    Attributes
    ----------
    fitness : Fitness
        The underlying Kartezio Fitness instance. Preserved so post-hoc robustness
        evaluation modules can access it directly.
    name : str
        The identifier for this objective.
    minimize : bool
        Whether this objective is to be minimized. Inherited from Kartezio conventions
        where fitness functions represent error/loss (0.0 is perfect).
    """

    def __init__(
        self,
        fitness: Fitness,
        name: Optional[str] = None,
        minimize: bool = True,
    ):
        """
        Initialize the PerformanceObjective adapter.

        Parameters
        ----------
        fitness : Fitness
            An instantiated Kartezio Fitness object (e.g., IoU(), AveragePrecision()).
        name : str, optional
            Display name. Defaults to the name of the wrapped fitness class.
        minimize : bool, default=True
            Whether lower values are preferred. In Kartezio, Fitness metrics
            represent loss (lower is better).
        """
        assert isinstance(
            fitness, Fitness
        ), f"Expected an instance of Fitness, got {type(fitness)}."
        self.fitness = fitness
        self.name = name if name is not None else fitness.name
        self.minimize = minimize

    def evaluate(self, individual: Any, context: Any = None) -> float:
        """
        Evaluate task performance for a given individual across various context formats.

        Supports dynamic resolution to avoid rigid caller coupling:
        1. Context with pre-computed raw fitness or population scores.
        2. Context containing `y_true` and `y_pred` data batches.
        3. Context containing raw input data and decoder for on-the-fly evaluation.

        Parameters
        ----------
        individual : Any
            The individual genotype or integer index within the population.
        context : Any, optional
            Evaluation context supporting several access patterns:
            - Population instance or object with `.score.fitness` / `.score.raw`
            - Dict containing 'y_true' and 'y_pred'
            - Object or dict containing 'decoder', 'x', 'y'

        Returns
        -------
        float
            Reduced scalar performance loss.
        """
        # Pattern 1: Context provides pre-computed population fitness
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

        # Pattern 2: Context contains ground truth and predictions directly
        if isinstance(context, dict):
            if "fitness" in context and isinstance(individual, int):
                return float(context["fitness"][individual])
            if "y_true" in context and "y_pred" in context:
                y_true = context["y_true"]
                y_pred = context["y_pred"]
                # If predictions are batched for population, index into individual
                if isinstance(individual, int) and len(y_pred) > individual:
                    y_pred_ind = y_pred[individual]
                else:
                    y_pred_ind = y_pred
                return float(np.mean(self.fitness.evaluate(y_true, y_pred_ind)))

        # Pattern 3: Context provides decoder and datasets to evaluate on the fly
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
        """
        Delegate batch evaluation to the underlying Fitness component.

        Parameters
        ----------
        y_true : DataBatch
            Ground truth annotations.
        y_pred : DataPopulation
            Predictions for each individual in the population.
        reduction : str, optional
            Reduction strategy ('mean', 'min', 'max', 'median', or 'raw').

        Returns
        -------
        np.ndarray
            Evaluated fitness scores.
        """
        return self.fitness.batch(y_true, y_pred, reduction=reduction)

    def __repr__(self) -> str:
        return (
            f"PerformanceObjective(name='{self.name}', "
            f"fitness={self.fitness.__class__.__name__}, minimize={self.minimize})"
        )
