# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""NSGA2Trainer - high-level facade for running multi-objective CGP with NSGA-II.

Usage example:

    from kartezio.moo import NSGA2Trainer

    trainer = NSGA2Trainer(
        n_inputs=1,
        n_nodes=30,
        libraries=default_matrix_lib(),
        endpoint=EndpointThreshold(128),
        objectives=[IoU(), ActiveNodeCount()],
        pop_size=50,
        n_iterations=200,
        seed=42,
    )
    pareto_front, history = trainer.fit(x_train, y_train)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from kartezio.callback import Callback, Event, EventType
from kartezio.core.components import Endpoint, Fitness, Library
from kartezio.core.initialization import RandomInit
from kartezio.evolution.decoder import DecoderCGP
from kartezio.helpers import Observable
from kartezio.mutation import MutationHandler, PointMutation
from kartezio.types import DataBatch

from kartezio.moo.callbacks import FileCallback
from kartezio.moo.pareto import ParetoFront
from kartezio.moo.population import MOOPopulation
from kartezio.moo.strategy import NSGA2Strategy, MOOGenerationState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_LOG_FILENAME_FMT = "nsga2_run_{timestamp}.log"


# ---------------------------------------------------------------------------
# NSGA2Trainer
# ---------------------------------------------------------------------------

class NSGA2Trainer(Observable):
    """High-level trainer that runs NSGA-II on a CGP representation.

    The trainer owns:
      - A DecoderCGP built from the supplied libraries and endpoint.
      - An NSGA2Strategy that handles selection and mutation.
      - An MOOPopulation that stores individuals, scores, and history.
      - A list of Fitness objectives. At least two must be provided.

    Parameters
    ----------
    n_inputs : int
        Number of input channels for the CGP graph.
    n_nodes : int
        Number of computational nodes in the CGP graph.
    libraries : Library or list of Library
        Primitive libraries for the CGP decoder.
    endpoint : Endpoint
        Output transformation applied to the decoded graph.
    objectives : list of Fitness
        At least two Fitness instances to optimise simultaneously.
        All objectives are minimised internally.
    pop_size : int
        Number of individuals in both the parent pool and the offspring pool.
        Default is 50.
    n_iterations : int
        Number of generations to run. Default is 200.
    seed : int or None
        Random seed for reproducibility. If None, no seed is set.
    store_failures : bool
        When True the failure log is kept in memory. Default is True.
    log_path : str or None
        Path for the FileCallback log. If None a file is created in the
        current working directory with a timestamped name.
    n_chromosomes : int
        Number of CGP chromosomes. Default is 1.
    """

    def __init__(
        self,
        n_inputs: int,
        n_nodes: int,
        libraries,
        endpoint: Endpoint,
        objectives: List[Fitness],
        pop_size: int = 50,
        n_iterations: int = 200,
        seed: Optional[int] = None,
        store_failures: bool = True,
        log_path: Optional[str] = None,
        n_chromosomes: int = 1,
    ):
        super().__init__()
        _validate_objectives(objectives)

        if not isinstance(libraries, list):
            libraries = [libraries]

        if seed is not None:
            np.random.seed(seed)

        self._objectives = objectives
        self._n_objectives = len(objectives)
        self._objective_names = _extract_objective_names(objectives)
        self._pop_size = pop_size
        self._n_iterations = n_iterations
        self._store_failures = store_failures
        self._seed = seed

        # Build decoder
        self._decoder = DecoderCGP(n_inputs, n_nodes, n_chromosomes, libraries, endpoint)

        # Build mutation and strategy
        mutation = PointMutation(self._decoder.adapter)
        initializer = RandomInit(mutation)
        mutation_handler = MutationHandler(mutation)
        self._strategy = NSGA2Strategy(initializer, mutation_handler, pop_size)

        # Population is created fresh in fit()
        self._population: Optional[MOOPopulation] = None

        # Callbacks
        self._log_path = log_path or _default_log_path()
        self._file_callback = FileCallback(self._log_path, self._objective_names)
        self._extra_callbacks: List[Callback] = []

    # ------------------------------------------------------------------
    # Public getters / setters
    # ------------------------------------------------------------------

    def get_decoder(self) -> DecoderCGP:
        return self._decoder

    def get_strategy(self) -> NSGA2Strategy:
        return self._strategy

    def get_population(self) -> Optional[MOOPopulation]:
        return self._population

    def get_objectives(self) -> List[Fitness]:
        return list(self._objectives)

    def get_objective_names(self) -> List[str]:
        return list(self._objective_names)

    def get_pop_size(self) -> int:
        return self._pop_size

    def get_n_iterations(self) -> int:
        return self._n_iterations

    def set_n_iterations(self, n_iterations: int) -> None:
        self._n_iterations = n_iterations

    def set_pop_size(self, pop_size: int) -> None:
        self._pop_size = pop_size
        self._strategy.set_pop_size(pop_size)

    def set_log_path(self, log_path: str) -> None:
        self._log_path = log_path
        self._file_callback.set_log_path(log_path)

    def add_callback(self, callback: Callback) -> None:
        """Attach an additional callback to the evolution loop."""
        self._extra_callbacks.append(callback)

    def set_endpoint(self, endpoint: Endpoint) -> None:
        self._decoder.endpoint = endpoint

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        x: DataBatch,
        y: DataBatch,
    ) -> Tuple[ParetoFront, List[np.ndarray]]:
        """Run NSGA-II evolution for n_iterations generations.

        Parameters
        ----------
        x : DataBatch
            Training inputs (same format used by KartezioTrainer).
        y : DataBatch
            Training labels (ground truth).

        Returns
        -------
        pareto_front : ParetoFront
            Container holding all Rank-0 individuals from the final generation.
        history : list of np.ndarray
            One 2-D array per generation, shape (pop_size, n_objectives + 1).
        """
        self._strategy.compile(self._n_iterations)
        self._population = _build_population(
            self._strategy.initializer,
            self._pop_size,
            self._n_objectives,
            self._store_failures,
        )

        # Register callbacks
        self._file_callback.set_decoder(self._decoder)
        self.attach(self._file_callback)
        for cb in self._extra_callbacks:
            cb.set_decoder(self._decoder)
            self.attach(cb)

        # Evaluate initial population
        self._evaluate_population(x, y, generation=0)
        self._population.snapshot()

        # Notify evolution start
        state = self._build_state(generation=0)
        self._notify(EventType.START_LOOP, state)

        for gen in range(1, self._n_iterations + 1):
            self._notify(EventType.START_STEP, state)

            # Expand population to 2 * pop_size by generating offspring
            self._strategy.reproduction(self._population)

            # Evaluate offspring only (slots pop_size .. 2*pop_size - 1)
            self._evaluate_offspring(x, y, generation=gen)

            # NSGA-II selection: reduces back to pop_size survivors
            state = self._strategy.selection(self._population)
            state.generation = gen

            # Record history snapshot for the surviving population
            self._population.snapshot()

            self._notify(EventType.END_STEP, state)

        self._notify(EventType.END_LOOP, state)

        pareto_front = _build_pareto_front(
            self._population,
            state.get_fronts(),
            self._objective_names,
        )
        return pareto_front, self._population.get_history()

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_population(self, x: DataBatch, y: DataBatch, generation: int) -> None:
        """Evaluate all individuals and write objective scores into the population."""
        self._population.initialise_raw()
        for i in range(self._population.size):
            _evaluate_individual(
                i, self._population, self._decoder, self._objectives, x, y, generation
            )

    def _evaluate_offspring(self, x: DataBatch, y: DataBatch, generation: int) -> None:
        """Evaluate only the offspring slots (indices pop_size .. size - 1)."""
        for i in range(self._pop_size, self._population.size):
            _evaluate_individual(
                i, self._population, self._decoder, self._objectives, x, y, generation
            )

    # ------------------------------------------------------------------
    # Observable helpers
    # ------------------------------------------------------------------

    def _notify(self, event_type: EventType, state: MOOGenerationState) -> None:
        event = Event(state.generation, event_type, state, force=True)
        self.notify(event)

    def _build_state(self, generation: int) -> MOOGenerationState:
        """Build a generation state summary from the current population."""
        from kartezio.moo.dominance import fast_non_dominated_sort
        objectives = self._population.get_objective_matrix()
        fronts = fast_non_dominated_sort(objectives)
        front0 = fronts[0] if fronts else []
        front0_objs = objectives[front0] if len(front0) > 0 else np.array([[]])
        best_objs = (
            np.min(front0_objs, axis=0)
            if len(front0) > 0
            else np.full(self._n_objectives, np.inf)
        )
        mean_time = float(np.mean(self._population.score.time))
        return MOOGenerationState(
            generation=generation,
            front_size=len(front0),
            best_objectives=best_objs,
            mean_time=mean_time,
            fronts=fronts,
        )


# ---------------------------------------------------------------------------
# Module-level helper functions (no nested functions, no tight coupling)
# ---------------------------------------------------------------------------

def _validate_objectives(objectives: List[Fitness]) -> None:
    """Raise ValueError when fewer than two objectives are provided."""
    if len(objectives) < 2:
        raise ValueError(
            "NSGA2Trainer requires at least two objectives but received "
            f"{len(objectives)}. Pass a list with two or more Fitness instances."
        )


def _extract_objective_names(objectives: List[Fitness]) -> List[str]:
    """Return a list of human-readable objective names."""
    return [obj.name if hasattr(obj, "name") else type(obj).__name__ for obj in objectives]


def _default_log_path() -> str:
    """Return a timestamped log filename in the current working directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = _DEFAULT_LOG_FILENAME_FMT.format(timestamp=timestamp)
    return os.path.join(os.getcwd(), filename)


def _build_population(
    initializer,
    pop_size: int,
    n_objectives: int,
    store_failures: bool,
) -> MOOPopulation:
    """Create and randomly initialise an MOOPopulation of size 2 * pop_size.

    The population always has 2 * pop_size slots: the first half holds the
    current parents and the second half is reserved for offspring.
    """
    population = MOOPopulation(2 * pop_size, n_objectives, store_failures)
    for i in range(2 * pop_size):
        population.individuals[i] = initializer.random()
    return population


def _evaluate_individual(
    individual_idx: int,
    population: MOOPopulation,
    decoder: DecoderCGP,
    objectives: List[Fitness],
    x: DataBatch,
    y: DataBatch,
    generation: int,
) -> None:
    """Decode and evaluate one individual, writing results into the population.

    On exception the individual is marked as failed (all objectives +inf).
    The exception is recorded in the failure log when store_failures is True.
    """
    genotype = population.individuals[individual_idx]
    try:
        y_pred, elapsed = decoder.decode(genotype, x)
        population.score.time[individual_idx] = elapsed
        scores = _compute_objective_scores(objectives, y, y_pred)
        population.score.raw[individual_idx] = scores
        population.score.fitness[individual_idx] = float(np.mean(scores))
    except Exception as exc:
        logger.warning(
            "Individual %d at generation %d raised %s: %s",
            individual_idx,
            generation,
            type(exc).__name__,
            exc,
        )
        population.record_failure(individual_idx, genotype, exc, generation)


def _compute_objective_scores(
    objectives: List[Fitness],
    y: DataBatch,
    y_pred,
) -> np.ndarray:
    """Return a 1-D array of one scalar per objective.

    Each Fitness.batch call returns a per-individual score; here there is
    always exactly one individual, so we take index 0.
    """
    scores = np.zeros(len(objectives), dtype=np.float32)
    for obj_idx, objective in enumerate(objectives):
        scores[obj_idx] = float(objective.batch(y, [y_pred], reduction="mean")[0])
    return scores


def _build_pareto_front(
    population: MOOPopulation,
    fronts: List[List[int]],
    objective_names: List[str],
) -> ParetoFront:
    """Construct a ParetoFront from the Rank-0 individuals in the population."""
    rank0 = fronts[0] if fronts else []
    entries = []
    objectives = population.get_objective_matrix()
    for idx in rank0:
        genotype = population.individuals[idx]
        scores = objectives[idx].copy()
        entries.append((genotype, scores))
    return ParetoFront(entries, objective_names, population.get_failure_log())
