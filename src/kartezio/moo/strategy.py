# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""NSGA2Strategy - mutation-only NSGA-II reproduction and selection.

This module implements the NSGA-II selection and reproduction operations.
It mirrors the structure of kartezio.evolution.strategy.OnePlusLambda
so that it fits naturally in the existing Strategy / GeneticAlgorithm scaffold.

Key classes
-----------
MOOGenerationState
    Lightweight value object carrying the information that FileCallback and
    the outer training loop need after each generation.

NSGA2Strategy
    The actual strategy: binary-tournament parent selection, mutation-only
    offspring generation, and NSGA-II environmental selection.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Tuple

import numpy as np

from kartezio.evolution.strategy import Strategy
from kartezio.mutation.handler import MutationHandler

from kartezio.moo.dominance import crowding_distance, fast_non_dominated_sort
from kartezio.moo.population import MOOPopulation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value object returned after each generation
# ---------------------------------------------------------------------------

class MOOGenerationState:
    """Carries summary statistics for one generation to callbacks and the trainer.

    Attributes
    ----------
    generation : int
        Current generation index (0-based).
    front_size : int
        Number of individuals on Rank-0 (the Pareto front).
    best_objectives : np.ndarray
        Per-objective minimum across all Rank-0 individuals.
    mean_time : float
        Mean evaluation time across the full population.
    fronts : list of list of int
        All fronts produced by the non-dominated sort.
    """

    def __init__(
        self,
        generation: int,
        front_size: int,
        best_objectives: np.ndarray,
        mean_time: float,
        fronts: List[List[int]],
    ):
        self.generation = generation
        self.front_size = front_size
        self.best_objectives = best_objectives
        self.mean_time = mean_time
        self.fronts = fronts

    def get_generation(self) -> int:
        return self.generation

    def get_front_size(self) -> int:
        return self.front_size

    def get_best_objectives(self) -> np.ndarray:
        return self.best_objectives

    def get_mean_time(self) -> float:
        return self.mean_time

    def get_fronts(self) -> List[List[int]]:
        return self.fronts


# ---------------------------------------------------------------------------
# NSGA-II strategy
# ---------------------------------------------------------------------------

class NSGA2Strategy(Strategy):
    """Mutation-only NSGA-II selection and reproduction.

    Parameters
    ----------
    initializer : RandomInit
        Used internally by NSGA2Trainer to create the initial population.
        Stored here so Strategy.compile can call it.
    mutation_handler : MutationHandler
        The existing Kartezio mutation handler applied to produce offspring.
    pop_size : int
        Size of both parent pool and offspring pool. Default is 50.
    """

    def __init__(
        self,
        initializer,
        mutation_handler: MutationHandler,
        pop_size: int = 50,
    ):
        self.initializer = initializer
        self.mutation_handler = mutation_handler
        self.pop_size = pop_size
        self._current_ranks: np.ndarray = np.array([])
        self._current_distances: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Getters / setters following Kartezio conventions
    # ------------------------------------------------------------------

    def get_pop_size(self) -> int:
        return self.pop_size

    def set_pop_size(self, pop_size: int) -> None:
        self.pop_size = pop_size

    def get_mutation_handler(self) -> MutationHandler:
        return self.mutation_handler

    def set_mutation_handler(self, mutation_handler: MutationHandler) -> None:
        self.mutation_handler = mutation_handler

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def compile(self, n_iterations: int) -> MOOPopulation:
        """Compile the mutation handler and return an uninitialised MOOPopulation.

        The actual n_objectives value is not known at this point; the caller
        (NSGA2Trainer) sets it before the first evaluation.
        This method returns None intentionally - NSGA2Trainer builds the
        MOOPopulation itself after calling compile.
        """
        self.mutation_handler.compile(n_iterations)

    def selection(self, population: MOOPopulation) -> MOOGenerationState:
        """Run the NSGA-II environmental selection on the combined population.

        Steps:
          1. Non-dominated sort of the combined parent + offspring pool.
          2. Fill the next generation front by front.
          3. When a front does not fit completely, sort by crowding distance
             (descending) and take the best-fitting individuals.

        Returns
        -------
        MOOGenerationState
            Summary of the generation for callbacks and the training log.
        """
        objectives = population.get_objective_matrix()
        if objectives is None:
            raise ValueError(
                "Objective matrix is None. Call initialise_raw before evaluation."
            )

        fronts = fast_non_dominated_sort(objectives)

        ranks = np.zeros(population.size, dtype=int)
        distances = np.zeros(population.size, dtype=float)

        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            dist = crowding_distance(front, objectives)
            for local_i, global_i in enumerate(front):
                distances[global_i] = dist[local_i]

        # Select pop_size individuals for the next generation
        survivors = _select_survivors(fronts, distances, self.pop_size)

        # Rebuild the population in-place using the survivors list
        _apply_survivors(population, survivors)

        self._current_ranks = ranks
        self._current_distances = distances

        # Build state summary
        front0 = fronts[0] if fronts else []
        front0_objectives = objectives[front0] if len(front0) > 0 else np.array([[]])
        best_objs = (
            np.min(front0_objectives, axis=0)
            if len(front0) > 0
            else np.full(objectives.shape[1], np.inf)
        )
        mean_time = float(np.mean(population.score.time))
        return MOOGenerationState(
            generation=0,  # updated by the trainer
            front_size=len(front0),
            best_objectives=best_objs,
            mean_time=mean_time,
            fronts=fronts,
        )

    def reproduction(self, population: MOOPopulation) -> None:
        """Generate offspring by binary tournament selection and mutation.

        For each offspring slot (the second half of the population after
        selection), a parent is chosen via binary tournament and mutated.
        """
        half = self.pop_size
        full = population.size
        for i in range(half, full):
            parent = self._binary_tournament(population)
            offspring = self.mutation_handler.mutate(parent.clone())
            population.individuals[i] = offspring

    # ------------------------------------------------------------------
    # Binary tournament selection
    # ------------------------------------------------------------------

    def _binary_tournament(self, population: MOOPopulation):
        """Select one parent by binary tournament.

        Two candidates are drawn at random from the first pop_size slots
        (the current survivors). The winner is chosen by:
          1. Lower Pareto rank wins.
          2. On rank tie, higher crowding distance wins.
          3. On distance tie, pick either candidate at random.
        """
        idx_a, idx_b = np.random.choice(self.pop_size, size=2, replace=False)
        return _tournament_winner(
            idx_a,
            idx_b,
            population,
            self._current_ranks,
            self._current_distances,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (no nested functions, no tight coupling)
# ---------------------------------------------------------------------------

def _select_survivors(
    fronts: List[List[int]],
    distances: np.ndarray,
    pop_size: int,
) -> List[int]:
    """Return a list of pop_size survivor indices using NSGA-II selection.

    Fronts are included in order until the population is filled. If a
    front does not fit completely, its members are sorted by crowding
    distance (descending) and the best-fitting subset is chosen.
    """
    survivors: List[int] = []
    for front in fronts:
        if len(survivors) + len(front) <= pop_size:
            survivors.extend(front)
        else:
            remaining = pop_size - len(survivors)
            sorted_by_dist = sorted(front, key=lambda idx: distances[idx], reverse=True)
            survivors.extend(sorted_by_dist[:remaining])
            break
    return survivors


def _apply_survivors(population: MOOPopulation, survivors: List[int]) -> None:
    """Rearrange the population so survivors occupy the first pop_size slots.

    Individuals, fitness values, raw objective values, and times are all
    reordered consistently. Extra slots (the offspring area) are zeroed out
    so stale data does not leak into the next generation.
    """
    new_individuals = [None] * population.size
    new_fitness = np.full(population.size, np.inf, dtype=np.float32)
    new_time = np.zeros(population.size, dtype=np.float32)
    n_obj = population.n_objectives
    new_raw = np.full((population.size, n_obj), np.inf, dtype=np.float32)

    for new_idx, old_idx in enumerate(survivors):
        new_individuals[new_idx] = population.individuals[old_idx]
        new_fitness[new_idx] = population.score.fitness[old_idx]
        new_time[new_idx] = population.score.time[old_idx]
        if population.score.raw is not None:
            new_raw[new_idx] = population.score.raw[old_idx]

    population.individuals = new_individuals
    population.score.fitness = new_fitness
    population.score.time = new_time
    population.score.raw = new_raw


def _tournament_winner(
    idx_a: int,
    idx_b: int,
    population: MOOPopulation,
    ranks: np.ndarray,
    distances: np.ndarray,
):
    """Return the winning individual from a binary tournament.

    Comparison criteria in order:
      1. Lower rank wins.
      2. Higher crowding distance wins (preserves diversity).
      3. Coin flip on a tie.
    """
    rank_a, rank_b = ranks[idx_a], ranks[idx_b]
    if rank_a < rank_b:
        return population.individuals[idx_a]
    if rank_b < rank_a:
        return population.individuals[idx_b]
    dist_a, dist_b = distances[idx_a], distances[idx_b]
    if dist_a > dist_b:
        return population.individuals[idx_a]
    if dist_b > dist_a:
        return population.individuals[idx_b]
    # tie - pick at random
    winner_idx = idx_a if np.random.rand() < 0.5 else idx_b
    return population.individuals[winner_idx]
