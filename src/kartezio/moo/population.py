# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""MOOPopulation - holds individuals, multi-objective scores, and history.

This class mirrors the Population class from kartezio.evolution.population
and extends it with:
  - A 3-D history list that grows each generation:
      history[generation] -> np.ndarray of shape (pop_size, n_objectives + 1)
      where the last column is execution time per individual.
  - An optional in-memory failure log that records runtime exceptions.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from kartezio.evolution.population import Population


class MOOPopulation(Population):
    """Population container for multi-objective CGP optimisation.

    Parameters
    ----------
    size : int
        Number of individuals in the population.
    n_objectives : int
        Number of optimisation objectives.
    store_failures : bool
        When True, failed evaluations are recorded in the failure log.
        Set to False to save memory when running large experiments.
    """

    def __init__(self, size: int, n_objectives: int, store_failures: bool = True):
        super().__init__(size)
        self.n_objectives = n_objectives
        self.store_failures = store_failures
        # In-memory list of dicts describing each failed evaluation.
        self.failure_log: List[Dict[str, Any]] = []
        # History is a list of 2-D snapshots, one per generation.
        # Each snapshot has shape (pop_size, n_objectives + 1).
        self.history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Failure recording
    # ------------------------------------------------------------------

    def record_failure(self, individual_idx: int, genotype, exc: Exception, generation: int) -> None:
        """Mark one individual as failed and log the exception.

        The individual's raw objective values are set to +np.inf which
        makes it strictly dominated by any valid individual. The failure
        entry is appended to the failure log only when store_failures is True.

        Parameters
        ----------
        individual_idx : int
            Index of the individual in the population.
        genotype : Genotype
            The genotype that caused the failure (stored for debugging).
        exc : Exception
            The exception that was raised during evaluation.
        generation : int
            The generation number in which the failure occurred.
        """
        if self.score.raw is not None:
            self.score.raw[individual_idx] = np.inf
        self.score.fitness[individual_idx] = np.inf
        self.score.time[individual_idx] = np.inf
        if self.store_failures:
            self.failure_log.append({
                "individual_idx": individual_idx,
                "genotype": genotype,
                "exception_type": type(exc).__name__,
                "exception_msg": str(exc),
                "generation": generation,
            })

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def snapshot(self) -> None:
        """Append one generation snapshot to the history list.

        The snapshot is a 2-D array of shape (pop_size, n_objectives + 1).
        The last column holds per-individual execution time.
        Call this once after each generation's evaluation.
        """
        raw = self.get_objective_matrix()
        if raw is None:
            raw = np.full((self.size, self.n_objectives), np.inf, dtype=np.float32)
        time_col = self.score.time[:, np.newaxis].astype(np.float32)
        frame = np.concatenate([raw, time_col], axis=1)
        self.history.append(frame)

    def clear_history(self) -> None:
        """Remove all stored history frames to free memory."""
        self.history = []

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_objective_matrix(self) -> np.ndarray:
        """Return the raw objective values as a (pop_size, n_objectives) array."""
        return self.score.raw

    def get_failure_log(self) -> List[Dict[str, Any]]:
        """Return the list of failure records."""
        return self.failure_log

    def get_history(self) -> List[np.ndarray]:
        """Return the list of per-generation snapshots."""
        return self.history

    # ------------------------------------------------------------------
    # Raw fitness initialisation helper
    # ------------------------------------------------------------------

    def initialise_raw(self) -> None:
        """Allocate the raw fitness array filled with +inf.

        Call this before the first evaluation so that record_failure can
        write into self.score.raw safely.
        """
        self.score.raw = np.full(
            (self.size, self.n_objectives), np.inf, dtype=np.float32
        )

    # KartezioComponent interface
    @classmethod
    def __from_dict__(cls, dict_infos: dict) -> "MOOPopulation":
        return cls(
            dict_infos["size"],
            dict_infos["n_objectives"],
            dict_infos.get("store_failures", True),
        )
