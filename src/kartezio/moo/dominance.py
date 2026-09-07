# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""Utility functions for NSGA-II: fast non-dominated sort and crowding distance.

Both functions operate purely on NumPy arrays and have no external dependencies.
All objectives are treated as minimisation objectives. Larger values are worse.
"""

import numpy as np
from typing import List


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """Return a list of fronts. Each front is a list of individual indices.

    Parameters
    ----------
    objectives : np.ndarray
        Shape (pop_size, n_objectives). Each row holds the objective values
        for one individual. All objectives are minimised.
    """
    pop_size = objectives.shape[0]
    # dominates[p] = list of individuals that p dominates
    dominates: List[List[int]] = [[] for _ in range(pop_size)]
    # dominated_count[p] = number of individuals that dominate p
    dominated_count = np.zeros(pop_size, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(pop_size):
        for q in range(pop_size):
            if p == q:
                continue
            # p dominates q: no worse in all and strictly better in at least one
            if _dominates(objectives[p], objectives[q]):
                dominates[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if vector a Pareto-dominates vector b.

    a dominates b when a is no worse than b in every objective and strictly
    better in at least one objective.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def crowding_distance(front: List[int], objectives: np.ndarray) -> np.ndarray:
    """Compute crowding distance for individuals in a given front.

    Parameters
    ----------
    front : List[int]
        Indices of individuals belonging to the same Pareto front.
    objectives : np.ndarray
        Shape (pop_size, n_objectives). Same array used in the sort.

    Returns
    -------
    np.ndarray
        1-D array of crowding distances, aligned with the order in front.
        Boundary individuals receive np.inf.
    """
    n = len(front)
    if n == 0:
        return np.array([])
    n_obj = objectives.shape[1]
    distances = np.zeros(n, dtype=float)
    front_objs = objectives[front]  # shape (n, n_obj)

    for m in range(n_obj):
        sorted_idx = np.argsort(front_objs[:, m])
        sorted_vals = front_objs[sorted_idx, m]
        # boundary points always get infinite distance
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        obj_range = sorted_vals[-1] - sorted_vals[0]
        if obj_range == 0.0:
            continue
        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (sorted_vals[i + 1] - sorted_vals[i - 1]) / obj_range
    return distances
