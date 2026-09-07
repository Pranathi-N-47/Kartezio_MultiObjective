# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""ParetoFront - container for the non-dominated set produced by NSGA-II.

After evolution completes, the NSGA2Trainer builds a ParetoFront from all
individuals on Rank 0. Decoding of genotypes is deferred until explicitly
requested so that the full population is not kept decoded in memory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ParetoFront:
    """Immutable container for the Pareto-optimal individuals found by NSGA-II.

    Parameters
    ----------
    entries : list of (genotype, objective_scores)
        Each entry pairs a genotype with its 1-D array of objective values.
    objective_names : list of str
        Human-readable names for each objective, used in string representations.
    failure_log : list of dict, optional
        The failure log from MOOPopulation. Stored here for downstream access.
    """

    def __init__(
        self,
        entries: List[Tuple[Any, np.ndarray]],
        objective_names: List[str],
        failure_log: Optional[List[Dict[str, Any]]] = None,
    ):
        self._entries = entries
        self._objective_names = objective_names
        self.failures = failure_log if failure_log is not None else []

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_size(self) -> int:
        """Return the number of individuals on the Pareto front."""
        return len(self._entries)

    def get_genotype(self, index: int):
        """Return the genotype of the individual at position index."""
        return self._entries[index][0]

    def get_scores(self, index: int) -> np.ndarray:
        """Return the objective scores of the individual at position index."""
        return self._entries[index][1]

    def get_all_genotypes(self) -> List[Any]:
        """Return a list of all genotypes on the front."""
        return [entry[0] for entry in self._entries]

    def get_all_scores(self) -> np.ndarray:
        """Return all objective scores as a 2-D array of shape (front_size, n_objectives)."""
        return np.array([entry[1] for entry in self._entries])

    def get_objective_names(self) -> List[str]:
        """Return the list of objective names."""
        return list(self._objective_names)

    def get_failure_log(self) -> List[Dict[str, Any]]:
        """Return the failure log from the evolution run."""
        return self.failures

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_all(self, decoder) -> List[Any]:
        """Decode every genotype on the front using the supplied decoder.

        Decoding is deferred to this call so that the full graph parsing
        overhead only occurs when the user explicitly needs the decoded
        pipelines (e.g. for robustness evaluation).

        Parameters
        ----------
        decoder : DecoderCGP
            The decoder to use. Should be the same decoder used during training.

        Returns
        -------
        list
            One decoded pipeline per individual on the front.
        """
        decoded = []
        for genotype, _ in self._entries:
            decoded.append(decoder.parse_to_graphs(genotype))
        return decoded

    # ------------------------------------------------------------------
    # Export stub (to be completed in a later iteration)
    # ------------------------------------------------------------------

    def export_as_functions(self, dir_path: str) -> None:
        """Placeholder for future export of front pipelines as Python functions.

        Parameters
        ----------
        dir_path : str
            Directory where the exported files will be written.
        """
        raise NotImplementedError(
            "export_as_functions is reserved for a future release."
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.get_size()

    def __repr__(self) -> str:
        names = ", ".join(self._objective_names)
        return f"ParetoFront(size={self.get_size()}, objectives=[{names}])"
