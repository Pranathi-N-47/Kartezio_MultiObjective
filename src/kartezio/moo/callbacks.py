# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""FileCallback - writes a human-readable evolution log after each generation.

Inherits from kartezio.callback.Callback so it integrates with the existing
Observable/Observer event system. The log format is plain CSV-style text so
any future UI can parse it easily.

Log line format (one per generation):
    Generation,<gen>,FrontSize,<n>,<ObjName1>,<v1>,<ObjName2>,<v2>,...,MeanTime,<t>
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from kartezio.callback import Callback, EventType


class FileCallback(Callback):
    """Write one log line per generation to a plain text file.

    Parameters
    ----------
    log_path : str
        Absolute or relative path of the output log file.
    objective_names : list of str
        Names of the objectives in the same order as the objective matrix.
    write_every : int
        Write a line every this many generations. Defaults to 1.
    """

    def __init__(
        self,
        log_path: str,
        objective_names: List[str],
        write_every: int = 1,
    ):
        super().__init__(frequency=write_every)
        self._log_path = log_path
        self._objective_names = objective_names
        self._file = None

    # ------------------------------------------------------------------
    # Getters / setters
    # ------------------------------------------------------------------

    def get_log_path(self) -> str:
        """Return the path of the log file."""
        return self._log_path

    def set_log_path(self, log_path: str) -> None:
        """Change the log path. Closes any open file handle first."""
        self._close_file()
        self._log_path = log_path

    def get_objective_names(self) -> List[str]:
        """Return the list of objective names."""
        return list(self._objective_names)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)
        self._file = open(self._log_path, "a", encoding="utf-8")

    def _close_file(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()
        self._file = None

    def _write_line(self, line: str) -> None:
        """Write a line to the log and flush immediately to minimise memory use."""
        if self._file is None or self._file.closed:
            self._open_file()
        self._file.write(line + "\n")
        self._file.flush()

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_evolution_start(self, iteration: int, state) -> None:
        """Write a header comment to the log file when evolution begins."""
        self._open_file()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write_line(f"# NSGA-II run started at {stamp}")
        obj_header = ",".join(self._objective_names)
        self._write_line(f"# Columns: Generation,FrontSize,{obj_header},MeanTime")

    def on_generation_end(self, iteration: int, state) -> None:
        """Write one summary line for the current generation.

        The state object is an MOOGenerationState produced by NSGA2Strategy.
        It carries front_size, best_objectives (minimum per objective on Rank-0),
        and mean_time.
        """
        front_size = getattr(state, "front_size", 0)
        best_objs = getattr(state, "best_objectives", [])
        mean_time = getattr(state, "mean_time", 0.0)

        obj_parts = ",".join(
            f"{name},{value:.6f}"
            for name, value in zip(self._objective_names, best_objs)
        )
        line = f"Generation,{iteration},FrontSize,{front_size},{obj_parts},MeanTime,{mean_time:.6f}"
        self._write_line(line)

    def on_evolution_end(self, iteration: int, state) -> None:
        """Flush and close the log file when evolution finishes."""
        self._write_line(f"# Evolution finished at generation {iteration}")
        self._close_file()
