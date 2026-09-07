# -------------------------------------------------------------------------
# Kartezio – Multi‑Objective Extension (NSGA‑II)
# Copyright (c) 2024‑2026 Inserm Transfert SA and co‑owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non‑commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""
Complexity and resource metrics for Multi-Objective Optimization (MOO).

This module implements the concrete ComplexityMetric components:
1. ProcessTime: Measures empirical wall-clock inference latency per image (retrieved
   from population cache without redundant re-execution).
2. ActiveNodeCount: Counts unique non-silent computational nodes in the active graph
   (hardware-agnostic structural metric).
"""
from typing import Any, Optional, Set, Tuple

from kartezio.core.components import Genotype, register
from kartezio.evolution.decoder import DecoderCGP
from kartezio.moo.objectives import ComplexityMetric


@register(ComplexityMetric)
class ProcessTime(ComplexityMetric):
    """
    Measures the inference latency (mean wall-clock time per image in seconds).

    To avoid expensive redundant inference during the evolutionary loop, this metric
    reads the wall-clock execution time pre-recorded in `population.score.time`
    (populated during `DecoderCGP.decode_population()`). If called in a standalone
    context where time is not yet measured, it provides a dynamic evaluation fallback.

    Attributes
    ----------
    name : str
        Metric identifier, defaults to "ProcessTime".
    minimize : bool
        Always True, as lower inference latency is preferred.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the ProcessTime complexity metric.

        Parameters
        ----------
        name : str, optional
            Custom name for this metric. Defaults to "ProcessTime".
        """
        super().__init__(name=name or "ProcessTime", minimize=True)

    def evaluate(self, individual: Any, context: Any = None) -> float:
        """
        Retrieve the inference time for the given individual.

        Supports dynamic context extraction without rigid caller coupling:
        - Direct Population object with `score.time`
        - EvaluationContext or dict containing `population` and optional `index`
        - Standalone fallback: executes `decoder.decode(genotype, x)` if time is not cached.

        Parameters
        ----------
        individual : Any
            The individual genotype or integer index within the population.
        context : Any, optional
            Context containing the evaluated Population instance or score metadata.

        Returns
        -------
        float
            Mean wall-clock inference time per image in seconds.
        """
        # Strategy 1: Context is directly a Population instance (or object with score.time)
        if hasattr(context, "score") and hasattr(context.score, "time"):
            if isinstance(individual, int):
                return float(context.score.time[individual])
            elif hasattr(context, "individuals") and individual in context.individuals:
                idx = context.individuals.index(individual)
                return float(context.score.time[idx])

        # Strategy 2: Context contains a .population attribute
        if hasattr(context, "population") and hasattr(context.population, "score"):
            idx = getattr(context, "index", individual)
            if isinstance(idx, int):
                return float(context.population.score.time[idx])

        # Strategy 3: Context is a dictionary
        if isinstance(context, dict):
            if "time" in context:
                time_val = context["time"]
                if isinstance(time_val, (list, tuple)) and isinstance(individual, int):
                    return float(time_val[individual])
                return float(time_val)
            if "population" in context and hasattr(context["population"], "score"):
                pop = context["population"]
                idx = context.get("index", individual)
                if isinstance(idx, int):
                    return float(pop.score.time[idx])

        # Strategy 4: Dynamic evaluation fallback (e.g. for standalone testing)
        decoder = getattr(context, "decoder", None)
        x_data = getattr(context, "x", None)
        if isinstance(context, dict):
            decoder = decoder or context.get("decoder")
            x_data = x_data or context.get("x")

        if decoder is not None and x_data is not None:
            genotype = individual
            if isinstance(individual, int) and hasattr(context, "population"):
                genotype = context.population.individuals[individual]
            _, elapsed_time = decoder.decode(genotype, x_data)
            return float(elapsed_time)

        raise ValueError(
            f"ProcessTime could not resolve inference time for individual {individual}. "
            "Context must provide access to an evaluated Population instance (`score.time`), "
            "a pre-measured 'time' value, or (decoder, x) for on-the-fly profiling."
        )


@register(ComplexityMetric)
class ActiveNodeCount(ComplexityMetric):
    """
    Counts the number of active (non-silent) processing nodes in an individual's CGP graph.

    In Cartesian Genetic Programming, genotypes contain non-coding ('silent') nodes that
    provide neutral mutations and evolutionary drift. This metric parses the Directed
    Acyclic Graph (DAG) backwards from the outputs to isolate only the nodes that actively
    contribute to the final prediction.

    This metric is 100% deterministic and hardware-independent.

    Attributes
    ----------
    name : str
        Metric identifier, defaults to "ActiveNodeCount".
    minimize : bool
        Always True, as fewer active nodes reflect a simpler, more efficient pipeline.
    decoder : DecoderCGP, optional
        Reference to the decoder used to parse genotypes into graphs. Can also be
        provided dynamically via context during evaluation.
    """

    def __init__(
        self,
        decoder: Optional[DecoderCGP] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize the ActiveNodeCount complexity metric.

        Parameters
        ----------
        decoder : DecoderCGP, optional
            The CGP decoder instance. If omitted here, it must be provided in context.
        name : str, optional
            Custom display name. Defaults to "ActiveNodeCount".
        """
        super().__init__(name=name or "ActiveNodeCount", minimize=True)
        self.decoder = decoder

    def _resolve_decoder(self, context: Any) -> DecoderCGP:
        """Resolve the decoder instance from self or context."""
        if self.decoder is not None:
            return self.decoder
        if hasattr(context, "decoder"):
            return context.decoder
        if isinstance(context, dict) and "decoder" in context:
            return context["decoder"]
        raise ValueError(
            "ActiveNodeCount requires a DecoderCGP instance. Provide it either "
            "at metric initialization or within the evaluation context."
        )

    def _resolve_genotype(self, individual: Any, context: Any) -> Genotype:
        """Resolve the Genotype object from individual or context."""
        if isinstance(individual, Genotype):
            return individual
        if isinstance(individual, int):
            if hasattr(context, "individuals"):
                return context.individuals[individual]
            if hasattr(context, "population") and hasattr(
                context.population, "individuals"
            ):
                return context.population.individuals[individual]
            if isinstance(context, dict) and "population" in context:
                return context["population"].individuals[individual]
        raise ValueError(
            f"ActiveNodeCount could not resolve Genotype for individual {individual}."
        )

    def evaluate(self, individual: Any, context: Any = None) -> float:
        """
        Count the number of unique active computational nodes.

        Excludes input channels (nodes with indices < n_inputs) to count only
        functional image-processing operations.

        Parameters
        ----------
        individual : Genotype or int
            The individual or its index in the population.
        context : Any, optional
            Evaluation context containing the decoder or population.

        Returns
        -------
        float
            The total count of active computational nodes.
        """
        decoder = self._resolve_decoder(context)
        genotype = self._resolve_genotype(individual, context)
        n_inputs = decoder.adapter.n_inputs

        # parse_to_graphs returns phenotype: list of graphs per chromosome
        phenotype = decoder.parse_to_graphs(genotype)

        unique_active_nodes: Set[Tuple[str, Any, int]] = set()
        for chrom_idx, chromosome_name in enumerate(genotype._chromosomes.keys()):
            chromosome_graphs = phenotype[chrom_idx]
            for graph in chromosome_graphs:
                for node in graph:
                    node_index, type_index = node
                    # Exclude raw input channels; count only active processing nodes
                    if node_index >= n_inputs:
                        unique_active_nodes.add(
                            (chromosome_name, type_index, node_index)
                        )

        return float(len(unique_active_nodes))
