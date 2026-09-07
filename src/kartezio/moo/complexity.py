# -------------------------------------------------------------------------
# Kartezio - Multi-Objective Extension (NSGA-II)
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""
Complexity and resource metrics for Multi-Objective Optimization (MOO).

Implements concrete metrics to evaluate pipeline efficiency, including
cached inference latency (ProcessTime) and active node count (ActiveNodeCount).
"""
from typing import Any, Optional, Set, Tuple

from kartezio.core.components import Genotype, register
from kartezio.evolution.decoder import DecoderCGP
from kartezio.moo.objectives import ComplexityMetric


@register(ComplexityMetric)
class ProcessTime(ComplexityMetric):
    """
    Measures the inference latency of an individual in seconds.

    Reads pre-recorded wall-clock execution time from population scores to avoid
    costly redundant inference, with a dynamic decoding fallback for standalone use.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the ProcessTime metric to minimize inference latency."""
        super().__init__(name=name or "ProcessTime", minimize=True)

    def evaluate(self, individual: Any, context: Any = None) -> float:
        """
        Retrieve or measure the wall-clock execution time for an individual.

        Extracts the cached runtime from the population score, or invokes the decoder
        to time execution on the test batch if no cache is available.
        """
        # 1. Check direct population score
        if hasattr(context, "score") and hasattr(context.score, "time"):
            if isinstance(individual, int):
                return float(context.score.time[individual])
            elif hasattr(context, "individuals") and individual in context.individuals:
                idx = context.individuals.index(individual)
                return float(context.score.time[idx])

        # 2. Check population attribute in context
        if hasattr(context, "population") and hasattr(context.population, "score"):
            idx = getattr(context, "index", individual)
            if isinstance(idx, int):
                return float(context.population.score.time[idx])

        # 3. Check dictionary context
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

        # 4. Fallback: run decoder directly if unmeasured
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
    Counts the number of active, non-silent processing nodes in an individual's CGP graph.

    Traces the computational Directed Acyclic Graph (DAG) backwards from output nodes to isolate
    only the active operations, providing a deterministic and hardware-independent metric.
    """

    def __init__(
        self,
        decoder: Optional[DecoderCGP] = None,
        name: Optional[str] = None,
    ):
        """Initialize the ActiveNodeCount metric with an optional pre-configured CGP decoder."""
        super().__init__(name=name or "ActiveNodeCount", minimize=True)
        self.decoder = decoder

    def _resolve_decoder(self, context: Any) -> DecoderCGP:
        """Resolve the DecoderCGP instance from internal attributes or evaluation context."""
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
        """Extract the Genotype object from the individual or population context."""
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
        Count active processing nodes in the CGP graph, excluding input channels.

        Parses the genotype into chromosome graphs and counts unique functional
        nodes that actively contribute to the output predictions.
        """
        decoder = self._resolve_decoder(context)
        genotype = self._resolve_genotype(individual, context)
        n_inputs = decoder.adapter.n_inputs

        phenotype = decoder.parse_to_graphs(genotype)

        unique_active_nodes: Set[Tuple[str, Any, int]] = set()
        for chrom_idx, chromosome_name in enumerate(genotype._chromosomes.keys()):
            chromosome_graphs = phenotype[chrom_idx]
            for graph in chromosome_graphs:
                for node in graph:
                    node_index, type_index = node
                    if node_index >= n_inputs:
                        unique_active_nodes.add(
                            (chromosome_name, type_index, node_index)
                        )

        return float(len(unique_active_nodes))
