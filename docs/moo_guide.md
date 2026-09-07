---
title: Multi-Objective Optimization Guide - Kartezio
description: In-depth technical guide to NSGA-II, Pareto dominance, custom complexity metrics, and robustness integration.
---

# Multi-Objective Optimization Guide

This guide provides a comprehensive technical breakdown of how **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) is integrated into Kartezio's Cartesian Genetic Programming (CGP) framework.

---

## 1. Why Multi-Objective CGP?

In standard genetic programming, a single scalar fitness function (e.g. mean IoU) dictates survival:

$$\text{fitness} = 1.0 - \text{IoU}(y_{\text{true}}, y_{\text{pred}})$$

This single-objective formulation creates several severe drawbacks:
1. **Code Bloat**: Evolution frequently inserts inactive or marginally beneficial operations that consume runtime with negligible accuracy gain.
2. **Suboptimal Trade-Offs**: Forcing multi-attribute criteria into a single weighted sum ($w_1 \cdot \text{loss} + w_2 \cdot \text{time} + w_3 \cdot \text{nodes}$) requires arbitrary weights that distort the Pareto frontier.
3. **Fragility**: Solutions tuned exclusively to a clean training set frequently fail when tested against slightly perturbed inputs.

By employing **NSGA-II**, Kartezio maintains an entire diverse population spanning the optimal trade-off frontier.

---

## 2. The NSGA-II Evolutionary Algorithm

The core engine is implemented in `kartezio.moo.strategy.NSGAIIStrategy` and operates through three foundational concepts:

### 2.1 Pareto Dominance

For two candidate pipelines $A$ and $B$ evaluated across $M$ minimization objectives:

$$A \prec B \iff \forall m \in \{1, \dots, M\}, f_m(A) \le f_m(B) \quad \text{and} \quad \exists m \in \{1, \dots, M\}, f_m(A) < f_m(B)$$

In plain words: **$A$ dominates $B$** if $A$ is no worse than $B$ on all objectives, and strictly better than $B$ on at least one objective.

```mermaid
graph TD
    subgraph Objective Space: IoU Loss vs Latency
    S1["Solution 1 (Loss: 0.05, Latency: 10ms)"] -->|Dominates| S3["Solution 3 (Loss: 0.08, Latency: 12ms)"]
    S2["Solution 2 (Loss: 0.09, Latency: 3ms)"] -->|Trade-off (Non-dominated)| S1
    end
```

### 2.2 Fast Non-Dominated Sorting

The population is sorted into hierarchical Pareto fronts $\mathcal{F}_0, \mathcal{F}_1, \dots$:
1. **Front $\mathcal{F}_0$ (Rank 0)**: The non-dominated set. No individual in the entire population dominates any solution in $\mathcal{F}_0$.
2. **Front $\mathcal{F}_1$ (Rank 1)**: The non-dominated set obtained after temporarily removing all solutions in $\mathcal{F}_0$.
3. **Subsequent Fronts**: Repeated until every individual in the combined parent-offspring pool is assigned a non-negative integer **rank**.

### 2.3 Crowding Distance

Within each Pareto front, solutions are sorted by their **crowding distance** to promote diversity and prevent clustering in a narrow region of the trade-off space:

$$d_i = \sum_{m=1}^M \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$

- Boundary solutions on the extremes of each objective receive an infinite crowding distance ($d = \infty$), ensuring that extreme specialists (e.g. absolute fastest, absolute most accurate) are always retained.
- Intermediate solutions in less populated regions receive higher crowding distance than solutions tightly packed together.

### 2.4 Survivor Selection Mechanism

At each evolutionary step:
1. The current population of size $N$ produces $N$ offspring via CGP point mutations (`NodeMutation`, `OutputMutation`).
2. Parents and offspring are pooled into a combined set of size $2N$.
3. Non-dominated sorting partitions the $2N$ pool into fronts $\mathcal{F}_0, \mathcal{F}_1, \dots$.
4. The next generation of size $N$ is filled front-by-front. When a front cannot be included completely without exceeding $N$, candidates from that front are selected based on **descending crowding distance**.

---

## 3. The Objective System Architecture

Kartezio MOO provides a clean, decoupled objective interface in `kartezio.moo.objectives`.

### 3.1 `ComplexityMetric` Base Class

All complexity metrics inherit from `ComplexityMetric`, which is decorated with `@fundamental()`:

```python
from abc import abstractmethod
from kartezio.core.components import fundamental


@fundamental()
class ComplexityMetric:
    """Base class for all complexity and resource metrics in Kartezio MOO."""

    @abstractmethod
    def evaluate(self, individual, context=None) -> float:
        """
        Evaluate complexity for a single individual.

        Parameters
        ----------
        individual : Genotype
            The CGP individual to evaluate.
        context : dict, optional
            Execution context containing population statistics, decoder, or timing.

        Returns
        -------
        float
            The scalar complexity value (lower is better / minimized).
        """
        pass
```

### 3.2 Built-In Complexity Metrics

Located in `kartezio.moo.complexity`:

| Metric | Class Name | Description | Source |
| :--- | :--- | :--- | :--- |
| **Active Node Count** | `ActiveNodeCount` | Counts the unique active (functional) nodes connecting inputs to outputs. | `decoder.parse_to_graphs(genotype)` |
| **Process Time** | `ProcessTime` | Wall-clock inference time in milliseconds per image. | Read directly from `population.score.time` without re-running inference. |
| **Operation Count** | `OperationCount` | Total arithmetic/morphological operations performed (sum of arities of active nodes). | Graph analysis of active primitives. |

### 3.3 Adapting Existing Fitness Metrics: `PerformanceObjective`

To allow any standard Kartezio `Fitness` subclass (e.g. `IoU`, `AveragePrecision`, `MSE`) to function seamlessly as an objective, Kartezio MOO provides the `PerformanceObjective` adapter:

```python
from kartezio.core.fitness import IoU
from kartezio.moo.objectives import PerformanceObjective

# Wrap existing Kartezio fitness
iou_objective = PerformanceObjective(fitness_metric=IoU())
```

---

## 4. Defining a Custom Complexity Metric

You can introduce custom domain-specific constraints—such as memory footprint, number of parameters, or energy consumption—by subclassing `ComplexityMetric` and registering it:

```python
from kartezio.core.components import register
from kartezio.moo.objectives import ComplexityMetric


@register(ComplexityMetric)
class MaxKernelSizeMetric(ComplexityMetric):
    """Measures the maximum spatial convolution kernel size used in active nodes."""

    def __init__(self):
        super().__init__()

    def evaluate(self, individual, context=None) -> float:
        # Access active nodes through decoder in context
        decoder = context.get("decoder") if context else None
        if decoder is None:
            return 0.0

        graph = decoder.parse_to_graphs(individual)
        max_kernel = 1.0

        for node_idx in graph.active_nodes:
            node = individual.nodes[node_idx]
            # Inspect parameters if primitive is a spatial filter
            if hasattr(node, "parameters") and len(node.parameters) > 0:
                kernel_param = float(node.parameters[0])
                if kernel_param > max_kernel:
                    max_kernel = kernel_param

        return max_kernel
```

Registering with `@register(ComplexityMetric)` makes your custom metric discoverable by Kartezio's dynamic component loader.

---

## 5. Dedicated Robustness Evaluation Module

In Kartezio MOO, robustness evaluation is a **first-class citizen**, not an afterthought.

### 5.1 Architecture

Implemented in `src/kartezio/moo/robustness.py`:
- `evaluate_pipeline`: Evaluates a single genotype on a test set against a `RobustnessSuite` of image perturbations.
- `evaluate_pareto_front`: Iterates over the Pareto front returned by NSGA-II, computes per-perturbation retention, and attaches a structured `RobustnessReport` to each solution.
- `summarise`: Prints a formatted table to the terminal.
- `to_dataframe`: Exports the Pareto front and robustness scores to a tabular DataFrame.

### 5.2 Built-In Perturbation Types

The default suite exposes five common real-world imaging distortions:
1. **Gaussian Noise**: Simulates electronic camera sensor thermal noise and photon shot noise.
2. **Gaussian Blur**: Simulates optical defocus and slight specimen depth variations.
3. **Brightness Shift**: Simulates light source fluctuations and illumination decay.
4. **Contrast Scaling**: Simulates staining intensity differences and sensor dynamic range variation.
5. **JPEG Compression**: Simulates lossy image compression artifacts from network transmission or disk storage.

### 5.3 Overall Robustness Score Calculation

For a baseline clean fitness $f_{\text{clean}}$ and perturbed fitness values $f_p$ across $P$ perturbations:

$$\text{Retention}_p = \max\left(0.0, \, 1.0 - \frac{|f_p - f_{\text{clean}}|}{\max(f_{\text{clean}}, \epsilon)}\right)$$

$$\text{Robustness Score} = \frac{1}{P} \sum_{p=1}^P \text{Retention}_p \quad \in [0.0, \, 1.0]$$

A score of `1.0` indicates complete immunity to the perturbation suite, whereas `0.0` indicates total failure.

### 5.4 Adding Custom Perturbations

You can extend `RobustnessSuite` with domain-specific distortions (e.g. motion blur, salt-and-pepper noise, Poisson noise):

```python
import cv2
import numpy as np
from kartezio.moo.robustness import RobustnessSuite


def apply_motion_blur(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Simulate stage motion blur during automated image acquisition."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)


# Add to custom suite
suite = RobustnessSuite()
suite.add_perturbation(
    name="motion_blur",
    fn=apply_motion_blur,
    kwargs={"kernel_size": 9},
)
```

---

## 6. Summary

Kartezio MOO unites the explainability of Cartesian Genetic Programming with the Pareto efficiency of NSGA-II and automated robustness guarantees:
- **No black-box models**: Every evolved solution is an interpretable OpenCV graph.
- **No GPU needed**: Runs efficiently on standard CPUs.
- **Guaranteed trade-offs**: Choose the best-balanced pipeline for cloud, lab, or edge environments with confidence.
