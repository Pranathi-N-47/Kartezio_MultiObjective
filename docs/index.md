---
title: Kartezio Multi-Objective - Explainable Vision with NSGA-II & Robustness
description: Cartesian Genetic Programming with NSGA-II Multi-Objective Optimization and Post-Hoc Robustness Evaluation
---

# Kartezio Multi-Objective (MOO)

<p align="center">
  <em>Automated, interpretable, and computationally frugal computer vision pipelines balanced across segmentation accuracy, execution speed, graph complexity, and real-world robustness.</em>
</p>

---

## Overview

**Kartezio Multi-Objective** extends the original [Kartezio](https://kartezio.com) Cartesian Genetic Programming (CGP) framework with state-of-the-art **NSGA-II multi-objective evolutionary optimization** and a comprehensive **post-hoc robustness evaluation battery**.

In high-stakes biomedical and industrial imaging, a single performance metric (such as IoU or Dice) is rarely sufficient:
- A pipeline that achieves 95% IoU might execute in 250 milliseconds with 40 complex morphological filters.
- Another pipeline might achieve 93% IoU in just 8 milliseconds with only 3 active operations.
- Furthermore, an algorithm that excels on pristine, clean benchmark images may catastrophically fail when confronted with microscope noise, uneven illumination, focus blur, or JPEG compression.

**Kartezio MOO** solves this dilemma by simultaneously evolving a **Pareto front** of non-dominated solutions and benchmarking every Pareto-optimal candidate against real-world imaging artefacts.

---

## ✨ Key Features

<div class="grid cards" markdown>

-   :material-chart-bell-curve-cumulative: __NSGA-II Multi-Objective Engine__

    ---

    True Pareto optimization powered by Fast Non-Dominated Sorting and Crowding Distance diversity maintenance. Jointly optimize segmentation quality alongside execution latency and graph size.

-   :material-shield-bug: __First-Class Robustness Battery__

    ---

    Every Pareto-optimal pipeline is automatically stress-tested against sensor noise, Gaussian blur, illumination shifts, contrast degradation, and compression artefacts with quantitative retention scoring.

-   :material-graph: __Transparent & Certifiable CGP__

    ---

    Zero black-box neural networks. Every evolved pipeline is an open directed acyclic graph (DAG) of standard OpenCV primitives that can be directly exported as Python code or C++ routines.

-   :material-cpu-64-bit: __Frugal & Edge-Ready__

    ---

    Runs entirely on standard single-core or multi-core CPUs. No GPU clusters, CUDA drivers, or terabytes of training data required. Ideal for edge devices and embedded microscopy setups.

</div>

---

## 🚀 Quick Start Example

Evolving a Pareto front of cell segmentation pipelines that balance Intersection-over-Union (IoU) accuracy against active graph size and inference latency:

```python
from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import IoU
from kartezio.moo import (
    ActiveNodeCount,
    KartezioMOOTrainer,
    PerformanceObjective,
    ProcessTime,
)
from kartezio.primitives.matrix import default_matrix_lib
from kartezio.utils.dataset import one_cell_dataset

# 1. Load annotated dataset (images and ground-truth masks)
train_x, train_y = one_cell_dataset()

# 2. Configure multi-objective optimization criteria
objectives = [
    PerformanceObjective(IoU()),  # Maximize segmentation accuracy (minimize loss)
    ActiveNodeCount(),  # Minimize active CGP graph complexity
    ProcessTime(),  # Minimize per-image inference latency
]

# 3. Initialize the NSGA-II Multi-Objective Trainer
trainer = KartezioMOOTrainer(
    n_inputs=1,
    n_nodes=20,
    libraries=default_matrix_lib(),
    endpoint=EndpointThreshold(128),
    objectives=objectives,
    population_size=20,
)

# 4. Evolve Pareto front (automatically evaluates robustness upon completion)
pareto_population = trainer.fit(
    n_generations=50,
    x_train=train_x,
    y_train=train_y,
    x_test=train_x,
    y_test=train_y,
)

# 5. Export Pareto-optimal solutions with robustness metrics to DataFrame
df = pareto_population.to_dataframe()
print(df)
```

---

## 📖 Documentation Navigation

<div class="grid cards" markdown>

-   :material-download: __[Installation Guide](installation.md)__

    ---

    Install Kartezio MOO and its dependencies using `uv` or `pip`, configure optional development packages, and verify your environment.

-   :material-rocket-launch: __[Quick Start Tutorial](quickstart.md)__

    ---

    Step-by-step walkthrough covering data preparation, objective definition, evolutionary search, console table reading, and CSV export.

-   :material-book-open-page-variant: __[Multi-Objective Guide](moo_guide.md)__

    ---

    Deep technical dive into NSGA-II non-dominated sorting, crowding distance mechanics, and creating custom complexity metrics.

-   :material-shield-check: __[Robustness Evaluation Guide](robustness_guide.md)__

    ---

    Comprehensive guide to the perturbation suite, severity parameters, retention scoring, and per-perturbation breakdown visualization.

-   :material-account-group: __[Collaboration Guide](collaboration.md)__

    ---

    Architecture specifications, module ownership, licensing compliance, and contribution workflows for team development.

</div>

---

## 📜 Citation & Licence

Kartezio is distributed for non-commercial academic and research purposes under the terms of the licence in the repository root. If you use Kartezio or its multi-objective extension in academic publications, please cite:

```bibtex
@article{cortacero2023evolutionary,
  title={Evolutionary design of explainable algorithms for biomedical image segmentation},
  author={Cortacero, K{\'e}vin and McKenzie, Brienne and M{\"u}ller, Sabina and Khazen, Roxana and others},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={7112},
  year={2023},
  publisher={Nature Publishing Group}
}
```
