---
title: Kartezio - Explainable Computer Vision
description: Cartesian Genetic Programming for Automated Vision Pipeline Design
---

# Kartezio

Automated, interpretable, and computationally frugal computer vision pipelines for biomedical imaging and beyond.

---

## Overview

**Kartezio** is a Cartesian Genetic Programming (CGP) framework that enables the automated design of fully interpretable image-processing pipelines. Built on OpenCV, Kartezio empowers researchers and engineers to discover novel computer vision solutions using only a handful of annotated samples and a single CPU core.

Originally developed for biomedical image segmentation and featured in Nature Communications, Kartezio's principles are domain-agnostic and apply to industrial quality control, satellite imagery, embedded vision, and robotics.

---

## Key Features

**Transparent & Certifiable** — Every evolved pipeline is an open directed acyclic graph (DAG) of standard OpenCV primitives that can be directly exported as Python code or C++ routines.

**Few-Shot Learning** — Evolve solutions from just a handful of annotated examples without requiring massive datasets or GPUs.

**Modular and Customizable** — Built from interchangeable Components that you can mix, match, or replace to suit your application.

**Frugal & Edge-Ready** — Runs entirely on standard CPUs. No GPU clusters, CUDA drivers, or terabytes of training data required.

---

## Quick Start Example

Evolving a cell segmentation pipeline:

```python
from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import IoU  
from kartezio.evolution.base import KartezioTrainer
from kartezio.primitives.matrix import default_matrix_lib
from kartezio.utils.dataset import one_cell_dataset

# 1. Load dataset
train_x, train_y = one_cell_dataset()

# 2. Configure components
libraries = default_matrix_lib()
endpoint = EndpointThreshold(threshold=128)

# 3. Initialize trainer
trainer = KartezioTrainer(
    n_inputs=1,
    n_nodes=20,
    libraries=libraries,
    endpoint=endpoint,
    fitness=IoU(),
    population_size=20,
)

# 4. Evolve pipeline
population = trainer.fit(
    n_generations=50,
    x_train=train_x,
    y_train=train_y,
    x_test=train_x,
    y_test=train_y,
)

# 5. Get best solution
best = population.best()
print(best)
```

---

## Documentation

- **[Installation Guide](installation.md)** — Install Kartezio and configure your environment
- **[Quick Start Tutorial](quickstart.md)** — Step-by-step walkthrough for new users
- **[Multi-Objective Optimization Guide](moo_guide.md)** — Deep technical guide to NSGA-II
- **[Robustness Evaluation Guide](robustness_guide.md)** — Comprehensive robustness testing guide

> Documentation note: this project page is intended to provide a quick overview of the Kartezio framework and links to detailed guides for installation, usage, and robustness analysis.

---

## Citation & Licence

Kartezio is distributed for non-commercial academic and research purposes under the terms of the licence in the repository root. If you use Kartezio in academic publications, please cite:

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
