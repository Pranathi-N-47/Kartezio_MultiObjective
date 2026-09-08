---
title: Robustness Evaluation Guide - Kartezio
description: Post-hoc robustness stress-testing, perturbation models, retention metrics, and visualization.
---

# Robustness Evaluation Guide

This guide provides a comprehensive overview of Kartezio's **post-hoc robustness evaluation module** (`kartezio.moo.robustness`).

---

## 1. Motivation: Why Test for Robustness?

Computer vision algorithms for biomedical microscopy and industrial inspection are typically evaluated on curated, clean test sets. However, when deployed into live clinical or manufacturing pipelines, they routinely encounter artefacts:

- Variations in microscope lamp intensity or LED aging.
- Electronic sensor thermal noise and high-gain ISO grain.
- Slight focal drift or optical defocus over multi-well plate scans.
- Reagent staining batch variability and illumination non-uniformity.
- Lossy compression introduced by PACS network transfers or archival storage.

An algorithm that achieves $96\%$ IoU on pristine test images might drop to $40\%$ under modest Gaussian noise if it relies on brittle, unregularized high-frequency edge filters. Conversely, a slightly simpler algorithm with $93\%$ clean IoU might retain $91\%$ performance across all disturbances.

**Kartezio MOO automatically stress-tests every pipeline that reaches the Pareto front**, giving researchers the quantitative evidence needed to select reliable, field-deployable solutions.

---

## 2. Built-In Perturbation Battery & Severity Parameters

The module provides five standardized perturbation transforms implemented using OpenCV (`cv2`) and NumPy:

| Perturbation | Physical Phenomenon | Severity Parameter & Default Range | Formula / Implementation |
| :--- | :--- | :--- | :--- |
| **Gaussian Noise** | Thermal sensor noise, low-light gain noise | Standard deviation $\sigma \in [10, 25]$ | $I_{\text{noisy}} = \text{clip}(I + \mathcal{N}(0, \sigma^2), 0, 255)$ |
| **Gaussian Blur** | Optical defocus, slight z-plane drift | Kernel size $k \in [3, 7]$, $\sigma_{\text{blur}} = 1.5$ | $I_{\text{blur}} = \text{GaussianBlur}(I, (k, k), \sigma)$ |
| **Brightness Shift** | Lamp decay, illumination flicker | Offset $\Delta \in [-30, +30]$ | $I_{\text{bright}} = \text{clip}(I + \Delta, 0, 255)$ |
| **Contrast Scaling** | Staining variation, dynamic range shifts | Factor $\alpha \in [0.7, 1.3]$ | $I_{\text{contrast}} = \text{clip}(\alpha \cdot (I - 128) + 128, 0, 255)$ |
| **JPEG Compression**| Lossy transmission, disk compression | Quality factor $Q \in [40, 75]$ | `cv2.imencode('.jpg', I, [IMWRITE_JPEG_QUALITY, Q])` |

---

## 3. The `RobustnessReport` Data Structure

When a pipeline is evaluated, `evaluate_pipeline()` returns an instance of `RobustnessReport` containing structured metrics:

```python
@dataclass
class RobustnessReport:
    clean_fitness: float  # Baseline fitness on pristine test images
    perturbed_fitness: (
        dict[str, float]  # Mean fitness under each perturbation type
    )
    fitness_drops: (
        dict[str, float]  # Absolute drop: |perturbed - clean| per perturbation
    )
    fitness_retentions: (
        dict[str, float]  # Retention ratio in [0, 1] per perturbation
    )
    mean_drop: float  # Average fitness drop across all perturbations
    std_drop: float  # Standard deviation of drops across perturbations
    overall_score: (
        float  # Overall robustness score: mean retention in [0.0, 1.0]
    )
    per_image_results: list[dict]  # Detailed sample-level breakdown
```

### Overall Score Computation
Retention for a perturbation $p$ is defined as:

$$\text{Retention}_p = \max\left(0.0, \, 1.0 - \frac{|f_p - f_{\text{clean}}|}{\max(|f_{\text{clean}}|, \epsilon)}\right)$$

$$\text{Robustness Score} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \text{Retention}_p$$

A pipeline with an overall score of `0.94` retains an average of $94\%$ of its baseline accuracy when subjected to the stress tests.

---

## 4. Worked Example: Annotated Console Output

Here is how results appear in the terminal when `robustness.summarise()` is called:

```text
====================================================================================================
                             KARTEZIO MOO PARETO FRONT & ROBUSTNESS SUMMARY
====================================================================================================
 Sol ID | Rank | IoU Loss | Active Nodes | Time (ms) | Robustness | Noise Ret. | Blur Ret. | JPEG Ret.
----------------------------------------------------------------------------------------------------
 #0     |   0  |   0.038  |      11      |    14.2   |   0.884    |   0.821    |   0.912   |   0.919
 #1     |   0  |   0.051  |       6      |     7.0   |   0.942    |   0.931    |   0.948   |   0.947
 #2     |   0  |   0.088  |       3      |     3.1   |   0.971    |   0.965    |   0.974   |   0.974
----------------------------------------------------------------------------------------------------
* Rank: 0 indicates non-dominated Pareto front solutions.
* IoU Loss: 1.0 - IoU (lower is better).
* Robustness: Mean retention across all 5 test perturbations [0.0 to 1.0, higher is better].
====================================================================================================
```

### Analysis of the Candidates:
- **Solution #0 (Max Accuracy Specialist)**: Achieves the lowest IoU loss (`0.038`, or $96.2\%$ IoU), but uses 11 nodes, takes $14.2\text{ ms}$, and its noise retention drops to `0.821` (sensitive to camera noise).
- **Solution #1 (Knee-Point Solution)**: Exceptional balance. IoU loss of `0.051` ($94.9\%$ IoU), only 6 active nodes, twice as fast ($7.0\text{ ms}$), and maintains `0.942` overall robustness.
- **Solution #2 (Ultra-Robust Edge Specialist)**: Smallest graph (3 nodes) and fastest inference ($3.1\text{ ms}$). Retains `97.1%` performance under severe perturbations.

---

## 5. Standalone Evaluation of Pre-Trained Pipelines

You can evaluate any individual pipeline or pre-existing Pareto front manually without running the full trainer:

```python
from kartezio.core.fitness import IoU
from kartezio.moo.robustness import (
    RobustnessSuite,
    evaluate_pareto_front,
    summarise,
    to_dataframe,
)
from kartezio.utils.dataset import one_cell_dataset

# 1. Load test data
test_x, test_y = one_cell_dataset()

# 2. Configure suite (or use default)
suite = RobustnessSuite()

# 3. Evaluate Pareto front
# front: list of Pareto-optimal genotypes from a previous run
# decoder: Kartezio CGP decoder instance
evaluated_front = evaluate_pareto_front(
    front=my_pareto_front,
    decoder=my_decoder,
    x_test=test_x,
    y_test=test_y,
    fitness_fn=IoU(),
    suite=suite,
)

# 4. Print formatted summary table to console
summarise(evaluated_front)

# 5. Convert to pandas DataFrame
df = to_dataframe(evaluated_front)
df.to_csv("robustness_benchmark.csv", index=False)
```

---

## 6. Visualizing the Per-Perturbation Breakdown

Visualizing how individual perturbations impact different candidate pipelines is straightforward with `matplotlib`:

```python
import matplotlib.pyplot as plt
import numpy as np

# Example retention scores for Solution #0 vs Solution #1
perturbations = ["Gaussian Noise", "Gaussian Blur", "Brightness", "Contrast", "JPEG"]
sol_0_retention = [0.821, 0.912, 0.890, 0.878, 0.919]
sol_1_retention = [0.931, 0.948, 0.940, 0.944, 0.947]

x = np.arange(len(perturbations))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width / 2, sol_0_retention, width, label="Solution #0 (Complex, 11 nodes)", color="#3f51b5")
rects2 = ax.bar(x + width / 2, sol_1_retention, width, label="Solution #1 (Balanced, 6 nodes)", color="#4caf50")

ax.set_ylabel("Fitness Retention Ratio [0 to 1]")
ax.set_title("Robustness Breakdown across Imaging Perturbations")
ax.set_xticks(x)
ax.set_xticklabels(perturbations)
ax.set_ylim(0.7, 1.0)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="Ideal (No degradation)")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig("robustness_breakdown.png", dpi=300)
plt.show()
```

---

## 7. Summary Checklist

When deploying an evolved pipeline into production:
- [x] Run `evaluate_pareto_front` with the full test set.
- [x] Check the **Noise Retention** column: if $< 0.85$, inspect whether the pipeline uses high-gain Laplacian/derivative filters without prior smoothing.
- [x] Inspect the **JPEG Retention** column if images will undergo transmission or cloud compression.
- [x] Use `to_dataframe()` to record complete performance reports for quality certification and reproducibility.
