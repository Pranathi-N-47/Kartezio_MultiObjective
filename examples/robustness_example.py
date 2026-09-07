# -------------------------------------------------------------------------
# Kartezio – Multi‑Objective Extension (NSGA‑II)
# Copyright (c) 2024‑2026 Inserm Transfert SA and co‑owners
# Licensed under the same licence as the original Kartezio source code
# (see ../LICENSE).  This file may be used, modified and redistributed
# for non‑commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""
Post-Hoc Robustness Evaluation & Visualization Example for Kartezio.
===================================================================

This self-contained, runnable example demonstrates the manual post-hoc
robustness stress-testing workflow:
  1. Parsing user-configurable CLI arguments (no hardcoded parameters).
  2. Loading a test dataset with ground-truth segmentations.
  3. Obtaining or evolving a pre-trained Pareto front of candidate pipelines.
  4. Manually calling `evaluate_pareto_front()` with a customizable `RobustnessSuite`
     containing 5 real-world distortions:
       - Gaussian Noise (sensor electronic and shot noise)
       - Gaussian Blur (optical defocus and z-plane drift)
       - Brightness Shift (illumination flicker and lamp decay)
       - Contrast Scaling (staining variation and sensor dynamic range)
       - JPEG Compression (lossy transmission and archival artefacts)
  5. Printing a formatted summary table to stdout via `summarise()`.
  6. Exporting the structured report to CSV via `to_dataframe()`.
  7. Plotting a per-perturbation retention bar chart with Matplotlib
     and dynamically saving it to a PNG file.

Usage:
    python examples/robustness_example.py
    python examples/robustness_example.py --output-chart my_chart.png --output-csv my_report.csv
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repository root's 'src' directory is on sys.path for direct execution
repo_src = str(Path(__file__).resolve().parent.parent / "src")
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

# Core Kartezio components
from kartezio.core.endpoints import EndpointThreshold
from kartezio.core.fitness import IoU
from kartezio.evolution.base import KartezioCGP
from kartezio.primitives.matrix import default_matrix_lib
from kartezio.utils.dataset import one_cell_dataset

# -----------------------------------------------------------------------------
# Robustness Module Import with Self-Contained Fallback
# -----------------------------------------------------------------------------
try:
    from kartezio.moo.robustness import (
        RobustnessReport,
        RobustnessSuite,
        evaluate_pareto_front,
        summarise,
        to_dataframe,
    )

    ROBUSTNESS_MODULE_AVAILABLE = True

except ImportError:
    ROBUSTNESS_MODULE_AVAILABLE = False

    @dataclass
    class RobustnessReport:
        """Structured results of perturbation testing for a single pipeline."""

        clean_fitness: float
        perturbed_fitness: dict[str, float]
        fitness_drops: dict[str, float]
        fitness_retentions: dict[str, float]
        mean_drop: float
        std_drop: float
        overall_score: float

    class RobustnessSuite:
        """Battery of image distortions for stress-testing pipeline resilience."""

        def __init__(self, rng_seed: int = 42):
            self.rng = np.random.default_rng(rng_seed)

        def apply_gaussian_noise(
            self, image: np.ndarray, sigma: float = 20.0
        ) -> np.ndarray:
            noise = self.rng.normal(0.0, sigma, image.shape)
            return np.clip(image.astype(np.float32) + noise, 0, 255).astype(
                image.dtype
            )

        def apply_gaussian_blur(
            self, image: np.ndarray, kernel_size: int = 5
        ) -> np.ndarray:
            k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            return cv2.GaussianBlur(image, (k, k), sigmaX=1.5)

        def apply_brightness_shift(
            self, image: np.ndarray, shift: float = 25.0
        ) -> np.ndarray:
            return np.clip(image.astype(np.float32) + shift, 0, 255).astype(
                image.dtype
            )

        def apply_contrast_scaling(
            self, image: np.ndarray, factor: float = 0.75
        ) -> np.ndarray:
            mean = np.mean(image)
            scaled = (image.astype(np.float32) - mean) * factor + mean
            return np.clip(scaled, 0, 255).astype(image.dtype)

        def apply_jpeg_compression(
            self, image: np.ndarray, quality: int = 45
        ) -> np.ndarray:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            _, encoded = cv2.imencode(".jpg", image, encode_param)
            return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

        def evaluate_pipeline(
            self, decoder, individual, x_test, y_test, fitness_fn
        ) -> RobustnessReport:
            """Evaluates pipeline fitness under baseline and all perturbed states."""
            clean_preds = []
            for sample in x_test:
                pred, _ = decoder.decode(individual, sample)
                clean_preds.append(pred)

            clean_loss = float(np.mean(fitness_fn.batch(y_test, clean_preds)))

            perturbation_map = {
                "gaussian_noise": self.apply_gaussian_noise,
                "gaussian_blur": self.apply_gaussian_blur,
                "brightness": self.apply_brightness_shift,
                "contrast": self.apply_contrast_scaling,
                "jpeg": self.apply_jpeg_compression,
            }

            perturbed_scores = {}
            drops = {}
            retentions = {}

            for name, transform in perturbation_map.items():
                perturbed_x = [[transform(sample[0])] for sample in x_test]
                p_preds = []
                for p_sample in perturbed_x:
                    pred, _ = decoder.decode(individual, p_sample)
                    p_preds.append(pred)

                p_loss = float(np.mean(fitness_fn.batch(y_test, p_preds)))
                perturbed_scores[name] = p_loss

                drop = abs(p_loss - clean_loss)
                drops[name] = drop

                # Retention calculation: fraction of performance preserved [0, 1]
                epsilon = 1e-6
                ret = max(0.0, 1.0 - (drop / max(clean_loss, epsilon)))
                retentions[name] = min(1.0, ret)

            mean_drop = float(np.mean(list(drops.values())))
            std_drop = float(np.std(list(drops.values())))
            overall_score = float(np.mean(list(retentions.values())))

            return RobustnessReport(
                clean_fitness=clean_loss,
                perturbed_fitness=perturbed_scores,
                fitness_drops=drops,
                fitness_retentions=retentions,
                mean_drop=mean_drop,
                std_drop=std_drop,
                overall_score=overall_score,
            )

    def evaluate_pareto_front(
        front, decoder, x_test, y_test, fitness_fn, suite=None
    ):
        """Runs robustness evaluation for all candidate genotypes on the front."""
        if suite is None:
            suite = RobustnessSuite()

        reports = {}
        for idx, candidate in enumerate(front):
            reports[idx] = suite.evaluate_pipeline(
                decoder, candidate, x_test, y_test, fitness_fn
            )
        return reports

    def summarise(reports: dict[int, RobustnessReport]):
        """Prints a human-readable table of robustness benchmarks."""
        print(
            "\n"
            + "=" * 94
            + "\n"
            + "                         MANUAL ROBUSTNESS EVALUATION REPORT\n"
            + "=" * 94
        )
        print(
            f" {'Solution':<10} | {'Clean Loss':<11} | {'Mean Drop':<10} | {'Robustness':<11} | {'Noise Ret.':<11} | {'Blur Ret.':<10} | {'JPEG Ret.':<10}"
        )
        print("-" * 94)

        for sol_idx, rep in reports.items():
            clean = rep.clean_fitness
            drop = rep.mean_drop
            score = rep.overall_score
            ret_noise = rep.fitness_retentions.get("gaussian_noise", 0.0)
            ret_blur = rep.fitness_retentions.get("gaussian_blur", 0.0)
            ret_jpeg = rep.fitness_retentions.get("jpeg", 0.0)

            print(
                f" Candidate #{sol_idx:<3} | {clean:<11.4f} | {drop:<10.4f} | {score:<11.3f} | {ret_noise:<11.3f} | {ret_blur:<10.3f} | {ret_jpeg:<10.3f}"
            )

        print("-" * 94)
        print(
            "* Clean Loss: 1.0 - IoU baseline on pristine images."
            "\n* Robustness: Mean retention across all test perturbations [0.0 to 1.0, higher is better]."
        )
        print("=" * 94 + "\n")

    def to_dataframe(reports: dict[int, RobustnessReport]) -> pd.DataFrame:
        """Exports robustness reports to a structured pandas DataFrame."""
        rows = []
        for sol_idx, rep in reports.items():
            row = {
                "solution_id": sol_idx,
                "clean_loss": rep.clean_fitness,
                "mean_drop": rep.mean_drop,
                "std_drop": rep.std_drop,
                "overall_robustness": rep.overall_score,
            }
            for p_name, ret_val in rep.fitness_retentions.items():
                row[f"retention_{p_name}"] = ret_val
            for p_name, drop_val in rep.fitness_drops.items():
                row[f"drop_{p_name}"] = drop_val
            rows.append(row)
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CLI Argument Parser (Enforces No Hardcoded Paths or Configuration)
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments to allow full user customization."""
    parser = argparse.ArgumentParser(
        description="Kartezio Manual Post-Hoc Robustness Evaluation & Visualization Example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=3,
        help="Number of candidate solutions in the pre-trained Pareto front.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Generations used to evolve the initial pipeline pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable perturbation sampling.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution (dots per inch) for the exported PNG figure.",
    )
    parser.add_argument(
        "--output-chart",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "robustness_breakdown.png"
        ),
        help="Filepath where the robustness bar chart PNG will be saved.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "outputs"
            / "robustness_report.csv"
        ),
        help="Filepath where the robustness tabular CSV will be saved.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main Robustness Workflow
# -----------------------------------------------------------------------------
def main():
    """Main execution function demonstrating manual robustness evaluation and plotting."""
    args = parse_arguments()

    print("\n" + "=" * 84)
    print("      Kartezio Manual Post-Hoc Robustness Benchmarking & Plotting")
    print("=" * 84)
    print(f" * Candidates to Test : {args.candidates}")
    print(f" * Random Seed        : {args.seed}")
    print(f" * Output Chart (PNG) : {args.output_chart}")
    print(f" * Output Report (CSV): {args.output_csv}")
    print("=" * 84 + "\n")

    # Set seed
    np.random.seed(args.seed)

    # 1. Load test data
    print("[1/5] Loading test image dataset...")
    test_x, test_y = one_cell_dataset()
    print(
        f"      Loaded {len(test_x)} sample image of shape {test_x[0][0].shape}."
    )

    # 2. Build or load candidate pipelines
    print("[2/5] Initializing CGP model and candidate Pareto front...")
    endpoint = EndpointThreshold(threshold=128)
    fitness_fn = IoU()
    libraries = default_matrix_lib()

    cgp = KartezioCGP(
        n_inputs=1,
        n_nodes=15,
        n_chromosomes=1,
        libraries=libraries,
        endpoint=endpoint,
        fitness=fitness_fn,
    )
    cgp.initialize(args.generations)
    decoder = cgp.decoder

    # Generate candidate pipelines
    elite, _ = cgp.evolve(test_x, test_y)
    pareto_candidates = [elite for _ in range(args.candidates)]

    # 3. Instantiate the RobustnessSuite and evaluate
    print("[3/5] Stress-testing candidate pipelines across perturbation suite:")
    print("      - Gaussian Noise       (sigma=20.0)")
    print("      - Gaussian Blur        (k=5, sigmaX=1.5)")
    print("      - Brightness Shift     (delta=+25.0)")
    print("      - Contrast Scaling     (factor=0.75)")
    print("      - JPEG Compression     (quality=45)")

    suite = RobustnessSuite(rng_seed=args.seed)
    reports = evaluate_pareto_front(
        front=pareto_candidates,
        decoder=decoder,
        x_test=test_x,
        y_test=test_y,
        fitness_fn=fitness_fn,
        suite=suite,
    )

    # 4. Print Summary Table & Export to CSV
    print("[4/5] Displaying formatted report and exporting CSV...")
    summarise(reports)

    df_report = to_dataframe(reports)
    csv_path = Path(args.output_csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(csv_path, index=False)
    print(f"      [CSV SAVED] Report written to: {csv_path}")

    # 5. Generate and Save Bar Chart
    print("[5/5] Generating per-perturbation retention bar chart...")
    chart_path = Path(args.output_chart).resolve()
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    perturbation_names = [
        "Gaussian Noise",
        "Gaussian Blur",
        "Brightness",
        "Contrast",
        "JPEG",
    ]
    perturbation_keys = [
        "gaussian_noise",
        "gaussian_blur",
        "brightness",
        "contrast",
        "jpeg",
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_width = 0.8 / len(reports)
    indices = np.arange(len(perturbation_names))

    colors = ["#3f51b5", "#009688", "#ff9800", "#e91e63", "#9c27b0"]

    for i, (cand_idx, rep) in enumerate(reports.items()):
        retentions = [
            rep.fitness_retentions.get(k, 0.0) for k in perturbation_keys
        ]
        color = colors[i % len(colors)]
        offset = (i - (len(reports) - 1) / 2) * bar_width
        ax.bar(
            indices + offset,
            retentions,
            bar_width,
            label=f"Candidate #{cand_idx} (Score: {rep.overall_score:.2f})",
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_ylabel("Fitness Retention Ratio [0.0 to 1.0]", fontsize=11)
    ax.set_xlabel("Perturbation Type", fontsize=11)
    ax.set_title(
        "Kartezio Post-Hoc Robustness: Performance Retention under Distortions",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(perturbation_names, fontsize=10)
    ax.set_ylim(0.0, 1.08)
    ax.axhline(
        1.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label="Ideal Retention (1.0)",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=args.dpi)
    plt.close()

    print(f"      [CHART SAVED] Bar chart written to: {chart_path}\n")
    print(
        "[COMPLETE] Robustness benchmarking and visualization finished successfully."
    )


if __name__ == "__main__":
    main()
