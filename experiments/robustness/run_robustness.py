# -------------------------------------------------------------------------
# Kartezio - Robustness Experiment
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""Example: apply basic perturbations to a single image and run each
Pareto-front pipeline from a completed NSGA-II training run.

This script demonstrates how to connect the new robustness module to the
existing Kartezio pipeline without modifying any existing source files.

Usage (from the repository root):

    python experiments/robustness/run_robustness.py \
        --pareto-dir /path/to/pareto_results \
        --image     /path/to/image.png \
        --severity  0.1

The script prints the output shapes produced by each pipeline for each
perturbation type, which can be extended into a quantitative evaluation
in later stages.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# ---------------------------------------------------------------------------
# These imports use the EXISTING Kartezio API - no modifications.
# ---------------------------------------------------------------------------
from kartezio.moo import NSGA2Trainer, ParetoFront

# ---------------------------------------------------------------------------
# These imports use the NEW robustness module.
# ---------------------------------------------------------------------------
from kartezio.robustness import ImagePerturbation, ParetoEvaluator, PerturbationType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_grayscale_image(path: str) -> list:
    """Load a grayscale image as a single-channel DataList.

    Returns a list containing one 2-D uint8 numpy array (H x W).
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {path!r}.")
    return [img]


def _load_pareto_front_and_decoder(pareto_dir: str):
    """Placeholder: load a ParetoFront and its decoder from a directory.

    In a real experiment this would deserialise the NSGA2Trainer state.
    Here we show the interface contract only.
    """
    raise NotImplementedError(
        "Provide a concrete implementation that loads ParetoFront and "
        "DecoderCGP from your saved training artefacts."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run robustness perturbations through Pareto-front pipelines."
    )
    parser.add_argument("--pareto-dir", required=True, help="Directory with saved Pareto front.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--severity",
        type=float,
        default=0.1,
        help="Perturbation severity in [0, 1]. Default: 0.1",
    )
    args = parser.parse_args()

    # 1. Load the input image.
    channels = _load_grayscale_image(args.image)
    print(f"Loaded image with {len(channels)} channel(s), shape={channels[0].shape}.")

    # 2. Load the Pareto front and decoder from a completed training run.
    #    Replace this call with your actual loading logic.
    try:
        pareto_front, decoder = _load_pareto_front_and_decoder(args.pareto_dir)
    except NotImplementedError as exc:
        print(f"[INFO] {exc}", file=sys.stderr)
        print("Exiting — implement _load_pareto_front_and_decoder to use this script.")
        return

    # 3. Create the evaluator.
    evaluator = ParetoEvaluator(pareto_front, decoder)
    print(f"Pareto front has {evaluator.front_size} individual(s).")

    # 4. Iterate over all four perturbation types.
    for p_type in PerturbationType:
        perturber = ImagePerturbation(p_type, severity=args.severity)
        print(f"\nPerturbation: {p_type.value}  (severity={args.severity})")

        # Run all Pareto-front pipelines for this perturbation.
        results = evaluator.run_all(channels, perturbation=perturber)

        for idx, (output, elapsed) in enumerate(results):
            # output is a DataBatch: list of DataList (one entry per image).
            # Since we pass one image, output[0] is the DataList for that image.
            per_image_output = output[0]
            shapes = [arr.shape for arr in per_image_output]
            print(
                f"  Individual {idx:3d} | time={elapsed:.4f}s | "
                f"output shapes: {shapes}"
            )


if __name__ == "__main__":
    main()
