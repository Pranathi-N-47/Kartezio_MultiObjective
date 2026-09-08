# -------------------------------------------------------------------------
# Kartezio - Robustness Module
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""Albumentations-based image perturbations for robustness evaluation.

This module defines the four basic perturbations used in the robustness
evaluation stage:

  1. GaussianNoise   - additive Gaussian noise.
  2. GaussianBlur    - Gaussian spatial blurring.
  3. Brightness      - multiplicative brightness shift.
  4. Contrast        - contrast adjustment.

Each perturbation is applied to images that are in Kartezio DataList format,
i.e. a list of 2-D (H x W) uint8 numpy arrays, one per input channel.

Implementation note (Albumentations 2.x API):
  - GaussNoise  : ``std_range`` is expressed in normalised [0, 1] range, where
                  1.0 corresponds to the full [0, 255] pixel range.
                  severity=0.1  -> std ~ 25 DN  (10 % of max range).
  - GaussianBlur: ``sigma_limit`` is in pixels; ``blur_limit`` (kernel size)
                  is derived from sigma so that the kernel is always odd.
  - RandomBrightnessContrast: ``brightness_limit`` and ``contrast_limit`` are
                  additive offsets in normalised [0, 1] range.

No existing Kartezio source files are modified by this module.
"""

from __future__ import annotations

from enum import Enum
from typing import List

import numpy as np

import albumentations as A

from kartezio.types import DataList


class PerturbationType(Enum):
    """Enumeration of the four supported image perturbation types."""

    GAUSSIAN_NOISE = "gaussian_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"


def _build_transform(perturbation: PerturbationType, severity: float) -> A.BasicTransform:
    """Build an Albumentations transform for the requested perturbation type.

    Parameters
    ----------
    perturbation : PerturbationType
        Which perturbation to construct.
    severity : float
        A continuous severity parameter in [0, 1].  Higher values produce
        stronger perturbations.  Its interpretation differs per type:

        GAUSSIAN_NOISE
            ``std_range`` (normalised) = ``(severity, severity)``.
            severity=0.1  -> Gaussian noise with std ~ 25 DN.
        GAUSSIAN_BLUR
            ``sigma_limit`` = ``(severity * 14.0, severity * 14.0)`` pixels.
            severity=0.5  -> sigma ~ 7 px; kernel size derived to be odd.
        BRIGHTNESS
            ``brightness_limit`` = ``(-severity, severity)`` (normalised).
            severity=0.2  -> up to ±51 DN brightness shift.
        CONTRAST
            ``contrast_limit`` = ``(-severity, severity)`` (normalised).
            severity=0.2  -> up to ±20 % contrast change.

    Returns
    -------
    A.BasicTransform
        An Albumentations transform that is always applied (p=1.0).

    Raises
    ------
    ValueError
        If ``severity`` is not in [0, 1] or ``perturbation`` is unknown.
    """
    if not (0.0 <= severity <= 1.0):
        raise ValueError(
            f"severity must be in [0, 1], got {severity!r}."
        )

    if perturbation is PerturbationType.GAUSSIAN_NOISE:
        # std_range is in normalised [0, 1] relative to the [0, 255] range.
        std_norm = severity
        return A.GaussNoise(std_range=(std_norm, std_norm), p=1.0)

    if perturbation is PerturbationType.GAUSSIAN_BLUR:
        # sigma in pixels; map severity [0, 1] -> [0, 14] px.
        sigma = severity * 14.0
        if sigma < 1e-6:
            # Zero-sigma: identity (pass through unchanged).
            sigma = 1e-6
        # Kernel size must be odd and >= 3.
        ksize = max(3, int(6 * sigma + 1))
        if ksize % 2 == 0:
            ksize += 1
        return A.GaussianBlur(
            blur_limit=(ksize, ksize),
            sigma_limit=(sigma, sigma),
            p=1.0,
        )

    if perturbation is PerturbationType.BRIGHTNESS:
        # brightness_limit is an additive offset in normalised [0, 1] range.
        limit = severity
        return A.RandomBrightnessContrast(
            brightness_limit=(-limit, limit),
            contrast_limit=(0.0, 0.0),
            p=1.0,
        )

    if perturbation is PerturbationType.CONTRAST:
        # contrast_limit is an additive factor in normalised [0, 1] range.
        limit = severity
        return A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0),
            contrast_limit=(-limit, limit),
            p=1.0,
        )

    raise ValueError(f"Unknown perturbation type: {perturbation!r}.")


class ImagePerturbation:
    """Apply a single Albumentations perturbation to a Kartezio DataList image.

    A Kartezio image (``DataList``) is a list of 2-D ``uint8`` numpy arrays,
    one per input channel.  This class applies the configured perturbation
    independently to each channel and returns a new ``DataList`` of the same
    shape.

    Parameters
    ----------
    perturbation : PerturbationType
        The type of perturbation to apply.
    severity : float
        Severity in [0, 1].  See ``_build_transform`` for details.

    Examples
    --------
    >>> perturber = ImagePerturbation(PerturbationType.GAUSSIAN_NOISE, severity=0.1)
    >>> perturbed_channels = perturber.apply(original_channels)
    """

    def __init__(self, perturbation: PerturbationType, severity: float = 0.1) -> None:
        if not isinstance(perturbation, PerturbationType):
            raise TypeError(
                f"perturbation must be a PerturbationType, "
                f"got {type(perturbation).__name__!r}."
            )
        self.perturbation = perturbation
        self.severity = severity
        self._transform = _build_transform(perturbation, severity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, channels: DataList) -> DataList:
        """Apply the perturbation to every channel of a single Kartezio image.

        Parameters
        ----------
        channels : DataList
            A list of 2-D ``uint8`` numpy arrays (H x W), one per channel.
            This is the format produced by Kartezio dataset readers and used
            as input to ``DecoderCGP.decode``.

        Returns
        -------
        DataList
            A new list of perturbed 2-D ``uint8`` numpy arrays with the same
            shapes as the inputs.  The original arrays are not modified.

        Raises
        ------
        ValueError
            If ``channels`` is empty or any channel is not a 2-D array.
        """
        if not channels:
            raise ValueError("channels must be a non-empty list of numpy arrays.")

        perturbed: List[np.ndarray] = []
        for channel in channels:
            channel_arr = np.asarray(channel, dtype=np.uint8)
            if channel_arr.ndim != 2:
                raise ValueError(
                    f"Each channel must be a 2-D array, "
                    f"got shape {channel_arr.shape!r}."
                )
            # Albumentations expects HWC; promote single channel to HxWx1.
            hwc = channel_arr[:, :, np.newaxis]
            result = self._transform(image=hwc)["image"]
            # Strip the singleton channel dimension back to HxW.
            perturbed.append(result[:, :, 0])
        return perturbed

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ImagePerturbation(perturbation={self.perturbation.value!r}, "
            f"severity={self.severity!r})"
        )
