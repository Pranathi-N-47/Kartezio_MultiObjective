# -------------------------------------------------------------------------
# Kartezio - Robustness Module
# Copyright (c) 2024-Present Inserm Transfert SA and co-owners
# Licensed under the same licence as the original Kartezio source code
# (see ../../LICENSE).  This file may be used, modified and redistributed
# for non-commercial research only, and must retain this header.
# -------------------------------------------------------------------------
"""ParetoEvaluator - interface between the robustness perturbation layer and
an existing Pareto-front Kartezio pipeline.

This module provides a single class, ``ParetoEvaluator``, that:

  1. Receives a Kartezio image (``DataList``) and an optional
     ``ImagePerturbation``.
  2. Optionally applies the perturbation to each channel of the image.
  3. Wraps the resulting image in the ``DataBatch`` format required by the
     existing ``DecoderCGP.decode`` method.
  4. Decodes a chosen genotype from the ``ParetoFront`` using the same
     ``DecoderCGP`` that was used during training.
  5. Returns the raw ``DataBatch`` output produced by the pipeline (i.e. the
     list of output arrays after the endpoint transform is applied), together
     with the elapsed inference time.

Interface contract
------------------
``pareto_front`` must expose:
  - ``get_size() -> int``
  - ``get_genotype(index: int) -> Genotype``

``decoder`` must expose:
  - ``decode(genotype, x: list) -> (list, float)``
    where ``x`` is a DataBatch (list of DataList) and the return value is
    ``(DataBatch, elapsed_time)``.

These are duck-typed to avoid triggering optional-dependency import chains
(``codecarbon``, etc.) that live in parent package ``__init__`` files.

No existing Kartezio source files are modified by this module.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .perturbations import ImagePerturbation


def _check_pareto_front(obj: Any) -> None:
    """Validate that *obj* exposes the ParetoFront interface."""
    required = ("get_size", "get_genotype")
    missing = [m for m in required if not callable(getattr(obj, m, None))]
    if missing:
        raise TypeError(
            f"pareto_front must expose callable methods {required}. "
            f"Missing: {missing}. Got type {type(obj).__name__!r}."
        )


def _check_decoder(obj: Any) -> None:
    """Validate that *obj* exposes the DecoderCGP interface needed here."""
    if not callable(getattr(obj, "decode", None)):
        raise TypeError(
            "decoder must expose a callable 'decode(genotype, x)' method. "
            f"Got type {type(obj).__name__!r}."
        )


class ParetoEvaluator:
    """Pass a (possibly perturbed) image through one pipeline from a ParetoFront.

    This class is the primary interface between the robustness perturbation
    layer and the existing Kartezio execution path.  It owns references to
    the ``ParetoFront`` produced by ``NSGA2Trainer.fit`` and the
    ``DecoderCGP`` that was used during training.  Neither object is
    modified.

    The output of ``run`` is whatever ``DecoderCGP.decode`` returns:
    a list with one entry per image in the batch.  Because a single image is
    passed, ``output[0]`` is a ``DataList`` (list of numpy arrays) produced
    by the pipeline endpoint.  The exact structure depends on the endpoint
    used during training (e.g., a watershed endpoint returns a labeled mask,
    a threshold endpoint returns a binary image).

    Parameters
    ----------
    pareto_front : ParetoFront
        The Pareto front produced by ``NSGA2Trainer.fit``.  Validated via
        duck typing: must expose ``get_size()`` and ``get_genotype(i)``.
    decoder : DecoderCGP
        The decoder used during training.  Validated via duck typing: must
        expose ``decode(genotype, x)``.

    Examples
    --------
    >>> evaluator = ParetoEvaluator(pareto_front, trainer.get_decoder())
    >>> output, elapsed = evaluator.run(
    ...     channels=image_channels,
    ...     individual_index=0,
    ...     perturbation=ImagePerturbation(PerturbationType.GAUSSIAN_NOISE, 0.1),
    ... )
    >>> per_image_output = output[0]   # DataList for the single input image
    """

    def __init__(self, pareto_front: Any, decoder: Any) -> None:
        _check_pareto_front(pareto_front)
        _check_decoder(decoder)
        self._pareto_front: Any = pareto_front
        self._decoder: Any = decoder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def front_size(self) -> int:
        """Number of individuals on the Pareto front."""
        return self._pareto_front.get_size()

    def run(
        self,
        channels: list,
        individual_index: int = 0,
        perturbation: Optional[ImagePerturbation] = None,
    ) -> Tuple[list, float]:
        """Apply an optional perturbation and run one Pareto-front pipeline.

        Parameters
        ----------
        channels : list of np.ndarray
            The input image as a list of 2-D ``uint8`` numpy arrays (H x W),
            one per channel.  This is the Kartezio ``DataList`` format.
        individual_index : int
            Index into the Pareto front (0-based).  Selects which genotype
            to decode and run.
        perturbation : ImagePerturbation or None
            When provided, the perturbation is applied to ``channels`` before
            passing the image to the pipeline.  When ``None`` the original
            channels are used unchanged.

        Returns
        -------
        output : list (DataBatch)
            The raw output of ``DecoderCGP.decode``.  This is a list with one
            entry per image in the batch.  Because a single image is passed
            here, ``output[0]`` is a ``DataList`` (list of numpy arrays)
            produced by the pipeline endpoint.
        elapsed : float
            Mean inference time in seconds as returned by the decoder.

        Raises
        ------
        IndexError
            If ``individual_index`` is out of range for the Pareto front.
        """
        if individual_index < 0 or individual_index >= self._pareto_front.get_size():
            raise IndexError(
                f"individual_index {individual_index!r} is out of range for a "
                f"ParetoFront of size {self._pareto_front.get_size()}."
            )

        # Optionally apply the perturbation to all channels.
        if perturbation is not None:
            channels = perturbation.apply(channels)

        # Wrap as a DataBatch: a list containing one DataList (one image).
        x = [channels]

        # Retrieve the genotype for the requested individual.
        genotype = self._pareto_front.get_genotype(individual_index)

        # Decode using the existing DecoderCGP - no modification to the
        # decoder or the Pareto front.  decode() returns (DataBatch, float).
        output, elapsed = self._decoder.decode(genotype, x)
        return output, elapsed

    def run_all(
        self,
        channels: list,
        perturbation: Optional[ImagePerturbation] = None,
    ) -> List[Tuple[list, float]]:
        """Run every individual on the Pareto front for one image.

        Parameters
        ----------
        channels : list of np.ndarray
            Input image as a list of 2-D ``uint8`` numpy arrays.
        perturbation : ImagePerturbation or None
            Optional perturbation applied once before iterating over the
            front.  All individuals receive the same perturbed image.

        Returns
        -------
        list of (DataBatch, float)
            One ``(output, elapsed)`` tuple per individual on the front,
            in the same order as the front indices (0 to front_size - 1).
        """
        # Apply perturbation once so that all individuals see the same image.
        if perturbation is not None:
            channels = perturbation.apply(channels)

        results: List[Tuple[list, float]] = []
        for idx in range(self._pareto_front.get_size()):
            x = [channels]
            genotype = self._pareto_front.get_genotype(idx)
            output, elapsed = self._decoder.decode(genotype, x)
            results.append((output, elapsed))
        return results

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ParetoEvaluator(front_size={self.front_size}, "
            f"decoder={type(self._decoder).__name__!r})"
        )
