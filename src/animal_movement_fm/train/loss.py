"""Tier 2 training losses for forecast and masked reconstruction.

Our current registered objective is radial pseudo-Huber in the window normalised displacement space with `delta=1.0`. The 
radial form depends only on 2D error magnitude, so it is rotation invariant and matches the geometry of the reported 
distance metrics. The same loss is then used for the forecast term and also auxiliary masked reconstruction term.
"""

from __future__ import annotations

import torch

HUBER_DELTA = 1.0  # fixed by prereg


def radial_pseudo_huber(
    error: torch.Tensor,
    *,
    delta: float = HUBER_DELTA,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Apply pseudo-Huber to the magnitude of a 2D displacement error.

    `error` is expected to have shape `(..., 2)`; loss computed radially over the final axis, then reduced with `"mean"`, `"sum"`, or `"none"`.
    """
    delta = float(delta)
    squared = error.pow(2).sum(dim=-1)
    loss = delta**2 * (torch.sqrt(1.0 + squared / delta**2) - 1.0)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"unknown reduction {reduction!r}")


def normalised_huber(
    pred_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    *,
    delta: float = HUBER_DELTA,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Apply radial pseudo-Huber to predicted and target displacements in normalised space.

    Both tensors must already be divided by the window scale. Keeping the loss in this space makes errors comparable across movement 
    scales; predictions are converted back to metres only for evaluation.
    """
    error = pred_scaled - target_scaled
    return radial_pseudo_huber(error, delta=delta, reduction=reduction)


def masked_reconstruction_loss(
    recon_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = HUBER_DELTA,
) -> torch.Tensor:
    """
    Compute the auxiliary reconstruction loss only over masked tokens.

    `mask` has shape `(batch, seq)` and marks tokens hidden from the trunk. If nothing is masked, a zero scalar connected to the computation 
    graph is returned so the training loop does not need a special case.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    if not mask.any():
        return recon_scaled.sum() * 0.0

    error = recon_scaled - target_scaled
    per_token = radial_pseudo_huber(error, delta=delta, reduction="none")
    return per_token[mask].mean()


def horizon_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = HUBER_DELTA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return `(loss, per_head)` for (batch, H, 2) predictions and targets with a (batch, H) target mask.

    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    # zeroed before the error so a non-finite missing target cannot reach the gradient through `where`
    target = torch.where(mask.unsqueeze(-1), target, torch.zeros_like(target))
    per_window = radial_pseudo_huber(pred - target, delta=delta, reduction="none")
    counts = mask.sum(dim=0)
    present = counts > 0
    head_means = torch.where(mask, per_window, torch.zeros_like(per_window)).sum(dim=0) / counts.clamp(min=1)
    loss = torch.where(present, head_means, torch.zeros_like(head_means)).sum() / present.sum()
    per_head = torch.where(present, head_means, torch.full_like(head_means, float("nan")))
    return loss, per_head
