"""Numerically stable Poincaré entailment cones.

This module follows Ganea, Bécigneul, and Hofmann (ICML 2018).  The cone
angle is measured at the apex, not at the origin.  For premise retrieval,
premises are cone apices and the current theorem/query is the contained point:

    premise p is admissible for theorem t iff z_t is in C(z_p).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _stable_acos(cosine: torch.Tensor) -> torch.Tensor:
    """Evaluate acos without the infinite endpoint derivative.

    Float32 dot products frequently round exactly to +/-1.  Clamping only the
    branch passed to acos keeps backpropagation finite, while the outer
    ``where`` restores the mathematically exact endpoint values.
    """
    tolerance = 1e-7 if cosine.dtype in (torch.float16, torch.float32) else 1e-12
    interior = torch.acos(
        cosine.clamp(min=-1.0 + tolerance, max=1.0 - tolerance)
    )
    interior = torch.where(
        cosine >= 1.0 - tolerance,
        torch.zeros_like(interior),
        interior,
    )
    return torch.where(
        cosine <= -1.0 + tolerance,
        torch.full_like(interior, math.pi),
        interior,
    )


def minimum_epsilon(cone_k: float) -> float:
    if cone_k <= 0:
        raise ValueError("cone_k must be positive")
    return 2.0 * cone_k / (1.0 + math.sqrt(1.0 + 4.0 * cone_k**2))


def validate_cone_parameters(cone_k: float, epsilon: float) -> None:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0, 1)")
    required = minimum_epsilon(cone_k)
    if epsilon + 1e-12 < required:
        raise ValueError(
            f"epsilon={epsilon} is too small for K={cone_k}; "
            f"need epsilon >= {required:.8f}"
        )


def project_to_annulus(
    points: torch.Tensor,
    epsilon: float,
    max_norm: float = 1.0 - 1e-5,
) -> torch.Tensor:
    if not 0.0 < epsilon < max_norm < 1.0:
        raise ValueError("require 0 < epsilon < max_norm < 1")
    norms = points.norm(dim=-1, keepdim=True)
    default_direction = torch.zeros_like(points)
    default_direction[..., 0] = 1.0
    directions = torch.where(
        norms > 1e-15,
        points / norms.clamp_min(1e-15),
        default_direction,
    )
    radii = norms.clamp(min=epsilon, max=max_norm)
    return directions * radii


def cone_half_aperture(
    apex: torch.Tensor,
    cone_k: float = 0.1,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Return psi(x)=asin(K(1-||x||²)/||x||) on the valid annulus."""
    validate_cone_parameters(cone_k, epsilon)
    radius = apex.norm(dim=-1)
    if torch.any(radius < epsilon - 1e-7):
        smallest = float(radius.detach().min().cpu())
        raise ValueError(
            f"cone apex radius {smallest:.8f} is below epsilon={epsilon}"
        )
    argument = cone_k * (1.0 - radius.square()) / radius.clamp_min(1e-15)
    return torch.asin(argument.clamp(min=-1.0, max=1.0))


def apex_angle(apex: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Xi(x,y)=pi-angle(O,x,y), the angle at cone apex x.

    Inputs broadcast over leading dimensions and must have the same final
    embedding dimension.  Identical points have angle zero.
    """
    x2 = apex.square().sum(dim=-1)
    y2 = target.square().sum(dim=-1)
    xy = (apex * target).sum(dim=-1)
    delta = (apex - target).norm(dim=-1)
    radicand = (1.0 + x2 * y2 - 2.0 * xy).clamp_min(0.0)
    numerator = xy * (1.0 + x2) - x2 * (1.0 + y2)
    denominator = (
        apex.norm(dim=-1)
        * delta
        * torch.sqrt(radicand.clamp_min(1e-30))
    )
    cosine = numerator / denominator.clamp_min(1e-30)
    cosine = cosine.clamp(min=-1.0, max=1.0)
    angle = _stable_acos(cosine)
    return torch.where(delta <= 1e-12, torch.zeros_like(angle), angle)


def origin_angle(apex: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The incorrect origin-centered comparator retained for the ablation."""
    cosine = F.cosine_similarity(apex, target, dim=-1, eps=1e-15)
    return _stable_acos(cosine.clamp(min=-1.0, max=1.0))


def cone_energy(
    apex: torch.Tensor,
    target: torch.Tensor,
    cone_k: float = 0.1,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Angular violation; zero means target belongs to the apex cone."""
    return F.relu(
        apex_angle(apex, target)
        - cone_half_aperture(apex, cone_k=cone_k, epsilon=epsilon)
    )


def origin_cone_energy(
    apex: torch.Tensor,
    target: torch.Tensor,
    cone_k: float = 0.1,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Origin-angle paper comparator; not a valid entailment-cone energy."""
    return F.relu(
        origin_angle(apex, target)
        - cone_half_aperture(apex, cone_k=cone_k, epsilon=epsilon)
    )


def inverse_cone_energy(
    query: torch.Tensor,
    premise_candidates: torch.Tensor,
    cone_k: float = 0.1,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """Score premises p for query t using t in C(p)."""
    return cone_energy(
        premise_candidates,
        query,
        cone_k=cone_k,
        epsilon=epsilon,
    )


def cone_rank_scores(
    energy: torch.Tensor,
    distance: torch.Tensor,
    containment_tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lexicographically rank contained points before fallback candidates."""
    membership = energy <= containment_tolerance
    scores = torch.where(
        membership,
        1e-6 * distance,
        1.0 + energy + 1e-6 * distance,
    )
    return scores, membership
