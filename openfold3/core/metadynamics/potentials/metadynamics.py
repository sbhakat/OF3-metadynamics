"""Metadynamics potential: Gaussian hills deposited in CV space.

The bias energy at step t is

    V(s; t) = weight * sum_i  h_i * exp(-0.5 * ((s(x) - s_i) / sigma_eff)^2)

where s_i are CV values recorded at past hill-deposition steps, h_i are
(optionally well-tempered) hill heights, and sigma_eff is the hill width
optionally inflated by the current diffusion noise level (AF3-ReD trick).

Gradient is taken by autograd via the base class.
"""

from __future__ import annotations
from typing import Callable

import torch

from openfold3.core.metadynamics.base import Potential


class MetadynamicsPotential(Potential):
    def __init__(
        self,
        cv_function: Callable[[torch.Tensor, dict], torch.Tensor],
        sigma: float = 2.0,
        hill_height: float = 0.5,
        hill_interval: int = 5,
        well_tempered: bool = False,
        bias_factor: float = 10.0,
        kT: float = 2.5,
        max_hills: int = 1000,
        weight: float = 1.0,
        warmup: float = 0.0,
        cutoff: float = 0.75,
        noise_tempered_sigma: bool = True,
    ):
        """
        Args:
            cv_function: callable (coords [B,S,N,3], batch dict) -> CV [B,S].
            sigma: hill width in CV units.
            hill_height: base hill height (CV-units energy).
            hill_interval: deposit every N diffusion steps.
            well_tempered: scale hill heights by exp(-V / (kT (gamma-1))).
            bias_factor: well-tempered gamma factor.
            kT: thermal energy scale used in well-tempered scaling.
            max_hills: pre-allocated buffer size; deposition stops at capacity.
            weight: global scale on the bias energy and its gradient.
            warmup: fraction of diffusion (0..1) before deposition activates.
            cutoff: fraction of diffusion (0..1) after which deposition stops.
            noise_tempered_sigma: if True, sigma_eff = sigma + current noise level.
        """
        self.cv_function = cv_function
        self.sigma = sigma
        self.hill_height = hill_height
        self.hill_interval = hill_interval
        self.well_tempered = well_tempered
        self.bias_factor = bias_factor
        self.kT = kT
        self.weight = weight
        self.warmup = warmup
        self.cutoff = cutoff
        self.noise_tempered_sigma = noise_tempered_sigma
        self._max_hills = max_hills

        # Lazily allocated state: shape depends on (B, S) at first call.
        self._hill_centers: torch.Tensor | None = None    # [max_hills, B, S]
        self._hill_heights: torch.Tensor | None = None    # [max_hills]
        self._count: int = 0

    # ------------------------------------------------------------------ utils

    def _allocate(self, coords: torch.Tensor) -> None:
        B, S = coords.shape[0], coords.shape[1]
        device, dtype = coords.device, coords.dtype
        self._hill_centers = torch.zeros(
            self._max_hills, B, S, device=device, dtype=dtype
        )
        self._hill_heights = torch.zeros(
            self._max_hills, device=device, dtype=dtype
        )

    def _gate_active(self, step_idx: int, num_steps: int) -> bool:
        """Active if the current relaxation fraction is within [warmup, cutoff]."""
        relaxation = step_idx / max(num_steps, 1)
        return self.warmup <= relaxation <= self.cutoff

    def _effective_sigma(self, noise_level: torch.Tensor) -> float:
        if self.noise_tempered_sigma:
            return self.sigma + float(noise_level.detach().item())
        return self.sigma

    # ------------------------------------------------------------------ API

    def energy(
        self,
        coords: torch.Tensor,
        batch: dict,
        step_idx: int,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        """Sum-of-Gaussians bias at the current CV value. Shape [B, S]."""
        if self._hill_centers is None:
            self._allocate(coords)
        s = self.cv_function(coords, batch)                   # [B, S]
        if self._count == 0:
            return torch.zeros_like(s)
        eff_sigma = self._effective_sigma(noise_level)
        centers = self._hill_centers[: self._count]           # [K, B, S]
        heights = self._hill_heights[: self._count]           # [K]
        diff = s.unsqueeze(0) - centers                       # [K, B, S]
        gauss = torch.exp(-0.5 * (diff / eff_sigma) ** 2)     # [K, B, S]
        V = (heights.view(-1, 1, 1) * gauss).sum(dim=0)       # [B, S]
        return self.weight * V

    def on_step_end(
        self,
        coords: torch.Tensor,
        batch: dict,
        step_idx: int,
        num_steps: int,
    ) -> None:
        """Deposit a hill at the current CV value, if gated on and at an
        interval-aligned step. Called by the sampler after each DDIM update."""
        if not self._gate_active(step_idx, num_steps):
            return
        if step_idx % self.hill_interval != 0:
            return
        if self._count >= self._max_hills:
            return  # buffer full; silently stop depositing.

        if self._hill_centers is None:
            self._allocate(coords)

        with torch.no_grad():
            s = self.cv_function(coords, batch)               # [B, S]
            if self.well_tempered and self._count > 0:
                V = self.energy(
                    coords,
                    batch,
                    step_idx,
                    noise_level=torch.tensor(0.0, device=coords.device),
                )
                scale = torch.exp(-V.mean() / (self.kT * (self.bias_factor - 1)))
                h = self.hill_height * scale
            else:
                h = torch.tensor(
                    self.hill_height, device=coords.device, dtype=coords.dtype
                )
            self._hill_centers[self._count] = s
            self._hill_heights[self._count] = h
            self._count += 1
