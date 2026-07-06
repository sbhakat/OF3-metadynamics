"""Potential interface for coordinate-space bias on OF3 diffusion sampling."""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class Potential(ABC):
    """Coordinate-space bias evaluated inside SampleDiffusion.forward.

    Subclasses implement `energy`; `gradient` is derived via autograd by default.
    Stateful subclasses override `on_step_end` (e.g. metadynamics hill deposition).
    """

    @abstractmethod
    def energy(
        self,
        coords: torch.Tensor,         # [B, S, N_atom, 3]
        batch: dict,
        step_idx: int,
        noise_level: torch.Tensor,    # scalar tensor
    ) -> torch.Tensor:                # [B, S] energy per (batch, sample)
        ...

    def gradient(
        self,
        coords: torch.Tensor,
        batch: dict,
        step_idx: int,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        with torch.enable_grad():
            coords_ = coords.detach().requires_grad_(True)
            e = self.energy(coords_, batch, step_idx, noise_level)
            if step_idx % 20 == 0:
                count = getattr(self, "_count", "?")
                e_max = float(e.detach().max()) if hasattr(e, "detach") else "?"
                print(
                    f"[POT-DIAG] step={step_idx} _count={count} "
                    f"e.requires_grad={e.requires_grad} e.max={e_max:.4e}",
                    flush=True,
                )
            if not e.requires_grad:
                return torch.zeros_like(coords)
            (g,) = torch.autograd.grad(e.sum(), coords_, create_graph=False)
            if step_idx % 20 == 0:
                print(
                    f"[POT-DIAG] step={step_idx} g.abs.max={float(g.abs().max()):.4e}",
                    flush=True,
                )
        return g

    def on_step_end(
        self,
        coords: torch.Tensor,
        batch: dict,
        step_idx: int,
        num_steps: int,
    ) -> None:
        """Override for stateful potentials. Default: no-op."""
        return None
