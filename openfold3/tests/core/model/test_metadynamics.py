"""Standalone tests for the metadynamics module — no OF3 model needed."""

import torch

from openfold3.core.metadynamics.cv.rg import rg_cv
from openfold3.core.metadynamics.potentials.metadynamics import MetadynamicsPotential


def _synthetic_batch(B=1, N_atom=30, seed=0):
    """Build a minimal batch with Cα-only atoms for testing."""
    torch.manual_seed(seed)
    # ref_atom_name_chars: encode "CA" for every atom.
    # _get_atom_name_mask expects [..., N_atom, 4, 64] one-hot.
    chars = torch.zeros(B, N_atom, 4, 64)
    # "CA" -> ord('C')-32 = 35, ord('A')-32 = 33, pad chars are 0.
    chars[:, :, 0, 35] = 1.0
    chars[:, :, 1, 33] = 1.0
    chars[:, :, 2, 0] = 1.0
    chars[:, :, 3, 0] = 1.0
    return {
        "ref_atom_name_chars": chars,
        "atom_mask": torch.ones(B, N_atom),
    }


# ===========================================================================
# MetadynamicsPotential (Rg-based)
# ===========================================================================


def test_rg_basic():
    batch = _synthetic_batch()
    coords = torch.randn(1, 2, 30, 3) * 5.0       # [B=1, S=2, N=30, 3]
    rg = rg_cv(coords, batch)
    assert rg.shape == (1, 2)
    assert (rg > 0).all()


def test_metadynamics_empty_energy_is_zero():
    batch = _synthetic_batch()
    coords = torch.randn(1, 2, 30, 3) * 5.0
    pot = MetadynamicsPotential(cv_function=rg_cv, sigma=1.0, hill_height=1.0)
    e = pot.energy(coords, batch, step_idx=0, noise_level=torch.tensor(0.0))
    assert torch.allclose(e, torch.zeros_like(e))


def test_metadynamics_deposits_hills():
    batch = _synthetic_batch()
    coords = torch.randn(1, 2, 30, 3) * 5.0
    pot = MetadynamicsPotential(
        cv_function=rg_cv,
        sigma=1.0, hill_height=1.0,
        hill_interval=1, warmup=0.0, cutoff=1.0,
        noise_tempered_sigma=False,
    )
    pot.on_step_end(coords, batch, step_idx=0, num_steps=10)
    pot.on_step_end(coords, batch, step_idx=1, num_steps=10)
    assert pot._count == 2

    # Energy should now be > 0 at the deposited location.
    e = pot.energy(coords, batch, step_idx=2, noise_level=torch.tensor(0.0))
    assert (e > 0).all()


def test_metadynamics_gradient_pushes_away():
    """The gradient at a deposited hill center should be near zero
    (we're at the peak); perturbed slightly, it should point away."""
    batch = _synthetic_batch()
    coords = torch.randn(1, 1, 30, 3) * 5.0
    pot = MetadynamicsPotential(
        cv_function=rg_cv,
        sigma=2.0, hill_height=1.0,
        hill_interval=1, warmup=0.0, cutoff=1.0,
        noise_tempered_sigma=False,
    )
    pot.on_step_end(coords, batch, step_idx=0, num_steps=10)

    # Perturb coords; gradient should be non-trivial.
    coords_perturbed = coords + 0.5 * torch.randn_like(coords)
    g = pot.gradient(coords_perturbed, batch, step_idx=1,
                     noise_level=torch.tensor(0.0))
    assert g.shape == coords.shape
    assert g.abs().sum() > 0


def test_metadynamics_gradient_empty_returns_zeros():
    """Before any hills are deposited, gradient must be zero — and not crash."""
    batch = _synthetic_batch()
    coords = torch.randn(1, 2, 30, 3) * 5.0
    pot = MetadynamicsPotential(cv_function=rg_cv, sigma=1.0, hill_height=1.0)
    g = pot.gradient(coords, batch, step_idx=0, noise_level=torch.tensor(0.0))
    assert g.shape == coords.shape
    assert torch.allclose(g, torch.zeros_like(g))
