"""Integration test for SampleDiffusion with metadynamics potentials.

Uses a MockDiffusionModule that returns xl_noisy unchanged (zero denoising),
so the only forces on the trajectory come from initial noise + bias potentials.
"""

import torch

from openfold3.core.model.structure.diffusion_module import SampleDiffusion
from openfold3.core.metadynamics.cv.rg import rg_cv
from openfold3.core.metadynamics.potentials.metadynamics import MetadynamicsPotential


class MockDiffusionModule:
    """No-op denoiser: returns xl_noisy unchanged."""
    def __call__(self, batch, xl_noisy, **kwargs):
        return xl_noisy


def _batch(B=1, N_atom=30):
    chars = torch.zeros(B, N_atom, 4, 64)
    chars[:, :, 0, 35] = 1.0   # 'C'
    chars[:, :, 1, 33] = 1.0   # 'A'
    return {
        "ref_atom_name_chars": chars,
        "atom_mask": torch.ones(B, N_atom),
        "token_mask": torch.ones(B, N_atom),
    }


def _make_sampler():
    return SampleDiffusion(
        gamma_0=0.0, gamma_min=0.0,
        noise_scale=0.0, step_scale=1.0,
        diffusion_module=MockDiffusionModule(),
    )


def _dummy_embeddings(B=1, S=1, N=30):
    # Shapes don't matter for the mock — they're not read.
    return (
        torch.zeros(B, S, N, 1),       # si_input
        torch.zeros(B, S, N, 1),       # si_trunk
        torch.zeros(B, S, N, N, 1),    # zij_trunk
    )


def test_sampler_runs_without_potentials():
    """Baseline: potentials=None reproduces unchanged SampleDiffusion behavior."""
    sampler = _make_sampler()
    batch = _batch()
    si_input, si_trunk, zij_trunk = _dummy_embeddings()
    noise_schedule = torch.linspace(10.0, 0.1, 5, dtype=torch.float32)

    torch.manual_seed(0)
    xl = sampler(
        batch=batch,
        si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
        noise_schedule=noise_schedule,
        no_rollout_samples=2,
        potentials=None,
    )
    assert xl.shape == (1, 2, 30, 3)
    assert torch.isfinite(xl).all()


def test_sampler_deposits_hills_during_run():
    """Hills accumulate inside the sampler loop."""
    sampler = _make_sampler()
    batch = _batch()
    si_input, si_trunk, zij_trunk = _dummy_embeddings()
    # 10 diffusion steps (schedule has 11 entries).
    noise_schedule = torch.linspace(10.0, 0.1, 11, dtype=torch.float32)

    pot = MetadynamicsPotential(
        cv_function=rg_cv,
        sigma=1.0, hill_height=0.5,
        hill_interval=1, warmup=0.0, cutoff=1.0,
        noise_tempered_sigma=False,
    )

    torch.manual_seed(0)
    sampler(
        batch=batch,
        si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
        noise_schedule=noise_schedule,
        no_rollout_samples=2,
        potentials=[pot],
    )
    # warmup=0, cutoff=1, hill_interval=1: all 10 steps deposit.
    assert pot._count == 10

def test_sampler_potentials_change_the_output():
    """Same seed, same inputs — only the bias should differ. Outputs must diverge.

    Uses gamma_0>0 and noise_scale>0 so the trajectory actually jitters in CV
    space between hill deposits; otherwise the gradient at the peak of a freshly
    deposited Gaussian is zero and the bias produces no observable effect.
    """
    batch = _batch()
    si_input, si_trunk, zij_trunk = _dummy_embeddings()
    noise_schedule = torch.linspace(10.0, 0.1, 6, dtype=torch.float32)

    # Stochastic sampler: gamma_0 inflates noise level; noise_scale injects randn.
    sampler = SampleDiffusion(
        gamma_0=1.0, gamma_min=0.0,
        noise_scale=1.0, step_scale=1.0,
        diffusion_module=MockDiffusionModule(),
    )

    torch.manual_seed(42)
    xl_baseline = sampler(
        batch=batch,
        si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
        noise_schedule=noise_schedule,
        no_rollout_samples=1,
        potentials=None,
    )

    pot = MetadynamicsPotential(
        cv_function=rg_cv,
        sigma=2.0, hill_height=2.0, weight=10.0,
        hill_interval=1, warmup=0.0, cutoff=1.0,
        noise_tempered_sigma=False,
    )

    torch.manual_seed(42)
    xl_biased = sampler(
        batch=batch,
        si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk,
        noise_schedule=noise_schedule,
        no_rollout_samples=1,
        potentials=[pot],
    )

    assert not torch.allclose(xl_baseline, xl_biased), \
        "Bias had no effect on the trajectory"
