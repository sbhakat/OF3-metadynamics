"""Run OF3 inference with a metadynamics-like coordinate-space bias.

Deposits Gaussian hills on a scalar collective variable (radius of gyration)
during diffusion sampling. The bias is applied inside SampleDiffusion.forward
via the patched `potentials` hook. Inference-time only — no retraining.
"""

import logging
from pathlib import Path

import click

from openfold3.core.config import config_utils
from openfold3.entry_points.import_utils import _torch_gpu_setup


logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--query-json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--inference-ckpt-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
)
@click.option(
    "--runner-yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--use-msa-server", type=bool, default=False)
@click.option("--use-templates", type=bool, default=False)
@click.option("--num-diffusion-samples", type=int, default=5)
@click.option("--num-model-seeds", type=int, default=1)
@click.option("--cv", type=click.Choice(["rg"]), default="rg",
              help="Collective variable to bias")
@click.option("--sigma", type=float, default=2.0,
              help="Hill width in CV units")
@click.option("--hill-height", type=float, default=0.5,
              help="Base hill height (energy units of the bias)")
@click.option("--hill-interval", type=int, default=5,
              help="Deposit a hill every N diffusion steps")
@click.option("--weight", type=float, default=1.0,
              help="Global scale on the bias gradient")
@click.option("--warmup", type=float, default=0.0,
              help="Fraction of diffusion before deposition activates")
@click.option("--cutoff", type=float, default=0.75,
              help="Fraction of diffusion after which deposition stops")
@click.option("--well-tempered", is_flag=True,
              help="Enable well-tempered metadynamics scaling")
@click.option("--bias-factor", type=float, default=10.0,
              help="Well-tempered gamma factor (only used if --well-tempered)")
def main(
    query_json,
    inference_ckpt_path,
    output_dir,
    runner_yaml,
    use_msa_server,
    use_templates,
    num_diffusion_samples,
    num_model_seeds,
    cv,
    sigma,
    hill_height,
    hill_interval,
    weight,
    warmup,
    cutoff,
    well_tempered,
    bias_factor,
):
    """Inference with a metadynamics-like bias on a collective variable."""
    print(f"[METAD] main() entered, query_json={query_json}", flush=True)
    _torch_gpu_setup()

    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )
    from openfold3.core.metadynamics.cv.rg import rg_cv
    from openfold3.core.metadynamics.potentials.metadynamics import (
        MetadynamicsPotential,
    )

    logging.basicConfig(level=logging.INFO)
    print("[METAD] imports done", flush=True)

    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()
    expt_config = InferenceExperimentConfig(
        inference_ckpt_path=inference_ckpt_path,
        **runner_args,
    )
    expt_runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples,
        num_model_seeds,
        use_msa_server,
        use_templates,
        output_dir,
    )
    print("[METAD] expt_runner constructed", flush=True)

    query_set = InferenceQuerySet.from_json(query_json)
    expt_runner.setup()
    print("[METAD] setup done — weights loaded", flush=True)

    cv_registry = {"rg": rg_cv}
    pot = MetadynamicsPotential(
        cv_function=cv_registry[cv],
        sigma=sigma,
        hill_height=hill_height,
        hill_interval=hill_interval,
        weight=weight,
        warmup=warmup,
        cutoff=cutoff,
        well_tempered=well_tempered,
        bias_factor=bias_factor,
        noise_tempered_sigma=False,
    )
    print(
        f"[METAD] Attached MetadynamicsPotential(cv={cv}, sigma={sigma}, "
        f"hill_height={hill_height}, hill_interval={hill_interval}, "
        f"weight={weight}, warmup={warmup}, cutoff={cutoff}, "
        f"well_tempered={well_tempered})",
        flush=True,
    )

    expt_runner.lightning_module.model.metadynamics_potentials = [pot]

    expt_runner.run(query_set)
    expt_runner.cleanup()

    print(f"[METAD] Final hill count: {pot._count} / {pot._max_hills}", flush=True)


if __name__ == "__main__":
    main()
