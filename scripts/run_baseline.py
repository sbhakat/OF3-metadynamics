"""Run standard OF3 inference — no metadynamics bias.

Mirrors run_metadynamics.py exactly but attaches no potential, so the output
is a plain baseline ensemble for fair comparison. Use the same
--num-diffusion-samples as your metad run.
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
def main(
    query_json,
    inference_ckpt_path,
    output_dir,
    runner_yaml,
    use_msa_server,
    use_templates,
    num_diffusion_samples,
    num_model_seeds,
):
    """Standard OF3 inference with no bias — baseline ensemble."""
    print(f"[BASELINE] main() entered, query_json={query_json}", flush=True)
    _torch_gpu_setup()

    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    logging.basicConfig(level=logging.INFO)
    print("[BASELINE] imports done", flush=True)

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
    print("[BASELINE] expt_runner constructed", flush=True)

    query_set = InferenceQuerySet.from_json(query_json)
    expt_runner.setup()
    print("[BASELINE] setup done — weights loaded", flush=True)

    # No potential attached — plain sampling.
    expt_runner.run(query_set)
    expt_runner.cleanup()

    print("[BASELINE] Done — no bias applied.", flush=True)


if __name__ == "__main__":
    main()
