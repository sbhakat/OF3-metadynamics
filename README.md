# OF3-metadynamics

Metadynamics-style conformational sampling for **OpenFold3**.

Standard OpenFold3 gives you one confident structure per protein. OF3-metadynamics
adds a **bias force** during diffusion sampling that pushes the model to explore
*different* conformations — so instead of one answer, you get a diverse ensemble.

No retraining is required. The bias is applied at inference time only.

---

## The idea 

During diffusion, we deposit Gaussian "hills" on a collective variable (CV) —
here the **radius of gyration** `Rg`. The accumulated hills form a bias energy:

```
V(x) = weight · Σ_i  h · exp( −(Rg(x) − Rg_i)² / (2σ²) )
```

- `Rg(x)`  — radius of gyration of the current structure
- `Rg_i`   — Rg where hill *i* was deposited
- `σ`      — hill width (`--sigma`)
- `h`      — hill height (`--hill-height`)
- `weight` — how hard the bias pushes (`--weight`)

The **gradient** of this energy is added to each diffusion step, nudging the
structure *away* from Rg values it has already visited. Fill up one basin, and
the model is forced to explore another.

```
x ← x_denoised  +  (score direction)  +  weight · ∇V(x)
```

---

## Install

OF3-metadynamics is built on top of OpenFold3. Install OpenFold3, then use this
tree (the bias code lives in `openfold3/core/metadynamics/`).

```bash
conda activate openfold3
pip install openfold3 
```

Run the unit tests (no GPU or weights needed):

```bash
python -m pytest openfold3/tests/core/model/test_metadynamics.py -v
```

---

## Quick start

```bash
python scripts/run_metadynamics.py \
    --query-json examples/example_inference_inputs/query_ubiquitin.json \
    --inference-ckpt-path of3-p2-155k.pt \
    --output-dir /tmp/of3-metad \
    --runner-yaml runner_no_ds.yml \
    --num-diffusion-samples 5 \
    --cv rg \
    --sigma 2.0 \
    --hill-height 2.0 \
    --hill-interval 5 \
    --weight 50.0 \
    --warmup 0.0 \
    --cutoff 0.75
```

This writes an ensemble of `.cif` structures to `--output-dir`.

### Run baseline

```bash
python scripts/run_baseline.py \
    --query-json query.json \
    --inference-ckpt-path of3-p2-155k.pt \
    --output-dir /tmp/of3-baseline-localmsa \
    --runner-yaml runner_no_ds.yml \
    --use-msa-server False \
    --use-templates False \
    --num-diffusion-samples 20
```

### Compare against baseline

Compute the Rg of every structure to see the spread:

```bash
python scripts/quick_rg.py /tmp/of3-baseline /tmp/of3-metad
```

Example result on ubiquitin (bias vs. no bias, same sample count):

```
baseline : mean 11.32 A, std 0.09 A   (one tight answer)
metad    : mean 13.30 A, std 1.63 A   (broad ensemble, 18x wider)
```

---

## Options

| Flag | Meaning | Typical |
|------|---------|---------|
| `--cv` | Collective variable to bias | `rg` |
| `--sigma` | Hill width (A of Rg) | `2.0` |
| `--hill-height` | Hill height | `0.5`-`2.0` |
| `--hill-interval` | Deposit a hill every N steps | `5` |
| `--weight` | How hard the bias pushes | `10`-`50` |
| `--warmup` | Fraction of diffusion before bias starts | `0.0` |
| `--cutoff` | Fraction of diffusion after which bias stops | `0.75` |
| `--well-tempered` | Self-limiting hills (gentler, converges) | off |
| `--bias-factor` | Well-tempered gamma (only with `--well-tempered`) | `10.0` |
| `--num-diffusion-samples` | Structures to generate | `5`-`20` |

### Tuning intuition

- **Not enough diversity?**  Raise `--weight`.
- **Structures look distorted?**  Lower `--weight`, or pull `--cutoff` earlier
  (e.g. `0.6`) so the model has more bias-free steps to clean up.
- **Want a stabler, converging bias?**  Add `--well-tempered`.

The `--cutoff` protects the final refinement steps: the bias picks the basin,
and the un-biased tail turns it into a physically valid structure.

---

## Supplying your own MSA (optional)

For best accuracy, supply a precomputed alignment instead of the online server.
Name the file with a recognized basename (e.g. `colabfold_main.a3m`) and point
the query JSON at it:

```json
{
  "queries": {
    "ubiquitin": {
      "chains": [
        {
          "molecule_type": "protein",
          "chain_ids": ["A"],
          "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
          "main_msa_file_paths": "/path/to/colabfold_main.a3m"
        }
      ]
    }
  }
}
```

Run with `--use-msa-server False`. The first sequence in the `.a3m` must be the
query itself.

---

## How it works (3 pieces)

1. **The CV** - `openfold3/core/metadynamics/cv/rg.py`
   A function that maps coordinates to a scalar (radius of gyration).

2. **The potential** - `openfold3/core/metadynamics/potentials/metadynamics.py`
   Deposits Gaussian hills on the CV and computes the bias gradient by autograd.

3. **The hook** - `openfold3/core/model/structure/diffusion_module.py`
   `SampleDiffusion.forward` adds the bias gradient to each diffusion step and
   deposits a hill after it.

The driver `scripts/run_metadynamics.py` wires these together and attaches the
potential to the model before inference.

---

## Notes

- **`runner_no_ds.yml` is required.** It sets `inference_mode: false` (so the
  bias gradient can be computed - Lightning's default blocks autograd) and
  `use_lma: true` (a portable attention backend that avoids a CUDA JIT step).
- **Fair comparisons:** the observed Rg range grows with sample count. Always
  compare baseline and metad at the *same* `--num-diffusion-samples`.
- **Debug output:** set `METAD_DEBUG=1` to print per-step bias magnitudes.

*Built using Claude Opus 4.8.*
