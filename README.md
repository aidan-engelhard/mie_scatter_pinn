# mie_scatter_pinn

Physics-informed neural networks for 2-D electromagnetic scattering from a
conospherical AFM tip above a dielectric substrate, for infrared atomic force
microscopy (IR-AFM).

The problem is the TM Helmholtz equation on `[0,1]²` with an absorbing tip
(`n = 3.0 + 30.0i`), a dielectric substrate (`n = 1.5 + 0.1i`) filling
`y < 0.4`, a plane wave incident at −45°, and a UPML absorbing band. The
surrogate is trained over two geometric parameters — the tip half-angle
`α ∈ [10°, 30°]` and the tip–sample gap `g ∈ [0.01, 0.20]` — so that a new
geometry is a forward pass rather than a new mesh and a new solve.

---

## Getting started

```bash
pip install torch numpy scipy matplotlib imageio
```

A GPU is assumed for training; everything else runs on CPU. On Windows set
`PYTHONIOENCODING=utf-8` before running anything, or the console printing of
`λ` and `φ` will raise a `UnicodeEncodeError` under the default cp1252 codec.

**The repository is about 3 GB** — trained weights are tracked, not ignored
(120 checkpoints across the arms), so a shallow clone is worth considering if
you only want the code.

To convince yourself the stack works, without spending GPU time:

```bash
python final_4/run.py --mlp --quick
```

---

## Layout

### The numbered directories are the development history

Each is a self-contained, single-geometry study that adds one piece of physics.
They are kept because they are the simplest place to see each mechanism in
isolation, and because the final code cites them by name in its comments.
Read them in order if you want to understand where the final loss came from.

| Directory | What it adds | Scripts |
|---|---|---|
| `1_pec/` | Perfectly conducting cylinder; introduces Fourier feature embeddings | `pec_fdm.py`, `pec_fourier_pinn.py`, `pec_mlp_pinn.py` |
| `2_cylinder/` | Penetrable cylinder; Sommerfeld vs UPML boundary treatments | `mie_fdm*.py`, `mie_pinn*.py` |
| `3_afm_tip/` | Conospherical tip geometry in place of a circle | `afm_tip_fdm.py`, `afm_tip_pinn.py` |
| `4_sample/` | Dielectric substrate and the Fresnel background field | `mie_fdm_sample.py`, `mie_pinn_sample.py` |
| `5_afm_pinn/` | Tip and substrate together — the full single-geometry problem | `afm_fdm.py`, `afm_pinn.py` |
| `6_variable_gap/` | First parametric sweep, over the gap only. See its own READMEs in `single_gap/` and `all_gaps/` | `variable_sample_*.py`, `single_deeponet.py` |
| `7_variable_tips/` | Parametric sweep over the half-angle; origin of the batched parametric code | `tip_sweep_*.py` |

Each `*_fdm.py` writes the finite-difference reference its `*_pinn.py`
counterpart trains against or is scored on.

### `final/` — the shared machinery

Nothing in `final/` is run directly except the evaluation scripts. The training
scripts are executed *through* an arm's launcher, which supplies the
configuration.

| File | Role |
|---|---|
| `final_fdm.py` | The reference solver. Vectorised UPML finite-difference sweep over both parameters; writes an arm's `inputs/fdm_final_<arm>.npz`. |
| `final_deeponet.py` | **The trainer.** All the physics, samplers, losses, optimisers, monitors and figures. ~3400 lines, and the single source of truth — the MLP is a flag, not a separate script. |
| `final_mlp.py` | Thin wrapper running `final_deeponet.py` with `ARCH = "mlp"`. Kept for provenance; the arms set `ARCH_OVERRIDE` instead. |
| `data_experiment.py` | **Shared configuration for every arm.** Budgets, batch sizes, the arm→grid mapping, and the generic `run_train` used by `run.py`. Change a budget here and every arm changes together. |
| `evaluate_arms.py` | Scores trained checkpoints against fresh FDM solutions on a `k × k` grid, writes per-model CSVs and heat maps. The main evaluation entry point. |
| `evaluate_final.py` | Rebuilds a network from a checkpoint and evaluates a single model. Its `build_nets` / `predict_scat` / `fdm_at_pairs` helpers are reused by the other evaluation scripts — import these rather than re-deriving the composite field. |
| `evaluate_extrap.py` | Same as `evaluate_arms.py` but on a parameter box *wider* than the training range. |
| `compare_k50.py` | Redraws several models' error maps on one shared colour scale, from CSVs already written. Recomputes nothing. |
| `inputs/` | Prebuilt evaluation references: `eval_refs_k50_lam1/` (2500 solves), `eval_refs_k5_lam1.npz`, `extrap_refs_lam1/`. Generating these is expensive — do not delete them. |
| `legacy/` | Copies of `final_deeponet.py` and `evaluate_arms.py` from before the per-run field figures and training movie were removed. Still runnable; see `legacy/README.md`. |

### `final_<arm>/` — the experiments

An **arm** is one sampling density of the parameter sweep. The name encodes the
grid, via `GRIDS` in `data_experiment.py`:

| Arm | Grid (α × gap) | Solves |
|---|---|---|
| `final_4` | 2 × 2 | 4 |
| `final_4x2` | 4 × 2 | 8 |
| `final_6x2` | 6 × 2 | 12 |
| `final_16` | 4 × 4 | 16 |
| `final_36` | 6 × 6 | 36 |
| `final_64` | 8 × 8 | 64 |
| `final_100` | 10 × 10 | 100 |

Every arm has the same four subdirectories:

```
final_4/
  inputs/    fdm_final_4.npz          the FDM reference solutions
  models/    final_4_mlp_d_weights.pt  trained checkpoints
  figures/   final_4_mlp_d_relL2.csv   training logs, loss plots, error maps
  run*.py    the launchers
  */legacy/  results from before the budgets were matched — see below
```

**`final_4` and `final_4x2` are the two arms the final comparison was run on.**
The others (`16`, `36`, `64`, `100`) were the earlier data-quantity study and
carry only the baseline `run.py`.

---

## Running things

### Building an arm's reference data

```bash
python final_4/run.py --fdm
```

`run.py` accepts `--fdm`, `--deeponet`, `--mlp`, `--both`, `--quick`,
`--warm`, `--eval` and `--eval-fine`. It trains the *baseline* configuration
from `data_experiment.py`; the cell launchers below override it.

### The comparison launchers (`final_4/`, `final_4x2/`)

Data is always present; the two binary choices are whether the physics
residuals are imposed at the geometries that have a reference solve, and
whether they are also imposed at parameter points that have none
(`N_PHYS_EXTRA` draws, redrawn during training).

| Launcher | Physics at solved geometries | Physics at data-free draws |
|---|---|---|
| `run_d.py` | — | — |
| `run_dp.py` | ✔ | — |
| `run_dpx.py` | ✔ | ✔ |
| `run_dx.py` | — | ✔ |

```bash
python final_4/run_dpx.py        # seed 0
python final_4/run_dpx.py 1      # seed 1, tagged _s1
```

Two further launchers:

- `run_dpx_wsweep.py <m>` — cell DPX with every physics weight scaled by `m`,
  at a reduced Adam-only budget. Used to locate the physics/data balance.
  `m = 0` removes the physics entirely and reproduces cell D at the same budget.
- `run_dpx_full.py <m> [seed]` — the same multiplier at the full matched budget.
- `run_d_noansatz.py` — cell D with `SKIN_ANSATZ` off, the ablation for the
  interior boundary-layer ansatz.

`run_mlp_phys.py`, `run_norm64.py` and `rundataonly.py` predate the cell naming
and are superseded by the table above. `run_mlp_phys.py` in particular is cell
DPX under an older tag — running it duplicates `run_dpx.py` under a different
name.

### Evaluation

```bash
python final/evaluate_arms.py --k 50 --arms 4 4x2
```

Discovers every checkpoint in the arms' `models/` directories by glob, scores
each on the `k × k` grid, and writes `<tag>_eval_k<k>.csv` and a heat map.
Flags: `--resolve` to regenerate references, `--solve-only`, and `--part I/N`
to split the FDM solves across parallel processes.

```bash
python final/compare_k50.py --arm 4x2 --archs mlp_d mlp_dpx_w0p01_full
```

---

## Conventions worth knowing

### Configuration flows one way

Launchers pass `CONFIG_OVERRIDES` and a handful of `*_OVERRIDE` globals into
`final_deeponet.py` through `runpy.run_path`. The trainer applies them at one
point, after every constant is defined, and **raises if a key does not already
name a module constant** — so a typo in a launcher cannot silently run the
wrong configuration.

⚠ **If you add a flag that other constants derive from**, recompute those
derived constants after the override block, as is done for `_B_INT_SCALE` and
`_MARGIN_IN` (which depend on `SKIN_ANSATZ`) and for the RNG seeding. Anything
derived *above* the block keeps its default-derived value, which produces a
configuration that is internally inconsistent and fails for reasons unrelated
to the flag. `lam` and `FDM_PATH` are deliberately read at their own point of
definition for the same reason.

⚠ **Square arms are keyed by integer, N×2 arms by string.** `ARM = 4`, not
`ARM = "4"` — the latter raises `KeyError` from `GRIDS`.

### Checkpoints

A run writes several files per tag:

| File | Meaning |
|---|---|
| `<tag>_weights.pt` | Final weights |
| `<tag>_best_weights.pt` | Lowest relL2 seen during training |
| `<tag>_latest_weights.pt` | Written periodically, so a running job can be scored |
| `<tag>_pre_warm_weights.pt` | For a continuation, the weights it started from |
| `<tag>_best_prev<N>_weights.pt` | An earlier run's best, archived rather than overwritten |

⚠ **`_best_` is selected on relL2 at the *training* geometries.** When the
physics constrains the parameter space this stops being a good proxy for
generalization and can invert — score `_latest_` and the final weights too.

Each checkpoint carries a `hyperparams` dict with everything needed to rebuild
and interpret it, including `ARCH`, `SKIN_ANSATZ`, `SEED`, `PHYS_AT_DATA` and
the loss weights. `evaluate_final.build_nets` reads it; anything that rebuilds
a network by hand must read it too, or the reconstructed field will be wrong.

### Continuation runs

Pointing `INIT_WEIGHTS_OVERRIDE` at a tag's **own** `<tag>_weights.pt` is
detected as a continuation: the step count carries on, the relL2 CSV is
appended to, figures are redrawn over the whole training, and the starting
weights are preserved as `<tag>_pre_warm_weights.pt`. Pointing it at a
*different* file is an ordinary warm start.

The loss figure stays per-run either way, because a continuation re-measures
its loss normalisation at a fresh initialisation and the normalised terms are
not comparable across the restart.

### `legacy/` directories

`final_4/models/legacy/`, `final_4/figures/legacy/` and the `final_4x2/`
equivalents hold every run trained *before* the training budgets were matched
across methods. Their L-BFGS budgets range from 10,000 to 250,000 evaluations
and each is a single seed, so they can be quoted individually but not compared
with each other. The evaluation scripts glob flat directories only, so
legacy results are never picked up by accident.

### Multiple GPUs

The workload is kernel-launch-bound rather than compute-bound, so a bigger card
is not necessarily faster — host CPU speed matters more. To pin a run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python final_4/run_dpx.py
```

`CUDA_DEVICE_ORDER=PCI_BUS_ID` is not optional: CUDA's default enumeration is a
speed heuristic and does not match `nvidia-smi -L`.

---

## Reproducing a result from scratch

```bash
python final_4x2/run.py --fdm                     # reference solves, once
python final_4x2/run_dpx_full.py 0.01             # train
python final/evaluate_arms.py --k 50 --arms 4x2   # score
```

The evaluation references in `final/inputs/eval_refs_k50_lam1/` already exist;
regenerating them is 2500 FDM solves and is only necessary if the physical
constants change.
