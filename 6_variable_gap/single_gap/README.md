# single_gap — runs trained on ONE geometry

Every run here fits a single cylinder position: gap index 2 (gap = 0.052) of the
ten gaps in `inputs/fdm_variable_sample_sweep.npz`, at lam = 0.5. They exist to
separate "can this physics be learnt at all" from "can one model span the gap
range", so the sweep runs in `../all_gaps/` differ from them only in the number
of geometries.

Read the checkpoints' `hyperparams` for the authoritative configuration; the
table below is from them, except where a run's checkpoint was later overwritten.

| folder | network | budget (Adam + L-BFGS closure evals) | data | checkpoint in `../models/` |
|---|---|---|---|---|
| `deeponet_gap2` | DeepONet, two independent nets | 10 000 + 50 000 | physics only | `variable_sample_deeponet_gap2_weights.pt` |
| `figures_single_mlp` | plain MLP | 3 000 + 250 000 | physics only | `variable_sample_single_mlp_gap2_weights.pt` |
| `figures_single_deeponet` | DeepONet, trunk depth 3 to match the MLP | 3 000 + 200 000 | physics only | `variable_sample_single_deeponet_gap2_weights.pt` |
| `figures_single_deeponet_10000_adam` | as above | 10 000 + 200 000 | physics only | overwritten by the run above |
| `figures_single_deeponet_short` | as above | 20 000 + 15 000 | physics only | `variable_sample_single_deeponet_short_gap2_weights.pt` |

What each one is for:

- **`deeponet_gap2`** — the first single-geometry run, made before the
  single-gap controls were set up as a bisection.
- **`figures_single_mlp`** — the control: one gap, a plain MLP, and
  `sample/mie_pinn_sample.py`'s weights and budget. Reached relL2 ~ 0.058, so
  the physics, losses, samplers and monitor are sound.
- **`figures_single_deeponet`** — the same control with a DeepONet head, so the
  only difference from the MLP run is the architecture. It imports the MLP's
  CONFIG rather than restating it, so budgets cannot drift.
- **`figures_single_deeponet_10000_adam`** — the earlier attempt at that run,
  with Adam at 10 000 instead of 3 000. Kept for its figures only.
- **`figures_single_deeponet_short`** — the `--short` diagnostic. Its budget and
  monitor cadence match `../all_gaps/figures_sweep_deeponet`'s short run, so the
  two relL2 logs can be compared step for step: one geometry against eight, with
  everything else identical. This folder has the CSVs, not just images.
