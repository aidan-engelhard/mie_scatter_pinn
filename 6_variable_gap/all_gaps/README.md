# all_gaps — runs trained ACROSS geometries

Every run here fits the same sweep: ten cylinder gaps from 0.01 to 0.20 at
lam = 0.5, eight for training and two held out, from
`inputs/fdm_variable_sample_sweep.npz`. The single-geometry controls are in
`../single_gap/`.

Two things vary between these runs, and it is worth keeping them apart:

- **the network.** `deeponet` is two independent networks, one for the field
  inside the cylinder and one for the scattered field outside, coupled only by
  the interface conditions. `single` is ONE shared body with two linear heads,
  from `single_deeponet.py`. "single" here means one shared network, NOT one
  geometry -- which is why folders named `single_data_*` are sweeps.
- **how much FDM data is used.** `W_DATA = 0` is physics only; larger values add
  supervision; "data only" switches the physics terms off entirely.

| folder | network | data | budget (Adam + L-BFGS evals) | checkpoint in `../models/` |
|---|---|---|---|---|
| `pure_physics_mlp` | MLP (batch 4) | physics only | 20 000 + 50 000 | `variable_sample_mlp_weights.pt` |
| `figures` | DeepONet, two nets | physics only | 10 000 + 5 000 | `variable_sample_deeponet_weights.pt` |
| `deeponet_data_10` | DeepONet, two nets | W_DATA = 10 | — | overwritten |
| `singe_data_10` (sic) | single shared body | W_DATA = 10 | — | overwritten |
| `single_data_1` | single shared body | W_DATA = 1 | 10 000 + 50 000 | `single_deeponet_weights_data_1.pt` |
| `figures_data_0_1` | single shared body | W_DATA = 0.1 | 20 000 + 50 000 | `single_deeponet_weights.pt` |
| `figures_data_only` | single shared body | data only (W_FIELD = W_PML = 0) | 10 000 + 30 000 | `single_deeponet_weights_data_only.pt` |
| `figures_sweep_deeponet` | DeepONet, two nets, batch 8 | physics only | 20 000 + 15 000 (`--short`) | `variable_sample_sweep_deeponet_short_weights.pt` |

Notes:

- **`figures_data_0_1` also carries the OLD loss weighting**, W_FIELD = 20 and
  W_PML = 2, against W_FIELD = 50 and W_PML = 0.5 everywhere else. Its result is
  therefore not comparable with the others on the data weight alone.
- **`figures_sweep_deeponet` holds only a relL2 CSV**, written by the full 8-gap
  sweep arm; the checkpoint surviving from that time is the `--short` one. Its
  budget and cadence match `../single_gap/figures_single_deeponet_short`.
- **Two runs have no checkpoint left**: a later run under the same tag
  overwrote it. Their figures are still here.
- `figures_data_0_1`, `figures_data_only`, `single_data_1` and `singe_data_10`
  also contain `*_gap_generalisation.png` and `*_gap_fields.png`, which are
  `evaluate_deeponet.py` outputs rather than training figures.
