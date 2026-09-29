"""final_4x2 MLP, with the physics also imposed where there is no FDM data.

The combination the 4x2 results point at but which has not been run. So far:

    mlp            physics + data at the 8 training geometries   unseen 0.0427
    deeponet_dataonly  data only, no physics                     unseen 0.0407
    deeponet_norm64    physics + data + data-free draws          unseen 0.0444

The MLP is the architecture that INTERPOLATES best -- its error map is nearly
flat (gap-axis spread 1.28x against the DeepONet's 2.6-4.4x), because the
parameters enter the same network as the coordinates, so nothing can specialise
per training geometry. N_PHYS_EXTRA is the only term that constrains the
surrogate BETWEEN training parameters. Neither has been tried with the other.

Budget is data_experiment.MATCHED_*, the one budget every method on final_4 and
final_4x2 is compared at, so the comparison is of objectives and not of
training length. The existing final_4x2_mlp it is compared against ran 250000
L-BFGS evaluations and has to be re-run at the matched budget before the two
numbers mean anything together.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

ARM, ARCH = "4x2", "mlp"
_, _, models, figs = DX._paths(ARM)
os.makedirs(models, exist_ok=True)
os.makedirs(figs, exist_ok=True)
npz = DX.npz_path(ARM)
if not os.path.exists(npz):
    raise SystemExit(f"{npz} not found -- run: python final_{ARM}/run.py --fdm")

cfg = {
    "SINGLE_GEOMETRY": False,
    "N_PARAM_BATCH": min(DX.PARAM_BATCH, DX.n_samples(ARM)),
    "ACCUM_CHUNK": DX.ACCUM_CHUNK,
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    "ADAM_EPOCHS": DX.MATCHED_ADAM, "LBFGS_STEPS": DX.MATCHED_LBFGS,
    # the one difference: physics at parameters with no data. 4 draws against 8
    # training geometries is what the DeepONet runs used; the gap axis is where
    # the error lives, so the draws are narrowed towards its middle.
    "N_PHYS_EXTRA": 4, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 0.5,
}

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": f"final_{ARM}_{ARCH}_phys",
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": None,
    "CONFIG_OVERRIDES": cfg,
})
