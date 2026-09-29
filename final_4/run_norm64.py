"""final_4 with the norm64 recipe developed on final_4x2.

Same architecture and training as final_4x2/run_norm64.py -- a narrower
DeepONet (BRANCH_DIM 64) with the BRANCH output normalised, and the physics
imposed at parameter points that have no FDM data -- on the 2x2 arm instead of
the 4x2 one. Against final_4's existing DeepONet and MLP (unseen 0.1258 and
0.1160 on the 5x5 grid) this asks what the recipe is worth at the sparsest
sampling, where there are only 4 solves: 2 half-angles x 2 gaps.

N_PHYS_EXTRA = 8, i.e. twice as many data-free draws as there are training
geometries. That is the ratio the 4x2 experiments pointed at: with 4 draws
against 8 geometries the surrogate still drifted towards fitting its training
points (seen/unseen 2.6x at 14000 Adam epochs, 6.1x at 40000), and here the
parameter space is covered even more thinly.

PHYS_SPAN_* are left at 1.0, so the draws cover the whole sweep. On 4x2 the
error was a ridge along the gap axis and narrowing the draws to its middle
(PHYS_SPAN_APEX = 0.5) flattened it but gave some back at the trained gaps;
with only two values on EACH axis here, there is no reason yet to favour one
direction.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

# INT, not "4": GRIDS keys the square arms by integer (the Nx2 arms are strings
# because "4x2" cannot be one), so "4" raises KeyError.
ARM, ARCH = 4, "deeponet"
_, _, models, figs = DX._paths(ARM)
os.makedirs(models, exist_ok=True)
os.makedirs(figs, exist_ok=True)
npz = DX.npz_path(ARM)
if not os.path.exists(npz):
    raise SystemExit(f"{npz} not found -- run: python final_{ARM}/run.py --fdm")

cfg = {
    "SINGLE_GEOMETRY": False,
    "N_PARAM_BATCH": min(DX.PARAM_BATCH, DX.n_samples(ARM)),   # all 4 geometries
    "ACCUM_CHUNK": DX.ACCUM_CHUNK,
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    # Adam-heavy on purpose: on 4x2, L-BFGS moved nothing in three runs, while
    # Adam carried the model from 0.1218 to 0.0444 on unseen parameters.
    "ADAM_EPOCHS": 40000, "LBFGS_STEPS": 0000,
    # the architecture that worked on 4x2
    "BRANCH_DIM": 64, "TRUNK_NORM": False, "BRANCH_NORM": True,
    # physics where there is no data
    "N_PHYS_EXTRA": 12, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 0.6, "PHYS_SPAN_APEX": 0.6,
}

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": f"final_{ARM}_{ARCH}_norm64",
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": None,
    "CONFIG_OVERRIDES": cfg,
})
