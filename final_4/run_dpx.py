"""final_4 MLP, cell DPX of the 2x2: data + physics at those geometries + physics at data-free draws -- everything on.

Two binary factors, data always on: physics at the geometries that have an FDM
solve (P), and physics at data-free parameter draws (X).

                    no draws      draws
    no phys@data       D           DX
    phys@data          DP          DPX

The claim under test is an INTERACTION, not a main effect: physics at the data
points is redundant there by construction, but may still pay for itself by
conditioning the weights to satisfy the operator where there is no data. If
DP <= D while DPX > DX, that is real. If DPX ~= DX, it is pure cost.

Every cell runs at data_experiment.MATCHED_* so the comparison is of objectives
and not of training length -- except D, which is trained to convergence at
DATAONLY_* instead; see the note there for why, and say so in the writeup.

SEED is set explicitly so repeats are reproducible and land in the checkpoint
alongside PHYS_AT_DATA, which is what lets a scored model be placed in the grid
later.

    python final_4/run_dpx.py [seed]
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

ARM, ARCH, CELL = 4, "mlp", "dpx"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
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
    "SEED": SEED,
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    "ADAM_EPOCHS": DX.MATCHED_ADAM, "LBFGS_STEPS": DX.MATCHED_LBFGS,
    "PHYS_AT_DATA": True,
    "N_PHYS_EXTRA": 8, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 1.0,
}

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": f"final_{ARM}_{ARCH}_{CELL}"
                        + (f"_s{SEED}" if SEED else ""),
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": None,
    "CONFIG_OVERRIDES": cfg,
})
