"""final_4x2 trained on DATA ONLY: identical to run.py --deeponet except that
every physics weight is zero, so the only difference is the loss."""
import os
import numpy as np
import runpy
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

ARM, ARCH = 4, "mlp"
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
    "ADAM_EPOCHS": DX.ADAM_EPOCHS, "LBFGS_STEPS": DX.LBFGS_STEPS,
    # the only difference from the PINN run
    "W_PDE_INT": 0.0, "W_PDE_EXT": 0.0, "W_PDE_SUB": 0.0,
    "W_FIELD": 0.0, "W_FLUX": 0.0, "W_PML": 0.0, "W_DIRICHLET": 0.0,
    "ADAM_EPOCHS": 15000,
    "LBFGS_STEPS": 10000,
}

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": f"final_{ARM}_{ARCH}_dataonly",
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": None,
    "CONFIG_OVERRIDES": cfg,
})