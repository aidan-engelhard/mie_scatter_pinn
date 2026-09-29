"""final_4x2 DeepONet, narrower and with the trunk output normalised:
BRANCH_DIM 64 (32 modes per component, against a measured participation of
16-34 and a rank-8 solution family) and TRUNK_NORM, which removes the
b -> c*b, t -> t/c freedom that L-BFGS sees as flat curvature. Everything
else matches the final_4x2 DeepONet it is meant to be compared with."""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

ARM, ARCH = "4x2", "deeponet"
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
    "TRUNK_NORM": False, "BRANCH_NORM": True, "BRANCH_DIM": 64,
    "N_PHYS_EXTRA": 4,
    "PHYS_SPAN_APEX": 0.5,
    "ADAM_EPOCHS": 15000,
    "LBFGS_STEPS": 10000,
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