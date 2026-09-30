"""final_4x2 cell DPX at a chosen physics-weight multiplier, full matched budget.

    python final_4x2/run_dpx_full.py 0.003
    python final_4x2/run_dpx_full.py 0.003 1        a second seed

The follow-up run_dpx_wsweep.py asks for: that scan was Adam only at half
MATCHED_ADAM, a scan for the shape rather than a result, and its best value has
to be given one full matched-budget run before a number is believed. On 4x2 the
scan put the unseen minimum between m = 0.003 (0.0204) and m = 0.01 (0.0214),
against cell D's 0.0246 and m = 0's 0.0241 at the scan's own budget; 0.03 was
already worse than D (0.0285).

Identical to run_dpx_wsweep.py except for the budget -- MATCHED_ADAM plus
MATCHED_LBFGS, the budget run_dpx.py gave m = 1 -- so this run sits in the same
comparison as cell DPX (unseen 0.0462) and the scan point it repeats.

The tag gets a _full suffix, final_4x2_mlp_dpx_w0p003_full, so the scan's
checkpoint for the same multiplier is not overwritten. Score it in range and
outside it:

    python final/evaluate_arms.py --k 50 --arms 4x2
    python final/evaluate_extrap.py --arms 4x2
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

if len(sys.argv) < 2:
    raise SystemExit("usage: run_dpx_full.py <multiplier> [seed]")
M = float(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MTAG = "w" + f"{M:g}".replace(".", "p")

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
    "SEED": SEED,
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    "ADAM_EPOCHS": DX.MATCHED_ADAM, "LBFGS_STEPS": DX.MATCHED_LBFGS,
    # cell DPX as run_dpx.py defines it for THIS arm
    "PHYS_AT_DATA": True,
    "N_PHYS_EXTRA": 4, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 0.5,
    # every physics term scaled together, as in run_dpx_wsweep.py
    "W_PDE_INT": 1.0 * M, "W_PDE_EXT": 1.0 * M, "W_PDE_SUB": 1.0 * M,
    "W_FIELD": 50.0 * M, "W_FLUX": 50.0 * M,
    "W_PML": 0.5 * M, "W_DIRICHLET": 1.0 * M,
}

print(f"=== DPX on {ARM}, multiplier {M:g}, FULL budget (Adam "
      f"{DX.MATCHED_ADAM}, L-BFGS {DX.MATCHED_LBFGS})  ->  W_FIELD/W_FLUX "
      f"{50.0 * M:g}, W_PDE_* {M:g}, W_PML {0.5 * M:g}, W_DATA {DX.W_DATA:g}")

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": f"final_{ARM}_{ARCH}_dpx_{MTAG}_full"
                        + (f"_s{SEED}" if SEED else ""),
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": None,
    "CONFIG_OVERRIDES": cfg,
})
