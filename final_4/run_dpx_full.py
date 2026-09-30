"""final_4 cell DPX at a chosen physics-weight multiplier, full matched budget.

    python final_4/run_dpx_full.py 0.03
    python final_4/run_dpx_full.py 0.03 1          a second seed

The follow-up run_dpx_wsweep.py asks for. That scan was Adam only at half
MATCHED_ADAM and ran m = 1 / 0.3 / 0.1 / 0.03, with the unseen error falling
all the way down to 0.0783 at m = 0.03 -- so its best value is the LOWEST one
tried, and the minimum is bracketed above but open below. The same scan on
final_4x2 found its minimum at 0.003-0.01.

The scan is also confounded by budget on this arm, in the direction that
matters here: run_dpx.py's m = 1 at the full matched budget scored unseen
0.0759, better than every scan point, while the scan's own m = 1 scored 0.166.
Half Adam leaves the large-m runs unconverged. So whether a small m beats m = 1
here at all is exactly what this run measures.

Identical to run_dpx_wsweep.py except for the budget -- MATCHED_ADAM plus
MATCHED_LBFGS, the budget run_dpx.py gave m = 1. The tag gets a _full suffix,
final_4_mlp_dpx_w0p03_full, so the scan's checkpoint for the same multiplier is
not overwritten. Score it in range and outside it:

    python final/evaluate_arms.py --k 50 --arms 4
    python final/evaluate_extrap.py --arms 4
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

if len(sys.argv) < 2:
    raise SystemExit("usage: run_dpx_full.py <multiplier> [seed] [arch]")
M = float(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ARCH_ARG = sys.argv[3] if len(sys.argv) > 3 else "mlp"
if ARCH_ARG not in ("mlp", "deeponet"):
    raise SystemExit(f"arch must be mlp or deeponet, not {ARCH_ARG!r}")
MTAG = "w" + f"{M:g}".replace(".", "p")

# The architecture comparison belongs in THIS configuration, not in the one
# with physics only at the solved geometries: that configuration is worse for
# both architectures, so a margin measured there says little about which
# architecture suits the problem.
#
# A DeepONet here is the narrowed variant -- latent dimension 64 with the
# branch output normalised -- and not the 256-wide default, because that is the
# configuration developed for this problem and the better of the two by a wide
# margin (0.0482 against 0.0757 on the 4x2 arm). Comparing a tuned MLP against
# an untuned DeepONet would repeat, in reverse, the unfairness this run exists
# to remove.
ARCH_EXTRA = {"mlp": {},
              "deeponet": {"BRANCH_DIM": 64, "BRANCH_NORM": True,
                           "TRUNK_NORM": False}}[ARCH_ARG]

ARM, ARCH = 4, ARCH_ARG
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
    "N_PHYS_EXTRA": 8, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 1.0,
    # every physics term scaled together, as in run_dpx_wsweep.py
    "W_PDE_INT": 1.0 * M, "W_PDE_EXT": 1.0 * M, "W_PDE_SUB": 1.0 * M,
    "W_FIELD": 50.0 * M, "W_FLUX": 50.0 * M,
    "W_PML": 0.5 * M, "W_DIRICHLET": 1.0 * M,
    **ARCH_EXTRA,
}

print(f"=== DPX {ARCH} on {ARM}, multiplier {M:g}, FULL budget (Adam "
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
