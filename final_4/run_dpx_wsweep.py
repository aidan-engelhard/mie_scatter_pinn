"""final_4 cell DPX with the physics weights scaled by a single multiplier.

    python final_4/run_dpx_wsweep.py 1.0
    python final_4/run_dpx_wsweep.py 0.3
    python final_4/run_dpx_wsweep.py 0.1
    python final_4/run_dpx_wsweep.py 0.03

WHAT THIS IS LOOKING FOR. DPX has NO generalisation gap -- on final_4x2 its
seen and unseen errors are 0.0472 and 0.0462, a ratio of 0.98 -- so whatever
lowers its fit should lower its unseen error almost one for one. It is
fit-limited, and what limits the fit is the physics competing with the data at
the geometries that have a solve. Measured at final_4: the data-only model's
field residual is 0.70 against the PINN's 8.4e-6 and its flux residual 40
against 9.2e-4, so the two objectives disagree by four orders of magnitude, and
W_FIELD = W_FLUX = 50 against W_DATA = 1 decides where the model lands between
them.

That the ~0.04 level is a CONFLICT floor and not a capacity floor is the reason
to expect anything here: final_16, 36 and 64 -- all cell DP -- sat at
0.039-0.041 on both seen and unseen whether they had 16, 36 or 64 solves, while
the same 4x128 architecture fits data alone to 0.0023.

The sweep has two known endpoints. At m = 1 it is DPX, which scored unseen
0.0462 on the 4x2 arm. As m -> 0 every physics term vanishes and it becomes
cell D, which scored 0.0246. So the useful question is not whether small m
beats large m -- it must, since it tends to the better endpoint -- but whether
some INTERIOR m beats BOTH: enough physics left to hold the parameter
dependence flat, little enough that the fit is no longer dragged to 0.047.
If the error falls monotonically to the m -> 0 end, there is no interior
optimum and the honest answer is that the physics does not pay here.

BUDGET. Adam only, at half MATCHED_ADAM, so four multipliers cost about four
hours in total. That is a scan for the shape, not a result: the ranking at
end-of-Adam has tracked the final ranking in every run so far, but the absolute
numbers will not match a full run, and the scan is mildly unfair to large m
because those configurations converge more slowly. Take the promising value and
give it one full matched-budget run before believing a number.

JUDGE IT ON UNSEEN ERROR, NOT ON THE CONSOLE. The relL2 the monitor prints is
measured at the training geometries. The whole point is what happens between
them, so score the saved weights:

    python final/evaluate_arms.py --k 50 --arms 4

and compare the unseen means, and the seen/unseen ratio -- the value worth
taking forward is the largest weight reduction that still holds that ratio near
1.0.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

if len(sys.argv) < 2:
    raise SystemExit(__doc__.split("\n\n")[1].strip())
M = float(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
# 1.0 -> "w1", 0.03 -> "w0p03"
MTAG = "w" + f"{M:g}".replace(".", "p")

ARM, ARCH = 4, "mlp"
_, _, models, figs = DX._paths(ARM)
os.makedirs(models, exist_ok=True)
os.makedirs(figs, exist_ok=True)
npz = DX.npz_path(ARM)
if not os.path.exists(npz):
    raise SystemExit(f"{npz} not found -- run: python final_{ARM}/run.py --fdm")

# final_deeponet.py's defaults, scaled together. Kept explicit rather than read
# from the module because CONFIG_OVERRIDES is applied after they are used to
# derive other constants, so the launcher has to state the values it wants.
cfg = {
    "SINGLE_GEOMETRY": False,
    "N_PARAM_BATCH": min(DX.PARAM_BATCH, DX.n_samples(ARM)),
    "ACCUM_CHUNK": DX.ACCUM_CHUNK,
    "SEED": SEED,
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    "ADAM_EPOCHS": DX.MATCHED_ADAM // 2, "LBFGS_STEPS": 0,
    # cell DPX
    "PHYS_AT_DATA": True,
    "N_PHYS_EXTRA": 8, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 1.0,
    # the swept quantity: every physics term, together
    "W_PDE_INT": 1.0 * M, "W_PDE_EXT": 1.0 * M, "W_PDE_SUB": 1.0 * M,
    "W_FIELD": 50.0 * M, "W_FLUX": 50.0 * M,
    "W_PML": 0.5 * M, "W_DIRICHLET": 1.0 * M,
}

print(f"=== DPX weight sweep, multiplier {M:g}  ->  W_FIELD/W_FLUX "
      f"{50.0 * M:g}, W_PDE_* {M:g}, W_PML {0.5 * M:g}, W_DATA {DX.W_DATA:g}")

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": f"final_{ARM}_{ARCH}_dpx_{MTAG}"
                        + (f"_s{SEED}" if SEED else ""),
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": None,
    "CONFIG_OVERRIDES": cfg,
})
