"""final_4x2 cell DPX with the physics weights scaled by a single multiplier.

    python final_4x2/run_dpx_wsweep.py 0.1
    python final_4x2/run_dpx_wsweep.py 0.03
    python final_4x2/run_dpx_wsweep.py 0.01
    python final_4x2/run_dpx_wsweep.py 0.003
    python final_4x2/run_dpx_wsweep.py 0

Same scan as final_4/run_dpx_wsweep.py, on the arm where the conclusion
currently goes the other way. Cell D wins here (unseen 0.0246 against DPX's
0.0462) and loses on final_4 (0.0970 against a swept DPX's 0.0783), and the
explanation on offer is a regime difference: final_4 has four solves at TWO
half-angles and is interpolation-limited, where physics buys something; 4x2 has
four half-angles and is not. This run decides that. If a small multiplier beats
D here too, the regime story is wrong and the honest reading is simply that the
weights were never tuned.

WHY THESE VALUES. On final_4 the multiplier ran 1 / 0.3 / 0.1 / 0.03 and the
error fell all the way down -- but that scan was confounded, because at 10000
Adam epochs the large-m runs were nowhere near converged (m = 1 ended with a
training-geometry relL2 of 0.1506 against m = 0.03's 0.0177). So m = 1 and
m = 0.3 are not worth repeating at this budget; they measure convergence rate,
not the weight. What the final_4 scan did establish is that the minimum lies
BELOW 0.03, since m -> 0 is cell D at 0.0970 and m = 0.03 reached 0.0783 --
bracketed above, open below. Hence the low end here.

m = 0 IS THE POINT OF THE THING. Every physics weight becomes zero, so the run
is cell D at the sweep's own budget. That anchors the low end INTERNALLY: on
final_4 the m = 0.03 result had to be compared against a cell D trained at
15000 Adam plus a converged L-BFGS phase, a different budget, so the interior
minimum was inferred across runs rather than demonstrated within one curve.
Here the whole curve is one controlled experiment. Do not skip it.

Adam only at half MATCHED_ADAM, about 1.5 h per multiplier on this arm.

JUDGE ON UNSEEN ERROR. The console monitor reports the training geometries,
which is the one thing this is not about:

    python final/evaluate_arms.py --k 50 --arms 4x2

Read the unseen mean and the seen/unseen ratio together. On final_4 the ratio
moved 1.1x -> 1.5x -> 2.2x -> 4.4x as m fell, with the best unseen error at
4.4x: some parametric flattening kept, most of the fit recovered. The value to
carry forward is the largest weight reduction that still holds the ratio down,
and it needs one full matched-budget run before it means anything.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

if len(sys.argv) < 2:
    raise SystemExit("usage: run_dpx_wsweep.py <multiplier> [seed]")
M = float(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
# 1.0 -> "w1", 0.03 -> "w0p03", 0 -> "w0"
MTAG = "w" + f"{M:g}".replace(".", "p")

ARM, ARCH = "4x2", "mlp"
_, _, models, figs = DX._paths(ARM)
os.makedirs(models, exist_ok=True)
os.makedirs(figs, exist_ok=True)
npz = DX.npz_path(ARM)
if not os.path.exists(npz):
    raise SystemExit(f"{npz} not found -- run: python final_{ARM}/run.py --fdm")

# final_deeponet.py's defaults, scaled together. Stated explicitly rather than
# read from the module because CONFIG_OVERRIDES is applied after they are used
# to derive other constants.
cfg = {
    "SINGLE_GEOMETRY": False,
    "N_PARAM_BATCH": min(DX.PARAM_BATCH, DX.n_samples(ARM)),
    "ACCUM_CHUNK": DX.ACCUM_CHUNK,
    "SEED": SEED,
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    "ADAM_EPOCHS": DX.MATCHED_ADAM // 2, "LBFGS_STEPS": 0,
    # cell DPX as run_dpx.py defines it for THIS arm: 4 draws, and the gap axis
    # narrowed to its middle half, which is where 4x2's error ridge sits
    "PHYS_AT_DATA": True,
    "N_PHYS_EXTRA": 4, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 0.5,
    # the swept quantity: every physics term, together
    "W_PDE_INT": 1.0 * M, "W_PDE_EXT": 1.0 * M, "W_PDE_SUB": 1.0 * M,
    "W_FIELD": 50.0 * M, "W_FLUX": 50.0 * M,
    "W_PML": 0.5 * M, "W_DIRICHLET": 1.0 * M,
}

print(f"=== DPX weight sweep on {ARM}, multiplier {M:g}  ->  W_FIELD/W_FLUX "
      f"{50.0 * M:g}, W_PDE_* {M:g}, W_PML {0.5 * M:g}, W_DATA {DX.W_DATA:g}"
      + ("   (m = 0: this is cell D at the sweep's budget)" if M == 0 else ""))

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
