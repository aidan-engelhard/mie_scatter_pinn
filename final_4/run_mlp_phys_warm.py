"""Continue final_4_mlp_phys for another 100000 L-BFGS evaluations.

The first run stopped at step 119597 with unseen relL2 0.0743 -- the best of
the arm, and the first model on it to beat bilinear interpolation of the four
EXACT FDM corner fields (0.1212). It got there on a HALVED L-BFGS budget: 100k
evaluations against the 250k the final_4_mlp control it beat by 37% was given,
and L-BFGS was still the phase doing the work (relL2 at the training
geometries 0.0698 -> 0.0411 over that phase, against Adam's 0.8670 -> 0.0711).
So there is no reason to think it had converged. This asks where it goes.

Two things to read from the result beyond the headline number:

  - whether the unseen error keeps falling with the anchors, or whether the two
    decouple. In the first run the _best_ checkpoint -- selected on the
    TRAINING-geometry relL2 -- was the WORST of the three saved models on
    unseen parameters (0.0757 against latest 0.0741). Anchor fit has stopped
    being a proxy for generalisation now that the physics constrains the
    parameter space, so score _latest_ as well as _best_.
  - the distance-to-corner profile. The draws' benefit grew with distance from
    the four training geometries (26% better than data-only in the far ring,
    worse than it in the nearest one). If more L-BFGS deepens that gradient,
    it is the draws improving; if it lifts the near ring instead, it is just
    the anchors.

Total L-BFGS after this run is ~240k, which finally makes the comparison
against final_4_mlp (269809) budget-matched.

CONTINUATION, not a new run: RUN_TAG is unchanged and INIT_WEIGHTS points at
that tag's own <tag>_weights.pt, which is what final_deeponet.py tests for. The
step count carries on from 119597, the relL2 CSV is appended to, and the
figures are redrawn over the whole training rather than a second set appearing.
The weights it starts from are copied to <tag>_pre_warm_weights.pt first, so a
continuation that goes wrong cannot destroy the 0.0743 model. The loss figure
stays per-run because a continuation measures its own L0 at a fresh
initialisation, so the normalised terms are not comparable across the restart.

Everything else is identical to final_4/run_mlp_phys.py -- same draws, same
spans, same weights -- so this only extends that run, it does not vary it.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

# INT, not "4": GRIDS keys the square arms by integer.
ARM, ARCH = 4, "mlp"
TAG = f"final_{ARM}_{ARCH}_phys"
_, _, models, figs = DX._paths(ARM)
npz = DX.npz_path(ARM)
init = os.path.join(models, f"{TAG}_weights.pt")
if not os.path.exists(init):
    raise SystemExit(f"{init} not found -- run final_{ARM}/run_mlp_phys.py first")

cfg = {
    "SINGLE_GEOMETRY": False,
    "N_PARAM_BATCH": min(DX.PARAM_BATCH, DX.n_samples(ARM)),
    "ACCUM_CHUNK": DX.ACCUM_CHUNK,
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    # no Adam: it would undo L-BFGS's work at a learning rate this model is
    # long past. The first run's Adam phase ended at 0.0711 and L-BFGS took it
    # to 0.0411.
    "ADAM_EPOCHS": 0, "LBFGS_STEPS": 100000,
    "N_PHYS_EXTRA": 8, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 1.0,
}

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": TAG,
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": init,
    "CONFIG_OVERRIDES": cfg,
})
