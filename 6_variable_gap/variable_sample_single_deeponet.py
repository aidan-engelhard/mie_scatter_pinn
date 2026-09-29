"""
VARIABLE SAMPLE — single-gap DeepONet, against the single-gap MLP control
===============================================================================
The second half of the bisection begun in variable_sample_single_mlp.py. That
control -- ONE gap, a plain 3x128 MLP, sample/mie_pinn_sample.py's weights and
budget -- reached relL2 ~ 0.058, falling monotonically through L-BFGS with no
climb. sample/mie_pinn_sample.py gets 0.045 on the same physics, so the
physics, losses, samplers and monitor in variable_sample_deeponet.py are sound
and the fault in the sweep runs lives in the parametric layer.

That layer has two parts, and this script separates them by changing exactly
one thing:

    this file          ONE gap, DeepONet          -> isolates the ARCHITECTURE
    the 8-gap runs     TEN gaps, DeepONet         -> adds the operator problem

CONFIG is IMPORTED from variable_sample_single_mlp.py rather than restated, so
the budget, weights, collocation counts, gap index and monitor cadence cannot
drift between the two runs. Only the architecture keys below are changed.

WHAT IS ACTUALLY BEING TESTED
-----------------------------
At a single gap the branch input is constant, so branch(features) is a fixed
vector b and the network collapses to u = sum_i b_i * trunk_i(x, y) -- a fixed
linear combination of trunk outputs. That is expressively equivalent to the MLP
control's 2-wide head, so this run does NOT test capacity. What survives the
collapse is exactly the two mechanisms the DeepONet adds:

  scaling degeneracy   b -> c*b, t -> t/c leaves u unchanged, so the loss has
                       flat directions in precisely the curvature L-BFGS is
                       estimating -- and L-BFGS does most of the work in these
                       runs. variable_sample_mlp.py's docstring names this.
  cancelling sum       u is a 128-term dot product, so grad^2 u is a sum of 128
                       second derivatives, evaluated against k_int^2 ~ 1.4e5 in
                       float32. The MLP's 2-wide head has no such sum.

TRUNK_HIDDEN is set to 3 to match the MLP control's depth. The default is 4,
which would make depth a second difference and blur the read.

Note the DeepONet carries MORE parameters than the MLP control at this shape
(the startup banner prints both counts). That asymmetry only sharpens a
negative result: if the larger network does worse, capacity is not the
explanation.

HOW TO READ THE RESULT
----------------------
    reaches ~0.06          the architecture is fine at one geometry, and the
                           8-gap failure is the operator problem itself --
                           the moving scatterer and the fixed trunk basis, i.e.
                           the n-width limitation the main file's header
                           already flags. The fix it suggests is
                           cylinder-local trunk coordinates.

    stalls high, or        the DeepONet head is the problem, and the two
    bottoms out then       mechanisms above are where to look. Cheapest next
    climbs                 probes: normalise the trunk outputs (kills the
                           scaling degeneracy), or shrink BRANCH_DIM (shortens
                           the cancelling sum).

As before, watch the SHAPE. The MLP control fell monotonically; a curve that
turns upward is the failure being chased and shows up long before the budget is
spent, so there is no need to let the run finish to learn that.

Prerequisite, as for the others:
    python variable_sample/variable_sample_fdm.py

OUTPUTS -- their own directory, so neither the sweep figures nor the MLP
control's are touched:
    variable_sample/figures_single_deeponet/
        variable_sample_single_deeponet_gap2_*.png, ..._training.mp4
    variable_sample/models/variable_sample_single_deeponet_gap2_weights.pt
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "variable_sample_deeponet.py")

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Importing is safe: the MLP launcher's runpy call is guarded by
# `if __name__ == "__main__"`, so this picks up CONFIG without running it.
from variable_sample_single_mlp import CONFIG as _MLP_CONFIG
from variable_sample_single_mlp import SHORT_BUDGET

CONFIG = dict(_MLP_CONFIG)
CONFIG.update({
    # Trunk at the MLP control's depth and width, so the head is the only
    # difference. TRUNK_HIDDEN defaults to 4.
    "TRUNK_HIDDEN":   3,
    "TRUNK_NEURONS":  128,
    # Branch and dot-product width: the file's defaults, restated so this
    # control keeps meaning its own thing as the defaults are edited.
    "BRANCH_HIDDEN":  3,
    "BRANCH_NEURONS": 64,
    "BRANCH_DIM":     256,
})

# ── --short: the sweep arm's short budget, at ONE geometry ──────────────────
# The point of this mode is a like-for-like loss trajectory against
# variable_sample_sweep_deeponet.py --short: identical ADAM_EPOCHS,
# LBFGS_STEPS and EVAL_EVERY, one geometry instead of eight. At ~1/8 the
# per-step cost it runs in roughly ten minutes, and comparing the two
# <RUN_TAG>_relL2.csv files at matched steps separates the two remaining
# explanations for the sweep's failure:
#
#   the sweep's loss tracks this one, just higher  -> budget. The sweep is on
#       the same descent and needs the full run. (The single-gap DeepONet took
#       ~200000 evals to reach relL2 0.075, and was still at ~0.62 at the
#       35000 mark where the sweep's short run stopped.)
#   this one keeps descending while the sweep flattens -> capacity. One basis
#       cannot serve eight geometries, and an N_PARAM_BATCH = 2 or 3 scaling
#       study locates where it breaks.
#
# Worth knowing before either conclusion: the objective itself is sound. At gap
# 2, on identical points and L0, the single-gap checkpoint scores relL2 0.0774
# at normalised loss 2.046e-03 and the sweep checkpoint relL2 1.1782 at
# 4.404e-02 -- the better field has the lower loss, by 21x. The sweep is not
# being misled by the loss; it is 21x short of minimising it.
SHORT = "--short" in sys.argv[1:]
RUN_TAG = "variable_sample_single_deeponet"
FIG_DIR = "variable_sample/figures_single_deeponet"
if SHORT:
    CONFIG.update(SHORT_BUDGET)
    RUN_TAG = "variable_sample_single_deeponet_short"
    FIG_DIR = "variable_sample/figures_single_deeponet_short"

_unknown = [a for a in sys.argv[1:] if a != "--short"]
if _unknown:
    raise SystemExit(f"unknown argument(s): {' '.join(_unknown)}"
                     "  (the only option is --short)")

if __name__ == "__main__":
    if not os.path.exists(_TARGET):
        raise SystemExit("variable_sample_deeponet.py not found next to this "
                         f"script: {_TARGET}")
    print(f"  mode: {'--short diagnostic' if SHORT else 'full'}   "
          f"ADAM_EPOCHS={CONFIG['ADAM_EPOCHS']}  "
          f"LBFGS_STEPS={CONFIG['LBFGS_STEPS']}  ->  {FIG_DIR}")
    runpy.run_path(
        _TARGET,
        init_globals={
            "ARCH_OVERRIDE":      "deeponet",
            "RUN_TAG_OVERRIDE":   RUN_TAG,
            "FIG_DIR_OVERRIDE":   FIG_DIR,
            "MODEL_DIR_OVERRIDE": "variable_sample/models",
            "CONFIG_OVERRIDES":   CONFIG,
        },
        run_name="__main__",
    )
