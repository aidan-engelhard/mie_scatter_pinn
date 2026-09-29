"""
VARIABLE SAMPLE — plain-MLP control for variable_sample_deeponet.py
===============================================================================
Runs the SAME script with the SAME physics, losses, samplers, budgets,
optimiser, monitor and figures, and swaps only the network:

    deeponet :  u = sum_i branch_i(features(gap)) * trunk_i(x, y)
    mlp      :  u = MLP([fourier(x, y), features(gap)]) -> (Re, Im)

Usage
-----
    python variable_sample/variable_sample_mlp.py

Prerequisite, same as the DeepONet: run variable_sample/variable_sample_fdm.py
first so variable_sample/inputs/fdm_variable_sample_sweep.npz exists.

WHY THIS IS A LAUNCHER RATHER THAN A COPY
-----------------------------------------
The question it answers is "does the architecture explain the difference",
which is only meaningful if literally everything else is identical. A forked
copy would drift on the first edit to either file. So the architecture lives
behind ARCH in variable_sample_deeponet.py and this script sets it, which makes
"apart from architecture, the same" a property of the code rather than a claim
in a comment.

Everything else is configured in variable_sample_deeponet.py as usual (the gap
sweep, the budgets, N_PARAM_BATCH, the weights); this file overrides only the
four names below. To change the MLP's own shape, edit MLP_HIDDEN / MLP_NEURONS
there.

WHAT THE COMPARISON IS FOR
--------------------------
Both networks take the same gap features as input, so both are parametric
surrogates over the sweep; they differ in how the parameter enters.

  deeponet  The parameter picks coefficients that multiply a FIXED spatial
            basis, so the field is constrained to a rank-BRANCH_DIM/2
            factorisation. It also carries a scaling degeneracy
            (b -> c*b, t -> t/c leaves u unchanged), which adds flat directions
            to exactly the curvature L-BFGS is estimating -- and L-BFGS does
            most of the work in these runs.
  mlp       The parameter is an ordinary input. No factorisation, no
            degeneracy, but also no separation between "where" and "which
            geometry", so nothing stops it from fitting each gap independently
            and interpolating badly between them.

The number that distinguishes them is the TEST relL2 -- the two held-out gaps --
against the train relL2, both plotted on <RUN_TAG>_relL2.png and broken out gap
by gap on <RUN_TAG>_error_vs_gap.png. A model that fits the eight training gaps
and misses the two held-out ones has learnt eight solutions rather than an
operator, and the gap between the two curves is what says so.

Note the default MLP is the DeepONet's TRUNK with a 2-wide head instead of a
BRANCH_DIM-wide one dotted with a branch, so it carries fewer parameters. The
startup banner prints both counts and the width that would give parity; set
MLP_NEURONS to that value in variable_sample_deeponet.py for a matched
comparison.

OUTPUTS
-------
Written beside the DeepONet's, tagged so nothing is overwritten:

    variable_sample/figures/variable_sample_mlp_relL2.png,  _loss.png,
                            _error_vs_gap.png, _sweep_grid.png,
                            _field_comparison.png, _training.mp4
    variable_sample/models/variable_sample_mlp_weights.pt

The checkpoint records ARCH, so any consumer rebuilds the right network.
"""

import os
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "variable_sample_deeponet.py")

if not os.path.exists(_TARGET):
    raise SystemExit(
        f"variable_sample_deeponet.py not found next to this script: {_TARGET}")

# run_name="__main__" so the target behaves exactly as when run directly.
# init_globals injects these BEFORE the module body executes, which is what lets
# variable_sample_deeponet.py pick them up via globals().get(...) at definition
# time.
runpy.run_path(
    _TARGET,
    init_globals={
        "ARCH_OVERRIDE":      "mlp",
        "RUN_TAG_OVERRIDE":   "variable_sample_mlp",
        "FIG_DIR_OVERRIDE":   "variable_sample/figures",
        "MODEL_DIR_OVERRIDE": "variable_sample/models",
    },
    run_name="__main__",
)
