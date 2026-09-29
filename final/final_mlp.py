"""
FINAL — plain-MLP control for final_deeponet.py
===============================================================================
Runs the SAME script with the SAME physics, losses, samplers, budgets,
optimiser, monitor and figures, and swaps only the network:

    deeponet :  u = sum_i branch_i(params) * trunk_i(x, y)
    mlp      :  u = MLP([fourier(x, y), params]) -> (Re, Im)

Usage
-----
    python final/final_mlp.py

Same prerequisites as final_deeponet.py: run final/final_fdm.py first so
final/inputs/fdm_final_sweep.npz exists for the geometry being trained.

WHY THIS IS A LAUNCHER RATHER THAN A COPY
-----------------------------------------
The question this answers is "does the architecture explain the gap", which is
only meaningful if literally everything else is identical. A forked 2400-line
copy would have drifted on the first edit to either file -- and both are being
actively changed. So the architecture lives behind ARCH in final_deeponet.py and
this script sets it, which makes "apart from architecture, the same" a property
of the code rather than a claim in a comment.

Everything is configured in final_deeponet.py as usual (SINGLE_GEOMETRY,
SKIN_ANSATZ, the budgets, the weights); this file overrides only the four names
below. To change the MLP's own shape, edit MLP_HIDDEN / MLP_NEURONS there.

WHAT THE COMPARISON IS FOR
--------------------------
The single-geometry DeepONet runs sit at relL2 0.33-0.46 where afm_pinn/, a
plain MLP on the same physics, reaches 0.081. Two explanations survive:

  1. the DeepONet form itself. u = sum_i b_i t_i has a scaling degeneracy
     (b -> a*b, t -> t/a leaves u unchanged), which adds flat directions to
     exactly the curvature L-BFGS is estimating -- and L-BFGS is what does the
     work in these runs. It is also a rank-BRANCH_DIM/2 factorisation, where the
     MLP has no such constraint.
  2. something non-architectural in this harness that differs from afm_pinn's --
     N_PML is one shared sample partitioned by region here against 1500 drawn
     per region there, and N_DIR_SIDE reproduces its 750 outer points only
     approximately.

This run separates them. If the MLP recovers ~0.08-0.13, (1) is the answer and
the parametric work should go into the architecture -- TRUNK_SEES_SDF, a
reference-domain map, or dropping the factorisation. If it lands with the
DeepONet at ~0.33, the harness is the problem and the architecture comparison is
not yet valid.

Note the default MLP is the DeepONet's TRUNK with a 2-wide head instead of a
256-wide one dotted with a branch, so it carries fewer parameters (0.57x at the current width). The
startup banner prints both counts, and the width that would give parity -- set
MLP_NEURONS to that value in final_deeponet.py for a matched comparison.

OUTPUTS
-------
Written beside the DeepONet's, tagged so nothing is overwritten:

    final/figures/final_mlp_relL2.png,  _loss.png,  _field_comparison.png, ...
    final/models/final_mlp_weights.pt

The checkpoint records ARCH, so evaluate_final.py rebuilds the right network.
"""

import os
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "final_deeponet.py")

if not os.path.exists(_TARGET):
    raise SystemExit(f"final_deeponet.py not found next to this script: {_TARGET}")

# run_name="__main__" so the target behaves exactly as when run directly.
# init_globals injects these BEFORE the module body executes, which is what lets
# final_deeponet.py pick them up via globals().get(...) at definition time.
runpy.run_path(
    _TARGET,
    init_globals={
        "ARCH_OVERRIDE":      "mlp",
        "RUN_TAG_OVERRIDE":   "final_mlp",
        "FIG_DIR_OVERRIDE":   "final/figures",
        "MODEL_DIR_OVERRIDE": "final/models",
    },
    run_name="__main__",
)
