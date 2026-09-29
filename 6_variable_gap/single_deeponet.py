"""
VARIABLE SAMPLE — one network over the whole domain, two heads
===============================================================================
Runs the SAME script with the SAME physics, losses, samplers, data terms,
budgets, optimiser, monitor and figures, and changes only what computes the two
fields:

    deeponet :  net_int and net_scat are two INDEPENDENT DeepONets,
                coupled only by the interface conditions
    single   :  ONE shared body over the whole domain, read out through two
                linear heads -- one for u_int, one for u_scat

Usage
-----
    python variable_sample/single_deeponet.py

Same prerequisite as the others: run variable_sample/variable_sample_fdm.py
first so variable_sample/inputs/fdm_variable_sample_sweep.npz exists.

WHY TRY THIS
------------
With two independent networks the only thing coupling them is the pair of
interface conditions on a curve that MOVES with the gap. Every gradient that
carries geometry information therefore has to cross between two separate
parameter sets through the 800 interface points -- 8.6% of the collocation
sample -- while the PDE, PML and Dirichlet terms are very nearly gap-blind (a
field satisfying the Helmholtz equation in air satisfies it in any subregion).
Sharing the body means the representation that knows where the cylinder is, is
the same one both fields are read from, and the interface gradients update
shared weights instead of having to agree across two networks.

WHY STILL TWO HEADS
-------------------
u_int is the TOTAL field inside a strong absorber, decaying over a skin depth
of lam/188; u_scat is the scattered field outside, oscillating at |k_sub| =
1.5k. They are different objects on scales a factor of 20 apart, so one readout
would be asking a single basis to be good at both. Two linear heads on a shared
body cost almost nothing and keep the readouts independent.

For the same reason the shared embedding carries BOTH Fourier bands -- one
scaled by |k_sub|, one by |k_int| -- concatenated, so each head can select the
frequencies it needs. That is the one place where the single architecture is
not simply "the same network twice".

WHAT IS UNCHANGED
-----------------
Everything else, deliberately. The physics was verified term by term against
sample/mie_pinn_sample.py and agrees to float32 round-off; the loss constants
are that script's; the data terms, the 16/4 split, INIT_OUT_RMS and the budgets
are exactly as the two-network runs used them. So a difference in the result is
attributable to the architecture and not to anything else -- which is only true
because this is a launcher rather than a fork.

WHAT TO LOOK FOR
----------------
The number that matters is the TEST relL2 on the held-out gaps, against the
two-network run's. Two secondary readings worth watching in the eval lines:

  rms|u_scat|(air) against its target -- the collapse detector. Physics-only
  runs walked to the "invisible cylinder", u_scat at 7% of amplitude with u_int
  30x too large, and the shared body does nothing by itself to prevent that.

  the train/test gap on <RUN_TAG>_relL2.png. Sharing the body raises capacity
  per field, so it could equally fit the 16 training gaps better and generalise
  no better.

OUTPUTS
-------
Written beside the others, tagged so nothing is overwritten:

    variable_sample/figures/single_deeponet_relL2.png,  _loss.png,
                            _error_vs_gap.png, _sweep_grid.png,
                            _field_comparison.png, _training.mp4
    variable_sample/models/single_deeponet_weights.pt

The checkpoint records ARCH, so evaluate_deeponet.py rebuilds the right network:

    python variable_sample/evaluate_deeponet.py --arch single
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
        "ARCH_OVERRIDE":      "single",
        "RUN_TAG_OVERRIDE":   "single_deeponet",
        "FIG_DIR_OVERRIDE":   "variable_sample/figures",
        "MODEL_DIR_OVERRIDE": "variable_sample/models",
    },
    run_name="__main__",
)
