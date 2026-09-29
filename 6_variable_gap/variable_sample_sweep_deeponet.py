"""
VARIABLE SAMPLE — the full gap sweep, at the budget that solved one geometry
===============================================================================
The third arm of the bisection. CONFIG chains down from the two single-gap
controls, so each arm adds exactly one thing:

    variable_sample_single_mlp.py        ONE gap, plain MLP    relL2 0.058
    variable_sample_single_deeponet.py   + DeepONet head       worked
    THIS FILE                            + all 8 training gaps   ?

Everything else -- budget, weights, collocation counts, monitor cadence,
trunk/branch shape -- is imported, not restated, so the only difference between
this run and the single-gap DeepONet that worked is the number of geometries.
If it fails, that is the one thing to explain.

WHY THE EARLIER SWEEP RUNS ARE NOT EVIDENCE ABOUT THIS ONE
----------------------------------------------------------
variable_sample/figures/ (commit b3b53d9) is the pure-physics sweep whose relL2
climbed from 0.88 to ~1.25 during L-BFGS. It had W_FIELD = W_FLUX = 20.0 and
W_PML = 2.0, i.e. a forcing:homogeneous weight ratio of 5:1 against
sample/mie_pinn_sample.py's 20:1 -- and field and flux are the ONLY
inhomogeneous terms, the only thing that imposes the incident wave at all. That
was fixed in 2384dbc. It also ran 50000 closure evals, ~2000 outer steps; at
that point in the single-gap MLP run relL2 was still ~0.4, seven times worse
than where it finished. So that curve is a run with the wrong objective,
stopped a fifth of the way in. It says little about this configuration.

WHAT IS GENUINELY UNTESTED HERE
-------------------------------
Both single-gap arms leave the branch inert: with a constant input,
branch(features) is a fixed vector and the network collapses to a fixed linear
combination of trunk outputs. Nothing about mapping gap -> coefficients has
been exercised yet, and that is the actual operator-learning machinery. This
run is the first test of it.

Two things also change that are not bugs but do make each step harder:
backward_batch returns the MEAN over geometries, so one update is a single
direction that must improve all eight at once through a shared basis -- where
they disagree, the components cancel; and the substrate, PML and outer-edge
samples are the same coordinates tiled across geometries, differing only
through the features, so there is less sampling diversity than per-geometry
draws would give. The multi-geometry code path itself is not a worry: it agrees
with the single-geometry path to 1.5e-07 on every loss term and 6.1e-07 on the
full parameter gradient, i.e. float32 round-off.

ADAM, AND WHY IT IS 20000 HERE
------------------------------
The first attempt at this run inherited the single-gap ADAM_EPOCHS of 3000 and
was stopped after 15517 L-BFGS evals. Its relL2 log
(variable_sample/figures_sweep_deeponet/) says two things.

Adam was still improving when it was cut off: over the 3000 epochs relL2(test)
went 0.981 -> 0.849, rms|u_scat| in air went 0.14x -> 0.79x of the FDM target,
and the implied correlation with the FDM field went 0.24 -> 0.45. All three
were still moving. 3000 epochs was the budget that covered ONE geometry.

L-BFGS then converged, fast, onto a fixed point that is wrong: within ~100
outer steps the amplitude overshot to 1.27x the target and crept to 1.34x,
while the correlation DROPPED to 0.34 and froze there -- 0.344, 0.339, 0.344,
0.343, 0.345 across 13000 evals. That is a plateau, not a divergence, so the
remaining 185000 evals would not have moved it. A field 34% too energetic with
a third of it correlated is what you get when the only thing pinning the
amplitude -- the field and flux pair on a circle of radius 0.05, ka = 0.63 --
is satisfied loosely: recovering an exterior field from Cauchy data on a
subwavelength circle is ill-conditioned, so a small boundary residual buys a
large far-field error, and L-BFGS drives residuals much harder than Adam does.
At one geometry the network got those residuals low enough; with one basis
serving eight, apparently not. That mechanism is consistent with the numbers
rather than demonstrated by them.

Raising Adam to 20000 is the one change the data directly justifies, and it is
cheap: at the 114 ms/epoch measured in that run it costs ~32 extra minutes
against an L-BFGS phase of several hours.

BUDGET
------
250000 closure evals is ~10000 outer L-BFGS steps at the measured ~24.6 evals
per step -- the same NUMBER OF OPTIMISER STEPS that solved one geometry, which
is the right equivalence. The cost per step is ~8x the single-gap run's,
because every step now evaluates eight geometries' residuals. Budget the
wall-clock accordingly: this is roughly 5x the old 50000-eval sweep.

If the relL2 curve is still descending at the end, extend the budget rather
than concluding anything -- the single-gap MLP only flattened around 150000.
N_PARAM_BATCH is the cost dial: below 8 it draws a random subset of the
training gaps each redraw, halving per-step cost at 4 but showing each gap half
as often.

    python variable_sample/variable_sample_sweep_deeponet.py            full
    python variable_sample/variable_sample_sweep_deeponet.py --short    diagnostic

--short keeps the configuration identical and cuts only the L-BFGS budget, to
15000 evals (~30 minutes at the 119 ms/eval measured), and tightens the monitor
to every 1000 evals so the Adam -> L-BFGS handover is resolved finely -- that is
where the last run went wrong, between its final Adam reading and its first
L-BFGS one 2513 evals later, with nothing in between.

15000 is enough because the failure it is looking for arrives early and then
stops moving: in the previous attempt the overshoot was already at 1.27x by the
first L-BFGS reading (2513 evals in) and the correlation was frozen at ~0.34
from there on. Anything that behaves like that will have shown itself well
inside this budget. It therefore answers "does the overshoot still happen with
more Adam, and which term pays for it" for about a seventh of the cost of the
full run -- ~1.1 h against ~7.3 h -- which is worth doing before committing to
the full one. It writes to its own tag and directory, so a short run and a full
one never overwrite each other.

HOW TO READ IT
--------------
Watch the TRAIN and TEST curves together on the relL2 plot, which is the whole
point of the sweep:

    both fall together       the operator is being learnt -- the thing the
                             directory exists to demonstrate.
    train falls, test        eight solutions memorised rather than an operator.
    stays high               The n-width limitation in the main file's header
                             is the first suspect, and its suggested fix is
                             cylinder-local trunk coordinates.
    both climb during        the failure mode from the earlier sweeps has
    L-BFGS                   survived the weight and budget fixes, and the
                             branch is then the only untested thing left.

The shape shows up long before the budget is spent, so there is no need to let
it finish to learn which of those is happening.

Prerequisite, as for the others:
    python variable_sample/variable_sample_fdm.py

OUTPUTS -- their own directory, leaving the earlier sweep figures in
variable_sample/figures/ intact for comparison:
    variable_sample/figures_sweep_deeponet/
        variable_sample_sweep_deeponet_*.png, ..._training.mp4
    variable_sample/models/variable_sample_sweep_deeponet_weights.pt
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "variable_sample_deeponet.py")

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Safe to import: that launcher's runpy call is behind `if __name__ ==
# "__main__"`, and it in turn imports CONFIG from variable_sample_single_mlp.py.
# So budget, weights, collocation counts and architecture all arrive from the
# runs that have already been validated, and cannot drift from them.
from variable_sample_single_deeponet import CONFIG as _SINGLE_CONFIG
from variable_sample_single_mlp import SHORT_BUDGET

CONFIG = dict(_SINGLE_CONFIG)
CONFIG.update({
    # The one change: the full training set instead of one gap.
    "SINGLE_GAP_INDEX": None,
    # All eight training gaps in every step. The main file clamps this to
    # len(train_idx), so 8 means "all of them" however the FDM split came out.
    "N_PARAM_BATCH": 8,
    # Geometries per backward pass. 8 is one graph for the whole batch --
    # fewest kernel launches, highest peak memory. Drop to 4 or 2 if the card
    # complains; the result is identical either way (verified: the chunked and
    # whole-batch paths agree to float32 round-off).
    "ACCUM_CHUNK": 8,
    # Eight geometries through one basis need more of the cheap phase than one
    # did. See ADAM, AND WHY IT IS 20000 HERE above -- the previous attempt's
    # Adam phase was still improving on all three monitored quantities when it
    # ended. This is the one place this launcher deliberately departs from the
    # single-gap arm it inherits from.
    "ADAM_EPOCHS": 20000,
})

# ── --short: same configuration, less L-BFGS ────────────────────────────────
SHORT = "--short" in sys.argv[1:]
RUN_TAG = "variable_sample_sweep_deeponet"
FIG_DIR = "variable_sample/figures_sweep_deeponet"
if SHORT:
    # SHARED with variable_sample_single_deeponet.py --short, so the two are
    # comparable step for step. See SHORT_BUDGET in variable_sample_single_mlp.py.
    CONFIG.update(SHORT_BUDGET)
    RUN_TAG = "variable_sample_sweep_deeponet_short"
    FIG_DIR = "variable_sample/figures_sweep_deeponet_short"

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
