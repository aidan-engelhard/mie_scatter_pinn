"""
VARIABLE SAMPLE — single-gap, plain-MLP control against sample/mie_pinn_sample.py
===============================================================================
Runs variable_sample_deeponet.py stripped down to the configuration that
sample/mie_pinn_sample.py solves: ONE geometry, a plain MLP, no gap features
doing any work, and the same optimiser budget. It is a diagnostic, not a
sweep -- it answers one question and then gets deleted or kept as a baseline.

WHY THIS EXISTS
---------------
sample/mie_pinn_sample.py reaches relL2 ~ 0.045 on this exact physics,
physics-only. variable_sample/single_gap/ is this script on ONE gap and it
plateaus at relL2 ~ 0.69 before climbing to 0.82. Since that run has no
parametric content at all -- one fixed cylinder, no held-out gaps, no operator
to learn -- the sweep, the DeepONet, the branch/trunk factorisation and the
moving-scatterer n-width problem are all ruled out as the cause. Something in
THIS FILE differs from mie_pinn_sample.py at a single geometry.

The physics has been cross-checked and is not the difference. Verified
identical: the background field (PINN u_bg vs the FDM's saved U_bg agree to
8.06e-07, i.e. float32 round-off), the Fresnel coefficients and the ky_sub
branch, k_int^2 / k_sub^2, the PML constants and the divergence-form UPML
operator term for term (including the Cxp/Cyp first-derivative terms), the
du_bg_air_dn_* sign fix, every sampler margin, the L0 assembly, and the
monitor (a zero prediction scores exactly relL2 = 1.0000).

That leaves three differences, and single_gap/ had all three at once:

  weights        forcing:homogeneous was 5:1 (W_FIELD/W_FLUX = 20, W_PML = 2)
                 against mie_pinn_sample.py's 20:1 (50, 50, 0.5). Field and
                 flux are the ONLY inhomogeneous terms -- everything else is
                 satisfied exactly by u = 0 -- so this ratio is what decides
                 how hard the incident wave is imposed at all.
  architecture   a DeepONet carries the b -> c*b, t -> t/c degeneracy, i.e.
                 flat directions in exactly the curvature L-BFGS estimates
                 (see variable_sample_mlp.py). Its output is also a
                 128-term dot product, so grad^2 u is a cancelling sum of 128
                 second derivatives against k_int^2 ~ 1.4e5, in float32.
  budget         single_gap/ ran 10000 Adam + 50000 closure evals ~ 2000 outer
                 L-BFGS steps. The mie_pinn_sample.py run that reached 0.045
                 used 3000 Adam + 10000 OUTER steps. At 2000 outer steps that
                 run was already at ~0.35 and still falling.

This launcher removes all three at once. It is deliberately NOT a bisection:
the point is to establish whether the configuration can reach 0.045 at all.

HOW TO READ THE RESULT
----------------------
    reaches ~0.045          the physics and losses in variable_sample_deeponet.py
                            are sound, and the fault is in the parametric layer.
                            Re-introduce one difference at a time (architecture
                            first, then gaps) with a known-good baseline under
                            it.

    stalls near 0.7 and     the discrepancy lives in this file at a single
    climbs                  geometry, and this is now the cheapest possible
                            reproduction -- one gap, one network, no gap
                            features -- to bisect against mie_pinn_sample.py.

Watch the relL2 curve's SHAPE, not just its final value: mie_pinn_sample.py
falls monotonically through L-BFGS. A curve that bottoms out early and then
climbs is the failure being chased, and it shows up long before the budget is
spent -- there is no need to let the full run finish to learn that.

CONFIGURATION
-------------
Everything below is set explicitly rather than inherited, so this control keeps
meaning its own thing as variable_sample_deeponet.py is edited. The values are
sample/mie_pinn_sample.py's, from the commit that produced relL2 = 0.045
(d455b3e): 3000 Adam epochs, 10000 outer L-BFGS steps, RESAMPLE_EVERY = 100,
N_FOURIER = 64, a 3x128 MLP, W_FIELD = W_FLUX = 50, W_PML = 0.5, N_PML = 2000.

LBFGS_STEPS is in CLOSURE EVALUATIONS here and in OUTER STEPS there. At
max_iter = 20 with a strong-Wolfe line search the measured conversion is ~24.6
evals per outer step (the inner loop runs the full 20 iterations plus ~4.6
line-search probes; this was measured directly, and is the same with the
default tolerances as with tightened ones). So 10000 outer steps ~ 250000
evals, which is the number set below.

Prerequisite, as for the DeepONet:
    python variable_sample/variable_sample_fdm.py

OUTPUTS -- written to their own directory, so nothing from the sweep runs is
overwritten:
    variable_sample/figures_single_mlp/variable_sample_single_mlp_gap2_*.png
                                       ..._training.mp4
    variable_sample/models/variable_sample_single_mlp_gap2_weights.pt
"""

import os
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "variable_sample_deeponet.py")

# Gap index 2 (gap = 0.052), matching variable_sample/single_gap/ so the two
# runs are directly comparable.
GAP_INDEX = 2

CONFIG = {
    # ── the control itself ──────────────────────────────────────────────────
    "SINGLE_GAP_INDEX": GAP_INDEX,

    # ── budget: sample/mie_pinn_sample.py's, converted to closure evals ─────
    "ADAM_EPOCHS":    3000,
    "LBFGS_STEPS":    200000,   # ~10000 outer steps at ~24.6 evals each
    "LBFGS_MAX_ITER": 20,
    "LR_ADAM":        8e-4,
    "RESAMPLE_LBFGS": 100,      # OUTER steps, as in the d455b3e run
    "RESAMPLE_ADAM":  250,

    # ── loss weights: forcing:homogeneous back to 20:1 ──────────────────────
    "W_PDE_INT": 1.0,
    "W_PDE_EXT": 1.0,
    "W_PDE_SUB": 1.0,
    "W_FIELD":   50.0,
    "W_FLUX":    50.0,
    "W_PML":     0.5,
    "W_PML_DIR": 1.0,
    "W_DATA":    0.0,           # physics only -- the whole point

    # ── architecture: sample/mie_pinn_sample.py's MLP, 3 x 128 ──────────────
    # ARCH is set through the existing ARCH_OVERRIDE hook below; these are the
    # shape constants MLPNet reads. They default to the TRUNK values (4 x 128),
    # which is one layer deeper than the reference.
    "N_FOURIER":   64,
    "MLP_HIDDEN":  3,
    "MLP_NEURONS": 128,

    # ── collocation counts: the reference's ─────────────────────────────────
    "N_INT":   1000,
    "N_AIR":   1500,
    "N_SUB":   1000,
    "N_IFACE": 800,
    "N_PML":   2000,            # N_PML_OUT follows this automatically

    # ── monitor cadence, in closure evals. 2500 ~ every 100 outer steps,
    #    matching the reference's cadence and giving ~100 points on the plot.
    "EVAL_EVERY": 2500,
}

# ── THE SHARED --short BUDGET ───────────────────────────────────────────────
# Both DeepONet arms take --short, and the only reason to run them is to
# compare their loss trajectories STEP FOR STEP: one geometry against eight, at
# an identical budget. Defining the three numbers here -- at the root of the
# import chain that both arms inherit from -- makes "the same budget" a
# property of the code rather than of two launchers agreeing by hand.
#
# ADAM_EPOCHS is part of it because the sweep arm raised Adam to 20000 after
# its first attempt was cut off mid-improvement, and the single-gap comparison
# is only meaningful at the same value.
SHORT_BUDGET = {
    "ADAM_EPOCHS": 20000,
    "LBFGS_STEPS": 15000,   # ~30 min at 8 geometries, ~4 min at one
    "EVAL_EVERY":  1000,    # resolve the Adam -> L-BFGS handover finely
}

# ARCH, the figure/model directories and the run tag go through the hooks that
# variable_sample_deeponet.py already reads with globals().get(); everything
# else goes through CONFIG_OVERRIDES, which it applies after its own constants
# are assigned. RUN_TAG picks up a "_gap2" suffix there, so the outputs are
# variable_sample_single_mlp_gap2_*.
# Guarded so variable_sample_single_deeponet.py can import CONFIG from here
# and change ONLY the architecture keys. "Apart from the architecture, the same
# run" is then a property of the code rather than a claim in a comment -- the
# same reasoning that makes variable_sample_mlp.py a launcher and not a copy.
if __name__ == "__main__":
    if not os.path.exists(_TARGET):
        raise SystemExit("variable_sample_deeponet.py not found next to this "
                         f"script: {_TARGET}")
    runpy.run_path(
        _TARGET,
        init_globals={
            "ARCH_OVERRIDE":      "mlp",
            "RUN_TAG_OVERRIDE":   "variable_sample_single_mlp",
            "FIG_DIR_OVERRIDE":   "variable_sample/figures_single_mlp",
            "MODEL_DIR_OVERRIDE": "variable_sample/models",
            "CONFIG_OVERRIDES":   CONFIG,
        },
        run_name="__main__",
    )
