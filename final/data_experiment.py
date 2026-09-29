"""
DATA-QUANTITY EXPERIMENT — shared configuration and runner
===============================================================================
How does a parametric surrogate over the AFM-tip sweep improve as the amount of
FDM supervision grows, with the physics loss present throughout?

The parameter RANGES are fixed at final_fdm.py's own -- half-angle 10-30 deg,
apex_y 0.41-0.60 (tip-sample gap 0.01-0.20) -- and only the SAMPLING DENSITY
inside them changes:

    final_4     2x2  =   4 FDM samples
    final_16    4x4  =  16
    final_36    6x6  =  36
    final_64    8x8  =  64
    final_100  10x10 = 100

and two arms that refine ONE axis, since final_4 showed two heights already
interpolate well across the tip-sample gap while the error depends mostly on the
half-angle (alpha x apex_y, so the height axis stays at its two ends):

    final_4x2   4x2  =   8
    final_6x2   6x2  =  12

Each directory holds its own inputs/, models/ and figures/, and a run.py that
drives the shared implementations in final/. Nothing is forked: final_fdm.py
and final_deeponet.py are the single copies, reached through the override hooks
they expose, so the five arms cannot drift from each other or from the original.

NO TRAIN/TEST SPLIT. Every solved grid point is training data (N_TEST = 0). A
held-out subset would both shrink the already-tiny 2x2 arm and measure the
wrong thing: the question is accuracy at parameters the model has NOT seen, and
the grids differ between arms, so a grid-point holdout is not even comparable
across them. evaluate_final.py answers it properly instead, solving a fresh FDM
reference at off-grid parameters -- by default the training-cell centres, which
are maximally far from every training point in both axes.

    Consequence to keep in mind: the relL2 the training run prints is a
    TRAINING number, because test_idx is set to train_idx. The honest figure
    for this experiment comes from evaluate_final.py.

BOTH ARCHITECTURES per arm, since final_deeponet.py already carries the MLP
behind ARCH. Weights are tagged <dir>_<arch>, e.g.
final_16/models/final_16_mlp_weights.pt, so provenance is unambiguous.

WAVELENGTH: lam = 1.0, chosen deliberately. It is what final_fdm.py and
final_deeponet.py are configured for, what the single-geometry run that reached
relL2 0.245 used, and what the afm_pinn run that reached 0.08 used -- the
docstring in final_fdm.py calls it the longest wavelength the operator learning
on this domain is known to handle.

    The one thing it costs: the only existing FULL-sweep number, relL2
    0.37-0.64 over 100 geometries, is at lam = 0.5
    (final/inputs/fdm_final_sweep_lam0.5.npz), so this experiment's 100-sample
    arm is NOT directly comparable to it. The in-family reference points at
    lam = 1.0 are the 0.245 single geometry and afm_pinn's 0.08.

Changing lam invalidates every solve, so if it is ever revisited, regenerate
all five arms' inputs/ from scratch.
"""

# A NOTE ON WHAT CONFIG_OVERRIDES CAN REACH. final_deeponet.py applies the
# override block just before it loads the npz, because FDM_PATH and
# SINGLE_GEOMETRY have to be settled by then. Constants defined AFTER that point
# -- RESAMPLE_ADAM, RESAMPLE_LBFGS, EVAL_EVERY -- are therefore not reachable,
# and passing them raises rather than silently doing nothing. None of them needs
# changing here: final/ already runs RESAMPLE_LBFGS = 25, the cadence whose
# absence was found to let L-BFGS memorise the collocation set in
# variable_sample/.

import os
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
FDM_SCRIPT = os.path.join(_HERE, "final_fdm.py")
NET_SCRIPT = os.path.join(_HERE, "final_deeponet.py")

# ── the arms: id -> (half-angle points, apex-height points) ─────────────────
# The square arms refine both axes together. The Nx2 arms keep the height axis
# at TWO points -- its ends, gap 0.01 and 0.20 -- and refine only the half-angle:
# final_4 showed that two heights already interpolate well across the gap, while
# the error depends mostly on the half-angle, so those arms ask how many ANGLES
# the surrogate needs. Directory and tag names follow the id: "4x2" lives in
# final_4x2/ and trains final_4x2_deeponet.
GRIDS = {4: (2, 2), 16: (4, 4), 36: (6, 6), 64: (8, 8), 100: (10, 10),
         "4x2": (4, 2), "6x2": (6, 2)}


def n_samples(arm):
    """FDM solves in an arm: the grid's two axes multiplied."""
    a, h = GRIDS[arm]
    return a * h


def grid_str(arm):
    a, h = GRIDS[arm]
    return f"{a}x{h}"


def arms_sorted():
    """Arm ids by sample count. GRIDS mixes int and str keys, so sorted() on the
    keys alone would raise."""
    return sorted(GRIDS, key=lambda a: (n_samples(a), str(a)))

# ── see THE WAVELENGTH IS A DECISION above ──────────────────────────────────
LAM = 1.0

# ── data weighting ──────────────────────────────────────────────────────────
# Deliberately light: the physics is meant to remain the primary constraint and
# the data only to steer it, which is the whole premise of the experiment. 0.1
# is a starting point, not a calibrated value -- every term is divided by its
# own L0 so 0.1 is 0.1 of a term that starts at 1, against W_FIELD/W_FLUX of 50
# on the two inhomogeneous terms. Tune it on the short runs before committing.
W_DATA = 1.0

# ── budgets ─────────────────────────────────────────────────────────────────
# final_deeponet.py's own values. Provisional: the point of --quick is to
# measure what these cost and how far they get before five arms x two
# architectures are launched at them.
ADAM_EPOCHS = 50000
LBFGS_STEPS = 200000

# --quick: enough to see the shape of the loss and relL2 curves and to time a
# run, not to converge anything.
QUICK_ADAM = 2000
QUICK_LBFGS = 5000

# ── the MATCHED budget ───────────────────────────────────────────────────────
# One budget for every method compared on final_4 and final_4x2, so the
# comparison is of objectives and not of training length. The runs done before
# this existed ranged from 10000 to 250000 L-BFGS evaluations -- mlp_dataonly
# beat mlp with a 25th of the optimisation and deeponet_dataonly beat deeponet
# with a tenth -- which makes every quantitative statement across them
# uninterpretable, even though the confound ran AGAINST the physics runs.
#
# 60000 comes from the logs rather than from taste. Taking the best-so-far
# relL2 over each completed L-BFGS phase and asking where it reaches a given
# fraction of everything that phase ever delivered:
#
#     run                          50%      80%      90%      end
#     final_4_deeponet           25751    74246   121697   248933
#     final_4_mlp                39442   149162   246722   249182
#     final_4_mlp_phys           21757    57570   115825   198780
#     final_4x2_deeponet         17272    47119    85745   249146
#     final_4x2_mlp              15725    61511   123165   248997
#     final_4x2_deeponet_dataonly 7624    16505    20315    24137
#
# The median 80% point is ~60000. The last 20% is not worth its cost, and on
# the only run measured on UNSEEN parameters at two depths it was negative:
# final_4_mlp_phys scored 0.0743 at 119597 L-BFGS evaluations and 0.0760 after
# another 100000, with 62% of that continuation's loss reduction going into the
# PML residual. The table above tracks the TRAINING geometries, which the same
# run showed is a poor proxy and latterly an inverted one.
#
# Matched in STEPS, not wall clock: a step is an optimiser evaluation of the
# objective, so equal steps compare the objectives. Methods that evaluate
# physics at extra parameter draws cost more per step and will take longer in
# hours; that is a property of the method, and matching wall clock instead
# would silently hand the cheap methods more optimisation.
MATCHED_ADAM = 20000
MATCHED_LBFGS = 60000

# The data-only cells are the exception, and are trained to convergence instead.
# They converge in minutes and then stop moving: final_4_mlp_dataonly's Adam
# phase was at relL2 0.0077 by step 2000, and its entire L-BFGS phase --
#
#     step 15632 (L-BFGS start)   relL2 0.001899   mag_ratio 0.7193241
#     step 20083                  relL2 0.001825   mag_ratio 0.7193354
#     step 24533 (end)            relL2 0.001764   mag_ratio 0.7193471
#
# -- moved relL2 by 1.4e-4 absolute, non-monotonically, with mag_ratio matching
# the FDM target of 0.7193345 to five decimals and the data term at 1.8e-6.
#
# The drift that makes a long budget matter elsewhere cannot happen here. A
# physics run keeps receiving gradient after the data fit plateaus -- which is
# how final_4_mlp_phys's continuation improved its anchors while its unseen
# error got worse -- but with data alone at 1.8e-6 there is nothing left pushing
# the weights anywhere. The asymmetry is deliberate and belongs in the writeup:
# the matched budget applies to the cells whose objective still has gradient.
DATAONLY_ADAM = 15000
DATAONLY_LBFGS = 10000

# --warm: continue training from an arm's existing checkpoint, e.g.
# final_4/models/final_4_deeponet_weights.pt. L-BFGS only (INIT_WEIGHTS in
# final_deeponet.py says why no Adam), saved under its own <dir>_<arch>_warm tag
# so the checkpoint it starts from is never overwritten. Its L-BFGS budget
# counts from zero: 250000 more closure evals on top of the original run.
WARM_ADAM_EPOCHS = 0
WARM_LBFGS_STEPS = 100000

# Geometries per optimiser step, for WARM runs only. Fresh runs stay at 4 so
# the five arms remain comparable with each other and with what is already
# trained. 4 is every geometry at once for final_4, but only 4 of 100 for
# final_100 -- and the batch is held for RESAMPLE_LBFGS = 25 outer steps, so
# each L-BFGS block there fits a different 4-geometry subproblem and rebuilds
# its curvature estimate on it. Raising this costs proportionally more per
# closure eval (8 geometries is ~2x the work of 4), so either expect twice the
# wall clock or halve WARM_LBFGS_STEPS. Capped at the arm's geometry count.
WARM_PARAM_BATCH = 64

# Geometries per optimiser step for FRESH runs, capped at the arm's geometry
# count (so final_16 at 16 sees every geometry at every step). The arms trained
# before this was introduced used 4; fresh runs at another value are not
# directly comparable with those.
PARAM_BATCH = 64

# GPU memory: the batch is evaluated and backpropagated ACCUM_CHUNK geometries
# at a time (the gradient is identical), so a large batch needs only the memory
# of ACCUM_CHUNK geometries. At 16 in one pass final_100 ran out of an 11.6 GB
# RTX 3080 Ti with 10.8 GB allocated, i.e. roughly 0.7 GB per geometry, so 8 fits
# with room to spare. Smaller chunks are slower; 0 = no chunking.
ACCUM_CHUNK = 8


def _paths(n):
    d = os.path.join(_REPO, f"final_{n}")
    return d, os.path.join(d, "inputs"), os.path.join(d, "models"), \
        os.path.join(d, "figures")


def npz_path(n):
    return os.path.join(_paths(n)[1], f"fdm_final_{n}.npz")


def run_fdm(n):
    """Solve the n-sample sweep and write it into final_<n>/inputs/."""
    n_alpha, n_height = GRIDS[n]
    _, inp, _, figs = _paths(n)
    os.makedirs(inp, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    print(f"=== FDM: {grid_str(n)} = {n_samples(n)} samples "
          f"({n_alpha} half-angles x {n_height} heights), lam = {LAM} "
          f"-> {npz_path(n)}")
    runpy.run_path(FDM_SCRIPT, run_name="__main__", init_globals={
        "LAM_OVERRIDE": LAM,
        "N_ALPHA_OVERRIDE": n_alpha,
        "N_HEIGHT_OVERRIDE": n_height,
        # None = solve every grid point, not just the one nearest a chosen tip.
        "SOLVE_ONE_NEAREST_OVERRIDE": None,
        "N_TEST_OVERRIDE": 0,            # no split -- see the module docstring
        "INPUT_DIR_OVERRIDE": inp,
        "OUT_NPZ_OVERRIDE": npz_path(n),
        "FIG_DIR_OVERRIDE": figs,
    })


def run_train(n, arch, quick=False, warm=False):
    """Train one architecture on the n-sample sweep. warm=True continues from
    the arm's existing <dir>_<arch> checkpoint into a <dir>_<arch>_warm one."""
    assert arch in ("deeponet", "mlp"), arch
    _, _, models, figs = _paths(n)
    os.makedirs(models, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    npz = npz_path(n)
    if not os.path.exists(npz):
        raise SystemExit(f"{npz} not found -- run with --fdm first")

    adam = QUICK_ADAM if quick else ADAM_EPOCHS
    lbfgs = QUICK_LBFGS if quick else LBFGS_STEPS
    tag = f"final_{n}_{arch}"
    init = None
    if warm:
        init = os.path.join(models, f"{tag}_weights.pt")
        if not os.path.exists(init):
            raise SystemExit(f"{init} not found -- --warm continues an arm that "
                             "has already been trained")
        adam = WARM_ADAM_EPOCHS
        lbfgs = QUICK_LBFGS if quick else WARM_LBFGS_STEPS
        # The tag does NOT change: final_deeponet.py recognises a run that
        # starts from its own checkpoint as a CONTINUATION -- the step count
        # carries on, the relL2 CSV is appended to, and every figure is redrawn
        # over the whole training instead of a second _warm set appearing. The
        # weights it starts from are kept as <tag>_pre_warm_weights.pt.
    cfg = {
        # Every solved point is a training geometry.
        "SINGLE_GEOMETRY": False,
        # N_PARAM_BATCH is computed from SINGLE_GEOMETRY before the override
        # block runs, so it has to be set explicitly. 4 is final_deeponet.py's
        # own sweep value; the 2x2 arm has only 4 points to draw from. Warm
        # runs use WARM_PARAM_BATCH instead -- see the note there.
        "N_PARAM_BATCH": min(WARM_PARAM_BATCH if warm else PARAM_BATCH,
                             n_samples(n)),
        "ACCUM_CHUNK": ACCUM_CHUNK,
        # All THREE regions. The substrate term was added after the
        # final_100 run showed that region carrying 33.7% of the squared
        # error at 0.352 relL2 while unsupervised, against 0.0013 inside
        # the tip and 0.248 in the air. Nothing but its own term reaches
        # it: l_pde_sub is homogeneous, so u = 0 satisfies it exactly.
        "W_DATA_SCAT": W_DATA,
        "W_DATA_INT": W_DATA,
        "W_DATA_SUB": W_DATA,
        "ADAM_EPOCHS": adam,
        "LBFGS_STEPS": lbfgs,
    }
    print(f"=== TRAIN {arch} on {n_samples(n)} samples ({grid_str(n)})"
          f"{'  [QUICK]' if quick else ''}"
          f"{f'  [WARM from {os.path.basename(init)}]' if warm else ''}"
          f": Adam {adam}, L-BFGS {lbfgs} evals, W_DATA {W_DATA}  -> {tag}")
    runpy.run_path(NET_SCRIPT, run_name="__main__", init_globals={
        "LAM_OVERRIDE": LAM,
        "FDM_PATH_OVERRIDE": npz,
        "ARCH_OVERRIDE": arch,
        "RUN_TAG_OVERRIDE": tag,
        "FIG_DIR_OVERRIDE": figs,
        "MODEL_DIR_OVERRIDE": models,
        "INIT_WEIGHTS_OVERRIDE": init,
        "CONFIG_OVERRIDES": cfg,
    })


def main(n, argv):
    """Shared entry point for the per-directory run.py launchers."""
    known = {"--fdm", "--deeponet", "--mlp", "--quick", "--both", "--eval",
             "--eval-fine", "--warm"}
    usage = (f"  usage: python final_{n}/run.py "
             "[--fdm] [--deeponet] [--mlp] [--both] [--warm] [--quick] [--eval] "
             "[--eval-fine]")
    bad = [a for a in argv if a not in known]
    if bad:
        raise SystemExit(f"unknown argument(s): {' '.join(bad)}\n{usage}")
    quick = "--quick" in argv
    warm = "--warm" in argv
    if warm and not ({"--deeponet", "--mlp", "--both"} & set(argv)):
        raise SystemExit(f"--warm says how to train, not what: add --deeponet, "
                         f"--mlp or --both\n{usage}")
    did = False
    if "--fdm" in argv:
        run_fdm(n)
        did = True
    for arch in ("deeponet", "mlp"):
        if f"--{arch}" in argv or "--both" in argv:
            run_train(n, arch, quick, warm)
            did = True
    if "--eval" in argv or "--eval-fine" in argv:
        # Imported here rather than at the top: it pulls in torch and
        # matplotlib, which --fdm does not need. It runs after any training
        # requested alongside it, so --both --eval scores the models just made.
        # For several arms at once, or to split the fine grid's reference
        # solves over processes (--part), use final/evaluate_arms.py directly.
        import evaluate_arms
        if "--eval" in argv:
            evaluate_arms.evaluate(arms=[n])
        if "--eval-fine" in argv:
            # 50x50: 2500 FDM solves the first time; a detailed heat map only
            evaluate_arms.evaluate(arms=[n], k=evaluate_arms.FINE_K)
        did = True
    if not did:
        raise SystemExit(f"nothing to do.\n{usage}")
