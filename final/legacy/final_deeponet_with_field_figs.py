"""
FINAL — DeepONet over TWO parameters (tip half-angle AND tip-sample height),
with the dielectric substrate present.

Derived from variable_tips/tip_sweep_deeponet.py, which supplies the parametric
DeepONet, the angle batching, the PML operator-branch split and the relL2
monitor. The substrate physics is grafted across from afm_pinn/afm_pinn.py.

    branch input : 8 features of (half_angle, apex_y)  -- see _raw_features
    trunk input  : (x, y) Fourier features only

apex_y is the y of the LOWEST point of the tip, so the tip-sample gap is
apex_y - Y_SUB. The (alpha, apex_y) grid arrives FLATTENED to one index pi from
final_fdm.py, which is what lets every per-parameter mechanism here -- caches,
train/test split, batching -- stay exactly as it was.

WHAT THE SUBSTRATE CHANGES relative to variable_tips:
  * background is Fresnel (incident + reflected above Y_SUB, transmitted
    below), not a bare plane wave, so the tip-interface conditions use u_bg_air
  * the exterior splits into TWO PDE regions, air (k_ext^2) and substrate
    (k_sub^2), both carried by ONE network (see the net_sub alias) because
    u_scat is C^1 across y = Y_SUB in s-polarisation
  * the PML band splits into THREE operator branches, not two: tip shank
    (net_int, k_int^2), air (net_scat, k_ext^2), substrate (net_scat, k_sub^2)
  * B_ext is scaled by |k_sub|, not k, so the trunk can resolve the faster
    substrate oscillation

WHY HEIGHT IS AN INTERESTING AXIS. With a substrate the background is a
STANDING WAVE in y of period pi/|ky| = 0.354. Sweeping the gap from 0.01 to
0.20 walks the apex across more than half of one period, so the near-field
varies non-monotonically rather than just weakening with distance.

    u(params, x, y) = sum_i  branch_i(params) * trunk_i(x, y)

branch : the PARAMETER vector -> BRANCH_DIM coefficients.  Currently that
         vector is the alpha-derived shape features from _raw_features(),
         but nothing here assumes it is only the half-angle: to add
         parameters (wavelength, refractive index, apex radius, ...) extend
         _raw_features() and bump PHYS_DIM, and the branch widens with it.
trunk  : (x, y) -> BRANCH_DIM spatial basis functions, via the same Fourier
         feature embedding the PINN uses. The trunk does NOT see the
         parameters -- that separation is the point of the architecture.

Re/Im split: BRANCH_DIM is halved, first half dotted for Re(u), second for
Im(u), following mie_deeponet_pml.py.

CAVEAT worth knowing before reading the results. A DeepONet is at its best
when the operator maps a parameter to a field on a FIXED domain. Here the
geometry itself moves with alpha -- the interface Gamma(alpha) and its
normals shift -- so a trunk basis that cannot depend on alpha has to span
every geometry at once with BRANCH_DIM/2 functions. If accuracy lags the
PINN noticeably that is the first thing to suspect, and TRUNK_SEES_SDF
below is the escape hatch.

Parametric PINN — 2D Dielectric AFM-Tip Scattering (TM mode), half-angle sweep
===============================================================================
Shape-parametric variant of afm_tip_pinn.py, structured like ellipse_pinn.py.
A single pair of networks is trained across all cone half-angles alpha in the
training split, with an alpha-derived physics feature vector as the extra
conditioning input:

  net_int  : learns u_int(x,y; alpha)   — total field  INSIDE  the tip (phi<0)
  net_scat : learns u_scat(x,y; alpha)  — scattered field OUTSIDE the tip

FIXED PARAMETERS (all identical to fdm_afm_tip_sweep.py)
--------------------------------------------------------
  N_RE, N_IM, lam (hence k), TIP_X, R_APEX, theta_inc, and the
  substrate (SUB_N_RE, SUB_N_IM, Y_SUB)

PHYSICS — identical PDE, BCs and PML treatment to afm_tip_pinn.py
------------------------------------------------------------------
  PDE interior  : (∇² + k_int²) u_int  = 0   (k_int² complex, n = 3+30i)
  PDE exterior  : (∇² + k²)     u_scat = 0
  Interface     : u_int = u_inc + u_scat        on phi = 0
  Flux          : ∂_n u_int = ∂_n(u_inc+u_scat) on phi = 0
                  (normals via autograd of the alpha-parametrised SDF)
  PML           : UPML-stretched Helmholtz in the PML band. Because the tip
                  SHANK crosses the top PML, the operator switches per point:
                  net_int + k_int² inside the shank, net_scat + k² outside
                  (exactly as in afm_tip_pinn.py) — and the inside mask now
                  depends on alpha.
  Dirichlet     : u_scat = 0 on the outer PML boundary
  All residuals non-dimensionalised by the local |k²| (afm_tip_pinn.py), and
  all loss terms normalised by their initialisation value L0 so every term
  starts at 1 (afm_tip_pinn.py).

GEOMETRY-DEPENDENT COLLOCATION SAMPLING  (ellipse_pinn.py structure)
--------------------------------------------------------------------
The interior region, exterior exclusion region, the interface itself, and
the inside-shank mask of the PML band all depend on alpha, so:

  * sample_inside / sample_air / sample_interface take (alpha, apex_y);
  * each parameter index gets its OWN spatial point set (lazy cache,
    cleared every RESAMPLE_ADAM epochs);
  * interface points are drawn in a thin shell |phi| < eps and projected
    onto phi = 0 along the SDF gradient; normals via autograd of phi_torch;
  * PML-band points and the outer Dirichlet points are shared across the
    parameter batch (their stretch tensors are geometry-independent), but
    each parameter gets its own inside-shank mask on those shared points
    (cached, invalidated when the PML sample is redrawn);
  * during L-BFGS both the parameter batch AND its spatial points are fixed
    per optimizer block (the fixed-batch fix for the L-BFGS NaN issue).

Usage
-----
Run fdm_afm_tip_sweep.py first to generate inputs/fdm_afm_tip_sweep.npz.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import map_coordinates
import csv
import shutil
import time
import imageio.v2 as imageio
from io import BytesIO

# ─────────────────────────────────────────────────────────────────────────────
# FIXED PARAMETERS — identical to fdm_afm_tip_sweep.py
# ─────────────────────────────────────────────────────────────────────────────
N_RE      = 3.0
N_IM      = 30.0
# ── Substrate (dielectric half-space) — as in afm_pinn/afm_pinn.py ──────────
SUB_N_RE = 1.5
SUB_N_IM = 0.1
Y_SUB    = 0.4     # flat substrate interface height
# Unit mapping from the physical AFM problem: R_APEX = 0.05 <-> 25 nm tip
# apex radius, so 1 script-unit = 500 nm. Target illumination 5-10 um therefore
# would map to lam = 10-20 here, but the unit domain is then deeply
# sub-wavelength and training has consistently failed to converge there (see
# afm_pinn_2/). lam = 1.0 matches the afm_pinn run that reached relL2 = 0.08.
# Must match the lam saved in final_fdm.py's meta (asserted below).
lam       = globals().get("LAM_OVERRIDE", 1.0)
k         = 2.0 * np.pi / lam
TIP_X     = 0.5
R_APEX    = 0.05
theta_inc = -np.pi / 4
kx_inc    = k * np.cos(theta_inc)
ky_inc    = k * np.sin(theta_inc)

k_ext_sq     = k**2
k_int_c      = (N_RE + 1j * N_IM) * k
k_int_sq_re  = float((N_RE**2 - N_IM**2) * k**2)   # Re(k_int²)
k_int_sq_im  = float(2 * N_RE * N_IM * k**2)        # Im(k_int²)
k_int_sq_mag = float(np.hypot(k_int_sq_re, k_int_sq_im))
k_ext_sq_mag = float(k_ext_sq)

# ── Substrate wavenumber and s-polarisation Fresnel coefficients ────────────
# s-pol (Ez out of plane) is the mode every solver in this repo enforces: u AND
# du/dn continuous. With these coefficients the background satisfies both
# matching conditions at y = Y_SUB exactly, so u_scat carries no interface line
# source and is purely tip-induced — the same object final_fdm.py solves for.
k_sub_c      = (SUB_N_RE + 1j * SUB_N_IM) * k
k_sub_sq_re  = float((SUB_N_RE**2 - SUB_N_IM**2) * k**2)
k_sub_sq_im  = float(2 * SUB_N_RE * SUB_N_IM * k**2)
k_sub_sq_mag = float(np.hypot(k_sub_sq_re, k_sub_sq_im))

_ky_sub = np.sqrt((SUB_N_RE + 1j * SUB_N_IM)**2 * k**2 - kx_inc**2 + 0j)
if _ky_sub.real > 0:
    _ky_sub = -_ky_sub          # Re < 0: downward-propagating
_q_air = abs(ky_inc)
_q_sub = -_ky_sub               # Re > 0
r_TM   = (_q_air - _q_sub) / (_q_air + _q_sub)
t_TM   = (2.0 * _q_air)  / (_q_air + _q_sub)
r_re, r_im = float(r_TM.real), float(r_TM.imag)
t_re, t_im = float(t_TM.real), float(t_TM.imag)
ky_sub_re, ky_sub_im = float(_ky_sub.real), float(_ky_sub.imag)

# (apex centre is now per-parameter: see _alpha_consts)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────────
# Set back to False to restore the full sweep; nothing else needs changing.
# ─────────────────────────────────────────────────────────────────────────────
SINGLE_GEOMETRY = True
# The grid point trained on, chosen nearest to afm_pinn's own tip: half-angle
# 20 deg, APEX_Y = 0.45 (a 0.05 gap above the substrate).
SINGLE_ALPHA_DEG = 20.0
SINGLE_APEX_Y    = 0.45

# Collocation counts, all matched to afm_pinn/afm_pinn.py. N_PML is its 1500
# PER REGION x 3 regions (air, substrate, tip shank), since this script draws
# one PML sample and partitions it by region instead of sampling each
# separately; N_DIR_SIDE x 4 reproduces its sample_pml_outer(N_PML // 2) = 750.
N_INT         = 1000   # interior collocation points  (inside tip)
N_EXT         = 1500   # AIR collocation points (outside tip, above Y_SUB)
N_SUB         = 1000   # SUBSTRATE collocation points (below Y_SUB)
N_IFACE       = 800    # interface phi=0 points
# Field and flux continuity are NOT enforced on the top IFACE_TOP_EXCLUDE of
# the shank, just below the outer boundary y = 1 + D_PML, where they contradict
# the Dirichlet condition -- see sample_interface_bc. 0 enforces them everywhere.
IFACE_TOP_EXCLUDE = 0.1
# One shared band sample, partitioned by mask into the three operator
# branches. A dedicated denser shank sample (N_PML_TIP = 1500, matching
# afm_pinn's per-region count) was tried and REVERTED: three runs with it were
# all worse than the 0.296 baseline at the same 20000 + 50000 budget. That
# includes the run where the shank points were ADDED on top (N_PML = 4500 and
# N_PML_TIP = 1500, 6000 total) rather than paid for out of the air and
# substrate budget, which reached 0.317 -- so the regression is not a matter of
# starving the other regions. Under SKIN_ANSATZ the field deep in the shank is
# ~0, so those points appear to carry almost no signal either way.
N_PML         = 4500   # PML-band collocation points (shared across batch)
N_DIR_SIDE    = 190    # outer Dirichlet points per side (×4 sides)
N_PARAM_BATCH = 1 if SINGLE_GEOMETRY else 4

# Evaluate all N_PARAM_BATCH half-angles in ONE forward per term instead of
# looping over them.
BATCH_ANGLES  = True
# Gradient accumulation for large batches: with BATCH_ANGLES, evaluate the
# batch ACCUM_CHUNK geometries at a time, backpropagating each chunk before the
# next, so peak GPU memory is that of ACCUM_CHUNK geometries while the gradient
# is exactly that of the whole batch. 0 = one pass over the whole batch. Needed
# because the double-backward graph grows linearly with the batch: final_100 at
# N_PARAM_BATCH = 16 ran out of an 11.6 GB card with 10.8 GB allocated.
ACCUM_CHUNK   = 0

# PHYSICS AT PARAMETERS WITH NO FDM DATA.
#
# Every loss term -- PDE, PML, interface, Dirichlet AND data -- is evaluated at
# the TRAINING geometries only, so nothing constrains the surrogate BETWEEN
# them: in the parameter direction a run here is no better determined than plain
# regression, and only the network's own smoothness interpolates. final_4x2 made
# that visible once the optimisation was fixed: relL2 0.0051 at its 8 training
# geometries against 0.1218 on the 50x50 grid, with the error flat in half-angle
# (4 trained values) and peaking between the two trained GAPS -- 0.032 and 0.086
# at gap 0.20 and 0.01, but 0.16 in the middle.
#
# The physics needs no FDM solve, so it can be imposed anywhere in the range.
# N_PHYS_EXTRA draws that many random (alpha, apex_y) per point-set and adds
# their PHYSICS terms (no data) to the objective, which is what makes the
# physics do parametric work rather than only regularising the field at points
# that already have data.
#
# Cost: one extra physics evaluation per draw, through the per-geometry path, so
# N_PHYS_EXTRA = 4 against a batch of 8 is roughly +50% per closure evaluation.
# The draws are refreshed with the collocation points, NOT every step: L-BFGS
# needs a fixed objective within its block.
N_PHYS_EXTRA      = 0
PHYS_EXTRA_WEIGHT = 1.0

# WHERE the draws land, as the fraction of each axis they span, centred on the
# middle of the sweep. 1.0 is the whole range; 0.5 is the middle half; 0.0 is
# the exact centre.
#
# Uniform draws spend most of their effort where the surrogate is already right.
# final_4x2's 50x50 heat map is a ridge along the GAP axis: the error is under
# 0.01 at the two trained gaps (0.01 and 0.20, the only values with data) and
# peaks near 0.09 in the middle, while it barely varies with half-angle, which
# has four trained values. Narrowing the height span puts the physics where that
# ridge is; leaving the alpha span at 1.0 keeps the draws spread along the axis
# that is already well covered, which costs nothing and avoids over-fitting one
# slice.
#
# Beware PHYS_SPAN_APEX = 0.0 with PHYS_SPAN_ALPHA = 0.0: every draw is then the
# same geometry, and N_PHYS_EXTRA copies of it.
PHYS_SPAN_ALPHA = 1.0
PHYS_SPAN_APEX  = 1.0


def _phys_span(lo, hi, frac):
    """The middle `frac` of [lo, hi]; frac = 1 is the whole range, 0 the centre."""
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * float(frac)
    return mid - half, mid + half

# ── PML parameters — keep identical to final_fdm.py ──────────────────────────
# Widened from 0.5/1e-8 to afm_pinn/afm_pinn.py's 1.0/1e-10; final_fdm.py
# carries the same change, so the reference and the trained field see the same
# absorbing boundary. See the note there on the residual leak at 0.5/1e-8.
D_PML         = 1.0
PML_POW       = 3
# NOT tied to k (see afm_pinn/afm_fdm.py for the derivation): round-trip PML
# reflection depends only on integral(sigma dx) = sigma_max*d_pml/(pow+1),
# wavelength-independent once the stretch factor uses a matched k_ref. Value
# must match final_fdm.py's sigma_max exactly (same PML_R0 target) since this
# script trains against that solver's field data.
PML_R0        = 1e-10   # target round-trip reflection coefficient
PML_SIGMA_MAX = -(PML_POW + 1) * np.log(PML_R0) / (2.0 * D_PML)

# Budgets and optimiser settings matched to afm_pinn/afm_pinn.py, except
# LBFGS_STEPS which is 50000 rather than its 80000.
ADAM_EPOCHS = 20000
LBFGS_STEPS = 100000
LBFGS_MAX_ITER = 20   # inner iterations per outer step (afm_pinn's max_iter)
LR_ADAM     = 8e-4

# Loss weights — same relative scheme as afm_tip_pinn.py; every term is
# divided by its initialisation value L0 so all terms start at 1.
W_PDE_INT   = 1.0
W_PDE_EXT   = 1.0
W_PDE_SUB   = 1.0
W_FIELD     = 50.0
W_FLUX      = 50.0
W_PML       = 0.5
W_DIRICHLET = 1.0

# Normalise field continuity by the physical surface scale (2/|n|)^2 instead of
# the measured L0. Tried at W_FIELD 50 and 10: it DID fix what it targeted (the
# surface residual fell from 100% of the physical field to 42%), but both runs
# came out worse overall than the measured-L0 baseline -- relL2 0.71 and 0.64
# against 0.18, and flux(out>0) went NEGATIVE, i.e. an inward-propagating
# scattered field.
#
# The reason looks structural rather than a weighting accident: with
# _MARGIN_IN = 1.88*delta no interior collocation point lies inside the
# lambda/188 boundary layer, so u_int near Gamma is not pinned by the interior
# PDE. Field continuity is then a constraint between a well-determined quantity
# (u_total from outside) and a free one (u_int), and tightening it moves u_int
# rather than fixing u_total -- ||u_int|| grew 7.95e-3 -> 2.63e-2, which drove
# d_n u_int ~ |k_int|*u_int up and made the FLUX residual climb during training.
# Worth revisiting only once the boundary layer is actually resolved.
# Set False to match afm_pinn/afm_pinn.py, which normalises field continuity by
# its MEASURED L0. Both of this script's own physical-L0 runs came out worse
# than the measured-L0 baseline anyway (relL2 0.71 and 0.64 against 0.18), for
# the structural reason set out above.
USE_PHYSICAL_L0_FIELD = False
# FDM supervision, per region. All three default to 0.0, i.e. physics only.
#   SCAT  u_scat in the AIR       (xy_out, above Y_SUB, outside the tip)
#   INT   u_total INSIDE the tip  (xy_in)
#   SUB   u_scat in the SUBSTRATE (xy_sub, below Y_SUB)
# The substrate term was missing until the final_100 run made the gap
# visible: with W_DATA_SCAT = W_DATA_INT = 1.0 that run reached relL2
# 0.0013 inside the tip and 0.248 in the air, but 0.352 in the substrate,
# which held 15.7% of the field norm and 33.7% of the squared error. The
# substrate's only constraint was l_pde_sub, which is homogeneous -- u = 0
# satisfies it exactly -- so nothing pinned the amplitude there except
# continuity with the air field across y = Y_SUB, and no value of the other
# two weights could reach it. variable_sample/ supervises all three.
W_DATA_SCAT = 0.0
W_DATA_INT  = 0.0
W_DATA_SUB  = 0.0

# Architecture
N_FOURIER = 64   # Fourier features for the TRUNK (spatial) net, wavelength band
# Second, higher-frequency band at the tip's own geometric scale, as in
# afm_pinn/afm_pinn.py. It was added there because the training animation showed
# the near-field null failing to reach the apex: the wavelength band has a
# period of ~1/|k_sub|, which cannot resolve structure on the scale of
# R_APEX = 0.05. Both trunks get it, matching that script's B_ext and B_int.
N_FOURIER_TIP = 32
K_TIP         = 1.0 / R_APEX   # period ~ 2*pi*R_APEX
PHYS_DIM  = 8    # length of the parameter vector fed to the BRANCH net.
                 # Currently alpha-derived shape features (see _raw_features);
                 # extend that function and raise this to add parameters.

# Branch: parameters -> coefficients. Small, since the parameter space is
# low-dimensional; widen it as PHYS_DIM grows.
BRANCH_HIDDEN  = 3
BRANCH_NEURONS = 64
# Trunk: (x,y) -> spatial basis. This carries the oscillatory structure, so it
# gets the capacity the PINN's MLP had.
TRUNK_HIDDEN   = 4
TRUNK_NEURONS  = 128
# Dot-product width. Split in half: first BRANCH_DIM/2 for Re, rest for Im, so
# each field component gets BRANCH_DIM/2 basis functions.
BRANCH_DIM     = 256

# ARCH = "mlp" only. Defaults to the trunk's depth and width, so the comparison
# is "same spatial network, different head". That leaves the MLP with fewer
# parameters, since the DeepONet also carries a branch. The startup banner prints
# both counts and the width that would give parity, computed rather than quoted.
MLP_HIDDEN     = TRUNK_HIDDEN
MLP_NEURONS    = TRUNK_NEURONS

# ─────────────────────────────────────────────────────────────────────────────
# SKIN ANSATZ  --  u_int = A(x,y) * exp(-i k_int phi(x,y))
SKIN_ANSATZ = True

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE  --  "deeponet" or "mlp"
#
# final_mlp.py runs THIS script with ARCH_OVERRIDE = "mlp" so the two
# architectures are compared through one code path: identical physics, losses,
# samplers, budgets, optimiser, monitor and figures, with the network as the
# only difference. Forking the file would have guaranteed drift.
#
#   deeponet : u = sum_i branch_i(params) * trunk_i(x,y)   (rank-BRANCH_DIM/2)
#   mlp      : u = MLP([fourier(x,y), params])             (no factorisation)
#
# The mlp path takes the SAME parameter vector as an input, concatenated onto
# the Fourier features, so it stays a parametric surrogate and will accept a
# geometry sweep unchanged; with SINGLE_GEOMETRY those 8 inputs are simply
# constant and it degenerates to afm_pinn/afm_pinn.py's plain network.
#
# What the comparison isolates: the DeepONet's separable form has a scaling
# degeneracy (b -> a*b, t -> t/a leaves u unchanged), which adds flat directions
# to exactly the curvature L-BFGS is trying to estimate -- and L-BFGS is what
# does the work in these runs. The MLP has no such degeneracy.
ARCH = globals().get("ARCH_OVERRIDE", "deeponet")
assert ARCH in ("deeponet", "mlp"), f"unknown ARCH {ARCH!r}"

# Output locations, so the two architectures do not overwrite each other.
FIG_DIR   = globals().get("FIG_DIR_OVERRIDE",   "final/figures")
MODEL_DIR = globals().get("MODEL_DIR_OVERRIDE", "final/models")
RUN_TAG   = globals().get("RUN_TAG_OVERRIDE",   "final_deeponet")
CKPT_PATH = f"{MODEL_DIR}/{RUN_TAG}_weights.pt"
# The BEST weights by the relL2 monitor, saved whenever it improves, because
# CKPT_PATH is only ever the LAST state and a run can end worse than it was:
# the first TRUNK_NORM run passed through 0.0508 at Adam epoch 10000 and
# finished at 0.298, with nothing keeping the good weights. Same format as
# CKPT_PATH plus best_relL2 / best_step, so every consumer reads it unchanged --
# evaluate_arms scores it as the arch "<arch>_best".
BEST_CKPT_PATH = f"{MODEL_DIR}/{RUN_TAG}_best_weights.pt"
# The CURRENT weights, rewritten at every monitor evaluation. The best file
# answers "what is the best model this run has produced"; this one answers "what
# is it doing NOW", which the relL2 monitor alone cannot: that metric is over the
# TRAINING geometries, so a run whose training number has stalled may still be
# improving (or degrading) at unseen parameters. Score this file mid-run to find
# out. Both saves go through _save_atomic, so a reader never catches a
# half-written checkpoint.
LATEST_CKPT_PATH = f"{MODEL_DIR}/{RUN_TAG}_latest_weights.pt"
SAVE_LATEST = True

# WARM START: a checkpoint to initialise both networks from (None = random).
# The weights are loaded AFTER the L0 pass, so the loss normalisation is the one
# a fresh run under this code would measure -- L0 taken on trained weights would
# shrink every term's ruler and change the balance between them. The Fourier
# matrices and the branch feature normalisation are checked against it (and the
# matrices copied) where they are created, before torch.compile or L0 use them.
# Launchers pair it with ADAM_EPOCHS = 0: Adam's full learning rate would throw
# away much of what the checkpoint holds, and L-BFGS rebuilds its optimiser at
# every redraw anyway, so a warm L-BFGS phase is the old one carrying on.
INIT_WEIGHTS = globals().get("INIT_WEIGHTS_OVERRIDE", None)

# CONTINUING A RUN. Starting from THIS tag's own checkpoint means the run is a
# continuation, not a new one: the step count carries on from where the previous
# one stopped, the relL2 CSV is appended to, and the relL2 and amplitude figures
# are redrawn over the WHOLE training. Those two are the same measurement in
# every run -- the same monitor against the same FDM reference -- so they join
# honestly.
#
# THE LOSS FIGURE IS PER RUN, and deliberately so: every term is logged as
# W * loss / L0, a continuation measures its own L0 at a fresh initialisation,
# and a change to the loss (a new term, a fixed one) makes the two sides measure
# different things. A continuation therefore writes its own
# <RUN_TAG>_loss_from<step>.png and leaves the earlier figure alone, rather than
# drawing one curve that silently steps at the restart.
#
# The MP4 cannot be extended either -- its frames are not kept -- so it covers
# the latest run only.
#
# The checkpoint it starts from is copied to <RUN_TAG>_pre_warm_weights.pt
# before training, so a continuation that goes wrong cannot destroy the model it
# began with.
CONTINUE_RUN = bool(INIT_WEIGHTS) and \
    os.path.abspath(INIT_WEIGHTS) == os.path.abspath(CKPT_PATH)
PREV_CKPT_PATH = f"{MODEL_DIR}/{RUN_TAG}_pre_warm_weights.pt"

# Everything the flag pulls with it, defined here so the startup banner can
# report the configuration before any of it is used.
#
# SHELL_DEPTH: interior points are drawn in |phi| < SHELL_DEPTH rather than over
# the whole tip cross-section. 15 skin depths puts the outer edge at exp(-15) =
# 3e-7 of the surface value, so nothing of consequence lives beyond it -- and
# past phi ~ -0.47 the float32 exp underflows anyway, which would give those
# points a dead gradient. Derived from k_int, so it tracks lam.
SHELL_DEPTH = 15.0 / abs(k_int_c.imag)

# Positive-side cap on phi inside the ansatz, in skin depths. See SkinAnsatz for
# why this must be strictly positive rather than 0.
PHI_CLAMP = 4.0 / abs(k_int_c.imag)

# B_INT band: depends on WHAT net_int represents.
#   ansatz off: net_int is u_int, which oscillates and decays at |k_int|, so the
#     band has to reach that scale.
#   ansatz on: net_int is the ENVELOPE A, and the whole point is that A is
#     smooth -- it varies along Gamma on the scale of the exterior field (1/k)
#     and of the apex (R_APEX), not on the skin depth. Keeping the |k_int| band
#     would invite the net to make A oscillate at 189 rad/unit and hand back
#     exactly the layer the ansatz just removed.
_B_INT_SCALE = k if SKIN_ANSATZ else abs(k_int_c)

# Interior/exterior collocation margins. 0.005 is afm_pinn/afm_pinn.py's value
# and the right one WITHOUT the ansatz. With it the margin has to shrink: 0.005
# is a whole skin depth, so it excluded precisely the region that sets the
# impedance. The ansatz makes points arbitrarily close to Gamma harmless
# (phi -> 0 gives exp -> 1, no stiffness), so the margin only has to keep
# collocation points off the surface itself.
_MARGIN_IN  = 1e-4 if SKIN_ANSATZ else 0.005
_MARGIN_OUT = 0.008

# Give the trunk the signed distance phi(x,y;alpha) as a third input. This
# breaks the strict branch/trunk separation, so it is OFF by default -- but a
# fixed trunk basis cannot follow a moving interface, and this is the standard
# remedy if the clean version underfits near Gamma.
TRUNK_SEES_SDF = False

# NORMALISE THE TRUNK OUTPUT, which removes the DeepONet's scaling degeneracy.
#
# u = sum_i b_i(params) * t_i(x, y) is unchanged by b -> c*b, t -> t/c, and more
# generally by mixing the basis, so the loss has flat valleys in exactly the
# curvature L-BFGS is trying to estimate -- and L-BFGS does most of the work in
# these runs. A LayerNorm on the trunk output (no learnable affine, so the
# checkpoint keys do not change) fixes the scale and mean of t at every point,
# leaving the branch alone to set amplitude and killing that freedom.
#
# Measured on final_4x2's trained DeepONet, which reached relL2 0.086 against
# the plain MLP's 0.035: the 128 modes per component have a participation ratio
# of only 16-34, while the 2500 FDM references are rank-8 to 0.003 -- so the
# width is 4-15x more than is used, and most of it is redundancy of exactly the
# kind this flag removes. Worth pairing with a smaller BRANCH_DIM (a
# CONFIG_OVERRIDES key, no code change needed).
#
# OFF by default: it changes what a checkpoint means, so existing runs stay
# comparable. evaluate_final.build_nets reads TRUNK_NORM from the checkpoint and
# rebuilds the same network, so a model trained with it scores correctly.
TRUNK_NORM = False
# LayerNorm divides by sqrt(sigma^2 + eps), and PyTorch's default eps of 1e-5
# assumes activations of order 1. This trunk's are not: measured on the first
# TRUNK_NORM run, the spread across modes is sigma ~ 1.2e-2 at the median and
# 3.4e-3 at the smallest, so the default eps is ~7% of the variance typically
# and ALL of it at the low end -- the normalisation stops being scale free
# exactly where sigma is small. That run reached relL2 0.0508 at Adam epoch
# 10000, then every term climbed and it ended at 0.298. 1e-10 keeps eps a
# safety floor rather than part of the denominator.
TRUNK_NORM_EPS = 1e-10

# NORMALISE THE BRANCH INSTEAD -- the same degeneracy, removed on the safer side.
#
# Fixing the scale of EITHER factor in u = b(params) . t(x, y) kills the
# b -> c*b, t -> t/c freedom. Doing it on the branch is much better behaved
# here, for two reasons measured on the TRUNK_NORM runs:
#
#   1. LayerNorm's gradient scales like 1/sigma, and the trunk's spread across
#      modes drifts DOWNWARD during training because normalisation divides its
#      scale away: sigma fell from a median of 2.2e-2 (eps 1e-5) to 3.1e-3
#      (eps 1e-10), with minima of 3.4e-3 and 1.5e-4, i.e. gradient
#      amplification up to 217x and 6610x respectively. Runs then sat on a
#      knife edge -- one reached relL2 0.046, the next exploded before epoch
#      2000 and went NaN in L-BFGS on a configuration differing only in the
#      Adam schedule length.
#   2. The PDE residuals differentiate in x, which touches ONLY the trunk. A
#      branch-side normalisation therefore never enters a second derivative,
#      while ||b|| is one number per geometry (measured 2.8-6.6, far from zero)
#      rather than a statistic over every collocation point.
#
# Scaled by sqrt(BRANCH_DIM) so the product keeps roughly its usual size.
BRANCH_NORM = False

FOURIER_DIM   = 2 * (N_FOURIER + N_FOURIER_TIP)
TRUNK_IN_DIM  = FOURIER_DIM + (1 if TRUNK_SEES_SDF else 0)
BRANCH_IN_DIM = PHYS_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")
if DEVICE.type != "cuda":
    # This fallback is silent and expensive: a full run measured ~3.9 s per
    # closure evaluation on CPU (~36 h) against ~0.8 ms of actual fp32 compute
    # on an RTX 3080 Ti. If a GPU was expected, stop and check the install
    # rather than letting this finish.
    print("  " + "!"*68)
    print("  !! CUDA NOT AVAILABLE - running on CPU.")
    print("  !! torch.cuda.is_available() is False, so a CPU-only torch build")
    print("  !! or an unallocated GPU will silently cost ~36 h per run.")
    print(f"  !! torch {torch.__version__}, built for CUDA "
          f"{torch.version.cuda}")
    print("  !! On CPU also set BATCH_ANGLES = False (batching only pays off")
    print("  !! on GPU; it measured 0.92x here) and cut LBFGS_STEPS.")
    print("  " + "!"*68)

# ─────────────────────────────────────────────────────────────────────────────
# Load FDM half-angle sweep
# ─────────────────────────────────────────────────────────────────────────────
FDM_PATH = globals().get("FDM_PATH_OVERRIDE",
                         "final/inputs/fdm_final_sweep.npz")
assert os.path.exists(FDM_PATH), \
    f"FDM sweep not found at '{FDM_PATH}'. Run fdm_afm_tip_sweep.py first."

# -- LAUNCHER OVERRIDES ------------------------------------------------------
# runpy.run_path(init_globals=...) injects names BEFORE this body runs, so a
# plain assignment above would overwrite whatever a launcher passed. Names in
# CONFIG_OVERRIDES are applied HERE instead: after every constant is set, and
# before anything derives from them (the npz load, the train/test handling,
# SINGLE_GEOMETRY, the networks, the samplers, L0).
#
# lam and FDM_PATH are NOT routed through here -- they are read at their own
# point of definition above, because other constants derive from them earlier
# than this block runs. A key that does not already name a module constant
# raises, so a typo in a launcher cannot look like a successful run of the
# wrong configuration.
_cfg_over = dict(globals().get("CONFIG_OVERRIDES", {}))
if _cfg_over:
    for _k in _cfg_over:
        if _k not in globals():
            raise SystemExit(
                f"CONFIG_OVERRIDES: {_k!r} does not name a constant in "
                f"final_deeponet.py -- check the spelling")
    globals().update(_cfg_over)
    assert ARCH in ("deeponet", "mlp"), f"unknown ARCH {ARCH!r}"
    print("  config overrides : "
          + ", ".join(f"{_k}={_cfg_over[_k]!r}" for _k in sorted(_cfg_over)))

_fdm       = np.load(FDM_PATH)
half_angle = _fdm["half_angle"].astype(np.float64)   # [N_PARAM], radians
apex_y_all = _fdm["apex_y"].astype(np.float64)      # [N_PARAM], apex heights
alpha_idx  = _fdm["alpha_idx"]                      # grid indices, for plots
height_idx = _fdm["height_idx"]
ALPHA_DEG_VALS = _fdm["alpha_deg_vals"]
APEX_Y_VALS    = _fdm["apex_y_vals"]
N_ALPHA_G  = len(ALPHA_DEG_VALS)
N_HEIGHT_G = len(APEX_Y_VALS)
alpha_deg  = np.rad2deg(half_angle)
train_idx  = _fdm["train_idx"]
test_idx   = _fdm["test_idx"]
FDM_X      = _fdm["X"]
FDM_Y      = _fdm["Y"]
FDM_N      = FDM_X.shape[0]

# Fields: [N_PARAM, FDM_N, FDM_N]
FDM_scat_re  = _fdm["U_scat_re"]
FDM_scat_im  = _fdm["U_scat_im"]
FDM_total_re = _fdm["U_total_re"]
FDM_total_im = _fdm["U_total_im"]

N_PARAM = len(half_angle)

# ── SINGLE-GEOMETRY OVERRIDE ────────────────────────────────────────────────
# The FDM sweep is loaded whole -- the branch feature normalisation, the grid
# plots and the parameter-index bookkeeping all assume the full 10x10 grid -- but
# training and evaluation are pinned to ONE index. test_idx is that same index,
# so the monitor reports the geometry actually being fitted; there is no
# generalisation claim to make from a single-geometry run and pretending
# otherwise by holding out a different one would only report noise.
SINGLE_PI = None
if SINGLE_GEOMETRY:
    # Prefer the indices final_fdm.py actually solved. Do NOT re-derive the
    # choice from SINGLE_ALPHA_DEG/SINGLE_APEX_Y when the file tells us: the two
    # rules agree only up to float rounding, and 20 deg sits exactly halfway
    # between the 18.89 and 21.11 grid values, so argmin picked different sides
    # in the two files and training ran against an array of zeros.
    if "solved_idx" in _fdm.files:
        _solved = np.asarray(_fdm["solved_idx"]).ravel()
        if len(_solved) == 1:
            SINGLE_PI = int(_solved[0])
        else:                       # full sweep on disk: pick nearest to target
            _d = ((alpha_deg[_solved] - SINGLE_ALPHA_DEG) /
                  max(np.ptp(alpha_deg), 1e-9))**2 + \
                 ((apex_y_all[_solved] - SINGLE_APEX_Y) /
                  max(np.ptp(apex_y_all), 1e-9))**2
            SINGLE_PI = int(_solved[int(np.argmin(_d))])
    else:                           # older sweep without the key: whole grid
        _d = ((alpha_deg - SINGLE_ALPHA_DEG) / max(np.ptp(alpha_deg), 1e-9))**2 + \
             ((apex_y_all - SINGLE_APEX_Y) / max(np.ptp(apex_y_all), 1e-9))**2
        SINGLE_PI = int(np.argmin(_d))

    # Guard: an unsolved index is all zeros, and every downstream number
    # (relL2, MON_TOT_REF, the amplitude target) silently becomes meaningless.
    _ref_norm = float(np.linalg.norm(FDM_scat_re[SINGLE_PI]) +
                      np.linalg.norm(FDM_scat_im[SINGLE_PI]))
    assert _ref_norm > 0.0, (
        f"FDM reference for pi={SINGLE_PI} is all zeros -- that grid point was "
        f"not solved. Re-run final_fdm.py with SOLVE_ONE_NEAREST covering it, "
        f"or set SOLVE_ONE_NEAREST = None for the full sweep.")

    train_idx = np.array([SINGLE_PI])
    test_idx  = np.array([SINGLE_PI])

# consistency check against the sweep's parameters
# meta = [N_RE, N_IM, lam, k, TIP_X, R_APEX, SUB_N_RE, SUB_N_IM, Y_SUB,
#         N_ALPHA, N_HEIGHT, theta_inc_deg]   (see final_fdm.py)
_meta = _fdm["meta"]
assert np.allclose(_meta[:3], [N_RE, N_IM, lam]), \
    f"Material/wave mismatch vs FDM sweep: sweep meta = {_meta[:3]}"
assert np.allclose(_meta[4:6], [TIP_X, R_APEX]), \
    f"Tip geometry mismatch vs FDM sweep: sweep meta = {_meta[4:6]}"
assert np.allclose(_meta[6:9], [SUB_N_RE, SUB_N_IM, Y_SUB]), \
    f"Substrate mismatch vs FDM sweep: sweep meta = {_meta[6:9]}"

print("=" * 66)
print("Final DeepONet — AFM Tip + Substrate (TM, half-angle x height sweep)")
print("=" * 66)
print(f"  n (fixed)      = {N_RE} + {N_IM}i,   λ (fixed) = {lam}")
print(f"  tip            : axis x={TIP_X}, R={R_APEX}  (apex height swept)")
print(f"  substrate      : n={SUB_N_RE}+{SUB_N_IM}i at y={Y_SUB},  "
      f"|r_TM|={abs(r_TM):.4f}")
print(f"  sweep grid     : {N_ALPHA_G} half-angles × {N_HEIGHT_G} heights "
      f"= {N_PARAM} combos")
print(f"  gap range      : {apex_y_all.min()-Y_SUB:.3f} .. "
      f"{apex_y_all.max()-Y_SUB:.3f}   "
      f"(background standing-wave period {np.pi/abs(ky_inc):.3f})")
print(f"  half-angles    : {N_PARAM} values ∈ "
      f"[{alpha_deg.min():.1f}°, {alpha_deg.max():.1f}°]")
print(f"  Train / Test   : {len(train_idx)} / {len(test_idx)}")
print(f"  test angles    = {[f'{alpha_deg[p]:.2f}°' for p in test_idx]}")
print(f"  Batch size (α) : {N_PARAM_BATCH}")
if ARCH == "mlp":
    print(f"  ARCH = mlp: u = MLP([fourier(x,y), params]) -> (Re, Im), "
          f"{MLP_HIDDEN}x{MLP_NEURONS} (in {FOURIER_DIM + PHYS_DIM})")
    print(f"    control for the DeepONet -- no rank-{BRANCH_DIM//2} "
          f"factorisation, no branch/trunk scaling degeneracy")
else:
    print(f"  ARCH = deeponet: branch {BRANCH_HIDDEN}x{BRANCH_NEURONS} "
          f"(in {BRANCH_IN_DIM}) | trunk {TRUNK_HIDDEN}x{TRUNK_NEURONS} "
          f"(in {TRUNK_IN_DIM}) | dot {BRANCH_DIM} "
          f"({BRANCH_DIM//2} basis fns per component)")
print(f"  Fourier {N_FOURIER} @ |k| + {N_FOURIER_TIP} @ tip scale "
      f"(K_TIP={K_TIP:.1f}) | TRUNK_SEES_SDF={TRUNK_SEES_SDF}"
      f" | TRUNK_NORM={TRUNK_NORM} | BRANCH_NORM={BRANCH_NORM}")
if N_PHYS_EXTRA:
    _ba = _phys_span(float(half_angle.min()), float(half_angle.max()),
                     PHYS_SPAN_ALPHA)
    _bh = _phys_span(float(apex_y_all.min()), float(apex_y_all.max()),
                     PHYS_SPAN_APEX)
    print(f"  physics also at {N_PHYS_EXTRA} DATA-FREE parameter draws per point "
          f"set (weight {PHYS_EXTRA_WEIGHT}), drawn in "
          f"alpha [{np.rad2deg(_ba[0]):.1f}, {np.rad2deg(_ba[1]):.1f}] deg "
          f"(span {PHYS_SPAN_ALPHA:g}), "
          f"apex_y [{_bh[0]:.3f}, {_bh[1]:.3f}] "
          f"= gap [{_bh[0]-Y_SUB:.3f}, {_bh[1]-Y_SUB:.3f}] "
          f"(span {PHYS_SPAN_APEX:g})")
print(f"  |k_int²| = {k_int_sq_mag:.1f}  (interior PDE non-dim scale)")
if SKIN_ANSATZ:
    print(f"  SKIN_ANSATZ ON: u_int = A*exp(-i k_int phi); net_int learns A")
    print(f"    skin depth 1/Im(k_int) = {1/abs(k_int_c.imag):.5f}, "
          f"shell = {SHELL_DEPTH:.4f} ({SHELL_DEPTH*abs(k_int_c.imag):.0f} skin "
          f"depths), margin = {_MARGIN_IN:.0e}")
    print(f"    B_int band = {_B_INT_SCALE:.2f} (envelope scale) "
          f"+ {K_TIP:.1f} (tip scale)")
else:
    print(f"  SKIN_ANSATZ OFF: net_int learns u_int directly, "
          f"B_int band = {_B_INT_SCALE:.1f}")
if SINGLE_GEOMETRY:
    print("-" * 66)
    print(f"  SINGLE_GEOMETRY: training on pi={SINGLE_PI} only "
          f"(α={alpha_deg[SINGLE_PI]:.2f}°, apex_y={apex_y_all[SINGLE_PI]:.4f}, "
          f"gap={apex_y_all[SINGLE_PI]-Y_SUB:.4f})")
    print(f"  nearest grid point to afm_pinn's tip "
          f"(α={SINGLE_ALPHA_DEG:.1f}°, apex_y={SINGLE_APEX_Y})")
    print("  every other knob matched to afm_pinn/afm_pinn.py, so the DeepONet")
    print("  architecture is the only remaining difference. ~0.08 => the")
    print("  architecture is fine and the problem is parametric; a stall => the")
    print("  rank-%d separable form cannot do this even at fixed geometry."
          % (BRANCH_DIM // 2))
print("=" * 66)

# ─────────────────────────────────────────────────────────────────────────────
# Alpha-parametrised signed distance function (numpy + torch)
# ─────────────────────────────────────────────────────────────────────────────
def _alpha_consts(alpha, apex_y):
    """Geometry constants for (half-angle, apex height). apex_y is the LOWEST
    point of the tip, so the apex sphere centre sits at apex_y + R_APEX."""
    sin_a = float(np.sin(alpha))
    cos_a = float(np.cos(alpha))
    apex_centre_y = apex_y + R_APEX
    cone_vertex_y = apex_centre_y - R_APEX / sin_a
    tangent_y     = apex_centre_y - R_APEX * sin_a
    return sin_a, cos_a, cone_vertex_y, tangent_y, apex_centre_y


def phi_np(x, y, alpha, apex_y):
    """Signed distance to the tip with half-angle alpha, apex at apex_y."""
    sin_a, cos_a, cone_vertex_y, tangent_y, apex_centre_y = _alpha_consts(alpha, apex_y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q_x    = np.abs(x - TIP_X)
    d_cone = q_x * cos_a - (y - cone_vertex_y) * sin_a
    d_apex = np.sqrt((x - TIP_X)**2 + (y - apex_centre_y)**2) - R_APEX
    return np.where(y < tangent_y, d_apex, d_cone)


def phi_torch(xy, alpha, apex_y):
    """Signed distance (torch). xy: [N,2] → [N,1]. Autograd-friendly."""
    sin_a, cos_a, cone_vertex_y, tangent_y, apex_centre_y = _alpha_consts(alpha, apex_y)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    q_x    = torch.abs(x - TIP_X)
    d_cone = q_x * cos_a - (y - cone_vertex_y) * sin_a
    d_apex = torch.sqrt((x - TIP_X)**2 + (y - apex_centre_y)**2 + 1e-12) - R_APEX
    return torch.where(y < tangent_y, d_apex, d_cone)


def phi_torch_vec(xy, geom):
    """Signed distance for PER-POINT geometry. xy: [N,2], geom: [N,2] or [1,2]
    holding (alpha, apex_y). Broadcasts, so a single row covers a whole block.

    phi_torch above takes python floats and is the right thing when one geometry
    governs the whole tensor; this one exists because the batched path
    concatenates several geometries into one tensor and the skin ansatz needs
    phi at every point."""
    alpha  = geom[:, 0:1]
    apex_y = geom[:, 1:2]
    sin_a  = torch.sin(alpha)
    cos_a  = torch.cos(alpha)
    apex_centre_y = apex_y + R_APEX
    cone_vertex_y = apex_centre_y - R_APEX / sin_a
    tangent_y     = apex_centre_y - R_APEX * sin_a
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    d_cone = torch.abs(x - TIP_X) * cos_a - (y - cone_vertex_y) * sin_a
    d_apex = torch.sqrt((x - TIP_X)**2 + (y - apex_centre_y)**2 + 1e-12) - R_APEX
    return torch.where(y < tangent_y, d_apex, d_cone)


def geom_row(alpha, apex_y):
    """[1,2] geometry tensor; broadcasts against any point block."""
    return torch.tensor([[float(alpha), float(apex_y)]],
                        dtype=torch.float32, device=DEVICE)


def outward_normal(xy, alpha, apex_y):
    """Outward unit normal via autograd of phi_torch. xy requires_grad."""
    d = phi_torch(xy, alpha, apex_y)
    g = torch.autograd.grad(d.sum(), xy, create_graph=True)[0]
    norm = torch.sqrt((g**2).sum(dim=1, keepdim=True) + 1e-12)
    return g / norm


def tip_mask(Xg, Yg, alpha, apex_y):
    return phi_np(Xg, Yg, alpha, apex_y) < 0.0


def tip_outline_xy(alpha, apex_y, y_top=1.0, n=200):
    sin_a, cos_a, cone_vertex_y, tangent_y, apex_centre_y = _alpha_consts(alpha, apex_y)
    yc = apex_centre_y
    ys = np.linspace(tangent_y, y_top, n)
    half_w  = (ys - cone_vertex_y) * np.tan(alpha)
    x_right = TIP_X + half_w
    x_left  = TIP_X - half_w
    a0 = -np.pi/2 - (np.pi/2 - alpha)
    a1 = -np.pi/2 + (np.pi/2 - alpha)
    ang = np.linspace(a0, a1, n)
    ax_arc = TIP_X + R_APEX * np.cos(ang)
    ay_arc = yc     + R_APEX * np.sin(ang)
    xs = np.concatenate([x_right[::-1], ax_arc, x_left])
    yy = np.concatenate([ys[::-1],      ay_arc, ys])
    return xs, yy

# ─────────────────────────────────────────────────────────────────────────────
# Physics feature normalisation — computed over all angles in the sweep.
# Smooth, shape-descriptive features of alpha (cf. the stretch features of
# ellipse_pinn.py): angle, its trig, and the shank half-width at the top of
# the physical box, which controls how much material sits in the domain.
# ─────────────────────────────────────────────────────────────────────────────
def _raw_features(alpha, apex_y):
    """Branch input: smooth, shape-descriptive functions of BOTH parameters.

    The five alpha features are inherited from variable_tips. The three height
    features are chosen for what actually controls the physics:
      * gap                 the tip-sample separation itself
      * log(gap)            the near-field varies far more steeply at small gap
                            than at large, so the log gives the branch a
                            well-conditioned coordinate near contact (gap 0.01
                            to 0.20 is a factor of 20)
      * cos(2*ky*gap)       the background is a standing wave in y of period
                            pi/|ky|; this is the phase of that interference at
                            the apex, i.e. whether the apex sits in a bright or
                            dark fringe. Without it the branch would have to
                            rediscover a known oscillation from samples.
    """
    _, _, cone_vertex_y, _, _ = _alpha_consts(alpha, apex_y)
    gap = apex_y - Y_SUB
    return np.array([
        alpha,
        np.sin(alpha),
        np.cos(alpha),
        np.tan(alpha),
        (1.0 - cone_vertex_y) * np.tan(alpha),   # shank half-width at y = 1
        gap,
        np.log(max(gap, 1e-6)),
        np.cos(2.0 * abs(ky_inc) * gap),
    ], dtype=np.float32)


_all_feats = np.stack([_raw_features(float(a_), float(h_))
                       for a_, h_ in zip(half_angle, apex_y_all)])
FEAT_MIN = _all_feats.min(axis=0)
FEAT_MAX = _all_feats.max(axis=0)


def param_to_features(alpha, apex_y):
    """Normalised [1, PHYS_DIM] feature tensor on DEVICE, values in [-1,1].
    Works for arbitrary alpha within the sweep's feature ranges, so the
    trained model can be queried at unseen half-angles."""
    raw   = _raw_features(float(alpha), float(apex_y))
    normd = 2.0 * (raw - FEAT_MIN) / (FEAT_MAX - FEAT_MIN + 1e-10) - 1.0
    return torch.tensor(normd, device=DEVICE).unsqueeze(0)

# ─────────────────────────────────────────────────────────────────────────────
# Fourier feature embeddings — k fixed → single-scale (as in afm_tip_pinn.py)
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(1)
# Scaled by |k_sub| (= 1.51k), not k: ONE exterior net carries the scattered
# field in both air and substrate, so the trunk basis must resolve the faster
# substrate oscillation (afm_pinn/afm_pinn.py).
B_ext = torch.cat([torch.randn(2, N_FOURIER,     device=DEVICE) * abs(k_sub_c),
                   torch.randn(2, N_FOURIER_TIP, device=DEVICE) * K_TIP], dim=1)

# _B_INT_SCALE is set with SKIN_ANSATZ near the top -- see the note there on why
# the interior band has to change when net_int switches to learning A.
B_int = torch.cat([torch.randn(2, N_FOURIER,     device=DEVICE) * _B_INT_SCALE,
                   torch.randn(2, N_FOURIER_TIP, device=DEVICE) * K_TIP], dim=1)

# Warm start, part 1 (see INIT_WEIGHTS): refuse an incompatible checkpoint now,
# before anything expensive, and take its Fourier matrices before torch.compile
# traces the embeddings. On the device that trained it the copy changes nothing,
# since B is drawn from the same seed there; elsewhere it is what makes the
# loaded weights mean the same basis. The weights themselves load after L0.
_INIT_CK = None
if INIT_WEIGHTS:
    if not os.path.exists(INIT_WEIGHTS):
        raise SystemExit(f"INIT_WEIGHTS not found: {INIT_WEIGHTS}")
    _INIT_CK = torch.load(INIT_WEIGHTS, map_location=DEVICE, weights_only=False)
    _ihp = _INIT_CK.get('hyperparams', {})
    for _key, _mine in (('ARCH', ARCH), ('SKIN_ANSATZ', SKIN_ANSATZ), ('lam', lam)):
        if _key in _ihp and _ihp[_key] != _mine:
            raise SystemExit(f"INIT_WEIGHTS {INIT_WEIGHTS}: {_key} = "
                             f"{_ihp[_key]!r}, but this run has {_mine!r}")
    if not (np.allclose(_INIT_CK['FEAT_MIN'], FEAT_MIN) and
            np.allclose(_INIT_CK['FEAT_MAX'], FEAT_MAX)):
        raise SystemExit(f"INIT_WEIGHTS {INIT_WEIGHTS} was trained on a different "
                         "parameter sweep: its branch feature normalisation "
                         "differs from this run's")
    if (tuple(_INIT_CK['B_ext'].shape) != tuple(B_ext.shape) or
            tuple(_INIT_CK['B_int'].shape) != tuple(B_int.shape)):
        raise SystemExit(f"INIT_WEIGHTS {INIT_WEIGHTS}: Fourier matrices are "
                         f"{tuple(_INIT_CK['B_ext'].shape)}, this run's are "
                         f"{tuple(B_ext.shape)}")
    _dB = max(float((_INIT_CK['B_ext'].to(DEVICE) - B_ext).abs().max()),
              float((_INIT_CK['B_int'].to(DEVICE) - B_int).abs().max()))
    B_ext = _INIT_CK['B_ext'].to(DEVICE).clone()
    B_int = _INIT_CK['B_int'].to(DEVICE).clone()
    print(f"  warm start from : {INIT_WEIGHTS}")
    print(f"    Fourier matrices taken from it (max |difference| from this "
          f"run's own draw: {_dB:.1e})")

# FOURIER_DIM / TRUNK_IN_DIM / BRANCH_IN_DIM are defined with the other
# architecture constants above, since the startup summary prints them.


# Trunk embeddings take ONLY coordinates -- the parameters go to the branch.
def embed_ext(xy, phys_feats=None):
    proj    = xy @ B_ext
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def embed_int(xy, phys_feats=None):
    proj    = xy @ B_int
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

# ─────────────────────────────────────────────────────────────────────────────
# Networks — parametric MLPs (ellipse_pinn.py architecture)
# ─────────────────────────────────────────────────────────────────────────────
def _mlp(d_in, d_hidden, n_hidden, d_out):
    layers = [nn.Linear(d_in, d_hidden), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(d_hidden, d_hidden), nn.Tanh()]
    layers += [nn.Linear(d_hidden, d_out)]
    net = nn.Sequential(*layers)
    for m in net.modules():                 # same init as the PINN
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)
    return net


class BranchNet(nn.Module):
    """Parameter vector -> BRANCH_DIM coefficients."""
    def __init__(self):
        super().__init__()
        self.net = _mlp(BRANCH_IN_DIM, BRANCH_NEURONS, BRANCH_HIDDEN, BRANCH_DIM)

    def forward(self, phys_feats):
        return self.net(phys_feats)          # [1 or M, BRANCH_DIM]


class TrunkNet(nn.Module):
    """(x, y) -> BRANCH_DIM spatial basis functions."""
    def __init__(self, embed_fn, sdf_fn=None):
        super().__init__()
        self.embed  = embed_fn
        self.sdf_fn = sdf_fn
        self.net = _mlp(TRUNK_IN_DIM, TRUNK_NEURONS, TRUNK_HIDDEN, BRANCH_DIM)
        # elementwise_affine=False: no parameters, so the state_dict is
        # identical with the flag on or off and checkpoints stay loadable.
        self.norm = (nn.LayerNorm(BRANCH_DIM, eps=TRUNK_NORM_EPS,
                                  elementwise_affine=False)
                     if TRUNK_NORM else None)

    def forward(self, xy, alpha=None):
        e = self.embed(xy)
        if TRUNK_SEES_SDF:
            # phi is O(0.1) while the Fourier features are O(1); scaling by k
            # puts it on a comparable footing
            e = torch.cat([e, self.sdf_fn(xy, alpha) * k], dim=-1)
        t = self.net(e)                      # [M, BRANCH_DIM]
        if self.norm is not None:
            t = self.norm(t)                 # see TRUNK_NORM
        return t


class DeepONet(nn.Module):
    """u(params, x, y) = sum_i branch_i(params) * trunk_i(x, y).

    BRANCH_DIM is split in half: the first half is dotted for Re(u), the second
    for Im(u), as in mie_deeponet_pml.py.

    phys_feats may be [1, PHYS_DIM] (one angle, broadcast over all points) or
    [M, PHYS_DIM] (the batched path, where consecutive blocks of rows belong to
    different half-angles). Both work unchanged: b * t broadcasts in the first
    case and multiplies row-wise in the second, which is exactly the required
    pairing of each point with its own parameters."""
    def __init__(self, embed_fn, sdf_fn=None):
        super().__init__()
        self.branch  = BranchNet()
        self.trunk   = TrunkNet(embed_fn, sdf_fn)
        self.bias_re = nn.Parameter(torch.zeros(1))
        self.bias_im = nn.Parameter(torch.zeros(1))
        self.half    = BRANCH_DIM // 2

    def forward(self, phys_feats, xy, alpha=None):
        b  = self.branch(phys_feats)         # [1 or M, BRANCH_DIM]
        if BRANCH_NORM:                      # see the note at BRANCH_NORM
            b = b * (BRANCH_DIM ** 0.5
                     / b.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        t  = self.trunk(xy, alpha)           # [M, BRANCH_DIM]
        bt = b * t
        Re = bt[:, :self.half].sum(dim=-1, keepdim=True) + self.bias_re
        Im = bt[:, self.half:].sum(dim=-1, keepdim=True) + self.bias_im
        return Re, Im


class MLPNet(nn.Module):
    """u(params, x, y) = MLP([fourier(x,y), params]) -> (Re, Im).

    The control for the DeepONet: same Fourier embedding, same depth and width
    as the trunk, same everything downstream -- but the parameters enter as
    ordinary inputs instead of multiplying a fixed spatial basis, so there is no
    rank-BRANCH_DIM/2 factorisation and no b/t scaling degeneracy.

    Presents the identical (phys_feats, xy, geom) signature as DeepONet, so it
    drops into every call site including the PML branch switch and the skin
    wrapper. phys_feats arrives as [1, PHYS_DIM] (one geometry broadcast over
    the block) or [M, PHYS_DIM] (the batched path); expand handles both.

    Depth/width default to the trunk's, which makes the comparison "same spatial
    network, different head". That leaves the MLP with fewer parameters, since
    the DeepONet also carries a branch -- see the count printed at startup. For
    a parameter-matched comparison instead, set MLP_NEURONS = 166 (solves
    3W^2 + 206W + 2 = 116418 at MLP_HIDDEN = 4)."""
    def __init__(self, embed_fn):
        super().__init__()
        self.embed = embed_fn
        self.net = _mlp(FOURIER_DIM + PHYS_DIM, MLP_NEURONS, MLP_HIDDEN, 2)

    def forward(self, phys_feats, xy, geom=None):
        e = self.embed(xy)
        p = phys_feats if phys_feats.shape[0] == e.shape[0] \
            else phys_feats.expand(e.shape[0], -1)
        out = self.net(torch.cat([e, p], dim=-1))
        return out[:, 0:1], out[:, 1:2]


class SkinAnsatz(nn.Module):
    """u_int = A(x,y) * exp(-i k_int phi),  A supplied by the wrapped DeepONet.

    With psi = -i k_int phi and k_int = k(N_RE + i N_IM):
        Re psi =  k*N_IM*phi        (phi < 0 inside, so this DECAYS)
        Im psi = -k*N_RE*phi
    so exp(psi) = e^{k N_IM phi} * (cos(k N_RE phi) - i sin(k N_RE phi)).

    On Gamma phi = 0, so exp(psi) = 1 and the wrapped net's raw output IS the
    surface field -- the interface terms need no change at all. And since
    grad(phi) = n with |grad phi| = 1, autograd yields
        dn u_int = dn A - i k_int A       on Gamma
    without anything having to resolve the layer.

    The wrapper is transparent to the optimiser and to checkpointing: it holds
    no parameters of its own and `.don` is saved directly, so the state_dict
    keys are identical with the ansatz on or off."""
    def __init__(self, don):
        super().__init__()
        self.don = don

    def forward(self, phys_feats, xy, geom=None):
        A_r, A_i = self.don(phys_feats, xy)
        if geom is None:
            raise RuntimeError(
                "SKIN_ANSATZ is on but net_int was called without a geometry; "
                "pass geom_row(alpha, apex_y) (or a per-point [M,2] tensor) as "
                "the third argument.")
        # Cap phi on the POSITIVE side, and not at zero. The cap exists because
        # the monitor, save_frame and evaluate_fields all call net_int on the
        # FULL grid, where phi > 0 would make exp(k*N_IM*phi) overflow float32
        # (phi > 0.47 at lam = 1.0).
        #
        # It must NOT be at 0: clamp has zero gradient beyond its bound, and the
        # interface points are projected to phi = +-1e-8, so about half of them
        # sit a hair on the positive side. Capping there would silently drop the
        # -i k_int A term from dn u_int on those points -- i.e. corrupt the flux
        # BC on half of Gamma, which is the one condition this ansatz exists to
        # get right. Verified: with a constant A the cap at 0 gives a mean
        # dn u/u of 0.61 * (-i k_int) while the median is exact.
        #
        # PHI_CLAMP = 4 skin depths is ~1300x the projection error and keeps the
        # factor below e^4, so gradients are exact everywhere that matters and
        # nothing downstream sees a large number.
        phi = torch.clamp(phi_torch_vec(xy, geom), max=PHI_CLAMP)
        amp = torch.exp(k * N_IM * phi)
        ph  = k * N_RE * phi
        E_r =  amp * torch.cos(ph)
        E_i = -amp * torch.sin(ph)
        return A_r * E_r - A_i * E_i, A_r * E_i + A_i * E_r


def _make_net(embed_fn):
    if ARCH == "mlp":
        return MLPNet(embed_fn).to(DEVICE)
    return DeepONet(embed_fn, phi_torch).to(DEVICE)


net_int  = _make_net(embed_int)   # total field inside tip (envelope A if ansatz)
net_scat = _make_net(embed_ext)   # scattered field outside
_don_int = net_int                # the parameterised module, for saving
if SKIN_ANSATZ:
    net_int = SkinAnsatz(_don_int).to(DEVICE)
# ── MERGED EXTERIOR (afm_pinn/afm_pinn.py) ──────────────────────────────────
# One network for the scattered field in BOTH air and substrate, mirroring the
# FDM's single unknown v with piecewise k^2. Legitimate for s-polarisation:
# u_scat and du_scat/dn are continuous across y = Y_SUB (t = 1 + r exactly), so
# the field is C^1 there and one smooth net represents it; only the second
# derivative kinks, and the samplers keep margins off the interface line.
# Splitting it into two networks is what caused the zero-solution collapse in
# sample/, so this is deliberately NOT two nets.
net_sub = net_scat


def raw_module(m):
    """The uncompiled module. torch.compile wraps in an OptimizedModule whose
    state_dict keys are prefixed '_orig_mod.', which would silently break
    checkpoint loading, so always save/load through this."""
    return getattr(m, "_orig_mod", m)


# torch.compile: on a launch-bound workload 'reduce-overhead' (CUDA graphs) can
# collapse the many tiny kernels this loss issues. OFF by default because it is
# unverified on the target GPU and because double-backward (create_graph=True)
# through a compiled module is a known rough edge -- it may error or silently
# fall back to eager, and warm-up costs minutes. To try it, set True and compare
# the elapsed time printed at Adam epoch 250 against a run with it False.
# TF32 matmuls (Ampere+). torch.compile warns that these are available, and they
# are ~8x the fp32 matmul throughput -- but LEFT OFF deliberately. TF32 keeps
# only a 10-bit mantissa, i.e. ~1e-3 relative precision against fp32's ~1e-7,
# and this loss lives on cancellations:
#   - the interior residual cancels lap(u_int) ~ 1.4e3 against k_int_sq*u ~ 1.4e5
#   - field continuity needs a residual ~0.02 out of O(1) quantities
#   - the PML residual currently sits at ~1.5e-3 raw, which is the SAME order as
#     TF32's own noise -- enabling it risks flooring the one term that took the
#     most work (W_PML = 20) to get converging at all.
# Worth testing once the run is otherwise stable: flip this and check whether
# l_pde_int and l_pml stop improving. Only helps if the workload is
# matmul-bound, which it was not before torch.compile.
USE_TF32 = False
if USE_TF32:
    torch.set_float32_matmul_precision('high')
    print("  TF32 matmuls enabled (float32_matmul_precision='high')")

USE_COMPILE = True
if USE_COMPILE:
    # AOTAutograd donates (frees) intermediate buffers during backward, which is
    # incompatible with create_graph=True -- and every PDE residual here needs
    # second derivatives, so the compiled backward raises:
    #   "compiled with non-empty donated buffers which requires
    #    create_graph=False and retain_graph=False"
    # Turning donation off is what makes compile usable for a PINN at all.
    try:
        import torch._functorch.config as _ft_config
        if hasattr(_ft_config, "donated_buffer"):
            _ft_config.donated_buffer = False
            print("  torch._functorch.config.donated_buffer = False "
                  "(required for create_graph=True)")
        else:
            print("  WARNING: torch._functorch.config.donated_buffer not found; "
                  "compile may fail on the double-backward")
    except Exception as _cfg_exc:
        print(f"  WARNING: could not disable donated_buffer: {_cfg_exc}")
    # dynamic=True is essential here, not optional: the PML branch subsets are
    # sized by how many band points fall inside the shank, which differs per
    # half-angle (170-500 of 3000) and changes at every resample. With static
    # shapes torch.compile would recompile for each new size and thrash.
    # mode='default', NOT 'reduce-overhead'. The latter's whole benefit is CUDA
    # graphs, which require STATIC shapes and so cannot apply alongside the
    # dynamic=True we need -- asking for it would silently get us nothing while
    # implying otherwise. Inductor's fusion still cuts the kernel count.
    #
    # To actually unlock CUDA graphs (the big win for a launch-bound loss like
    # this one) the varying tensor would have to go: pad each PML branch to a
    # fixed size and mask, so every shape is constant. That reintroduces a
    # little of the waste the branch split removed, but far less, and it is the
    # route worth taking if compile alone is disappointing.
    _cmode   = "default"
    net_int  = torch.compile(net_int,  mode=_cmode, dynamic=True)
    net_scat = torch.compile(net_scat, mode=_cmode, dynamic=True)
    print(f"  torch.compile enabled (mode={_cmode}, dynamic=True)")

    # Force compilation NOW, exercising the second-derivative path, which is the
    # known rough edge for compiled modules. torch.compile is lazy, so without
    # this the first failure would surface deep in training; and on failure we
    # fall back to eager rather than losing the run. (This could not be
    # validated locally: inductor's CPU backend needs MSVC, absent on the dev
    # box, so the fallback below is also what exercises that path there.)
    try:
        _w_pf = torch.zeros(1, PHYS_DIM, device=DEVICE)
        _w_xy = torch.rand(64, 2, device=DEVICE, requires_grad=True)
        # net_int needs a geometry once SKIN_ANSATZ is on; net_scat ignores the
        # third argument, so one call shape covers both.
        _w_gm = geom_row(half_angle[0], apex_y_all[0])
        for _w_net in (net_int, net_scat):
            _w_r, _w_i = _w_net(_w_pf, _w_xy, _w_gm)
            _w_g  = torch.autograd.grad(_w_r.sum(), _w_xy, create_graph=True)[0]
            _w_gg = torch.autograd.grad(_w_g[:, 0].sum(), _w_xy,
                                        create_graph=True)[0]
            _w_gg.sum().backward()
        print("  torch.compile warm-up OK (double-backward compiled)")
    except Exception as _w_exc:
        net_int, net_scat = raw_module(net_int), raw_module(net_scat)
        USE_COMPILE = False
        print(f"  torch.compile FAILED -> falling back to eager: "
              f"{type(_w_exc).__name__}: {str(_w_exc).splitlines()[-1][:160]}")
    finally:
        for _w_p in list(net_int.parameters()) + list(net_scat.parameters()):
            _w_p.grad = None          # warm-up must not seed the real gradients

n_int_p  = sum(p.numel() for p in net_int.parameters())
n_scat_p = sum(p.numel() for p in net_scat.parameters())
print(f"\n  net_int  parameters : {n_int_p:,}")
print(f"  net_scat parameters : {n_scat_p:,}")

# The other architecture's count, so a run's log states its own handicap rather
# than leaving the comparison to be reconstructed later.
def _mlp_params(d_in, d_h, n_h, d_out):
    p = d_in * d_h + d_h
    for _ in range(n_h - 1):
        p += d_h * d_h + d_h
    return p + d_h * d_out + d_out

_p_don = (_mlp_params(TRUNK_IN_DIM, TRUNK_NEURONS, TRUNK_HIDDEN, BRANCH_DIM)
          + _mlp_params(BRANCH_IN_DIM, BRANCH_NEURONS, BRANCH_HIDDEN, BRANCH_DIM)
          + 2)
_p_mlp = _mlp_params(FOURIER_DIM + PHYS_DIM, MLP_NEURONS, MLP_HIDDEN, 2)
# Width that would give parity, solved rather than quoted: the input dimension
# moves whenever a Fourier band is added, and a hard-coded number goes stale
# silently (it already did once -- 166 was right before the tip-scale band).
_w_parity = next(w for w in range(8, 4096)
                 if _mlp_params(FOURIER_DIM + PHYS_DIM, w, MLP_HIDDEN, 2) >= _p_don)
print(f"  per-net comparison  : deeponet {_p_don:,}  vs  mlp {_p_mlp:,}  "
      f"({_p_mlp / _p_don:.2f}x)")
if ARCH == "mlp" and abs(_p_mlp / _p_don - 1.0) > 0.1:
    print(f"    NOTE this MLP has {_p_mlp/_p_don:.2f}x the DeepONet's parameters. "
          f"For parity set MLP_NEURONS = {_w_parity}.")

# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────
# ── Background field (Fresnel, no tip) ──────────────────────────────────────
# Phase referenced to the interface, matching final_fdm.py exactly. The AIR
# branch is what the tip surface ever sees: the tip's lowest point is apex_y
# >= 0.41 > Y_SUB, so the whole scatterer sits in air.
def u_bg_air_re(xy):
    ph_i = kx_inc * xy[:, 0:1] + ky_inc * (xy[:, 1:2] - Y_SUB)
    ph_r = kx_inc * xy[:, 0:1] - ky_inc * (xy[:, 1:2] - Y_SUB)
    return torch.cos(ph_i) + (r_re * torch.cos(ph_r) - r_im * torch.sin(ph_r))

def u_bg_air_im(xy):
    ph_i = kx_inc * xy[:, 0:1] + ky_inc * (xy[:, 1:2] - Y_SUB)
    ph_r = kx_inc * xy[:, 0:1] - ky_inc * (xy[:, 1:2] - Y_SUB)
    return torch.sin(ph_i) + (r_re * torch.sin(ph_r) + r_im * torch.cos(ph_r))

def u_bg_sub_re(xy):
    s   = xy[:, 1:2] - Y_SUB
    amp = torch.exp(-ky_sub_im * s)
    ph  = kx_inc * xy[:, 0:1] + ky_sub_re * s
    return amp * (t_re * torch.cos(ph) - t_im * torch.sin(ph))

def u_bg_sub_im(xy):
    s   = xy[:, 1:2] - Y_SUB
    amp = torch.exp(-ky_sub_im * s)
    ph  = kx_inc * xy[:, 0:1] + ky_sub_re * s
    return amp * (t_re * torch.sin(ph) + t_im * torch.cos(ph))

# Normal derivative of the AIR background, used by the tip flux condition.
# d/dn[exp(i k.r)] = i(k.n)exp(i k.r) for each of the two plane waves; the
# reflected one has k_r = (kx, -ky).
def du_bg_air_dn_re(xy, nxy):
    ph_i = kx_inc * xy[:, 0:1] + ky_inc * (xy[:, 1:2] - Y_SUB)
    ph_r = kx_inc * xy[:, 0:1] - ky_inc * (xy[:, 1:2] - Y_SUB)
    kn_i = kx_inc * nxy[:, 0:1] + ky_inc * nxy[:, 1:2]
    kn_r = kx_inc * nxy[:, 0:1] - ky_inc * nxy[:, 1:2]
    d_i  = -kn_i * torch.sin(ph_i)
    d_r  = kn_r * (-r_re * torch.sin(ph_r) - r_im * torch.cos(ph_r))
    return d_i + d_r

def du_bg_air_dn_im(xy, nxy):
    ph_i = kx_inc * xy[:, 0:1] + ky_inc * (xy[:, 1:2] - Y_SUB)
    ph_r = kx_inc * xy[:, 0:1] - ky_inc * (xy[:, 1:2] - Y_SUB)
    kn_i = kx_inc * nxy[:, 0:1] + ky_inc * nxy[:, 1:2]
    kn_r = kx_inc * nxy[:, 0:1] - ky_inc * nxy[:, 1:2]
    d_i  = kn_i * torch.cos(ph_i)
    d_r  = kn_r * (r_re * torch.cos(ph_r) - r_im * torch.sin(ph_r))
    return d_i + d_r

def laplacian(u, xy):
    g   = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(g[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return uxx + uyy

def normal_deriv(u, xy, nxy):
    g = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    return g[:, 0:1] * nxy[:, 0:1] + g[:, 1:2] * nxy[:, 1:2]

# ─────────────────────────────────────────────────────────────────────────────
# Collocation samplers — GEOMETRY-DEPENDENT: they take alpha.
# Margins as in afm_tip_pinn.py so PDE points sit strictly on one side of
# the tip surface.
# ─────────────────────────────────────────────────────────────────────────────


def _tip_bbox(alpha, apex_y):
    """Bounding box around the tip (including the shank in the top PML)."""
    _, _, cone_vertex_y, _, _ = _alpha_consts(alpha, apex_y)
    y_hi   = 1.0 + D_PML
    half_w = (y_hi - cone_vertex_y) * np.tan(alpha)
    x_lo = TIP_X - half_w - 0.02
    x_hi = TIP_X + half_w + 0.02
    y_lo = apex_y - R_APEX - 0.02
    return x_lo, x_hi, y_lo, y_hi


def sample_inside(n, alpha, apex_y):
    """Interior collocation points.

    SKIN_ANSATZ on: a graded shell hugging Gamma. Built by pushing surface
    points inward along -n rather than by rejection sampling the bounding box,
    which gives exact control of the depth distribution -- and the depth is
    quadratically graded so roughly half the points land within a quarter of the
    shell, i.e. inside the layer that determines the impedance. Under the old
    uniform-over-the-cross-section sampling with _MARGIN_IN = 0.005, essentially
    no point sat inside the layer at all and the rest were evaluating 0 = 0.

    SKIN_ANSATZ off: the original uniform sampling over the whole tip (incl. the
    shank inside the top PML), unchanged."""
    if SKIN_ANSATZ:
        xy_s, n_s = sample_interface(n, alpha, apex_y)
        u = torch.rand(n, 1, device=DEVICE)
        depth = _MARGIN_IN + (SHELL_DEPTH - _MARGIN_IN) * u**2
        return (xy_s - depth * n_s).detach().requires_grad_(True)

    x_lo, x_hi, y_lo, y_hi = _tip_bbox(alpha, apex_y)
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.column_stack([
            np.random.uniform(x_lo, x_hi, n * 8),
            np.random.uniform(y_lo, y_hi, n * 8),
        ])
        d = phi_np(xy[:, 0], xy[:, 1], alpha, apex_y)
        pts.append(xy[d < -_MARGIN_IN])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)


def sample_air(n, alpha, apex_y):
    """Points outside the tip AND above the substrate: the k_ext^2 region."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 4, 2)
        d = phi_np(xy[:, 0], xy[:, 1], alpha, apex_y)
        pts.append(xy[(d > _MARGIN_OUT) & (xy[:, 1] > Y_SUB + 0.005)])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)


def sample_substrate(n):
    """Points strictly inside the substrate: the k_sub^2 region. Independent of
    the tip parameters — the tip never reaches below Y_SUB (apex_y >= 0.41) —
    so unlike the other samplers this one needs no (alpha, apex_y)."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 3, 2)
        pts.append(xy[xy[:, 1] < Y_SUB - 0.005])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)


def sample_interface(n, alpha, apex_y):
    """
    Points on the tip surface phi = 0 with autograd outward normals.
    Thin-shell rejection + projection along the SDF gradient, exactly as
    afm_tip_pinn.py, but with the alpha-parametrised SDF.
    """
    x_lo, x_hi, y_lo, y_hi = _tip_bbox(alpha, apex_y)
    eps = 0.02
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.column_stack([
            np.random.uniform(x_lo, x_hi, n * 12),
            np.random.uniform(y_lo, y_hi, n * 12),
        ])
        d = phi_np(xy[:, 0], xy[:, 1], alpha, apex_y)
        pts.append(xy[np.abs(d) < eps])
    xy = np.vstack(pts)[:n]

    xy_t = torch.tensor(xy.astype(np.float32), device=DEVICE, requires_grad=True)
    d_t  = phi_torch(xy_t, alpha, apex_y)
    n_t  = outward_normal(xy_t, alpha, apex_y)
    xy_proj = (xy_t - d_t * n_t).detach()

    xy_proj.requires_grad_(True)
    nxy = outward_normal(xy_proj, alpha, apex_y).detach()

    xy_proj = xy_proj.detach().clone().requires_grad_(True)
    return xy_proj, nxy


def sample_interface_bc(n, alpha, apex_y):
    """Surface points for the field and flux continuity TERMS: sample_interface
    without the top IFACE_TOP_EXCLUDE of the shank below the outer boundary.

    Where the shank meets y = 1 + D_PML, continuity contradicts the outer
    Dirichlet condition: Dirichlet wants u_scat = 0 there, continuity wants
    u_scat = u_int - u_bg ~ -u_bg with |u_bg| ~ 1. On the trained final_4 model
    that last 0.1 held ~6% of the surface points but 74-99% of the squared field
    residual (rms 0.36-0.44 against <= 0.04 everywhere else); its gradient was
    opposed by the Dirichlet and PML ones, so l_field could not fall during
    L-BFGS, and each 800-point redraw landed between 0.24x and 2.25x of its mean
    depending on how many corner points it caught. The band is deep in the PML,
    where the field has no physical meaning, and final_fdm.py settles the same
    corner by letting its boundary rows win -- which is what dropping the
    continuity points here does. sample_inside keeps the FULL surface, so the
    interior shell under the band is unchanged."""
    if IFACE_TOP_EXCLUDE <= 0.0:
        return sample_interface(n, alpha, apex_y)
    y_max = 1.0 + D_PML - IFACE_TOP_EXCLUDE
    xys, nrms, have = [], [], 0
    while have < n:
        xy, nxy = sample_interface(n, alpha, apex_y)
        keep = xy[:, 1].detach() <= y_max
        xys.append(xy.detach()[keep])
        nrms.append(nxy[keep])
        have += int(keep.sum())
    return (torch.cat(xys)[:n].clone().requires_grad_(True),
            torch.cat(nrms)[:n])

# ─────────────────────────────────────────────────────────────────────────────
# PML sampler — points shared across the parameter batch. The complex UPML
# coefficient tensors Cx = s_y/s_x, Cy = s_x/s_y, Cx' = -s_y s_x'/s_x²,
# Cy' = -s_x s_y'/s_y², s_x s_y are geometry-independent and precomputed here
# once per sample (ellipse_pinn.py style). What DOES depend on alpha is the
# inside-shank partition, computed per parameter in get_pml_parts below.
# ─────────────────────────────────────────────────────────────────────────────
def _pml_sigma(coord, lo, hi):
    s = np.zeros_like(coord, dtype=float)
    mask_lo = coord < lo
    mask_hi = coord > hi
    s[mask_lo] = PML_SIGMA_MAX * ((lo - coord[mask_lo]) / D_PML) ** PML_POW
    s[mask_hi] = PML_SIGMA_MAX * ((coord[mask_hi] - hi) / D_PML) ** PML_POW
    return s


def _pml_sigma_deriv(coord, lo, hi):
    s = np.zeros_like(coord, dtype=float)
    mask_lo = coord < lo
    mask_hi = coord > hi
    if PML_POW >= 1:
        s[mask_lo] = (PML_POW * PML_SIGMA_MAX / D_PML) * \
                     ((lo - coord[mask_lo]) / D_PML) ** (PML_POW - 1)
        s[mask_hi] = (PML_POW * PML_SIGMA_MAX / D_PML) * \
                     ((coord[mask_hi] - hi) / D_PML) ** (PML_POW - 1)
    return s


def sample_pml(n):
    """Shared PML-band sample: xy (requires_grad), coefficient dict, and the
    raw numpy coordinates (needed for the per-alpha inside masks)."""
    pts = []
    while len(pts) == 0 or sum(len(p) for p in pts) < n:
        xy = np.random.uniform(-D_PML, 1.0 + D_PML, (n * 4, 2))
        in_phys = ((xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) &
                   (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0))
        pts.append(xy[~in_phys])
    xy = np.vstack(pts)[:n]

    sig_x  = _pml_sigma(xy[:, 0], 0.0, 1.0)
    sig_y  = _pml_sigma(xy[:, 1], 0.0, 1.0)
    dsig_x = _pml_sigma_deriv(xy[:, 0], 0.0, 1.0)
    dsig_y = _pml_sigma_deriv(xy[:, 1], 0.0, 1.0)

    # ── MEDIUM-AWARE STRETCH ────────────────────────────────────────────────
    # s = 1 + i*sigma/k_ref, and k_ref must be the LOCAL wavenumber. This used
    # to reference k everywhere, including the substrate -- while both
    # final_fdm.py (the reference this script is scored against) and
    # afm_pinn/afm_pinn.py (which reaches relL2 0.081) are medium-aware. At
    # lam = 1.0 that made the substrate stretch wrong by |k_sub|/k = 1.50 in x
    # and |ky_sub|/k = 1.33 in y, mismatching the PML entry and reflecting back
    # into the physical domain -- across 47% of the PML band, which is how much
    # of it lies below Y_SUB. The config note above already stated the
    # precondition ("wavelength-independent once the stretch factor uses a
    # matched k_ref"); the code simply did not meet it.
    #
    # The y reference is |ky_sub|, not |k_sub|: the wave crossing the BOTTOM
    # PML travels vertically with the substrate's actual propagation constant,
    # and at 45 deg incidence those differ by ~13%. Same reasoning, and same
    # values, as final_fdm.py's SX/SY.
    #
    # Selecting on y < Y_SUB reproduces final_fdm.py exactly: its bottom PML
    # (y < 0) is wholly below Y_SUB and its top PML (y > 1) wholly above, and
    # in the left/right PML at y < Y_SUB sigma_y is zero, so s_y = 1 either way.
    # Verified against a transcription of that loop over 1512 points spanning
    # every side and corner of the band: max |s difference| = 4e-16.
    #
    # KNOWN SUBTLETY. k_ref_x now depends on y, so s_x jumps across y = Y_SUB
    # inside the LEFT and RIGHT PML, where sigma_x is nonzero. The residual's
    # Cyp term assumes d(s_x)/dy = 0, which that line violates. It is measure
    # zero, no collocation point lands on it, and the medium genuinely does
    # change there -- final_fdm.py has the same jump and absorbs it into its
    # conservative flux stencil, as does afm_pinn/afm_pinn.py by sampling the
    # air and substrate PML separately with constant k_ref each. s_y has no
    # such jump: sigma_y = 0 at y = Y_SUB, so s_y = 1 on both sides.
    in_sub  = xy[:, 1] < Y_SUB
    k_ref_x = np.where(in_sub, abs(k_sub_c),  k)
    k_ref_y = np.where(in_sub, abs(_ky_sub),  k)

    sx_c  = 1.0 + 1j * sig_x / k_ref_x
    sy_c  = 1.0 + 1j * sig_y / k_ref_y
    dsx_c = 1j * dsig_x / k_ref_x
    dsy_c = 1j * dsig_y / k_ref_y

    Cx_c   = sy_c / sx_c
    Cy_c   = sx_c / sy_c
    Cxp_c  = -sy_c * dsx_c / sx_c**2
    Cyp_c  = -sx_c * dsy_c / sy_c**2
    sxsy_c = sx_c * sy_c

    def _t(arr):
        return torch.tensor(arr.astype(np.float32), device=DEVICE).unsqueeze(1)

    xy_t = torch.tensor(xy.astype(np.float32), device=DEVICE,
                        requires_grad=True)
    coef = {
        'Cx_re':   _t(Cx_c.real),   'Cx_im':   _t(Cx_c.imag),
        'Cy_re':   _t(Cy_c.real),   'Cy_im':   _t(Cy_c.imag),
        'Cxp_re':  _t(Cxp_c.real),  'Cxp_im':  _t(Cxp_c.imag),
        'Cyp_re':  _t(Cyp_c.real),  'Cyp_im':  _t(Cyp_c.imag),
        'sxsy_re': _t(sxsy_c.real), 'sxsy_im': _t(sxsy_c.imag),
    }
    return {'xy': xy_t, 'coef': coef, 'xy_np': xy}


def sample_dirichlet(n_per_side):
    """Points on the outer PML boundary — Dirichlet u_scat = 0."""
    t  = np.random.rand(n_per_side)
    lo = -D_PML; hi = 1.0 + D_PML
    zeros_t = np.full(n_per_side, lo)
    ones_t  = np.full(n_per_side, hi)
    xy = np.vstack([
        np.stack([t * (hi - lo) + lo, zeros_t], 1),  # bottom
        np.stack([t * (hi - lo) + lo, ones_t],  1),  # top
        np.stack([zeros_t, t * (hi - lo) + lo], 1),  # left
        np.stack([ones_t,  t * (hi - lo) + lo], 1),  # right
    ])
    return torch.tensor(xy.astype(np.float32), device=DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# Per-parameter caches
# ─────────────────────────────────────────────────────────────────────────────
pts_cache       = {}  # pi → (xy_in, xy_out, xy_sub, xy_if, n_if)
pml_parts_cache = {}  # pi → [branch dicts] partitioning the CURRENT PML points


def sample_spatial_for_params(alpha, apex_y):
    xy_in       = sample_inside(N_INT, alpha, apex_y)
    xy_out      = sample_air(N_EXT, alpha, apex_y)
    xy_sub      = sample_substrate(N_SUB)
    xy_if, n_if = sample_interface_bc(N_IFACE, alpha, apex_y)
    return xy_in, xy_out, xy_sub, xy_if, n_if


def sample_spatial_for(pi):
    return sample_spatial_for_params(float(half_angle[pi]), float(apex_y_all[pi]))


def get_pts_at(key, alpha, apex_y):
    """Points for ANY (alpha, apex_y). `key` is only the cache identity: an int
    for a training geometry, a ('phys', alpha, apex_y) tuple for one of the
    data-free draws (see N_PHYS_EXTRA). Both are cleared together on a redraw."""
    if key not in pts_cache:
        pts_cache[key] = sample_spatial_for_params(alpha, apex_y)
    return pts_cache[key]


def get_pts(pi):
    pi = int(pi)
    return get_pts_at(pi, float(half_angle[pi]), float(apex_y_all[pi]))


def get_pml_parts_at(key, pml, alpha, apex_y):
    """PARTITION the shared PML points into the THREE operator branches for this
    (alpha, apex_y): inside the shank (net_int, k_int²), air outside the shank
    (net_scat, k_ext²) and substrate (net_scat, k_sub²).

    Previously a 0/1 mask blended both networks' outputs at every point:
        pR = mask*piR + (1-mask)*psR
    which required evaluating BOTH nets, plus their second derivatives, on all
    N_PML points -- and 84-94% of net_int's share was then multiplied by zero
    and discarded (measured: 6.15/10.91/16.37% inside at alpha=10/20/30 deg).
    The PML term was 72% of every closure evaluation.

    Splitting is EXACTLY equivalent, not an approximation: the mask is a
    non-differentiable 0/1 constant, so at a mask==1 point pR == piR and hence
    d(pR)/dx == d(piR)/dx. Each point is owned by exactly one branch. Summing
    squared residuals over all three branches and dividing by N_PML reproduces the
    old .mean() over all points.

    Cache is cleared whenever the PML sample is redrawn. As with get_pts_at,
    `key` is only the cache identity, so a data-free parameter draw partitions
    the same band through the same code.
    """
    if key not in pml_parts_cache:
        xy_np  = pml['xy_np']
        d      = phi_np(xy_np[:, 0], xy_np[:, 1], alpha, apex_y)
        inside = d < 0.0
        in_sub = xy_np[:, 1] < Y_SUB          # substrate half-space
        parts  = []
        # The tip never dips below Y_SUB, so inside and in_sub are disjoint
        # and these three selections partition the band exactly.
        for sel, use_int, ksq_re, ksq_im in (
                (inside,            True,  k_int_sq_re, k_int_sq_im),
                (~inside & ~in_sub, False, k_ext_sq,    0.0),
                (~inside &  in_sub, False, k_sub_sq_re, k_sub_sq_im)):
            if not sel.any():
                continue
            idx  = torch.tensor(np.nonzero(sel)[0], device=DEVICE)
            xy_s = torch.tensor(pml['xy_np'][sel].astype(np.float32),
                                device=DEVICE, requires_grad=True)
            parts.append({
                'xy':      xy_s,
                # coefficients are constants -- index_select, no grad needed
                'coef':    {kk: v.index_select(0, idx)
                            for kk, v in pml['coef'].items()},
                'use_int': use_int,
                'ksq_re':  ksq_re,
                'ksq_im':  ksq_im,
                # ksq_mag is now a scalar per branch. The old code computed
                # sqrt(ksq_re²+ksq_im²+1e-12) pointwise from the blended
                # tensors, which equals exactly this on each subset.
                'ksq_mag': float(np.hypot(ksq_re, ksq_im)),
            })
        pml_parts_cache[key] = parts
    return pml_parts_cache[key]


def get_pml_parts(pi, pml):
    pi = int(pi)
    return get_pml_parts_at(pi, pml, float(half_angle[pi]), float(apex_y_all[pi]))


# ── data-free parameter draws (see N_PHYS_EXTRA) ────────────────────────────
# The sweep's own range: physics may be imposed anywhere inside it, since it
# needs no FDM solve.
PHYS_A_LO, PHYS_A_HI = float(half_angle.min()), float(half_angle.max())
PHYS_H_LO, PHYS_H_HI = float(apex_y_all.min()),  float(apex_y_all.max())
phys_geoms = []


def draw_phys_geoms():
    """Redraw the data-free parameter points. Called where the collocation
    points are redrawn, NOT every step: within an L-BFGS block the objective
    has to stay fixed, or the line search is chasing a moving target.

    PHYS_SPAN_ALPHA / PHYS_SPAN_APEX narrow the draws towards the centre of the
    sweep -- see the note there on why the gap axis is where they are worth
    spending."""
    a_lo, a_hi = _phys_span(PHYS_A_LO, PHYS_A_HI, PHYS_SPAN_ALPHA)
    h_lo, h_hi = _phys_span(PHYS_H_LO, PHYS_H_HI, PHYS_SPAN_APEX)
    phys_geoms[:] = [
        (float(a), float(h)) for a, h in
        zip(np.random.uniform(a_lo, a_hi, N_PHYS_EXTRA),
            np.random.uniform(h_lo, h_hi, N_PHYS_EXTRA))]
    return phys_geoms


# ─────────────────────────────────────────────────────────────────────────────
# FDM data interpolation
# ─────────────────────────────────────────────────────────────────────────────
def interp_fdm(field_grid, xy_torch):
    """Bilinear interpolation of a [FDM_N, FDM_N] grid at xy_torch points."""
    xy_np    = xy_torch.detach().cpu().numpy()
    i_coords = xy_np[:, 1] * (FDM_N - 1)   # y → row
    j_coords = xy_np[:, 0] * (FDM_N - 1)   # x → col
    coords   = np.array([i_coords, j_coords])
    vals     = map_coordinates(field_grid, coords, order=1, mode='nearest')
    return torch.tensor(vals.astype(np.float32), device=DEVICE).unsqueeze(1)


def in_fdm_square(xy_torch):
    """[M,1] float mask: 1 where a point lies inside the FDM data's [0,1]^2.

    The FDM fields are stored on the physical square only, and interp_fdm's
    mode='nearest' silently CLAMPS anything outside it to the edge row/column.
    Only the interior data term can reach outside: sample_inside pushes tip
    surface points inward, and those come from _tip_bbox, which runs up to
    y = 1 + D_PML through the shank in the top PML. Measured on final_4, 61-70%
    of interior points sat above y = 1 and read air values of 0.2-0.3 from the
    y = 1 row -- all of l_data_int, against a true interior error of ~0.004."""
    xy = xy_torch.detach()
    return ((xy >= 0.0) & (xy <= 1.0)).all(dim=1, keepdim=True).float()


def masked_mean(sq, mask):
    """Mean of sq over the points where mask is 1 (0 if there are none)."""
    return (sq * mask).sum() / mask.sum().clamp_min(1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Loss-normalisation globals — set after the L0 pass below; initialised to 1
# so loss_fn_single is callable before they are computed.
# ─────────────────────────────────────────────────────────────────────────────
L0_PDE_INT = L0_PDE_EXT = L0_PDE_SUB = L0_FIELD = L0_FLUX = 1.0
L0_PML = L0_DIRICHLET = L0_DATA_SCAT = L0_DATA_INT = L0_DATA_SUB = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Loss function for a single half-angle index.
# PDE, BCs, PML operator and non-dimensionalisation are IDENTICAL to
# afm_tip_pinn.py; the spatial points belong to THIS parameter's geometry and
# both networks are conditioned on the alpha feature vector (ellipse_pinn.py).
# ─────────────────────────────────────────────────────────────────────────────
def loss_fn_single(pi, pml, xy_dir, geom=None):
    """All ten terms at ONE geometry.

    `pi` indexes the FDM sweep, so that geometry has data. Passing `geom` as an
    (alpha, apex_y) pair instead evaluates the PHYSICS at an arbitrary parameter
    point -- no FDM solve exists there, so the three data terms return zero (see
    N_PHYS_EXTRA)."""
    if geom is None:
        pi = int(pi)
        alpha, apex_y = float(half_angle[pi]), float(apex_y_all[pi])
        key = pi
    else:
        alpha, apex_y = float(geom[0]), float(geom[1])
        key = ("phys", alpha, apex_y)
    xy_in, xy_out, xy_sub, xy_if, n_if = get_pts_at(key, alpha, apex_y)
    phys_feats = param_to_features(alpha, apex_y)
    # [1,2] geometry row; broadcasts over any point block. Ignored by net_scat
    # and by net_int when SKIN_ANSATZ is off.
    gm = geom_row(alpha, apex_y)

    # ── 1. Interior PDE: (∇² + k_int²) u_int = 0  (complex k_int²)
    iR, iI   = net_int(phys_feats, xy_in, gm)
    lapR_in  = laplacian(iR, xy_in)
    lapI_in  = laplacian(iI, xy_in)
    # Divide by |k_int²| — relative equation violation (afm_tip_pinn.py)
    res_in_R = (lapR_in + k_int_sq_re * iR - k_int_sq_im * iI) / k_int_sq_mag
    res_in_I = (lapI_in + k_int_sq_re * iI + k_int_sq_im * iR) / k_int_sq_mag
    l_pde_int = (res_in_R**2 + res_in_I**2).mean()

    # ── 2. Exterior PDE: (∇² + k²) u_scat = 0
    sR, sI   = net_scat(phys_feats, xy_out)
    lapR_ex  = laplacian(sR, xy_out)
    lapI_ex  = laplacian(sI, xy_out)
    res_ex_R = (lapR_ex + k_ext_sq * sR) / k_ext_sq_mag
    res_ex_I = (lapI_ex + k_ext_sq * sI) / k_ext_sq_mag
    l_pde_ext = (res_ex_R**2 + res_ex_I**2).mean()

    # ── 2b. Substrate PDE: (∇² + k_sub²) u_scat = 0, complex k_sub².
    # Same network as the air region (merged exterior) — only k² differs.
    bR, bI    = net_sub(phys_feats, xy_sub)
    lapR_sb   = laplacian(bR, xy_sub)
    lapI_sb   = laplacian(bI, xy_sub)
    res_sb_R  = (lapR_sb + k_sub_sq_re * bR - k_sub_sq_im * bI) / k_sub_sq_mag
    res_sb_I  = (lapI_sb + k_sub_sq_re * bI + k_sub_sq_im * bR) / k_sub_sq_mag
    l_pde_sub = (res_sb_R**2 + res_sb_I**2).mean()

    # ── 3. Interface: Ez continuity  u_int = u_inc + u_scat  on phi = 0
    iR_if, iI_if = net_int(phys_feats, xy_if, gm)
    sR_if, sI_if = net_scat(phys_feats, xy_if)
    inc_R_if     = u_bg_air_re(xy_if)   # tip is wholly in air
    inc_I_if     = u_bg_air_im(xy_if)
    l_field = ((iR_if - (sR_if + inc_R_if))**2 +
               (iI_if - (sI_if + inc_I_if))**2).mean()

    # ── 4. Interface: flux continuity with SDF-autograd normals
    dn_iR   = normal_deriv(iR_if, xy_if, n_if)
    dn_iI   = normal_deriv(iI_if, xy_if, n_if)
    dn_sR   = normal_deriv(sR_if, xy_if, n_if)
    dn_sI   = normal_deriv(sI_if, xy_if, n_if)
    dn_incR = du_bg_air_dn_re(xy_if, n_if)
    dn_incI = du_bg_air_dn_im(xy_if, n_if)
    l_flux  = ((dn_iR - (dn_sR + dn_incR))**2 +
               (dn_iI - (dn_sI + dn_incI))**2).mean()

    # ── 5. PML: UPML-stretched Helmholtz in the PML band. Operator switches
    #    per point (afm_tip_pinn.py): inside the shank → net_int + k_int²;
    #    outside → net_scat + k². The inside mask depends on alpha.
    # Each branch owns a disjoint subset of the PML points, so only the network
    # that actually governs a point is evaluated there (see get_pml_parts).
    pml_sq = None
    for part in get_pml_parts_at(key, pml, alpha, apex_y):
        xy_p = part['xy']
        C    = part['coef']

        net    = net_int if part['use_int'] else net_scat
        pR, pI = net(phys_feats, xy_p, gm)

        ksq_re, ksq_im = part['ksq_re'], part['ksq_im']

        gR   = torch.autograd.grad(pR.sum(), xy_p, create_graph=True)[0]
        pRx, pRy = gR[:, 0:1], gR[:, 1:2]
        pRxx = torch.autograd.grad(gR[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
        pRyy = torch.autograd.grad(gR[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

        gI   = torch.autograd.grad(pI.sum(), xy_p, create_graph=True)[0]
        pIx, pIy = gI[:, 0:1], gI[:, 1:2]
        pIxx = torch.autograd.grad(gI[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
        pIyy = torch.autograd.grad(gI[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

        # Complex reaction coefficient  K = k²(x) * s_x s_y
        K_re = ksq_re * C['sxsy_re'] - ksq_im * C['sxsy_im']
        K_im = ksq_re * C['sxsy_im'] + ksq_im * C['sxsy_re']

        # Non-dimensionalise by the LOCAL |k²| (afm_tip_pinn.py) -- constant
        # within a branch, so a scalar rather than a pointwise tensor.
        ksq_mag = part['ksq_mag']

        res_pml_R = (C['Cx_re'] * pRxx - C['Cx_im'] * pIxx
                   + C['Cxp_re'] * pRx - C['Cxp_im'] * pIx
                   + C['Cy_re'] * pRyy - C['Cy_im'] * pIyy
                   + C['Cyp_re'] * pRy - C['Cyp_im'] * pIy
                   + (K_re * pR - K_im * pI)) / ksq_mag
        res_pml_I = (C['Cx_re'] * pIxx + C['Cx_im'] * pRxx
                   + C['Cxp_re'] * pIx + C['Cxp_im'] * pRx
                   + C['Cy_re'] * pIyy + C['Cy_im'] * pRyy
                   + C['Cyp_re'] * pIy + C['Cyp_im'] * pRy
                   + (K_re * pI + K_im * pR)) / ksq_mag

        s      = (res_pml_R**2 + res_pml_I**2).sum()
        pml_sq = s if pml_sq is None else pml_sq + s

    # sum over all three branches / N_PML == the old .mean() over all PML points
    l_pml = pml_sq / float(N_PML)

    # ── 6. Dirichlet on the outer PML boundary — u_scat = 0
    dR, dI = net_scat(phys_feats, xy_dir)
    l_dirichlet = (dR**2 + dI**2).mean()

    total = (W_PDE_INT   * l_pde_int   / L0_PDE_INT +
             W_PDE_EXT   * l_pde_ext   / L0_PDE_EXT +
             W_PDE_SUB   * l_pde_sub   / L0_PDE_SUB +
             W_FIELD     * l_field     / L0_FIELD   +
             W_FLUX      * l_flux      / L0_FLUX    +
             W_PML       * l_pml       / L0_PML     +
             W_DIRICHLET * l_dirichlet / L0_DIRICHLET)

    # ── 7. FDM data supervision — indexed by THIS half-angle (off by default)
    # Guarded: each interp_fdm call does a GPU->CPU sync, a scipy
    # map_coordinates, and a CPU->GPU copy. Four of them per parameter per
    # closure evaluation (x N_PARAM_BATCH) is a large serialising cost whose
    # result was then multiplied by W_DATA_* = 0 and discarded.
    # geom is not None: a data-free draw, so there is nothing to compare with
    if geom is None and (W_DATA_SCAT > 0.0 or W_DATA_INT > 0.0
                         or W_DATA_SUB > 0.0):
        fdm_sR = interp_fdm(FDM_scat_re[pi],  xy_out)
        fdm_sI = interp_fdm(FDM_scat_im[pi],  xy_out)
        l_data_scat = ((sR - fdm_sR)**2 + (sI - fdm_sI)**2).mean()

        # Only interior points the FDM actually covers -- see in_fdm_square.
        fdm_iR = interp_fdm(FDM_total_re[pi], xy_in)
        fdm_iI = interp_fdm(FDM_total_im[pi], xy_in)
        l_data_int = masked_mean((iR - fdm_iR)**2 + (iI - fdm_iI)**2,
                                 in_fdm_square(xy_in))

        # The FDM's u_scat is defined over the WHOLE grid, substrate included,
        # so this needs no new reference -- only the comparison that was never
        # made. bR/bI are net_sub's output at xy_sub, already computed for
        # l_pde_sub above.
        fdm_bR = interp_fdm(FDM_scat_re[pi],  xy_sub)
        fdm_bI = interp_fdm(FDM_scat_im[pi],  xy_sub)
        l_data_sub = ((bR - fdm_bR)**2 + (bI - fdm_bI)**2).mean()

        total = total + W_DATA_SCAT * l_data_scat / L0_DATA_SCAT \
                      + W_DATA_INT  * l_data_int  / L0_DATA_INT \
                      + W_DATA_SUB  * l_data_sub  / L0_DATA_SUB
    else:
        # zeros_like keeps device and dtype in step with the live tensors
        l_data_scat = torch.zeros_like(l_pde_int)
        l_data_int  = torch.zeros_like(l_pde_int)
        l_data_sub  = torch.zeros_like(l_pde_int)

    return (total, l_pde_int, l_pde_ext, l_pde_sub, l_field, l_flux, l_pml,
            l_dirichlet, l_data_scat, l_data_int, l_data_sub)


# ─────────────────────────────────────────────────────────────────────────────
# Batched-over-half-angles path. Every angle's collocation set is concatenated
# into one tensor per term, with a matching per-point physics-feature matrix, so
# each term costs ONE forward (and one double-backward) for the whole batch.
#
# The networks are pointwise, so concatenation changes nothing mathematically:
# each output row depends only on its own input row, hence autograd's Laplacian
# and normal derivatives are per-point regardless of what else is in the tensor.
#
# NORMALISATION. The per-angle path returns, for each term, the mean over that
# angle's points, and backward_batch then averages those over the batch. Here a
# single global mean reproduces that exactly because every angle contributes the
# SAME number of points to a given term. The one exception is l_pml, where the
# inside/outside split is angle-dependent (6-17% inside): there each angle still
# contributes exactly N_PML points in total, so the global squared sum divided
# by (N_PML * n_angles) equals the mean of the per-angle values.
# ─────────────────────────────────────────────────────────────────────────────
def _specs_from_indices(param_indices):
    """(cache key, alpha, apex_y, sweep index) for training geometries."""
    out = []
    for p in param_indices:
        p = int(p)
        out.append((p, float(half_angle[p]), float(apex_y_all[p]), p))
    return out


def _specs_from_geoms(geoms):
    """The same for DATA-FREE draws (see N_PHYS_EXTRA): no sweep index, and the
    cache key is the parameter pair, so their points and PML partition are
    cached and reused exactly like a training geometry's."""
    return [(("phys", float(a), float(h)), float(a), float(h), None)
            for a, h in geoms]


def _as_specs(items):
    """Accept either sweep indices (what every caller used to pass) or specs."""
    items = list(items)
    if items and isinstance(items[0], tuple) and len(items[0]) == 4:
        return items
    return _specs_from_indices(items)


def _phys_extra_chunks():
    """The data-free draws, split like a training batch so ACCUM_CHUNK bounds
    memory here too."""
    specs = _specs_from_geoms(phys_geoms)
    size  = ACCUM_CHUNK if 0 < ACCUM_CHUNK < len(specs) else len(specs)
    return [specs[i:i + size] for i in range(0, len(specs), size)]


batch_cache = {}   # tuple(cache keys) → assembled batched tensors
# The chunk keys of the batch backward_batch is currently evaluating: those are
# the only entries worth keeping. Empty = keep just the one batch being built.
_BATCH_LIVE_KEYS = [set()]


def build_batched(param_indices, pml, xy_dir):
    specs = _as_specs(param_indices)
    key = tuple(s[0] for s in specs)
    if key in batch_cache:
        return batch_cache[key]
    # Only the current batch's chunks are ever live. Adam draws a fresh angle
    # combination every epoch (so keys essentially never repeat), while L-BFGS
    # reuses the same chunks for RESAMPLE_LBFGS*max_iter closure evals -- where
    # this cache does the work. Dropping everything that is not a chunk of the
    # current batch keeps the concatenated copies from piling up between
    # resamples.
    live = _BATCH_LIVE_KEYS[0] if key in _BATCH_LIVE_KEYS[0] else set()
    for stale in [k for k in batch_cache if k not in live]:
        batch_cache.pop(stale)

    def leaf(ts):
        """Fresh leaf from concatenated per-angle tensors. Building a genuine
        leaf (rather than differentiating through torch.cat) keeps the
        double-backward graph identical in shape to the per-angle path."""
        return torch.cat(ts, dim=0).detach().requires_grad_(True)

    feats  = {s[0]: param_to_features(s[1], s[2]) for s in specs}
    rows   = lambda ck, m: feats[ck].expand(m, -1)
    # Per-point geometry, built exactly like the per-point features: the skin
    # ansatz needs phi at every point, and a concatenated block spans several
    # geometries, so one [1,2] row per angle will not do here.
    geoms  = {s[0]: geom_row(s[1], s[2]) for s in specs}
    grows  = lambda ck, m: geoms[ck].expand(m, -1)

    pts    = {s[0]: get_pts_at(s[0], s[1], s[2]) for s in specs}
    xy_in  = leaf([pts[ck][0] for ck in key])
    xy_out = leaf([pts[ck][1] for ck in key])
    xy_sub = leaf([pts[ck][2] for ck in key])
    xy_if  = leaf([pts[ck][3] for ck in key])
    n_if   = torch.cat([pts[ck][4] for ck in key], dim=0).detach()

    f_in   = torch.cat([rows(ck, N_INT)   for ck in key], dim=0)
    f_out  = torch.cat([rows(ck, N_EXT)   for ck in key], dim=0)
    f_sub  = torch.cat([rows(ck, N_SUB)   for ck in key], dim=0)
    f_if   = torch.cat([rows(ck, N_IFACE) for ck in key], dim=0)

    g_in   = torch.cat([grows(ck, N_INT)   for ck in key], dim=0)
    g_if   = torch.cat([grows(ck, N_IFACE) for ck in key], dim=0)

    # ── PML: gather the THREE operator branches get_pml_parts defines -- tip
    #    shank (net_int, k_int²), air (net_scat, k²) and substrate (net_scat,
    #    k_sub²) -- across ALL angles. A branch is keyed by its OPERATOR, not
    #    only by which network it uses: keying on use_int alone merged air and
    #    substrate and gave the substrate PML the air wavenumber, so this path
    #    (the one training runs, BATCH_ANGLES) disagreed with loss_fn_single by
    #    6x on l_pml. Constants are branch-wide, so each branch is one
    #    homogeneous tensor.
    groups = {}
    for ck, _a, _h, _fdm in specs:
        for part in get_pml_parts_at(ck, pml, _a, _h):
            op  = (part['use_int'], part['ksq_re'], part['ksq_im'])
            grp = groups.setdefault(op, {'xy': [], 'coef': [], 'feats': [],
                                         'geom': [], 'ksq_mag': part['ksq_mag']})
            m = part['xy'].shape[0]
            grp['xy'].append(part['xy'])
            grp['coef'].append(part['coef'])
            grp['feats'].append(rows(ck, m))
            grp['geom'].append(grows(ck, m))
    branches = []
    for (use_int, ksq_re, ksq_im), grp in groups.items():
        cfs = grp['coef']
        branches.append({
            'xy':      leaf(grp['xy']),
            'coef':    {kk: torch.cat([c[kk] for c in cfs], dim=0)
                        for kk in cfs[0]},
            'feats':   torch.cat(grp['feats'], dim=0),
            'geom':    torch.cat(grp['geom'], dim=0),
            'use_int': use_int,
            'ksq_re':  ksq_re,
            'ksq_im':  ksq_im,
            'ksq_mag': grp['ksq_mag'],
        })

    n_dir = xy_dir.shape[0]
    out = {
        'n':       len(key),
        'xy_in':   xy_in,  'f_in':  f_in,  'g_in': g_in,
        'xy_out':  xy_out, 'f_out': f_out,
        'xy_sub':  xy_sub, 'f_sub': f_sub,
        'xy_if':   xy_if,  'f_if':  f_if, 'n_if': n_if, 'g_if': g_if,
        'pml':     branches,
        'xy_dir':  torch.cat([xy_dir] * len(key), dim=0),
        'f_dir':   torch.cat([rows(ck, n_dir) for ck in key], dim=0),
        # sweep index per block, None for a data-free draw (see N_PHYS_EXTRA)
        'fdm':     [s[3] for s in specs],
    }
    batch_cache[key] = out
    return out


def loss_fn_batched(param_indices, pml, xy_dir):
    """Same terms as loss_fn_single, averaged over the parameter batch.

    Takes either sweep indices or the (cache_key, alpha, apex_y, fdm_index)
    specs built by _as_specs. A batch whose specs carry no fdm_index is a set of
    DATA-FREE draws (see N_PHYS_EXTRA): the physics is evaluated exactly as for
    a training batch and the three data terms are zero. Mixing the two in one
    call is refused rather than silently averaging over different subsets."""
    B = build_batched(param_indices, pml, xy_dir)
    n = B['n']

    # ── 1. Interior PDE
    xy_in    = B['xy_in']
    iR, iI   = net_int(B['f_in'], xy_in, B['g_in'])
    res_in_R = (laplacian(iR, xy_in) + k_int_sq_re * iR - k_int_sq_im * iI) / k_int_sq_mag
    res_in_I = (laplacian(iI, xy_in) + k_int_sq_re * iI + k_int_sq_im * iR) / k_int_sq_mag
    l_pde_int = (res_in_R**2 + res_in_I**2).mean()

    # ── 2. Exterior PDE
    xy_out   = B['xy_out']
    sR, sI   = net_scat(B['f_out'], xy_out)
    res_ex_R = (laplacian(sR, xy_out) + k_ext_sq * sR) / k_ext_sq_mag
    res_ex_I = (laplacian(sI, xy_out) + k_ext_sq * sI) / k_ext_sq_mag
    l_pde_ext = (res_ex_R**2 + res_ex_I**2).mean()

    # ── 2b. Substrate PDE (merged exterior net, complex k_sub²)
    xy_sub    = B['xy_sub']
    bR, bI    = net_sub(B['f_sub'], xy_sub)
    res_sb_R  = (laplacian(bR, xy_sub) + k_sub_sq_re * bR - k_sub_sq_im * bI) / k_sub_sq_mag
    res_sb_I  = (laplacian(bI, xy_sub) + k_sub_sq_re * bI + k_sub_sq_im * bR) / k_sub_sq_mag
    l_pde_sub = (res_sb_R**2 + res_sb_I**2).mean()

    # ── 3./4. Interface field + flux continuity
    xy_if, n_if  = B['xy_if'], B['n_if']
    iR_if, iI_if = net_int(B['f_if'], xy_if, B['g_if'])
    sR_if, sI_if = net_scat(B['f_if'], xy_if)
    l_field = ((iR_if - (sR_if + u_bg_air_re(xy_if)))**2 +
               (iI_if - (sI_if + u_bg_air_im(xy_if)))**2).mean()

    dn_incR = du_bg_air_dn_re(xy_if, n_if)
    dn_incI = du_bg_air_dn_im(xy_if, n_if)
    l_flux  = ((normal_deriv(iR_if, xy_if, n_if)
                - (normal_deriv(sR_if, xy_if, n_if) + dn_incR))**2 +
               (normal_deriv(iI_if, xy_if, n_if)
                - (normal_deriv(sI_if, xy_if, n_if) + dn_incI))**2).mean()

    # ── 5. PML, one forward per operator branch across the whole batch
    pml_sq = None
    for part in B['pml']:
        xy_p   = part['xy']
        C      = part['coef']
        net    = net_int if part['use_int'] else net_scat
        pR, pI = net(part['feats'], xy_p, part['geom'])
        ksq_re, ksq_im, ksq_mag = part['ksq_re'], part['ksq_im'], part['ksq_mag']

        gR   = torch.autograd.grad(pR.sum(), xy_p, create_graph=True)[0]
        pRx, pRy = gR[:, 0:1], gR[:, 1:2]
        pRxx = torch.autograd.grad(gR[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
        pRyy = torch.autograd.grad(gR[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

        gI   = torch.autograd.grad(pI.sum(), xy_p, create_graph=True)[0]
        pIx, pIy = gI[:, 0:1], gI[:, 1:2]
        pIxx = torch.autograd.grad(gI[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
        pIyy = torch.autograd.grad(gI[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

        K_re = ksq_re * C['sxsy_re'] - ksq_im * C['sxsy_im']
        K_im = ksq_re * C['sxsy_im'] + ksq_im * C['sxsy_re']

        res_pml_R = (C['Cx_re'] * pRxx - C['Cx_im'] * pIxx
                   + C['Cxp_re'] * pRx - C['Cxp_im'] * pIx
                   + C['Cy_re'] * pRyy - C['Cy_im'] * pIyy
                   + C['Cyp_re'] * pRy - C['Cyp_im'] * pIy
                   + (K_re * pR - K_im * pI)) / ksq_mag
        res_pml_I = (C['Cx_re'] * pIxx + C['Cx_im'] * pRxx
                   + C['Cxp_re'] * pIx + C['Cxp_im'] * pRx
                   + C['Cy_re'] * pIyy + C['Cy_im'] * pRyy
                   + C['Cyp_re'] * pIy + C['Cyp_im'] * pRy
                   + (K_re * pI + K_im * pR)) / ksq_mag

        s      = (res_pml_R**2 + res_pml_I**2).sum()
        pml_sq = s if pml_sq is None else pml_sq + s

    # each angle contributes exactly N_PML points across the three branches
    l_pml = pml_sq / float(N_PML * n)

    # ── 6. Dirichlet
    dR, dI = net_scat(B['f_dir'], B['xy_dir'])
    l_dirichlet = (dR**2 + dI**2).mean()

    total = (W_PDE_INT   * l_pde_int   / L0_PDE_INT +
             W_PDE_EXT   * l_pde_ext   / L0_PDE_EXT +
             W_PDE_SUB   * l_pde_sub   / L0_PDE_SUB +
             W_FIELD     * l_field     / L0_FIELD   +
             W_FLUX      * l_flux      / L0_FLUX    +
             W_PML       * l_pml       / L0_PML     +
             W_DIRICHLET * l_dirichlet / L0_DIRICHLET)

    # ── 7. FDM data supervision (off by default; see loss_fn_single)
    _fdm = B['fdm']
    _has_data = [p is not None for p in _fdm]
    if any(_has_data) and not all(_has_data):
        raise RuntimeError("loss_fn_batched: a batch must be either all "
                           "training geometries or all data-free draws")
    if all(_has_data) and (W_DATA_SCAT > 0.0 or W_DATA_INT > 0.0
                           or W_DATA_SUB > 0.0):
        idxs = [int(p) for p in _fdm]
        fdm_sR = torch.cat([interp_fdm(FDM_scat_re[p],  xy_out[j*N_EXT:(j+1)*N_EXT])
                            for j, p in enumerate(idxs)], dim=0)
        fdm_sI = torch.cat([interp_fdm(FDM_scat_im[p],  xy_out[j*N_EXT:(j+1)*N_EXT])
                            for j, p in enumerate(idxs)], dim=0)
        l_data_scat = ((sR - fdm_sR)**2 + (sI - fdm_sI)**2).mean()

        fdm_iR = torch.cat([interp_fdm(FDM_total_re[p], xy_in[j*N_INT:(j+1)*N_INT])
                            for j, p in enumerate(idxs)], dim=0)
        fdm_iI = torch.cat([interp_fdm(FDM_total_im[p], xy_in[j*N_INT:(j+1)*N_INT])
                            for j, p in enumerate(idxs)], dim=0)
        # Masked to the FDM square (see in_fdm_square), and averaged PER
        # GEOMETRY before the batch mean: the surviving count differs between
        # geometries, so a pooled mean would weight them unequally and stop
        # matching loss_fn_single.
        _sq_int = ((iR - fdm_iR)**2 + (iI - fdm_iI)**2).reshape(n, N_INT)
        _m_int  = in_fdm_square(xy_in).reshape(n, N_INT)
        l_data_int = ((_sq_int * _m_int).sum(dim=1)
                      / _m_int.sum(dim=1).clamp_min(1.0)).mean()

        # Same block layout as xy_out/xy_in: build_batched concatenates each
        # geometry's own N_SUB draw, so geometry j owns rows [j*N_SUB,(j+1)*N_SUB).
        fdm_bR = torch.cat([interp_fdm(FDM_scat_re[p],  xy_sub[j*N_SUB:(j+1)*N_SUB])
                            for j, p in enumerate(idxs)], dim=0)
        fdm_bI = torch.cat([interp_fdm(FDM_scat_im[p],  xy_sub[j*N_SUB:(j+1)*N_SUB])
                            for j, p in enumerate(idxs)], dim=0)
        l_data_sub = ((bR - fdm_bR)**2 + (bI - fdm_bI)**2).mean()

        total = total + W_DATA_SCAT * l_data_scat / L0_DATA_SCAT \
                      + W_DATA_INT  * l_data_int  / L0_DATA_INT \
                      + W_DATA_SUB  * l_data_sub  / L0_DATA_SUB
    else:
        l_data_scat = torch.zeros_like(l_pde_int)
        l_data_int  = torch.zeros_like(l_pde_int)
        l_data_sub  = torch.zeros_like(l_pde_int)

    return (total, l_pde_int, l_pde_ext, l_pde_sub, l_field, l_flux, l_pml,
            l_dirichlet, l_data_scat, l_data_int, l_data_sub)


def backward_batch(param_indices, pml, xy_dir):
    """Accumulate the batch gradient. Caller must zero_grad() before and step()
    after. Returns detached term means.

    BATCH_ANGLES: one forward/backward over all half-angles at once (fewer,
    larger GPU kernels). Otherwise the memory-frugal per-angle loop
    (ellipse_pinn.py): backpropagate each angle immediately and free its graph.
    Both produce the same gradient; see loss_fn_batched on normalisation.

    N_PHYS_EXTRA adds the PHYSICS at that many data-free parameter draws on top,
    weighted PHYS_EXTRA_WEIGHT, through the per-geometry path. The returned
    terms stay the TRAINING geometries' means, so the CSV and the loss figure
    keep meaning what they always did; the extra contribution is reported
    separately in _PHYS_EXTRA_LAST."""
    n = len(param_indices)
    if BATCH_ANGLES:
        # ACCUM_CHUNK: loss_fn_batched returns the MEAN over the geometries it
        # is given, so a chunk of m weighs m/n. The weighted chunk gradients
        # sum to the full batch's, and so do the weighted term values.
        size   = ACCUM_CHUNK if 0 < ACCUM_CHUNK < n else n
        chunks = [param_indices[i:i + size] for i in range(0, n, size)]
        live = {tuple(int(p) for p in c) for c in chunks}
        if N_PHYS_EXTRA and phys_geoms:   # keep the draws' tensors cached too
            live |= {tuple(s[0] for s in c) for c in _phys_extra_chunks()}
        _BATCH_LIVE_KEYS[0] = live
        run = None
        for chunk in chunks:
            w = len(chunk) / n
            t, *ts = loss_fn_batched(chunk, pml, xy_dir)
            (t * w).backward()
            vals = [w * x.detach() for x in [t] + ts]
            run  = vals if run is None else [va + vb for va, vb in zip(run, vals)]
        _physics_extra_backward(pml, xy_dir)
        return run
    run = None
    for pi in param_indices:
        t, *ts = loss_fn_single(int(pi), pml, xy_dir)
        (t / n).backward()
        vals = [t.detach()] + [x.detach() for x in ts]
        run  = vals if run is None else [va + vb for va, vb in zip(run, vals)]
    _physics_extra_backward(pml, xy_dir)
    return [v / n for v in run]


_PHYS_EXTRA_LAST = [0.0]


def _physics_extra_backward(pml, xy_dir):
    """Physics at the data-free parameter draws, backpropagated onto the same
    gradient (see N_PHYS_EXTRA). The whole set counts PHYS_EXTRA_WEIGHT.

    BATCHED, for the same reason the training batch is: the cost of these terms
    on a GPU is kernel launches, not arithmetic. Issued one draw at a time they
    made an Adam epoch 4x more expensive (0.21 -> 0.85 s at 8 training
    geometries and 4 draws), because each per-geometry call launches as many
    kernels as the whole batched call does for eight. On CPU, where there is
    nothing to amortise, the same loop cost only the expected +50%."""
    if not N_PHYS_EXTRA or not phys_geoms:
        _PHYS_EXTRA_LAST[0] = 0.0
        return 0.0
    tot = 0.0
    if BATCH_ANGLES:
        chunks = _phys_extra_chunks()
        n_all  = sum(len(c) for c in chunks)
        for chunk in chunks:
            w = PHYS_EXTRA_WEIGHT * len(chunk) / n_all
            t_e, *_ = loss_fn_batched(chunk, pml, xy_dir)
            (t_e * w).backward()
            tot += float(t_e) * w
    else:
        w = PHYS_EXTRA_WEIGHT / len(phys_geoms)
        for gm in phys_geoms:
            t_e, *_ = loss_fn_single(None, pml, xy_dir, geom=gm)
            (t_e * w).backward()
            tot += float(t_e) * w
    _PHYS_EXTRA_LAST[0] = tot
    return tot

# ─────────────────────────────────────────────────────────────────────────────
# Radial-flux diagnostic — detects the ±k sign degeneracy (afm_tip_pinn.py).
# Net outward time-averaged power through a loop enclosing the apex should
# be POSITIVE for a correct outgoing solution.
# ─────────────────────────────────────────────────────────────────────────────
# The loop is centred on the apex, which now MOVES with the parameter, and its
# radius must stay clear of the substrate as well as the tip: at the tightest
# gap (apex_y = 0.41, Y_SUB = 0.4) only ~0.01 separates them, so a fixed radius
# would put the loop inside the substrate and mix media into the flux integral.
_FLUX_N = 256


def _flux_radius(apex_y):
    return float(min(apex_y - R_APEX - 0.02,        # clear of the tip apex
                     apex_y - Y_SUB - 0.002,        # clear of the substrate
                     0.5 - 0.02))                   # inside the physical box


def radial_flux_diag(pi):
    pi     = int(pi)
    alpha  = float(half_angle[pi])
    apex_y = float(apex_y_all[pi])
    _FLUX_R = _flux_radius(apex_y)
    theta = np.linspace(0, 2 * np.pi, _FLUX_N, endpoint=False)
    xs = TIP_X + _FLUX_R * np.cos(theta)
    ys = apex_y + _FLUX_R * np.sin(theta)
    good = (phi_np(xs, ys, alpha, apex_y) > 0.0) & (xs > 0) & (xs < 1) & \
           (ys > Y_SUB) & (ys < 1)          # air only: k is uniform there
    xs, ys = xs[good], ys[good]
    nx, ny = np.cos(theta[good]), np.sin(theta[good])
    xy  = torch.tensor(np.stack([xs, ys], 1).astype(np.float32),
                       device=DEVICE, requires_grad=True)
    nxy = torch.tensor(np.stack([nx, ny], 1).astype(np.float32), device=DEVICE)
    phys_feats = param_to_features(alpha, apex_y)
    uR, uI = net_scat(phys_feats, xy)
    dnR = normal_deriv(uR, xy, nxy)
    dnI = normal_deriv(uI, xy, nxy)
    Sr = (uR * dnI - uI * dnR)   # Im( conj(u) du/dn )
    return float(Sr.mean().detach().cpu())

# ─────────────────────────────────────────────────────────────────────────────
# Loss normalisation — average each raw term over K_L0 fresh random batches
# of the untrained networks and store as L0_* (afm_tip_pinn.py scheme).
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing L0 loss normalisation ...")
K_L0   = 5
_accum = {key: 0.0 for key in
          ['lpi', 'lpe', 'lps', 'lf', 'lfl', 'lpml', 'ldir', 'lds', 'ldi',
           'ldsb']}

for _ in range(K_L0):
    pts_cache.clear()
    pml_parts_cache.clear()
    batch_cache.clear()
    _pml    = sample_pml(N_PML)
    _xy_dir = sample_dirichlet(N_DIR_SIDE)
    _batch  = np.random.choice(train_idx, size=N_PARAM_BATCH, replace=False)
    for pi in _batch:
        with torch.no_grad():
            pass  # (graphs are needed for laplacians; just don't backward)
        _, _lpi, _lpe, _lps, _lf, _lfl, _lpml, _ldir, _lds, _ldi, _ldsb = \
            loss_fn_single(int(pi), _pml, _xy_dir)
        _accum['lpi']  += _lpi.item();  _accum['lpe']  += _lpe.item()
        _accum['lps']  += _lps.item()
        _accum['lf']   += _lf.item();   _accum['lfl']  += _lfl.item()
        _accum['lpml'] += _lpml.item(); _accum['ldir'] += _ldir.item()
        _accum['lds']  += _lds.item();  _accum['ldi']  += _ldi.item()
        _accum['ldsb'] += _ldsb.item()

_den = K_L0 * N_PARAM_BATCH
L0_PDE_INT   = max(_accum['lpi']  / _den, 1e-12)
L0_PDE_EXT   = max(_accum['lpe']  / _den, 1e-12)
L0_PDE_SUB   = max(_accum['lps']  / _den, 1e-12)
L0_FIELD     = max(_accum['lf']   / _den, 1e-12)
L0_FLUX      = max(_accum['lfl']  / _den, 1e-12)
L0_PML       = max(_accum['lpml'] / _den, 1e-12)
L0_DIRICHLET = max(_accum['ldir'] / _den, 1e-12)
L0_DATA_SCAT = max(_accum['lds']  / _den, 1e-12)
L0_DATA_INT  = max(_accum['ldi']  / _den, 1e-12)
L0_DATA_SUB  = max(_accum['ldsb'] / _den, 1e-12)

pts_cache.clear()
pml_parts_cache.clear()
batch_cache.clear()

print(f"  (averaged over {K_L0}×{N_PARAM_BATCH} param draws)")
print(f"  L0_PDE_INT   = {L0_PDE_INT:.3e}")
print(f"  L0_PDE_EXT   = {L0_PDE_EXT:.3e}")
print(f"  L0_PDE_SUB   = {L0_PDE_SUB:.3e}")
print(f"  L0_FIELD     = {L0_FIELD:.3e}   (measured, before override below)")
print(f"  L0_FLUX      = {L0_FLUX:.3e}")
print(f"  L0_PML       = {L0_PML:.3e}")
print(f"  L0_DIRICHLET = {L0_DIRICHLET:.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# Override L0_FIELD with a PHYSICAL scale set by the refractive index.
#
# The measured L0 is the residual of UNTRAINED networks, which for field
# continuity is dominated by |u_inc| = 1 -- so L0_FIELD ~ 1 regardless of the
# material. But the quantity the condition is about is the TOTAL field on the
# tip surface, and for a strongly absorbing tip that is small: the Leontovich
# impedance estimate gives
#       |u_total| on Gamma  ~  2k/|k_int|  =  2/|n|
# with |n| = sqrt(N_RE^2 + N_IM^2). At n = 3+30i that is 2/30.15 = 0.066, and
# the FDM reference gives 0.012-0.024, so the O(1) ruler over-estimates the
# scale by ~15-50x (~230x once squared).
#
# Consequence of the old normalisation, measured on the trained net: the
# residual on Gamma was ~0.10 against |u_total| ~ 0.10, i.e. the condition was
# violated by the WHOLE physical quantity while the loss reported 3.9% of L0.
# Using the physical scale makes "l_field/L0_FIELD = 1" mean "the residual
# equals the surface field", which is the honest unit.
#
# NOTE the flux condition needs no such fix: its physical scale is
#       |d_n u_total| ~ |k_int| * 2k/|k_int| = 2k     (independent of n)
# giving (2k)^2 = 4k^2, within ~5x of the measured L0_FLUX, so that one is
# already about right. Left alone deliberately.
U_REF_GAMMA   = 2.0 / abs(N_RE + 1j * N_IM)     # = 2k/|k_int|
L0_FIELD_PHYS = U_REF_GAMMA**2
_l0_field_measured = L0_FIELD

if USE_PHYSICAL_L0_FIELD:
    L0_FIELD = L0_FIELD_PHYS
    print(f"\n  L0_FIELD override (physical surface scale):")
    print(f"    |n| = {abs(N_RE + 1j*N_IM):.3f}   u_ref = 2/|n| = {U_REF_GAMMA:.4f}")
    print(f"    L0_FIELD {_l0_field_measured:.3e} -> {L0_FIELD:.3e} "
          f"({_l0_field_measured/L0_FIELD:.0f}x more weight on field continuity)")
    print(f"    (flux left on its measured L0; its physical scale 4k^2 = "
          f"{4*k**2:.1f} is within ~5x of {L0_FLUX:.1f})")
else:
    print(f"\n  L0_FIELD: using the MEASURED value {L0_FIELD:.3e} "
          f"(physical scale would be {L0_FIELD_PHYS:.3e}, {_l0_field_measured/L0_FIELD_PHYS:.0f}x)")

# ─────────────────────────────────────────────────────────────────────────────
# Optimiser
# ─────────────────────────────────────────────────────────────────────────────
# Warm start, part 2: the weights, now that L0 has been measured at a fresh
# initialisation. load_state_dict copies in place, so the compiled modules keep
# pointing at the same parameter tensors.
if _INIT_CK is not None:
    raw_module(_don_int).load_state_dict(_INIT_CK['net_int_state_dict'])
    raw_module(net_scat).load_state_dict(_INIT_CK['net_scat_state_dict'])
    print(f"  warm start: network weights loaded from {INIT_WEIGHTS} "
          f"(L0 above is from the random initialisation, as in a fresh run)")
    if CONTINUE_RUN:
        os.makedirs(MODEL_DIR, exist_ok=True)
        shutil.copy2(INIT_WEIGHTS, PREV_CKPT_PATH)
        print(f"    continuing this tag; the weights it starts from are kept at "
              f"{PREV_CKPT_PATH}")
    _INIT_CK = None

all_params = list(net_int.parameters()) + list(net_scat.parameters())
optimizer  = torch.optim.Adam(all_params, lr=LR_ADAM)
# max(..., 1): a warm start runs ADAM_EPOCHS = 0, and a zero-length schedule
# has no meaning -- the scheduler is simply never stepped then.
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(ADAM_EPOCHS, 1), eta_min=1e-5)

# ─────────────────────────────────────────────────────────────────────────────
# Frame capture helpers (MP4 animation)
# ─────────────────────────────────────────────────────────────────────────────
gif_frames = []
N_grid_gif = 200
_FW, _FH   = 912 / 90, 816 / 90


def save_frame(label, total_loss, pi_sample):
    """4-panel field plot for one representative (half-angle, height)."""
    pi_sample = int(pi_sample)
    alpha  = float(half_angle[pi_sample])
    apex_y = float(apex_y_all[pi_sample])
    phys_feats = param_to_features(alpha, apex_y)

    net_int.eval(); net_scat.eval()
    with torch.no_grad():
        xg = np.linspace(0, 1, N_grid_gif)
        yg = np.linspace(0, 1, N_grid_gif)
        Xg_f, Yg_f = np.meshgrid(xg, yg)
        in_tip_f = tip_mask(Xg_f, Yg_f, alpha, apex_y)
        xy_ev_f = torch.tensor(
            np.stack([Xg_f.ravel(), Yg_f.ravel()], axis=1).astype(np.float32),
            device=DEVICE)
        iR_g, iI_g = net_int(phys_feats, xy_ev_f, geom_row(alpha, apex_y))
        sR_g, sI_g = net_scat(phys_feats, xy_ev_f)

    U_int_f  = (iR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * iI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))
    U_scat_f = (sR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * sI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))
    _bga_f    = (np.exp(1j * (kx_inc * Xg_f + ky_inc * (Yg_f - Y_SUB))) +
                 r_TM * np.exp(1j * (kx_inc * Xg_f - ky_inc * (Yg_f - Y_SUB))))
    _bgs_f    = t_TM * np.exp(1j * (kx_inc * Xg_f + _ky_sub * (Yg_f - Y_SUB)))
    U_inc_f   = np.where(Yg_f < Y_SUB, _bgs_f, _bga_f)
    U_total_f = np.where(in_tip_f, U_int_f, U_inc_f + U_scat_f)
    U_scat_plot_f = U_total_f - U_inc_f   # scattered = total - incident

    matplotlib.rcParams.update({'font.size': 9})
    fig_f, axes_f = plt.subplots(2, 2, figsize=(_FW, _FH))
    fig_f.suptitle(f"{label}  |  α={np.rad2deg(alpha):.2f}°  |  "
                   f"Loss={total_loss:.3e}", fontsize=12)
    fig_f.tight_layout(rect=[0, 0, 1, 0.96])
    ox, oy = tip_outline_xy(alpha, apex_y)
    outline_kw_f = dict(color='cyan', lw=1.5, zorder=5, ls='--')
    sub_kw_f     = dict(color='yellow', lw=1.2, zorder=6, ls='--')

    def _panel(ax_, data, title, cmap, symmetric):
        valid = data[np.isfinite(data)]
        scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
        scale = max(scale, 1e-12)
        vmin  = -scale if symmetric else 0.0
        im    = ax_.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                           cmap=cmap, vmin=vmin, vmax=scale,
                           interpolation='bilinear')
        ax_.plot(ox, oy, **outline_kw_f)
        ax_.axhline(Y_SUB, **sub_kw_f)      # sample surface
        ax_.set_title(title, fontsize=10)
        ax_.set_xlabel("x"); ax_.set_ylabel("y")
        ax_.set_xlim(0, 1); ax_.set_ylim(0, 1)
        fig_f.colorbar(im, ax=ax_, shrink=0.90)

    _panel(axes_f[0, 0], U_scat_plot_f.real,    "Scattered   Re(u_scat)",  "RdBu_r",  True)
    _panel(axes_f[0, 1], U_total_f.real,        "Total       Re(u_total)", "RdBu_r",  True)
    _panel(axes_f[1, 0], np.abs(U_scat_plot_f), "Scattered   |u_scat|",    "hot",     False)
    _panel(axes_f[1, 1], np.abs(U_total_f),     "Total       |u_total|",   "inferno", False)

    buf = BytesIO()
    fig_f.savefig(buf, format='png', dpi=90)
    plt.close(fig_f)
    buf.seek(0)
    gif_frames.append(imageio.imread(buf))
    net_int.train(); net_scat.train()


# Representative parameter for frame capture — middle of training set
_pi_mid = int(train_idx[len(train_idx) // 2])

# ─────────────────────────────────────────────────────────────────────────────
# History buffers
# ─────────────────────────────────────────────────────────────────────────────
loss_history   = []
# x-coordinate for every loss sample, in the same units as hist_eval_ep so the
# loss plot and the relL2 plot share an axis (see the Phase 2 note on why that
# unit is cumulative closure evals, not outer steps).
hist_step      = []
hist_pde_int   = []
hist_pde_ext   = []
hist_pde_sub   = []
hist_field     = []
hist_flux      = []
hist_pml       = []
hist_dirichlet = []
hist_data_scat = []
hist_data_int  = []
hist_data_sub  = []

# ── a continuation picks up the previous run's step count ───────────────────
# STEP_OFFSET puts this run's steps after the last one's, so relL2, amplitude
# and the CSV are cumulative over the whole training. The loss lists above stay
# EMPTY: this run's loss figure covers this run only -- see CONTINUE_RUN.
_HIST_KEYS = ("loss_history", "hist_step", "hist_pde_int", "hist_pde_ext",
              "hist_pde_sub", "hist_field", "hist_flux", "hist_pml",
              "hist_dirichlet", "hist_data_scat", "hist_data_int",
              "hist_data_sub")
STEP_OFFSET = 0
_PREV_EVAL_ROWS = []
if CONTINUE_RUN:
    _csv_prev = f"{FIG_DIR}/{RUN_TAG}_relL2.csv"
    if os.path.exists(_csv_prev):
        with open(_csv_prev, newline="") as _f:
            _PREV_EVAL_ROWS = list(csv.DictReader(_f))
        if _PREV_EVAL_ROWS:
            STEP_OFFSET = int(float(_PREV_EVAL_ROWS[-1]["step"]))
    if STEP_OFFSET:
        print(f"  continuing: {len(_PREV_EVAL_ROWS)} earlier evaluations, "
              f"this run's steps start at {STEP_OFFSET}")
    else:
        print(f"  continuing, but {_csv_prev} has no earlier evaluations to "
              f"carry on from; this run's steps start at 0")

# Suffix that keeps a continuation's own-run outputs beside the earlier ones.
_SEG = f"_from{STEP_OFFSET}" if STEP_OFFSET else ""

# ─────────────────────────────────────────────────────────────────────────────
# Periodic relL2 monitoring
#
# The loss is a poor proxy for solution quality here -- across runs the flux
# term has been ANTI-correlated with relL2 -- so a long run needs the actual
# metric tracked as it goes, not just at the end. This reuses exactly the
# definition used in the final report (relative_l2_vs_fdm) so the numbers are
# directly comparable.
#
# Cost: forward-only on MON_N^2 points per test angle, no autograd. That is
# well under one training epoch's work, so at EVAL_EVERY = 2000 the overhead is
# a few hundredths of a percent.
# ─────────────────────────────────────────────────────────────────────────────
EVAL_EVERY = 2000
EVAL_EVERY_LBFGS = 25     # in outer L-BFGS steps -- see Phase 2 loop below.
# Matches afm_pinn/afm_pinn.py's cadence. At 500 a 100k-eval budget produced
# only ~8 points across the entire L-BFGS phase, so the phase that does most of
# the work was the least resolved part of the curve. The monitor is forward-only
# on MON_N^2 points per test angle, so ~160 evaluations over a long run stay
# negligible against the training cost.
MON_N      = 200          # same grid as the final evaluation

_mg = np.linspace(0, 1, MON_N)
MON_Xg, MON_Yg = np.meshgrid(_mg, _mg)
MON_xy = torch.tensor(
    np.stack([MON_Xg.ravel(), MON_Yg.ravel()], axis=1).astype(np.float32),
    device=DEVICE)
_mba = (np.exp(1j * (kx_inc * MON_Xg + ky_inc * (MON_Yg - Y_SUB))) +
        r_TM * np.exp(1j * (kx_inc * MON_Xg - ky_inc * (MON_Yg - Y_SUB))))
_mbs = t_TM * np.exp(1j * (kx_inc * MON_Xg + _ky_sub * (MON_Yg - Y_SUB)))
MON_U_INC   = np.where(MON_Yg < Y_SUB, _mbs, _mba)
MON_NRM_INC = float(np.linalg.norm(MON_U_INC))

# FDM reference interpolated onto the monitor grid, once
_mon_coords = np.array([MON_Yg.ravel() * (FDM_N - 1),
                        MON_Xg.ravel() * (FDM_N - 1)])
MON_FDM = {}
for _pi in range(N_PARAM):
    _re = map_coordinates(FDM_scat_re[_pi], _mon_coords, order=1, mode='nearest')
    _im = map_coordinates(FDM_scat_im[_pi], _mon_coords, order=1, mode='nearest')
    MON_FDM[_pi] = (_re + 1j * _im).reshape(MON_N, MON_N)

# what ||u_total||/||u_inc|| SHOULD be, from the same reference
MON_TOT_REF = float(np.mean([
    np.linalg.norm(MON_U_INC + MON_FDM[int(p)]) / MON_NRM_INC for p in test_idx]))


def monitor_metrics(idxs):
    """(mean relL2 vs FDM, mean ||u_total||/||u_inc||) over the given angles."""
    net_int.eval(); net_scat.eval()
    rels, mags = [], []
    with torch.no_grad():
        for pi in idxs:
            pi    = int(pi)
            alpha  = float(half_angle[pi])
            apex_y = float(apex_y_all[pi])
            pf     = param_to_features(alpha, apex_y)
            iR, iI = net_int(pf, MON_xy, geom_row(alpha, apex_y))
            sR, sI = net_scat(pf, MON_xy)
            U_int  = (iR.cpu().numpy().reshape(MON_N, MON_N) +
                      1j * iI.cpu().numpy().reshape(MON_N, MON_N))
            U_scat = (sR.cpu().numpy().reshape(MON_N, MON_N) +
                      1j * sI.cpu().numpy().reshape(MON_N, MON_N))
            # U_total: net_int inside the tip, u_inc + net_scat outside
            U_total = np.where(tip_mask(MON_Xg, MON_Yg, alpha, apex_y),
                               U_int, MON_U_INC + U_scat)
            U_sc    = U_total - MON_U_INC        # composite, matches FDM's u_scat
            fdm     = MON_FDM[pi]
            rels.append(np.linalg.norm(U_sc - fdm) /
                        (np.linalg.norm(fdm) + 1e-12))
            mags.append(np.linalg.norm(U_total) / MON_NRM_INC)
    net_int.train(); net_scat.train()
    return float(np.mean(rels)), float(np.mean(mags))


hist_eval_ep, hist_eval_rel, hist_eval_mag = [], [], []
for _r in _PREV_EVAL_ROWS:            # empty unless this run continues another
    hist_eval_ep.append(float(_r["step"]))
    hist_eval_rel.append(float(_r["relL2"]))
    hist_eval_mag.append(float(_r["mag_ratio"]))
print(f"\n  relL2 monitor: every {EVAL_EVERY} epochs on {len(test_idx)} test "
      f"angles, {MON_N}x{MON_N} grid")
print(f"  ||u_total||/||u_inc|| target from FDM = {MON_TOT_REF:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Adam
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()

# -- relL2 log ---------------------------------------------------------------
# Every figure here is written only after training finishes, so a run stopped
# early -- the normal way to act on a curve that has already gone wrong, and on
# a budget of several hundred thousand closure evaluations the normal way to
# act on one that has flattened -- left nothing but console scrollback. Each
# monitor evaluation is appended here and flushed immediately, so the file is
# complete and readable at any instant, including after a kill.
#
# The w_* columns are the NORMALISED, WEIGHTED contributions W[i] * term / L0[i]
# -- the numbers that actually sum to `total`, and so the ones that say which
# term the optimiser is trading when the loss falls and relL2 does not. The raw
# terms stay on the console and in the loss figure; these are not those.
#
# Both phases go through log_eval, so the file and the console cannot disagree
# about what was measured, and runs can be compared by reading the CSVs rather
# than by eye across figures -- which is what the five data-quantity arms need.
EVAL_CSV = f"{FIG_DIR}/{RUN_TAG}_relL2.csv"
_EVAL_TERMS = ("pde_int", "pde_ext", "pde_sub", "field", "flux", "pml",
               "dirichlet", "data_scat", "data_int", "data_sub")
if _PREV_EVAL_ROWS:
    # append: the header and the earlier rows stay, so one CSV holds the whole
    # training. The w_* columns are W*loss/L0, and a continuation measures its
    # own L0 at a fresh initialisation, so those are not comparable across the
    # restart -- the phase column and the step jump mark where it happened.
    _eval_csv = open(EVAL_CSV, "a", buffering=1)
    print(f"\n  relL2 log     : {EVAL_CSV}  (appending to "
          f"{len(_PREV_EVAL_ROWS)} earlier evaluations)")
else:
    _eval_csv = open(EVAL_CSV, "w", buffering=1)
    _eval_csv.write("phase,step,relL2,mag_ratio,target_mag,seconds,total,"
                    + ",".join("w_" + _t for _t in _EVAL_TERMS) + "\n")
    print(f"\n  relL2 log     : {EVAL_CSV}")


def log_eval(phase, step, rel, mag, vals):
    """One monitor evaluation -> the history lists and a flushed CSV row.

    `step` is the relL2 plot's x coordinate: the Adam epoch, or ADAM_EPOCHS
    plus the L-BFGS closure-evaluation count. The phase column tells them
    apart. `vals` is (total,) + the ten raw terms, in the order the loss
    functions return them."""
    # A NaN never recovers: the weights are already gone, and the rest of the
    # budget would grind out NaN rows and NaN figures on top of them. Stop, and
    # leave BEST_CKPT_PATH holding the last good model.
    if not (np.isfinite(rel) and np.isfinite(mag) and np.isfinite(vals[0])):
        print(f"  !! non-finite monitor at step {step}: relL2={rel}, "
              f"total={vals[0]} -- stopping this run.")
        if _BEST["step"] is not None:
            print(f"     the best weights (relL2 {_BEST['rel']:.4f} at step "
                  f"{_BEST['step']}) are kept at {BEST_CKPT_PATH}")
        _STOP[0] = True
        return
    hist_eval_ep.append(step)
    hist_eval_rel.append(rel)
    hist_eval_mag.append(mag)
    _w = (W_PDE_INT, W_PDE_EXT, W_PDE_SUB, W_FIELD, W_FLUX, W_PML,
          W_DIRICHLET, W_DATA_SCAT, W_DATA_INT, W_DATA_SUB)
    _l0 = (L0_PDE_INT, L0_PDE_EXT, L0_PDE_SUB, L0_FIELD, L0_FLUX, L0_PML,
           L0_DIRICHLET, L0_DATA_SCAT, L0_DATA_INT, L0_DATA_SUB)
    contrib = [w * v / l0 for w, v, l0 in zip(_w, vals[1:], _l0)]
    _eval_csv.write(f"{phase},{step},{rel:.6e},{mag:.6e},{MON_TOT_REF:.6e},"
                    f"{time.time() - t0:.1f},{vals[0]:.6e},"
                    + ",".join(f"{c:.6e}" for c in contrib) + "\n")
    _eval_csv.flush()
    # TRUNK_NORM's gradient scales like 1/sigma, and sigma DRIFTS DOWN as
    # training divides the trunk's scale away -- which is how two runs blew up.
    # Watching it here makes that visible while it happens.
    _s = trunk_sigma_min()
    if np.isfinite(_s):
        print(f"     trunk sigma_min {_s:.2e}  -> gradient amplification up to "
              f"{1.0 / np.sqrt(_s**2 + TRUNK_NORM_EPS):.0f}x")
    # Keep the best weights the monitor has seen. CKPT_PATH is only ever the
    # LAST state, and a run can end far worse than it was -- see BEST_CKPT_PATH.
    _ck = None
    if rel < _BEST["rel"]:
        _BEST.update(rel=rel, step=step)
        os.makedirs(MODEL_DIR, exist_ok=True)
        _ck = checkpoint_dict()
        _save_atomic({**_ck, 'best_relL2': rel, 'best_step': step},
                     BEST_CKPT_PATH)
    if SAVE_LATEST:
        # the state as of THIS evaluation, whether or not it is the best
        os.makedirs(MODEL_DIR, exist_ok=True)
        _save_atomic({**(_ck if _ck is not None else checkpoint_dict()),
                      'latest_relL2': rel, 'latest_step': step},
                     LATEST_CKPT_PATH)


_BEST = {"rel": float("inf"), "step": None}
_STOP = [False]          # set by log_eval when the monitor goes non-finite


def trunk_sigma_min():
    """Smallest spread across the trunk's modes over the monitor grid.

    Only meaningful under TRUNK_NORM, which divides by sqrt(sigma^2 + eps) and
    whose gradient therefore scales like 1/sigma. nan when it does not apply."""
    if ARCH != "deeponet" or not TRUNK_NORM:
        return float("nan")
    _tr = raw_module(net_scat).trunk
    with torch.no_grad():
        _t = _tr.net(_tr.embed(MON_xy))
    return float(_t.std(dim=1, unbiased=False).min())


def _save_atomic(obj, path):
    """torch.save through a temporary file in the same directory, then replace.
    A run writes these while other machines read them, and os.replace is atomic,
    so a reader sees either the old checkpoint or the new one, never a partial."""
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checkpoint_dict():
    """Everything a consumer needs to rebuild and interpret these weights.

    One definition for both the final save and every best-so-far save, so the
    two cannot drift apart."""
    return {
        # _don_int, not net_int: with SKIN_ANSATZ the latter is the wrapper,
        # whose state_dict would prefix every key with 'don.' and break
        # evaluate_final.py. The wrapper holds no parameters, so saving the
        # inner DeepONet loses nothing and keeps one checkpoint format for both
        # settings of the flag.
        'net_int_state_dict':  raw_module(_don_int).state_dict(),
        'net_scat_state_dict': raw_module(net_scat).state_dict(),
        'B_ext': B_ext.cpu(),
        'B_int': B_int.cpu(),
        'FEAT_MIN':   FEAT_MIN,
        'FEAT_MAX':   FEAT_MAX,
        'half_angle': half_angle,
        'apex_y': apex_y_all,
        'train_idx':  train_idx,
        'test_idx':   test_idx,
        'hyperparams': {
            # N_FOURIER is the WAVELENGTH band only; B_ext/B_int also carry a
            # tip-scale band, so the trunk input is 2*(N_FOURIER +
            # N_FOURIER_TIP). Consumers should take the width from B.shape[1]
            # rather than from these -- evaluate_final.py does.
            'N_FOURIER': N_FOURIER, 'N_FOURIER_TIP': N_FOURIER_TIP,
            'K_TIP': K_TIP, 'PHYS_DIM': PHYS_DIM,
            # net_int's output means different things under these two settings
            # -- u_int itself when False, the envelope A when True -- so any
            # consumer of net_int_state_dict must read this before interpreting
            # it.
            'SKIN_ANSATZ': SKIN_ANSATZ, 'SHELL_DEPTH': SHELL_DEPTH,
            # net_int/net_scat are MLPNet, not DeepONet, when this is 'mlp'
            'ARCH': ARCH, 'MLP_HIDDEN': MLP_HIDDEN, 'MLP_NEURONS': MLP_NEURONS,
            'BRANCH_HIDDEN': BRANCH_HIDDEN, 'BRANCH_NEURONS': BRANCH_NEURONS,
            'TRUNK_HIDDEN': TRUNK_HIDDEN, 'TRUNK_NEURONS': TRUNK_NEURONS,
            'BRANCH_DIM': BRANCH_DIM, 'TRUNK_SEES_SDF': TRUNK_SEES_SDF,
            # read back by evaluate_final.build_nets: with it on, the trunk
            # output is LayerNorm'd (at this eps), and scoring without that
            # would rebuild a different net
            'TRUNK_NORM': TRUNK_NORM, 'TRUNK_NORM_EPS': TRUNK_NORM_EPS,
            'BRANCH_NORM': BRANCH_NORM,
            # physics imposed at this many data-free parameter draws per
            # point-set, weighted PHYS_EXTRA_WEIGHT (see N_PHYS_EXTRA)
            'N_PHYS_EXTRA': N_PHYS_EXTRA,
            'PHYS_EXTRA_WEIGHT': PHYS_EXTRA_WEIGHT,
            'PHYS_SPAN_ALPHA': PHYS_SPAN_ALPHA,
            'PHYS_SPAN_APEX': PHYS_SPAN_APEX,
            'TIP_X': TIP_X, 'R_APEX': R_APEX,
            'SUB_N_RE': SUB_N_RE, 'SUB_N_IM': SUB_N_IM, 'Y_SUB': Y_SUB,
            'N_RE': N_RE, 'N_IM': N_IM, 'lam': lam,
            'ADAM_EPOCHS': ADAM_EPOCHS, 'LBFGS_STEPS': LBFGS_STEPS,
            'W_PDE_INT': W_PDE_INT, 'W_PDE_EXT': W_PDE_EXT,
            'W_PDE_SUB': W_PDE_SUB,
            'W_FIELD': W_FIELD, 'W_FLUX': W_FLUX,
            'W_PML': W_PML, 'W_DIRICHLET': W_DIRICHLET,
            'W_DATA_SCAT': W_DATA_SCAT, 'W_DATA_INT': W_DATA_INT,
            'W_DATA_SUB': W_DATA_SUB,
            'D_PML': D_PML, 'PML_POW': PML_POW,
            'PML_SIGMA_MAX': PML_SIGMA_MAX, 'N_PML': N_PML,
            'MARGIN_IN': _MARGIN_IN, 'MARGIN_OUT': _MARGIN_OUT,
            'IFACE_TOP_EXCLUDE': IFACE_TOP_EXCLUDE,
            # where this run's weights started: None = random initialisation
            'INIT_WEIGHTS': INIT_WEIGHTS,
        },
    }


# A PREVIOUS RUN'S BEST IS NEVER OVERWRITTEN. The best file is named after the
# tag, so re-running a tag used to destroy it -- which cost two good models,
# one at relL2 0.046 and one at 0.025, when later attempts under the same tag
# blew up. A continuation instead INHERITS the standing best, so only a genuine
# improvement replaces it.
os.makedirs(MODEL_DIR, exist_ok=True)
if os.path.exists(BEST_CKPT_PATH):
    try:
        _prev_rel = float(torch.load(BEST_CKPT_PATH, map_location="cpu",
                                     weights_only=False).get("best_relL2",
                                                             float("inf")))
    except Exception as _exc:
        _prev_rel = float("inf")
        print(f"  existing best checkpoint unreadable ({type(_exc).__name__}); "
              f"archiving it rather than trusting it")
    if CONTINUE_RUN:
        _BEST["rel"] = _prev_rel
        print(f"  continuing: the standing best is relL2 {_prev_rel:.4f}; only "
              f"an improvement on that will replace it")
    else:
        _i = 1
        while os.path.exists(f"{MODEL_DIR}/{RUN_TAG}_best_prev{_i}_weights.pt"):
            _i += 1
        _archived = f"{MODEL_DIR}/{RUN_TAG}_best_prev{_i}_weights.pt"
        os.replace(BEST_CKPT_PATH, _archived)
        print(f"  an earlier run's best (relL2 {_prev_rel:.4f}) is kept at "
              f"{_archived}; this run starts its own")


if INIT_WEIGHTS:
    # Where the loaded model starts, on THIS run's loss and monitor, so the relL2
    # curve and CSV open at the checkpoint instead of at the first L-BFGS eval.
    _b0 = np.random.choice(train_idx, size=N_PARAM_BATCH, replace=False)
    _v0 = tuple(v.item() for v in backward_batch(
        _b0, sample_pml(N_PML), sample_dirichlet(N_DIR_SIDE)))
    for _p in all_params:
        _p.grad = None                  # evaluation only: seed no real gradient
    pts_cache.clear()
    pml_parts_cache.clear()
    batch_cache.clear()
    _rel0, _mag0 = monitor_metrics(test_idx)
    log_eval("warm", STEP_OFFSET, _rel0, _mag0, _v0)
    print(f"  [warm start]  relL2(test) = {_rel0:.4f}   "
          f"||u_total||/||u_inc|| = {_mag0:.4f}  (target {MON_TOT_REF:.4f})")
    save_frame("Warm start", _v0[0], _pi_mid)

print(f"\n--- Phase 1: Adam ({ADAM_EPOCHS} epochs) ---")
print(f"{'Ep':>5} | {'Total':>8} | {'PDEin':>8} | {'PDEex':>8} | "
      f"{'Field':>8} | {'Flux':>8} | {'PML':>8} | {'Dirich':>8} | {'s':>5}")
print("-" * 100)

RESAMPLE_ADAM = 250  # resample spatial points every N Adam epochs (afm_pinn)

pml_sample = sample_pml(N_PML)
xy_dir     = sample_dirichlet(N_DIR_SIDE)
draw_phys_geoms()

for ep in range(1, ADAM_EPOCHS + 1):

    # ── resample: clear per-parameter caches; shared sets re-drawn ───────────
    if ep % RESAMPLE_ADAM == 0:
        pts_cache.clear()
        pml_parts_cache.clear()
        batch_cache.clear()
        pml_sample = sample_pml(N_PML)
        xy_dir     = sample_dirichlet(N_DIR_SIDE)
        draw_phys_geoms()

    # ── mini-batch of half-angles from the training split ────────────────────
    batch_idx = np.random.choice(train_idx, size=N_PARAM_BATCH, replace=False)

    optimizer.zero_grad()
    tot, lpi, lpe, lps, lf, lfl, lpml, ldir, lds, ldi, ldsb = tuple(
        v.item() for v in backward_batch(batch_idx, pml_sample, xy_dir))
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()
    scheduler.step()

    loss_history.append(tot)
    hist_step.append(STEP_OFFSET + ep)
    hist_pde_int.append(lpi);   hist_pde_ext.append(lpe)
    hist_pde_sub.append(lps)
    hist_field.append(lf);      hist_flux.append(lfl)
    hist_pml.append(lpml);      hist_dirichlet.append(ldir)
    hist_data_scat.append(lds); hist_data_int.append(ldi)
    hist_data_sub.append(ldsb)

    if ep % 250 == 0 or ep == 1:
        _flux = radial_flux_diag(_pi_mid)
        print(f"{ep:>5} | {tot:>8.2e} | {lpi:>8.2e} | {lpe:>8.2e} | "
              f"{lps:>8.2e} | "
              f"{lf:>8.2e} | {lfl:>8.2e} | {lpml:>8.2e} | {ldir:>8.2e} | "
              f"{time.time()-t0:>5.1f}   flux(out>0)={_flux:+.2e}"
              + (f"   phys@unseen {_PHYS_EXTRA_LAST[0]:.2e}"
                 if N_PHYS_EXTRA else ""))

    if ep == 1 or ep % EVAL_EVERY == 0:
        _rel, _mag = monitor_metrics(test_idx)
        log_eval("adam", STEP_OFFSET + ep, _rel, _mag,
                 (tot, lpi, lpe, lps, lf, lfl, lpml, ldir, lds, ldi, ldsb))
        print(f"  [eval ep {ep:>6}]  relL2(test) = {_rel:.4f}   "
              f"||u_total||/||u_inc|| = {_mag:.4f}  (target {MON_TOT_REF:.4f})")

    if ep == 1 or ep % 250 == 0:
        save_frame(f"Adam epoch {ep:>5}", tot, _pi_mid)

    if _STOP[0]:            # the monitor went non-finite; see log_eval
        break

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: L-BFGS  (float32, eval-bounded)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Phase 2: L-BFGS ({LBFGS_STEPS} closure evals, float32) ---")

# CADENCE HELD AT afm_pinn's 25.
#
# A previous edit moved the
# cadence to 100 as well, on a justification that turned out to be false: the
# note claimed "afm_tip_pinn.py resamples every 100 outer steps and converges",
# whereas afm_tip/afm_tip_pinn.py uses 25, having moved OFF 100 because ~3000
# evals per point set was "looser, and more exposed to memorising a frozen
# collocation set". afm_pinn/afm_pinn.py -- the run that reaches relL2 0.084 --
# also uses 25 and rebuilds the optimiser every time.
#
# CARRYING THE CURVATURE HISTORY ACROSS A REDRAW WAS TESTED AND REVERTED.
# Runs with it on and with it off both came out worse than this 0.296
# configuration, so the optimiser is rebuilt at every redraw exactly as
# afm_pinn/afm_pinn.py and afm_tip/afm_tip_pinn.py do. The null result is
# consistent with what the mechanism could buy, which was never much: with
# max_iter = 20 a single outer step appends up to 20 (s, y) pairs, so the
# history_size = 50 buffer fills after ~3 outer steps and is full for ~22 of
# every 25. Rebuilding costs ~60 evals of degraded steps per 500 -- a ~12% duty
# cycle, not a permanent absence of curvature -- and since the buffer is a
# rolling window either way, keeping it does not extend how far back the model
# reaches.
RESAMPLE_LBFGS = 25   # resample every N outer steps (afm_pinn's cadence).

# At max_iter = 20 and a 50,000-eval budget (~2500 outer steps) a cadence of 25
# gives ~100 redraws, i.e. ~500 closure evals per point set -- the same exposure
# afm_pinn and afm_tip_pinn run at. (Earlier text here claimed afm_tip_pinn.py
# used 100 and converged; it does not, see the RESAMPLE_LBFGS note above.)
# Freezing the objective for the WHOLE run lets L-BFGS memorise the collocation
# set instead of solving the PDE: with 10000 the reported flux term reached
# 2.5e-2 on the frozen points but was 1.8e+01 at freshly drawn interface points
# (~700x worse), while the field term generalised (1.4e-3 -> 3.0e-3). The
# parameter batch is redrawn here too -- it was previously also drawn once, so
# 12 of the 20 angles never entered the L-BFGS phase at all.

# ── Precision: float32 throughout, as in afm_tip_pinn.py ─────────────────────
# This phase previously ran in float64. On a consumer Ampere card (RTX 3080 Ti)
# FP64 executes at 1/64 of the FP32 rate, which dominated the whole training
# cost. It bought no accuracy: evaluating every residual term at identical
# points with identical weights in both precisions agrees to <= 8.1e-6 relative
#   pde_int 4.915125e-04 (f64) vs 4.915165e-04 (f32)
#   pde_ext 1.553128e-05  |  field 1.408736e-01  |  flux 1.706132e+01  (< 5e-7)
# The normalised float32 noise floor is 1.19e-07 (squared 1.4e-14), far below
# any term training reaches. The apparent need for float64 was a symptom of the
# frozen collocation set, not of precision.


def make_closure(param_batch, pml_c, xy_dir_c, last_terms):
    def closure():
        lbfgs.zero_grad()
        vals = [v.item() for v in backward_batch(param_batch, pml_c, xy_dir_c)]
        # No gradient clipping here -- clipping inside an L-BFGS closure corrupts
        # the strong-Wolfe curvature check and stalls the line search.
        step_count[0] += 1
        last_terms[:] = vals
        return torch.tensor(vals[0], device=DEVICE)
    return closure


step_count = [0]
outer_step = 0
lbfgs      = None
last_terms = [0.0] * 10

# NOTE ON THE BUDGET: this loop is bounded by CLOSURE EVALUATIONS, not by outer
# steps. `while outer_step < LBFGS_STEPS` with max_iter=30 meant >= 3000*30 =
# 90,000 inner iterations plus strong-Wolfe line-search probes on top -- a ~3x
# silent increase over the eval-bounded form, while the constant kept its name.
# step_count[0] is incremented once per closure call and printed as "evals".
while step_count[0] < LBFGS_STEPS and not _STOP[0]:

    if outer_step % RESAMPLE_LBFGS == 0:
        param_batch_lb = np.random.choice(train_idx, size=N_PARAM_BATCH,
                                          replace=False)
        pts_cache.clear()
        pml_parts_cache.clear()
        batch_cache.clear()
        pml_lb    = sample_pml(N_PML)
        xy_dir_lb = sample_dirichlet(N_DIR_SIDE)
        for pi in param_batch_lb:
            get_pts(int(pi))
            get_pml_parts(int(pi), pml_lb)
        draw_phys_geoms()

        # lr and history_size matched to afm_pinn/afm_pinn.py (0.1, 50). With
        # SINGLE_GEOMETRY the parameter batch redrawn just above is always the
        # same index, so unlike the sweep run every block continues on the SAME
        # objective and the curvature history is rebuilt on a consistent
        # problem -- which is the condition afm_pinn's L-BFGS phase enjoyed.
        lbfgs = torch.optim.LBFGS(all_params, lr=0.1, max_iter=LBFGS_MAX_ITER,
                                  history_size=50, line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-12, tolerance_change=1e-16)

    closure = make_closure(param_batch_lb, pml_lb, xy_dir_lb, last_terms)
    lbfgs.step(closure)
    outer_step += 1

    tot, lpi, lpe, lps, lf, lfl, lpml, ldir, lds, ldi, ldsb = last_terms
    loss_history.append(tot)
    hist_step.append(STEP_OFFSET + ADAM_EPOCHS + step_count[0])
    hist_pde_int.append(lpi);   hist_pde_ext.append(lpe)
    hist_pde_sub.append(lps)
    hist_field.append(lf);      hist_flux.append(lfl)
    hist_pml.append(lpml);      hist_dirichlet.append(ldir)
    hist_data_scat.append(lds); hist_data_int.append(ldi)
    hist_data_sub.append(ldsb)

    # The try/except guards are gone: they only existed because the nets were
    # float64 here while both helpers build float32 tensors internally, so every
    # attempt raised and was swallowed -- the L-BFGS phase contributed no frames
    # to the MP4 and reported flux(out>0)=nan throughout.
    if outer_step % 25 == 0:
        _flux = radial_flux_diag(_pi_mid)
        print(f"  outer {outer_step:>5} (evals {step_count[0]:>6}) | "
              f"loss {tot:.4e} | Field {lf:.2e} | Flux {lfl:.2e} | "
              f"PML {lpml:.2e} | Dirich {ldir:.2e} | {time.time()-t0:.1f}s   "
              f"flux(out>0)={_flux:+.2e}"
              + (f"   phys@unseen {_PHYS_EXTRA_LAST[0]:.2e}"
                 if N_PHYS_EXTRA else ""))
        save_frame(f"L-BFGS outer {outer_step}", tot, _pi_mid)

    # Same relL2-vs-FDM monitor as Phase 1, on the same global "optimiser step"
    # axis so hist_eval_* is one continuous series spanning Adam -> L-BFGS
    # instead of stopping at the phase boundary.
    # Gated on outer L-BFGS steps (EVAL_EVERY_LBFGS), not on the Adam-scaled
    # EVAL_EVERY -- outer_step tops out at a few thousand (LBFGS_STEPS bounds
    # closure evals, and each outer step burns several), so it would almost
    # never land on a multiple of EVAL_EVERY=2000.
    # The axis position is the cumulative CLOSURE-EVAL count, not outer_step:
    # one outer step is LBFGS_MAX_ITER inner iterations plus line-search probes,
    # so plotting against outer_step compressed the whole L-BFGS phase into a
    # sliver next to ADAM_EPOCHS and made Adam look like most of the run.
    _global_step = STEP_OFFSET + ADAM_EPOCHS + step_count[0]
    if outer_step % EVAL_EVERY_LBFGS == 0:
        _rel, _mag = monitor_metrics(test_idx)
        log_eval("lbfgs", _global_step, _rel, _mag,
                 (tot, lpi, lpe, lps, lf, lfl, lpml, ldir, lds, ldi, ldsb))
        print(f"  [eval step {_global_step:>6}]  relL2(test) = {_rel:.4f}   "
              f"||u_total||/||u_inc|| = {_mag:.4f}  (target {MON_TOT_REF:.4f})")

print(f"\nTotal training time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save model
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
torch.save(checkpoint_dict(), CKPT_PATH)
print(f"Saved: {CKPT_PATH}")
if _BEST["step"] is not None:
    print(f"Best relL2 {_BEST['rel']:.4f} at step {_BEST['step']}: "
          f"{BEST_CKPT_PATH}")
    if hist_eval_rel and _BEST["rel"] < 0.95 * hist_eval_rel[-1]:
        print(f"  NOTE: the run ended at {hist_eval_rel[-1]:.4f}, worse than its "
              f"best. The file above holds the better model.")


# ─────────────────────────────────────────────────────────────────────────────
# Training loss plot
# ─────────────────────────────────────────────────────────────────────────────
# The per-epoch loss history, saved because the figure below is only a picture
# of it and the lists are otherwise in memory only. One file per run segment,
# for the same reason the loss figure is per run (see CONTINUE_RUN).
_LOSS_HIST_PATH = f"{FIG_DIR}/{RUN_TAG}_loss_history{_SEG}.npz"
np.savez_compressed(_LOSS_HIST_PATH,
                    **{_k: np.asarray(globals()[_k], dtype=np.float32)
                       for _k in _HIST_KEYS})
print(f"Saved: {_LOSS_HIST_PATH}  ({len(hist_step)} loss samples)")


def _mark_phases(ax):
    """Adam -> L-BFGS, and where a continuation picked the run up."""
    if ADAM_EPOCHS:
        ax.axvline(STEP_OFFSET + ADAM_EPOCHS, color='gray', ls='--', lw=1.5,
                   label=f'Adam → L-BFGS (step {STEP_OFFSET + ADAM_EPOCHS})')
    if STEP_OFFSET:
        ax.axvline(STEP_OFFSET, color='black', ls='-.', lw=1.4,
                   label=f'continued from step {STEP_OFFSET}')


fig_loss, ax_loss = plt.subplots(figsize=(12, 5))
steps = hist_step
ax_loss.semilogy(steps, loss_history,   color='black',        lw=1.5, label='Total')
ax_loss.semilogy(steps, hist_pde_int,   color='royalblue',    lw=0.8, label='PDE int')
ax_loss.semilogy(steps, hist_pde_ext,   color='steelblue',    lw=0.8, label='PDE air')
ax_loss.semilogy(steps, hist_pde_sub,   color='darkorange',   lw=0.8, label='PDE sub')
ax_loss.semilogy(steps, hist_field,     color='tomato',       lw=0.8, label='Field BC')
ax_loss.semilogy(steps, hist_flux,      color='orange',       lw=0.8, label='Flux BC')
ax_loss.semilogy(steps, hist_pml,       color='mediumpurple', lw=1.2, label='PML')
ax_loss.semilogy(steps, hist_dirichlet, color='darkcyan',     lw=0.8, label='Dirichlet')
_mark_phases(ax_loss)
ax_loss.set_xlabel("cumulative step (Adam epochs, then L-BFGS closure evals)")
ax_loss.set_ylabel("Loss (log scale)")
ax_loss.set_title("Final AFM-tip+substrate DeepONet (half-angle sweep) — "
                  "Training Loss"
                  + (f"   (this run only, steps {STEP_OFFSET}+; L0 is measured "
                     f"per run, so earlier segments are not on this scale)"
                     if STEP_OFFSET else ""))
ax_loss.legend(fontsize=8, ncol=2)
ax_loss.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{RUN_TAG}_loss{_SEG}.png", dpi=150,
            bbox_inches='tight')
print(f"Saved: {FIG_DIR}/{RUN_TAG}_loss{_SEG}.png")

# ─────────────────────────────────────────────────────────────────────────────
# Convergence of the ACTUAL metric (see the monitor block above for why the
# loss curve is not a reliable stand-in for this)
# ─────────────────────────────────────────────────────────────────────────────
# relL2 and the amplitude ratio get their OWN figures. They were on one pair of
# twinned axes, which made both harder to read than either alone: two y-scales,
# one log and one linear, with the log axis's decade gridlines meaningless for
# the linear series. They also answer different questions -- relL2 is the
# accuracy, the ratio is a convergence diagnostic that says whether the field
# has reached its physical amplitude at all -- and are usually looked at one at
# a time. Both series are in <RUN_TAG>_relL2.csv if they need overlaying.
if hist_eval_ep:
    _x_label = "cumulative step (Adam epochs, then L-BFGS closure evals)"

    fig_ev, ax_ev = plt.subplots(figsize=(11, 5))
    ax_ev.semilogy(hist_eval_ep, hist_eval_rel, 'o-', color='crimson',
                   ms=4, lw=1.5, label='relL2 vs FDM (test angles)')
    _mark_phases(ax_ev)
    ax_ev.set_xlabel(_x_label)
    ax_ev.set_ylabel("relative L2 error vs FDM")
    ax_ev.grid(True, alpha=0.3, which='both')
    ax_ev.legend(loc='best', fontsize=9)
    ax_ev.set_title("Final AFM-tip+substrate DeepONet — convergence of relL2 "
                    "(Adam + L-BFGS)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{RUN_TAG}_relL2.png", dpi=150, bbox_inches='tight')
    plt.close(fig_ev)
    print(f"Saved: {FIG_DIR}/{RUN_TAG}_relL2.png")

    fig_mg, ax_mg = plt.subplots(figsize=(11, 5))
    ax_mg.plot(hist_eval_ep, hist_eval_mag, 's-', color='steelblue',
               ms=4, lw=1.2, label=r'$\|u_{tot}\|/\|u_{inc}\|$')
    ax_mg.axhline(MON_TOT_REF, color='k', ls=':', lw=1.5,
                  label=f'FDM target ({MON_TOT_REF:.4f})')
    _mark_phases(ax_mg)
    ax_mg.set_xlabel(_x_label)
    ax_mg.set_ylabel(r'$\|u_{tot}\|/\|u_{inc}\|$')
    ax_mg.grid(True, alpha=0.3)
    ax_mg.legend(loc='best', fontsize=9)
    ax_mg.set_title("Final AFM-tip+substrate DeepONet — field amplitude "
                    "against the FDM")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{RUN_TAG}_magnitude.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig_mg)
    print(f"Saved: {FIG_DIR}/{RUN_TAG}_magnitude.png")

    _eval_csv.close()
    print(f"Saved: {EVAL_CSV}  ({len(hist_eval_ep)} evaluations)")
    print(f"  relL2(test): {hist_eval_rel[0]:.4f} (ep {hist_eval_ep[0]}) -> "
          f"{hist_eval_rel[-1]:.4f} (ep {hist_eval_ep[-1]})")
    print(f"  ||u_tot||/||u_inc||: {hist_eval_mag[0]:.4f} -> "
          f"{hist_eval_mag[-1]:.4f}   (FDM target {MON_TOT_REF:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
N_grid = 200
xg = np.linspace(0, 1, N_grid)
yg = np.linspace(0, 1, N_grid)
Xg, Yg = np.meshgrid(xg, yg)
xy_ev  = torch.tensor(
    np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32),
    device=DEVICE)


def evaluate_fields(pi):
    """PINN (U_int, U_scat_comp, U_inc, U_total) on the eval grid for angle pi.
    U_scat_comp = U_total - U_inc EVERYWHERE (matches the FDM definition;
    inside the strongly absorbing tip net_scat is unconstrained, so the
    composite is the fair quantity to compare)."""
    pi = int(pi)
    alpha  = float(half_angle[pi])
    apex_y = float(apex_y_all[pi])
    phys_feats = param_to_features(alpha, apex_y)
    net_int.eval(); net_scat.eval()
    with torch.no_grad():
        iR_g, iI_g = net_int(phys_feats, xy_ev, geom_row(alpha, apex_y))
        sR_g, sI_g = net_scat(phys_feats, xy_ev)
    net_int.train(); net_scat.train()
    U_int  = (iR_g.cpu().numpy().reshape(N_grid, N_grid) +
              1j * iI_g.cpu().numpy().reshape(N_grid, N_grid))
    U_scat = (sR_g.cpu().numpy().reshape(N_grid, N_grid) +
              1j * sI_g.cpu().numpy().reshape(N_grid, N_grid))
    # Background: Fresnel, matching final_fdm.py
    _bg_air_g = (np.exp(1j * (kx_inc * Xg + ky_inc * (Yg - Y_SUB))) +
                 r_TM * np.exp(1j * (kx_inc * Xg - ky_inc * (Yg - Y_SUB))))
    _bg_sub_g = t_TM * np.exp(1j * (kx_inc * Xg + _ky_sub * (Yg - Y_SUB)))
    U_inc     = np.where(Yg < Y_SUB, _bg_sub_g, _bg_air_g)
    in_tip  = tip_mask(Xg, Yg, alpha, apex_y)
    U_total = np.where(in_tip, U_int, U_inc + U_scat)
    U_scat_comp = U_total - U_inc
    return U_int, U_scat_comp, U_inc, U_total


def fdm_scat_on_grid(pi):
    pi     = int(pi)
    scale  = FDM_N - 1
    coords = np.array([Yg.ravel() * scale, Xg.ravel() * scale])
    fdm_re = map_coordinates(FDM_scat_re[pi], coords, order=1, mode='nearest')
    fdm_im = map_coordinates(FDM_scat_im[pi], coords, order=1, mode='nearest')
    return (fdm_re + 1j * fdm_im).reshape(N_grid, N_grid)


def relative_l2_vs_fdm(pi):
    """Relative L2 error of PINN u_scat (= u_total - u_inc) against FDM."""
    _, U_scat_pinn, _, _ = evaluate_fields(pi)
    U_scat_fdm = fdm_scat_on_grid(pi)
    return float(np.linalg.norm(U_scat_pinn - U_scat_fdm) /
                 (np.linalg.norm(U_scat_fdm) + 1e-12))

# ─────────────────────────────────────────────────────────────────────────────
# Generalisation evaluation — ALL half-angles (train and test)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Generalisation Evaluation ---")
# With SINGLE_GEOMETRY only one index was ever trained; scoring the other 99
# would report the branch net extrapolating to parameters it has never seen,
# which is not what this run is asking.
_eval_pis  = [SINGLE_PI] if SINGLE_GEOMETRY else list(range(N_PARAM))
errors_all = np.full(N_PARAM, np.nan)
for pi in _eval_pis:
    errors_all[pi] = relative_l2_vs_fdm(pi)
    tag = "TEST " if pi in set(test_idx) else "train"
    print(f"  [{pi+1:>2}/{N_PARAM}] {tag}  α={alpha_deg[pi]:5.2f}°  "
          f"relL2={errors_all[pi]:.4f}")

test_errors  = errors_all[test_idx]
train_errors = errors_all[train_idx]
print(f"\n  Mean relL2 (test)  : {np.nanmean(test_errors):.4f}")
print(f"  Mean relL2 (train) : {np.nanmean(train_errors):.4f}")
print(f"  Max  relL2 (test)  : {np.nanmax(test_errors):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Error vs half-angle (1D parameter → line plot with test markers)
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(alpha_deg[train_idx], errors_all[train_idx], 'o-',
         color='steelblue', ms=5, lw=1.2, label='train')
ax1.plot(alpha_deg[test_idx], errors_all[test_idx], 'r*',
         ms=15, mec='white', zorder=5, label='test (interpolation)')
ax1.set_xlabel("Cone half-angle α  [deg]")
ax1.set_ylabel("Relative L2 error vs FDM u_scat")
ax1.set_title("Final AFM-tip+substrate DeepONet — error over the half-angle sweep")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{RUN_TAG}_error_vs_angle.png",
            dpi=130, bbox_inches='tight')
print(f"Saved: {FIG_DIR}/{RUN_TAG}_error_vs_angle.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — PINN sweep overviews: four 4×5 grids over all half-angles
# Same layout as the FDM sweep figures so the two can be compared side-by-side.
# ─────────────────────────────────────────────────────────────────────────────
# One row per half-angle, one column per height -- the same layout
# final_fdm.py uses, so the trained sweep and the reference sweep are
# directly comparable panel by panel.
N_ROWS, N_COLS = N_ALPHA_G, N_HEIGHT_G

_pinn_fields = {}
if not SINGLE_GEOMETRY:
    for pi in range(N_PARAM):
        _, U_s, _, U_t = evaluate_fields(pi)
        _pinn_fields[pi] = (U_s, U_t)


def plot_sweep_grid(get_data, title, cmap, symmetric, fname):
    matplotlib.rcParams.update({'font.size': 7})
    fig, axes_g = plt.subplots(N_ROWS, N_COLS,
                               figsize=(3 * N_COLS, 3 * N_ROWS), squeeze = False)
    outline_kw_g = dict(color='cyan', lw=1.0, zorder=5, ls='--')
    sub_kw_g     = dict(color='yellow', lw=0.8, zorder=6, ls='--')
    for pi in range(N_PARAM):
        ax_g  = axes_g[alpha_idx[pi], height_idx[pi]]
        data  = get_data(pi)
        scale = max(float(np.percentile(np.abs(data), 98)), 1e-12)
        vmin  = -scale if symmetric else 0.0
        ax_g.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                    cmap=cmap, vmin=vmin, vmax=scale,
                    interpolation='bilinear')
        ox, oy = tip_outline_xy(float(half_angle[pi]), float(apex_y_all[pi]))
        ax_g.plot(ox, oy, **outline_kw_g)
        ax_g.axhline(Y_SUB, **sub_kw_g)     # sample surface
        tag = " [TEST]" if pi in set(test_idx) else ""
        ax_g.set_title(f"α={alpha_deg[pi]:.2f}°{tag}  "
                       f"relL2={errors_all[pi]:.3f}", fontsize=6.5)
        ax_g.set_xticks([]); ax_g.set_yticks([])
        ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
    # A full tensor grid leaves no spare panels, but keep the guard in case
    # N_PARAM is ever a subset of N_ALPHA_G x N_HEIGHT_G.
    _used = {(int(alpha_idx[q]), int(height_idx[q])) for q in range(N_PARAM)}
    for _r in range(N_ROWS):
        for _c in range(N_COLS):
            if (_r, _c) not in _used:
                axes_g[_r, _c].axis('off')
    fig.suptitle(f"Final AFM-tip+substrate DeepONet (half-angle sweep) — {title}",
                 fontsize=11)
    plt.tight_layout()
    path = f"{FIG_DIR}/{RUN_TAG}_{fname}.png"
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


if SINGLE_GEOMETRY:
    print("Skipped: sweep-grid figures (SINGLE_GEOMETRY — 99 of 100 panels "
          "would show untrained parameters)")
else:
    plot_sweep_grid(lambda pi: _pinn_fields[pi][0].real,
                    "Re(u_scat)", "RdBu_r", True, "sweep_scat_re")
    plot_sweep_grid(lambda pi: np.abs(_pinn_fields[pi][0]),
                    "|u_scat|", "hot", False, "sweep_scat_mag")
    plot_sweep_grid(lambda pi: _pinn_fields[pi][1].real,
                    "Re(u_total)", "RdBu_r", True, "sweep_total_re")
    plot_sweep_grid(lambda pi: np.abs(_pinn_fields[pi][1]),
                    "|u_total|", "inferno", False, "sweep_total_mag")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Field comparison: PINN vs FDM for the test half-angles
# ─────────────────────────────────────────────────────────────────────────────
sorted_test = test_idx[np.argsort(test_errors)]
n_show      = min(4, len(sorted_test))
show_pi     = [int(sorted_test[j * (len(sorted_test) - 1) // max(n_show - 1, 1)])
               for j in range(n_show)]
show_labels = ['Best', 'Low-mid', 'High-mid', 'Worst'][:n_show]

matplotlib.rcParams.update({'font.size': 8})
fig3, axes3 = plt.subplots(3, n_show, figsize=(5 * n_show, 12),
                           squeeze=False)
fig3.suptitle("PINN vs FDM — Re(u_scat) for test half-angles\n"
              "(sorted by relative L2 error)", fontsize=11, fontweight='bold')
outline_kw3 = dict(color='cyan', lw=1.0, zorder=5, ls='--')
sub_kw3     = dict(color='yellow', lw=0.9, zorder=6, ls='--')

for col, (pi, lab) in enumerate(zip(show_pi, show_labels)):
    alpha  = float(half_angle[pi])
    apex_y = float(apex_y_all[pi])
    err    = errors_all[pi]

    _, U_scat_pinn, _, _ = evaluate_fields(pi)
    U_scat_fdm = fdm_scat_on_grid(pi)
    ox, oy = tip_outline_xy(alpha, apex_y)

    vmax = max(float(np.percentile(np.abs(U_scat_fdm.real), 98)), 1e-10)
    kw   = dict(origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r',
                vmin=-vmax, vmax=vmax, interpolation='bilinear')
    title_str = f"{lab} — α={np.rad2deg(alpha):.2f}°\nrelL2={err:.3f}"

    im_p = axes3[0, col].imshow(U_scat_pinn.real, **kw)
    axes3[0, col].plot(ox, oy, **outline_kw3)
    axes3[0, col].axhline(Y_SUB, **sub_kw3)
    axes3[0, col].set_title(f"{title_str}\nPINN Re(u_scat)", fontsize=7)
    axes3[0, col].set_xlim(0, 1); axes3[0, col].set_ylim(0, 1)
    plt.colorbar(im_p, ax=axes3[0, col], shrink=0.8)

    im_f = axes3[1, col].imshow(U_scat_fdm.real, **kw)
    axes3[1, col].plot(ox, oy, **outline_kw3)
    axes3[1, col].axhline(Y_SUB, **sub_kw3)
    axes3[1, col].set_title("FDM Re(u_scat)", fontsize=7)
    axes3[1, col].set_xlim(0, 1); axes3[1, col].set_ylim(0, 1)
    plt.colorbar(im_f, ax=axes3[1, col], shrink=0.8)

    diff   = U_scat_pinn.real - U_scat_fdm.real
    vmax_d = max(float(np.percentile(np.abs(diff), 99)), 1e-10)
    im_d   = axes3[2, col].imshow(diff, origin='lower', extent=[0, 1, 0, 1],
                                  cmap='PiYG', vmin=-vmax_d, vmax=vmax_d,
                                  interpolation='bilinear')
    axes3[2, col].plot(ox, oy, **outline_kw3)
    axes3[2, col].axhline(Y_SUB, **sub_kw3)
    axes3[2, col].set_title("PINN − FDM", fontsize=7)
    axes3[2, col].set_xlim(0, 1); axes3[2, col].set_ylim(0, 1)
    plt.colorbar(im_d, ax=axes3[2, col], shrink=0.8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{RUN_TAG}_field_comparison.png",
            dpi=130, bbox_inches='tight')
print(f"Saved: {FIG_DIR}/{RUN_TAG}_field_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble MP4 from captured frames
# ─────────────────────────────────────────────────────────────────────────────
if gif_frames:
    mp4_path = f"{FIG_DIR}/{RUN_TAG}_training.mp4"
    fps = 6
    hold_frames = fps * 3
    frames_out  = gif_frames + [gif_frames[-1]] * hold_frames
    writer_kw   = dict(fps=fps, codec="libx264", pixelformat="yuv420p",
                       output_params=["-crf", "18"])
    imageio.mimwrite(mp4_path, frames_out, format="mp4", **writer_kw)
    print(f"Saved training animation: {mp4_path}")

print("\nDone.")