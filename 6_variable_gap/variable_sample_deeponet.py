"""
VARIABLE SAMPLE — parametric DeepONet / MLP over the tip-sample gap
===============================================================================
Learns u_scat(gap; x, y) for a cylinder above a dielectric substrate, with the
gap between the cylinder and the substrate as the operator input.

WHY THIS DIRECTORY EXISTS
-------------------------
final/ solves a cone-plus-apex AFM tip and, on a single geometry, sits at
relL2 ~ 0.30 after 20000 Adam + 50000 L-BFGS. sample/mie_pinn_sample.py solves
a CYLINDER above the same substrate and converges essentially onto the FDM in
about 10000 L-BFGS steps. A generalisation experiment is only informative if
the single-geometry problem underneath it is solved, so this directory rebuilds
the parametric machinery of final/ on top of the physics of sample/.

Everything physical here is taken from sample/mie_pinn_sample.py without
modification: the cylinder (a = 0.05, n = 3 + 30i), the substrate
(n = 1.5 + 0.1i below Y_SUB = 0.4), lam = 0.5, 45-degree incidence, the UPML
(d = 0.6, sigma_max = 6k, order 3), the loss weights, the collocation counts,
and the merged-exterior network arrangement. What is new is the gap parameter
and the branch/trunk head that consumes it.

WHAT IS SWEPT
-------------
    gap = cy - a - Y_SUB    over 10 values in [0.01, 0.20]

Two interior gaps are held out for testing, so the generalisation number is
interpolation rather than extrapolation. The split is drawn with a seeded RNG
in variable_sample_fdm.py and read from the npz here, so the two scripts cannot
disagree about it.

NETWORK ARRANGEMENT (from sample/mie_pinn_sample.py)
----------------------------------------------------
    net_int   total field INSIDE the cylinder
    net_scat  scattered field OUTSIDE it -- air AND substrate, one network

net_sub is an ALIAS of net_scat, not a third network. That is the change which
made the cylinder-above-substrate problem converge at all: with separate air
and substrate networks, any radiation crossing y = Y_SUB (which sits 0.01-0.20
below the cylinder) is penalised by the flat-interface losses unless both nets
move in lockstep, and sitting still at zero avoids that coordination barrier
entirely. One exterior network removes it, exactly as the FDM treats u_scat as
one unknown with piecewise k^2. It is legitimate for the s-polarisation solved
here because u_scat and du_scat/dn are both continuous across y = Y_SUB
(t = 1 + r), so the field is C^1 there; only the second derivative kinks, and
the PDE samplers keep +-0.005 margins around Y_SUB so no residual is ever
evaluated on the kink. The flat-interface loss terms are then identically zero
and are not computed.

ARCHITECTURE SWITCH
-------------------
    deeponet : u = sum_i branch_i(features(gap)) * trunk_i(x, y)
    mlp      : u = MLP([fourier(x, y), features(gap)]) -> (Re, Im)

variable_sample_mlp.py runs THIS file with ARCH_OVERRIDE = "mlp", so the two
architectures share one code path -- identical physics, losses, samplers,
budgets, optimiser, monitor and figures, with the network as the only
difference. Forking the file would guarantee drift.

A KNOWN LIMITATION, STATED UP FRONT
-----------------------------------
The cylinder MOVES with the parameter (cy = Y_SUB + a + gap), so across the
sweep the discs at gap 0.01 and gap 0.20 barely overlap. A DeepONet's trunk is
a FIXED spatial basis, and representing a field whose scatterer sits in a
different place for every parameter is the Kolmogorov n-width problem: the
number of basis functions needed grows quickly with how far the geometry moves.
This is inherent to the experiment being run, both architectures face it, and
it is not worked around here. If the sweep turns out to be limited by it rather
than by optimisation, the standard fix is to give the interior trunk
CYLINDER-LOCAL coordinates (x - cx, y - cy) -- under which the interior problem
becomes gap-independent apart from the incident phase -- before reaching for
more capacity.

PREREQUISITE
------------
    python variable_sample/variable_sample_fdm.py

OUTPUTS
-------
    variable_sample/figures/<RUN_TAG>_loss.png
                            <RUN_TAG>_relL2.png
                            <RUN_TAG>_error_vs_gap.png
                            <RUN_TAG>_sweep_grid.png
                            <RUN_TAG>_field_comparison.png
                            <RUN_TAG>_training.mp4
    variable_sample/models/<RUN_TAG>_weights.pt
"""

import os
import time
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS — every value from sample/mie_pinn_sample.py
# ─────────────────────────────────────────────────────────────────────────────
N_RE  = 3.0      # cylinder: near-PEC, strong skin effect
N_IM  = 30.0
SUB_N_RE = 1.5   # substrate: dielectric half-space
SUB_N_IM = 0.1
Y_SUB    = 0.4
lam      = 0.5
k        = 2.0 * np.pi / lam
a        = 0.05  # cylinder radius
cx       = 0.5   # cylinder centre x (fixed; only cy is swept)

theta_inc = -np.pi / 4
kx = k * np.cos(theta_inc)
ky = k * np.sin(theta_inc)

d_pml     = 0.6
sigma_max = 6.0 * k
pml_order = 3

X_MIN, X_MAX = -d_pml, 1.0 + d_pml
Y_MIN, Y_MAX = -d_pml, 1.0 + d_pml

# ── Derived wavenumbers ─────────────────────────────────────────────────────
k_int       = (N_RE + 1j * N_IM) * k
k_int_sq_re = float((N_RE ** 2 - N_IM ** 2) * k ** 2)
k_int_sq_im = float(2 * N_RE * N_IM * k ** 2)
k_ext_sq    = float(k ** 2)
k_sub       = (SUB_N_RE + 1j * SUB_N_IM) * k
k_sub_sq_re = float((SUB_N_RE ** 2 - SUB_N_IM ** 2) * k ** 2)
k_sub_sq_im = float(2 * SUB_N_RE * SUB_N_IM * k ** 2)

# ── s-polarisation Fresnel pair (see the FDM script for the derivation) ─────
ky_sub = np.sqrt(k_sub ** 2 - kx ** 2 + 0j)
if ky_sub.real > 0:
    ky_sub = -ky_sub
q_air = abs(ky)
q_sub = -ky_sub
r_s   = (q_air - q_sub) / (q_air + q_sub)
t_s   = (2.0 * q_air) / (q_air + q_sub)

r_re, r_im = float(r_s.real), float(r_s.imag)
t_re, t_im = float(t_s.real), float(t_s.imag)
ky_sub_re  = float(ky_sub.real)
ky_sub_im  = float(ky_sub.imag)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collocation counts, all from sample/mie_pinn_sample.py.
N_INT   = 1000   # interior PDE points (inside the cylinder)
N_AIR   = 1500   # air PDE points (outside the cylinder, above Y_SUB)
N_SUB   = 1000   # substrate PDE points (below Y_SUB)
N_IFACE = 800    # cylinder boundary points
N_PML   = 2000   # PML collocation points PER REGION (air + substrate)
N_PML_OUT = N_PML // 2   # outer-edge Dirichlet points

# ── SINGLE-GAP CONTROL ──────────────────────────────────────────────────────
# Train on ONE gap instead of the sweep. None = the full sweep.
SINGLE_GAP_INDEX = None

N_PARAM_BATCH = 8

# How many geometries share one backward pass. The batch is split into chunks
# of this size; each chunk is concatenated into a single graph, backwarded, and
# freed before the next is built, and the gradients add.
#
# Mathematically identical for any value: the gradient of a sum is the sum of
# the gradients, and each geometry keeps its 1/N_PARAM_BATCH weight. Verified
# across chunk sizes against the whole-batch path -- all ten terms and the full
# parameter gradient agree to float32 round-off.
#
# IT IS A MEMORY-AGAINST-LAUNCHES DIAL, and both ends hurt:
#
#   1                 peak memory of ONE geometry, but N_PARAM_BATCH separate
#                     backward sequences per term. At N_PARAM_BATCH = 16 that
#                     is 112 sequences per closure evaluation against 7 for the
#                     whole batch -- 16x the launch overhead, which is what
#                     made training slow after this was introduced to fix an
#                     out-of-memory error.
#   N_PARAM_BATCH     one graph for everything: fewest launches, and the peak
#                     that exhausted an 11.6 GiB card at 16 geometries.
#   in between        peak memory of `chunk` geometries, launches divided by
#                     `chunk`. 4 costs a quarter of the memory of the full
#                     batch and a quarter of the launches of full accumulation.
#
# None or 0 means the whole batch in one graph. Start at 4 and raise it until
# memory complains -- that is the fastest setting that still fits.
ACCUM_CHUNK = 8

BATCH_GEOMETRIES = True

# Evaluate net_scat's FOUR exterior terms -- pde_air, pde_sub, pml_air, pml_sub
# -- in a single forward and double-backward instead of four, with k^2 and the
# PML stretch supplied per point.
#
#   "none"  four separate passes
#   "pml"   the two PML bands share one pass; the two PDE samples stay separate
#   "all"   one pass over all four
MERGE_EXTERIOR = "none"
assert MERGE_EXTERIOR in ("none", "pml", "all")

ADAM_EPOCHS = 10000
LBFGS_STEPS = 5000   # CLOSURE EVALUATIONS, not outer steps
LBFGS_MAX_ITER = 20
LR_ADAM = 8e-4

# Loss weights — sample/mie_pinn_sample.py's values. Every term is divided by
# its value at initialisation (L0_*)
W_PDE_INT = 1.0
W_PDE_EXT = 1.0
W_PDE_SUB = 1.0
W_FIELD   = 20.0   # field continuity on the cylinder boundary
W_FLUX    = 20.0   # flux  continuity on the cylinder boundary
W_PML     = 2.0
W_PML_DIR = 1.0

# ── FDM DATA LOSS ───────────────────────────────────────────────────────────
# Supervision from the FDM solutions at the TRAINING gaps, applied alongside
# the physics for the whole run (not a decaying warm start).
W_DATA = 0.0

RESAMPLE_LBFGS = 100
RESAMPLE_ADAM  = 250   # sample/mie_pinn_sample.py's Adam cadence

EVAL_EVERY = 250       # relL2 monitor cadence, in Adam epochs / L-BFGS evals

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
N_FOURIER = 64   # Fourier features per network, as in sample/

# Branch input width. See _raw_features for what the entries are and why.
PHYS_DIM = 5

# Branch: features(gap) -> coefficients. Small, since the parameter space is
# one-dimensional.
BRANCH_HIDDEN  = 3
BRANCH_NEURONS = 64
# Trunk: (x, y) -> spatial basis.
TRUNK_HIDDEN  = 4
TRUNK_NEURONS = 128
# Dot-product width, split in half: the first BRANCH_DIM/2 coefficients are
# dotted for Re(u), the rest for Im(u).
BRANCH_DIM = 256

MLP_HIDDEN  = TRUNK_HIDDEN
MLP_NEURONS = TRUNK_NEURONS

# Where the L0 normalisation constants come from.
#
#   "sample"    the values measured from sample/mie_pinn_sample.py at this
#               geometry. Fixing them makes the OBJECTIVE FUNCTION identical to
#               that script's
#   "measured"  measure them from this network at initialisation (the previous
#               behaviour).
L0_SOURCE = "measured"
# ARCH="single" additionally REQUIRES "measured" -- see the guard beside the L0
# assembly for why.

# Rescale each network at initialisation so its output has this rms over the
# unit box. None = leave the initialisation alone (the old behaviour).
INIT_OUT_RMS = 7.70e-02

ARCH = globals().get("ARCH_OVERRIDE", "deeponet")
assert ARCH in ("deeponet", "mlp", "single"), f"unknown ARCH {ARCH!r}"

FIG_DIR   = globals().get("FIG_DIR_OVERRIDE",   "variable_sample/figures")
MODEL_DIR = globals().get("MODEL_DIR_OVERRIDE", "variable_sample/models")
RUN_TAG   = globals().get("RUN_TAG_OVERRIDE",   "variable_sample_deeponet")
if SINGLE_GAP_INDEX is not None:
    RUN_TAG = f"{RUN_TAG}_gap{SINGLE_GAP_INDEX}"
CKPT_PATH = f"{MODEL_DIR}/{RUN_TAG}_weights.pt"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# torch.compile
USE_COMPILE = True

# ─────────────────────────────────────────────────────────────────────────────
# FDM reference
# ─────────────────────────────────────────────────────────────────────────────
FDM_PATH = "variable_sample/inputs/fdm_variable_sample_sweep.npz"
if not os.path.exists(FDM_PATH):
    raise SystemExit(f"{FDM_PATH} not found -- run "
                     f"variable_sample/variable_sample_fdm.py first")

_fdm = np.load(FDM_PATH)
FDM_X   = _fdm["X"]
FDM_Y   = _fdm["Y"]
GAP_VALS = _fdm["gap"].astype(np.float64)
CY_VALS  = _fdm["cy"].astype(np.float64)
N_PARAM  = len(GAP_VALS)

FDM_Scat  = _fdm["U_scat_re"]  + 1j * _fdm["U_scat_im"]
FDM_Total = _fdm["U_total_re"] + 1j * _fdm["U_total_im"]
FDM_in_cyl = _fdm["in_cyl"].astype(bool)
FDM_in_sub = _fdm["in_sub"].astype(bool)

solved_idx = set(int(p) for p in np.asarray(_fdm["solved_idx"]).ravel())
train_idx  = np.array([p for p in _fdm["train_idx"] if int(p) in solved_idx])
test_idx   = np.array([p for p in _fdm["test_idx"]  if int(p) in solved_idx])

if SINGLE_GAP_INDEX is not None:
    _p = int(SINGLE_GAP_INDEX)
    if _p not in solved_idx:
        raise SystemExit(f"gap index {_p} was not solved by the FDM sweep "
                         f"(solved: {sorted(solved_idx)})")
    # test_idx is the SAME index: with one gap there is no held-out set, and
    # the monitor's "test" column is then a duplicate of "train" rather than a
    # generalisation number. The banner says so.
    train_idx = np.array([_p])
    test_idx  = np.array([_p])

if len(train_idx) == 0:
    raise SystemExit("no solved training gaps in the npz -- was the FDM sweep "
                     "run with SOLVE_SUBSET set?")
for _p in list(train_idx) + list(test_idx):
    if np.linalg.norm(FDM_Scat[int(_p)]) == 0.0:
        raise SystemExit(f"FDM reference for index {_p} is identically zero")

N_PARAM_BATCH = min(N_PARAM_BATCH, len(train_idx))

# Grid geometry, for interp_fdm and the monitor.
_x0, _y0 = float(FDM_X[0, 0]), float(FDM_Y[0, 0])
_dx = float(FDM_X[0, 1] - FDM_X[0, 0])
_dy = float(FDM_Y[1, 0] - FDM_Y[0, 0])

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers — the gap is the only thing that moves
# ─────────────────────────────────────────────────────────────────────────────
def cy_of(pi):
    return float(CY_VALS[int(pi)])


def gap_of(pi):
    return float(GAP_VALS[int(pi)])


def cyl_mask(Xg, Yg, cy):
    return (Xg - cx) ** 2 + (Yg - cy) ** 2 <= a ** 2


# ─────────────────────────────────────────────────────────────────────────────
# Branch features
# ─────────────────────────────────────────────────────────────────────────────
def _raw_features(gap):
    """Branch input: smooth, physically-motivated functions of the gap.

      gap                 the parameter itself
      log(gap)            the near field varies far more steeply at small gap
                          than at large (0.01 to 0.20 is a factor of 20), so
                          the log is the well-conditioned coordinate near
                          contact
      cos(2|ky| gap)      the background above the substrate is a standing wave
                          in y with period pi/|ky|; this is the phase of that
      sin(2|ky| gap)      interference at the cylinder.
      a/(a + gap)         a bounded proxy for the cylinder-image coupling
                          strength, which grows as the cylinder approaches the
                          surface. Bounded rather than a/gap so the feature
                          stays O(1) at the smallest gap
    """
    g = float(gap)
    return np.array([
        g,
        np.log(max(g, 1e-6)),
        np.cos(2.0 * abs(ky) * g),
        np.sin(2.0 * abs(ky) * g),
        a / (a + g),
    ], dtype=np.float32)


assert len(_raw_features(GAP_VALS[0])) == PHYS_DIM, \
    "PHYS_DIM does not match _raw_features"

_all_feats = np.stack([_raw_features(g) for g in GAP_VALS])
FEAT_MIN = _all_feats.min(axis=0)
FEAT_MAX = _all_feats.max(axis=0)


def param_to_features(gap):
    """Normalised [1, PHYS_DIM] feature row on DEVICE, values in [-1, 1].
    Defined for any gap in the sweep's range, so the trained model can be
    queried at gaps it never saw."""
    raw   = _raw_features(gap)
    normd = 2.0 * (raw - FEAT_MIN) / (FEAT_MAX - FEAT_MIN + 1e-10) - 1.0
    return torch.tensor(normd, device=DEVICE).unsqueeze(0)


FEATS = {int(p): param_to_features(gap_of(p)) for p in range(N_PARAM)}

# ─────────────────────────────────────────────────────────────────────────────
# Fourier feature embeddings
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(1)
B_ext = torch.randn(2, N_FOURIER, device=DEVICE) * abs(k_sub)
B_int = torch.randn(2, N_FOURIER, device=DEVICE) * abs(k_int)

FOURIER_DIM   = 2 * N_FOURIER
TRUNK_IN_DIM  = FOURIER_DIM
BRANCH_IN_DIM = PHYS_DIM


def embed_ext(xy):
    proj = xy @ B_ext
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def embed_int(xy):
    proj = xy @ B_int
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def embed_both(xy):
    """Both bands, for the single shared network.

    u_int decays over a skin depth of lam/188 while u_scat oscillates at
    |k_sub| = 1.5k -- a factor of 20 apart -- so a shared spatial body cannot
    serve both from one band. Concatenating them lets each head select what it
    needs. B_ext and B_int are read at CALL time rather than concatenated once
    at import, so restoring them from a checkpoint (which any consumer must do,
    since they are drawn on DEVICE and the CUDA and CPU RNG streams differ)
    also fixes this embedding."""
    proj = torch.cat([xy @ B_ext, xy @ B_int], dim=-1)
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────
def _mlp(d_in, d_hidden, n_hidden, d_out):
    layers = [nn.Linear(d_in, d_hidden), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(d_hidden, d_hidden), nn.Tanh()]
    layers += [nn.Linear(d_hidden, d_out)]
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            nn.init.zeros_(m.bias)
    return net


class BranchNet(nn.Module):
    """features(gap) -> BRANCH_DIM coefficients."""

    def __init__(self):
        super().__init__()
        self.net = _mlp(BRANCH_IN_DIM, BRANCH_NEURONS, BRANCH_HIDDEN, BRANCH_DIM)

    def forward(self, phys_feats):
        return self.net(phys_feats)


class TrunkNet(nn.Module):
    """(x, y) -> BRANCH_DIM spatial basis functions."""

    def __init__(self, embed_fn):
        super().__init__()
        self.embed = embed_fn
        self.net = _mlp(TRUNK_IN_DIM, TRUNK_NEURONS, TRUNK_HIDDEN, BRANCH_DIM)

    def forward(self, xy):
        return self.net(self.embed(xy))


class DeepONet(nn.Module):
    """u(gap, x, y) = sum_i branch_i(features) * trunk_i(x, y).

    BRANCH_DIM is split in half: the first half is dotted for Re(u), the second
    for Im(u). phys_feats is [1, PHYS_DIM] and broadcasts over all points of the
    block, which is the required pairing since one block is one geometry."""

    def __init__(self, embed_fn):
        super().__init__()
        self.branch  = BranchNet()
        self.trunk   = TrunkNet(embed_fn)
        self.bias_re = nn.Parameter(torch.zeros(1))
        self.bias_im = nn.Parameter(torch.zeros(1))
        self.half    = BRANCH_DIM // 2

    def forward(self, phys_feats, xy):
        bt = self.branch(phys_feats) * self.trunk(xy)
        Re = bt[:, :self.half].sum(dim=-1, keepdim=True) + self.bias_re
        Im = bt[:, self.half:].sum(dim=-1, keepdim=True) + self.bias_im
        return Re, Im


class MLPNet(nn.Module):
    """u(gap, x, y) = MLP([fourier(x, y), features(gap)]) -> (Re, Im).

    The control for the DeepONet: same Fourier embedding, same depth and width
    as the trunk, same everything downstream -- but the parameter enters as an
    ordinary input instead of multiplying a fixed spatial basis, so there is no
    rank-BRANCH_DIM/2 factorisation and no branch/trunk scaling degeneracy."""

    def __init__(self, embed_fn):
        super().__init__()
        self.embed = embed_fn
        self.net = _mlp(FOURIER_DIM + PHYS_DIM, MLP_NEURONS, MLP_HIDDEN, 2)

    def forward(self, phys_feats, xy):
        e = self.embed(xy)
        p = phys_feats if phys_feats.shape[0] == e.shape[0] \
            else phys_feats.expand(e.shape[0], -1)
        out = self.net(torch.cat([e, p], dim=-1))
        return out[:, 0:1], out[:, 1:2]


class SharedTwoHead(nn.Module):
    """ONE network over the whole domain, read out through two heads.

        body(embed_both(x, y))  ->  TRUNK_NEURONS      shared representation
        head_int                ->  BRANCH_DIM basis for u_int
        head_scat               ->  BRANCH_DIM basis for u_scat
        branch(features(gap))   ->  2 * BRANCH_DIM coefficients, split

    WHY. With two independent networks the only thing coupling them is the pair
    of interface conditions on a curve that MOVES with the parameter, so every
    gradient that carries geometry has to cross between two separate parameter
    sets through 800 points. Sharing the body means the representation that
    knows where the cylinder is, is the same one both fields are read from, and
    the interface gradients update shared weights.

    WHY STILL TWO HEADS. The two fields are different objects on different
    scales -- u_int is the TOTAL field inside a strong absorber, u_scat the
    scattered field outside -- and forcing them through one readout would ask
    one basis to be good at both. The heads are linear layers on a shared body,
    so they cost little and keep the two readouts independent.

    The physics, losses, samplers, data terms and budgets are untouched: only
    what computes u_int and u_scat changes."""

    def __init__(self):
        super().__init__()
        emb = 2 * (2 * N_FOURIER)           # cos and sin of both bands
        body = [nn.Linear(emb, TRUNK_NEURONS), nn.Tanh()]
        for _ in range(TRUNK_HIDDEN - 1):
            body += [nn.Linear(TRUNK_NEURONS, TRUNK_NEURONS), nn.Tanh()]
        self.body = nn.Sequential(*body)
        self.head_int = nn.Linear(TRUNK_NEURONS, BRANCH_DIM)
        self.head_scat = nn.Linear(TRUNK_NEURONS, BRANCH_DIM)
        self.branch = _mlp(BRANCH_IN_DIM, BRANCH_NEURONS, BRANCH_HIDDEN,
                           2 * BRANCH_DIM)
        self.bias = nn.Parameter(torch.zeros(4))   # re/im for each head
        self.half = BRANCH_DIM // 2
        for m in self.modules():                   # same init as the others
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def _read(self, b, t, b0):
        bt = b * t
        return (bt[:, :self.half].sum(dim=-1, keepdim=True) + self.bias[b0],
                bt[:, self.half:].sum(dim=-1, keepdim=True) + self.bias[b0 + 1])

    def forward_int(self, feats, xy):
        h = self.body(embed_both(xy))
        return self._read(self.branch(feats)[:, :BRANCH_DIM],
                          self.head_int(h), 0)

    def forward_scat(self, feats, xy):
        h = self.body(embed_both(xy))
        return self._read(self.branch(feats)[:, BRANCH_DIM:],
                          self.head_scat(h), 2)


class _Head(nn.Module):
    """A view onto one head of the shared core, presenting the same
    (phys_feats, xy) signature as DeepONet and MLPNet so every call site --
    the losses, the PML branch, the monitor, the figures -- is unchanged.

    Both views hold the SAME core, so their .parameters() overlap completely;
    all_params below deduplicates by identity, which torch.optim requires."""

    def __init__(self, core, which):
        super().__init__()
        self.core = core
        self.which = which

    def forward(self, phys_feats, xy):
        return (self.core.forward_int if self.which == "int"
                else self.core.forward_scat)(phys_feats, xy)


def _make_net(embed_fn):
    return (DeepONet(embed_fn) if ARCH == "deeponet" else MLPNet(embed_fn)).to(DEVICE)


if ARCH == "single":
    _core = SharedTwoHead().to(DEVICE)
    net_int = _Head(_core, "int").to(DEVICE)
    net_scat = _Head(_core, "scat").to(DEVICE)
else:
    net_int  = _make_net(embed_int)   # total field inside the cylinder
    net_scat = _make_net(embed_ext)   # scattered field outside (air + substrate)

def _final_linear(net):
    """The layer that scales the output. For the DeepONet the output is
    sum_i b_i t_i (+ zero biases at init), so scaling the TRUNK's last layer
    scales the output exactly; for the MLP it is the last layer outright."""
    if isinstance(net, _Head):
        # u = sum_i b_i t_i with zero biases at init, and t is this head's
        # output, so scaling the head scales that field alone -- the two heads
        # rescale independently even though the body is shared.
        return (net.core.head_int if net.which == "int" else net.core.head_scat)
    return net.trunk.net[-1] if isinstance(net, DeepONet) else net.net[-1]


def _rescale_init(net, target):
    """Scale the network's initial output to `target` rms over the unit box.
    Exact in one pass: the output is linear in the final layer's weight and
    bias, so one multiplication lands on the target rather than approaching
    it."""
    xy = torch.rand(4000, 2, device=DEVICE)

    def rms():
        # Averaged over EVERY parameter row: a DeepONet's output magnitude
        # depends on its branch input, so measuring at one gap would leave the
        # others mis-scaled. For the MLP the spread is negligible and the
        # average costs nothing.
        with torch.no_grad():
            acc = [ (lambda r, i: (r ** 2 + i ** 2).mean())(*net(f, xy))
                    for f in FEATS.values() ]
        return float(torch.sqrt(torch.stack(acc).mean()))

    cur = rms()
    if cur < 1e-30:
        return cur, cur
    with torch.no_grad():
        lin = _final_linear(net)
        lin.weight.mul_(target / cur)
        lin.bias.mul_(target / cur)
    return cur, rms()


_init_rms = None
if INIT_OUT_RMS is not None:
    _b_int = _rescale_init(net_int, INIT_OUT_RMS)
    _b_scat = _rescale_init(net_scat, INIT_OUT_RMS)
    _init_rms = (_b_int, _b_scat)

_raw_int, _raw_scat = net_int, net_scat   # pre-compile handles, for state_dict


def raw_module(m):
    return getattr(m, "_orig_mod", m)


if USE_COMPILE:
    try:
        import torch._functorch.config as _ft_config
        if hasattr(_ft_config, "donated_buffer"):
            _ft_config.donated_buffer = False
            print("  torch._functorch.config.donated_buffer = False "
                  "(required for create_graph=True)")
    except Exception as _cfg_exc:
        print(f"  WARNING: could not disable donated_buffer: {_cfg_exc}")
    net_int  = torch.compile(net_int,  mode="default", dynamic=True)
    net_scat = torch.compile(net_scat, mode="default", dynamic=True)
    try:
        _w_xy = torch.rand(64, 2, device=DEVICE, requires_grad=True)
        _w_pf = FEATS[0]
        for _w_net in (net_int, net_scat):
            _w_r, _w_i = _w_net(_w_pf, _w_xy)
            _w_g  = torch.autograd.grad(_w_r.sum(), _w_xy, create_graph=True)[0]
            _w_gg = torch.autograd.grad(_w_g[:, 0].sum(), _w_xy,
                                        create_graph=True)[0]
            _w_gg.sum().backward()
        print("  torch.compile warm-up OK (double-backward compiled)")
    except Exception as _w_exc:
        net_int, net_scat = _raw_int, _raw_scat
        USE_COMPILE = False
        print(f"  torch.compile FAILED -> falling back to eager: "
              f"{type(_w_exc).__name__}: {str(_w_exc).splitlines()[-1][:160]}")
    finally:
        for _w_p in list(net_int.parameters()) + list(net_scat.parameters()):
            _w_p.grad = None

# MERGED EXTERIOR
net_sub = net_scat

# Deduplicated by identity: with ARCH="single" the two views share one core,
# so the concatenation lists every parameter twice and torch.optim rejects
# duplicates. A no-op for the other two architectures, whose nets are distinct.
_seen = set()
all_params = [p for p in list(net_int.parameters()) + list(net_scat.parameters())
              if not (id(p) in _seen or _seen.add(id(p)))]
n_int_p  = sum(p.numel() for p in net_int.parameters())
n_scat_p = sum(p.numel() for p in net_scat.parameters())
n_total_p = sum(p.numel() for p in all_params)

# ─────────────────────────────────────────────────────────────────────────────
# Startup banner
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 78)
print(f"VARIABLE SAMPLE — {ARCH.upper()} over the tip-sample gap")
print("=" * 78)
print(f"  device        : {DEVICE}   torch.compile: {USE_COMPILE}")
print(f"  cylinder      : a = {a}, n = {N_RE} + {N_IM}i, cx = {cx}")
print(f"  substrate     : n = {SUB_N_RE} + {SUB_N_IM}i below Y_SUB = {Y_SUB}")
print(f"  lam           : {lam}   k = {k:.4f}   theta_inc = "
      f"{np.degrees(theta_inc):.0f} deg")
print(f"  gap sweep     : {N_PARAM} values in "
      f"[{GAP_VALS[0]:.3f}, {GAP_VALS[-1]:.3f}]  (cy in "
      f"[{CY_VALS[0]:.3f}, {CY_VALS[-1]:.3f}])")
if SINGLE_GAP_INDEX is None:
    print(f"  train / test  : {len(train_idx)} / {len(test_idx)}   "
          f"test gaps = {[f'{gap_of(p):.3f}' for p in test_idx]}")
else:
    print(f"  SINGLE GAP    : index {SINGLE_GAP_INDEX}, "
          f"gap = {gap_of(SINGLE_GAP_INDEX):.4f}, "
          f"cy = {cy_of(SINGLE_GAP_INDEX):.4f}")
    print(f"                  control run -- the branch input is constant, so "
          f"the 'test' column below")
    print(f"                  repeats 'train' and is NOT a generalisation "
          f"number. Target ~0.1, from")
    print(f"                  sample/mie_pinn_sample.py at gap 0.05.")
print(f"  param batch   : {N_PARAM_BATCH} geometries per step"
      f"   ({'concatenated' if BATCH_GEOMETRIES else 'looped'}"
      f"{', merge=' + MERGE_EXTERIOR if BATCH_GEOMETRIES else ''})")
print(f"  collocation   : int {N_INT}, air {N_AIR}, sub {N_SUB}, "
      f"iface {N_IFACE}, pml {N_PML}x2, outer {N_PML_OUT}  (per geometry)")
print(f"  budget        : {ADAM_EPOCHS} Adam + {LBFGS_STEPS} L-BFGS evals")
_chunk_eff = (N_PARAM_BATCH if ACCUM_CHUNK in (None, 0)
              else min(max(1, int(ACCUM_CHUNK)), N_PARAM_BATCH))
print(f"  accum chunk   : {_chunk_eff} of {N_PARAM_BATCH} geometries per "
      f"backward  ({-(-N_PARAM_BATCH // _chunk_eff)} sequences per term)")
print(f"  data loss     : W_DATA = {W_DATA}" + ("  (off)" if W_DATA == 0 else
      f"  on the {len(train_idx)} TRAINING gaps only"))
if ARCH == "deeponet":
    print(f"  branch/trunk  : {BRANCH_HIDDEN}x{BRANCH_NEURONS} / "
          f"{TRUNK_HIDDEN}x{TRUNK_NEURONS}, BRANCH_DIM {BRANCH_DIM}")
elif ARCH == "single":
    print(f"  single core   : body {TRUNK_HIDDEN}x{TRUNK_NEURONS} shared over "
          f"the WHOLE domain, two linear heads -> BRANCH_DIM {BRANCH_DIM}")
    print(f"                  branch {BRANCH_HIDDEN}x{BRANCH_NEURONS} -> "
          f"{2 * BRANCH_DIM} coefficients, split between the heads")
    print(f"                  embedding carries BOTH Fourier bands "
          f"(|k_sub| = {abs(k_sub):.1f} and |k_int| = {abs(k_int):.1f})")
    print(f"                  the high band leaks into the exterior residual "
          f"at init (~{(abs(k_int) / abs(k_sub)) ** 4:.0e}x), so L0_SOURCE "
          f"must be 'measured' -- it is {L0_SOURCE!r}")
else:
    print(f"  mlp           : {MLP_HIDDEN}x{MLP_NEURONS}")
    # Width giving parameter parity with one DeepONet, solved rather than
    # quoted, so it cannot go stale when the Fourier or feature widths change.
    _din = FOURIER_DIM + PHYS_DIM
    _tgt = (BRANCH_IN_DIM * BRANCH_NEURONS + BRANCH_NEURONS
            + (BRANCH_HIDDEN - 1) * (BRANCH_NEURONS ** 2 + BRANCH_NEURONS)
            + BRANCH_NEURONS * BRANCH_DIM + BRANCH_DIM
            + TRUNK_IN_DIM * TRUNK_NEURONS + TRUNK_NEURONS
            + (TRUNK_HIDDEN - 1) * (TRUNK_NEURONS ** 2 + TRUNK_NEURONS)
            + TRUNK_NEURONS * BRANCH_DIM + BRANCH_DIM + 2)
    # (MLP_HIDDEN-1) W^2 + (d_in + MLP_HIDDEN + 2) W + 2 = _tgt
    _A = MLP_HIDDEN - 1
    _B = _din + MLP_HIDDEN + 2
    _C = 2 - _tgt
    _W = (-_B + np.sqrt(_B ** 2 - 4 * _A * _C)) / (2 * _A) if _A else -_C / _B
    print(f"                  parameter parity with one DeepONet would need "
          f"MLP_NEURONS = {int(round(_W))}")
if ARCH == "single":
    print(f"  parameters    : {n_total_p:,} shared "
          f"(the two views report {n_int_p:,} each -- the same core)")
else:
    print(f"  parameters    : net_int {n_int_p:,}   net_scat {n_scat_p:,}"
          f"   total {n_total_p:,}")
if _init_rms is not None:
    print(f"  init out rms  : net_int {_init_rms[0][0]:.3e} -> "
          f"{_init_rms[0][1]:.3e}   net_scat {_init_rms[1][0]:.3e} -> "
          f"{_init_rms[1][1]:.3e}   (target {INIT_OUT_RMS:.3e}, "
          f"sample/mie_pinn_sample.py's)")
else:
    print(f"  init out rms  : left as initialised (INIT_OUT_RMS = None)")
print("=" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# Background field (Fresnel, no cylinder) — gap-independent
# ─────────────────────────────────────────────────────────────────────────────
def u_bg_air_re(xy):
    p_i = kx * xy[:, 0:1] + ky * (xy[:, 1:2] - Y_SUB)
    p_r = kx * xy[:, 0:1] - ky * (xy[:, 1:2] - Y_SUB)
    return torch.cos(p_i) + (r_re * torch.cos(p_r) - r_im * torch.sin(p_r))


def u_bg_air_im(xy):
    p_i = kx * xy[:, 0:1] + ky * (xy[:, 1:2] - Y_SUB)
    p_r = kx * xy[:, 0:1] - ky * (xy[:, 1:2] - Y_SUB)
    return torch.sin(p_i) + (r_re * torch.sin(p_r) + r_im * torch.cos(p_r))


def u_bg_sub_re(xy):
    s   = xy[:, 1:2] - Y_SUB
    amp = torch.exp(-ky_sub_im * s)
    ph  = kx * xy[:, 0:1] + ky_sub_re * s
    return amp * (t_re * torch.cos(ph) - t_im * torch.sin(ph))


def u_bg_sub_im(xy):
    s   = xy[:, 1:2] - Y_SUB
    amp = torch.exp(-ky_sub_im * s)
    ph  = kx * xy[:, 0:1] + ky_sub_re * s
    return amp * (t_re * torch.sin(ph) + t_im * torch.cos(ph))


def du_bg_air_dn_re(xy, nxy):
    nx, ny = nxy[:, 0:1], nxy[:, 1:2]
    p_i = kx * xy[:, 0:1] + ky * (xy[:, 1:2] - Y_SUB)
    p_r = kx * xy[:, 0:1] - ky * (xy[:, 1:2] - Y_SUB)
    kn_i = kx * nx + ky * ny
    kn_r = kx * nx - ky * ny
    return (-kn_i * torch.sin(p_i)
            - kn_r * (r_re * torch.sin(p_r) + r_im * torch.cos(p_r)))


def du_bg_air_dn_im(xy, nxy):
    nx, ny = nxy[:, 0:1], nxy[:, 1:2]
    p_i = kx * xy[:, 0:1] + ky * (xy[:, 1:2] - Y_SUB)
    p_r = kx * xy[:, 0:1] - ky * (xy[:, 1:2] - Y_SUB)
    kn_i = kx * nx + ky * ny
    kn_r = kx * nx - ky * ny
    return (kn_i * torch.cos(p_i)
            + kn_r * (r_re * torch.cos(p_r) - r_im * torch.sin(p_r)))


# ─────────────────────────────────────────────────────────────────────────────
# Autograd helpers
# ─────────────────────────────────────────────────────────────────────────────
def laplacian(u, xy):
    g   = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(g[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    return uxx + uyy


def normal_deriv(u, xy, nxy):
    g = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    return g[:, 0:1] * nxy[:, 0:1] + g[:, 1:2] * nxy[:, 1:2]


# ─────────────────────────────────────────────────────────────────────────────
# UPML — divergence-form stretched Helmholtz, identical to
# sample/mie_pinn_sample.py (and to the conservative FDM stencil).
#
#   d/dx( (s_y/s_x) du/dx ) + d/dy( (s_x/s_y) du/dy ) + k^2 s_x s_y u = 0
#
# Expanding the divergences gives the two second-derivative terms, the two
# FIRST-derivative terms carrying the gradient of the stretch, and the reaction
# term. Dropping the first-derivative group is what makes a PML reflect.
# ─────────────────────────────────────────────────────────────────────────────
def _pml_sigma(coord, lo, hi):
    s = np.zeros_like(coord, dtype=float)
    m = coord < lo
    s[m] = sigma_max * ((lo - coord[m]) / d_pml) ** pml_order
    m = coord > hi
    s[m] = sigma_max * ((coord[m] - hi) / d_pml) ** pml_order
    return s


def _pml_sigma_deriv(coord, lo, hi):
    s = np.zeros_like(coord, dtype=float)
    if pml_order >= 1:
        m = coord < lo
        s[m] = (pml_order * sigma_max / d_pml) * \
               ((lo - coord[m]) / d_pml) ** (pml_order - 1)
        m = coord > hi
        s[m] = (pml_order * sigma_max / d_pml) * \
               ((coord[m] - hi) / d_pml) ** (pml_order - 1)
    return s


def _pml_stretch_tensors(xy, k_ref):
    """Precompute the stretch tensors for an (n, 2) numpy array of PML points.
    Points are graded against the PHYSICAL box [0,1]^2 -- the stretch is purely
    geometric; the medium enters only through k_ref and, in the residual,
    through k^2."""
    sx = 1.0 + 1j * _pml_sigma(xy[:, 0], 0.0, 1.0) / k_ref
    sy = 1.0 + 1j * _pml_sigma(xy[:, 1], 0.0, 1.0) / k_ref
    T = lambda v: torch.tensor(v.astype(np.float32), device=DEVICE).unsqueeze(1)
    return (torch.tensor(xy.astype(np.float32), device=DEVICE,
                         requires_grad=True),
            T(sx.real), T(sx.imag), T(sy.real), T(sy.imag),
            T(_pml_sigma_deriv(xy[:, 0], 0.0, 1.0) / k_ref),
            T(_pml_sigma_deriv(xy[:, 1], 0.0, 1.0) / k_ref))


def pml_residual_sq(net, phys_feats, xy_pml, k2_re, k2_im):
    """Squared residual of the divergence-form UPML Helmholtz operator, PER
    POINT. k2_re / k2_im may be python floats (one medium for the whole block)
    or [M, 1] tensors (a block spanning both media); every use of them below is
    elementwise, so the two cases need no separate code."""
    xy_p, sx_re, sx_im, sy_re, sy_im, dsx_im, dsy_im = xy_pml

    pR, pI = net(phys_feats, xy_p)

    gR   = torch.autograd.grad(pR.sum(), xy_p, create_graph=True)[0]
    pRx, pRy = gR[:, 0:1], gR[:, 1:2]
    pRxx = torch.autograd.grad(gR[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
    pRyy = torch.autograd.grad(gR[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

    gI   = torch.autograd.grad(pI.sum(), xy_p, create_graph=True)[0]
    pIx, pIy = gI[:, 0:1], gI[:, 1:2]
    pIxx = torch.autograd.grad(gI[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
    pIyy = torch.autograd.grad(gI[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

    # Cx = s_y/s_x, Cy = s_x/s_y  (coefficients of u_xx, u_yy)
    sx_sq = sx_re ** 2 + sx_im ** 2
    sy_sq = sy_re ** 2 + sy_im ** 2
    Cx_re = (sy_re * sx_re + sy_im * sx_im) / sx_sq
    Cx_im = (sy_im * sx_re - sy_re * sx_im) / sx_sq
    Cy_re = (sx_re * sy_re + sx_im * sy_im) / sy_sq
    Cy_im = (sx_im * sy_re - sx_re * sy_im) / sy_sq

    # Cx' = -s_y s_x'/s_x^2, Cy' = -s_x s_y'/s_y^2, with s' purely imaginary
    sx2_re = sx_re ** 2 - sx_im ** 2
    sx2_im = 2.0 * sx_re * sx_im
    sx2_sq = sx2_re ** 2 + sx2_im ** 2
    sy2_re = sy_re ** 2 - sy_im ** 2
    sy2_im = 2.0 * sy_re * sy_im
    sy2_sq = sy2_re ** 2 + sy2_im ** 2

    n_xr =  dsx_im * sy_im
    n_xi = -dsx_im * sy_re
    Cxp_re = (n_xr * sx2_re + n_xi * sx2_im) / sx2_sq
    Cxp_im = (n_xi * sx2_re - n_xr * sx2_im) / sx2_sq

    n_yr =  dsy_im * sx_im
    n_yi = -dsy_im * sx_re
    Cyp_re = (n_yr * sy2_re + n_yi * sy2_im) / sy2_sq
    Cyp_im = (n_yi * sy2_re - n_yr * sy2_im) / sy2_sq

    sxsy_re = sx_re * sy_re - sx_im * sy_im
    sxsy_im = sx_re * sy_im + sx_im * sy_re
    K_re = k2_re * sxsy_re - k2_im * sxsy_im
    K_im = k2_re * sxsy_im + k2_im * sxsy_re

    res_R = (Cx_re * pRxx - Cx_im * pIxx + Cxp_re * pRx - Cxp_im * pIx
             + Cy_re * pRyy - Cy_im * pIyy + Cyp_re * pRy - Cyp_im * pIy
             + K_re * pR - K_im * pI)
    res_I = (Cx_re * pIxx + Cx_im * pRxx + Cxp_re * pIx + Cxp_im * pRx
             + Cy_re * pIyy + Cy_im * pRyy + Cyp_re * pIy + Cyp_im * pRy
             + K_re * pI + K_im * pR)
    return res_R ** 2 + res_I ** 2


def pml_pde_residual(net, phys_feats, xy_pml, k2_re, k2_im):
    """Mean-squared residual over the block."""
    return pml_residual_sq(net, phys_feats, xy_pml, k2_re, k2_im).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Collocation samplers. Only the four cylinder-relative ones depend on the gap;
# the substrate, PML and outer-edge samples are shared across the batch.
# ─────────────────────────────────────────────────────────────────────────────
def _to_t(arr):
    return torch.tensor(arr.astype(np.float32), device=DEVICE,
                        requires_grad=True)


def sample_inside(n, cy):
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 6, 2)
        r2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
        pts.append(xy[r2 < (a - 0.005) ** 2])
    return _to_t(np.vstack(pts)[:n])


def sample_air(n, cy):
    """Outside the cylinder AND above the substrate."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 4, 2)
        r2 = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
        m = (r2 > (a + 0.008) ** 2) & (xy[:, 1] > Y_SUB + 0.005)
        pts.append(xy[m])
    return _to_t(np.vstack(pts)[:n])


def sample_substrate(n):
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 3, 2)
        pts.append(xy[xy[:, 1] < Y_SUB - 0.005])
    return _to_t(np.vstack(pts)[:n])


def sample_interface(n, cy):
    """Cylinder boundary with outward normals."""
    th = np.random.rand(n) * 2 * np.pi
    xy = np.stack([cx + a * np.cos(th), cy + a * np.sin(th)], 1)
    nxy = np.stack([np.cos(th), np.sin(th)], 1)
    return _to_t(xy), torch.tensor(nxy.astype(np.float32), device=DEVICE)


def sample_pml_air(n):
    """PML points in air: outside [0,1]^2 and above Y_SUB.

    Gap-independent: the cylinder lives at x = 0.5 with cy in [0.46, 0.65], so
    it is inside the physical box for every gap in the sweep and never
    intersects the band.
    """
    pts = []
    while sum(len(p) for p in pts) < n:
        x = np.random.uniform(X_MIN, X_MAX, n * 4)
        y = np.random.uniform(Y_MIN, Y_MAX, n * 4)
        m = ((x < 0) | (x > 1) | (y > 1)) & (y > Y_SUB)
        pts.append(np.stack([x[m], y[m]], axis=1))
    return _pml_stretch_tensors(np.vstack(pts)[:n], float(k))


def sample_pml_sub(n):
    """PML points in the substrate: outside [0,1]^2 and below Y_SUB."""
    pts = []
    while sum(len(p) for p in pts) < n:
        x = np.random.uniform(X_MIN, X_MAX, n * 4)
        y = np.random.uniform(Y_MIN, Y_SUB, n * 4)
        m = (x < 0) | (x > 1) | (y < 0)
        pts.append(np.stack([x[m], y[m]], axis=1))
    return _pml_stretch_tensors(np.vstack(pts)[:n], float(abs(k_sub)))


def sample_pml_outer(n):
    """Points on the outermost PML edge, where u_scat = 0 is imposed."""
    edge = np.random.randint(0, 4, n)
    t = np.random.rand(n)
    x = np.where(edge == 2, X_MIN,
        np.where(edge == 3, X_MAX, X_MIN + t * (X_MAX - X_MIN)))
    y = np.where(edge == 0, Y_MAX,
        np.where(edge == 1, Y_MIN, Y_MIN + t * (Y_MAX - Y_MIN)))
    return _to_t(np.stack([x, y], axis=1))


def fdm_interp(grid, xy_t):
    """Bilinear sample of one [N, N] complex FDM grid at the given points.

    Evaluated ONCE per redraw and cached with the points, not per closure
    evaluation: L-BFGS calls the closure ~2000 times per point set, and this
    costs a device->host sync plus a scipy call each time it runs."""
    xy = xy_t.detach().cpu().numpy()
    cols = (xy[:, 0] - _x0) / _dx
    rows = (xy[:, 1] - _y0) / _dy
    out = []
    for part in (grid.real, grid.imag):
        v = map_coordinates(part, [rows, cols], order=1, mode="nearest")
        out.append(torch.tensor(v.astype(np.float32),
                                device=DEVICE).unsqueeze(1))
    return out[0], out[1]


# ── Per-geometry point cache, cleared on every redraw ────────────────────────
_pts_cache = {}


def get_pts(pi):
    """(xy_in, xy_air, xy_if, n_if, data_in, data_air, data_sub).

    The three data entries are (Re, Im) target pairs, or None when W_DATA is 0.
    The substrate points are shared across geometries but their TARGET is not,
    so it is cached per geometry alongside the rest."""
    pi = int(pi)
    if pi not in _pts_cache:
        cy = cy_of(pi)
        xy_if, n_if = sample_interface(N_IFACE, cy)
        xy_in = sample_inside(N_INT, cy)
        xy_air = sample_air(N_AIR, cy)
        if W_DATA > 0.0:
            d_in = fdm_interp(FDM_Total[pi], xy_in)     # u_total inside
            d_air = fdm_interp(FDM_Scat[pi], xy_air)    # u_scat in air
            d_sub = fdm_interp(FDM_Scat[pi], xy_sub_shared)
        else:
            d_in = d_air = d_sub = None
        _pts_cache[pi] = (xy_in, xy_air, xy_if, n_if, d_in, d_air, d_sub)
    return _pts_cache[pi]


# ─────────────────────────────────────────────────────────────────────────────
# Loss — one geometry. The batched loss is the mean of these over the batch.
#
# Terms, all from sample/mie_pinn_sample.py:
#   pde_int   (grad^2 + k_int^2) u_int = 0            inside the cylinder
#   pde_ext   (grad^2 + k^2)     u_scat = 0           in air
#   pde_sub   (grad^2 + k_sub^2) u_scat = 0           in the substrate
#   field     u_int = u_bg + u_scat                   on the cylinder boundary
#   flux      d_n u_int = d_n u_bg + d_n u_scat       on the cylinder boundary
#   pml_air / pml_sub                                 in the two PML regions
#   pml_dir   u_scat = 0                              on the outer PML edge
#
# The flat-interface field and flux terms are absent by construction: net_sub
# IS net_scat, and the background jumps are analytically zero (t = 1 + r), so
# both residuals would be identically zero and would cost two forwards and four
# first-derivative backwards per closure evaluation to confirm it.
# ─────────────────────────────────────────────────────────────────────────────
def loss_terms_single(pi, pml_air, pml_sub, xy_out_dir):
    feats = FEATS[int(pi)]
    # [:4] -- get_pts also returns the three FDM data targets, read
    # separately below when W_DATA > 0.
    xy_in, xy_air, xy_if, n_if = get_pts(pi)[:4]

    # ── interior: complex k_int^2 couples the real and imaginary equations
    iR, iI = net_int(feats, xy_in)
    res_R = laplacian(iR, xy_in) + k_int_sq_re * iR - k_int_sq_im * iI
    res_I = laplacian(iI, xy_in) + k_int_sq_re * iI + k_int_sq_im * iR
    l_pde_int = (res_R ** 2 + res_I ** 2).mean()

    # ── air
    sR, sI = net_scat(feats, xy_air)
    res_R = laplacian(sR, xy_air) + k_ext_sq * sR
    res_I = laplacian(sI, xy_air) + k_ext_sq * sI
    l_pde_ext = (res_R ** 2 + res_I ** 2).mean()

    # ── substrate (same network)
    bR, bI = net_sub(feats, xy_sub_shared)
    res_R = laplacian(bR, xy_sub_shared) + k_sub_sq_re * bR - k_sub_sq_im * bI
    res_I = laplacian(bI, xy_sub_shared) + k_sub_sq_re * bI + k_sub_sq_im * bR
    l_pde_sub = (res_R ** 2 + res_I ** 2).mean()

    # ── cylinder boundary: field and flux continuity
    iR_if, iI_if = net_int(feats, xy_if)
    sR_if, sI_if = net_scat(feats, xy_if)
    bgR, bgI = u_bg_air_re(xy_if), u_bg_air_im(xy_if)
    l_field = ((iR_if - (sR_if + bgR)) ** 2 +
               (iI_if - (sI_if + bgI)) ** 2).mean()

    dn_iR = normal_deriv(iR_if, xy_if, n_if)
    dn_iI = normal_deriv(iI_if, xy_if, n_if)
    dn_sR = normal_deriv(sR_if, xy_if, n_if)
    dn_sI = normal_deriv(sI_if, xy_if, n_if)
    dn_bR = du_bg_air_dn_re(xy_if, n_if)
    dn_bI = du_bg_air_dn_im(xy_if, n_if)
    l_flux = ((dn_iR - (dn_sR + dn_bR)) ** 2 +
              (dn_iI - (dn_sI + dn_bI)) ** 2).mean()

    # ── PML
    l_pml_air = pml_pde_residual(net_scat, feats, pml_air, k_ext_sq, 0.0)
    l_pml_sub = pml_pde_residual(net_sub, feats, pml_sub,
                                 k_sub_sq_re, k_sub_sq_im)

    oR, oI = net_scat(feats, xy_out_dir)
    l_pml_dir = (oR ** 2 + oI ** 2).mean()

    # ── FDM supervision, training geometries only
    if W_DATA > 0.0:
        d_in, d_air, d_sub = get_pts(pi)[4:]
        l_data_int = ((iR - d_in[0]) ** 2 + (iI - d_in[1]) ** 2).mean()
        l_data_scat = 0.5 * (((sR - d_air[0]) ** 2 + (sI - d_air[1]) ** 2).mean()
                             + ((bR - d_sub[0]) ** 2
                                + (bI - d_sub[1]) ** 2).mean())
    else:
        l_data_int = torch.zeros((), device=DEVICE)
        l_data_scat = torch.zeros((), device=DEVICE)

    return (l_pde_int, l_pde_ext, l_pde_sub, l_field, l_flux,
            l_pml_air, l_pml_sub, l_pml_dir, l_data_int, l_data_scat)


# L0_SOURCE is set with the other training constants near the top, so the
# startup banner can report it.
# Measured by executing sample/mie_pinn_sample.py to its L0 block with its
# cylinder moved to this sweep's index-2 centre.
L0_SAMPLE = [1.975e+09,   # pde_int
             4.230e+03,   # pde_air
             3.629e+03,   # pde_sub
             1.223e+00,   # field
             1.106e+03,   # flux
             1.226e+04,   # pml_air
             8.058e+03,   # pml_sub
             7.686e-03]   # pml_dir

# L0 defaults, so the first (measuring) call to loss_batch is well defined.
L0 = [1.0] * 8

W = [W_PDE_INT, W_PDE_EXT, W_PDE_SUB, W_FIELD, W_FLUX,
     W_PML, W_PML, W_PML_DIR, W_DATA, W_DATA]


# ---------------------------------------------------------------------------
# BATCHED-OVER-GEOMETRIES PATH  (final/final_deeponet.py's build_batched)
#
# Every geometry's collocation set is concatenated into ONE tensor per term,
# with a matching per-point feature matrix, so each term costs one forward and
# one double-backward for the whole batch instead of N_PARAM_BATCH of each.

_batch_cache = {}


def build_batched(param_indices, pml_air, pml_sub, xy_out_dir):
    """Concatenate one batch's collocation sets. Cached on the index tuple:
    Adam redraws the batch every RESAMPLE_ADAM epochs, but L-BFGS reuses one
    for RESAMPLE_LBFGS * max_iter closure evaluations, which is where paying
    the concatenation once rather than per evaluation matters."""
    key = tuple(int(p) for p in param_indices)
    if key in _batch_cache:
        return _batch_cache[key]
    # NOT cleared here. With ACCUM_CHUNK there are ceil(N_PARAM_BATCH/chunk)
    # distinct keys live at once, and clearing on every miss would rebuild each
    # chunk's concatenation on every closure evaluation instead of once per
    # point set -- which is most of what the cache is for. redraw_shared()
    # clears it, so the entries never outlive their points.

    def leaf(ts):
        """A genuine leaf rather than differentiating through torch.cat, so the
        double-backward graph keeps the same shape as the per-geometry path."""
        return torch.cat(ts, dim=0).detach().requires_grad_(True)

    n = len(key)
    rows = lambda pi, m: FEATS[pi].expand(m, -1)

    # Geometry-dependent samples: each geometry brings its own points.
    xy_in  = leaf([get_pts(pi)[0] for pi in key])
    xy_if  = leaf([get_pts(pi)[2] for pi in key])
    n_if   = torch.cat([get_pts(pi)[3] for pi in key], dim=0).detach()

    # Shared samples: the SAME coordinates for every geometry, tiled so each
    # block pairs with its own features. Tiling costs a copy but keeps one
    # kernel per term and one code path for both architectures.
    def tile_pml(part):
        xy, coef = part[0], part[1:]
        return (leaf([xy] * n),) + tuple(torch.cat([c] * n, dim=0)
                                         for c in coef)

    out = {
        "n": n,
        "xy_in":  xy_in,  "f_in":  torch.cat([rows(pi, N_INT) for pi in key], 0),
        "xy_if":  xy_if,  "f_if":  torch.cat([rows(pi, N_IFACE) for pi in key], 0),
        "n_if":   n_if,
        "xy_dir":  torch.cat([xy_out_dir] * n, dim=0).detach(),
        "f_dir":   torch.cat([rows(pi, xy_out_dir.shape[0]) for pi in key], 0),
    }

    if W_DATA > 0.0:
        cat2 = lambda j, c: torch.cat([get_pts(pi)[j][c] for pi in key], dim=0)
        out.update({
            "d_in":  (cat2(4, 0), cat2(4, 1)),
            "d_air": (cat2(5, 0), cat2(5, 1)),
            "d_sub": (cat2(6, 0), cat2(6, 1)),
        })

    air_pts = [get_pts(pi)[1] for pi in key]
    f_air = torch.cat([rows(pi, N_AIR) for pi in key], 0)
    f_sub = torch.cat([rows(pi, N_SUB) for pi in key], 0)
    f_pml = torch.cat([rows(pi, N_PML) for pi in key], 0)

    def col(m, v):
        return torch.full((m, 1), float(v), device=DEVICE)

    assert not (MERGE_EXTERIOR == "all" and W_DATA > 0.0), (
        'MERGE_EXTERIOR = "all" folds the air and substrate samples into the '
        'PML pass, which returns residuals rather than field values, so the '
        'data terms have nothing to compare against. Use "none" or "pml" with '
        "W_DATA > 0 (the default is \"none\", which measured fastest anyway).")

    if MERGE_EXTERIOR == "all":
        # Segment order: air PDE, substrate PDE, air PML, substrate PML.
        segs = (n * N_AIR, n * N_SUB, n * N_PML, n * N_PML)
        n_phys = segs[0] + segs[1]        # inside [0,1]^2, so the stretch is 1

        def coef(j):
            """Stretch component j of (sx_re, sx_im, sy_re, sy_im, dsx, dsy),
            per point: unity on the physical-domain segments, the sampled
            values on the two PML bands."""
            return torch.cat([col(n_phys, 1.0 if j in (0, 2) else 0.0)]
                             + [pml_air[1 + j]] * n
                             + [pml_sub[1 + j]] * n, dim=0)

        out.update({
            "segs":  segs,
            "ext":   (leaf(air_pts + [xy_sub_shared] * n
                           + [pml_air[0]] * n + [pml_sub[0]] * n),)
                     + tuple(coef(j) for j in range(6)),
            "f_ext": torch.cat([f_air, f_sub, f_pml, f_pml], dim=0),
            "k2_re": torch.cat([col(segs[0], k_ext_sq), col(segs[1], k_sub_sq_re),
                                col(segs[2], k_ext_sq), col(segs[3], k_sub_sq_re)]),
            "k2_im": torch.cat([col(segs[0], 0.0), col(segs[1], k_sub_sq_im),
                                col(segs[2], 0.0), col(segs[3], k_sub_sq_im)]),
        })
    else:
        out.update({
            "xy_air": leaf(air_pts),             "f_air": f_air,
            "xy_sub": leaf([xy_sub_shared] * n), "f_sub": f_sub,
        })
        if MERGE_EXTERIOR == "pml":
            # Both bands need the full operator, so this adds no arithmetic --
            # only k^2 and the reference wavenumber differ, and both are already
            # baked per point (the reference wavenumber into the stretch).
            m = n * N_PML
            out.update({
                "segs":  (m, m),
                "ext":   (leaf([pml_air[0]] * n + [pml_sub[0]] * n),)
                         + tuple(torch.cat([pml_air[1 + j]] * n
                                           + [pml_sub[1 + j]] * n, dim=0)
                                 for j in range(6)),
                "f_ext": torch.cat([f_pml, f_pml], dim=0),
                "k2_re": torch.cat([col(m, k_ext_sq), col(m, k_sub_sq_re)]),
                "k2_im": torch.cat([col(m, 0.0), col(m, k_sub_sq_im)]),
            })
        else:
            out.update({
                "pml_air": tile_pml(pml_air),
                "pml_sub": tile_pml(pml_sub),
                "f_pml":   f_pml,
            })

    _batch_cache[key] = out
    return out


def loss_terms_batched(param_indices, pml_air, pml_sub, xy_out_dir):
    """The same eight terms as loss_terms_single, over the whole batch."""
    B = build_batched(param_indices, pml_air, pml_sub, xy_out_dir)

    xy_in = B["xy_in"]
    iR, iI = net_int(B["f_in"], xy_in)
    res_R = laplacian(iR, xy_in) + k_int_sq_re * iR - k_int_sq_im * iI
    res_I = laplacian(iI, xy_in) + k_int_sq_re * iI + k_int_sq_im * iR
    l_pde_int = (res_R ** 2 + res_I ** 2).mean()

    if MERGE_EXTERIOR == "all":
        # One pass over air PDE + substrate PDE + both PML bands; the four
        # reported terms are slice means of the per-point residual.
        sq = pml_residual_sq(net_scat, B["f_ext"], B["ext"],
                             B["k2_re"], B["k2_im"])
        a, b, c, _ = B["segs"]
        l_pde_ext = sq[:a].mean()
        l_pde_sub = sq[a:a + b].mean()
        l_pml_air = sq[a + b:a + b + c].mean()
        l_pml_sub = sq[a + b + c:].mean()
    else:
        xy_air = B["xy_air"]
        sR, sI = net_scat(B["f_air"], xy_air)
        res_R = laplacian(sR, xy_air) + k_ext_sq * sR
        res_I = laplacian(sI, xy_air) + k_ext_sq * sI
        l_pde_ext = (res_R ** 2 + res_I ** 2).mean()

        xy_sub = B["xy_sub"]
        bR, bI = net_sub(B["f_sub"], xy_sub)
        res_R = laplacian(bR, xy_sub) + k_sub_sq_re * bR - k_sub_sq_im * bI
        res_I = laplacian(bI, xy_sub) + k_sub_sq_re * bI + k_sub_sq_im * bR
        l_pde_sub = (res_R ** 2 + res_I ** 2).mean()

        if MERGE_EXTERIOR == "pml":
            sq = pml_residual_sq(net_scat, B["f_ext"], B["ext"],
                                 B["k2_re"], B["k2_im"])
            m = B["segs"][0]
            l_pml_air = sq[:m].mean()
            l_pml_sub = sq[m:].mean()
        else:
            l_pml_air = pml_pde_residual(net_scat, B["f_pml"], B["pml_air"],
                                         k_ext_sq, 0.0)
            l_pml_sub = pml_pde_residual(net_sub, B["f_pml"], B["pml_sub"],
                                         k_sub_sq_re, k_sub_sq_im)

    xy_if, n_if = B["xy_if"], B["n_if"]
    iR_if, iI_if = net_int(B["f_if"], xy_if)
    sR_if, sI_if = net_scat(B["f_if"], xy_if)
    bgR, bgI = u_bg_air_re(xy_if), u_bg_air_im(xy_if)
    l_field = ((iR_if - (sR_if + bgR)) ** 2 +
               (iI_if - (sI_if + bgI)) ** 2).mean()

    dn_iR = normal_deriv(iR_if, xy_if, n_if)
    dn_iI = normal_deriv(iI_if, xy_if, n_if)
    dn_sR = normal_deriv(sR_if, xy_if, n_if)
    dn_sI = normal_deriv(sI_if, xy_if, n_if)
    dn_bR = du_bg_air_dn_re(xy_if, n_if)
    dn_bI = du_bg_air_dn_im(xy_if, n_if)
    l_flux = ((dn_iR - (dn_sR + dn_bR)) ** 2 +
              (dn_iI - (dn_sI + dn_bI)) ** 2).mean()

    oR, oI = net_scat(B["f_dir"], B["xy_dir"])
    l_pml_dir = (oR ** 2 + oI ** 2).mean()

    # ── FDM supervision, training geometries only
    if W_DATA > 0.0:
        d_in, d_air, d_sub = B["d_in"], B["d_air"], B["d_sub"]
        l_data_int = ((iR - d_in[0]) ** 2 + (iI - d_in[1]) ** 2).mean()
        l_data_scat = 0.5 * (((sR - d_air[0]) ** 2
                              + (sI - d_air[1]) ** 2).mean()
                             + ((bR - d_sub[0]) ** 2
                                + (bI - d_sub[1]) ** 2).mean())
    else:
        l_data_int = torch.zeros((), device=DEVICE)
        l_data_scat = torch.zeros((), device=DEVICE)

    return (l_pde_int, l_pde_ext, l_pde_sub, l_field, l_flux,
            l_pml_air, l_pml_sub, l_pml_dir, l_data_int, l_data_scat)


def loss_batch(param_indices, pml_air, pml_sub, xy_out_dir):
    """Mean over the geometry batch of each normalised term, plus the total."""
    if BATCH_GEOMETRIES:
        terms = loss_terms_batched(param_indices, pml_air, pml_sub, xy_out_dir)
    else:
        acc = None
        for pi in param_indices:
            t = loss_terms_single(pi, pml_air, pml_sub, xy_out_dir)
            acc = list(t) if acc is None else [x + y for x, y in zip(acc, t)]
        terms = [t / float(len(param_indices)) for t in acc]
    total = sum(w * t / l0 for w, t, l0 in zip(W, terms, L0))
    return total, terms


def backward_batch(param_indices, pml_air, pml_sub, xy_out_dir):
    """Populate .grad for the batch and return [total] + the ten terms, as
    plain floats. The single place both training phases step through, so the
    chunking cannot drift away from the whole-batch path."""
    idx = [int(p) for p in param_indices]
    m = float(len(idx))
    # The looping path evaluates one geometry at a time by construction, so a
    # chunk larger than 1 has nothing to concatenate there.
    chunk = len(idx) if ACCUM_CHUNK in (None, 0) else max(1, int(ACCUM_CHUNK))
    if not BATCH_GEOMETRIES:
        chunk = 1

    tot, acc = 0.0, None
    for c0 in range(0, len(idx), chunk):
        sl = idx[c0:c0 + chunk]
        if BATCH_GEOMETRIES and len(sl) > 1:
            terms = loss_terms_batched(sl, pml_air, pml_sub, xy_out_dir)
        elif len(sl) == 1:
            terms = loss_terms_single(sl[0], pml_air, pml_sub, xy_out_dir)
        else:
            acc_t = None
            for pi in sl:
                t = loss_terms_single(pi, pml_air, pml_sub, xy_out_dir)
                acc_t = list(t) if acc_t is None else \
                    [x + y for x, y in zip(acc_t, t)]
            terms = [t / float(len(sl)) for t in acc_t]

        # terms are the mean over THIS chunk; weighting by len(sl)/m makes the
        # accumulated total the mean over the whole batch, which is what
        # loss_batch returns and what the L0 constants were measured against.
        wc = len(sl) / m
        part = sum(w * t / l0 for w, t, l0 in zip(W, terms, L0)) * wc
        part.backward()
        tot += float(part.detach())
        v = torch.stack([t.detach() for t in terms]).cpu().numpy() * wc
        acc = v if acc is None else acc + v
    return [tot] + acc.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Shared (gap-independent) point sets. Redrawn alongside the per-geometry ones.
# ─────────────────────────────────────────────────────────────────────────────
def redraw_shared():
    global xy_sub_shared
    _pts_cache.clear()
    _batch_cache.clear()
    xy_sub_shared = sample_substrate(N_SUB)
    return sample_pml_air(N_PML), sample_pml_sub(N_PML), \
        sample_pml_outer(N_PML_OUT)


xy_sub_shared = None
pml_air_c, pml_sub_c, xy_dir_c = redraw_shared()

# ─────────────────────────────────────────────────────────────────────────────
# Loss normalisation 
# ─────────────────────────────────────────────────────────────────────────────
TERM_NAMES = ["pde_int", "pde_air", "pde_sub", "field", "flux",
              "pml_air", "pml_sub", "pml_dir", "data_int", "data_scat"]

# Both data terms are normalised by the REFERENCE's own mean square 
_tr = [int(p) for p in train_idx]
L0_DATA = max(float(np.mean([np.mean(np.abs(FDM_Scat[p]) ** 2) for p in _tr])),
              1e-12)
L0_DATA_INT = L0_DATA_SCAT = L0_DATA
# The data loss must never see a held-out gap, or the test relL2 stops being a
# generalisation measure. Every batch is drawn from train_idx, and this makes
# that a checked property rather than an assumed one.
_TEST_SET = set(int(p) for p in test_idx)
if W_DATA > 0.0 and SINGLE_GAP_INDEX is None:
    assert not (_TEST_SET & set(int(p) for p in train_idx)), \
        "train and test gaps overlap -- the data loss would leak"

print("Loss normalisation constants ...")
# Measured ONE GEOMETRY AT A TIME and averaged, never as one batched graph.
# The mean of the per-geometry means is what loss_batch returns anyway, so the
# numbers are the same -- but building the whole batch here would hold
# N_PARAM_BATCH double-backward graphs at once, which is exactly the peak that
# ACCUM_CHUNK exists to bound. At N_PARAM_BATCH = 16 that alone was enough to
# exhaust an 11.6 GiB card before training had started, whatever ACCUM_CHUNK
# was set to, because this call does not go through backward_batch.
_n0_batch = train_idx[:N_PARAM_BATCH]
_acc0 = None
for _p0 in _n0_batch:
    _t0 = loss_terms_single(int(_p0), pml_air_c, pml_sub_c, xy_dir_c)
    _v0 = np.array([float(t.detach()) for t in _t0])
    _acc0 = _v0 if _acc0 is None else _acc0 + _v0
    del _t0                       # frees this geometry's graph before the next
_measured = [max(v / len(_n0_batch), 1e-12) for v in _acc0]
# L0_SAMPLE describes a network whose EXTERIOR field is built from the |k_sub|
# band alone. ARCH="single" shares one embedding carrying BOTH bands, so at
# initialisation its exterior residual is dominated by |k_int| content the
# exterior field does not need. Measured, L0_pde_air comes out 2.0e+05 times
# sample's and L0_pml_air 1.5e+05 times -- against (|k_int|/|k_sub|)^4 =
# 1.6e+05, a second derivative squared, which is exactly what that leakage
# predicts. Dividing by sample's constants would start those terms five orders
# out of balance and spend the run undoing it, so the combination is refused
# rather than left to fail quietly.
if ARCH == "single" and L0_SOURCE == "sample":
    raise SystemExit(
        'ARCH="single" with L0_SOURCE="sample" is refused: sample\'s L0 '
        "constants assume a single-band exterior network and this one is "
        "two-band, so the exterior terms would start ~2e5 out of balance. "
        'Use L0_SOURCE="measured", which normalises this architecture on its '
        "own initialisation and starts every term at 1.")

L0 = (list(L0_SAMPLE) if L0_SOURCE == "sample" else _measured[:8]) \
     + [L0_DATA_INT, L0_DATA_SCAT]
for _nm, _v, _m in zip(TERM_NAMES, L0, _measured):
    _tag = "" if L0_SOURCE != "sample" else f"   (this net measured {_m:.3e})"
    print(f"  L0_{_nm:<8s} = {_v:.3e}{_tag}")
print(f"  source: {L0_SOURCE}")

# ─────────────────────────────────────────────────────────────────────────────
# relL2 monitor. The loss is a poor proxy for solution quality so the composite
# scattered field is compared against the FDM on a subsampled grid:
#   inside the cylinder  net_int - u_bg_air   (net_int is the TOTAL field)
#   in the substrate     net_scat
#   in air               net_scat
# which is the same object the FDM's v is, everywhere.
# ─────────────────────────────────────────────────────────────────────────────
_ms = 3
MON_X = FDM_X[::_ms, ::_ms]
MON_Y = FDM_Y[::_ms, ::_ms]
MON_shape = MON_X.shape
MON_xy = torch.tensor(np.stack([MON_X.ravel(), MON_Y.ravel()], 1)
                      .astype(np.float32), device=DEVICE)
MON_in_sub = FDM_in_sub[::_ms, ::_ms]
MON_ref    = {int(p): FDM_Scat[int(p)][::_ms, ::_ms]
              for p in list(train_idx) + list(test_idx)}
MON_in_cyl = {int(p): FDM_in_cyl[int(p)][::_ms, ::_ms]
              for p in list(train_idx) + list(test_idx)}

# Air-region rms|u_scat| of the reference, averaged over the monitored gaps --
# the number a collapsed-to-zero run fails to reach.
MON_TARGET_RMS = float(np.mean([
    np.sqrt(np.mean(np.abs(MON_ref[int(p)][~MON_in_cyl[int(p)] & ~MON_in_sub]) ** 2))
    for p in list(train_idx) + list(test_idx)]))


def predict_scat(pi):
    """Composite u_scat on the monitor grid for one gap, as numpy complex."""
    feats = FEATS[int(pi)]
    with torch.no_grad():
        sR, sI = net_scat(feats, MON_xy)
        iR, iI = net_int(feats, MON_xy)
        bgR, bgI = u_bg_air_re(MON_xy), u_bg_air_im(MON_xy)
    to_c = lambda R, I: (R.cpu().numpy().reshape(MON_shape) +
                         1j * I.cpu().numpy().reshape(MON_shape))
    return np.where(MON_in_cyl[int(pi)],
                    to_c(iR, iI) - to_c(bgR, bgI), to_c(sR, sI))


def monitor(indices):
    """(mean relL2, mean air rms|u_scat|) over the given parameter indices."""
    net_int.eval(); net_scat.eval()
    rels, amps = [], []
    for pi in indices:
        Us  = predict_scat(pi)
        ref = MON_ref[int(pi)]
        rels.append(float(np.linalg.norm(Us - ref) /
                          (np.linalg.norm(ref) + 1e-12)))
        air = ~MON_in_cyl[int(pi)] & ~MON_in_sub
        amps.append(float(np.sqrt(np.mean(np.abs(Us[air]) ** 2))))
    net_int.train(); net_scat.train()
    return float(np.mean(rels)), float(np.mean(amps))


# ─────────────────────────────────────────────────────────────────────────────
# Training animation — one representative gap, captured periodically.
# The first TEST gap, so the animation shows generalisation rather than fit.
# ─────────────────────────────────────────────────────────────────────────────
FRAME_PI = int(test_idx[0]) if len(test_idx) else int(train_idx[0])
gif_frames = []
N_gif = 200
_FW, _FH = 912 / 90, 816 / 90


def save_frame(label, total_loss):
    cy_f = cy_of(FRAME_PI)
    feats = FEATS[FRAME_PI]
    net_int.eval(); net_scat.eval()
    g = np.linspace(0, 1, N_gif)
    Xg, Yg = np.meshgrid(g, g)
    xy_ev = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], 1)
                         .astype(np.float32), device=DEVICE)
    with torch.no_grad():
        iR, iI = net_int(feats, xy_ev)
        sR, sI = net_scat(feats, xy_ev)
    net_int.train(); net_scat.train()
    to_c = lambda R, I: (R.cpu().numpy().reshape(N_gif, N_gif) +
                         1j * I.cpu().numpy().reshape(N_gif, N_gif))
    U_int, U_s = to_c(iR, iI), to_c(sR, sI)

    in_cyl_f = cyl_mask(Xg, Yg, cy_f)
    in_sub_f = Yg < Y_SUB
    bg_air = (np.exp(1j * (kx * Xg + ky * (Yg - Y_SUB))) +
              r_s * np.exp(1j * (kx * Xg - ky * (Yg - Y_SUB))))
    bg_sub = t_s * np.exp(1j * (kx * Xg + ky_sub * (Yg - Y_SUB)))
    U_scat_f  = np.where(in_cyl_f, 0j, U_s)
    U_total_f = np.where(in_cyl_f, U_int,
                         np.where(in_sub_f, bg_sub + U_s, bg_air + U_s))

    matplotlib.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(_FW, _FH))
    fig.suptitle(f"{label}   |   gap = {gap_of(FRAME_PI):.3f} [TEST]   "
                 f"|   Loss = {total_loss:.3e}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    circ = dict(color="cyan", lw=1.5, fill=False, zorder=5, ls="--")
    subk = dict(color="yellow", lw=1.2, zorder=6, ls="--")

    def _panel(ax, data, title, cmap, sym):
        v = data[np.isfinite(data)]
        sc = max(float(np.percentile(np.abs(v), 98)) if len(v) else 1.0, 1e-12)
        im = ax.imshow(data, origin="lower", extent=[0, 1, 0, 1], cmap=cmap,
                       vmin=-sc if sym else 0.0, vmax=sc,
                       interpolation="bilinear")
        ax.add_patch(plt.Circle((cx, cy_f), a, **circ))
        ax.axhline(Y_SUB, **subk)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.90)

    _panel(axes[0, 0], U_scat_f.real,     "Scattered  Re(u_scat)",  "RdBu_r", True)
    _panel(axes[0, 1], U_total_f.real,    "Total      Re(u_total)", "RdBu_r", True)
    _panel(axes[1, 0], np.abs(U_scat_f),  "Scattered  |u_scat|",    "hot",    False)
    _panel(axes[1, 1], np.abs(U_total_f), "Total      |u_total|",   "inferno", False)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    if HAVE_IMAGEIO:
        gif_frames.append(imageio.imread(buf))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Adam
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(all_params, lr=LR_ADAM)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(ADAM_EPOCHS, 1), eta_min=1e-5)

loss_history = []
hist_step    = []
hist_terms   = [[] for _ in TERM_NAMES]
hist_eval_step, hist_eval_train, hist_eval_test, hist_eval_amp = [], [], [], []

t0 = time.time()
print(f"\n--- Phase 1: Adam ({ADAM_EPOCHS} epochs) ---")
print(f"{'Ep':>6} | {'Total':>9} | {'PDEin':>8} | {'PDEair':>8} | {'PDEsub':>8} "
      f"| {'Field':>8} | {'Flux':>8} | {'PMLair':>8} | {'PMLsub':>8} "
      f"| {'PMLdir':>8} | {'DatInt':>8} | {'DatSc':>8} | {'s':>6}")
print("-" * 118)

param_batch = train_idx[:N_PARAM_BATCH]

for ep in range(1, ADAM_EPOCHS + 1):
    if ep == 1 or ep % RESAMPLE_ADAM == 0:
        param_batch = np.random.choice(train_idx, size=N_PARAM_BATCH,
                                       replace=False)
        pml_air_c, pml_sub_c, xy_dir_c = redraw_shared()
        for _p in param_batch:
            get_pts(_p)

    optimizer.zero_grad()
    vals = backward_batch(param_batch, pml_air_c, pml_sub_c, xy_dir_c)
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()
    scheduler.step()

    loss_history.append(vals[0])
    hist_step.append(ep)
    for _h, _v in zip(hist_terms, vals[1:]):
        _h.append(_v)

    if ep == 1 or ep % RESAMPLE_ADAM == 0:
        print(f"{ep:>6} | {vals[0]:>9.2e} | " +
              " | ".join(f"{v:>8.2e}" for v in vals[1:]) +
              f" | {time.time()-t0:>6.1f}")

    if ep == 1 or ep % EVAL_EVERY == 0:
        _rl_tr, _amp = monitor(train_idx)
        _rl_te, _    = monitor(test_idx) if len(test_idx) else (float("nan"), 0)
        hist_eval_step.append(ep)
        hist_eval_train.append(_rl_tr)
        hist_eval_test.append(_rl_te)
        hist_eval_amp.append(_amp)
        print(f"  [eval ep {ep:>6}]  relL2 train {_rl_tr:.4f}  "
              f"test {_rl_te:.4f}  |  rms|u_scat|(air) {_amp:.4f} "
              f"(target {MON_TARGET_RMS:.4f})")

    if ep == 1 or ep % 500 == 0:
        save_frame(f"Adam epoch {ep}", vals[0])

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: L-BFGS
#
# The loop is bounded by CLOSURE EVALUATIONS, not outer steps: with
# max_iter = 20 an outer-step bound would silently mean 20x the work plus the
# strong-Wolfe line-search probes on top.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Phase 2: L-BFGS ({LBFGS_STEPS} closure evals) ---")

last_terms = [None]


def make_closure(indices, pml_a, pml_s, xy_d):
    """Called many times per lbfgs.step() -- once per inner iteration and once
    per rejected trial point in the line search -- so it must not log or sync.
    It stashes the detached term tensors as one stack; the loop reads them once
    per outer step."""
    def closure():
        lbfgs.zero_grad()
        vals = backward_batch(indices, pml_a, pml_s, xy_d)
        last_terms[0] = vals
        # L-BFGS reads the VALUE from the return and the gradients from
        # .grad, which backward_batch has already populated, so a detached
        # scalar is all it needs here.
        return torch.tensor(vals[0], device=DEVICE)
    return closure


step_count = [0]
outer_step = 0
lbfgs = None
_t_lbfgs = time.time()

while step_count[0] < LBFGS_STEPS:
    if outer_step % RESAMPLE_LBFGS == 0:
        param_batch = np.random.choice(train_idx, size=N_PARAM_BATCH,
                                       replace=False)
        pml_air_c, pml_sub_c, xy_dir_c = redraw_shared()
        for _p in param_batch:
            get_pts(_p)
        lbfgs = torch.optim.LBFGS(all_params, lr=0.1,
                                  max_iter=LBFGS_MAX_ITER, history_size=50,
                                  line_search_fn="strong_wolfe")

    _before = step_count[0]
    closure = make_closure(param_batch, pml_air_c, pml_sub_c, xy_dir_c)

    def _counting_closure():
        step_count[0] += 1
        return closure()

    lbfgs.step(_counting_closure)
    outer_step += 1

    vals = last_terms[0]
    loss_history.append(vals[0])
    hist_step.append(ADAM_EPOCHS + step_count[0])
    for _h, _v in zip(hist_terms, vals[1:]):
        _h.append(_v)

    if step_count[0] // 500 != _before // 500:
        print(f"{ADAM_EPOCHS + step_count[0]:>6} | {vals[0]:>9.2e} | " +
              " | ".join(f"{v:>8.2e}" for v in vals[1:]) +
              f" | {time.time()-t0:>6.1f}")

    if step_count[0] // EVAL_EVERY != _before // EVAL_EVERY:
        _rl_tr, _amp = monitor(train_idx)
        _rl_te, _    = monitor(test_idx) if len(test_idx) else (float("nan"), 0)
        hist_eval_step.append(ADAM_EPOCHS + step_count[0])
        hist_eval_train.append(_rl_tr)
        hist_eval_test.append(_rl_te)
        hist_eval_amp.append(_amp)
        print(f"  [eval L-BFGS {step_count[0]:>6}]  relL2 train {_rl_tr:.4f}  "
              f"test {_rl_te:.4f}  |  rms|u_scat|(air) {_amp:.4f} "
              f"(target {MON_TARGET_RMS:.4f})")

    if step_count[0] // 2000 != _before // 2000:
        save_frame(f"L-BFGS eval {step_count[0]}", vals[0])

print(f"\nTotal training time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
torch.save({
    "net_int_state_dict":  raw_module(net_int).state_dict(),
    "net_scat_state_dict": raw_module(net_scat).state_dict(),
    "B_ext": B_ext.cpu(), "B_int": B_int.cpu(),
    "FEAT_MIN": FEAT_MIN, "FEAT_MAX": FEAT_MAX,
    "gap": GAP_VALS, "cy": CY_VALS,
    "train_idx": train_idx, "test_idx": test_idx,
    "hyperparams": {
        # Consumers should take the trunk width from B_ext.shape[1] rather than
        # from N_FOURIER, so an added Fourier band cannot desynchronise them.
        "N_FOURIER": N_FOURIER, "PHYS_DIM": PHYS_DIM,
        "INIT_OUT_RMS": INIT_OUT_RMS, "L0_SOURCE": L0_SOURCE,
        "ACCUM_CHUNK": ACCUM_CHUNK,
        "W_DATA": W_DATA, "L0_DATA_INT": L0_DATA_INT,
        "L0_DATA_SCAT": L0_DATA_SCAT,
        "RESAMPLE_LBFGS": RESAMPLE_LBFGS,
        "ARCH": ARCH, "MLP_HIDDEN": MLP_HIDDEN, "MLP_NEURONS": MLP_NEURONS,
        "BRANCH_HIDDEN": BRANCH_HIDDEN, "BRANCH_NEURONS": BRANCH_NEURONS,
        "TRUNK_HIDDEN": TRUNK_HIDDEN, "TRUNK_NEURONS": TRUNK_NEURONS,
        "BRANCH_DIM": BRANCH_DIM,
        "N_RE": N_RE, "N_IM": N_IM, "SUB_N_RE": SUB_N_RE,
        "SUB_N_IM": SUB_N_IM, "Y_SUB": Y_SUB, "lam": lam, "a": a, "cx": cx,
        "theta_inc": float(theta_inc),
        "d_pml": d_pml, "sigma_max": sigma_max, "pml_order": pml_order,
        "ADAM_EPOCHS": ADAM_EPOCHS, "LBFGS_STEPS": LBFGS_STEPS,
        "N_PARAM_BATCH": N_PARAM_BATCH,
        "W_PDE_INT": W_PDE_INT, "W_PDE_EXT": W_PDE_EXT,
        "W_PDE_SUB": W_PDE_SUB, "W_FIELD": W_FIELD, "W_FLUX": W_FLUX,
        "W_PML": W_PML, "W_PML_DIR": W_PML_DIR,
        "N_INT": N_INT, "N_AIR": N_AIR, "N_SUB": N_SUB,
        "N_IFACE": N_IFACE, "N_PML": N_PML,
    },
}, CKPT_PATH)
print(f"Saved: {CKPT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Loss plot
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(hist_step, loss_history, color="black", lw=1.5, label="Total")
_colors = ["royalblue", "darkorange", "seagreen", "crimson", "purple",
           "grey", "brown", "olive", "magenta", "teal"]
for _nm, _h, _c in zip(TERM_NAMES, hist_terms, _colors):
    ax.semilogy(hist_step, _h, color=_c, lw=0.8, label=_nm)
ax.axvline(ADAM_EPOCHS, color="k", ls=":", lw=1.0)
ax.text(ADAM_EPOCHS, ax.get_ylim()[1], "  L-BFGS", va="top", fontsize=9)
ax.set_xlabel("Adam epoch / L-BFGS closure evaluation")
ax.set_ylabel("loss term (unnormalised)")
ax.set_title(f"{RUN_TAG} — training loss")
ax.legend(fontsize=8, ncol=3)
ax.grid(alpha=0.3)
plt.savefig(f"{FIG_DIR}/{RUN_TAG}_loss.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR}/{RUN_TAG}_loss.png")

# ─────────────────────────────────────────────────────────────────────────────
# relL2 plot — train and test on one axis, which is the generalisation picture
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(hist_eval_step, hist_eval_train, color="royalblue", lw=1.5,
            marker="o", ms=2.5, label=f"train ({len(train_idx)} gaps)")
if len(test_idx):
    ax.semilogy(hist_eval_step, hist_eval_test, color="crimson", lw=1.5,
                marker="s", ms=2.5, label=f"test ({len(test_idx)} gaps)")
ax.axvline(ADAM_EPOCHS, color="k", ls=":", lw=1.0)
ax.set_xlabel("Adam epoch / L-BFGS closure evaluation")
ax.set_ylabel("relL2 vs FDM")
ax.set_title(f"{RUN_TAG} — relative L2 error against the FDM")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.savefig(f"{FIG_DIR}/{RUN_TAG}_relL2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR}/{RUN_TAG}_relL2.png")

# ─────────────────────────────────────────────────────────────────────────────
# Per-gap error — the generalisation result, gap by gap
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Generalisation ---")
per_gap = {}
for pi in list(train_idx) + list(test_idx):
    Us  = predict_scat(pi)
    ref = MON_ref[int(pi)]
    per_gap[int(pi)] = float(np.linalg.norm(Us - ref) /
                             (np.linalg.norm(ref) + 1e-12))

_tr = [per_gap[int(p)] for p in train_idx]
_te = [per_gap[int(p)] for p in test_idx] if len(test_idx) else [float("nan")]
for pi in sorted(per_gap):
    tag = "TEST " if pi in set(int(p) for p in test_idx) else "train"
    print(f"  [{pi:>2}] {tag}  gap = {gap_of(pi):.4f}   relL2 = {per_gap[pi]:.4f}")
print(f"\n  Mean relL2 (train) : {np.mean(_tr):.4f}")
print(f"  Mean relL2 (test)  : {np.mean(_te):.4f}")
print(f"  Max  relL2 (test)  : {np.max(_te):.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot([gap_of(p) for p in train_idx], _tr, "o", color="royalblue",
        ms=7, label="train")
if len(test_idx):
    ax.plot([gap_of(p) for p in test_idx], _te, "s", color="crimson",
            ms=8, label="test (held out)")
ax.set_xlabel("gap  (cylinder bottom to substrate)")
ax.set_ylabel("relL2 vs FDM")
ax.set_title(f"{RUN_TAG} — error across the gap sweep")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig(f"{FIG_DIR}/{RUN_TAG}_error_vs_gap.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR}/{RUN_TAG}_error_vs_gap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Sweep grid — predicted |u_scat| at every gap
# ─────────────────────────────────────────────────────────────────────────────
if SINGLE_GAP_INDEX is not None:
    print(f"Skipped: sweep grid (SINGLE_GAP_INDEX — {N_PARAM - 1} of "
          f"{N_PARAM} panels would show an untrained parameter)")
ncol = 5 if N_PARAM <= 25 else 10
nrow = int(np.ceil(N_PARAM / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.5 * nrow))
axes = np.atleast_2d(axes)
circ_g = dict(color="cyan", lw=1.0, fill=False, zorder=5, ls="--")
sub_g  = dict(color="yellow", lw=0.8, zorder=6, ls="--")
_test_set = set(int(p) for p in test_idx)
for p in range(N_PARAM):
    ax = axes.ravel()[p]
    if int(p) in MON_ref:
        data = np.abs(predict_scat(p))
        ax.imshow(data, origin="lower", extent=[0, 1, 0, 1], cmap="hot",
                  vmin=0.0, vmax=max(float(np.percentile(data, 99)), 1e-12),
                  interpolation="bilinear")
        ttl = f"gap {gap_of(p):.3f}  relL2 {per_gap[int(p)]:.3f}"
    else:
        ax.text(0.5, 0.5, "no FDM", ha="center", va="center", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ttl = f"gap {gap_of(p):.3f}"
    ax.add_patch(plt.Circle((cx, cy_of(p)), a, **circ_g))
    ax.axhline(Y_SUB, **sub_g)
    ax.set_title(ttl + (" [TEST]" if p in _test_set else ""), fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
for q in range(N_PARAM, nrow * ncol):
    axes.ravel()[q].axis("off")
fig.suptitle(f"{RUN_TAG} — predicted |u_scat| across the gap sweep",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.8)
if SINGLE_GAP_INDEX is None:
    plt.savefig(f"{FIG_DIR}/{RUN_TAG}_sweep_grid.png", dpi=140,
                bbox_inches="tight")
    print(f"Saved: {FIG_DIR}/{RUN_TAG}_sweep_grid.png")
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Field comparison — PINN / FDM / difference, at the test gaps plus two others
# ─────────────────────────────────────────────────────────────────────────────
show = list(dict.fromkeys(
    [int(p) for p in test_idx] +
    [int(train_idx[0]), int(train_idx[len(train_idx) // 2]),
     int(train_idx[-1])]))[:4]
fig, axes = plt.subplots(3, len(show), figsize=(3.4 * len(show), 9.6),
                         squeeze=False)
circ3 = dict(color="cyan", lw=1.0, fill=False, zorder=5, ls="--")
sub3  = dict(color="yellow", lw=0.9, zorder=6, ls="--")
for col, pi in enumerate(show):
    Us  = predict_scat(pi)
    ref = MON_ref[int(pi)]
    sc  = max(float(np.percentile(np.abs(ref), 99)), 1e-12)
    tag = " [TEST]" if pi in _test_set else ""
    for row, (data, ttl) in enumerate(
            [(Us.real,  f"gap {gap_of(pi):.3f}{tag}\nPINN Re(u_scat)"),
             (ref.real, "FDM Re(u_scat)"),
             (Us.real - ref.real, "PINN - FDM")]):
        ax = axes[row, col]
        ax.imshow(data, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
                  vmin=-sc, vmax=sc, interpolation="bilinear")
        ax.add_patch(plt.Circle((cx, cy_of(pi)), a, **circ3))
        ax.axhline(Y_SUB, **sub3)
        ax.set_title(ttl, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{RUN_TAG} — scattered field against the FDM", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{FIG_DIR}/{RUN_TAG}_field_comparison.png", dpi=140,
            bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG_DIR}/{RUN_TAG}_field_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Training animation
# ─────────────────────────────────────────────────────────────────────────────
if HAVE_IMAGEIO and gif_frames:
    mp4 = f"{FIG_DIR}/{RUN_TAG}_training.mp4"
    try:
        imageio.mimwrite(mp4, gif_frames, format="mp4", fps=8,
                         macro_block_size=None)
        print(f"Saved: {mp4}")
    except Exception as e:
        # imageio needs an ffmpeg or pyav backend for mp4; neither is required
        # for anything else here, so a missing one must not fail the run after
        # a multi-hour training.
        print(f"  MP4 write skipped ({type(e).__name__}: "
              f"{str(e).splitlines()[0][:120]})")
