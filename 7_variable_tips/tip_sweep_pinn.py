"""
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
  N_RE, N_IM, lam (hence k), TIP_X, APEX_Y, R_APEX, theta_inc

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

  * sample_inside / sample_outside / sample_interface take alpha;
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
import time
import imageio.v2 as imageio
from io import BytesIO

# ─────────────────────────────────────────────────────────────────────────────
# FIXED PARAMETERS — identical to fdm_afm_tip_sweep.py
# ─────────────────────────────────────────────────────────────────────────────
N_RE      = 3.0
N_IM      = 30.0
lam       = 0.5
k         = 2.0 * np.pi / lam
TIP_X     = 0.5
APEX_Y    = 0.45
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

_APEX_CENTRE_Y = APEX_Y + R_APEX

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

N_INT         = 1000   # interior collocation points  (inside tip)
N_EXT         = 2000   # exterior collocation points  (outside tip, [0,1]²)
N_IFACE       = 800    # interface phi=0 points
N_PML         = 3000   # PML-band collocation points (shared across batch)
N_DIR_SIDE    = 500    # outer Dirichlet points per side (×4 sides)
N_PARAM_BATCH = 4      # half-angles per gradient step (was 8). Every closure
                       # costs N_PARAM_BATCH x the single-angle work, so this
                       # halves it; with 16 training angles two steps still
                       # cover them all.

# Evaluate all N_PARAM_BATCH half-angles in ONE forward per term instead of
# looping over them. At these sizes (1000-3000 points into a 4x128 MLP) the GPU
# kernels are tiny and launch-latency bound, so 8 sequential passes waste most
# of the card; concatenating them gives ~8x fewer, larger launches. The cost is
# peak memory: one graph covering every angle rather than one angle at a time.
# Set False to fall back to the per-angle path (loss_fn_single) if this OOMs.
BATCH_ANGLES  = True

# ── PML parameters — keep identical to fdm_afm_tip_sweep.py ──────────────────
D_PML         = 0.5
PML_POW       = 3
PML_SIGMA_MAX = 20.0

ADAM_EPOCHS = 40000   
LBFGS_STEPS = 100000       
LBFGS_MAX_ITER = 24   # inner iterations per outer step. 
LR_ADAM     = 5e-4

# Loss weights — same relative scheme as afm_tip_pinn.py; every term is
# divided by its initialisation value L0 so all terms start at 1.
W_PDE_INT   = 1.0
W_PDE_EXT   = 1.0
W_FIELD     = 50.0
W_FLUX      = 50.0
W_PML       = 20.0
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
USE_PHYSICAL_L0_FIELD = False
W_DATA_SCAT = 0.0
W_DATA_INT  = 0.0

# Architecture
N_FOURIER = 32
HIDDEN    = 4
NEURONS   = 128
PHYS_DIM  = 5    # alpha-derived shape features (see _raw_features)

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
FDM_PATH = "variable_tips/inputs/fdm_afm_tip_sweep.npz"
assert os.path.exists(FDM_PATH), \
    f"FDM sweep not found at '{FDM_PATH}'. Run fdm_afm_tip_sweep.py first."

_fdm       = np.load(FDM_PATH)
half_angle = _fdm["half_angle"].astype(np.float64)   # [N_PARAM], radians
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

# consistency check against the sweep's parameters
_meta = _fdm["meta"]   # [N_RE, N_IM, lam, k, TIP_X, APEX_Y, R_APEX]
assert np.allclose(_meta[:3], [N_RE, N_IM, lam]), \
    f"Material/wave mismatch vs FDM sweep: sweep meta = {_meta[:3]}"
assert np.allclose(_meta[4:], [TIP_X, APEX_Y, R_APEX]), \
    f"Geometry mismatch vs FDM sweep: sweep meta = {_meta[4:]}"

print("=" * 66)
print("Parametric PINN — AFM Tip (TM, half-angle sweep)")
print("=" * 66)
print(f"  n (fixed)      = {N_RE} + {N_IM}i,   λ (fixed) = {lam}")
print(f"  tip (fixed)    : apex=({TIP_X},{APEX_Y}), R={R_APEX}")
print(f"  half-angles    : {N_PARAM} values ∈ "
      f"[{alpha_deg.min():.1f}°, {alpha_deg.max():.1f}°]")
print(f"  Train / Test   : {len(train_idx)} / {len(test_idx)}")
print(f"  test angles    = {[f'{alpha_deg[p]:.2f}°' for p in test_idx]}")
print(f"  Batch size (α) : {N_PARAM_BATCH}")
print(f"  MLP: {HIDDEN} hidden × {NEURONS} neurons | Fourier {N_FOURIER} | "
      f"PhysDim {PHYS_DIM}")
print(f"  |k_int²| = {k_int_sq_mag:.1f}  (interior PDE non-dim scale)")
print("=" * 66)

# ─────────────────────────────────────────────────────────────────────────────
# Alpha-parametrised signed distance function (numpy + torch)
# ─────────────────────────────────────────────────────────────────────────────
def _alpha_consts(alpha):
    sin_a = float(np.sin(alpha))
    cos_a = float(np.cos(alpha))
    cone_vertex_y = APEX_Y + R_APEX - R_APEX / sin_a
    tangent_y     = _APEX_CENTRE_Y - R_APEX * sin_a
    return sin_a, cos_a, cone_vertex_y, tangent_y


def phi_np(x, y, alpha):
    """Signed distance to the tip with half-angle alpha (numpy)."""
    sin_a, cos_a, cone_vertex_y, tangent_y = _alpha_consts(alpha)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q_x    = np.abs(x - TIP_X)
    d_cone = q_x * cos_a - (y - cone_vertex_y) * sin_a
    d_apex = np.sqrt((x - TIP_X)**2 + (y - _APEX_CENTRE_Y)**2) - R_APEX
    return np.where(y < tangent_y, d_apex, d_cone)


def phi_torch(xy, alpha):
    """Signed distance (torch). xy: [N,2] → [N,1]. Autograd-friendly."""
    sin_a, cos_a, cone_vertex_y, tangent_y = _alpha_consts(alpha)
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    q_x    = torch.abs(x - TIP_X)
    d_cone = q_x * cos_a - (y - cone_vertex_y) * sin_a
    d_apex = torch.sqrt((x - TIP_X)**2 + (y - _APEX_CENTRE_Y)**2 + 1e-12) - R_APEX
    return torch.where(y < tangent_y, d_apex, d_cone)


def outward_normal(xy, alpha):
    """Outward unit normal via autograd of phi_torch. xy requires_grad."""
    d = phi_torch(xy, alpha)
    g = torch.autograd.grad(d.sum(), xy, create_graph=True)[0]
    norm = torch.sqrt((g**2).sum(dim=1, keepdim=True) + 1e-12)
    return g / norm


def tip_mask(Xg, Yg, alpha):
    return phi_np(Xg, Yg, alpha) < 0.0


def tip_outline_xy(alpha, y_top=1.0, n=200):
    sin_a, cos_a, cone_vertex_y, tangent_y = _alpha_consts(alpha)
    yc = _APEX_CENTRE_Y
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
def _raw_features(alpha):
    _, _, cone_vertex_y, _ = _alpha_consts(alpha)
    return np.array([
        alpha,
        np.sin(alpha),
        np.cos(alpha),
        np.tan(alpha),
        (1.0 - cone_vertex_y) * np.tan(alpha),   # shank half-width at y = 1
    ], dtype=np.float32)


_all_feats = np.stack([_raw_features(float(av)) for av in half_angle])
FEAT_MIN = _all_feats.min(axis=0)
FEAT_MAX = _all_feats.max(axis=0)


def param_to_features(alpha):
    """Normalised [1, PHYS_DIM] feature tensor on DEVICE, values in [-1,1].
    Works for arbitrary alpha within the sweep's feature ranges, so the
    trained model can be queried at unseen half-angles."""
    raw   = _raw_features(float(alpha))
    normd = 2.0 * (raw - FEAT_MIN) / (FEAT_MAX - FEAT_MIN + 1e-10) - 1.0
    return torch.tensor(normd, device=DEVICE).unsqueeze(0)

# ─────────────────────────────────────────────────────────────────────────────
# Fourier feature embeddings — k fixed → single-scale (as in afm_tip_pinn.py)
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(1)
B_ext = torch.randn(2, N_FOURIER, device=DEVICE) * k
B_int = torch.randn(2, N_FOURIER, device=DEVICE) * abs(k_int_c)

FOURIER_DIM = 2 * N_FOURIER
IN_DIM      = FOURIER_DIM + PHYS_DIM


def _phys_rows(phys_feats, n):
    """Accept either a single [1, PHYS_DIM] row to broadcast over n points (the
    per-angle path, and all the eval/plot helpers) or an already per-point
    [n, PHYS_DIM] matrix (the batched path, where consecutive blocks of rows
    belong to different half-angles)."""
    return phys_feats.expand(n, -1) if phys_feats.shape[0] == 1 else phys_feats


def embed_ext(xy, phys_feats):
    proj    = xy @ B_ext
    fourier = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
    return torch.cat([fourier, _phys_rows(phys_feats, xy.shape[0])], dim=-1)


def embed_int(xy, phys_feats):
    proj    = xy @ B_int
    fourier = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
    return torch.cat([fourier, _phys_rows(phys_feats, xy.shape[0])], dim=-1)

# ─────────────────────────────────────────────────────────────────────────────
# Networks — parametric MLPs (ellipse_pinn.py architecture)
# ─────────────────────────────────────────────────────────────────────────────
class ParametricMLP(nn.Module):
    """Outputs [Re(u), Im(u)]; receives physics features alongside xy."""
    def __init__(self, embed_fn):
        super().__init__()
        self.embed_fn = embed_fn
        layers = [nn.Linear(IN_DIM, NEURONS), nn.Tanh()]
        for _ in range(HIDDEN - 1):
            layers += [nn.Linear(NEURONS, NEURONS), nn.Tanh()]
        layers += [nn.Linear(NEURONS, 2)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, phys_feats, xy):
        out = self.net(self.embed_fn(xy, phys_feats))
        return out[:, 0:1], out[:, 1:2]


net_int  = ParametricMLP(embed_int).to(DEVICE)   # total field inside tip
net_scat = ParametricMLP(embed_ext).to(DEVICE)   # scattered field outside


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
        for _w_net in (net_int, net_scat):
            _w_r, _w_i = _w_net(_w_pf, _w_xy)
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

# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────
def u_inc_re(xy):
    return torch.cos(kx_inc * xy[:, 0:1] + ky_inc * xy[:, 1:2])

def u_inc_im(xy):
    return torch.sin(kx_inc * xy[:, 0:1] + ky_inc * xy[:, 1:2])

def du_inc_dn_re(xy, nxy):
    phase = kx_inc * xy[:, 0:1] + ky_inc * xy[:, 1:2]
    kn    = kx_inc * nxy[:, 0:1] + ky_inc * nxy[:, 1:2]
    return -kn * torch.sin(phase)

def du_inc_dn_im(xy, nxy):
    phase = kx_inc * xy[:, 0:1] + ky_inc * xy[:, 1:2]
    kn    = kx_inc * nxy[:, 0:1] + ky_inc * nxy[:, 1:2]
    return  kn * torch.cos(phase)

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
# EXPERIMENT, ACTIVE (control value is 0.005): the interior skin depth is
#   delta = 1/Im(k_int) = 0.00265 = lambda/188
# so _MARGIN_IN = 0.005 is 1.88*delta -- every interior collocation point sits
# BEYOND the boundary layer, where the true field has already decayed to ~15%.
# Nothing therefore teaches net_int the layer profile, yet the true solution
# satisfies d_n u_int = -i k_int u_int automatically because of it. Without
# that profile the field and flux interface conditions are mutually
# inconsistent, which is why enforcing field continuity harder makes the flux
# residual climb during training.
# At 2e-4 (0.075*delta) the layer is ~10% of the tip's area, so uniform
# rejection sampling would put ~100 of N_INT points inside it, and B_int is
# scaled by |k_int| = 379 so the network can already represent that scale.
_MARGIN_IN  = 2e-4     # was 0.005 (= 1.88*delta); 2e-4 = 0.075*delta
_MARGIN_OUT = 0.008


def _tip_bbox(alpha):
    """Bounding box around the tip (including the shank in the top PML)."""
    _, _, cone_vertex_y, _ = _alpha_consts(alpha)
    y_hi   = 1.0 + D_PML
    half_w = (y_hi - cone_vertex_y) * np.tan(alpha)
    x_lo = TIP_X - half_w - 0.02
    x_hi = TIP_X + half_w + 0.02
    y_lo = APEX_Y - R_APEX - 0.02
    return x_lo, x_hi, y_lo, y_hi


def sample_inside(n, alpha):
    """Points strictly inside the tip (incl. shank inside the top PML)."""
    x_lo, x_hi, y_lo, y_hi = _tip_bbox(alpha)
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.column_stack([
            np.random.uniform(x_lo, x_hi, n * 8),
            np.random.uniform(y_lo, y_hi, n * 8),
        ])
        d = phi_np(xy[:, 0], xy[:, 1], alpha)
        pts.append(xy[d < -_MARGIN_IN])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)


def sample_outside(n, alpha):
    """Points strictly outside the tip in the physical box [0,1]²."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 4, 2)
        d = phi_np(xy[:, 0], xy[:, 1], alpha)
        pts.append(xy[d > _MARGIN_OUT])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)


def sample_interface(n, alpha):
    """
    Points on the tip surface phi = 0 with autograd outward normals.
    Thin-shell rejection + projection along the SDF gradient, exactly as
    afm_tip_pinn.py, but with the alpha-parametrised SDF.
    """
    x_lo, x_hi, y_lo, y_hi = _tip_bbox(alpha)
    eps = 0.02
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.column_stack([
            np.random.uniform(x_lo, x_hi, n * 12),
            np.random.uniform(y_lo, y_hi, n * 12),
        ])
        d = phi_np(xy[:, 0], xy[:, 1], alpha)
        pts.append(xy[np.abs(d) < eps])
    xy = np.vstack(pts)[:n]

    xy_t = torch.tensor(xy.astype(np.float32), device=DEVICE, requires_grad=True)
    d_t  = phi_torch(xy_t, alpha)
    n_t  = outward_normal(xy_t, alpha)
    xy_proj = (xy_t - d_t * n_t).detach()

    xy_proj.requires_grad_(True)
    nxy = outward_normal(xy_proj, alpha).detach()

    xy_proj = xy_proj.detach().clone().requires_grad_(True)
    return xy_proj, nxy

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

    sx_c  = 1.0 + 1j * sig_x / k
    sy_c  = 1.0 + 1j * sig_y / k
    dsx_c = 1j * dsig_x / k
    dsy_c = 1j * dsig_y / k

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
pts_cache       = {}  # pi → (xy_in, xy_out, xy_if, n_if)
pml_parts_cache = {}  # pi → [branch dicts] partitioning the CURRENT PML points


def sample_spatial_for(pi):
    alpha = float(half_angle[pi])
    xy_in       = sample_inside(N_INT, alpha)
    xy_out      = sample_outside(N_EXT, alpha)
    xy_if, n_if = sample_interface(N_IFACE, alpha)
    return xy_in, xy_out, xy_if, n_if


def get_pts(pi):
    pi = int(pi)
    if pi not in pts_cache:
        pts_cache[pi] = sample_spatial_for(pi)
    return pts_cache[pi]


def get_pml_parts(pi, pml):
    """PARTITION the shared PML points into the two operator branches for this
    alpha: inside the shank (net_int, k_int²) and outside it (net_scat, k²).

    Previously a 0/1 mask blended both networks' outputs at every point:
        pR = mask*piR + (1-mask)*psR
    which required evaluating BOTH nets, plus their second derivatives, on all
    N_PML points -- and 84-94% of net_int's share was then multiplied by zero
    and discarded (measured: 6.15/10.91/16.37% inside at alpha=10/20/30 deg).
    The PML term was 72% of every closure evaluation.

    Splitting is EXACTLY equivalent, not an approximation: the mask is a
    non-differentiable 0/1 constant, so at a mask==1 point pR == piR and hence
    d(pR)/dx == d(piR)/dx. Each point is owned by exactly one branch. Summing
    squared residuals over both branches and dividing by N_PML reproduces the
    old .mean() over all points.

    Cache is cleared whenever the PML sample is redrawn.
    """
    pi = int(pi)
    if pi not in pml_parts_cache:
        alpha  = float(half_angle[pi])
        d      = phi_np(pml['xy_np'][:, 0], pml['xy_np'][:, 1], alpha)
        inside = d < 0.0
        parts  = []
        for sel, use_int, ksq_re, ksq_im in (
                (inside,  True,  k_int_sq_re, k_int_sq_im),
                (~inside, False, k_ext_sq,    0.0)):
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
        pml_parts_cache[pi] = parts
    return pml_parts_cache[pi]

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

# ─────────────────────────────────────────────────────────────────────────────
# Loss-normalisation globals — set after the L0 pass below; initialised to 1
# so loss_fn_single is callable before they are computed.
# ─────────────────────────────────────────────────────────────────────────────
L0_PDE_INT = L0_PDE_EXT = L0_FIELD = L0_FLUX = 1.0
L0_PML = L0_DIRICHLET = L0_DATA_SCAT = L0_DATA_INT = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Loss function for a single half-angle index.
# PDE, BCs, PML operator and non-dimensionalisation are IDENTICAL to
# afm_tip_pinn.py; the spatial points belong to THIS parameter's geometry and
# both networks are conditioned on the alpha feature vector (ellipse_pinn.py).
# ─────────────────────────────────────────────────────────────────────────────
def loss_fn_single(pi, pml, xy_dir):
    pi = int(pi)
    xy_in, xy_out, xy_if, n_if = get_pts(pi)
    phys_feats = param_to_features(half_angle[pi])

    # ── 1. Interior PDE: (∇² + k_int²) u_int = 0  (complex k_int²)
    iR, iI   = net_int(phys_feats, xy_in)
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

    # ── 3. Interface: Ez continuity  u_int = u_inc + u_scat  on phi = 0
    iR_if, iI_if = net_int(phys_feats, xy_if)
    sR_if, sI_if = net_scat(phys_feats, xy_if)
    inc_R_if     = u_inc_re(xy_if)
    inc_I_if     = u_inc_im(xy_if)
    l_field = ((iR_if - (sR_if + inc_R_if))**2 +
               (iI_if - (sI_if + inc_I_if))**2).mean()

    # ── 4. Interface: flux continuity with SDF-autograd normals
    dn_iR   = normal_deriv(iR_if, xy_if, n_if)
    dn_iI   = normal_deriv(iI_if, xy_if, n_if)
    dn_sR   = normal_deriv(sR_if, xy_if, n_if)
    dn_sI   = normal_deriv(sI_if, xy_if, n_if)
    dn_incR = du_inc_dn_re(xy_if, n_if)
    dn_incI = du_inc_dn_im(xy_if, n_if)
    l_flux  = ((dn_iR - (dn_sR + dn_incR))**2 +
               (dn_iI - (dn_sI + dn_incI))**2).mean()

    # ── 5. PML: UPML-stretched Helmholtz in the PML band. Operator switches
    #    per point (afm_tip_pinn.py): inside the shank → net_int + k_int²;
    #    outside → net_scat + k². The inside mask depends on alpha.
    # Each branch owns a disjoint subset of the PML points, so only the network
    # that actually governs a point is evaluated there (see get_pml_parts).
    pml_sq = None
    for part in get_pml_parts(pi, pml):
        xy_p = part['xy']
        C    = part['coef']

        net    = net_int if part['use_int'] else net_scat
        pR, pI = net(phys_feats, xy_p)

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

    # sum over both branches / N_PML == the old .mean() over all PML points
    l_pml = pml_sq / float(N_PML)

    # ── 6. Dirichlet on the outer PML boundary — u_scat = 0
    dR, dI = net_scat(phys_feats, xy_dir)
    l_dirichlet = (dR**2 + dI**2).mean()

    total = (W_PDE_INT   * l_pde_int   / L0_PDE_INT +
             W_PDE_EXT   * l_pde_ext   / L0_PDE_EXT +
             W_FIELD     * l_field     / L0_FIELD   +
             W_FLUX      * l_flux      / L0_FLUX    +
             W_PML       * l_pml       / L0_PML     +
             W_DIRICHLET * l_dirichlet / L0_DIRICHLET)

    # ── 7. FDM data supervision — indexed by THIS half-angle (off by default)
    # Guarded: each interp_fdm call does a GPU->CPU sync, a scipy
    # map_coordinates, and a CPU->GPU copy. Four of them per parameter per
    # closure evaluation (x N_PARAM_BATCH) is a large serialising cost whose
    # result was then multiplied by W_DATA_* = 0 and discarded.
    if W_DATA_SCAT > 0.0 or W_DATA_INT > 0.0:
        fdm_sR = interp_fdm(FDM_scat_re[pi],  xy_out)
        fdm_sI = interp_fdm(FDM_scat_im[pi],  xy_out)
        l_data_scat = ((sR - fdm_sR)**2 + (sI - fdm_sI)**2).mean()

        fdm_iR = interp_fdm(FDM_total_re[pi], xy_in)
        fdm_iI = interp_fdm(FDM_total_im[pi], xy_in)
        l_data_int = ((iR - fdm_iR)**2 + (iI - fdm_iI)**2).mean()

        total = total + W_DATA_SCAT * l_data_scat / L0_DATA_SCAT \
                      + W_DATA_INT  * l_data_int  / L0_DATA_INT
    else:
        # zeros_like keeps device and dtype in step with the live tensors
        l_data_scat = torch.zeros_like(l_pde_int)
        l_data_int  = torch.zeros_like(l_pde_int)

    return (total, l_pde_int, l_pde_ext, l_field, l_flux, l_pml, l_dirichlet,
            l_data_scat, l_data_int)


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
batch_cache = {}   # tuple(param_indices) → assembled batched tensors


def build_batched(param_indices, pml, xy_dir):
    key = tuple(int(p) for p in param_indices)
    if key in batch_cache:
        return batch_cache[key]
    # Only one batch is ever live. Adam draws a fresh angle combination every
    # epoch (so the key essentially never repeats), while L-BFGS reuses one for
    # RESAMPLE_LBFGS*max_iter closure evals -- where this cache does the work.
    # Evicting keeps the concatenated copies from piling up between resamples.
    batch_cache.clear()

    def leaf(ts):
        """Fresh leaf from concatenated per-angle tensors. Building a genuine
        leaf (rather than differentiating through torch.cat) keeps the
        double-backward graph identical in shape to the per-angle path."""
        return torch.cat(ts, dim=0).detach().requires_grad_(True)

    feats  = {pi: param_to_features(half_angle[pi]) for pi in key}
    rows   = lambda pi, m: feats[pi].expand(m, -1)

    xy_in  = leaf([get_pts(pi)[0] for pi in key])
    xy_out = leaf([get_pts(pi)[1] for pi in key])
    xy_if  = leaf([get_pts(pi)[2] for pi in key])
    n_if   = torch.cat([get_pts(pi)[3] for pi in key], dim=0).detach()

    f_in   = torch.cat([rows(pi, N_INT)   for pi in key], dim=0)
    f_out  = torch.cat([rows(pi, N_EXT)   for pi in key], dim=0)
    f_if   = torch.cat([rows(pi, N_IFACE) for pi in key], dim=0)

    # ── PML: gather the two operator branches across ALL angles. Constants
    #    (k², |k²|) are branch-wide, so each branch is one homogeneous tensor.
    branches = []
    for use_int in (True, False):
        xys, cfs, fts = [], [], []
        for pi in key:
            for part in get_pml_parts(pi, pml):
                if part['use_int'] is not use_int:
                    continue
                xys.append(part['xy'])
                cfs.append(part['coef'])
                fts.append(rows(pi, part['xy'].shape[0]))
        if not xys:
            continue
        branches.append({
            'xy':      leaf(xys),
            'coef':    {kk: torch.cat([c[kk] for c in cfs], dim=0)
                        for kk in cfs[0]},
            'feats':   torch.cat(fts, dim=0),
            'use_int': use_int,
            'ksq_re':  k_int_sq_re if use_int else k_ext_sq,
            'ksq_im':  k_int_sq_im if use_int else 0.0,
            'ksq_mag': k_int_sq_mag if use_int else k_ext_sq_mag,
        })

    n_dir = xy_dir.shape[0]
    out = {
        'n':       len(key),
        'xy_in':   xy_in,  'f_in':  f_in,
        'xy_out':  xy_out, 'f_out': f_out,
        'xy_if':   xy_if,  'f_if':  f_if, 'n_if': n_if,
        'pml':     branches,
        'xy_dir':  torch.cat([xy_dir] * len(key), dim=0),
        'f_dir':   torch.cat([rows(pi, n_dir) for pi in key], dim=0),
    }
    batch_cache[key] = out
    return out


def loss_fn_batched(param_indices, pml, xy_dir):
    """Same nine terms as loss_fn_single, averaged over the parameter batch."""
    B = build_batched(param_indices, pml, xy_dir)
    n = B['n']

    # ── 1. Interior PDE
    xy_in    = B['xy_in']
    iR, iI   = net_int(B['f_in'], xy_in)
    res_in_R = (laplacian(iR, xy_in) + k_int_sq_re * iR - k_int_sq_im * iI) / k_int_sq_mag
    res_in_I = (laplacian(iI, xy_in) + k_int_sq_re * iI + k_int_sq_im * iR) / k_int_sq_mag
    l_pde_int = (res_in_R**2 + res_in_I**2).mean()

    # ── 2. Exterior PDE
    xy_out   = B['xy_out']
    sR, sI   = net_scat(B['f_out'], xy_out)
    res_ex_R = (laplacian(sR, xy_out) + k_ext_sq * sR) / k_ext_sq_mag
    res_ex_I = (laplacian(sI, xy_out) + k_ext_sq * sI) / k_ext_sq_mag
    l_pde_ext = (res_ex_R**2 + res_ex_I**2).mean()

    # ── 3./4. Interface field + flux continuity
    xy_if, n_if  = B['xy_if'], B['n_if']
    iR_if, iI_if = net_int(B['f_if'], xy_if)
    sR_if, sI_if = net_scat(B['f_if'], xy_if)
    l_field = ((iR_if - (sR_if + u_inc_re(xy_if)))**2 +
               (iI_if - (sI_if + u_inc_im(xy_if)))**2).mean()

    dn_incR = du_inc_dn_re(xy_if, n_if)
    dn_incI = du_inc_dn_im(xy_if, n_if)
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
        pR, pI = net(part['feats'], xy_p)
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

    # each angle contributes exactly N_PML points across the two branches
    l_pml = pml_sq / float(N_PML * n)

    # ── 6. Dirichlet
    dR, dI = net_scat(B['f_dir'], B['xy_dir'])
    l_dirichlet = (dR**2 + dI**2).mean()

    total = (W_PDE_INT   * l_pde_int   / L0_PDE_INT +
             W_PDE_EXT   * l_pde_ext   / L0_PDE_EXT +
             W_FIELD     * l_field     / L0_FIELD   +
             W_FLUX      * l_flux      / L0_FLUX    +
             W_PML       * l_pml       / L0_PML     +
             W_DIRICHLET * l_dirichlet / L0_DIRICHLET)

    # ── 7. FDM data supervision (off by default; see loss_fn_single)
    if W_DATA_SCAT > 0.0 or W_DATA_INT > 0.0:
        idxs = [int(p) for p in param_indices]
        fdm_sR = torch.cat([interp_fdm(FDM_scat_re[p],  xy_out[j*N_EXT:(j+1)*N_EXT])
                            for j, p in enumerate(idxs)], dim=0)
        fdm_sI = torch.cat([interp_fdm(FDM_scat_im[p],  xy_out[j*N_EXT:(j+1)*N_EXT])
                            for j, p in enumerate(idxs)], dim=0)
        l_data_scat = ((sR - fdm_sR)**2 + (sI - fdm_sI)**2).mean()

        fdm_iR = torch.cat([interp_fdm(FDM_total_re[p], xy_in[j*N_INT:(j+1)*N_INT])
                            for j, p in enumerate(idxs)], dim=0)
        fdm_iI = torch.cat([interp_fdm(FDM_total_im[p], xy_in[j*N_INT:(j+1)*N_INT])
                            for j, p in enumerate(idxs)], dim=0)
        l_data_int = ((iR - fdm_iR)**2 + (iI - fdm_iI)**2).mean()

        total = total + W_DATA_SCAT * l_data_scat / L0_DATA_SCAT \
                      + W_DATA_INT  * l_data_int  / L0_DATA_INT
    else:
        l_data_scat = torch.zeros_like(l_pde_int)
        l_data_int  = torch.zeros_like(l_pde_int)

    return (total, l_pde_int, l_pde_ext, l_field, l_flux, l_pml, l_dirichlet,
            l_data_scat, l_data_int)


def backward_batch(param_indices, pml, xy_dir):
    """Accumulate the batch gradient. Caller must zero_grad() before and step()
    after. Returns detached term means.

    BATCH_ANGLES: one forward/backward over all half-angles at once (fewer,
    larger GPU kernels). Otherwise the memory-frugal per-angle loop
    (ellipse_pinn.py): backpropagate each angle immediately and free its graph.
    Both produce the same gradient; see loss_fn_batched on normalisation."""
    n = len(param_indices)
    if BATCH_ANGLES:
        t, *ts = loss_fn_batched(param_indices, pml, xy_dir)
        t.backward()
        return [t.detach()] + [x.detach() for x in ts]
    run = None
    for pi in param_indices:
        t, *ts = loss_fn_single(int(pi), pml, xy_dir)
        (t / n).backward()
        vals = [t.detach()] + [x.detach() for x in ts]
        run  = vals if run is None else [va + vb for va, vb in zip(run, vals)]
    return [v / n for v in run]

# ─────────────────────────────────────────────────────────────────────────────
# Radial-flux diagnostic — detects the ±k sign degeneracy (afm_tip_pinn.py).
# Net outward time-averaged power through a loop enclosing the apex should
# be POSITIVE for a correct outgoing solution.
# ─────────────────────────────────────────────────────────────────────────────
_FLUX_R = float(min(APEX_Y - R_APEX - 0.02, 0.5 - 0.02))
_FLUX_N = 256


def radial_flux_diag(pi):
    alpha = float(half_angle[int(pi)])
    theta = np.linspace(0, 2 * np.pi, _FLUX_N, endpoint=False)
    xs = TIP_X + _FLUX_R * np.cos(theta)
    ys = APEX_Y + _FLUX_R * np.sin(theta)
    good = (phi_np(xs, ys, alpha) > 0.0) & (xs > 0) & (xs < 1) & \
           (ys > 0) & (ys < 1)
    xs, ys = xs[good], ys[good]
    nx, ny = np.cos(theta[good]), np.sin(theta[good])
    xy  = torch.tensor(np.stack([xs, ys], 1).astype(np.float32),
                       device=DEVICE, requires_grad=True)
    nxy = torch.tensor(np.stack([nx, ny], 1).astype(np.float32), device=DEVICE)
    phys_feats = param_to_features(alpha)
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
          ['lpi', 'lpe', 'lf', 'lfl', 'lpml', 'ldir', 'lds', 'ldi']}

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
        _, _lpi, _lpe, _lf, _lfl, _lpml, _ldir, _lds, _ldi = \
            loss_fn_single(int(pi), _pml, _xy_dir)
        _accum['lpi']  += _lpi.item();  _accum['lpe']  += _lpe.item()
        _accum['lf']   += _lf.item();   _accum['lfl']  += _lfl.item()
        _accum['lpml'] += _lpml.item(); _accum['ldir'] += _ldir.item()
        _accum['lds']  += _lds.item();  _accum['ldi']  += _ldi.item()

_den = K_L0 * N_PARAM_BATCH
L0_PDE_INT   = max(_accum['lpi']  / _den, 1e-12)
L0_PDE_EXT   = max(_accum['lpe']  / _den, 1e-12)
L0_FIELD     = max(_accum['lf']   / _den, 1e-12)
L0_FLUX      = max(_accum['lfl']  / _den, 1e-12)
L0_PML       = max(_accum['lpml'] / _den, 1e-12)
L0_DIRICHLET = max(_accum['ldir'] / _den, 1e-12)
L0_DATA_SCAT = max(_accum['lds']  / _den, 1e-12)
L0_DATA_INT  = max(_accum['ldi']  / _den, 1e-12)

pts_cache.clear()
pml_parts_cache.clear()
batch_cache.clear()

print(f"  (averaged over {K_L0}×{N_PARAM_BATCH} param draws)")
print(f"  L0_PDE_INT   = {L0_PDE_INT:.3e}")
print(f"  L0_PDE_EXT   = {L0_PDE_EXT:.3e}")
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
all_params = list(net_int.parameters()) + list(net_scat.parameters())
optimizer  = torch.optim.Adam(all_params, lr=LR_ADAM)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=ADAM_EPOCHS, eta_min=1e-5)

# ─────────────────────────────────────────────────────────────────────────────
# Frame capture helpers (MP4 animation)
# ─────────────────────────────────────────────────────────────────────────────
gif_frames = []
N_grid_gif = 200
_FW, _FH   = 912 / 90, 816 / 90


def save_frame(label, total_loss, pi_sample):
    """4-panel field plot for one representative half-angle."""
    pi_sample = int(pi_sample)
    alpha = float(half_angle[pi_sample])
    phys_feats = param_to_features(alpha)

    net_int.eval(); net_scat.eval()
    with torch.no_grad():
        xg = np.linspace(0, 1, N_grid_gif)
        yg = np.linspace(0, 1, N_grid_gif)
        Xg_f, Yg_f = np.meshgrid(xg, yg)
        in_tip_f = tip_mask(Xg_f, Yg_f, alpha)
        xy_ev_f = torch.tensor(
            np.stack([Xg_f.ravel(), Yg_f.ravel()], axis=1).astype(np.float32),
            device=DEVICE)
        iR_g, iI_g = net_int(phys_feats, xy_ev_f)
        sR_g, sI_g = net_scat(phys_feats, xy_ev_f)

    U_int_f  = (iR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * iI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))
    U_scat_f = (sR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * sI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))
    U_inc_f   = np.exp(1j * (kx_inc * Xg_f + ky_inc * Yg_f))
    U_total_f = np.where(in_tip_f, U_int_f, U_inc_f + U_scat_f)
    U_scat_plot_f = U_total_f - U_inc_f   # scattered = total - incident

    matplotlib.rcParams.update({'font.size': 9})
    fig_f, axes_f = plt.subplots(2, 2, figsize=(_FW, _FH))
    fig_f.suptitle(f"{label}  |  α={np.rad2deg(alpha):.2f}°  |  "
                   f"Loss={total_loss:.3e}", fontsize=12)
    fig_f.tight_layout(rect=[0, 0, 1, 0.96])
    ox, oy = tip_outline_xy(alpha)
    outline_kw_f = dict(color='cyan', lw=1.5, zorder=5, ls='--')

    def _panel(ax_, data, title, cmap, symmetric):
        valid = data[np.isfinite(data)]
        scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
        scale = max(scale, 1e-12)
        vmin  = -scale if symmetric else 0.0
        im    = ax_.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                           cmap=cmap, vmin=vmin, vmax=scale,
                           interpolation='bilinear')
        ax_.plot(ox, oy, **outline_kw_f)
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
hist_pde_int   = []
hist_pde_ext   = []
hist_field     = []
hist_flux      = []
hist_pml       = []
hist_dirichlet = []
hist_data_scat = []
hist_data_int  = []

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
MON_N      = 200          # same grid as the final evaluation

_mg = np.linspace(0, 1, MON_N)
MON_Xg, MON_Yg = np.meshgrid(_mg, _mg)
MON_xy = torch.tensor(
    np.stack([MON_Xg.ravel(), MON_Yg.ravel()], axis=1).astype(np.float32),
    device=DEVICE)
MON_U_INC   = np.exp(1j * (kx_inc * MON_Xg + ky_inc * MON_Yg))
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
            alpha = float(half_angle[pi])
            pf    = param_to_features(alpha)
            iR, iI = net_int(pf, MON_xy)
            sR, sI = net_scat(pf, MON_xy)
            U_int  = (iR.cpu().numpy().reshape(MON_N, MON_N) +
                      1j * iI.cpu().numpy().reshape(MON_N, MON_N))
            U_scat = (sR.cpu().numpy().reshape(MON_N, MON_N) +
                      1j * sI.cpu().numpy().reshape(MON_N, MON_N))
            # U_total: net_int inside the tip, u_inc + net_scat outside
            U_total = np.where(tip_mask(MON_Xg, MON_Yg, alpha),
                               U_int, MON_U_INC + U_scat)
            U_sc    = U_total - MON_U_INC        # composite, matches FDM's u_scat
            fdm     = MON_FDM[pi]
            rels.append(np.linalg.norm(U_sc - fdm) /
                        (np.linalg.norm(fdm) + 1e-12))
            mags.append(np.linalg.norm(U_total) / MON_NRM_INC)
    net_int.train(); net_scat.train()
    return float(np.mean(rels)), float(np.mean(mags))


hist_eval_ep, hist_eval_rel, hist_eval_mag = [], [], []
print(f"\n  relL2 monitor: every {EVAL_EVERY} epochs on {len(test_idx)} test "
      f"angles, {MON_N}x{MON_N} grid")
print(f"  ||u_total||/||u_inc|| target from FDM = {MON_TOT_REF:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Adam
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
print(f"\n--- Phase 1: Adam ({ADAM_EPOCHS} epochs) ---")
print(f"{'Ep':>5} | {'Total':>8} | {'PDEin':>8} | {'PDEex':>8} | "
      f"{'Field':>8} | {'Flux':>8} | {'PML':>8} | {'Dirich':>8} | {'s':>5}")
print("-" * 100)

RESAMPLE_ADAM = 10   # resample spatial points every N Adam epochs

pml_sample = sample_pml(N_PML)
xy_dir     = sample_dirichlet(N_DIR_SIDE)

for ep in range(1, ADAM_EPOCHS + 1):

    # ── resample: clear per-parameter caches; shared sets re-drawn ───────────
    if ep % RESAMPLE_ADAM == 0:
        pts_cache.clear()
        pml_parts_cache.clear()
        batch_cache.clear()
        pml_sample = sample_pml(N_PML)
        xy_dir     = sample_dirichlet(N_DIR_SIDE)

    # ── mini-batch of half-angles from the training split ────────────────────
    batch_idx = np.random.choice(train_idx, size=N_PARAM_BATCH, replace=False)

    optimizer.zero_grad()
    tot, lpi, lpe, lf, lfl, lpml, ldir, lds, ldi = tuple(
        v.item() for v in backward_batch(batch_idx, pml_sample, xy_dir))
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()
    scheduler.step()

    loss_history.append(tot)
    hist_pde_int.append(lpi);   hist_pde_ext.append(lpe)
    hist_field.append(lf);      hist_flux.append(lfl)
    hist_pml.append(lpml);      hist_dirichlet.append(ldir)
    hist_data_scat.append(lds); hist_data_int.append(ldi)

    if ep % 250 == 0 or ep == 1:
        _flux = radial_flux_diag(_pi_mid)
        print(f"{ep:>5} | {tot:>8.2e} | {lpi:>8.2e} | {lpe:>8.2e} | "
              f"{lf:>8.2e} | {lfl:>8.2e} | {lpml:>8.2e} | {ldir:>8.2e} | "
              f"{time.time()-t0:>5.1f}   flux(out>0)={_flux:+.2e}")

    if ep == 1 or ep % EVAL_EVERY == 0:
        _rel, _mag = monitor_metrics(test_idx)
        hist_eval_ep.append(ep)
        hist_eval_rel.append(_rel)
        hist_eval_mag.append(_mag)
        print(f"  [eval ep {ep:>6}]  relL2(test) = {_rel:.4f}   "
              f"||u_total||/||u_inc|| = {_mag:.4f}  (target {MON_TOT_REF:.4f})")

    if ep == 1 or ep % 100 == 0:
        save_frame(f"Adam epoch {ep:>5}", tot, _pi_mid)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: L-BFGS  (float32, eval-bounded)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Phase 2: L-BFGS ({LBFGS_STEPS} closure evals, float32) ---")

RESAMPLE_LBFGS = 25   # resample every N outer steps.
# At max_iter=30 and a 30,000-eval budget (~1000 outer steps) this gives ~40
# redraws, i.e. ~750 closure evals per point set. afm_tip_pinn.py resamples
# every 100 outer steps (~3000 evals per set) and converges; we go stricter
# because memorisation is the failure being fixed here.
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
last_terms = [0.0] * 9

# NOTE ON THE BUDGET: this loop is bounded by CLOSURE EVALUATIONS, not by outer
# steps. `while outer_step < LBFGS_STEPS` with max_iter=30 meant >= 3000*30 =
# 90,000 inner iterations plus strong-Wolfe line-search probes on top -- a ~3x
# silent increase over the eval-bounded form, while the constant kept its name.
# step_count[0] is incremented once per closure call and printed as "evals".
while step_count[0] < LBFGS_STEPS:

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

        lbfgs = torch.optim.LBFGS(all_params, lr=0.05, max_iter=LBFGS_MAX_ITER,
                                  history_size=20, line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-12, tolerance_change=1e-16)

    closure = make_closure(param_batch_lb, pml_lb, xy_dir_lb, last_terms)
    lbfgs.step(closure)
    outer_step += 1

    tot, lpi, lpe, lf, lfl, lpml, ldir, lds, ldi = last_terms
    loss_history.append(tot)
    hist_pde_int.append(lpi);   hist_pde_ext.append(lpe)
    hist_field.append(lf);      hist_flux.append(lfl)
    hist_pml.append(lpml);      hist_dirichlet.append(ldir)
    hist_data_scat.append(lds); hist_data_int.append(ldi)

    # The try/except guards are gone: they only existed because the nets were
    # float64 here while both helpers build float32 tensors internally, so every
    # attempt raised and was swallowed -- the L-BFGS phase contributed no frames
    # to the MP4 and reported flux(out>0)=nan throughout.
    if outer_step % 25 == 0:
        _flux = radial_flux_diag(_pi_mid)
        print(f"  outer {outer_step:>5} (evals {step_count[0]:>6}) | "
              f"loss {tot:.4e} | Field {lf:.2e} | Flux {lfl:.2e} | "
              f"PML {lpml:.2e} | Dirich {ldir:.2e} | {time.time()-t0:.1f}s   "
              f"flux(out>0)={_flux:+.2e}")
        save_frame(f"L-BFGS outer {outer_step}", tot, _pi_mid)

print(f"\nTotal training time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save model
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("variable_tips/models", exist_ok=True)
torch.save({
    'net_int_state_dict':  raw_module(net_int).state_dict(),
    'net_scat_state_dict': raw_module(net_scat).state_dict(),
    'B_ext': B_ext.cpu(),
    'B_int': B_int.cpu(),
    'FEAT_MIN':   FEAT_MIN,
    'FEAT_MAX':   FEAT_MAX,
    'half_angle': half_angle,
    'train_idx':  train_idx,
    'test_idx':   test_idx,
    'hyperparams': {
        'N_FOURIER': N_FOURIER, 'HIDDEN': HIDDEN, 'NEURONS': NEURONS,
        'PHYS_DIM': PHYS_DIM,
        'TIP_X': TIP_X, 'APEX_Y': APEX_Y, 'R_APEX': R_APEX,
        'N_RE': N_RE, 'N_IM': N_IM, 'lam': lam,
        'ADAM_EPOCHS': ADAM_EPOCHS, 'LBFGS_STEPS': LBFGS_STEPS,
        'W_FIELD': W_FIELD, 'W_FLUX': W_FLUX,
        'W_PML': W_PML, 'W_DIRICHLET': W_DIRICHLET,
        'D_PML': D_PML, 'PML_POW': PML_POW,
        'PML_SIGMA_MAX': PML_SIGMA_MAX, 'N_PML': N_PML,
        'MARGIN_IN': _MARGIN_IN, 'MARGIN_OUT': _MARGIN_OUT,
    },
}, "variable_tips/models/afm_tip_pinn_parametric_weights.pt")
print("Saved: variable_tips/models/afm_tip_pinn_parametric_weights.pt")

# ─────────────────────────────────────────────────────────────────────────────
# Training loss plot
# ─────────────────────────────────────────────────────────────────────────────
fig_loss, ax_loss = plt.subplots(figsize=(12, 5))
steps = range(len(loss_history))
ax_loss.semilogy(steps, loss_history,   color='black',        lw=1.5, label='Total')
ax_loss.semilogy(steps, hist_pde_int,   color='royalblue',    lw=0.8, label='PDE int')
ax_loss.semilogy(steps, hist_pde_ext,   color='steelblue',    lw=0.8, label='PDE ext')
ax_loss.semilogy(steps, hist_field,     color='tomato',       lw=0.8, label='Field BC')
ax_loss.semilogy(steps, hist_flux,      color='orange',       lw=0.8, label='Flux BC')
ax_loss.semilogy(steps, hist_pml,       color='mediumpurple', lw=1.2, label='PML')
ax_loss.semilogy(steps, hist_dirichlet, color='darkcyan',     lw=0.8, label='Dirichlet')
ax_loss.axvline(ADAM_EPOCHS, color='gray', ls='--', lw=1.5,
                label=f'Adam → L-BFGS (step {ADAM_EPOCHS})')
ax_loss.set_xlabel("Optimiser step")
ax_loss.set_ylabel("Loss (log scale)")
ax_loss.set_title("Parametric AFM-tip PINN (half-angle sweep) — Training Loss")
ax_loss.legend(fontsize=8, ncol=2)
ax_loss.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("variable_tips/figures/afm_tip_pinn_parametric_loss.png", dpi=150,
            bbox_inches='tight')
print("Saved: variable_tips/figures/afm_tip_pinn_parametric_loss.png")

# ─────────────────────────────────────────────────────────────────────────────
# Convergence of the ACTUAL metric (see the monitor block above for why the
# loss curve is not a reliable stand-in for this)
# ─────────────────────────────────────────────────────────────────────────────
if hist_eval_ep:
    fig_ev, ax_ev = plt.subplots(figsize=(11, 5))
    ax_ev.semilogy(hist_eval_ep, hist_eval_rel, 'o-', color='crimson',
                   ms=4, lw=1.5, label='relL2 vs FDM (test angles)')
    ax_ev.set_xlabel("Adam epoch")
    ax_ev.set_ylabel("relative L2 error vs FDM", color='crimson')
    ax_ev.tick_params(axis='y', labelcolor='crimson')
    ax_ev.grid(True, alpha=0.3)

    ax_ev2 = ax_ev.twinx()
    ax_ev2.plot(hist_eval_ep, hist_eval_mag, 's--', color='steelblue',
                ms=4, lw=1.2, label=r'$\|u_{tot}\|/\|u_{inc}\|$')
    ax_ev2.axhline(MON_TOT_REF, color='steelblue', ls=':', lw=1.5)
    ax_ev2.set_ylabel(r'$\|u_{tot}\|/\|u_{inc}\|$   (dotted = FDM target)',
                      color='steelblue')
    ax_ev2.tick_params(axis='y', labelcolor='steelblue')

    _l1, _b1 = ax_ev.get_legend_handles_labels()
    _l2, _b2 = ax_ev2.get_legend_handles_labels()
    ax_ev.legend(_l1 + _l2, _b1 + _b2, loc='best', fontsize=9)
    ax_ev.set_title("Parametric AFM-tip PINN — convergence of relL2 during Adam")
    plt.tight_layout()
    plt.savefig("variable_tips/figures/afm_tip_pinn_parametric_relL2.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig_ev)
    print("Saved: variable_tips/figures/afm_tip_pinn_parametric_relL2.png")
    print(f"  relL2(test): {hist_eval_rel[0]:.4f} (ep {hist_eval_ep[0]}) -> "
          f"{hist_eval_rel[-1]:.4f} (ep {hist_eval_ep[-1]})")

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
    alpha = float(half_angle[pi])
    phys_feats = param_to_features(alpha)
    net_int.eval(); net_scat.eval()
    with torch.no_grad():
        iR_g, iI_g = net_int(phys_feats, xy_ev)
        sR_g, sI_g = net_scat(phys_feats, xy_ev)
    net_int.train(); net_scat.train()
    U_int  = (iR_g.cpu().numpy().reshape(N_grid, N_grid) +
              1j * iI_g.cpu().numpy().reshape(N_grid, N_grid))
    U_scat = (sR_g.cpu().numpy().reshape(N_grid, N_grid) +
              1j * sI_g.cpu().numpy().reshape(N_grid, N_grid))
    U_inc   = np.exp(1j * (kx_inc * Xg + ky_inc * Yg))
    in_tip  = tip_mask(Xg, Yg, alpha)
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
errors_all = np.full(N_PARAM, np.nan)
for pi in range(N_PARAM):
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
ax1.set_title("Parametric AFM-tip PINN — error over the half-angle sweep")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
plt.tight_layout()
plt.savefig("variable_tips/figures/afm_tip_pinn_parametric_error_vs_angle.png",
            dpi=130, bbox_inches='tight')
print("Saved: variable_tips/figures/afm_tip_pinn_parametric_error_vs_angle.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — PINN sweep overviews: four 4×5 grids over all half-angles
# Same layout as the FDM sweep figures so the two can be compared side-by-side.
# ─────────────────────────────────────────────────────────────────────────────
N_ROWS, N_COLS = 4, 5

_pinn_fields = {}
for pi in range(N_PARAM):
    _, U_s, _, U_t = evaluate_fields(pi)
    _pinn_fields[pi] = (U_s, U_t)


def plot_sweep_grid(get_data, title, cmap, symmetric, fname):
    matplotlib.rcParams.update({'font.size': 7})
    fig, axes_g = plt.subplots(N_ROWS, N_COLS,
                               figsize=(3 * N_COLS, 3 * N_ROWS))
    outline_kw_g = dict(color='cyan', lw=1.0, zorder=5, ls='--')
    for pi in range(N_PARAM):
        ax_g  = axes_g[pi // N_COLS, pi % N_COLS]
        data  = get_data(pi)
        scale = max(float(np.percentile(np.abs(data), 98)), 1e-12)
        vmin  = -scale if symmetric else 0.0
        ax_g.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                    cmap=cmap, vmin=vmin, vmax=scale,
                    interpolation='bilinear')
        ox, oy = tip_outline_xy(float(half_angle[pi]))
        ax_g.plot(ox, oy, **outline_kw_g)
        tag = " [TEST]" if pi in set(test_idx) else ""
        ax_g.set_title(f"α={alpha_deg[pi]:.2f}°{tag}  "
                       f"relL2={errors_all[pi]:.3f}", fontsize=6.5)
        ax_g.set_xticks([]); ax_g.set_yticks([])
        ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
    fig.suptitle(f"Parametric AFM-tip PINN (half-angle sweep) — {title}",
                 fontsize=11)
    plt.tight_layout()
    path = f"variable_tips/figures/afm_tip_pinn_parametric_{fname}.png"
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


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
fig3, axes3 = plt.subplots(3, n_show, figsize=(5 * n_show, 12))
fig3.suptitle("PINN vs FDM — Re(u_scat) for test half-angles\n"
              "(sorted by relative L2 error)", fontsize=11, fontweight='bold')
outline_kw3 = dict(color='cyan', lw=1.0, zorder=5, ls='--')

for col, (pi, lab) in enumerate(zip(show_pi, show_labels)):
    alpha = float(half_angle[pi])
    err   = errors_all[pi]

    _, U_scat_pinn, _, _ = evaluate_fields(pi)
    U_scat_fdm = fdm_scat_on_grid(pi)
    ox, oy = tip_outline_xy(alpha)

    vmax = max(float(np.percentile(np.abs(U_scat_fdm.real), 98)), 1e-10)
    kw   = dict(origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r',
                vmin=-vmax, vmax=vmax, interpolation='bilinear')
    title_str = f"{lab} — α={np.rad2deg(alpha):.2f}°\nrelL2={err:.3f}"

    im_p = axes3[0, col].imshow(U_scat_pinn.real, **kw)
    axes3[0, col].plot(ox, oy, **outline_kw3)
    axes3[0, col].set_title(f"{title_str}\nPINN Re(u_scat)", fontsize=7)
    axes3[0, col].set_xlim(0, 1); axes3[0, col].set_ylim(0, 1)
    plt.colorbar(im_p, ax=axes3[0, col], shrink=0.8)

    im_f = axes3[1, col].imshow(U_scat_fdm.real, **kw)
    axes3[1, col].plot(ox, oy, **outline_kw3)
    axes3[1, col].set_title("FDM Re(u_scat)", fontsize=7)
    axes3[1, col].set_xlim(0, 1); axes3[1, col].set_ylim(0, 1)
    plt.colorbar(im_f, ax=axes3[1, col], shrink=0.8)

    diff   = U_scat_pinn.real - U_scat_fdm.real
    vmax_d = max(float(np.percentile(np.abs(diff), 99)), 1e-10)
    im_d   = axes3[2, col].imshow(diff, origin='lower', extent=[0, 1, 0, 1],
                                  cmap='PiYG', vmin=-vmax_d, vmax=vmax_d,
                                  interpolation='bilinear')
    axes3[2, col].plot(ox, oy, **outline_kw3)
    axes3[2, col].set_title("PINN − FDM", fontsize=7)
    axes3[2, col].set_xlim(0, 1); axes3[2, col].set_ylim(0, 1)
    plt.colorbar(im_d, ax=axes3[2, col], shrink=0.8)

plt.tight_layout()
plt.savefig("variable_tips/figures/afm_tip_pinn_parametric_field_comparison.png",
            dpi=130, bbox_inches='tight')
print("Saved: variable_tips/figures/afm_tip_pinn_parametric_field_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble MP4 from captured frames
# ─────────────────────────────────────────────────────────────────────────────
if gif_frames:
    mp4_path = "variable_tips/figures/afm_tip_pinn_parametric_training.mp4"
    fps = 6
    hold_frames = fps * 3
    frames_out  = gif_frames + [gif_frames[-1]] * hold_frames
    writer_kw   = dict(fps=fps, codec="libx264", pixelformat="yuv420p",
                       output_params=["-crf", "18"])
    imageio.mimwrite(mp4_path, frames_out, format="mp4", **writer_kw)
    print(f"Saved training animation: {mp4_path}")

print("\nDone.")