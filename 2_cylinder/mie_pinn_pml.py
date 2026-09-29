"""
PINN — 2D Dielectric Cylinder Scattering (TM mode)
================================================================
Plane wave scattering by a dielectric cylinder with a tunable complex
refractive index. Unlike the PEC version, the field penetrates the cylinder
so we need two separate networks:

  net_int  : learns u_int(x,y)   — total field  INSIDE  the cylinder
  net_scat : learns u_scat(x,y)  — scattered field OUTSIDE the cylinder

PHYSICS
-------
Incident field (TM):
    u_inc = exp(i (k x + k y)) = exp(i k (x cosθ + y sinθ))

PDEs:
    (∇² + k_int²) u_int  = 0    inside  D  (disk of radius a)
    (∇² + k_ext²) u_scat = 0    outside D

where
    k_ext = k                          (real, exterior wavenumber)
    k_int = (n_re + i n_im) * k        (complex, interior wavenumber)

k_int² = (n_re + i n_im)² k²  is complex:
    Re(k_int²) = (n_re² - n_im²) k²
    Im(k_int²) = 2 n_re n_im k²

For a complex k_int², the Helmholtz equation for u_int splits as:
    ∇²(Re u_int) + Re(k_int²) Re(u_int) - Im(k_int²) Im(u_int) = 0
    ∇²(Im u_int) + Re(k_int²) Im(u_int) + Im(k_int²) Re(u_int) = 0

For the exterior (k_ext² real):
    ∇²(Re u_scat) + k² Re(u_scat) = 0
    ∇²(Im u_scat) + k² Im(u_scat) = 0

BOUNDARY CONDITIONS
-------------------
On the cylinder boundary ∂D (outward normal n̂):

  1. Ez continuity (field BC):
       u_int = u_inc + u_scat

  2. Normal derivative continuity (flux BC, μ_int = μ_ext = 1):
       ∂u_int/∂n = ∂u_inc/∂n + ∂u_scat/∂n

On the outer domain boundary ∂Ω:
  Sommerfeld ABC (active, drives training):
       ∂u_scat/∂n - i k u_scat = 0   on square boundary [0,1]²
  UPML (diagnostic only, W_PML=0):
       stretched Helmholtz in PML annular band

LOSS TERMS
----------
    L_pde_int  : Helmholtz residual for u_int  (inside cylinder)
    L_pde_scat : Helmholtz residual for u_scat (outside cylinder)
    L_field    : u_int = u_inc + u_scat        on ∂D
    L_flux     : ∂u_int/∂n = ∂(u_inc+u_scat)/∂n  on ∂D
    L_abc      : Sommerfeld on square boundary  (W_ABC > 0, active)
    L_pml      : PML-stretched Helmholtz        (W_PML = 0, diagnostic only)

TUNABLE PARAMETERS
------------------
  N_RE : real part of refractive index
  N_IM : imaginary part (>0 = absorption)
  a    : cylinder radius
  k    : exterior wavenumber
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import map_coordinates
import os
import time
import imageio.v2 as imageio
from io import BytesIO

# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
N_RE   = 3.0    # real part of refractive index
N_IM   = 30.0    # imaginary part  (0 = lossless dielectric, >0 = absorbing)
lam    = 0.5    # wavelength 
k      = 2.0 * np.pi / lam   # exterior wavenumber  (wavelength = 1)
a      = 0.05           # cylinder radius
cx, cy = 0.5, 0.5       # cylinder centre
# Incident wave direction
theta_inc = -np.pi / 4
kx = k * np.cos(theta_inc)
ky = k * np.sin(theta_inc)

# ─────────────────────────────────────────────────────────────────────────────
# Derived wavenumbers
# ─────────────────────────────────────────────────────────────────────────────
k_int        = (N_RE + 1j * N_IM) * k
k_int_sq_re  = float((N_RE**2 - N_IM**2) * k**2)   # Re(k_int²)
k_int_sq_im  = float(2 * N_RE * N_IM * k**2)        # Im(k_int²)
k_ext_sq     = k**2                                  # k² exterior (real)

# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

N_INT   = 1000   # interior PDE points  (inside cylinder)
N_EXT   = 2000   # exterior PDE points  (outside cylinder)
N_IFACE = 800    # interface boundary ∂D
N_OUT   = 200    # outer square boundary (Sommerfeld ABC)

ADAM_EPOCHS = 5000
LBFGS_STEPS = 10000
LR_ADAM     = 8e-4

# Loss weights — all dimensionless and comparable after per-term normalisation.
# Each term is divided by its value at initialisation so every term starts at 1.
# These weights then express genuine relative importance, not raw magnitude.
W_PDE_INT  = 1.0
W_PDE_EXT  = 1.0
W_FIELD    = 50.0
W_FLUX     = 50.0
W_ABC      = 0.0    # Sommerfeld on square boundary 
W_PML      = 1.0    # PML term 
W_DIRICHLET = 0.0    # Dirichlet on outer PML boundary

# ── PML parameters — keep identical to mie_fdm_pml.py ────────────────────────
D_PML        = 0.5     # PML thickness in physical units (~0.5 wavelengths at k=2π)
PML_POW      = 3       # polynomial grading exponent m — must match mie_fdm_pml.py
PML_SIGMA_MAX = 20.0  # peak conductivity (absolute, same units as k)
N_PML        = 8000     # number of PML collocation points

# ── Mixed training (radial → full coordinates) ────────────────────────────────
# net_scat output = alpha * NN1(r) + beta * NN2(x,y)
# USE_BLEND = True  : alpha decays 1→0 and beta grows 0→1 via sigmoid over the
#                     full Adam+L-BFGS training run (radial → full-coord blend).
# USE_BLEND = False : alpha=0, beta=1 throughout — pure full-coordinate NN2 only.
USE_BLEND          = False  # set False to skip the radial branch entirely
BLEND_EPOCH_CENTRE = 0.5   # fraction of total steps where alpha ≈ beta ≈ 0.5
BLEND_SHARPNESS    = 8.0   # controls steepness of the sigmoid transition


N_FOURIER = 32
HIDDEN    = 3
NEURONS   = 96

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")
ka = k * a

# ─────────────────────────────────────────────────────────────────────────────
# Load FDM reference solution
# ─────────────────────────────────────────────────────────────────────────────
FDM_PATH = "inputs/fdm_solution.npz"
if os.path.exists(FDM_PATH):
    _fdm = np.load(FDM_PATH)
    FDM_X        = _fdm["X"]           # shape [N_fdm, N_fdm]
    FDM_Y        = _fdm["Y"]
    FDM_Scat_re  = _fdm["U_scat_re"]   # scattered field real part
    FDM_Scat_im  = _fdm["U_scat_im"]
    FDM_Int_re   = (_fdm["U_total_re"] # total field inside = interior field
                    - _fdm["U_inc_re"])
    FDM_Int_im   = (_fdm["U_total_im"]
                    - _fdm["U_inc_im"])
    # Store total field too for convenience
    FDM_Total_re = _fdm["U_total_re"]
    FDM_Total_im = _fdm["U_total_im"]
    FDM_N        = FDM_X.shape[0]
    USE_FDM_DATA = True
    print(f"  Loaded FDM reference: {FDM_PATH}  (grid {FDM_N}x{FDM_N})")
else:
    USE_FDM_DATA = False
    print(f"  WARNING: FDM reference not found at {FDM_PATH} — data loss disabled")

# Weight for FDM data loss — set to 0 to disable
W_DATA_SCAT = 0.0   # scattered field outside cylinder
W_DATA_INT  = 0.0   # interior field inside cylinder

def interp_fdm(field_grid, xy_torch):
    """
    Bilinear interpolation of a 2D FDM field at arbitrary (x,y) points.
    field_grid : numpy array shape [N_fdm, N_fdm], rows=y, cols=x
    xy_torch   : torch tensor shape [M, 2], columns = (x, y)
    Returns    : torch tensor shape [M, 1] on DEVICE
    """
    xy_np  = xy_torch.detach().cpu().numpy()
    x_pts  = xy_np[:, 0]
    y_pts  = xy_np[:, 1]
    # Map physical coords [0,1] to pixel coords [0, N_fdm-1]
    i_coords = y_pts * (FDM_N - 1)   # y → row index
    j_coords = x_pts * (FDM_N - 1)   # x → col index
    coords   = np.array([i_coords, j_coords])
    vals     = map_coordinates(field_grid, coords, order=1, mode='nearest')
    return torch.tensor(vals.astype(np.float32), device=DEVICE).unsqueeze(1)

print("=" * 60)
print("Dielectric Cylinder PINN (TM mode, two networks)")
print("=" * 60)
print(f"  n            = {N_RE} + {N_IM}i")
print(f"  k_ext        = {k:.4f}")
print(f"  k_int        = {k_int:.4f}")
print(f"  a            = {a},   ka = {ka:.3f}")
print(f"  Re(k_int²)   = {k_int_sq_re:.4f}")
print(f"  Im(k_int²)   = {k_int_sq_im:.4f}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Fourier feature embeddings — one per network, scaled to the relevant k
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(1)
B_ext = torch.randn(2, N_FOURIER, device=DEVICE) * k
B_int = torch.randn(2, N_FOURIER, device=DEVICE) * abs(k_int)

# Radial Fourier features: 1-D random frequencies applied to r = ||(x,y)-(cx,cy)||
# These are purely functions of distance so NN1 is forced to be radially symmetric.
B_rad = torch.randn(1, N_FOURIER, device=DEVICE) * k   # shape [1, N_FOURIER]

def embed_ext(xy):
    proj = xy @ B_ext
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

def embed_int(xy):
    proj = xy @ B_int
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

def embed_rad(xy):
    """Embed the scalar radial distance r from the scatterer centre."""
    r = torch.sqrt((xy[:, 0:1] - cx)**2 + (xy[:, 1:2] - cy)**2 + 1e-8)
    proj = r.float() @ B_rad          # [N, N_FOURIER]
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

IN_DIM     = 2 * N_FOURIER   # used by both MLP and RadialMLP

# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Outputs [Re(u), Im(u)] as two separate scalars."""
    def __init__(self, embed_fn):
        super().__init__()
        self.embed = embed_fn
        layers = [nn.Linear(IN_DIM, NEURONS), nn.Tanh()]
        for _ in range(HIDDEN - 1):
            layers += [nn.Linear(NEURONS, NEURONS), nn.Tanh()]
        layers += [nn.Linear(NEURONS, 2)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        out = self.net(self.embed(xy))
        return out[:, 0:1], out[:, 1:2]   # Re, Im


class MixedScatNet(nn.Module):
    """
    Mixed radial + full-coordinate scattered-field network.

    Output: u_scat(x,y) = alpha * NN1(r)  +  beta * NN2(x,y)

    where  r = sqrt((x-cx)^2 + (y-cy)^2)  is the radial distance from the
    scatterer centre.

    NN1 (radial branch)  — embed_rad → MLP.
        Because its embedding depends only on r, it is strictly radially
        symmetric. This gives the network an inductive bias toward the
        near-circular symmetry of the scattered field.

    NN2 (full branch)    — embed_ext → MLP  (same Fourier features as before).
        This has full (x,y) access so it can capture the slight angular
        asymmetry introduced by the plane-wave direction.

    alpha, beta are *not* learnable; they are set externally via set_blend()
    and follow a smooth sigmoid schedule during training.
    """
    def __init__(self):
        super().__init__()
        self.nn1 = MLP(embed_rad)   # radially symmetric branch
        self.nn2 = MLP(embed_ext)   # full-coordinate branch
        # blending weights (not Parameters — set by the scheduler)
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('beta',  torch.tensor(0.0))

    def set_blend(self, alpha: float, beta: float):
        """Called by the training loop to update the blending coefficients."""
        self.alpha.fill_(alpha)
        self.beta.fill_(beta)

    def forward(self, xy):
        r1, i1 = self.nn1(xy)
        r2, i2 = self.nn2(xy)
        return (self.alpha * r1 + self.beta * r2,
                self.alpha * i1 + self.beta * i2)


TOTAL_STEPS = ADAM_EPOCHS + LBFGS_STEPS   # blend spans the full training run

def blend_schedule(step: int) -> tuple[float, float]:
    """
    Compute (alpha, beta) for the current global training step.

    If USE_BLEND is False: always returns (0.0, 1.0) — pure full-coordinate NN2.

    If USE_BLEND is True: sigmoid over TOTAL_STEPS steps:
      At step 0            : alpha ≈ 1, beta ≈ 0  (pure radial)
      At BLEND_EPOCH_CENTRE: alpha = beta = 0.5
      At step TOTAL_STEPS  : alpha ≈ 0, beta ≈ 1  (pure full-coord)

    The outputs satisfy alpha + beta = 1 at all times.
    """
    if not USE_BLEND:
        return 0.0, 1.0
    t = (step / max(TOTAL_STEPS - 1, 1) - BLEND_EPOCH_CENTRE) * BLEND_SHARPNESS
    beta  = float(torch.sigmoid(torch.tensor(t)))
    alpha = 1.0 - beta
    return alpha, beta


net_int  = MLP(embed_int).to(DEVICE)       # total field inside cylinder
net_scat = MixedScatNet().to(DEVICE)       # mixed radial+full scattered field

n_int  = sum(p.numel() for p in net_int.parameters())
n_scat = sum(p.numel() for p in net_scat.parameters())
print(f"\n  net_int      parameters: {n_int:,}")
print(f"  net_scat     parameters: {n_scat:,}")
if USE_BLEND:
    print(f"  (net_scat = alpha*NN1(r) + beta*NN2(x,y), sigmoid blend over Adam+L-BFGS steps)")
else:
    print(f"  (net_scat = NN2(x,y) only — blend disabled, alpha=0, beta=1 throughout)")

# ─────────────────────────────────────────────────────────────────────────────
# Incident field  u_inc = exp(ikx)
# ─────────────────────────────────────────────────────────────────────────────
# u_inc = exp(i(kx*x + ky*y))
def u_inc_re(xy):   return torch.cos(kx * xy[:, 0:1] + ky * xy[:, 1:2])
def u_inc_im(xy):   return torch.sin(kx * xy[:, 0:1] + ky * xy[:, 1:2])

# ∂u_inc/∂n = i(kx nx + ky ny) u_inc
# Re: -(kx nx + ky ny) sin(kx x + ky y)
# Im:  (kx nx + ky ny) cos(kx x + ky y)
def du_inc_dn_re(xy, nxy):
    phase = kx * xy[:, 0:1] + ky * xy[:, 1:2]
    kn    = kx * nxy[:, 0:1] + ky * nxy[:, 1:2]
    return -kn * torch.sin(phase)

def du_inc_dn_im(xy, nxy):
    phase = kx * xy[:, 0:1] + ky * xy[:, 1:2]
    kn    = kx * nxy[:, 0:1] + ky * nxy[:, 1:2]
    return  kn * torch.cos(phase)
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
# Collocation samplers
# ─────────────────────────────────────────────────────────────────────────────
def sample_inside(n):
    """Points strictly inside the cylinder."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 6, 2)
        r2 = (xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2
        pts.append(xy[r2 < (a - 0.005)**2])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)

def sample_outside(n):
    """Points strictly outside the cylinder in [0,1]²."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 4, 2)
        r2 = (xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2
        pts.append(xy[r2 > (a + 0.008)**2])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)

def sample_interface(n):
    """Points on cylinder boundary ∂D with outward normals."""
    theta = np.random.rand(n) * 2 * np.pi
    x  = cx + a * np.cos(theta);  y  = cy + a * np.sin(theta)
    nx = np.cos(theta);            ny = np.sin(theta)
    xy  = torch.tensor(np.stack([x, y], 1).astype(np.float32),
                       device=DEVICE, requires_grad=True)
    nxy = torch.tensor(np.stack([nx, ny], 1).astype(np.float32), device=DEVICE)
    return xy, nxy
'''
def sample_outer(n_per_side):
    """Points on square boundary [0,1]², outward normals — for Sommerfeld ABC."""
    t = np.random.rand(n_per_side)
    zeros = np.zeros(n_per_side); ones = np.ones(n_per_side)
    xy = np.vstack([
        np.stack([t, zeros], 1),   # bottom  (n̂ = -ŷ)
        np.stack([t, ones],  1),   # top     (n̂ = +ŷ)
        np.stack([zeros, t], 1),   # left    (n̂ = -x̂)
        np.stack([ones,  t], 1),   # right   (n̂ = +x̂)
    ])
    nx = np.concatenate([ zeros,  zeros, -ones,  ones])
    ny = np.concatenate([-ones,    ones,  zeros, zeros])
    nxy = np.stack([nx, ny], 1)
    xy_t  = torch.tensor(xy.astype(np.float32),  device=DEVICE, requires_grad=True)
    nxy_t = torch.tensor(nxy.astype(np.float32), device=DEVICE)
    return xy_t, nxy_t
'''
def sample_outer(n_per_side):
    """Points on outer PML boundary [-D_PML, 1+D_PML]², outward normals."""
    lo = -D_PML; hi = 1.0 + D_PML
    t = np.random.rand(n_per_side) * (hi - lo) + lo
    lo_arr = np.full(n_per_side, lo)
    hi_arr = np.full(n_per_side, hi)
    xy = np.vstack([
        np.stack([t,      lo_arr], 1),   # bottom  (n̂ = -ŷ)
        np.stack([t,      hi_arr], 1),   # top     (n̂ = +ŷ)
        np.stack([lo_arr, t     ], 1),   # left    (n̂ = -x̂)
        np.stack([hi_arr, t     ], 1),   # right   (n̂ = +x̂)
    ])
    nx = np.concatenate([np.zeros(n_per_side), np.zeros(n_per_side),
                         -np.ones(n_per_side),  np.ones(n_per_side)])
    ny = np.concatenate([-np.ones(n_per_side),  np.ones(n_per_side),
                          np.zeros(n_per_side), np.zeros(n_per_side)])
    nxy = np.stack([nx, ny], 1)
    xy_t  = torch.tensor(xy.astype(np.float32),  device=DEVICE, requires_grad=True)
    nxy_t = torch.tensor(nxy.astype(np.float32), device=DEVICE)
    return xy_t, nxy_t

def _pml_sigma(coord, lo, hi):
    """
    Polynomial PML conductivity profile.
    Returns absolute conductivity σ (same units as k).
    Zero inside [lo, hi], rises as (depth/d_pml)^PML_POW outside.
    """
    s = np.zeros_like(coord, dtype=float)
    mask_lo = coord < lo
    mask_hi = coord > hi
    s[mask_lo] = PML_SIGMA_MAX * ((lo - coord[mask_lo]) / D_PML) ** PML_POW
    s[mask_hi] = PML_SIGMA_MAX * ((coord[mask_hi] - hi) / D_PML) ** PML_POW
    return s

def _pml_sigma_deriv(coord, lo, hi):
    """
    Derivative of the polynomial PML conductivity profile.
    ds/dx = (m * sigma_max / d_pml) * (depth/d_pml)^(m-1)
    Sign: positive when penetrating from either side.
    """
    s = np.zeros_like(coord, dtype=float)
    mask_lo = coord < lo
    mask_hi = coord > hi
    if PML_POW >= 1:
        s[mask_lo] = (PML_POW * PML_SIGMA_MAX / D_PML) * ((lo - coord[mask_lo]) / D_PML) ** (PML_POW - 1)
        s[mask_hi] = (PML_POW * PML_SIGMA_MAX / D_PML) * ((coord[mask_hi] - hi) / D_PML) ** (PML_POW - 1)
    return s

def sample_pml(n):
    """
    Points in the PML annular band [-D_PML, 1+D_PML]^2 outside [0,1]^2.
    Returns xy (requires_grad=True) plus the UPML stretching tensors
    (sx_re, sx_im, sy_re, sy_im) precomputed from the FDM-matched profile.

    s_x(x) = 1 - i σ_x(x) / k,  s_y(y) = 1 - i σ_y(y) / k
    """
    pts = []
    while len(pts) == 0 or sum(len(p) for p in pts) < n:
        xy = np.random.uniform(-D_PML, 1.0 + D_PML, (n * 4, 2))
        in_phys = ((xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0) &
                   (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0))
        pts.append(xy[~in_phys])
    xy = np.vstack(pts)[:n]

    # Complex stretching — s = 1 + i σ/k  (σ in same units as k)
    sx = 1.0 + 1j * _pml_sigma(xy[:, 0], 0.0, 1.0) / k
    sy = 1.0 + 1j * _pml_sigma(xy[:, 1], 0.0, 1.0) / k

    xy_t    = torch.tensor(xy.astype(np.float32),        device=DEVICE, requires_grad=True)
    sx_re_t = torch.tensor(sx.real.astype(np.float32),   device=DEVICE).unsqueeze(1)
    sx_im_t = torch.tensor(sx.imag.astype(np.float32),   device=DEVICE).unsqueeze(1)
    sy_re_t = torch.tensor(sy.real.astype(np.float32),   device=DEVICE).unsqueeze(1)
    sy_im_t = torch.tensor(sy.imag.astype(np.float32),   device=DEVICE).unsqueeze(1)
    dsx_im_t = torch.tensor((_pml_sigma_deriv(xy[:, 0], 0.0, 1.0) / k).astype(np.float32), device=DEVICE).unsqueeze(1)
    dsy_im_t = torch.tensor((_pml_sigma_deriv(xy[:, 1], 0.0, 1.0) / k).astype(np.float32), device=DEVICE).unsqueeze(1)
    #dsx_im_t = torch.tensor(np.zeros_like(xy[:, 0]).astype(np.float32), device=DEVICE).unsqueeze(1)
    #dsy_im_t = torch.tensor(np.zeros_like(xy[:, 1]).astype(np.float32), device=DEVICE).unsqueeze(1)
    return xy_t, sx_re_t, sx_im_t, sy_re_t, sy_im_t, dsx_im_t, dsy_im_t

def sample_pml_outer(n_per_side):
    """Points on the outer PML boundary — Dirichlet u_scat = 0."""
    t = np.random.rand(n_per_side)
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

'''
xy_test, *_ = sample_pml(1000)
xy_np = xy_test.detach().cpu().numpy()
sigma_x = _pml_sigma(xy_np[:, 0], 0.0, 1.0)
sigma_y = _pml_sigma(xy_np[:, 1], 0.0, 1.0)
print("xy range:", xy_np.min(axis=0), xy_np.max(axis=0))
print("sigma_x min/max:", sigma_x.min(), sigma_x.max())
print("sigma_y min/max:", sigma_y.min(), sigma_y.max())
print("nonzero sigma_x:", (sigma_x > 0).sum(), "out of", len(sigma_x))
print("nonzero sigma_y:", (sigma_y > 0).sum(), "out of", len(sigma_y))
'''
# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────
def loss_fn(xy_in, xy_out, xy_if, n_if, xy_pml, xy_ob, n_ob, xy_dirichlet):

    # ── 1. Interior PDE: (∇² + k_int²) u_int = 0
    # k_int² is complex: splits into coupled real/imag equations
    # Re: ∇²uR + kR² uR - kI² uI = 0
    # Im: ∇²uI + kR² uI + kI² uR = 0
    iR, iI = net_int(xy_in)
    lapR_in = laplacian(iR, xy_in)
    lapI_in = laplacian(iI, xy_in)
    res_in_R = lapR_in + k_int_sq_re * iR - k_int_sq_im * iI
    res_in_I = lapI_in + k_int_sq_re * iI + k_int_sq_im * iR
    l_pde_int = (res_in_R**2 + res_in_I**2).mean()

    # ── 2. Exterior PDE: (∇² + k²) u_scat = 0  (k² real)
    sR, sI = net_scat(xy_out)
    lapR_ex = laplacian(sR, xy_out)
    lapI_ex = laplacian(sI, xy_out)
    res_ex_R = lapR_ex + k_ext_sq * sR
    res_ex_I = lapI_ex + k_ext_sq * sI
    l_pde_ext = (res_ex_R**2 + res_ex_I**2).mean()

    # ── 3. Interface: Ez continuity  u_int = u_inc + u_scat  on ∂D
    iR_if, iI_if   = net_int(xy_if)
    sR_if, sI_if   = net_scat(xy_if)
    inc_R_if        = u_inc_re(xy_if)
    inc_I_if        = u_inc_im(xy_if)
    l_field = ((iR_if - (sR_if + inc_R_if))**2 +
               (iI_if - (sI_if + inc_I_if))**2).mean()

    # ── 4. Interface: flux continuity  ∂u_int/∂n = ∂u_scat/∂n + ∂u_inc/∂n  on ∂D
    dn_iR = normal_deriv(iR_if, xy_if, n_if)
    dn_iI = normal_deriv(iI_if, xy_if, n_if)
    dn_sR = normal_deriv(sR_if, xy_if, n_if)
    dn_sI = normal_deriv(sI_if, xy_if, n_if)
    dn_incR = du_inc_dn_re(xy_if, n_if)
    dn_incI = du_inc_dn_im(xy_if, n_if)
    l_flux = ((dn_iR - (dn_sR + dn_incR))**2 +
              (dn_iI - (dn_sI + dn_incI))**2).mean()

    # ── 5. PML loss: UPML-stretched Helmholtz for u_scat in the PML band
    xy_p, sx_re, sx_im, sy_re, sy_im, dsx_im, dsy_im = xy_pml

    pR, pI = net_scat(xy_p)

    gR   = torch.autograd.grad(pR.sum(), xy_p, create_graph=True)[0]
    pRx  = gR[:, 0:1]
    pRy  = gR[:, 1:2]
    pRxx = torch.autograd.grad(gR[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
    pRyy = torch.autograd.grad(gR[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

    gI   = torch.autograd.grad(pI.sum(), xy_p, create_graph=True)[0]
    pIx  = gI[:, 0:1]
    pIy  = gI[:, 1:2]
    pIxx = torch.autograd.grad(gI[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
    pIyy = torch.autograd.grad(gI[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

    # Cx = s_y / s_x,  Cy = s_x / s_y
    sx_sq = sx_re**2 + sx_im**2
    sy_sq = sy_re**2 + sy_im**2

    Cx_re = (sy_re * sx_re + sy_im * sx_im) / sx_sq
    Cx_im = (sy_im * sx_re - sy_re * sx_im) / sx_sq

    Cy_re = (sx_re * sy_re + sx_im * sy_im) / sy_sq
    Cy_im = (sx_im * sy_re - sx_re * sy_im) / sy_sq

    # Cx' = -s_y * s_x' / s_x²,  s_x' = i*dsx_im  (purely imaginary)
    # numerator: -s_y * i*dsx_im = (-i*dsx_im)(sy_re + i*sy_im)
    #          = (dsx_im*sy_im) + i(-dsx_im*sy_re)   [note: -(i*a)(b+ic) = ac - iab]
    sx2_re = sx_re**2 - sx_im**2
    sx2_im = 2.0 * sx_re * sx_im
    sx2_sq = sx2_re**2 + sx2_im**2

    sy2_re = sy_re**2 - sy_im**2
    sy2_im = 2.0 * sy_re * sy_im
    sy2_sq = sy2_re**2 + sy2_im**2

    num_Cxp_re =  dsx_im * sy_im
    num_Cxp_im = -dsx_im * sy_re
    Cxp_re = (num_Cxp_re * sx2_re + num_Cxp_im * sx2_im) / sx2_sq
    Cxp_im = (num_Cxp_im * sx2_re - num_Cxp_re * sx2_im) / sx2_sq

    num_Cyp_re =  dsy_im * sx_im
    num_Cyp_im = -dsy_im * sx_re
    Cyp_re = (num_Cyp_re * sy2_re + num_Cyp_im * sy2_im) / sy2_sq
    Cyp_im = (num_Cyp_im * sy2_re - num_Cyp_re * sy2_im) / sy2_sq

    # s_x * s_y
    sxsy_re = sx_re * sy_re - sx_im * sy_im
    sxsy_im = sx_re * sy_im + sx_im * sy_re

    res_pml_R = (Cx_re * pRxx - Cx_im * pIxx
               + Cxp_re * pRx - Cxp_im * pIx
               + Cy_re * pRyy - Cy_im * pIyy
               + Cyp_re * pRy - Cyp_im * pIy
               + k_ext_sq * (sxsy_re * pR - sxsy_im * pI))
    res_pml_I = (Cx_re * pIxx + Cx_im * pRxx
               + Cxp_re * pIx + Cxp_im * pRx
               + Cy_re * pIyy + Cy_im * pRyy
               + Cyp_re * pIy + Cyp_im * pRy
               + k_ext_sq * (sxsy_re * pI + sxsy_im * pR))
    l_abc = (res_pml_R**2 + res_pml_I**2).mean()

    # ── 6. Sommerfeld ABC on square outer boundary — active, breaks ±k degeneracy
    #   ∂u_scat/∂n - ik u_scat = 0
    #   Re: ∂uR/∂n + k uI = 0
    #   Im: ∂uI/∂n - k uR = 0
    oR, oI = net_scat(xy_ob)
    dn_oR  = normal_deriv(oR, xy_ob, n_ob)
    dn_oI  = normal_deriv(oI, xy_ob, n_ob)
    l_sommerfeld = ((dn_oR + k * oI)**2 + (dn_oI - k * oR)**2).mean()

    # ── 7. Dirichlet boundary on outer PML boundary — u_scat = 0
    dR, dI = net_scat(xy_dirichlet)
    l_dirichlet = (dR**2 + dI**2).mean()

    total = (W_PDE_INT * l_pde_int   / L0_PDE_INT +
             W_PDE_EXT * l_pde_ext   / L0_PDE_EXT +
             W_FIELD   * l_field     / L0_FIELD   +
             W_FLUX    * l_flux      / L0_FLUX    +
             W_ABC     * l_sommerfeld / L0_ABC     +
             W_PML     * l_abc       / L0_PML     +
             W_DIRICHLET * l_dirichlet / L0_DIRICHLET)

    # ── 7. FDM data loss — penalise deviation from reference solution
    l_data_scat = torch.tensor(0.0, device=DEVICE)
    l_data_int  = torch.tensor(0.0, device=DEVICE)
    if USE_FDM_DATA:
        # Scattered field at exterior points
        fdm_sR = interp_fdm(FDM_Scat_re, xy_out)
        fdm_sI = interp_fdm(FDM_Scat_im, xy_out)
        l_data_scat = ((sR - fdm_sR)**2 + (sI - fdm_sI)**2).mean()

        # Interior field at interior points
        # FDM u_int = u_total - u_inc  (net_int learns total field inside)
        fdm_iR = interp_fdm(FDM_Total_re, xy_in)
        fdm_iI = interp_fdm(FDM_Total_im, xy_in)
        l_data_int = ((iR - fdm_iR)**2 + (iI - fdm_iI)**2).mean()

        total = total + W_DATA_SCAT * l_data_scat / L0_DATA_SCAT \
                      + W_DATA_INT  * l_data_int  / L0_DATA_INT

    return total, l_pde_int, l_pde_ext, l_field, l_flux, l_sommerfeld, l_abc, l_dirichlet, l_data_scat, l_data_int

# ─────────────────────────────────────────────────────────────────────────────
# Optimiser — trains both networks simultaneously
# ─────────────────────────────────────────────────────────────────────────────
all_params = list(net_int.parameters()) + list(net_scat.parameters())
optimizer  = torch.optim.Adam(all_params, lr=LR_ADAM)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=ADAM_EPOCHS, eta_min=1e-5)

gif_frames = []   # will hold PNG bytes for each frame

# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate networks and render the 6-panel field plot into a frame
# ─────────────────────────────────────────────────────────────────────────────
N_grid_gif = 200   # lower resolution for speed during training
# figsize chosen so that width*dpi and height*dpi are both divisible by 16
# (avoids ffmpeg macro_block_size resize warning).
# At dpi=90: 912×816 px  →  figsize ≈ (10.13, 9.07)
_FW, _FH = 912 / 90, 816 / 90

def save_frame(label, total_loss):
    """Evaluate both networks on a coarse grid and capture a 4-panel plot
    showing only the scattered and total fields (no incident field)."""
    net_int.eval(); net_scat.eval()
    with torch.no_grad():
        xg = np.linspace(0, 1, N_grid_gif)
        yg = np.linspace(0, 1, N_grid_gif)
        Xg, Yg = np.meshgrid(xg, yg)
        R2      = (Xg - cx)**2 + (Yg - cy)**2
        in_cyl  = R2 <= a**2
        xy_ev   = torch.tensor(
            np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32),
            device=DEVICE)
        iR_g, iI_g = net_int(xy_ev)
        sR_g, sI_g = net_scat(xy_ev)

    U_int_f  = (iR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * iI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))
    U_scat_f = (sR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * sI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))
    U_inc_f   = np.exp(1j * (kx * Xg + ky * Yg))
    U_total_f = np.where(in_cyl, U_int_f, U_inc_f + U_scat_f)

    matplotlib.rcParams.update({'font.size': 9})
    fig_f, axes_f = plt.subplots(2, 2, figsize=(_FW, _FH))
    # rect top=0.96 reserves 4 % headroom so suptitle is never clipped
    fig_f.suptitle(f"{label}   |   Loss = {total_loss:.3e}", fontsize=13)
    fig_f.tight_layout(rect=[0, 0, 1, 0.96])

    circle_kw_f = dict(color='cyan', lw=1.5, fill=False, zorder=5, ls='--')

    def _panel(ax, data, title, cmap, symmetric):
        valid = data[np.isfinite(data)]
        scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
        scale = max(scale, 1e-12)
        vmin  = -scale if symmetric else 0.0
        im    = ax.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                          cmap=cmap, vmin=vmin, vmax=scale,
                          interpolation='bilinear')
        ax.add_patch(plt.Circle((cx, cy), a, **circle_kw_f))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig_f.colorbar(im, ax=ax, shrink=0.90)

    _panel(axes_f[0, 0], U_scat_f.real,     "Scattered   Re(u_scat)",  "RdBu_r", True)
    _panel(axes_f[0, 1], U_total_f.real,    "Total       Re(u_total)", "RdBu_r", True)
    _panel(axes_f[1, 0], np.abs(U_scat_f),  "Scattered   |u_scat|",   "hot",     False)
    _panel(axes_f[1, 1], np.abs(U_total_f), "Total       |u_total|",  "inferno", False)

    buf = BytesIO()
    fig_f.savefig(buf, format='png', dpi=90)   # fixed canvas — no bbox_inches='tight'
    plt.close(fig_f)
    buf.seek(0)
    gif_frames.append(imageio.imread(buf))
    net_int.train(); net_scat.train()

loss_history = []
hist_pde_int  = []
hist_pde_ext  = []
hist_field    = []
hist_flux     = []
hist_abc      = []
hist_dirichlet = []
hist_data_scat = []
hist_data_int  = []
hist_alpha    = []   # track blending weights over training
hist_beta     = []
t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Loss normalisation — compute each term's initial value on a fresh batch and
# store as L0_* globals. loss_fn divides every term by its L0 so all terms
# start at 1 and the weights express genuine relative importance rather than
# compensating for differences in natural magnitude.
# Computed once at initialisation and never updated — changing L0 during
# training would shift the loss landscape discontinuously.
# ─────────────────────────────────────────────────────────────────────────────
print("Computing loss normalisation constants ...")

# Initialise L0 globals to 1 so loss_fn is callable before they are set
L0_PDE_INT = L0_PDE_EXT = L0_FIELD = L0_FLUX = L0_ABC = L0_PML = L0_SRC = 1.0
L0_DATA_SCAT = L0_DATA_INT = L0_DIRICHLET = 1.0

# Sample a fresh batch — identical sizes to training batches
_xy_in_n                  = sample_inside(N_INT)
_xy_out_n                 = sample_outside(N_EXT)
_xy_if_n, _nif_n          = sample_interface(N_IFACE)
_xy_pml_n                 = sample_pml(N_PML)
_xy_ob_n, _nob_n          = sample_outer(N_OUT // 4)
_xy_dirichlet_n            = sample_pml_outer(N_PML)

_, _lpi, _lpe, _lf, _lfl, _labc, _la, _ldir, _lds, _ldi = loss_fn(
    _xy_in_n, _xy_out_n, _xy_if_n, _nif_n, _xy_pml_n, _xy_ob_n, _nob_n, _xy_dirichlet_n)

L0_PDE_INT = max(_lpi.item(),  1e-12)
L0_PDE_EXT = max(_lpe.item(),  1e-12)
L0_FIELD   = max(_lf.item(),   1e-12)
L0_FLUX    = max(_lfl.item(),  1e-12)
L0_ABC     = max(_labc.item(), 1e-12)
L0_PML     = max(_la.item(),   1e-12)
L0_DIRICHLET = max(_ldir.item(), 1e-12)
if USE_FDM_DATA:
    L0_DATA_SCAT = max(_lds.item(), 1e-12)
    L0_DATA_INT  = max(_ldi.item(), 1e-12)

print(f"  L0_PDE_INT = {L0_PDE_INT:.3e}")
print(f"  L0_PDE_EXT = {L0_PDE_EXT:.3e}")
print(f"  L0_FIELD   = {L0_FIELD:.3e}")
print(f"  L0_FLUX    = {L0_FLUX:.3e}")
print(f"  L0_ABC     = {L0_ABC:.3e}")
print(f"  L0_PML     = {L0_PML:.3e}")
print(f"  L0_DIRICHLET = {L0_DIRICHLET:.3e}")
if USE_FDM_DATA:
    print(f"  L0_DATA_SCAT= {L0_DATA_SCAT:.3e}")
    print(f"  L0_DATA_INT = {L0_DATA_INT:.3e}")

print(f"\n--- Phase 1: Adam ({ADAM_EPOCHS} epochs) ---")
if USE_BLEND:
    print(f"  Blend: starts radial (alpha=1, beta=0), transitions to full-coord (alpha=0, beta=1)")
    print(f"  Sigmoid spans full training ({TOTAL_STEPS} steps), centre at step "
          f"{int(BLEND_EPOCH_CENTRE * TOTAL_STEPS)}, sharpness={BLEND_SHARPNESS}")
else:
    print(f"  Blend disabled: alpha=0, beta=1 (pure full-coordinate NN2) throughout")
print(f"{'Ep':>5} | {'Total':>8} | {'PDEin':>8} | {'PDEex':>8} | "
      f"{'Field':>8} | {'Flux':>8} | {'Somm':>8} | {'PML':>9} | {'Dirichlet':>10} | "
      f"{'alpha':>6} | {'s':>5}")
print("-" * 110)

hist_somm = []   # Sommerfeld loss history
hist_pml  = []   # PML loss history (diagnostic)

for ep in range(1, ADAM_EPOCHS + 1):
    # ── Update blending weights ──────────────────────────────────────────────
    alpha, beta = blend_schedule(ep - 1)   # global step index: 0 … ADAM_EPOCHS-1
    net_scat.set_blend(alpha, beta)
    hist_alpha.append(alpha)
    hist_beta.append(beta)

    if ep == 1 or ep % 250 == 0:
        xy_in           = sample_inside(N_INT)
        xy_out          = sample_outside(N_EXT)
        xy_if, n_if     = sample_interface(N_IFACE)
        xy_pml_data     = sample_pml(N_PML)
        xy_ob, n_ob     = sample_outer(N_OUT // 4)
        xy_dirichlet    = sample_pml_outer(N_PML)

    optimizer.zero_grad()
    loss, lpi, lpe, lf, lfl, labc, lpml, lds, ldi, ldir = loss_fn(
        xy_in, xy_out, xy_if, n_if, xy_pml_data, xy_ob, n_ob, xy_dirichlet)
    xy_p, sx_re, sx_im, sy_re, sy_im, dsx_im, dsy_im = xy_pml_data
    #print("sx_im sign check:", sx_im.min().item(), sx_im.max().item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())
    hist_pde_int.append(lpi.item())
    hist_pde_ext.append(lpe.item())
    hist_field.append(lf.item())
    hist_flux.append(lfl.item())
    hist_somm.append(labc.item())
    hist_pml.append(lpml.item())
    hist_abc.append(labc.item())
    hist_dirichlet.append(ldir.item())
    hist_data_scat.append(lds.item())
    hist_data_int.append(ldi.item())

    if ep % 250 == 0 or ep == 1:
        print(f"{ep:>5} | {loss.item():>8.2e} | {lpi.item():>8.2e} | {lpe.item():>8.2e} | "
              f"{lf.item():>8.2e} | {lfl.item():>8.2e} | {labc.item():>8.2e} | {lpml.item():>9.2e} | "
              f"{ldir.item():>10.2e} | {alpha:>6.3f} | {time.time()-t0:>5.1f}")

    # ── Capture frame every 100 epochs (and at epoch 1) ─────────────────────
    if ep == 1 or ep % 100 == 0:
        save_frame(f"Adam epoch {ep:>5}", loss.item())

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: L-BFGS — blend locked to full-coordinate mode (alpha=0, beta=1)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Phase 2: L-BFGS ({LBFGS_STEPS} steps) ---")
print(f"{'Step':>5} | {'Total':>8} | {'PDEin':>8} | {'PDEex':>8} | "
      f"{'Field':>8} | {'Flux':>8} | {'Somm':>8} | {'PML':>9} | {'Dirichlet':>10} | "
      f"{'alpha':>6} | {'s':>5}")
print("-" * 110)

RESAMPLE_EVERY = 300   # reinitialise optimizer and resample every N outer steps

def make_closure(xy_in, xy_out, xy_if, nif, xy_pml, xy_ob, nob, xy_dirichlet):
    def closure():
        lbfgs.zero_grad()
        # update blend: global step = ADAM_EPOCHS + current L-BFGS step
        _alpha, _beta = blend_schedule(ADAM_EPOCHS + step_count[0])
        net_scat.set_blend(_alpha, _beta)
        loss, lpi, lpe, lf, lfl, labc, lpml, lds, ldi, ldir = loss_fn(
            xy_in, xy_out, xy_if, nif, xy_pml, xy_ob, nob, xy_dirichlet)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        step_count[0] += 1
        loss_history.append(loss.item())
        hist_pde_int.append(lpi.item())
        hist_pde_ext.append(lpe.item())
        hist_field.append(lf.item())
        hist_flux.append(lfl.item())
        hist_somm.append(labc.item())
        hist_pml.append(lpml.item())
        hist_abc.append(labc.item())
        hist_dirichlet.append(ldir.item())
        hist_data_scat.append(lds.item())
        hist_data_int.append(ldi.item())
        if step_count[0] % 250 == 0:
            print(f"{step_count[0]:>5} | {loss.item():>8.2e} | {lpi.item():>8.2e} | {lpe.item():>8.2e} | "
                  f"{lf.item():>8.2e} | {lfl.item():>8.2e} | {labc.item():>8.2e} | {lpml.item():>9.2e} | "
                  f"{ldir.item():>10.2e} | {_alpha:>6.3f} | {time.time()-t0:>5.1f}")
        if step_count[0] == 1 or step_count[0] % 100 == 0:
            save_frame(f"L-BFGS {step_count[0]}", loss.item())
        return loss
    return closure

step_count = [0]
lbfgs      = None   # initialised on first resample below

for outer in range(LBFGS_STEPS):
    if step_count[0] >= LBFGS_STEPS:
        break

    # resample and reinitialise optimizer every RESAMPLE_EVERY outer steps
    if outer % RESAMPLE_EVERY == 0:
        xy_in_lb         = sample_inside(N_INT)
        xy_out_lb        = sample_outside(N_EXT)
        xy_if_lb, nif_lb = sample_interface(N_IFACE)
        xy_pml_lb        = sample_pml(N_PML)
        xy_ob_lb, nob_lb = sample_outer(N_OUT // 4)
        xy_dirichlet_lb   = sample_pml_outer(N_PML)
        lbfgs = torch.optim.LBFGS(all_params, lr=0.05, max_iter=20,
                                   history_size=20, line_search_fn='strong_wolfe')

    closure = make_closure(xy_in_lb, xy_out_lb, xy_if_lb, nif_lb, xy_pml_lb, xy_ob_lb, nob_lb, xy_dirichlet_lb)
    lbfgs.step(closure)

print(f"\nTotal training time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on dense grid
# ─────────────────────────────────────────────────────────────────────────────
N_grid = 300
xg = np.linspace(0, 1, N_grid)
yg = np.linspace(0, 1, N_grid)
Xg, Yg = np.meshgrid(xg, yg)
R2     = (Xg - cx)**2 + (Yg - cy)**2
in_cyl = R2 <= a**2

xy_ev = torch.tensor(
    np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32), device=DEVICE)

net_int.eval(); net_scat.eval()
with torch.no_grad():
    iR_g, iI_g = net_int(xy_ev)
    sR_g, sI_g = net_scat(xy_ev)

U_int  = (iR_g.cpu().numpy().reshape(N_grid, N_grid) +
          1j * iI_g.cpu().numpy().reshape(N_grid, N_grid))
U_scat = (sR_g.cpu().numpy().reshape(N_grid, N_grid) +
          1j * sI_g.cpu().numpy().reshape(N_grid, N_grid))
U_inc  = np.exp(1j * (kx * Xg + ky * Yg))

# Composite total field: interior from net_int, exterior from u_inc + u_scat
U_total = np.where(in_cyl, U_int, U_inc + U_scat)

print(f"\nField check:")
print(f"  Max |U_total|  = {np.max(np.abs(U_total)):.4f}")
print(f"  Inside mean|u| = {np.mean(np.abs(U_int[in_cyl])):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

r_line = np.linspace(a + 0.01, 0.98, 200)
xy_line = torch.tensor(np.stack([np.full_like(r_line, 0.5),
                                  0.5 + r_line], axis=1).astype(np.float32),
                        device=DEVICE, requires_grad=False)
with torch.no_grad():
    sR_line, sI_line = net_scat(xy_line)
amp_line = (sR_line**2 + sI_line**2).sqrt().cpu().numpy()
plt.plot(r_line, amp_line)
plt.xlabel('r from centre'); plt.ylabel('|u_scat|')
plt.title('Radial amplitude profile')
plt.savefig('radial_profile.png')

matplotlib.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

circle_kw = dict(color='cyan', lw=1.5, fill=False, zorder=5, ls='--')

def panel(ax, data, title, cmap, symmetric):
    valid = data[np.isfinite(data)]
    scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
    scale = max(scale, 1e-12)
    vmin  = -scale if symmetric else 0.0
    im    = ax.imshow(data, origin='lower', extent=[0,1,0,1],
                      cmap=cmap, vmin=vmin, vmax=scale, interpolation='bilinear')
    ax.add_patch(plt.Circle((cx,cy), a, **circle_kw))
    ax.set_title(title, fontsize=16); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.95)

panel(axes[0,0], U_inc.real,      "Incident    Re(u_inc)",   "RdBu_r", True)
panel(axes[0,1], U_scat.real,     "Scattered   Re(u_scat)",  "RdBu_r", True)
panel(axes[0,2], U_total.real,    "Total       Re(u_total)", "RdBu_r", True)
panel(axes[1,0], np.abs(U_inc),   "Incident    |u_inc|",     "viridis", False)
panel(axes[1,1], np.abs(U_scat),  "Scattered   |u_scat|",    "hot",    False)
panel(axes[1,2], np.abs(U_total), "Total       |u_total|",   "inferno", False)

plt.tight_layout()
os.makedirs("figures/mie_2d", exist_ok=True)
plt.savefig("figures/mie_2d/dielectric_pinn_fields.png", dpi=150, bbox_inches='tight')
print("Saved: figures/mie_2d/dielectric_pinn_fields.png")

# Training loss
fig2, ax2 = plt.subplots(figsize=(11, 5))
steps = range(len(loss_history))
ax2.semilogy(steps, loss_history,    color='black',      lw=1.5, label='Total')
ax2.semilogy(steps, hist_pde_int,    color='royalblue',  lw=0.8, label='PDE int')
ax2.semilogy(steps, hist_pde_ext,    color='steelblue',  lw=0.8, label='PDE ext')
ax2.semilogy(steps, hist_field,      color='tomato',     lw=0.8, label='Field BC')
ax2.semilogy(steps, hist_flux,       color='orange',     lw=0.8, label='Flux BC')
ax2.semilogy(steps, hist_somm,       color='mediumseagreen', lw=1.2, label='Sommerfeld (active, W_ABC)')
ax2.semilogy(steps, hist_pml,        color='mediumpurple',   lw=1.2, ls='--',
             label='PML (diagnostic, W_PML=0)')
ax2.axvline(ADAM_EPOCHS, color='gray', ls='--', lw=1.5,
            label=f'Adam → L-BFGS (step {ADAM_EPOCHS})')
ax2.set_xlabel("Optimiser step")
ax2.set_ylabel("Loss (log scale)")
ax2.set_title("Dielectric Cylinder PINN — Training Loss by Term")
ax2.legend(fontsize=8, ncol=2, loc='upper right')
ax2.grid(True, alpha=0.3)

# Overlay blend schedule on a twin y-axis (Adam phase only)
#ax2b = ax2.twinx()
#adam_steps = range(ADAM_EPOCHS)
#ax2b.plot(adam_steps, hist_alpha, color='darkorange', lw=1.5, ls=':', label='alpha (radial)')
#ax2b.plot(adam_steps, hist_beta,  color='dodgerblue', lw=1.5, ls=':', label='beta  (full-xy)')
#ax2b.set_ylabel("Blend weight (alpha / beta)", color='gray')
#ax2b.set_ylim(-0.05, 1.25)
#ax2b.tick_params(axis='y', labelcolor='gray')
#ax2b.legend(fontsize=8, loc='center right')

plt.tight_layout()
plt.savefig(f"figures/mie_2d/dielectric_pinn_loss.png", dpi=150, bbox_inches='tight')
print(f"Saved: figures/mie_2d/dielectric_pinn_loss.png")

# plt.show()
print("\nDone.")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble MP4 from captured frames  (plays natively in PowerPoint)
# ─────────────────────────────────────────────────────────────────────────────
if gif_frames:
    mp4_path = "figures/mie_2d/training_animation.mp4"
    # ~6 fps gives a readable pace; last frame held for 3 s (≈18 duplicate frames)
    fps = 6
    hold_frames = fps * 3          # repeat the final frame for 3 seconds
    frames_out  = gif_frames + [gif_frames[-1]] * hold_frames
    writer_kw   = dict(fps=fps, codec="libx264", pixelformat="yuv420p",
                       output_params=["-crf", "18"])   # high quality, PowerPoint-compatible
    imageio.mimwrite(mp4_path, frames_out, format="mp4", **writer_kw)
    print(f"Saved training animation ({len(gif_frames)} frames + {hold_frames} hold): {mp4_path}")