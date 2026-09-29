"""
PINN — 2D Dielectric AFM-Tip Scattering (TM mode)
================================================================
Plane wave scattering by a conospherical AFM tip descending from the top
of the domain, with a tunable complex refractive index. The scatterer is
no longer a circle: it is defined implicitly by a signed distance function
phi(x,y) (negative inside the tip, zero on the surface, positive outside).

  net_int  : learns u_int(x,y)   — total field  INSIDE  the tip  (phi < 0)
  net_scat : learns u_scat(x,y)  — scattered field OUTSIDE the tip (phi > 0)

GEOMETRY — conospherical AFM tip
--------------------------------
The tip is modelled as a cone of half-angle HALF_ANGLE whose apex points
DOWNWARD, capped at the apex by a sphere (circle in 2D) of radius R_APEX.
The cone axis is the vertical line x = TIP_X. The apex sits at height
APEX_Y; the shank widens upward and runs straight out through the TOP of
the domain (and through the top PML). Because the shank is invariant in y,
a y-PML terminates it without spurious reflection.

phi(x,y) < 0  inside the tip
phi(x,y) = 0  on the tip surface
phi(x,y) > 0  outside the tip
Outward normal on the surface:  n_hat = ∇phi / |∇phi|  (autograd).

A radial-flux diagnostic (net outward time-averaged power through a loop
enclosing the apex) is printed during training to catch any residual
incoming-branch contamination that the loss curve alone would hide.

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


LOSS TERMS
----------
    L_pde_int  : Helmholtz residual for u_int  (inside cylinder)
    L_pde_scat : Helmholtz residual for u_scat (outside cylinder)
    L_field    : u_int = u_inc + u_scat        on ∂D
    L_flux     : ∂u_int/∂n = ∂(u_inc+u_scat)/∂n  on ∂D
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
N_IM   = 30.0    # imaginary part of refractive index
lam    = 0.5    # wavelength 
k      = 2.0 * np.pi / lam   # exterior wavenumber

# ── AFM tip geometry ─────────────────────────────────────────────────────────
# The tip axis is the vertical line x = TIP_X. The rounded apex points DOWN and
# sits at height APEX_Y; the cone opens upward with half-angle HALF_ANGLE and the
# shank runs straight out through the top boundary (y = 1) and the top PML.
TIP_X       = 0.5                 # x-coordinate of the tip axis
APEX_Y      = 0.45                # y-coordinate of the rounded apex
R_APEX      = 0.05                # apex radius of curvature
HALF_ANGLE  = np.deg2rad(20.0)    # cone half-angle

cx, cy = TIP_X, APEX_Y
a      = R_APEX

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

# ── Non-dimensionalization scales for the PDE/PML residuals ───────────────────
k_int_sq_mag = float(np.hypot(k_int_sq_re, k_int_sq_im))   # |k_int²|
k_ext_sq_mag = float(k_ext_sq)                              # |k_ext²| (already real)
print(f"  |k_int²| (interior PDE non-dim scale) = {k_int_sq_mag:.4f}")
print(f"  |k_ext²| (exterior PDE non-dim scale) = {k_ext_sq_mag:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Signed distance function for the conospherical AFM tip
# ─────────────────────────────────────────────────────────────────────────────

_SIN_A = float(np.sin(HALF_ANGLE))
_COS_A = float(np.cos(HALF_ANGLE))
_APEX_CENTRE_Y = APEX_Y + R_APEX                          # apex sphere centre
_CONE_VERTEX_Y = APEX_Y + R_APEX - R_APEX / _SIN_A        # sharp-cone vertex
_TANGENT_Y     = _APEX_CENTRE_Y - R_APEX * _SIN_A         # cap/flank switch height

def phi_np(x, y):
    """
    Signed distance to the conospherical AFM tip (numpy)
    Negative inside the tip, positive outside, 0 on the surface
    x, y : numpy arrays
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q_x = np.abs(x - TIP_X)
    d_cone = q_x * _COS_A - (y - _CONE_VERTEX_Y) * _SIN_A
    d_apex = np.sqrt((x - TIP_X)**2 + (y - _APEX_CENTRE_Y)**2) - R_APEX
    use_apex = y < _TANGENT_Y
    return np.where(use_apex, d_apex, d_cone)

def phi_torch(xy):
    """
    Signed distance to the conospherical AFM tip.
    xy : tensor [N,2]. Returns [N,1]. Used for autograd outward normals.
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    q_x = torch.abs(x - TIP_X)
    d_cone = q_x * _COS_A - (y - _CONE_VERTEX_Y) * _SIN_A
    d_apex = torch.sqrt((x - TIP_X)**2 + (y - _APEX_CENTRE_Y)**2 + 1e-12) - R_APEX
    use_apex = (y < _TANGENT_Y)
    return torch.where(use_apex, d_apex, d_cone)

def outward_normal(xy):
    """
    Outward unit normal on the tip surface via autograd of phi_torch.
    xy : tensor [N,2] with requires_grad=True. Returns [N,2].
    """
    d = phi_torch(xy)
    g = torch.autograd.grad(d.sum(), xy, create_graph=True)[0]
    norm = torch.sqrt((g**2).sum(dim=1, keepdim=True) + 1e-12)
    return g / norm

def tip_mask(Xg, Yg):
    """Boolean mask (phi < 0) marking the tip interior on a meshgrid."""
    return phi_np(Xg, Yg) < 0.0

def tip_outline_xy(y_top=1.0, n=200):
    """
    Return (x, y) polyline tracing the tip surface for plot overlays
    """
    yc = _APEX_CENTRE_Y
    ys = np.linspace(_TANGENT_Y, y_top, n)
    half_w = (ys - _CONE_VERTEX_Y) * np.tan(HALF_ANGLE)
    x_right = TIP_X + half_w
    x_left  = TIP_X - half_w
    a0 = -np.pi/2 - (np.pi/2 - HALF_ANGLE)
    a1 = -np.pi/2 + (np.pi/2 - HALF_ANGLE)
    ang = np.linspace(a0, a1, n)
    ax_arc = TIP_X + R_APEX * np.cos(ang)
    ay_arc = yc     + R_APEX * np.sin(ang)
    xs = np.concatenate([x_right[::-1], ax_arc, x_left])
    yy = np.concatenate([ys[::-1],      ay_arc, ys])
    return xs, yy

# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

N_INT   = 1000   # interior PDE points
N_EXT   = 2000   # exterior PDE points
N_IFACE = 800    # interface boundary ∂D
N_OUT   = 200    # outer square boundary

ADAM_EPOCHS = 2000
LBFGS_STEPS = 30000   # closure-eval budget (not outer steps) -- see Phase 2
LR_ADAM     = 8e-4

# Loss weights 
# Each term is divided by its value at initialisation so every term starts at 1.
W_PDE_INT  = 1.0
W_PDE_EXT  = 1.0
W_FIELD    = 50.0
W_FLUX     = 50.0 
W_PML      = 5.0    # PML term 
W_DIRICHLET = 1.0    # Dirichlet on outer PML boundary 

# ── PML parameters ────────────────────────
D_PML        = 0.5     # PML thickness
PML_POW      = 3       # polynomial grading exponent m
PML_SIGMA_MAX = 20.0  # peak conductivity
N_PML        = 2500     # number of PML collocation points

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

# Weight for FDM data loss
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

def embed_ext(xy):
    proj = xy @ B_ext
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

def embed_int(xy):
    proj = xy @ B_int
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

IN_DIM = 2 * N_FOURIER 

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

net_int  = MLP(embed_int).to(DEVICE)       # total field inside cylinder
net_scat = MLP(embed_ext).to(DEVICE)       # scattered field

n_int  = sum(p.numel() for p in net_int.parameters())
n_scat = sum(p.numel() for p in net_scat.parameters())
print(f"\n  net_int      parameters: {n_int:,}")
print(f"  net_scat     parameters: {n_scat:,}")
print(f"  (net_scat = NN(x,y), full-coordinate scattered field)")

# ─────────────────────────────────────────────────────────────────────────────
# Incident field
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
# Small margins so PDE points sit strictly on one side of the tip surface
_MARGIN_IN  = 0.005
_MARGIN_OUT = 0.008

# The interior points are sampled from a bounding box around the tip for efficiency.
def _tip_bbox():
    y_hi = 1.0 + D_PML
    half_w = (y_hi - _CONE_VERTEX_Y) * np.tan(HALF_ANGLE)
    x_lo = TIP_X - half_w - 0.02
    x_hi = TIP_X + half_w + 0.02
    y_lo = APEX_Y - R_APEX - 0.02
    return x_lo, x_hi, y_lo, y_hi

def sample_inside(n):
    """Points strictly inside the AFM tip (including the part of the shank inside the top PML)."""
    x_lo, x_hi, y_lo, y_hi = _tip_bbox()
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.column_stack([
            np.random.uniform(x_lo, x_hi, n * 8),
            np.random.uniform(y_lo, y_hi, n * 8),
        ])
        d = phi_np(xy[:, 0], xy[:, 1])
        pts.append(xy[d < -_MARGIN_IN])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)

def sample_outside(n):
    """Points strictly outside the tip in the physical box [0,1]²."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 4, 2)
        d = phi_np(xy[:, 0], xy[:, 1])
        pts.append(xy[d > _MARGIN_OUT])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)

def sample_interface(n):
    """
    Points on the tip surface phi = 0 with autograd outward normals.
    Draw points in the tip bounding box, keep those within a thin shell
    |phi| < eps, then project them onto the surface along the SDF gradient.
    Normals come from autograd of phi_torch.
    """
    x_lo, x_hi, y_lo, y_hi = _tip_bbox()
    eps = 0.02
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.column_stack([
            np.random.uniform(x_lo, x_hi, n * 12),
            np.random.uniform(y_lo, y_hi, n * 12),
        ])
        d = phi_np(xy[:, 0], xy[:, 1])
        pts.append(xy[np.abs(d) < eps])
    xy = np.vstack(pts)[:n]

    # Project onto phi = 0 along the gradient, and grab outward normals.
    xy_t = torch.tensor(xy.astype(np.float32), device=DEVICE, requires_grad=True)
    d_t  = phi_torch(xy_t)
    n_t  = outward_normal(xy_t)
    xy_proj = (xy_t - d_t * n_t).detach()

    # Recompute exact outward normals at the projected surface points.
    xy_proj.requires_grad_(True)
    nxy = outward_normal(xy_proj).detach()

    xy_proj = xy_proj.detach().clone().requires_grad_(True)
    return xy_proj, nxy

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

    # Mask: which PML points fall INSIDE the tip material (shank in the top PML).
    # These points must use the interior wavenumber k_int² and be evaluated on
    # net_int (the interior total-field network), not net_scat. Points outside
    # the tip use the exterior k² on net_scat as before.
    d_pml    = phi_np(xy[:, 0], xy[:, 1])
    inside_t = torch.tensor((d_pml < 0.0).astype(np.float32), device=DEVICE).unsqueeze(1)
    return xy_t, sx_re_t, sx_im_t, sy_re_t, sy_im_t, dsx_im_t, dsy_im_t, inside_t

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

# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────
def loss_fn(xy_in, xy_out, xy_if, n_if, xy_pml, xy_dirichlet):

    # ── 1. Interior PDE: (∇² + k_int²) u_int = 0
    # k_int² is complex: splits into coupled real/imag equations
    # Re: ∇²uR + kR² uR - kI² uI = 0
    # Im: ∇²uI + kR² uI + kI² uR = 0
    iR, iI = net_int(xy_in)
    lapR_in = laplacian(iR, xy_in)
    lapI_in = laplacian(iI, xy_in)
    # Divide by |k_int²| so the residual is a RELATIVE equation violation,
    # not one whose absolute scale is inflated ~900x by the material contrast.
    res_in_R = (lapR_in + k_int_sq_re * iR - k_int_sq_im * iI) / k_int_sq_mag
    res_in_I = (lapI_in + k_int_sq_re * iI + k_int_sq_im * iR) / k_int_sq_mag
    l_pde_int = (res_in_R**2 + res_in_I**2).mean()

    # ── 2. Exterior PDE: (∇² + k²) u_scat = 0  (k² real)
    sR, sI = net_scat(xy_out)
    lapR_ex = laplacian(sR, xy_out)
    lapI_ex = laplacian(sI, xy_out)
    # Divide by |k_ext²| — same non-dimensionalization as the interior term,
    # for consistency (k_ext² is O(1)-ish here so this mainly keeps units
    # matched rather than fixing a large gap, since the exterior contrast is mild).
    res_ex_R = (lapR_ex + k_ext_sq * sR) / k_ext_sq_mag
    res_ex_I = (lapI_ex + k_ext_sq * sI) / k_ext_sq_mag
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

    # ── 5. PML loss: UPML-stretched Helmholtz in the PML band.
    #    The stretching tensors s_x, s_y depend only on geometry so they are identical everywhere.
    #    But the reaction wavenumber and the target network differ:
    #      * OUTSIDE the tip:  operator acts on net_scat with k_ext² (real).
    #      * INSIDE  the tip: operator acts on net_int with k_int² (complex).
    xy_p, sx_re, sx_im, sy_re, sy_im, dsx_im, dsy_im, inside_pml = xy_pml

    # Evaluate BOTH networks; select per-point via the inside/outside mask.
    psR, psI = net_scat(xy_p)   # exterior branch
    piR, piI = net_int(xy_p)    # interior branch (used only where inside_pml=1)

    pR = inside_pml * piR + (1.0 - inside_pml) * psR
    pI = inside_pml * piI + (1.0 - inside_pml) * psI

    # Position-dependent reaction wavenumber squared (complex):
    #   outside: (k_ext², 0);  inside: (Re k_int², Im k_int²)
    ksq_re = inside_pml * k_int_sq_re + (1.0 - inside_pml) * k_ext_sq
    ksq_im = inside_pml * k_int_sq_im            # zero outside (real k_ext²)

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

    # Complex reaction coefficient  K = k² * s_x s_y  with k² = ksq_re + i ksq_im
    #   K_re = ksq_re*sxsy_re - ksq_im*sxsy_im
    #   K_im = ksq_re*sxsy_im + ksq_im*sxsy_re
    K_re = ksq_re * sxsy_re - ksq_im * sxsy_im
    K_im = ksq_re * sxsy_im + ksq_im * sxsy_re

    # Non-dimensionalize by the LOCAL |k²| at each point (not a single global
    # constant): points blend between k_int² inside the shank and k_ext²
    # outside it within the same PML band, so each point's residual should be
    # scaled by its own characteristic reaction magnitude.
    ksq_mag = torch.sqrt(ksq_re**2 + ksq_im**2 + 1e-12)

    res_pml_R = (Cx_re * pRxx - Cx_im * pIxx
               + Cxp_re * pRx - Cxp_im * pIx
               + Cy_re * pRyy - Cy_im * pIyy
               + Cyp_re * pRy - Cyp_im * pIy
               + (K_re * pR - K_im * pI)) / ksq_mag
    res_pml_I = (Cx_re * pIxx + Cx_im * pRxx
               + Cxp_re * pIx + Cxp_im * pRx
               + Cy_re * pIyy + Cy_im * pRyy
               + Cyp_re * pIy + Cyp_im * pRy
               + (K_re * pI + K_im * pR)) / ksq_mag
    l_pml = (res_pml_R**2 + res_pml_I**2).mean()

    # ── 6. Dirichlet boundary on outer PML boundary — u_scat = 0
    dR, dI = net_scat(xy_dirichlet)
    l_dirichlet = (dR**2 + dI**2).mean()

    total = (W_PDE_INT * l_pde_int   / L0_PDE_INT +
             W_PDE_EXT * l_pde_ext   / L0_PDE_EXT +
             W_FIELD   * l_field     / L0_FIELD   +
             W_FLUX    * l_flux      / L0_FLUX    +
             W_PML     * l_pml       / L0_PML     +
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

    return total, l_pde_int, l_pde_ext, l_field, l_flux, l_pml, l_dirichlet, l_data_scat, l_data_int

# ─────────────────────────────────────────────────────────────────────────────
# Radial-flux diagnostic — detects the ±k sign degeneracy
# ─────────────────────────────────────────────────────────────────────────────
# The time-averaged radial power flux of the scattered field through a loop
# enclosing the apex should be NET OUTWARD for a correct (outgoing) solution:
#     S_r ∝ Im( conj(u) ∂u/∂r )
# A contaminated / incoming-branch solution shows near-zero or inward net flux
# even when the PDE + PML residuals look converged. This is far more sensitive
# to the sign problem than the loss curve, so we monitor it during training.
_FLUX_R = float(min(APEX_Y - R_APEX - 0.02, 0.5 - 0.02))   # loop radius (in box)
_FLUX_N = 256
def radial_flux_diag():
    theta = np.linspace(0, 2 * np.pi, _FLUX_N, endpoint=False)
    xs = TIP_X + _FLUX_R * np.cos(theta)
    ys = APEX_Y + _FLUX_R * np.sin(theta)
    # Keep only sample points that lie in the exterior AND inside the box.
    good = (phi_np(xs, ys) > 0.0) & (xs > 0) & (xs < 1) & (ys > 0) & (ys < 1)
    xs, ys = xs[good], ys[good]
    nx, ny = np.cos(theta[good]), np.sin(theta[good])   # outward radial normal
    xy = torch.tensor(np.stack([xs, ys], 1).astype(np.float32),
                      device=DEVICE, requires_grad=True)
    nxy = torch.tensor(np.stack([nx, ny], 1).astype(np.float32), device=DEVICE)
    uR, uI = net_scat(xy)
    dnR = normal_deriv(uR, xy, nxy)
    dnI = normal_deriv(uI, xy, nxy)
    # Im( conj(u) du/dn ) = uR*dnI - uI*dnR
    Sr = (uR * dnI - uI * dnR)
    return float(Sr.mean().detach().cpu())

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
_FW, _FH = 912 / 90, 816 / 90

def save_frame(label, total_loss):
    """Evaluate both networks on a coarse grid and capture a 4-panel plot
    showing only the scattered and total fields."""
    net_int.eval(); net_scat.eval()
    with torch.no_grad():
        xg = np.linspace(0, 1, N_grid_gif)
        yg = np.linspace(0, 1, N_grid_gif)
        Xg, Yg = np.meshgrid(xg, yg)
        in_cyl  = tip_mask(Xg, Yg)
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

    # Scattered field (total - incident)
    U_scat_plot_f = U_total_f - U_inc_f

    matplotlib.rcParams.update({'font.size': 9})
    fig_f, axes_f = plt.subplots(2, 2, figsize=(_FW, _FH))
    fig_f.suptitle(f"{label}   |   Loss = {total_loss:.3e}", fontsize=13)
    fig_f.tight_layout(rect=[0, 0, 1, 0.96])

    circle_kw_f = dict(color='cyan', lw=1.5, zorder=5, ls='--')
    _tip_ox, _tip_oy = tip_outline_xy(y_top=1.0)

    def _panel(ax, data, title, cmap, symmetric):
        valid = data[np.isfinite(data)]
        scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
        scale = max(scale, 1e-12)
        vmin  = -scale if symmetric else 0.0
        im    = ax.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                          cmap=cmap, vmin=vmin, vmax=scale,
                          interpolation='bilinear')
        ax.plot(_tip_ox, _tip_oy, **circle_kw_f)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig_f.colorbar(im, ax=ax, shrink=0.90)

    _panel(axes_f[0, 0], U_scat_plot_f.real,     "Scattered   Re(u_scat)",  "RdBu_r", True)
    _panel(axes_f[0, 1], U_total_f.real,    "Total       Re(u_total)", "RdBu_r", True)
    _panel(axes_f[1, 0], np.abs(U_scat_plot_f),  "Scattered   |u_scat|",   "hot",     False)
    _panel(axes_f[1, 1], np.abs(U_total_f), "Total       |u_total|",  "inferno", False)

    buf = BytesIO()
    fig_f.savefig(buf, format='png', dpi=90)
    plt.close(fig_f)
    buf.seek(0)
    gif_frames.append(imageio.imread(buf))
    net_int.train(); net_scat.train()

loss_history   = []
hist_pde_int   = []
hist_pde_ext   = []
hist_field     = []
hist_flux      = []
hist_dirichlet = []
hist_pml       = []
hist_data_scat = []
hist_data_int  = []
t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Loss normalisation — compute each term's initial value and store as L0_*
# globals. loss_fn divides every term by its L0 averaged over K fresh random 
# batches on the untrained network.
# ─────────────────────────────────────────────────────────────────────────────

# Initialise L0 globals to 1 so loss_fn is callable before they are set
L0_PDE_INT = L0_PDE_EXT = L0_FIELD = L0_FLUX = L0_ABC = L0_PML = L0_SRC = 1.0
L0_DATA_SCAT = L0_DATA_INT = L0_DIRICHLET = 1.0

K_L0 = 15   # number of fresh random batches to average over
_accum = {k: 0.0 for k in
          ['lpi', 'lpe', 'lf', 'lfl', 'lpml', 'ldir', 'lds', 'ldi']}

for _ in range(K_L0):
    # Sample a fresh batch — identical sizes to training batches
    _xy_in_n                  = sample_inside(N_INT)
    _xy_out_n                 = sample_outside(N_EXT)
    _xy_if_n, _nif_n          = sample_interface(N_IFACE)
    _xy_pml_n                 = sample_pml(N_PML)
    _xy_dirichlet_n            = sample_pml_outer(N_PML)

    _, _lpi, _lpe, _lf, _lfl, _lpml, _ldir, _lds, _ldi = loss_fn(
        _xy_in_n, _xy_out_n, _xy_if_n, _nif_n, _xy_pml_n, _xy_dirichlet_n)

    _accum['lpi']  += _lpi.item()
    _accum['lpe']  += _lpe.item()
    _accum['lf']   += _lf.item()
    _accum['lfl']  += _lfl.item()
    _accum['lpml'] += _lpml.item()
    _accum['ldir'] += _ldir.item()
    if USE_FDM_DATA:
        _accum['lds'] += _lds.item()
        _accum['ldi'] += _ldi.item()

L0_PDE_INT = max(_accum['lpi']  / K_L0, 1e-12)
L0_PDE_EXT = max(_accum['lpe']  / K_L0, 1e-12)
L0_FIELD   = max(_accum['lf']   / K_L0, 1e-12)
L0_FLUX    = max(_accum['lfl']  / K_L0, 1e-12)
L0_PML     = max(_accum['lpml']   / K_L0, 1e-12)
L0_DIRICHLET = max(_accum['ldir'] / K_L0, 1e-12)
if USE_FDM_DATA:
    L0_DATA_SCAT = max(_accum['lds'] / K_L0, 1e-12)
    L0_DATA_INT  = max(_accum['ldi'] / K_L0, 1e-12)

print(f"  (averaged over {K_L0} batches)")
print(f"  L0_PDE_INT = {L0_PDE_INT:.3e}")
print(f"  L0_PDE_EXT = {L0_PDE_EXT:.3e}")
print(f"  L0_FIELD   = {L0_FIELD:.3e}")
print(f"  L0_FLUX    = {L0_FLUX:.3e}")
print(f"  L0_PML     = {L0_PML:.3e}")
print(f"  L0_DIRICHLET = {L0_DIRICHLET:.3e}")
if USE_FDM_DATA:
    print(f"  L0_DATA_SCAT= {L0_DATA_SCAT:.3e}")
    print(f"  L0_DATA_INT = {L0_DATA_INT:.3e}")

def print_normalized_contributions(step_label, lpi, lpe, lf, lfl, lpml, ldir, lds, ldi):
    """
    Diagnostic: print each term's ACTUAL normalized-and-weighted contribution
    to `total` (W_x * l_x / L0_x), not the raw residual.
    """
    contribs = {
        'PDE int':    W_PDE_INT   * lpi  / L0_PDE_INT,
        'PDE ext':    W_PDE_EXT   * lpe  / L0_PDE_EXT,
        'Field BC':   W_FIELD     * lf   / L0_FIELD,
        'Flux BC':    W_FLUX      * lfl  / L0_FLUX,
        'PML':        W_PML       * lpml   / L0_PML,
        'Dirichlet':  W_DIRICHLET * ldir / L0_DIRICHLET,
    }
    if USE_FDM_DATA:
        contribs['Data scat'] = W_DATA_SCAT * lds / L0_DATA_SCAT
        contribs['Data int']  = W_DATA_INT  * ldi / L0_DATA_INT
    total_check = sum(contribs.values())
    print(f"  [{step_label}] normalized+weighted contributions to total "
          f"(sum={total_check:.3e}):")
    for name, val in contribs.items():
        pct = 100.0 * val / total_check if total_check > 0 else 0.0
        print(f"      {name:<11s} {val:>10.3e}  ({pct:5.1f}%)")

# Print the initial (step-0) breakdown right away
print_normalized_contributions(
    "init", _accum['lpi']/K_L0, _accum['lpe']/K_L0, _accum['lf']/K_L0,
    _accum['lfl']/K_L0, _accum['lpml']/K_L0,
    _accum['ldir']/K_L0, _accum['lds']/K_L0 if USE_FDM_DATA else 0.0,
    _accum['ldi']/K_L0 if USE_FDM_DATA else 0.0)

print(f"\n--- Phase 1: Adam ({ADAM_EPOCHS} epochs) ---")
print(f"{'Ep':>5} | {'Total':>8} | {'PDEin':>8} | {'PDEex':>8} | "
      f"{'Field':>8} | {'Flux':>8} | {'PML':>9} | {'Dirichlet':>10} | "
      f"{'s':>5}")
print("-" * 110)

for ep in range(1, ADAM_EPOCHS + 1):
    if ep == 1 or ep % 25000 == 0:
        xy_in           = sample_inside(N_INT)
        xy_out          = sample_outside(N_EXT)
        xy_if, n_if     = sample_interface(N_IFACE)
        xy_pml_data     = sample_pml(N_PML)
        xy_dirichlet    = sample_pml_outer(N_PML)

    optimizer.zero_grad()
    loss, lpi, lpe, lf, lfl, lpml, ldir, lds, ldi = loss_fn(
        xy_in, xy_out, xy_if, n_if, xy_pml_data, xy_dirichlet)
    xy_p, sx_re, sx_im, sy_re, sy_im, dsx_im, dsy_im, inside_pml = xy_pml_data
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())
    hist_pde_int.append(lpi.item())
    hist_pde_ext.append(lpe.item())
    hist_field.append(lf.item())
    hist_flux.append(lfl.item())
    hist_pml.append(lpml.item())
    hist_dirichlet.append(ldir.item())
    hist_data_scat.append(lds.item())
    hist_data_int.append(ldi.item())

    if ep % 250 == 0 or ep == 1:
        _flux = radial_flux_diag()
        print(f"{ep:>5} | {loss.item():>8.2e} | {lpi.item():>8.2e} | {lpe.item():>8.2e} | "
              f"{lf.item():>8.2e} | {lfl.item():>8.2e} | {lpml.item():>9.2e} | "
              f"{ldir.item():>10.2e} | {time.time()-t0:>5.1f}"
              f"   flux(out>0)={_flux:+.2e}")
        print_normalized_contributions(f"Adam ep {ep}", lpi.item(), lpe.item(),
                                        lf.item(), lfl.item(),
                                        lpml.item(), ldir.item(), lds.item(), ldi.item())

    # ── Capture frame every 100 epochs (and at epoch 1) ─────────────────────
    if ep == 1 or ep % 100 == 0:
        save_frame(f"Adam epoch {ep:>5}", loss.item())

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: L-BFGS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Phase 2: L-BFGS ({LBFGS_STEPS} closure evals) ---")
print(f"{'Eval':>6} | {'Total':>8} | {'PDEin':>8} | {'PDEex':>8} | "
      f"{'Field':>8} | {'Flux':>8} | {'PML':>9} | {'Dirichlet':>10} | "
      f"{'s':>5}")
print("-" * 110)

LBFGS_LR         = 0.05    # L-BFGS learning rate
LBFGS_MAX_ITER   = 24      # inner iterations per optimiser step -- matches
                           # final_deeponet.py's tested config
RESAMPLE_EVERY   = 25      # in OUTER steps, matching final_deeponet.py: at
                           # max_iter=24 that's every ~600 closure evals (the
                           # old "every 100 outer steps" was every ~3000 evals
                           # under the previous max_iter=30 outer-step-bounded
                           # loop -- looser, and more exposed to memorising a
                           # frozen collocation set).

def resample_batch():
    return (sample_inside(N_INT), sample_outside(N_EXT),
            *sample_interface(N_IFACE), sample_pml(N_PML), sample_pml_outer(N_PML))

batch = resample_batch()
last_terms = [None]   # stacked loss terms from the closure's last evaluation

def make_closure(xy_in, xy_out, xy_if, nif, xy_pml, xy_dirichlet):
    # Called MANY times per lbfgs.step() (once per inner iteration plus every
    # rejected trial point during the Wolfe line search) -- must not log or
    # do a second forward pass. Stashes the detached term tensors instead of
    # the loop re-running loss_fn(*batch) after every outer step (which used
    # to burn one full extra evaluation per outer step purely on logging).
    def closure():
        lbfgs.zero_grad()
        terms = loss_fn(xy_in, xy_out, xy_if, nif, xy_pml, xy_dirichlet)
        terms[0].backward()
        step_count[0] += 1
        last_terms[0] = torch.stack([t.detach() for t in terms])
        return terms[0]
    return closure

lbfgs = torch.optim.LBFGS(all_params, lr=LBFGS_LR, max_iter=LBFGS_MAX_ITER,
                          history_size=20, line_search_fn='strong_wolfe',
                          tolerance_grad=1e-12, tolerance_change=1e-16)

step_count = [0]   # closure evaluations -- the real compute unit
outer_step = 0

# Bounded by CLOSURE EVALUATIONS, not outer .step() calls: a
# `for _ in range(LBFGS_STEPS)` form with max_iter=24 would mean up to
# LBFGS_STEPS*24 inner iterations plus strong-Wolfe line-search probes on
# top -- see final_deeponet.py's Phase 2 for the same fix and the measured
# ~3x silent blow-up that motivated it.
while step_count[0] < LBFGS_STEPS:
    if outer_step % RESAMPLE_EVERY == 0:
        batch = resample_batch()

    closure = make_closure(*batch)
    lbfgs.step(closure)
    outer_step += 1

    (loss, lpi, lpe, lf, lfl, lpml, ldir, lds, ldi) = last_terms[0].tolist()

    loss_history.append(loss)
    hist_pde_int.append(lpi)
    hist_pde_ext.append(lpe)
    hist_field.append(lf)
    hist_flux.append(lfl)
    hist_pml.append(lpml)
    hist_dirichlet.append(ldir)
    hist_data_scat.append(lds)
    hist_data_int.append(ldi)

    if outer_step % 25 == 0:
        _flux = radial_flux_diag()
        print(f"{step_count[0]:>6} | {loss:>8.2e} | {lpi:>8.2e} | {lpe:>8.2e} | "
              f"{lf:>8.2e} | {lfl:>8.2e} | {lpml:>9.2e} | "
              f"{ldir:>10.2e} | {time.time()-t0:>5.1f}"
              f"   flux(out>0)={_flux:+.2e}")
        print_normalized_contributions(f"L-BFGS eval {step_count[0]}", lpi, lpe,
                                        lf, lfl, lpml, ldir, lds, ldi)
    if outer_step == 1 or outer_step % 25 == 0:
        save_frame(f"L-BFGS eval {step_count[0]}", loss)

print(f"\nTotal training time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC: spatial heatmaps of the PDE/PML residuals.
# ─────────────────────────────────────────────────────────────────────────────
_RESIDUAL_CHUNK = 4000

def _eval_in_chunks(xy_np, eval_fn, chunk=_RESIDUAL_CHUNK):
    """
    Run eval_fn (which builds its own requires_grad tensor internally and
    returns a [n,1] magnitude tensor) over xy_np in chunks, concatenating
    the detached results. Keeps peak autograd-graph memory bounded regardless
    of how large the requested grid is.
    """
    n = xy_np.shape[0]
    out = np.empty((n, 1), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        out[start:end] = eval_fn(xy_np[start:end])
    return out

def _residual_grid_interior(ng=200):
    """
    Interior PDE residual |res_int| over the tip's bounding box (including
    the part of the shank inside the top PML, matching sample_inside's
    domain — this is exactly the region flagged as "inside the tip, at the
    top"). Points outside the tip (phi >= 0) are masked to NaN.
    """
    x_lo, x_hi, y_lo, y_hi = _tip_bbox()
    xg = np.linspace(x_lo, x_hi, ng)
    yg = np.linspace(y_lo, y_hi, ng)
    Xg, Yg = np.meshgrid(xg, yg)
    mask_in = phi_np(Xg, Yg) < 0.0
    xy_np = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)

    def eval_fn(chunk_np):
        xy = torch.tensor(chunk_np, device=DEVICE, requires_grad=True)
        iR, iI = net_int(xy)
        lapR = laplacian(iR, xy)
        lapI = laplacian(iI, xy)
        res_R = (lapR + k_int_sq_re * iR - k_int_sq_im * iI) / k_int_sq_mag
        res_I = (lapI + k_int_sq_re * iI + k_int_sq_im * iR) / k_int_sq_mag
        return torch.sqrt(res_R**2 + res_I**2).detach().cpu().numpy()

    mag = _eval_in_chunks(xy_np, eval_fn).reshape(ng, ng)
    mag[~mask_in] = np.nan
    return Xg, Yg, mag, (x_lo, x_hi, y_lo, y_hi)

def _residual_grid_exterior(ng=200):
    """
    Exterior PDE residual |res_ext| over the physical box [0,1]^2. Points
    inside the tip (phi < 0) are masked to NaN.
    """
    xg = np.linspace(0.0, 1.0, ng)
    yg = np.linspace(0.0, 1.0, ng)
    Xg, Yg = np.meshgrid(xg, yg)
    mask_out = phi_np(Xg, Yg) >= 0.0
    xy_np = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)

    def eval_fn(chunk_np):
        xy = torch.tensor(chunk_np, device=DEVICE, requires_grad=True)
        sR, sI = net_scat(xy)
        lapR = laplacian(sR, xy)
        lapI = laplacian(sI, xy)
        res_R = (lapR + k_ext_sq * sR) / k_ext_sq_mag
        res_I = (lapI + k_ext_sq * sI) / k_ext_sq_mag
        return torch.sqrt(res_R**2 + res_I**2).detach().cpu().numpy()

    mag = _eval_in_chunks(xy_np, eval_fn).reshape(ng, ng)
    mag[~mask_out] = np.nan
    return Xg, Yg, mag, (0.0, 1.0, 0.0, 1.0)

def _residual_grid_pml(ng=220):
    """
    PML residual |res_pml| over the padded box [-D_PML, 1+D_PML]^2. Points
    strictly inside [0,1]^2 (governed by the ordinary, non-stretched PDEs
    instead) are masked to NaN. Reuses the EXACT formula from loss_fn's PML
    term, including the inside-tip/outside-tip blend via inside_pml.
    """
    lo, hi = -D_PML, 1.0 + D_PML
    xg = np.linspace(lo, hi, ng)
    yg = np.linspace(lo, hi, ng)
    Xg, Yg = np.meshgrid(xg, yg)
    in_phys = (Xg >= 0.0) & (Xg <= 1.0) & (Yg >= 0.0) & (Yg <= 1.0)
    xy_np = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)

    def eval_fn(chunk_np):
        xy_p = torch.tensor(chunk_np, device=DEVICE, requires_grad=True)

        sx = 1.0 + 1j * _pml_sigma(chunk_np[:, 0], 0.0, 1.0) / k
        sy = 1.0 + 1j * _pml_sigma(chunk_np[:, 1], 0.0, 1.0) / k
        sx_re = torch.tensor(sx.real.astype(np.float32), device=DEVICE).unsqueeze(1)
        sx_im = torch.tensor(sx.imag.astype(np.float32), device=DEVICE).unsqueeze(1)
        sy_re = torch.tensor(sy.real.astype(np.float32), device=DEVICE).unsqueeze(1)
        sy_im = torch.tensor(sy.imag.astype(np.float32), device=DEVICE).unsqueeze(1)
        dsx_im = torch.tensor((_pml_sigma_deriv(chunk_np[:, 0], 0.0, 1.0) / k).astype(np.float32),
                              device=DEVICE).unsqueeze(1)
        dsy_im = torch.tensor((_pml_sigma_deriv(chunk_np[:, 1], 0.0, 1.0) / k).astype(np.float32),
                              device=DEVICE).unsqueeze(1)

        d_pml = phi_np(chunk_np[:, 0], chunk_np[:, 1])
        inside_pml = torch.tensor((d_pml < 0.0).astype(np.float32), device=DEVICE).unsqueeze(1)

        psR, psI = net_scat(xy_p)
        piR, piI = net_int(xy_p)
        pR = inside_pml * piR + (1.0 - inside_pml) * psR
        pI = inside_pml * piI + (1.0 - inside_pml) * psI

        ksq_re = inside_pml * k_int_sq_re + (1.0 - inside_pml) * k_ext_sq
        ksq_im = inside_pml * k_int_sq_im

        gR   = torch.autograd.grad(pR.sum(), xy_p, create_graph=True)[0]
        pRx, pRy = gR[:, 0:1], gR[:, 1:2]
        pRxx = torch.autograd.grad(gR[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
        pRyy = torch.autograd.grad(gR[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

        gI   = torch.autograd.grad(pI.sum(), xy_p, create_graph=True)[0]
        pIx, pIy = gI[:, 0:1], gI[:, 1:2]
        pIxx = torch.autograd.grad(gI[:, 0].sum(), xy_p, create_graph=True)[0][:, 0:1]
        pIyy = torch.autograd.grad(gI[:, 1].sum(), xy_p, create_graph=True)[0][:, 1:2]

        sx_sq = sx_re**2 + sx_im**2
        sy_sq = sy_re**2 + sy_im**2
        Cx_re = (sy_re * sx_re + sy_im * sx_im) / sx_sq
        Cx_im = (sy_im * sx_re - sy_re * sx_im) / sx_sq
        Cy_re = (sx_re * sy_re + sx_im * sy_im) / sy_sq
        Cy_im = (sx_im * sy_re - sx_re * sy_im) / sy_sq

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

        sxsy_re = sx_re * sy_re - sx_im * sy_im
        sxsy_im = sx_re * sy_im + sx_im * sy_re

        K_re = ksq_re * sxsy_re - ksq_im * sxsy_im
        K_im = ksq_re * sxsy_im + ksq_im * sxsy_re

        ksq_mag = torch.sqrt(ksq_re**2 + ksq_im**2 + 1e-12)

        res_pml_R = (Cx_re * pRxx - Cx_im * pIxx
                   + Cxp_re * pRx - Cxp_im * pIx
                   + Cy_re * pRyy - Cy_im * pIyy
                   + Cyp_re * pRy - Cyp_im * pIy
                   + (K_re * pR - K_im * pI)) / ksq_mag
        res_pml_I = (Cx_re * pIxx + Cx_im * pRxx
                   + Cxp_re * pIx + Cxp_im * pRx
                   + Cy_re * pIyy + Cy_im * pRyy
                   + Cyp_re * pIy + Cyp_im * pRy
                   + (K_re * pI + K_im * pR)) / ksq_mag

        return torch.sqrt(res_pml_R**2 + res_pml_I**2).detach().cpu().numpy()

    mag = _eval_in_chunks(xy_np, eval_fn).reshape(ng, ng)
    mag[in_phys] = np.nan   # not the relevant operator inside the physical box
    return Xg, Yg, mag, (lo, hi, lo, hi)

def plot_residual_heatmap(Xg, Yg, mag, extent, title, save_path, y_top_outline=1.0):
    """Log10 heatmap of a residual magnitude field, with the tip outline overlaid."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    log_mag = np.log10(mag + 1e-14)
    im = ax.imshow(log_mag, origin='lower', extent=extent, cmap='inferno',
                   interpolation='nearest', aspect='equal')
    ox, oy = tip_outline_xy(y_top=y_top_outline)
    ax.plot(ox, oy, color='cyan', lw=1.3, ls='--')
    ax.axhline(1.0, color='white', lw=0.8, ls=':', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    cb = plt.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label('log10(residual magnitude)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()

os.makedirs("afm_tip/figures", exist_ok=True)
print("\nComputing spatial residual heatmaps ...")

Xg_i, Yg_i, mag_i, ext_i = _residual_grid_interior(ng=200)
plot_residual_heatmap(Xg_i, Yg_i, mag_i, ext_i,
                      "Interior PDE residual |res_int|",
                      "afm_tip/figures/residual_heatmap_interior.png",
                      y_top_outline=ext_i[3])
_flat_i = mag_i.ravel()
_pk_i = int(np.nanargmax(_flat_i))
print(f"  Interior PDE residual: max={_flat_i[_pk_i]:.3e} at "
      f"(x,y)=({Xg_i.ravel()[_pk_i]:.3f}, {Yg_i.ravel()[_pk_i]:.3f})")

Xg_e, Yg_e, mag_e, ext_e = _residual_grid_exterior(ng=200)
plot_residual_heatmap(Xg_e, Yg_e, mag_e, ext_e,
                      "Exterior PDE residual |res_ext|",
                      "afm_tip/figures/residual_heatmap_exterior.png")
_flat_e = mag_e.ravel()
_pk_e = int(np.nanargmax(_flat_e))
print(f"  Exterior PDE residual: max={_flat_e[_pk_e]:.3e} at "
      f"(x,y)=({Xg_e.ravel()[_pk_e]:.3f}, {Yg_e.ravel()[_pk_e]:.3f})")

Xg_p, Yg_p, mag_p, ext_p = _residual_grid_pml(ng=220)
plot_residual_heatmap(Xg_p, Yg_p, mag_p, ext_p,
                      "PML residual |res_pml|",
                      "afm_tip/figures/residual_heatmap_pml.png",
                      y_top_outline=ext_p[3])
_flat_p = mag_p.ravel()
_pk_p = int(np.nanargmax(_flat_p))
print(f"  PML residual: max={_flat_p[_pk_p]:.3e} at "
      f"(x,y)=({Xg_p.ravel()[_pk_p]:.3f}, {Yg_p.ravel()[_pk_p]:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on dense grid
# ─────────────────────────────────────────────────────────────────────────────
N_grid = 300
xg = np.linspace(0, 1, N_grid)
yg = np.linspace(0, 1, N_grid)
Xg, Yg = np.meshgrid(xg, yg)
in_cyl = tip_mask(Xg, Yg)

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

# Scattered field defined as (total - incident), matching the FDM script.
# Exterior: U_total - U_inc = U_scat (raw net_scat output, unchanged).
# Interior: U_total - U_inc = U_int - U_inc ≈ -U_inc in the absorbing tip,
#           instead of the unconstrained net_scat extrapolation. This makes the
#           interior scattered panel agree with the FDM reference.
U_scat_plot = U_total - U_inc

print(f"\nField check:")
print(f"  Max |U_total|  = {np.max(np.abs(U_total)):.4f}")
print(f"  Inside mean|u| = {np.mean(np.abs(U_int[in_cyl])):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

circle_kw = dict(color='cyan', lw=1.5, zorder=5, ls='--')
_tip_ox, _tip_oy = tip_outline_xy(y_top=1.0)

def panel(ax, data, title, cmap, symmetric):
    valid = data[np.isfinite(data)]
    scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
    scale = max(scale, 1e-12)
    vmin  = -scale if symmetric else 0.0
    im    = ax.imshow(data, origin='lower', extent=[0,1,0,1],
                      cmap=cmap, vmin=vmin, vmax=scale, interpolation='bilinear')
    ax.plot(_tip_ox, _tip_oy, **circle_kw)
    ax.set_title(title, fontsize=16); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.95)

panel(axes[0,0], U_inc.real,      "Incident    Re(u_inc)",   "RdBu_r", True)
panel(axes[0,1], U_scat_plot.real, "Scattered   Re(u_scat)",  "RdBu_r", True)
panel(axes[0,2], U_total.real,    "Total       Re(u_total)", "RdBu_r", True)
panel(axes[1,0], np.abs(U_inc),   "Incident    |u_inc|",     "viridis", False)
panel(axes[1,1], np.abs(U_scat_plot), "Scattered   |u_scat|",    "hot",    False)
panel(axes[1,2], np.abs(U_total), "Total       |u_total|",   "inferno", False)

plt.tight_layout()
os.makedirs("afm_tip/figures", exist_ok=True)
plt.savefig("afm_tip/figures/afm_tip_pinn_fields.png", dpi=150, bbox_inches='tight')
print("Saved: afm_tip/figures/afm_tip_pinn_fields.png")

# Training loss
fig2, ax2 = plt.subplots(figsize=(11, 5))
steps = range(len(loss_history))
ax2.semilogy(steps, loss_history,    color='black',      lw=1.5, label='Total')
ax2.semilogy(steps, hist_pde_int,    color='royalblue',  lw=0.8, label='PDE int')
ax2.semilogy(steps, hist_pde_ext,    color='steelblue',  lw=0.8, label='PDE ext')
ax2.semilogy(steps, hist_field,      color='tomato',     lw=0.8, label='Field BC')
ax2.semilogy(steps, hist_flux,       color='orange',     lw=0.8, label='Flux BC')
ax2.semilogy(steps, hist_pml,        color='mediumpurple',   lw=1.2, label='PML')
ax2.semilogy(steps, hist_dirichlet,  color='darkcyan',  lw=0.8, label='Dirichlet')
ax2.axvline(ADAM_EPOCHS, color='gray', ls='--', lw=1.5,
            label=f'Adam → L-BFGS (step {ADAM_EPOCHS})')
ax2.set_xlabel("Optimiser step")
ax2.set_ylabel("Loss (log scale)")
ax2.set_title("AFM tip PINN — Training Loss by Term")
ax2.legend(fontsize=8, ncol=2, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"afm_tip/figures/afm_tip_pinn_loss.png", dpi=150, bbox_inches='tight')
print(f"Saved: afm_tip/figures/afm_tip_pinn_loss.png")

# plt.show()
print("\nDone.")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble MP4 from captured frames  (plays natively in PowerPoint)
# ─────────────────────────────────────────────────────────────────────────────
if gif_frames:
    mp4_path = "afm_tip/figures/training_animation.mp4"
    # ~6 fps gives a readable pace; last frame held for 3 s (≈18 duplicate frames)
    fps = 6
    hold_frames = fps * 3          # repeat the final frame for 3 seconds
    frames_out  = gif_frames + [gif_frames[-1]] * hold_frames
    writer_kw   = dict(fps=fps, codec="libx264", pixelformat="yuv420p",
                       output_params=["-crf", "18"])   # high quality, PowerPoint-compatible
    imageio.mimwrite(mp4_path, frames_out, format="mp4", **writer_kw)
    print(f"Saved training animation ({len(gif_frames)} frames + {hold_frames} hold): {mp4_path}")