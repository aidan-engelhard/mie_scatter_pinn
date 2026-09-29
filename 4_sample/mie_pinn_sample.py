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
# ── Tip (near-PEC, strong absorber)
N_RE  = 3.0    # real part of refractive index
N_IM  = 30.0   # large imaginary part → |n|≫1, near-PEC / strong skin effect

# ── Substrate (dielectric half-space)
SUB_N_RE = 1.5   # e.g. SiO₂-like
SUB_N_IM = 0.1
Y_SUB    = 0.4   # flat interface height
lam      = 0.5   # wavelength in vacuum (sets scale)
k        = 2.0 * np.pi / lam   # exterior wavenumber  (wavelength = 1)
a        = 0.05           # cylinder radius
cx, cy   = 0.5, 0.5       # cylinder centre

# Incident wave direction
theta_inc = -np.pi / 4
kx = k * np.cos(theta_inc)
ky = k * np.sin(theta_inc)

# Parameters for PML absorbing layer outside the domain Ω (for ABC loss)
d_pml     = 0.6     # PML thickness — matched to mie_fdm_sample.py
sigma_max = 6.0 * k # peak PML conductivity — matched to mie_fdm_sample.py (≈75 at lam=0.5)
pml_order = 3

X_MIN, X_MAX = -d_pml, 1.0 + d_pml
Y_MIN, Y_MAX =  0.0 - d_pml, 1.0 + d_pml

# ─────────────────────────────────────────────────────────────────────────────
# Derived wavenumbers
# ─────────────────────────────────────────────────────────────────────────────
k_int        = (N_RE + 1j * N_IM) * k
k_int_sq_re  = float((N_RE**2 - N_IM**2) * k**2)   # Re(k_int²)
k_int_sq_im  = float(2 * N_RE * N_IM * k**2)        # Im(k_int²)
k_ext_sq     = k**2                                  # k² exterior (real)
k_sub        = (SUB_N_RE + 1j * SUB_N_IM) * k
k_sub_sq_re  = float((SUB_N_RE**2 - SUB_N_IM**2) * k**2)
k_sub_sq_im  = float(2 * SUB_N_RE * SUB_N_IM * k**2)

# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

N_INT   = 1000   # interior PDE points  (inside cylinder)
N_AIR   = 1500   # air PDE points (outside cylinder) — ~30+ pts/λ² at lam=0.5; 200 was far below Nyquist
N_SUB   = 1000   # substrate PDE points (below y=Y_SUB)
N_IFACE = 800    # interface boundary ∂D
N_OUT   = 200    # outer boundary ∂Ω
N_FLAT  = 400
N_PML = 1500   # PML collocation points PER REGION (air + substrate).
               # History: 4000/region was 8000 of the ~11,500 second-derivative
               # points per closure (~70% of its cost) -- the same imbalance
               # found and fixed in tip_sweep_pinn.py. 2000/region halved that;
               # 1500 now matches afm_pinn_3/afm_pinn.py, which solves the SAME
               # air+substrate+PML geometry at the same N_AIR/N_SUB. The two
               # branches are then 3000 of 6500 double-backward points per
               # closure instead of 4000 of 7500 -- ~13% off every closure
               # evaluation. The band is an absorber, not physics; it does not
               # deserve more points than the physical domain.

ADAM_EPOCHS = 0000
LBFGS_STEPS = 50000   # closure-eval budget (not outer steps) -- see Phase 2.
                      # Was 20000 OUTER steps, which at max_iter=20 meant up to
                      # 400,000 closure evals plus strong-Wolfe probes on top:
                      # 5x tip_sweep_pinn's 100k budget and 13x afm_tip_pinn's
                      # 30k. Every other script in the repo counts EVALS; this
                      # one was the last holdout, and it is the single reason
                      # the run takes as long as it does. 80000 matches
                      # afm_pinn_3/afm_pinn.py, the closest sibling.
LR_ADAM     = 8e-4

W_PDE_INT    = 1.0
W_PDE_EXT    = 1.0
W_PDE_SUB    = 1.0
W_FIELD      = 50.0   # Ez continuity on ∂D
W_FLUX       = 50.0   # ∂Ez/∂n continuity on ∂D
W_FLAT_FIELD = 20.0  # Ez continuity on flat interface
W_FLAT_FLUX  = 20.0  # ∂Ez/∂n continuity on flat interface
W_PML        = 0.5
W_PML_DIR    = 1.0

N_FOURIER = 64
HIDDEN    = 3
NEURONS   = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {DEVICE}")
ka = k * a

# ─────────────────────────────────────────────────────────────────────────────
# Load FDM reference solution
# ─────────────────────────────────────────────────────────────────────────────
FDM_PATH = "sample/inputs/afm_fdm_solution.npz"
if os.path.exists(FDM_PATH):
    _fdm = np.load(FDM_PATH)
    FDM_X        = _fdm["X"]           # shape [N_fdm, N_fdm]
    FDM_Y        = _fdm["Y"]
    # FDM scattered field = tip-induced only (background already subtracted)
    # Valid everywhere: air, substrate, and inside cylinder
    FDM_Scat_re  = _fdm["U_scat_re"]
    FDM_Scat_im  = _fdm["U_scat_im"]
    # FDM total field inside cylinder — net_int learns total field, so compare directly
    FDM_Total_re = _fdm["U_total_re"]
    FDM_Total_im = _fdm["U_total_im"]
    FDM_N        = FDM_X.shape[0]
    USE_FDM_DATA = True
    print(f"  Loaded FDM reference: {FDM_PATH}  (grid {FDM_N}x{FDM_N})")
else:
    USE_FDM_DATA = False
    print(f"  WARNING: FDM reference not found at {FDM_PATH} — data loss disabled")

# ── FDM-data WARM START (default OFF — fallback only) ───────────────────────
# History. The physics-only run of the THREE-network version collapsed to
# u_scat ~ 2e-4 against the FDM's 1.28 (figures/pinn_afm_fields.png), and the
# formulation was verified clean at the FDM solution (PDE residuals to machine
# precision, interface conditions identically, background derivatives vs
# autograd to 3e-15) — an optimisation failure. The warm start below was built
# as the escape: pull the nets toward the FDM reference for WARMSTART_EPOCHS,
# then decay the data weight to zero and let physics refine.
#
# It is now DISABLED by default because the collapse mechanism turned out to
# be the two-exterior-network split, not a too-weak forcing:
# cylinder/mie_pinn_pml.py solves the identical cylinder (same a, n, weights)
# physics-only in free space without collapsing, and the merged exterior net
# (see net_sub alias below) restores exactly that structure. If a run with
# WARMSTART_EPOCHS = 0 still collapses (watch rms|u_scat| in the monitor: the
# failed run sat at ~2e-4, the FDM target is 0.467), set this back to ~800 as
# insurance — the machinery is fully wired.
#
# W_DATA_WARM calibration, if used: at init the normalised physics terms total
# ~145 (sum of W_*), and each L0-normalised data term starts at ~1, so
# W_DATA = 1 contributed only ~1/145 of the objective — which is why earlier
# data experiments moved little. 100 per term makes the data pull dominant
# while keeping the solution PDE-feasible.
WARMSTART_EPOCHS = 0     # full-strength data pull for this many epochs (0 = off)
W_DATA_WARM      = 100.0
# After WARMSTART_EPOCHS the data weight is not cut to zero but HALVED every
# WARMSTART_HALFLIFE epochs, and only zeroed (guard kicks in, interp_fdm work
# stops) once it falls below W_DATA_FLOOR. A hard cliff was tried first: in a
# shrunken smoke run the physics-only phase slid straight back toward the zero
# basin after the cut (relL2 0.25 -> 0.77, rms|u_scat| 0.42 -> 0.26 and still
# falling), so the handover to the physics losses must be gradual.
WARMSTART_HALFLIFE = 400
W_DATA_FLOOR       = 0.5     # below this, weights snap to exactly 0

# Live weights for the FDM data terms; decayed by the Adam loop per the
# schedule above. L-BFGS runs with whatever value survives at ADAM_EPOCHS
# (zero if Adam is long enough, which the default budget ensures:
# 100 * 0.5^((3000-800)/400) ~ 2.2 -> ... choose budgets so it floors first).
W_DATA_SCAT = W_DATA_WARM if WARMSTART_EPOCHS > 0 else 0.0
W_DATA_SUB  = W_DATA_WARM if WARMSTART_EPOCHS > 0 else 0.0
W_DATA_INT  = W_DATA_WARM if WARMSTART_EPOCHS > 0 else 0.0

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
# Scaled by |k_sub| (= 1.51k), not k: with the merged exterior net (below) one
# network carries the scattered field in BOTH air and substrate, so its Fourier
# features must resolve the faster substrate oscillation.
B_ext = torch.randn(2, N_FOURIER, device=DEVICE) * abs(k_sub)
B_int = torch.randn(2, N_FOURIER, device=DEVICE) * abs(k_int)
B_sub = torch.randn(2, N_FOURIER, device=DEVICE) * abs(k_sub)

def embed_ext(xy):
    proj = xy @ B_ext
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

def embed_int(xy):
    proj = xy @ B_int
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

def embed_sub(xy):
    proj = xy @ B_sub
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

net_int  = MLP(embed_int).to(DEVICE)   # total field inside cylinder
net_scat = MLP(embed_ext).to(DEVICE)   # scattered field outside cylinder

# ── torch.compile (ported from variable_tips/tip_sweep_pinn.py, where it gave
# a measured speedup on the same GPU). Two non-negotiables for a PINN:
#   - donated_buffer=False: AOTAutograd otherwise frees intermediate buffers
#     during backward, which is incompatible with the create_graph=True that
#     every second-derivative residual here requires.
#   - the warm-up below forces compilation through the double-backward path NOW
#     and falls back to eager on failure, instead of dying mid-training.
USE_COMPILE = True
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
        for _w_net in (net_int, net_scat):
            _w_r, _w_i = _w_net(_w_xy)
            _w_g  = torch.autograd.grad(_w_r.sum(), _w_xy, create_graph=True)[0]
            _w_gg = torch.autograd.grad(_w_g[:, 0].sum(), _w_xy,
                                        create_graph=True)[0]
            _w_gg.sum().backward()
        print("  torch.compile warm-up OK (double-backward compiled)")
    except Exception as _w_exc:
        net_int  = getattr(net_int,  "_orig_mod", net_int)
        net_scat = getattr(net_scat, "_orig_mod", net_scat)
        USE_COMPILE = False
        print(f"  torch.compile FAILED -> falling back to eager: "
              f"{type(_w_exc).__name__}: {str(_w_exc).splitlines()[-1][:160]}")
    finally:
        for _w_p in list(net_int.parameters()) + list(net_scat.parameters()):
            _w_p.grad = None          # warm-up must not seed real gradients
# ── MERGED EXTERIOR: net_sub is an ALIAS of net_scat, not a third network ────
# Decisive A/B within this repo: cylinder/mie_pinn_pml.py — the IDENTICAL
# cylinder (a=0.05, n=3+30i, W_FIELD=W_FLUX=50, physics-only) in free space —
# converges to a clean radiating solution, while this script with the same
# cylinder collapsed to u_scat ~ 2e-4. Same forcing circle, same index, same
# weights; the difference is the substrate machinery. With separate air and
# substrate networks, any radiation crossing y = Y_SUB (0.05 below the
# cylinder!) is penalised by the flat-interface losses unless BOTH nets move
# in lockstep — a coordination barrier the free-space script does not have,
# and sitting still at zero avoids it entirely.
#
# One exterior network removes that barrier, exactly as the FDM treats the
# scattered field as ONE unknown v with piecewise k². This is legitimate for
# the s-polarisation solved here: u_scat AND du_scat/dn are continuous across
# the flat interface (the background satisfies both matching conditions,
# t = 1 + r), so the scattered field is C^1 there and one smooth network can
# represent it. Only the second derivative kinks (k² jumps); the PDE samplers
# already keep +-0.005 margins around Y_SUB, so no residual is evaluated on
# the kink itself. The flat-interface loss terms become identically zero
# (same net on both sides, analytic jump zero) — continuity is now enforced
# by construction instead of by soft penalty.
net_sub = net_scat   # alias: u_total = u_bg_sub + net_scat below Y_SUB

n_int  = sum(p.numel() for p in net_int.parameters())
n_scat = sum(p.numel() for p in net_scat.parameters())
print(f"\n  net_int  parameters: {n_int:,}")
print(f"  net_scat parameters: {n_scat:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Background field (and normal derivatives) computed using Fresnel coefficients
# ─────────────────────────────────────────────────────────────────────────────
 
# Precomputed Fresnel scalar quantities
ky_sub = np.sqrt((SUB_N_RE + 1j*SUB_N_IM)**2 * k**2 - kx**2 + 0j)
if ky_sub.real > 0: ky_sub = -ky_sub   # Re(ky_sub) < 0: downward-propagating
# s-polarisation (TE / out-of-plane-E) Fresnel coefficients — the correct pair
# for a scalar field with u AND du/dy continuous at y=Y_SUB, which is what the
# flat-interface losses below enforce. With these, the background satisfies
# both matching conditions exactly, so the imposed jumps (field AND flux) are
# analytically zero and u_scat is purely tip-induced — same object as the FDM.
# (Names r_TM/t_TM kept for downstream code; "TM mode" here = Ez out of plane
# = s-polarisation in optics naming.)
q_air = abs(ky)
q_sub = -ky_sub          # Re(q_sub) > 0
r_TM  = (q_air - q_sub) / (q_air + q_sub)
t_TM  = (2.0 * q_air)   / (q_air + q_sub)

# Split into real/imag for use in torch functions
r_re, r_im = float(r_TM.real), float(r_TM.imag)
t_re, t_im = float(t_TM.real), float(t_TM.imag)
ky_sub_re  = float(ky_sub.real)
ky_sub_im  = float(ky_sub.imag)

def u_bg_air_re(xy):
    # u_bg_air = exp(i(kx x + ky(y-Y_SUB))) + r_TM exp(i(kx x - ky(y-Y_SUB)))
    # (phase referenced to the interface, matching mie_fdm_sample.py)
    phase_inc  = kx * xy[:,0:1] + ky * (xy[:,1:2] - Y_SUB)
    phase_refl = kx * xy[:,0:1] - ky * (xy[:,1:2] - Y_SUB)
    refl_re    = r_re * torch.cos(phase_refl) - r_im * torch.sin(phase_refl)
    return torch.cos(phase_inc) + refl_re

def u_bg_air_im(xy):
    phase_inc  = kx * xy[:,0:1] + ky * (xy[:,1:2] - Y_SUB)
    phase_refl = kx * xy[:,0:1] - ky * (xy[:,1:2] - Y_SUB)
    refl_im    = r_re * torch.sin(phase_refl) + r_im * torch.cos(phase_refl)
    return torch.sin(phase_inc) + refl_im

def u_bg_sub_re(xy):
    # u_bg_sub = t_TM exp(i(kx x + ky_sub(y - Y_SUB)))
    # ky_sub is complex: exp(i(a+ib)s) = exp(-b*s)(cos(a*s) + i sin(a*s))
    s   = xy[:,1:2] - Y_SUB
    amp = torch.exp(-ky_sub_im * s)
    ph  = kx * xy[:,0:1] + ky_sub_re * s
    return amp * (t_re * torch.cos(ph) - t_im * torch.sin(ph))

def u_bg_sub_im(xy):
    s   = xy[:,1:2] - Y_SUB
    amp = torch.exp(-ky_sub_im * s)
    ph  = kx * xy[:,0:1] + ky_sub_re * s
    return amp * (t_re * torch.sin(ph) + t_im * torch.cos(ph))

def du_bg_air_dy_re(xy):
    # d/dy of u_bg_air at y=Y_SUB, n_hat = +y
    # d/dy[exp(i ky y)] = iky exp(i ky y)  -> Re part: -ky sin(phase)
    # d/dy[r exp(-i ky y + ...)] = -iky r exp(...)  -> careful with signs
    phase_inc  = kx * xy[:,0:1] + ky * (xy[:,1:2] - Y_SUB)
    phase_refl = kx * xy[:,0:1] - ky * (xy[:,1:2] - Y_SUB)
    # d/dy u_inc: i*ky * exp(i phase_inc)  -> Re = -ky sin, Im = ky cos
    dinc_re = -ky * torch.sin(phase_inc)
    # d/dy u_refl: i*(-ky) * r_TM * exp(i phase_refl)
    drefl_re = ky * (r_re * torch.sin(phase_refl) + r_im * torch.cos(phase_refl))
    return dinc_re + drefl_re

def du_bg_air_dy_im(xy):
    phase_inc  = kx * xy[:,0:1] + ky * (xy[:,1:2] - Y_SUB)
    phase_refl = kx * xy[:,0:1] - ky * (xy[:,1:2] - Y_SUB)
    dinc_im  =  ky * torch.cos(phase_inc)
    # d/dy u_refl = -i*ky * r * exp(i*phase_refl);
    # Im[-i*ky*(r_re+i r_im)(cos+i sin)] = -ky*(r_re*cos - r_im*sin)
    # (sign previously flipped — verified against autograd)
    drefl_im = -ky * (r_re * torch.cos(phase_refl) - r_im * torch.sin(phase_refl))
    return dinc_im + drefl_im

def du_bg_sub_dy_re(xy):
    # d/dy u_bg_sub = i*ky_sub * u_bg_sub
    # i*(a+ib) = ia - b  ->  Re[i*ky_sub * u] = -ky_sub_im*Re(u) - ky_sub_re*Im(u)
    ubR = u_bg_sub_re(xy)
    ubI = u_bg_sub_im(xy)
    return -ky_sub_im * ubR - ky_sub_re * ubI

def du_bg_sub_dy_im(xy):
    ubR = u_bg_sub_re(xy)
    ubI = u_bg_sub_im(xy)
    return  ky_sub_re * ubR - ky_sub_im * ubI

def du_bg_air_dn_re(xy, nxy):
    nx = nxy[:,0:1];  ny = nxy[:,1:2]
    phase_inc  = kx * xy[:,0:1] + ky * (xy[:,1:2] - Y_SUB)
    phase_refl = kx * xy[:,0:1] - ky * (xy[:,1:2] - Y_SUB)
    kn_inc  =  kx*nx + ky*ny    # k_inc  . n
    kn_refl =  kx*nx - ky*ny    # k_refl . n  (ky flipped)
    # Re[ i*kn_inc * exp(i*phase_inc) ] = -kn_inc * sin(phase_inc)
    d_inc_re  = -kn_inc  * torch.sin(phase_inc)
    # Re[ i*kn_refl * r_TM * exp(i*phase_refl) ]
    #   = -kn_refl*(r_re*sin - r_im*cos)  ... wait, r_TM complex:
    #   i*kn*(r_re+ir_im)*(cos+isin) -> Re = -kn*(r_re*sin + r_im*cos)... 
    # Careful: Re[i*kn*(r_re+ir_im)(cos(p)+isin(p))]
    #        = Re[(i*kn*r_re - kn*r_im)(cos+isin)]
    #        = -kn*r_re*sin(p) - kn*r_im*cos(p)
    d_refl_re = -kn_refl * (r_re * torch.sin(phase_refl)
                           + r_im * torch.cos(phase_refl))
    return d_inc_re + d_refl_re

def du_bg_air_dn_im(xy, nxy):
    nx = nxy[:,0:1];  ny = nxy[:,1:2]
    phase_inc  = kx * xy[:,0:1] + ky * (xy[:,1:2] - Y_SUB)
    phase_refl = kx * xy[:,0:1] - ky * (xy[:,1:2] - Y_SUB)
    kn_inc  = kx*nx + ky*ny
    kn_refl = kx*nx - ky*ny
    # Im[i*kn*exp(i*phase)] = kn*cos(phase)
    d_inc_im  =  kn_inc  * torch.cos(phase_inc)
    # Im[i*kn*(r_re+ir_im)(cos+isin)]
    #   = kn*r_re*cos(p) - kn*r_im*sin(p)
    d_refl_im =  kn_refl * (r_re * torch.cos(phase_refl)
                            - r_im * torch.sin(phase_refl))
    return d_inc_im + d_refl_im

# ─────────────────────────────────────────────────────────────────────────────
# UPML stretched-Helmholtz loss — identical formulation to afm_tip_pinn.py /
# mie_pinn_pml.py.
#
# The absorbing layer is imposed as the DIVERGENCE-FORM stretched Helmholtz
# operator
#
#   ∂/∂x ( (s_y/s_x) ∂u/∂x ) + ∂/∂y ( (s_x/s_y) ∂u/∂y ) + k² s_x s_y u = 0
#
# with complex coordinate stretch  s_x(x) = 1 + i σ_x(x)/k_ref ,
#                                  s_y(y) = 1 + i σ_y(y)/k_ref .
#
# Expanding the divergences yields FOUR groups of terms: the two second-
# derivative terms, the two FIRST-derivative terms carrying the gradient of the
# stretch (C'_x, C'_y — these were missing in the previous version of this
# file, which is what caused the PML to reflect), and the reaction term. The
# stretch tensors and their derivatives are precomputed in numpy inside the
# samplers, exactly as in the two reference scripts.
# ─────────────────────────────────────────────────────────────────────────────

def _pml_sigma(coord, lo, hi):
    """
    Polynomial PML conductivity profile σ (same units as k).
    Zero inside [lo, hi], rising as (depth/d_pml)^pml_order outside.
    """
    s = np.zeros_like(coord, dtype=float)
    mask_lo = coord < lo
    mask_hi = coord > hi
    s[mask_lo] = sigma_max * ((lo - coord[mask_lo]) / d_pml) ** pml_order
    s[mask_hi] = sigma_max * ((coord[mask_hi] - hi) / d_pml) ** pml_order
    return s

def _pml_sigma_deriv(coord, lo, hi):
    """
    Magnitude of the derivative of the polynomial PML conductivity profile,
    dσ/dn = (m·σ_max/d_pml)·(depth/d_pml)^(m-1).  (Same sign convention as the
    reference scripts: positive when penetrating from either side.)
    """
    s = np.zeros_like(coord, dtype=float)
    mask_lo = coord < lo
    mask_hi = coord > hi
    if pml_order >= 1:
        s[mask_lo] = (pml_order * sigma_max / d_pml) * ((lo - coord[mask_lo]) / d_pml) ** (pml_order - 1)
        s[mask_hi] = (pml_order * sigma_max / d_pml) * ((coord[mask_hi] - hi) / d_pml) ** (pml_order - 1)
    return s

def _pml_stretch_tensors(xy, k_ref):
    """
    Precompute the UPML stretch tensors for an (N,2) numpy array of PML points.

      s_x(x) = 1 + i σ_x(x)/k_ref ,   s_y(y) = 1 + i σ_y(y)/k_ref

    Returns (xy_t, sx_re, sx_im, sy_re, sy_im, dsx_im, dsy_im) where the last
    two are  dσ/k_ref  (the imaginary part of s' , since Re(s) is constant).
    Points are graded against the PHYSICAL box [0,1]² — the stretch is purely
    geometric; the medium enters only through k_ref and, in the residual,
    through k².
    """
    sx = 1.0 + 1j * _pml_sigma(xy[:, 0], 0.0, 1.0) / k_ref
    sy = 1.0 + 1j * _pml_sigma(xy[:, 1], 0.0, 1.0) / k_ref

    xy_t     = torch.tensor(xy.astype(np.float32), device=DEVICE, requires_grad=True)
    sx_re_t  = torch.tensor(sx.real.astype(np.float32), device=DEVICE).unsqueeze(1)
    sx_im_t  = torch.tensor(sx.imag.astype(np.float32), device=DEVICE).unsqueeze(1)
    sy_re_t  = torch.tensor(sy.real.astype(np.float32), device=DEVICE).unsqueeze(1)
    sy_im_t  = torch.tensor(sy.imag.astype(np.float32), device=DEVICE).unsqueeze(1)
    dsx_im_t = torch.tensor((_pml_sigma_deriv(xy[:, 0], 0.0, 1.0) / k_ref).astype(np.float32),
                            device=DEVICE).unsqueeze(1)
    dsy_im_t = torch.tensor((_pml_sigma_deriv(xy[:, 1], 0.0, 1.0) / k_ref).astype(np.float32),
                            device=DEVICE).unsqueeze(1)
    return xy_t, sx_re_t, sx_im_t, sy_re_t, sy_im_t, dsx_im_t, dsy_im_t

def pml_pde_residual(net, xy_pml, k2_re, k2_im):
    """
    Residual of the divergence-form UPML-stretched Helmholtz equation, in the
    exact form used by afm_tip_pinn.py / mie_pinn_pml.py (generalised to a
    complex reaction wavenumber k² = k2_re + i·k2_im so it serves both the air
    PML on net_scat, k²=k_ext², and the substrate PML on net_sub, k²=k_sub²).

    `xy_pml` is the tuple produced by _pml_stretch_tensors:
        (xy_p, sx_re, sx_im, sy_re, sy_im, dsx_im, dsy_im)
    Returns the mean-squared residual (scalar).
    """
    xy_p, sx_re, sx_im, sy_re, sy_im, dsx_im, dsy_im = xy_pml

    pR, pI = net(xy_p)

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

    # Cx = s_y / s_x,  Cy = s_x / s_y   (coefficients of u_xx, u_yy)
    sx_sq = sx_re**2 + sx_im**2
    sy_sq = sy_re**2 + sy_im**2

    Cx_re = (sy_re * sx_re + sy_im * sx_im) / sx_sq
    Cx_im = (sy_im * sx_re - sy_re * sx_im) / sx_sq

    Cy_re = (sx_re * sy_re + sx_im * sy_im) / sy_sq
    Cy_im = (sx_im * sy_re - sx_re * sy_im) / sy_sq

    # Cx' = -s_y s_x' / s_x²,  Cy' = -s_x s_y' / s_y²   (coefficients of u_x, u_y)
    # with s_x' = i·dsx_im purely imaginary.
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

    # s_x s_y  and the complex reaction coefficient  K = k² s_x s_y
    sxsy_re = sx_re * sy_re - sx_im * sy_im
    sxsy_im = sx_re * sy_im + sx_im * sy_re
    K_re = k2_re * sxsy_re - k2_im * sxsy_im
    K_im = k2_re * sxsy_im + k2_im * sxsy_re

    res_R = (Cx_re * pRxx - Cx_im * pIxx
           + Cxp_re * pRx - Cxp_im * pIx
           + Cy_re * pRyy - Cy_im * pIyy
           + Cyp_re * pRy - Cyp_im * pIy
           + (K_re * pR - K_im * pI))
    res_I = (Cx_re * pIxx + Cx_im * pRxx
           + Cxp_re * pIx + Cxp_im * pRx
           + Cy_re * pIyy + Cy_im * pRyy
           + Cyp_re * pIy + Cyp_im * pRy
           + (K_re * pI + K_im * pR))

    return (res_R**2 + res_I**2).mean()

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

def sample_air(n):
    """Outside cylinder AND above substrate (y > Y_SUB)."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 4, 2)
        r2 = (xy[:, 0] - cx)**2 + (xy[:, 1] - cy)**2
        mask = (r2 > (a + 0.008)**2) & (xy[:, 1] > Y_SUB + 0.005)
        pts.append(xy[mask])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)

def sample_substrate(n):
    """Points strictly inside substrate (y < Y_SUB)."""
    pts = []
    while sum(len(p) for p in pts) < n:
        xy = np.random.rand(n * 3, 2)
        mask = xy[:, 1] < Y_SUB - 0.005
        pts.append(xy[mask])
    return torch.tensor(np.vstack(pts)[:n].astype(np.float32),
                        device=DEVICE, requires_grad=True)

def sample_flat_interface(n):
    """Points on flat substrate surface y=Y_SUB, normal = +ŷ (pointing into air)."""
    x   = np.random.rand(n)
    y   = np.full(n, Y_SUB)
    nxy = np.stack([np.zeros(n), np.ones(n)], axis=1)   # n̂ = +ŷ
    xy  = torch.tensor(np.stack([x, y], axis=1).astype(np.float32),
                       device=DEVICE, requires_grad=True)
    nxy = torch.tensor(nxy.astype(np.float32), device=DEVICE)
    return xy, nxy

def sample_interface(n):
    """Points on cylinder boundary ∂D with outward normals."""
    theta = np.random.rand(n) * 2 * np.pi
    x  = cx + a * np.cos(theta);  y  = cy + a * np.sin(theta)
    nx = np.cos(theta);            ny = np.sin(theta)
    xy  = torch.tensor(np.stack([x, y], 1).astype(np.float32),
                       device=DEVICE, requires_grad=True)
    nxy = torch.tensor(np.stack([nx, ny], 1).astype(np.float32), device=DEVICE)
    return xy, nxy

def sample_pml_air(n):
    """Points in PML region where medium is air (y > Y_SUB, outside physical domain)."""
    pts = []
    while sum(len(p) for p in pts) < n:
        x = np.random.uniform(X_MIN, X_MAX, n*4)
        y = np.random.uniform(Y_MIN, Y_MAX, n*4)
        # In PML = outside [0,1]^2, in air = y > Y_SUB, outside cylinder
        in_pml = (x < 0) | (x > 1) | (y > 1)
        in_air = y > Y_SUB
        r2 = (x - cx)**2 + (y - cy)**2
        mask = in_pml & in_air & (r2 > (a + 0.01)**2)
        pts.append(np.stack([x[mask], y[mask]], axis=1))
    xy = np.vstack(pts)[:n]
    # Air PML: geometric stretch referenced to the exterior (vacuum) wavenumber.
    return _pml_stretch_tensors(xy, float(k))

def sample_pml_sub(n):
    """Points in PML region where medium is substrate (y < Y_SUB)."""
    pts = []
    while sum(len(p) for p in pts) < n:
        x = np.random.uniform(X_MIN, X_MAX, n*4)
        y = np.random.uniform(Y_MIN, Y_SUB, n*4)
        in_pml = (x < 0) | (x > 1) | (y < 0)
        mask = in_pml
        pts.append(np.stack([x[mask], y[mask]], axis=1))
    xy = np.vstack(pts)[:n]
    # Substrate PML: stretch referenced to the substrate wavenumber magnitude,
    # so σ/k_ref matches the local wave as it does for the air PML.
    return _pml_stretch_tensors(xy, float(abs(k_sub)))

def sample_pml_outer(n):
    """Points on the outermost edge of the PML — Dirichlet u=0 here."""
    pts = []
    # Sample on all four outer edges
    for _ in range(n * 4):
        edge = np.random.randint(4)
        if edge == 0:   # top
            pts.append([np.random.uniform(X_MIN, X_MAX), Y_MAX])
        elif edge == 1: # bottom
            pts.append([np.random.uniform(X_MIN, X_MAX), Y_MIN])
        elif edge == 2: # left
            pts.append([X_MIN, np.random.uniform(Y_MIN, Y_MAX)])
        else:           # right
            pts.append([X_MAX, np.random.uniform(Y_MIN, Y_MAX)])
    pts = np.array(pts[:n], dtype=np.float32)
    return torch.tensor(pts, device=DEVICE, requires_grad=True)

# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────
def loss_fn(xy_in, xy_out, xy_if, n_if, xy_sub, xy_fl, n_fl, xy_pml_air, xy_pml_sub, xy_pml_out):

    # ── Interior PDE: (∇² + k_int²) u_int = 0
    # k_int² is complex: splits into coupled real/imag equations
    # Re: ∇²uR + kR² uR - kI² uI = 0
    # Im: ∇²uI + kR² uI + kI² uR = 0
    iR, iI = net_int(xy_in)
    lapR_in = laplacian(iR, xy_in)
    lapI_in = laplacian(iI, xy_in)
    res_in_R = lapR_in + k_int_sq_re * iR - k_int_sq_im * iI
    res_in_I = lapI_in + k_int_sq_re * iI + k_int_sq_im * iR
    l_pde_int = (res_in_R**2 + res_in_I**2).mean()

    # ── Exterior PDE: (∇² + k²) u_scat = 0  (k² real)
    sR, sI = net_scat(xy_out)
    lapR_ex = laplacian(sR, xy_out)
    lapI_ex = laplacian(sI, xy_out)
    res_ex_R = lapR_ex + k_ext_sq * sR
    res_ex_I = lapI_ex + k_ext_sq * sI
    l_pde_ext = (res_ex_R**2 + res_ex_I**2).mean()

    # ── Substrate PDE: (∇² + k_sub²) u_sub = 0
    bR, bI  = net_sub(xy_sub)
    lapR_sub = laplacian(bR, xy_sub)
    lapI_sub = laplacian(bI, xy_sub)
    res_sub_R = lapR_sub + k_sub_sq_re * bR - k_sub_sq_im * bI
    res_sub_I = lapI_sub + k_sub_sq_re * bI + k_sub_sq_im * bR
    l_pde_sub = (res_sub_R**2 + res_sub_I**2).mean()

    # ── Cylinder Interface: Ez continuity  u_int = u_inc + u_scat  on ∂D
    iR_if, iI_if   = net_int(xy_if)
    sR_if, sI_if   = net_scat(xy_if)
    bg_R_if = u_bg_air_re(xy_if)
    bg_I_if = u_bg_air_im(xy_if)
    l_field = ((iR_if - (sR_if + bg_R_if))**2 +
            (iI_if - (sI_if + bg_I_if))**2).mean()

    # ── Cylinder Interface: flux continuity  ∂u_int/∂n = ∂u_scat/∂n + ∂u_inc/∂n  on ∂D
    dn_iR = normal_deriv(iR_if, xy_if, n_if)
    dn_iI = normal_deriv(iI_if, xy_if, n_if)
    dn_sR = normal_deriv(sR_if, xy_if, n_if)
    dn_sI = normal_deriv(sI_if, xy_if, n_if)
    dn_bgR_if = du_bg_air_dn_re(xy_if, n_if)   # autograd through u_bg_air
    dn_bgI_if = du_bg_air_dn_im(xy_if, n_if)
    l_flux = ((dn_iR - (dn_sR + dn_bgR_if))**2 +
            (dn_iI - (dn_sI + dn_bgI_if))**2).mean()
    
    # ── 7./8. Flat interface: field + flux continuity across y=Y_SUB.
    # With the MERGED exterior (net_sub is net_scat) both residuals are
    # identically zero: same network on both sides, and the background jumps
    # are analytically zero (t = 1 + r). Computing them would spend two
    # forwards and four first-derivative backwards per closure evaluation on
    # exact zeros, so they are skipped. The split-net branch is kept for the
    # case where net_sub is ever made a separate network again.
    if net_sub is net_scat:
        l_flat_field = torch.tensor(0.0, device=DEVICE)
        l_flat_flux  = torch.tensor(0.0, device=DEVICE)
    else:
        sR_fl, sI_fl = net_scat(xy_fl)    # u_scat_air at interface
        bR_fl, bI_fl = net_sub(xy_fl)     # u_scat_sub at interface

        # Known jump in background at interface
        jump_R = u_bg_sub_re(xy_fl) - u_bg_air_re(xy_fl)
        jump_I = u_bg_sub_im(xy_fl) - u_bg_air_im(xy_fl)

        l_flat_field = ((sR_fl - bR_fl - jump_R)**2 +
                        (sI_fl - bI_fl - jump_I)**2).mean()

        dn_sR_fl  = normal_deriv(sR_fl, xy_fl, n_fl)
        dn_sI_fl  = normal_deriv(sI_fl, xy_fl, n_fl)
        dn_bR_fl  = normal_deriv(bR_fl, xy_fl, n_fl)
        dn_bI_fl  = normal_deriv(bI_fl, xy_fl, n_fl)
        # Known flux jump from background
        djump_R = du_bg_sub_dy_re(xy_fl) - du_bg_air_dy_re(xy_fl)
        djump_I = du_bg_sub_dy_im(xy_fl) - du_bg_air_dy_im(xy_fl)

        l_flat_flux = ((dn_sR_fl - dn_bR_fl - djump_R)**2 +
                       (dn_sI_fl - dn_bI_fl - djump_I)**2).mean()

    # PML residual for net_scat in air PML region
    l_pml_air   = pml_pde_residual(net_scat, xy_pml_air, k_ext_sq, 0.0)

    # PML residual for net_sub in substrate PML region
    l_pml_sub   = pml_pde_residual(net_sub, xy_pml_sub, k_sub_sq_re, k_sub_sq_im)

    # Dirichlet on outer PML edge: u_scat = 0. With the merged exterior a
    # second net_sub evaluation would recompute the identical outputs and
    # double-count the term, so it runs only for genuinely separate nets.
    # (L0_PML_DIR is measured with the same definition, so the normalised
    # term is unaffected by which branch is active.)
    sR_out, sI_out = net_scat(xy_pml_out)
    if net_sub is net_scat:
        l_pml_dirichlet = (sR_out**2 + sI_out**2).mean()
    else:
        bR_out, bI_out = net_sub(xy_pml_out)
        l_pml_dirichlet = (sR_out**2 + sI_out**2 +
                           bR_out**2 + bI_out**2).mean()
    
    
    # Per-term normalisation: each term is divided by its value at initialisation
    # (L0_*, computed once below) so every term starts at ~1 and the W_* weights
    # express genuine relative importance rather than compensating for the very
    # different natural magnitudes of, e.g., the interior PDE residual (huge for
    # n=3+30i) versus the interface terms. Same scheme as afm_tip_pinn / mie_pinn_pml.
    total = (W_PDE_INT    * l_pde_int    / L0_PDE_INT    +
             W_PDE_EXT    * l_pde_ext    / L0_PDE_EXT    +
             W_PDE_SUB    * l_pde_sub    / L0_PDE_SUB    +
             W_FIELD      * l_field      / L0_FIELD      +
             W_FLUX       * l_flux       / L0_FLUX       +
             W_FLAT_FIELD * l_flat_field / L0_FLAT_FIELD +
             W_FLAT_FLUX  * l_flat_flux  / L0_FLAT_FLUX  +
             W_PML        * l_pml_air    / L0_PML_AIR    +
             W_PML        * l_pml_sub    / L0_PML_SUB    +
             W_PML_DIR    * l_pml_dirichlet / L0_PML_DIR)

    # ── 9. FDM data loss — penalise deviation from reference solution
    l_data_scat = torch.tensor(0.0, device=DEVICE)
    l_data_sub  = torch.tensor(0.0, device=DEVICE)
    l_data_int  = torch.tensor(0.0, device=DEVICE)
    # Guarded on the live weights, not just data availability: after the
    # warm start switches off, each interp_fdm call would still cost a
    # GPU->CPU sync + scipy interpolation + copy back, multiplied by zero.
    if USE_FDM_DATA and (W_DATA_SCAT > 0.0 or W_DATA_SUB > 0.0
                         or W_DATA_INT > 0.0):
        # net_scat learns u_scat in air — compare against FDM scattered field
        fdm_sR = interp_fdm(FDM_Scat_re, xy_out)
        fdm_sI = interp_fdm(FDM_Scat_im, xy_out)
        l_data_scat = ((sR - fdm_sR)**2 + (sI - fdm_sI)**2).mean()

        # net_sub learns u_scat in substrate — compare against FDM scattered field
        # (FDM_Scat is the tip-induced scattered field everywhere, substrate included)
        bR_sub, bI_sub = net_sub(xy_sub)
        fdm_bR = interp_fdm(FDM_Scat_re, xy_sub)
        fdm_bI = interp_fdm(FDM_Scat_im, xy_sub)
        l_data_sub = ((bR_sub - fdm_bR)**2 + (bI_sub - fdm_bI)**2).mean()

        # net_int learns u_total inside cylinder — compare against FDM total field
        fdm_iR = interp_fdm(FDM_Total_re, xy_in)
        fdm_iI = interp_fdm(FDM_Total_im, xy_in)
        l_data_int = ((iR - fdm_iR)**2 + (iI - fdm_iI)**2).mean()

        total = total + (W_DATA_SCAT * l_data_scat / L0_DATA_SCAT +
                         W_DATA_SUB  * l_data_sub  / L0_DATA_SUB +
                         W_DATA_INT  * l_data_int  / L0_DATA_INT)

    return total, l_pde_int, l_pde_ext, l_pde_sub, l_field, l_flux, l_flat_field, l_flat_flux, l_pml_air, l_pml_sub, l_pml_dirichlet, l_data_scat, l_data_sub, l_data_int

# ─────────────────────────────────────────────────────────────────────────────
# Optimiser — trains both networks simultaneously
# ─────────────────────────────────────────────────────────────────────────────
# net_sub is an alias of net_scat — including it again would hand the
# optimiser duplicate parameters (torch.optim rejects that).
all_params = list(net_int.parameters()) + list(net_scat.parameters())
optimizer  = torch.optim.Adam(all_params, lr=LR_ADAM)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=ADAM_EPOCHS, eta_min=1e-5)

loss_history = []
hist_pde_int  = []
hist_pde_air  = []
hist_pde_sub  = []
hist_field    = []
hist_flux     = []
hist_flat_field = []
hist_flat_flux  = []
hist_pml_air    = []
hist_pml_sub    = []
hist_pml_dir    = []
hist_data_scat = []
hist_data_sub  = []
hist_data_int  = []

# ─────────────────────────────────────────────────────────────────────────────
# Training animation — capture a coarse-grid snapshot of the scattered/total
# fields periodically and assemble them into an MP4 at the end (same approach as
# afm_tip_pinn.py / mie_pinn_pml.py).
# ─────────────────────────────────────────────────────────────────────────────
gif_frames = []               # holds rendered RGB frames
N_grid_gif = 200              # coarse grid for fast in-training rendering
# Frame cadence. Each frame is a 200x200 forward pass plus a 2x2 matplotlib
# figure with four colorbars rendered to PNG -- CPU work the GPU waits behind.
# At an 80,000-eval L-BFGS budget (~4000 outer steps) these give ~100 Adam
# frames and ~40 L-BFGS frames.
FRAME_EVERY_ADAM  = 100       # capture a frame every N Adam epochs
FRAME_EVERY_LBFGS = 10        # capture a frame every N L-BFGS steps
_FW, _FH = 912 / 90, 816 / 90

def save_frame(label, total_loss):
    """Evaluate the three networks on a coarse grid and capture a 2x2 snapshot
    (scattered + total field, real part and magnitude) for the animation."""
    net_int.eval(); net_scat.eval(); net_sub.eval()
    with torch.no_grad():
        xg = np.linspace(0, 1, N_grid_gif)
        yg = np.linspace(0, 1, N_grid_gif)
        Xg, Yg = np.meshgrid(xg, yg)
        R2       = (Xg - cx)**2 + (Yg - cy)**2
        in_cyl_f = R2 <= a**2
        in_sub_f = Yg < Y_SUB
        xy_ev = torch.tensor(
            np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32), device=DEVICE)
        iR_g, iI_g = net_int(xy_ev)
        sR_g, sI_g = net_scat(xy_ev)
        # alias-aware, as in monitor_relL2(): with the merged exterior this is
        # the identical 40,000-point forward pass repeated once per frame.
        bR_g, bI_g = (sR_g, sI_g) if net_sub is net_scat else net_sub(xy_ev)

    U_int_f  = (iR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * iI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))
    U_sa_f   = (sR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * sI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))   # scat air
    U_ss_f   = (bR_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif) +
                1j * bI_g.cpu().numpy().reshape(N_grid_gif, N_grid_gif))   # scat sub

    U_bg_air_f = (np.exp(1j*(kx*Xg + ky*(Yg - Y_SUB))) +
                  r_TM * np.exp(1j*(kx*Xg - ky*(Yg - Y_SUB))))
    U_bg_sub_f = t_TM * np.exp(1j*(kx*Xg + ky_sub*(Yg - Y_SUB)))

    U_scat_f  = np.where(in_cyl_f, 0j,
                np.where(in_sub_f, U_ss_f, U_sa_f))
    U_total_f = np.where(in_cyl_f, U_int_f,
                np.where(in_sub_f, U_bg_sub_f + U_ss_f, U_bg_air_f + U_sa_f))

    matplotlib.rcParams.update({'font.size': 9})
    fig_f, axes_f = plt.subplots(2, 2, figsize=(_FW, _FH))
    fig_f.suptitle(f"{label}   |   Loss = {total_loss:.3e}", fontsize=13)
    fig_f.tight_layout(rect=[0, 0, 1, 0.96])

    circle_kw_f = dict(color='cyan',   lw=1.5, fill=False, zorder=5, ls='--')
    sub_kw_f    = dict(color='yellow', lw=1.2, zorder=6, ls='--')

    def _panel(ax, data, title, cmap, symmetric):
        valid = data[np.isfinite(data)]
        scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
        scale = max(scale, 1e-12)
        vmin  = -scale if symmetric else 0.0
        im    = ax.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                          cmap=cmap, vmin=vmin, vmax=scale, interpolation='bilinear')
        ax.add_patch(plt.Circle((cx, cy), a, **circle_kw_f))
        ax.axhline(Y_SUB, **sub_kw_f)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig_f.colorbar(im, ax=ax, shrink=0.90)

    _panel(axes_f[0, 0], U_scat_f.real,     "Scattered   Re(u_scat)",  "RdBu_r", True)
    _panel(axes_f[0, 1], U_total_f.real,    "Total       Re(u_total)", "RdBu_r", True)
    _panel(axes_f[1, 0], np.abs(U_scat_f),  "Scattered   |u_scat|",    "hot",    False)
    _panel(axes_f[1, 1], np.abs(U_total_f), "Total       |u_total|",   "inferno", False)

    buf = BytesIO()
    fig_f.savefig(buf, format='png', dpi=90)   # fixed canvas — no bbox_inches='tight'
    plt.close(fig_f)
    buf.seek(0)
    gif_frames.append(imageio.imread(buf))
    net_int.train(); net_scat.train(); net_sub.train()

# ─────────────────────────────────────────────────────────────────────────────
# Loss normalisation — compute each term's value on a fresh batch at
# initialisation and store as L0_* globals. loss_fn divides every term by its L0
# so all terms start at ~1 and the W_* weights express genuine relative
# importance. Computed once and never updated (changing L0 mid-training would
# shift the loss landscape discontinuously).
# ─────────────────────────────────────────────────────────────────────────────
# Defaults = 1.0 so the first loss_fn call (below) is well defined.
L0_PDE_INT = L0_PDE_EXT = L0_PDE_SUB = 1.0
L0_FIELD = L0_FLUX = L0_FLAT_FIELD = L0_FLAT_FLUX = 1.0
L0_PML_AIR = L0_PML_SUB = L0_PML_DIR = 1.0
L0_DATA_SCAT = L0_DATA_SUB = L0_DATA_INT = 1.0

print("Computing loss normalisation constants ...")
_xy_in_n         = sample_inside(N_INT)
_xy_air_n        = sample_air(N_AIR)
_xy_if_n, _nif_n = sample_interface(N_IFACE)
_xy_sub_n        = sample_substrate(N_SUB)
_xy_fl_n, _nfl_n = sample_flat_interface(N_FLAT)
_xy_pa_n         = sample_pml_air(N_PML)
_xy_ps_n         = sample_pml_sub(N_PML)
_xy_po_n         = sample_pml_outer(N_PML // 2)

(_, _lpi, _lpe, _lpsub, _lf, _lfl, _lflf, _lflfx,
 _lpa, _lpsb, _lpd, _lds, _ldsu, _ldi) = loss_fn(
    _xy_in_n, _xy_air_n, _xy_if_n, _nif_n, _xy_sub_n, _xy_fl_n, _nfl_n,
    _xy_pa_n, _xy_ps_n, _xy_po_n)

L0_PDE_INT    = max(_lpi.item(),   1e-12)
L0_PDE_EXT    = max(_lpe.item(),   1e-12)
L0_PDE_SUB    = max(_lpsub.item(), 1e-12)
L0_FIELD      = max(_lf.item(),    1e-12)
L0_FLUX       = max(_lfl.item(),   1e-12)
L0_FLAT_FIELD = max(_lflf.item(),  1e-12)
L0_FLAT_FLUX  = max(_lflfx.item(), 1e-12)
L0_PML_AIR    = max(_lpa.item(),   1e-12)
L0_PML_SUB    = max(_lpsb.item(),  1e-12)
L0_PML_DIR    = max(_lpd.item(),   1e-12)
if USE_FDM_DATA:
    L0_DATA_SCAT = max(_lds.item(),  1e-12)
    L0_DATA_SUB  = max(_ldsu.item(), 1e-12)
    L0_DATA_INT  = max(_ldi.item(),  1e-12)

print(f"  L0_PDE_INT    = {L0_PDE_INT:.3e}")
print(f"  L0_PDE_EXT    = {L0_PDE_EXT:.3e}")
print(f"  L0_PDE_SUB    = {L0_PDE_SUB:.3e}")
print(f"  L0_FIELD      = {L0_FIELD:.3e}")
print(f"  L0_FLUX       = {L0_FLUX:.3e}")
print(f"  L0_FLAT_FIELD = {L0_FLAT_FIELD:.3e}")
print(f"  L0_FLAT_FLUX  = {L0_FLAT_FLUX:.3e}")
print(f"  L0_PML_AIR    = {L0_PML_AIR:.3e}")
print(f"  L0_PML_SUB    = {L0_PML_SUB:.3e}")
print(f"  L0_PML_DIR    = {L0_PML_DIR:.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# relL2 monitor vs FDM (the loss is a poor proxy for solution quality — the
# tip sweep showed loss terms can be anti-correlated with relL2). Composite
# u_scat: net_scat in air, net_sub in substrate, net_int − u_bg inside the
# cylinder, matching the FDM's scattered field v everywhere.
# ─────────────────────────────────────────────────────────────────────────────
EVAL_EVERY = 250
if USE_FDM_DATA:
    _ms         = 2                       # subsample stride of the FDM grid
    MON_Xg      = FDM_X[::_ms, ::_ms]
    MON_Yg      = FDM_Y[::_ms, ::_ms]
    MON_ref     = (FDM_Scat_re + 1j*FDM_Scat_im)[::_ms, ::_ms]
    MON_in_cyl  = _fdm["in_cyl"][::_ms, ::_ms].astype(bool)
    MON_in_sub  = _fdm["in_sub"][::_ms, ::_ms].astype(bool)
    MON_xy      = torch.tensor(
        np.stack([MON_Xg.ravel(), MON_Yg.ravel()], axis=1).astype(np.float32),
        device=DEVICE)
    MON_nrm_ref = float(np.linalg.norm(MON_ref))

    def monitor_relL2():
        """(relL2 vs FDM u_scat, rms|net_scat| in air) on the monitor grid."""
        net_int.eval(); net_scat.eval(); net_sub.eval()
        with torch.no_grad():
            sR, sI = net_scat(MON_xy)
            # alias-aware: with the merged exterior this would be an identical
            # 18k-point forward pass repeated
            bR, bI = (sR, sI) if net_sub is net_scat else net_sub(MON_xy)
            iR, iI = net_int(MON_xy)
            bgR    = u_bg_air_re(MON_xy)
            bgI    = u_bg_air_im(MON_xy)
        net_int.train(); net_scat.train(); net_sub.train()
        shp    = MON_Xg.shape
        to_c   = lambda R, I: (R.cpu().numpy().reshape(shp) +
                               1j*I.cpu().numpy().reshape(shp))
        Us_air = to_c(sR, sI)
        Us_sub = to_c(bR, bI)
        Us_cyl = to_c(iR, iI) - to_c(bgR, bgI)
        Us     = np.where(MON_in_cyl, Us_cyl,
                 np.where(MON_in_sub, Us_sub, Us_air))
        rel    = float(np.linalg.norm(Us - MON_ref) / (MON_nrm_ref + 1e-12))
        amp    = float(np.sqrt(np.mean(np.abs(Us_air[~MON_in_cyl & ~MON_in_sub])**2)))
        return rel, amp

    hist_eval_ep, hist_eval_rel, hist_eval_amp = [], [], []
    print(f"\n  relL2 monitor every {EVAL_EVERY} epochs on "
          f"{MON_Xg.shape[0]}x{MON_Xg.shape[1]} FDM grid points")
    print(f"  FDM rms|u_scat| (air) target = "
          f"{np.sqrt(np.mean(np.abs(MON_ref[~MON_in_cyl & ~MON_in_sub])**2)):.4f}"
          f"   (the failed physics-only run produced ~2e-4)")

t0 = time.time()

print(f"\n--- Phase 1: Adam ({ADAM_EPOCHS} epochs; "
      f"FDM warm start for the first {WARMSTART_EPOCHS}) ---")
print(f"{'Ep':>5} | {'Total':>8} | {'PDEin':>8} | {'PDEair':>8} | {'PDEsub':>8} "
      f"| {'Field':>8} | {'Flux':>8} | {'FlatF':>8} | {'FlatX':>8} | {'PML_Air':>8} "
      f"| {'PML_Sub':>8} | {'PML_Dir':>8} | {'DatSc':>8} | {'DatSub':>8} | {'DatIn':>8} | {'s':>5}")
print("-" * 95)

for ep in range(1, ADAM_EPOCHS + 1):
    # ── warm-start schedule: full strength, then geometric decay, then exactly
    #    zero (loss_fn reads these module globals at call time; its weight
    #    guard skips all interp_fdm work once they are 0).
    if WARMSTART_EPOCHS > 0 and ep > WARMSTART_EPOCHS and W_DATA_SCAT > 0.0:
        _w = W_DATA_WARM * 0.5 ** ((ep - WARMSTART_EPOCHS) / WARMSTART_HALFLIFE)
        if _w < W_DATA_FLOOR:
            _w = 0.0
            if USE_FDM_DATA:
                _rel, _amp = monitor_relL2()
                print(f"  == data weight floored to 0 at epoch {ep}: "
                      f"relL2={_rel:.4f}, rms|u_scat|={_amp:.4f} — "
                      f"physics-only from here ==")
        W_DATA_SCAT = W_DATA_SUB = W_DATA_INT = _w

    if ep == 1 or ep % 250 == 0:
        xy_in           = sample_inside(N_INT)
        xy_air          = sample_air(N_AIR)
        xy_if, n_if     = sample_interface(N_IFACE)
        xy_sub          = sample_substrate(N_SUB)
        xy_fl, n_fl     = sample_flat_interface(N_FLAT)
        xy_pml_air      = sample_pml_air(N_PML)
        xy_pml_sub      = sample_pml_sub(N_PML)
        xy_pml_out      = sample_pml_outer(N_PML//2)

    optimizer.zero_grad()
    loss, lpi, lpe, lps, lf, lfl, lflf, lflfx, l_pml_air, l_pml_sub, l_pml_dirichlet, lds, ldsu, ldi = loss_fn(xy_in, xy_air, xy_if, n_if, xy_sub, xy_fl, n_fl, xy_pml_air, xy_pml_sub, xy_pml_out)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()
    scheduler.step()

    # One device->host transfer for all 14 terms; separate .item() calls each
    # force their own GPU sync, i.e. 14 pipeline drains per epoch.
    (loss_v, lpi_v, lpe_v, lps_v, lf_v, lfl_v, lflf_v, lflfx_v,
     lpa_v, lpsb_v, lpd_v, lds_v, ldsu_v, ldi_v) = torch.stack(
        [loss, lpi, lpe, lps, lf, lfl, lflf, lflfx,
         l_pml_air, l_pml_sub, l_pml_dirichlet, lds, ldsu, ldi]
    ).detach().cpu().tolist()
    loss_history.append(loss_v)
    hist_pde_int.append(lpi_v)
    hist_pde_air.append(lpe_v)
    hist_pde_sub.append(lps_v)
    hist_field.append(lf_v)
    hist_flux.append(lfl_v)
    hist_flat_field.append(lflf_v)
    hist_flat_flux.append(lflfx_v)
    hist_pml_air.append(lpa_v)
    hist_pml_sub.append(lpsb_v)
    hist_pml_dir.append(lpd_v)
    hist_data_scat.append(lds_v)
    hist_data_sub.append(ldsu_v)
    hist_data_int.append(ldi_v)

    if ep % 250 == 0 or ep == 1:
        print(f"{ep:>5} | {loss_v:>8.2e} | {lpi_v:>8.2e} | {lpe_v:>8.2e} | {lps_v:>8.2e} | "
              f"{lf_v:>8.2e} | {lfl_v:>8.2e} | {lflf_v:>8.2e} | {lflfx_v:>8.2e} | {lpa_v:>8.2e} | "
              f"{lpsb_v:>8.2e} | {lpd_v:>8.2e} | {lds_v:>8.2e} | {ldsu_v:>8.2e} | {ldi_v:>8.2e} | "
              f"{time.time()-t0:>5.1f}")

    if USE_FDM_DATA and (ep == 1 or ep % EVAL_EVERY == 0):
        _rel, _amp = monitor_relL2()
        hist_eval_ep.append(ep)
        hist_eval_rel.append(_rel)
        hist_eval_amp.append(_amp)
        _tag = "warm" if (WARMSTART_EPOCHS > 0 and ep <= WARMSTART_EPOCHS) \
               else "phys"
        print(f"  [eval ep {ep:>5} {_tag}]  relL2 = {_rel:.4f}   "
              f"rms|u_scat|(air) = {_amp:.4f}")

    # ── Capture animation frame every FRAME_EVERY_ADAM epochs ──────────────
    if ep == 1 or ep % FRAME_EVERY_ADAM == 0:
        save_frame(f"Adam epoch {ep}", loss_v)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: L-BFGS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n--- Phase 2: L-BFGS ({LBFGS_STEPS} closure evals) ---")
xy_in_lb         = sample_inside(N_INT)
xy_air_lb        = sample_air(N_AIR)
xy_if_lb, nif_lb = sample_interface(N_IFACE)
xy_sub_lb        = sample_substrate(N_SUB)
xy_fl_lb, nfl_lb = sample_flat_interface(N_FLAT)
xy_pml_air_lb      = sample_pml_air(N_PML)
xy_pml_sub_lb      = sample_pml_sub(N_PML)
xy_pml_out_lb      = sample_pml_outer(N_PML//2)

RESAMPLE_EVERY = 25   # in OUTER steps, matching afm_pinn_3/afm_pinn.py and
                      # afm_tip_pinn.py. At max_iter=20 that is ~500 closure
                      # evals per frozen point set. The previous 100 froze one
                      # set for ~2000 evals -- the regime tip_sweep_pinn.py
                      # documented as letting L-BFGS memorise the collocation
                      # points instead of solving the PDE (its flux term read
                      # 2.5e-2 on the frozen set and 1.8e+01 on fresh points).

last_terms = [None]   # stacked loss terms from the latest closure evaluation

def make_closure(xy_in, xy_air, xy_if, nif, xy_sub, xy_fl, nfl,
                 xy_pml_air, xy_pml_sub, xy_pml_out):
    # Clean closure: it is called MANY times per lbfgs.step() (once per inner
    # iteration plus every rejected trial point during the Wolfe line search),
    # so it must not log or sync — it only evaluates loss + gradient. It DOES
    # stash the detached term tensors (one stack, no device->host transfer);
    # the loop reads them once per outer step. This replaces the previous
    # full loss_fn re-evaluation after each lbfgs.step(), which at max_iter=20
    # spent ~1 of every 21 evaluations purely on logging.
    def closure():
        lbfgs.zero_grad()
        terms = loss_fn(
            xy_in, xy_air, xy_if, nif, xy_sub, xy_fl, nfl,
            xy_pml_air, xy_pml_sub, xy_pml_out)
        terms[0].backward()
        step_count[0] += 1          # the budget unit: ONE closure evaluation
        last_terms[0] = torch.stack([t.detach() for t in terms])
        return terms[0]
    return closure

LBFGS_KW = dict(lr=0.1, max_iter=20, history_size=50,
                line_search_fn='strong_wolfe',
                tolerance_grad=1e-12, tolerance_change=1e-16)
lbfgs = torch.optim.LBFGS(all_params, **LBFGS_KW)
step_count = [0]
outer_step = 0

# EVAL-bounded, as in afm_pinn_3 / afm_tip_pinn / tip_sweep_pinn. The previous
# `for outer in range(LBFGS_STEPS)` counted OUTER steps and incremented
# step_count once per step, so the break never fired before the loop ended --
# 20000 outer steps x max_iter=20 = up to 400,000 closure evals, each one a
# full forward + double-backward over every collocation set. Counting evals
# inside the closure makes LBFGS_STEPS mean what it means everywhere else.
while step_count[0] < LBFGS_STEPS:
    if outer_step % RESAMPLE_EVERY == 0:
        xy_in_lb         = sample_inside(N_INT)
        xy_air_lb        = sample_air(N_AIR)
        xy_if_lb, nif_lb = sample_interface(N_IFACE)
        xy_sub_lb        = sample_substrate(N_SUB)
        xy_fl_lb, nfl_lb = sample_flat_interface(N_FLAT)
        xy_pml_air_lb    = sample_pml_air(N_PML)
        xy_pml_sub_lb    = sample_pml_sub(N_PML)
        xy_pml_out_lb    = sample_pml_outer(N_PML // 2)
        lbfgs = torch.optim.LBFGS(all_params, **LBFGS_KW)

    closure = make_closure(xy_in_lb, xy_air_lb, xy_if_lb, nif_lb,
                           xy_sub_lb, xy_fl_lb, nfl_lb,
                           xy_pml_air_lb, xy_pml_sub_lb, xy_pml_out_lb)
    lbfgs.step(closure)
    outer_step += 1

    # ── Log ONCE per outer L-BFGS step, from the values the closure already
    #    computed at its last (accepted) evaluation: one device->host transfer
    #    instead of a full loss_fn re-evaluation plus 14 separate .item() syncs.
    (loss_v, lpi_v, lpe_v, lps_v, lf_v, lfl_v, lflf_v, lflfx_v,
     lpa_v, lpsb_v, lpd_v, lds_v, ldsu_v, ldi_v) = last_terms[0].cpu().tolist()
    loss_history.append(loss_v)
    hist_pde_int.append(lpi_v)
    hist_pde_air.append(lpe_v)
    hist_pde_sub.append(lps_v)
    hist_field.append(lf_v)
    hist_flux.append(lfl_v)
    hist_flat_field.append(lflf_v)
    hist_flat_flux.append(lflfx_v)
    hist_pml_air.append(lpa_v)
    hist_pml_sub.append(lpsb_v)
    hist_pml_dir.append(lpd_v)
    hist_data_scat.append(lds_v)
    hist_data_sub.append(ldsu_v)
    hist_data_int.append(ldi_v)
    if outer_step % 25 == 0 or outer_step == 1:
        print(f"{step_count[0]:>5} | {loss_v:>8.2e} | {lpi_v:>8.2e} | {lpe_v:>8.2e} | {lps_v:>8.2e} | "
              f"{lf_v:>8.2e} | {lfl_v:>8.2e} | {lflf_v:>8.2e} | {lflfx_v:>8.2e} | {lpa_v:>8.2e} | "
              f"{lpsb_v:>8.2e} | {lpd_v:>8.2e} | {lds_v:>8.2e} | {ldsu_v:>8.2e} | {ldi_v:>8.2e} | "
              f"{time.time()-t0:>5.1f}")
        if USE_FDM_DATA:
            _rel, _amp = monitor_relL2()
            hist_eval_ep.append(ADAM_EPOCHS + step_count[0])
            hist_eval_rel.append(_rel)
            hist_eval_amp.append(_amp)
            print(f"  [eval L-BFGS {step_count[0]:>5}]  relL2 = {_rel:.4f}   "
                  f"rms|u_scat|(air) = {_amp:.4f}")

    # ── Capture animation frame every 100 outer steps (and at step 1) ───────
    if outer_step == 1 or outer_step % FRAME_EVERY_LBFGS == 0:
        save_frame(f"L-BFGS {step_count[0]}", loss_v)

print(f"\nTotal training time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on dense grid
# ─────────────────────────────────────────────────────────────────────────────
N_grid = 300
xg = np.linspace(0, 1, N_grid)
yg = np.linspace(0, 1, N_grid)
Xg, Yg = np.meshgrid(xg, yg)
R2       = (Xg - cx)**2 + (Yg - cy)**2
in_cyl   = R2 <= a**2
in_sub_ev = Yg < Y_SUB

xy_ev = torch.tensor(
    np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32), device=DEVICE)

net_int.eval(); net_scat.eval(); net_sub.eval()
with torch.no_grad():
    iR_g, iI_g = net_int(xy_ev)
    sR_g, sI_g = net_scat(xy_ev)
    bR_g, bI_g = (sR_g, sI_g) if net_sub is net_scat else net_sub(xy_ev)

U_int      = (iR_g.cpu().numpy().reshape(N_grid, N_grid) +
              1j * iI_g.cpu().numpy().reshape(N_grid, N_grid))
U_scat_air = (sR_g.cpu().numpy().reshape(N_grid, N_grid) +
              1j * sI_g.cpu().numpy().reshape(N_grid, N_grid))
U_scat_sub = (bR_g.cpu().numpy().reshape(N_grid, N_grid) +
              1j * bI_g.cpu().numpy().reshape(N_grid, N_grid))

# ── Background field (Fresnel, no tip)
U_bg_air = (np.exp(1j*(kx*Xg + ky*(Yg - Y_SUB))) +
            r_TM * np.exp(1j*(kx*Xg - ky*(Yg - Y_SUB))))
U_bg_sub = t_TM * np.exp(1j*(kx*Xg + ky_sub*(Yg - Y_SUB)))

# Composite background — zero inside cylinder (no background field definition there)
U_bg = np.where(in_sub_ev, U_bg_sub, U_bg_air)

# ── Composite scattered field — tip-induced only, per region
U_scat = np.where(in_cyl,    0j,          # undefined inside tip — set to 0
          np.where(in_sub_ev, U_scat_sub,
                              U_scat_air))

# ── Composite total field
U_total = np.where(in_cyl,
              U_int,
          np.where(in_sub_ev,
              U_bg_sub + U_scat_sub,
              U_bg_air + U_scat_air))

print(f"\nField check:")
print(f"  Max |U_total|        = {np.max(np.abs(U_total)):.4f}")
print(f"  Tip   mean |U_total| = {np.mean(np.abs(U_total[in_cyl])):.4f}")
print(f"  Sub   mean |U_total| = {np.mean(np.abs(U_total[in_sub_ev & ~in_cyl])):.4f}")
print(f"  Air   mean |U_total| = {np.mean(np.abs(U_total[~in_sub_ev & ~in_cyl])):.4f}")
print(f"  Max |U_scat|         = {np.max(np.abs(U_scat)):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot — 3x3: background | scattered | total
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

circle_kw   = dict(color='cyan',   lw=1.5, fill=False, zorder=5, ls='--')
sub_line_kw = dict(color='yellow', lw=1.5, zorder=6,  ls='--')

def panel(ax, data, title, cmap, symmetric):
    valid = data[np.isfinite(data)]
    scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
    scale = max(scale, 1e-12)
    vmin  = -scale if symmetric else 0.0
    im    = ax.imshow(data, origin='lower', extent=[0,1,0,1],
                      cmap=cmap, vmin=vmin, vmax=scale, interpolation='bilinear')
    ax.add_patch(plt.Circle((cx, cy), a, **circle_kw))
    ax.axhline(Y_SUB, **sub_line_kw)
    ax.axhspan(0, Y_SUB, alpha=0.07, color='white', zorder=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.95)

# Row 0 — background (Fresnel, no tip)
panel(axes[0,0], U_bg.real,      "Background  Re(u_bg)\n[Fresnel, no tip]",   "RdBu_r", True)
panel(axes[0,1], U_bg.imag,      "Background  Im(u_bg)",                       "RdBu_r", True)
panel(axes[0,2], np.abs(U_bg),   "Background  |u_bg|",                         "viridis", False)

# Row 1 — scattered (tip-induced only)
panel(axes[1,0], U_scat.real,    "Scattered   Re(u_scat)\n[tip-induced only]", "RdBu_r", True)
panel(axes[1,1], U_scat.imag,    "Scattered   Im(u_scat)",                     "RdBu_r", True)
panel(axes[1,2], np.abs(U_scat), "Scattered   |u_scat|",                       "hot",    False)

# Row 2 — total field
panel(axes[2,0], U_total.real,    "Total       Re(u_total)",                   "RdBu_r", True)
panel(axes[2,1], U_total.imag,    "Total       Im(u_total)",                   "RdBu_r", True)
panel(axes[2,2], np.abs(U_total), "Total       |u_total|",                     "inferno", False)

handles = [
    matplotlib.patches.Circle((0,0), 0.1, ec='cyan', fc='none', ls='--', lw=1.5,
                               label='Tip boundary'),
    matplotlib.lines.Line2D([0],[0], color='yellow', ls='--', lw=1.5,
                             label=f'Substrate interface (y={Y_SUB})'),
]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 0.005))

fig.suptitle(
    f"AFM Tip + Substrate PINN  |  "
    f"n_tip={N_RE}+{N_IM}i  n_sub={SUB_N_RE}+{SUB_N_IM}i  "
    f"theta={np.degrees(theta_inc):.0f} deg\n"
    f"u_scat = tip-induced only (Fresnel background subtracted)",
    fontsize=11, y=1.01)

plt.tight_layout(rect=[0, 0.03, 1, 1])
os.makedirs("sample/figures", exist_ok=True)
plt.savefig("sample/figures/pinn_afm_fields.png", dpi=150, bbox_inches='tight')
print("Saved: sample/figures/pinn_afm_fields.png")


# Training loss
fig2, ax2 = plt.subplots(figsize=(11, 5))
steps = range(len(loss_history))
ax2.semilogy(steps, loss_history,    color='black',      lw=1.5, label='Total')
ax2.semilogy(steps, hist_pde_int,    color='royalblue',  lw=0.8, label='PDE int')
ax2.semilogy(steps, hist_pde_air,    color='dodgerblue',  lw=0.8, label='PDE air')
ax2.semilogy(steps, hist_pde_sub,    color='cyan',  lw=0.8, label='PDE sub')
ax2.semilogy(steps, hist_field,      color='yellow',     lw=0.8, label='Field BC')
ax2.semilogy(steps, hist_flux,       color='orange',     lw=0.8, label='Flux BC')
ax2.semilogy(steps, hist_flat_field, color='seagreen',   lw=0.8, label='Flat field BC')
ax2.semilogy(steps, hist_flat_flux,  color='olive',      lw=0.8, label='Flat flux BC')
ax2.semilogy(steps, hist_pml_air,    color='red', lw=0.8, label='PML air')
ax2.semilogy(steps, hist_pml_sub,    color='lightcoral', lw=0.8, label='PML sub')
ax2.semilogy(steps, hist_pml_dir,    color='maroon', lw=0.8, label='PML dirichlet')
ax2.semilogy(steps, hist_data_scat,  color='blueviolet',   lw=0.8, label='Data scat')
ax2.semilogy(steps, hist_data_sub,   color='magenta',       lw=0.8, label='Data sub')
ax2.semilogy(steps, hist_data_int,   color='deeppink',       lw=0.8, label='Data int')
ax2.axvline(ADAM_EPOCHS, color='gray', ls='--', lw=1.5,
            label=f'Adam → L-BFGS (step {ADAM_EPOCHS})')
ax2.set_xlabel("Optimiser step")
ax2.set_ylabel("Loss (log scale)")
ax2.set_title("Dielectric Cylinder PINN — Training Loss by Term")
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sample/figures/afm_pinn_loss.png", dpi=150, bbox_inches='tight')
print("Saved: sample/figures/afm_pinn_loss.png")

# ── relL2 convergence (the metric, not the loss) ─────────────────────────────
if USE_FDM_DATA and hist_eval_ep:
    fig_ev, ax_ev = plt.subplots(figsize=(11, 5))
    ax_ev.semilogy(hist_eval_ep, hist_eval_rel, 'o-', color='crimson',
                   ms=4, lw=1.5, label='relL2 vs FDM u_scat')
    if WARMSTART_EPOCHS > 0:
        ax_ev.axvline(WARMSTART_EPOCHS, color='gray', ls='--', lw=1.2,
                      label=f'warm start ends (ep {WARMSTART_EPOCHS})')
    ax_ev.axvline(ADAM_EPOCHS, color='black', ls=':', lw=1.2,
                  label=f'Adam → L-BFGS (ep {ADAM_EPOCHS})')
    ax_ev.set_xlabel("epoch (L-BFGS closure evals offset by ADAM_EPOCHS)")
    ax_ev.set_ylabel("relative L2 error vs FDM")
    ax_ev.grid(True, alpha=0.3)
    ax_ev2 = ax_ev.twinx()
    ax_ev2.plot(hist_eval_ep, hist_eval_amp, 's--', color='steelblue',
                ms=4, lw=1.2, label='rms|u_scat| (air)')
    ax_ev2.set_ylabel("rms|u_scat| in air   (watch for decay toward 0 = "
                      "re-collapse)", color='steelblue')
    ax_ev2.tick_params(axis='y', labelcolor='steelblue')
    _h1, _l1 = ax_ev.get_legend_handles_labels()
    _h2, _l2 = ax_ev2.get_legend_handles_labels()
    ax_ev.legend(_h1 + _h2, _l1 + _l2, loc='best', fontsize=9)
    ax_ev.set_title("AFM tip + substrate PINN — relL2 convergence")
    plt.tight_layout()
    plt.savefig("sample/figures/afm_pinn_relL2.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig_ev)
    print("Saved: sample/figures/afm_pinn_relL2.png")
    print(f"  relL2: {hist_eval_rel[0]:.4f} (start) -> {hist_eval_rel[-1]:.4f} (end)")

# plt.show()
print("\nDone.")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble MP4 from captured frames (plays natively in PowerPoint / browsers)
# ─────────────────────────────────────────────────────────────────────────────
if gif_frames:
    mp4_path = "sample/figures/training_animation.mp4"
    fps = 6
    hold_frames = fps * 3                       # hold the final frame for ~3 s
    frames_out  = gif_frames + [gif_frames[-1]] * hold_frames
    writer_kw   = dict(fps=fps, codec="libx264", pixelformat="yuv420p",
                       output_params=["-crf", "18"])   # high quality, PPT-compatible
    imageio.mimwrite(mp4_path, frames_out, format="mp4", **writer_kw)
    print(f"Saved training animation ({len(gif_frames)} frames + {hold_frames} hold): {mp4_path}")