"""
FDM Sweep — 2D Dielectric AFM-Tip Scattering (TM mode), HALF-ANGLE sweep
=========================================================================
Shape-parametric extension of afm_tip_fdm.py, structured exactly like
fdm_ellipse_sweep.py.

The scatterer is the conospherical AFM tip of afm_tip_fdm.py — a cone of
half-angle alpha opening upward, apex pointing down, capped by a circle of
radius R_APEX, shank running straight out through the top boundary and the
top PML. The sweep iterates over

    N_ALPHA = 20 half-angles  alpha ∈ [10°, 30°]

and solves the scattered-field UPML system once per angle. Everything else
(scattered-field formulation, UPML stencil, PML profile, grid, Dirichlet
outer boundary, material n = 3 + 30i) is identical to afm_tip_fdm.py. Only
the interior mask in_tip(alpha) and the RHS source region change — the PML
stretch factors and off-diagonal stencil coefficients are geometry-
independent, so they are precomputed once outside the sweep loop
(vectorised assembly, as in fdm_ellipse_sweep.py).

TIP CROSSING THE TOP PML
------------------------
As in afm_tip_fdm.py, the mask is built from the signed distance function
on the FULL grid (physical + PML), so the shank material and contrast
source automatically extend into the top PML for every angle. The straight
y-invariant shank is exactly what a y-stretched PML terminates cleanly.

OUTPUT
------
inputs/fdm_afm_tip_sweep.npz with:
    half_angle           : [N_ALPHA]            cone half-angles (radians)
    U_scat_re/im         : [N_ALPHA, Nph, Nph]  scattered field (physical dom.)
    U_total_re/im        : [N_ALPHA, Nph, Nph]  total field
    in_tip               : [N_ALPHA, Nph, Nph]  interior masks
    X, Y                 : [Nph, Nph]           physical-domain grid
    train_idx, test_idx  : parameter-index split (test = interpolation angles)
    meta                 : [N_RE, N_IM, lam, k, TIP_X, APEX_Y, R_APEX]

Run this before afm_tip_pinn_parametric.py.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — material/wave identical to afm_tip_fdm.py / afm_tip_pinn.py
# ─────────────────────────────────────────────────────────────────────────────
N_RE   = 3.0
N_IM   = 30.0
lam    = 0.5
k      = 2.0 * np.pi / lam
theta_inc = -np.pi / 4
kx = k * np.cos(theta_inc)
ky = k * np.sin(theta_inc)

# ── AFM conospherical tip geometry (fixed except the half-angle) ─────────────
TIP_X   = 0.5                 # x-coordinate of the tip axis
APEX_Y  = 0.45                # y-coordinate of the rounded apex
R_APEX  = 0.05                # apex radius of curvature

N = 401   # grid points over the FULL (physical + PML) domain

# ── Half-angle sweep ─────────────────────────────────────────────────────────
N_ALPHA      = 20
ALPHA_DEG    = np.linspace(10.0, 30.0, N_ALPHA)
half_angle   = np.deg2rad(ALPHA_DEG)          # [N_ALPHA], radians

# ── Train/test split ─────────────────────────────────────────────────────────
# Test angles drawn only from the interior indices 1..N_ALPHA-2, so every
# test case is an INTERPOLATION in the half-angle (both extreme angles stay
# in training). Same philosophy as fdm_ellipse_sweep.py.
rng       = np.random.default_rng(0)
interior  = np.arange(1, N_ALPHA - 1)
test_idx  = np.sort(rng.choice(interior, size=4, replace=False))
train_idx = np.array([p for p in range(N_ALPHA) if p not in set(test_idx)])

# ── PML parameters — identical to afm_tip_fdm.py ─────────────────────────────
d_pml     = 0.5
sigma_max = 20.0
pml_order = 3

# ─────────────────────────────────────────────────────────────────────────────
# Signed distance function for the conospherical AFM tip, parametrised by the
# cone half-angle alpha. Same construction as afm_tip_fdm.py:
#   apex_centre_y = APEX_Y + R_APEX
#   cone_vertex_y = APEX_Y + R_APEX - R_APEX / sin(alpha)
#   tangent_y     = apex_centre_y - R_APEX * sin(alpha)
# ─────────────────────────────────────────────────────────────────────────────
_APEX_CENTRE_Y = APEX_Y + R_APEX


def _alpha_consts(alpha):
    sin_a = float(np.sin(alpha))
    cos_a = float(np.cos(alpha))
    cone_vertex_y = APEX_Y + R_APEX - R_APEX / sin_a
    tangent_y     = _APEX_CENTRE_Y - R_APEX * sin_a
    return sin_a, cos_a, cone_vertex_y, tangent_y


def phi_np(x, y, alpha):
    """Signed distance to the conospherical tip with half-angle alpha.
    Negative inside, positive outside, ~0 on the surface."""
    sin_a, cos_a, cone_vertex_y, tangent_y = _alpha_consts(alpha)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q_x    = np.abs(x - TIP_X)
    d_cone = q_x * cos_a - (y - cone_vertex_y) * sin_a
    d_apex = np.sqrt((x - TIP_X)**2 + (y - _APEX_CENTRE_Y)**2) - R_APEX
    return np.where(y < tangent_y, d_apex, d_cone)


def tip_outline_xy(alpha, y_top=1.0, n=200):
    """(x, y) polyline tracing the tip surface for plot overlays."""
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
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────
k_int    = (N_RE + 1j * N_IM) * k
k_int_sq = k_int**2
k_ext_sq = k**2
delta_k2 = k_int_sq - k_ext_sq

print("=" * 66)
print("FDM SWEEP — AFM Tip half-angle (TM, scattered-field + UPML)")
print("=" * 66)
print(f"  n            = {N_RE} + {N_IM}i   (matches afm_tip_pinn.py)")
print(f"  lam / k      = {lam} / {k:.4f}")
print(f"  tip          : apex=({TIP_X},{APEX_Y}), R={R_APEX}, kR={k*R_APEX:.3f}")
print(f"  half-angles  = {ALPHA_DEG[0]:.1f}° .. {ALPHA_DEG[-1]:.1f}°  "
      f"({N_ALPHA} values)")
print(f"  combos       = {N_ALPHA}  (train {len(train_idx)} / test {len(test_idx)})")
print(f"  test angles  = {[f'{ALPHA_DEG[p]:.2f}°' for p in test_idx]}")
print(f"  PML: d={d_pml}, sigma_max={sigma_max/k:.1f}k, order={pml_order}")

# ─────────────────────────────────────────────────────────────────────────────
# Grid — identical to afm_tip_fdm.py
# ─────────────────────────────────────────────────────────────────────────────
X_LO, X_HI = 0.0, 1.0
Y_LO, Y_HI = 0.0, 1.0

x_full_lo = X_LO - d_pml;  x_full_hi = X_HI + d_pml
y_full_lo = Y_LO - d_pml;  y_full_hi = Y_HI + d_pml

x_coords = np.linspace(x_full_lo, x_full_hi, N)
y_coords = np.linspace(y_full_lo, y_full_hi, N)
hx = abs((x_full_hi - x_full_lo) / (N - 1))
hy = abs((y_full_hi - y_full_lo) / (N - 1))

Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing='ij')   # [N, N]

print(f"  Grid {N}×{N}, h={hx:.4f}  →  {R_APEX/hx:.1f} cells across the apex "
      f"radius (increase N if too coarse)")
print("=" * 66)

U_inc_grid = np.exp(1j * (kx * X_grid + ky * Y_grid))

# ─────────────────────────────────────────────────────────────────────────────
# PML stretch factors — geometry-independent, computed ONCE
# ─────────────────────────────────────────────────────────────────────────────
def pml_sigma(coord, lo, hi):
    s = np.zeros(len(coord))
    mask_lo = coord < lo
    mask_hi = coord > hi
    s[mask_lo] = sigma_max * ((lo - coord[mask_lo]) / d_pml) ** pml_order
    s[mask_hi] = sigma_max * ((coord[mask_hi] - hi) / d_pml) ** pml_order
    return s

sigma_x_arr = pml_sigma(x_coords, X_LO, X_HI)
sigma_y_arr = pml_sigma(y_coords, Y_LO, Y_HI)

SX = (1.0 + 1j * sigma_x_arr[np.newaxis, :] / k) * np.ones((N, N), dtype=complex)
SY = (1.0 + 1j * sigma_y_arr[:, np.newaxis] / k) * np.ones((N, N), dtype=complex)

# ─────────────────────────────────────────────────────────────────────────────
# Precompute the geometry-INDEPENDENT part of the UPML stencil
#   (s_y/s_x) h_x⁻² ∂²v/∂x²  +  (s_x/s_y) h_y⁻² ∂²v/∂y²  +  k²(i,j) s_x s_y v
# Off-diagonal coefficients c_x, c_y and the -2(c_x+c_y) part of the diagonal
# never change across the sweep; only k²(i,j) s_x s_y and the RHS do.
# ─────────────────────────────────────────────────────────────────────────────
inv_hx2 = 1.0 / hx**2
inv_hy2 = 1.0 / hy**2
n_dof   = N * N

C_X  = (SY / SX) * inv_hx2
C_Y  = (SX / SY) * inv_hy2
SXSY = SX * SY

IDX = np.arange(n_dof).reshape(N, N)          # flat(i,j) = i*N + j
interior_mask = np.zeros((N, N), dtype=bool)
interior_mask[1:-1, 1:-1] = True
I_int = IDX[interior_mask]
I_bnd = IDX[~interior_mask]

cx_int   = C_X[interior_mask]
cy_int   = C_Y[interior_mask]
sxsy_int = SXSY[interior_mask]

rows_static = np.concatenate([I_int] * 5 + [I_bnd])
cols_static = np.concatenate([I_int + 1, I_int - 1,          # j±1 → x neighbours
                              I_int + N, I_int - N,          # i±1 → y neighbours
                              I_int,                          # diagonal
                              I_bnd])                         # boundary identity


def assemble_and_solve(in_tip):
    """Assemble the UPML scattered-field system for a given tip mask and
    solve. Identical stencil to afm_tip_fdm.py, vectorised."""
    k2_int_flat = np.where(in_tip, k_int_sq, k_ext_sq)[interior_mask]
    diag = -2.0 * (cx_int + cy_int) + k2_int_flat * sxsy_int

    vals = np.concatenate([cx_int, cx_int, cy_int, cy_int, diag,
                           np.ones(len(I_bnd), dtype=complex)])
    A = coo_matrix((vals, (rows_static, cols_static)),
                   shape=(n_dof, n_dof)).tocsc()

    rhs = np.zeros(n_dof, dtype=complex)
    src = in_tip & interior_mask
    rhs[IDX[src]] = -delta_k2 * SXSY[src] * U_inc_grid[src]

    return spsolve(A, rhs).reshape(N, N)


# ─────────────────────────────────────────────────────────────────────────────
# Physical-subdomain extraction indices
# ─────────────────────────────────────────────────────────────────────────────
j_lo = np.searchsorted(x_coords, X_LO - hx / 2)
j_hi = np.searchsorted(x_coords, X_HI + hx / 2)
i_lo = np.searchsorted(y_coords, Y_LO - hy / 2)
i_hi = np.searchsorted(y_coords, Y_HI + hy / 2)

X_ph = X_grid[i_lo:i_hi, j_lo:j_hi]
Y_ph = Y_grid[i_lo:i_hi, j_lo:j_hi]
N_ph = X_ph.shape[0]
U_inc_ph = np.exp(1j * (kx * X_ph + ky * Y_ph))

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
U_scat_re_all  = np.zeros((N_ALPHA, N_ph, N_ph), dtype=np.float32)
U_scat_im_all  = np.zeros((N_ALPHA, N_ph, N_ph), dtype=np.float32)
U_total_re_all = np.zeros((N_ALPHA, N_ph, N_ph), dtype=np.float32)
U_total_im_all = np.zeros((N_ALPHA, N_ph, N_ph), dtype=np.float32)
in_tip_all     = np.zeros((N_ALPHA, N_ph, N_ph), dtype=bool)

t0 = time.time()
for p in range(N_ALPHA):
    alpha = float(half_angle[p])

    # Tip material mask on the FULL grid (physical + PML) — the shank
    # automatically extends into the top PML, as in afm_tip_fdm.py.
    in_tip = phi_np(X_grid, Y_grid, alpha) < 0.0

    t1 = time.time()
    V_full  = assemble_and_solve(in_tip)
    U_scat  = V_full[i_lo:i_hi, j_lo:j_hi]
    U_total = U_scat + U_inc_ph
    in_tip_ph = in_tip[i_lo:i_hi, j_lo:j_hi]

    U_scat_re_all[p]  = U_scat.real;   U_scat_im_all[p]  = U_scat.imag
    U_total_re_all[p] = U_total.real;  U_total_im_all[p] = U_total.imag
    in_tip_all[p]     = in_tip_ph

    tag = "TEST " if p in set(test_idx) else "train"
    print(f"  [{p+1:>2}/{N_ALPHA}] {tag}  alpha={ALPHA_DEG[p]:5.2f}°  "
          f"max|u_scat|={np.max(np.abs(U_scat)):.3f}  "
          f"n_interior={in_tip_ph.sum():>6}  ({time.time()-t1:.1f}s)")

print(f"\nSweep done in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("variable_tips/inputs", exist_ok=True)
np.savez_compressed(
    "variable_tips/inputs/fdm_afm_tip_sweep.npz",
    half_angle=half_angle.astype(np.float32),
    U_scat_re=U_scat_re_all,   U_scat_im=U_scat_im_all,
    U_total_re=U_total_re_all, U_total_im=U_total_im_all,
    in_tip=in_tip_all,
    X=X_ph.astype(np.float32), Y=Y_ph.astype(np.float32),
    train_idx=train_idx, test_idx=test_idx,
    meta=np.array([N_RE, N_IM, lam, k, TIP_X, APEX_Y, R_APEX]))
print("Saved: variable_tips/inputs/fdm_afm_tip_sweep.npz")

# ─────────────────────────────────────────────────────────────────────────────
# Overview figures — four 4×5 grids over the half-angle sweep:
#   Re(u_scat), |u_scat|, Re(u_total), |u_total|
# Panels ordered by increasing half-angle (left→right, top→bottom).
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("variable_tips/figures", exist_ok=True)
matplotlib.rcParams.update({'font.size': 7})
N_ROWS, N_COLS = 4, 5
outline_kw = dict(color='cyan', lw=1.0, zorder=5, ls='--')


def plot_sweep_grid(get_data, title, cmap, symmetric, fname):
    """4×5 panel grid over all half-angles.
    get_data(p) returns the [N_ph, N_ph] field to show for combo p."""
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(3 * N_COLS, 3 * N_ROWS))
    for p in range(N_ALPHA):
        ax_plot = axes[p // N_COLS, p % N_COLS]
        data  = get_data(p)
        scale = max(float(np.percentile(np.abs(data), 98)), 1e-12)
        vmin  = -scale if symmetric else 0.0
        ax_plot.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                       cmap=cmap, vmin=vmin, vmax=scale)
        ox, oy = tip_outline_xy(float(half_angle[p]))
        ax_plot.plot(ox, oy, **outline_kw)
        tag = " [TEST]" if p in set(test_idx) else ""
        ax_plot.set_title(f"α={ALPHA_DEG[p]:.2f}°{tag}", fontsize=7)
        ax_plot.set_xticks([]); ax_plot.set_yticks([])
        ax_plot.set_xlim(0, 1); ax_plot.set_ylim(0, 1)
    fig.suptitle(f"FDM AFM-tip half-angle sweep — {title},  "
                 f"n={N_RE}+{N_IM}i,  λ={lam}", fontsize=11)
    plt.tight_layout()
    path = f"variable_tips/figures/{fname}.png"
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


plot_sweep_grid(lambda p: U_scat_re_all[p],
                "Re(u_scat)", "RdBu_r", True,
                "fdm_afm_tip_sweep_scat_re")
plot_sweep_grid(lambda p: np.hypot(U_scat_re_all[p], U_scat_im_all[p]),
                "|u_scat|", "hot", False,
                "fdm_afm_tip_sweep_scat_mag")
plot_sweep_grid(lambda p: U_total_re_all[p],
                "Re(u_total)", "RdBu_r", True,
                "fdm_afm_tip_sweep_total_re")
plot_sweep_grid(lambda p: np.hypot(U_total_re_all[p], U_total_im_all[p]),
                "|u_total|", "inferno", False,
                "fdm_afm_tip_sweep_total_mag")

print("\nDone.")