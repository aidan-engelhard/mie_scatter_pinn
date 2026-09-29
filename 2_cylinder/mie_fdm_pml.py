"""
FDM Reference — 2D Dielectric Cylinder Scattering (TM mode)
============================================================
Computes the exact (up to discretisation error) solution to the same
problem as the dielectric_pinn.py script, using a sparse direct solve.

SCATTERED-FIELD FORMULATION WITH UPML
--------------------------------------
Solves directly for v = u_scat = u_total - u_inc.

Substituting u_total = v + u_inc into the Helmholtz equation:
    (∇² + k²(x,y)) (v + u_inc) = 0

Since u_inc satisfies (∇² + k_ext²) u_inc = 0 everywhere, this gives:
    (∇² + k²(x,y)) v = -(k²(x,y) - k_ext²) u_inc

The RHS source is nonzero ONLY inside the cylinder where k² = k_int² ≠ k_ext²:
    source(i,j) = -(k_int² - k_ext²) * u_inc(i,j)   inside cylinder
    source(i,j) = 0                                    everywhere else

This is exact. u_inc appears only on the RHS so the PML only ever sees
the outgoing scattered field v, which it can absorb cleanly.

UPML STENCIL
-------------
The coordinate-stretched Helmholtz equation in the UPML formulation is:

    (s_y/s_x²) ∂²v/∂x²  +  (s_x/s_y²) ∂²v/∂y²  +  k²(i,j) s_x s_y v = rhs

where s_x(x) = 1 + i σ_x(x)/k, s_y(y) = 1 + i σ_y(y)/k are the complex
stretching factors. The cross-factors s_y/s_x² and s_x/s_y² are the correct
UPML tensor form (cf. Gedney 1996), accounting for the Jacobian of the
stretch in both directions simultaneously.

On the physical domain s_x = s_y = 1 and we recover the standard stencil.
Outer boundary: v = 0 (Dirichlet) — field is fully absorbed before reaching it.

PARAMETERS (keep identical to dielectric_pinn.py)
--------------------------------------------------
  N_RE, N_IM : refractive index
  k, a, cx, cy : wavenumber, radius, centre
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — keep identical to dielectric_pinn.py
# ─────────────────────────────────────────────────────────────────────────────
N_RE   = 3.0
N_IM   = 30.0
lam    = 0.5
k      = 2.0 * np.pi / lam
a      = 0.05
cx, cy = 0.5, 0.5
theta_inc = -np.pi / 4
kx = k * np.cos(theta_inc)
ky = k * np.sin(theta_inc)

N = 401   # grid points over the PHYSICAL domain [0,1]²

# ── PML parameters ────────────────────────────────────────────────────────────
d_pml     = 0.5         # PML thickness in physical units (~0.5 wavelengths at k=2π)
sigma_max = 20.0      # peak conductivity  (same convention as mie_fdm_sample)
pml_order = 3            # polynomial grading exponent

# ─────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────
k_int    = (N_RE + 1j * N_IM) * k
k_int_sq = k_int**2
k_ext_sq = k**2
delta_k2 = k_int_sq - k_ext_sq   # source contrast, nonzero inside cylinder only

print("=" * 62)
print("FDM — Dielectric Cylinder (TM, scattered-field + UPML)")
print("=" * 62)
print(f"  n         = {N_RE} + {N_IM}i")
print(f"  k_ext     = {k:.4f}")
print(f"  k_int     = {k_int:.4f}")
print(f"  delta_k²  = {delta_k2:.4f}  (source contrast)")
print(f"  a         = {a},   ka = {k*a:.3f}")
print(f"  PML: d={d_pml}, sigma_max={sigma_max/k:.1f}k, order={pml_order}")
print(f"  Physical  = {N}×{N}")

# ─────────────────────────────────────────────────────────────────────────────
# Grid — uniform over the full domain including PML on all sides
# ─────────────────────────────────────────────────────────────────────────────
X_LO, X_HI = 0.0, 1.0
Y_LO, Y_HI = 0.0, 1.0

x_full_lo = X_LO - d_pml;  x_full_hi = X_HI + d_pml
y_full_lo = Y_LO - d_pml;  y_full_hi = Y_HI + d_pml

x_coords = np.linspace(x_full_lo, x_full_hi, N)
y_coords = np.linspace(y_full_lo, y_full_hi, N)
hx = (x_full_hi - x_full_lo) / (N - 1)
hy = (y_full_lo - y_full_hi) / (N - 1)   # negative — y increases upward; use abs below
hx = abs(hx); hy = abs(hy)

Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing='ij')   # [N, N]

R2     = (X_grid - cx)**2 + (Y_grid - cy)**2
in_cyl = R2 <= a**2

print(f"  Full grid = {N}×{N}  over "
      f"[{x_full_lo:.2f},{x_full_hi:.2f}]×[{y_full_lo:.2f},{y_full_hi:.2f}]")
print("=" * 62)

# ─────────────────────────────────────────────────────────────────────────────
# Incident field on the full grid (RHS source only — never enters the stencil)
# ─────────────────────────────────────────────────────────────────────────────
U_inc_grid = np.exp(1j * (kx * X_grid + ky * Y_grid))

# ─────────────────────────────────────────────────────────────────────────────
# k²(x,y) on the full grid
# ─────────────────────────────────────────────────────────────────────────────
k2 = np.where(in_cyl, k_int_sq, k_ext_sq).astype(complex)

# ─────────────────────────────────────────────────────────────────────────────
# PML stretch factors  s_x(x), s_y(y)
# σ(ξ) = sigma_max * (depth/d_pml)^pml_order,  depth = distance into PML
# s(ξ) = 1 + i σ(ξ) / k
# ─────────────────────────────────────────────────────────────────────────────
def pml_sigma(coord, lo, hi):
    """Polynomial PML conductivity profile, zero inside [lo, hi]."""
    s = np.zeros(len(coord))
    mask_lo = coord < lo
    mask_hi = coord > hi
    s[mask_lo] = sigma_max * ((lo - coord[mask_lo]) / d_pml) ** pml_order
    s[mask_hi] = sigma_max * ((coord[mask_hi] - hi) / d_pml) ** pml_order
    return s

sigma_x_arr = pml_sigma(x_coords, X_LO, X_HI)   # shape [N]
sigma_y_arr = pml_sigma(y_coords, Y_LO, Y_HI)   # shape [N]

# s_x[i,j] depends only on x_coords[j]; s_y[i,j] depends only on y_coords[i]
SX = (1.0 + 1j * sigma_x_arr[np.newaxis, :] / k) * np.ones((N, N), dtype=complex)
SY = (1.0 + 1j * sigma_y_arr[:, np.newaxis] / k) * np.ones((N, N), dtype=complex)

print(f"\nPML check:")
print(f"  s_x range: [{SX.real.min():.3f},{SX.real.max():.3f}] + "
      f"i[{SX.imag.min():.3f},{SX.imag.max():.3f}]")
print(f"  s_y range: [{SY.real.min():.3f},{SY.real.max():.3f}] + "
      f"i[{SY.imag.min():.3f},{SY.imag.max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble scattered-field system
#
# UPML-stretched equation for v = u_scat:
#   (s_y/s_x²) ∂²v/∂x²  +  (s_x/s_y²) ∂²v/∂y²  +  k²(i,j) s_x s_y v
#       = -(k²(i,j) - k_ext²) * s_x * s_y * u_inc(i,j)
#
# Second derivatives discretised with 2nd-order centred differences.
# Outer boundary: v = 0 (Dirichlet).
# ─────────────────────────────────────────────────────────────────────────────
inv_hx2 = 1.0 / hx**2
inv_hy2 = 1.0 / hy**2
n_dof   = N * N

def flat(i, j): return i * N + j

print(f"\nAssembling {n_dof:,}-DOF scattered-field system ...")
t0 = time.time()

rows_l, cols_l, vals_l = [], [], []
rhs = np.zeros(n_dof, dtype=complex)

for i in range(N):
    for j in range(N):
        I = flat(i, j)

        # Outer Dirichlet: v = 0 (PML absorbs everything before here)
        if i == 0 or i == N - 1 or j == 0 or j == N - 1:
            rows_l.append(I); cols_l.append(I); vals_l.append(1.0)
            rhs[I] = 0.0
            continue

        sx = SX[i, j]
        sy = SY[i, j]

        c_x = (sy / sx) * inv_hx2
        c_y = (sx / sy) * inv_hy2
        c_d = -2.0 * (c_x + c_y) + k2[i, j] * sx * sy

        rows_l += [I,            I,            I,            I,            I]
        cols_l += [flat(i, j+1), flat(i, j-1), flat(i+1, j), flat(i-1, j), I]
        vals_l += [c_x,          c_x,          c_y,          c_y,          c_d]

        # Source term: only inside the cylinder
        if in_cyl[i, j]:
            rhs[I] = -(k_int_sq - k_ext_sq) * sx * sy * U_inc_grid[i, j]
        # else rhs[I] = 0.0 (already initialised)

A_mat = coo_matrix((vals_l, (rows_l, cols_l)), shape=(n_dof, n_dof)).tocsc()
print(f"  Built in {time.time()-t0:.2f}s  (nnz = {A_mat.nnz:,})")

# ─────────────────────────────────────────────────────────────────────────────
# Solve for v = u_scat
# ─────────────────────────────────────────────────────────────────────────────
print("Solving ...")
t1 = time.time()
v_flat = spsolve(A_mat, rhs)
print(f"  Solved in {time.time()-t1:.2f}s")

V_full = v_flat.reshape(N, N)   # u_scat over the full (physical + PML) grid

# ─────────────────────────────────────────────────────────────────────────────
# Extract physical subdomain [X_LO, X_HI] × [Y_LO, Y_HI]
# ─────────────────────────────────────────────────────────────────────────────
j_lo = np.searchsorted(x_coords, X_LO - hx / 2)
j_hi = np.searchsorted(x_coords, X_HI + hx / 2)
i_lo = np.searchsorted(y_coords, Y_LO - hy / 2)
i_hi = np.searchsorted(y_coords, Y_HI + hy / 2)

U_scat    = V_full   [i_lo:i_hi, j_lo:j_hi]
X_ph      = X_grid   [i_lo:i_hi, j_lo:j_hi]
Y_ph      = Y_grid   [i_lo:i_hi, j_lo:j_hi]
in_cyl_ph = in_cyl   [i_lo:i_hi, j_lo:j_hi]

U_inc   = np.exp(1j * (kx * X_ph + ky * Y_ph))
U_total = U_scat + U_inc

print(f"\nField check (physical domain):")
print(f"  Max |U_total|          = {np.max(np.abs(U_total)):.4f}")
print(f"  Inside  mean |U_total| = {np.mean(np.abs(U_total[in_cyl_ph])):.4f}")
print(f"  Outside mean |U_total| = {np.mean(np.abs(U_total[~in_cyl_ph])):.4f}")
print(f"  Max |U_scat|           = {np.max(np.abs(U_scat)):.4f}")
print(f"  Scat/Total ratio       = {np.max(np.abs(U_scat))/np.max(np.abs(U_total)):.3f}")
print(f"  Total time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save — same interface as original mie_fdm.py
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("inputs", exist_ok=True)
np.savez("inputs/fdm_solution.npz",
         X=X_ph, Y=Y_ph,
         U_total_re=U_total.real, U_total_im=U_total.imag,
         U_scat_re=U_scat.real,   U_scat_im=U_scat.imag,
         U_inc_re=U_inc.real,     U_inc_im=U_inc.imag,
         in_cyl=in_cyl_ph,
         params=np.array([N_RE, N_IM, k, a, cx, cy]))
print("Saved FDM solution: inputs/fdm_solution.npz")

# ─────────────────────────────────────────────────────────────────────────────
# Plot — same 2×3 layout as original
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

circle_kw = dict(color='cyan', lw=1.5, fill=False, zorder=5, ls='--')

def panel(ax, data, title, cmap, symmetric):
    valid = data[np.isfinite(data)]
    scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
    scale = max(scale, 1e-12)
    vmin  = -scale if symmetric else 0.0
    im    = ax.pcolormesh(X_ph, Y_ph, data, shading='auto',
                          cmap=cmap, vmin=vmin, vmax=scale)
    ax.add_patch(plt.Circle((cx, cy), a, **circle_kw))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, shrink=0.95)

panel(axes[0,0], U_inc.real,      "Incident    Re(u_inc)",   "RdBu_r", True)
panel(axes[0,1], U_scat.real,     "Scattered   Re(u_scat)",  "RdBu_r", True)
panel(axes[0,2], U_total.real,    "Total       Re(u_total)", "RdBu_r", True)
panel(axes[1,0], np.abs(U_inc),   "Incident    |u_inc|",     "viridis", False)
panel(axes[1,1], np.abs(U_scat),  "Scattered   |u_scat|",    "hot",    False)
panel(axes[1,2], np.abs(U_total), "Total       |u_total|",   "inferno", False)

plt.suptitle(
    f"FDM — Scattered-field + UPML  "
    f"(N={N}, d_pml={d_pml}, σ_max={sigma_max/k:.1f}k)",
    fontsize=13)
plt.tight_layout()
os.makedirs("figures/mie_2d", exist_ok=True)
plt.savefig("figures/mie_2d/fdm_fields_pml.png", dpi=150, bbox_inches='tight')
print("Saved: figures/mie_2d/fdm_fields_pml.png")

# plt.show()
print("\nDone.")