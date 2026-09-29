"""
FDM — AFM Tip + Substrate, Scattered-Field Formulation with Medium-Aware PML
=============================================================================

SCATTERED-FIELD FORMULATION
----------------------------
Instead of solving for u_total directly (total-field formulation), we solve
for the scattered field v = u_scat = u_total - u_bg explicitly.

Substituting u_total = v + u_bg into the Helmholtz equation:
    (∇² + k²) (v + u_bg) = 0

Since u_bg satisfies (∇² + k²_bg) u_bg = 0 where k²_bg is the background k²
(no tip), we get:
    (∇² + k²) v = -(k² - k²_bg) u_bg

The right-hand side is nonzero ONLY inside the cylinder, where k² = k_int²
but k²_bg = k_ext² (air). Everywhere else k² = k²_bg so the source is zero.

    source(i,j) = -(k_int² - k_ext²) * u_bg(i,j)   inside cylinder
    source(i,j) = 0                                   everywhere else

This is exact — no discretisation consistency issue because u_bg appears
only on the RHS, not in the stencil. The scattered field v is driven purely
by the tip's dielectric contrast, which is what we want physically.

BOUNDARY CONDITIONS
-------------------
On the outer PML edge: v = 0 (Dirichlet).
The PML attenuates v to zero before it reaches the boundary, so this is
clean with no approximation needed.

PML
---
Divergence-form UPML, identical in physics to mie_pinn_sample.py:

    ∂/∂x [ (s_y/s_x) ∂v/∂x ] + ∂/∂y [ (s_x/s_y) ∂v/∂y ] + k² s_x s_y v = RHS

discretised with a CONSERVATIVE flux stencil (coefficients evaluated at cell
faces via arithmetic averaging). This is the finite-difference equivalent of
the PINN's divergence-form residual, and the face differencing captures the
stretch-gradient (first-derivative) terms automatically — no explicit s' is
needed. (The previous version of this file used the non-divergence coefficients
s_y/s_x², s_x/s_y² with no stretch-gradient terms, which is a different — and
less well absorbing — operator.)

Medium-aware: k_ref = |k_sub| in the substrate region (y < Y_SUB),
k_ref = k_ext in air. Polynomial grading order 3. Both s_x and s_y are active
simultaneously in the corners.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
N_RE     = 3.0
N_IM     = 30.0
SUB_N_RE = 1.5
SUB_N_IM = 0.1
Y_SUB    = 0.4
lam      = 0.5
k        = 2.0 * np.pi / lam
a        = 0.05
cx, cy   = 0.5, 0.5

theta_inc = -(np.pi) / 4
kx = k * np.cos(theta_inc)
ky = k * np.sin(theta_inc)   # negative (downward)

# Physical domain bounds
X_LO, X_HI = 0.0, 1.0   # physical domain x range
Y_LO, Y_HI = 0.0, 1.0   # physical domain y range

# PML settings — kept identical to mie_pinn_sample.py so both solvers use the
# same absorbing layer.
d_pml     = 0.6       # PML thickness in physical units (~0.6 wavelengths); matches PINN
sigma_max = 6.0 * k   # peak conductivity
pml_order = 3

N = 601   # grid points over full domain including PML

# ─────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────
k_int    = (N_RE + 1j * N_IM) * k
k_int_sq = k_int**2
k_ext_sq = k**2
k_sub    = (SUB_N_RE + 1j * SUB_N_IM) * k
k_sub_sq = k_sub**2

# Fresnel TM coefficients for flat interface
# ky_sub for the transmitted FIELD must have Re < 0 to propagate downward,
# matching the incident ky < 0. numpy sqrt gives the Re > 0 root, so we flip.
ky_sub = np.sqrt(k_sub_sq - kx**2 + 0j)
if ky_sub.real > 0:
    ky_sub = -ky_sub   # ensure Re(ky_sub) < 0: downward-propagating

# s-polarisation (TE / out-of-plane-E) Fresnel coefficients, matching the
# continuity this solver actually enforces: u AND du/dy continuous at y=Y_SUB
# (the plain 5-point stencil with k² only in the reaction term is the discrete
# form of ∇²u + k²(y)u, whose natural interface conditions are exactly those).
# With these coefficients the background satisfies BOTH matching conditions,
# so u_scat carries no interface line source and is purely tip-induced.
#   q_air = |ky|  (vertical wavenumber, air, downward magnitude)
#   q_sub = -ky_sub  (Re > 0)
#   r_s = (q_air - q_sub)/(q_air + q_sub),  t_s = 1 + r_s
# (Variable names r_TM/t_TM kept to avoid touching downstream code; note the
# script's "TM mode" = Ez out of plane = s-polarisation in optics naming.)
q_air = abs(ky)
q_sub = -ky_sub          # Re(q_sub) > 0
r_TM  = (q_air - q_sub) / (q_air + q_sub)
t_TM  = (2.0 * q_air)   / (q_air + q_sub)

# Source contrast: only nonzero inside cylinder
delta_k2 = k_int_sq - k_ext_sq   # k²_tip - k²_air

print("=" * 65)
print("FDM + PML — Scattered-Field Formulation")
print("=" * 65)
print(f"  n_tip      = {N_RE} + {N_IM}i")
print(f"  n_sub      = {SUB_N_RE} + {SUB_N_IM}i")
print(f"  theta_inc  = {np.degrees(theta_inc):.0f} deg  "
      f"(kx={kx:.3f}, ky={ky:.3f})")
print(f"  r_TM       = {r_TM:.4f}")
print(f"  t_TM       = {t_TM:.4f}")
print(f"  ky_sub     = {ky_sub:.4f}")
print(f"  delta_k2   = {delta_k2:.4f}  (tip source contrast)")
print(f"  Physical domain: x=[{X_LO}, {X_HI}], y=[{Y_LO}, {Y_HI}]")
print(f"  PML: d={d_pml}, sigma_max={sigma_max/k:.1f}k, order={pml_order}")
print(f"  Grid: {N}x{N} over "
      f"[{X_LO-d_pml:.2f}, {X_HI+d_pml:.2f}] x "
      f"[{Y_LO-d_pml:.2f}, {Y_HI+d_pml:.2f}]")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
# Full domain including PML on all sides
x_full_lo = X_LO - d_pml;  x_full_hi = X_HI + d_pml
y_full_lo = Y_LO - d_pml;  y_full_hi = Y_HI + d_pml

x_span = x_full_hi - x_full_lo
y_span = y_full_hi - y_full_lo

# Use a single uniform spacing; grids may differ if domain is non-square.
# We build separate x and y coordinate arrays.
hx = x_span / (N - 1)
hy = y_span / (N - 1)

x_coords = np.linspace(x_full_lo, x_full_hi, N)
y_coords = np.linspace(y_full_lo, y_full_hi, N)

Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing='ij')

R2         = (X_grid - cx)**2 + (Y_grid - cy)**2
in_cyl     = R2 <= a**2
in_sub_all = Y_grid < Y_SUB   # substrate region across full grid incl. PML

# ─────────────────────────────────────────────────────────────────────────────
# Background field u_bg — Fresnel (no tip)
# This appears only on the RHS, never in the stencil
# ─────────────────────────────────────────────────────────────────────────────
# Above interface: incident + reflected (phase origin at y = Y_SUB)
u_bg_air = (np.exp(1j * (kx * X_grid + ky * (Y_grid - Y_SUB))) +
            r_TM * np.exp(1j * (kx * X_grid - ky * (Y_grid - Y_SUB))))

u_bg_sub = t_TM * np.exp(1j * (kx * X_grid + ky_sub * (Y_grid - Y_SUB)))
u_bg = np.where(in_sub_all, u_bg_sub, u_bg_air)

# ─────────────────────────────────────────────────────────────────────────────
# k²(x,y) for the TOTAL field equation
# ─────────────────────────────────────────────────────────────────────────────
k2 = np.where(in_cyl,     k_int_sq,
     np.where(in_sub_all, k_sub_sq, k_ext_sq)).astype(complex)

# ─────────────────────────────────────────────────────────────────────────────
# Medium-aware PML stretch factors
# ─────────────────────────────────────────────────────────────────────────────
def pml_sigma(coord, lo, hi):
    """Polynomial PML conductivity profile, zero inside [lo, hi]."""
    s = np.zeros(len(coord))
    s[coord < lo] = sigma_max * ((lo - coord[coord < lo]) / d_pml) ** pml_order
    s[coord > hi] = sigma_max * ((coord[coord > hi] - hi) / d_pml) ** pml_order
    return s

sigma_x_raw = pml_sigma(x_coords, X_LO, X_HI)
sigma_y_raw = pml_sigma(y_coords, Y_LO, Y_HI)

print("Precomputing PML stretch factors ...")
SX = np.ones((N, N), dtype=complex)
SY = np.ones((N, N), dtype=complex)

for i in range(N):
    y_val = y_coords[i]
    # k_ref for x-stretching depends on which medium the row is in
    k_ref_x = abs(k_sub) if y_val < Y_SUB else k
    SX[i, :] = 1.0 + 1j * sigma_x_raw / k_ref_x

    # k_ref for y-stretching: only active outside physical domain
    if y_val < Y_LO:
        k_ref_y = abs(k_sub)   # bottom PML — substrate side
    elif y_val > Y_HI:
        k_ref_y = k            # top PML — air side
    else:
        k_ref_y = None         # inside physical domain: no y-stretching

    if k_ref_y is not None:
        SY[i, :] = 1.0 + 1j * sigma_y_raw[i] / k_ref_y
    # else SY[i,:] stays 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Divergence-form UPML coefficient fields (same operator as the PINN):
#   A = s_y/s_x  (coefficient inside ∂x[ A ∂x v ])
#   B = s_x/s_y  (coefficient inside ∂y[ B ∂y v ])
#   REACT = k²(x,y) · s_x · s_y   (reaction term)
# These are differenced at cell FACES below, which is what makes the scheme
# conservative and reproduces the stretch-gradient terms of the PINN residual.
# ─────────────────────────────────────────────────────────────────────────────
A_coef = SY / SX
B_coef = SX / SY
REACT  = k2 * SX * SY

# ─────────────────────────────────────────────────────────────────────────────
# Assemble scattered-field system
#
# Equation for v = u_scat (divergence-form UPML, same operator as the PINN):
#   ∂/∂x[(sᵧ/sₓ) ∂v/∂x] + ∂/∂y[(sₓ/sᵧ) ∂v/∂y] + k²(i,j) sₓ sᵧ v
#       = -(k²(i,j) - k²_bg(i,j)) * u_bg(i,j)        (source only inside cylinder)
#
# Discretised conservatively: the ∂x[A ∂x v] term becomes
#   [A_{j+½}(v_{j+1}-v_j) - A_{j-½}(v_j-v_{j-1})]/hx²  with A_{j±½} face averages.
# Non-square domain: separate hx, hy. Boundary: v = 0 on outer PML edge.
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

        on_boundary = (i == 0 or i == N-1 or j == 0 or j == N-1)

        if on_boundary:
            rows_l.append(I); cols_l.append(I); vals_l.append(1.0)
            rhs[I] = 0.0

        else:
            # Conservative flux stencil for the divergence-form UPML.
            # Face coefficients = arithmetic mean of the neighbouring node values
            # (standard second-order conservative differencing of ∂x[A ∂x v] etc.).
            aE = 0.5 * (A_coef[i, j] + A_coef[i, j+1]) * inv_hx2   # east  face (j+½)
            aW = 0.5 * (A_coef[i, j] + A_coef[i, j-1]) * inv_hx2   # west  face (j-½)
            bN = 0.5 * (B_coef[i, j] + B_coef[i+1, j]) * inv_hy2   # north face (i+½)
            bS = 0.5 * (B_coef[i, j] + B_coef[i-1, j]) * inv_hy2   # south face (i-½)
            c_d = -(aE + aW + bN + bS) + REACT[i, j]

            rows_l += [I,            I,            I,            I,            I ]
            cols_l += [flat(i, j+1), flat(i, j-1), flat(i+1, j), flat(i-1, j), I ]
            vals_l += [aE,           aW,           bN,           bS,           c_d]

            # Source: nonzero only inside cylinder (which lies in the physical
            # region, so s_x = s_y = 1 there and the factor below is exactly 1).
            if in_cyl[i, j]:
                rhs[I] = -(k_int_sq - k_ext_sq) * SX[i, j] * SY[i, j] * u_bg[i, j]
            else:
                rhs[I] = 0.0

A_mat = coo_matrix((vals_l, (rows_l, cols_l)), shape=(n_dof, n_dof)).tocsc()
print(f"  Built in {time.time()-t0:.2f}s  (nnz = {A_mat.nnz:,})")

# ─────────────────────────────────────────────────────────────────────────────
# Solve for v = u_scat
# ─────────────────────────────────────────────────────────────────────────────
print("Solving ...")
t1 = time.time()
v_flat = spsolve(A_mat, rhs)
print(f"  Solved in {time.time()-t1:.2f}s")

V_full = v_flat.reshape(N, N)   # u_scat over full domain

# ─────────────────────────────────────────────────────────────────────────────
# Extract physical domain [X_LO, X_HI] x [Y_LO, Y_HI]
# ─────────────────────────────────────────────────────────────────────────────
i_lo = np.searchsorted(y_coords, Y_LO - hy/2)
i_hi = np.searchsorted(y_coords, Y_HI + hy/2)
j_lo = np.searchsorted(x_coords, X_LO - hx/2)
j_hi = np.searchsorted(x_coords, X_HI + hx/2)

U_scat    = V_full   [i_lo:i_hi, j_lo:j_hi]
U_bg_ph   = u_bg     [i_lo:i_hi, j_lo:j_hi]
X_ph      = X_grid   [i_lo:i_hi, j_lo:j_hi]
Y_ph      = Y_grid   [i_lo:i_hi, j_lo:j_hi]
in_cyl_ph = in_cyl   [i_lo:i_hi, j_lo:j_hi]
in_sub_ph = Y_ph < Y_SUB

U_total = U_scat + U_bg_ph
U_inc   = np.exp(1j * (kx * X_ph + ky * Y_ph))

print(f"\nField check (physical domain):")
print(f"  Max |U_total|        = {np.max(np.abs(U_total)):.4f}")
print(f"  Tip   mean |U_total| = {np.mean(np.abs(U_total[in_cyl_ph])):.4f}")
print(f"  Sub   mean |U_total| = {np.mean(np.abs(U_total[in_sub_ph & ~in_cyl_ph])):.4f}")
print(f"  Air   mean |U_total| = {np.mean(np.abs(U_total[~in_cyl_ph & ~in_sub_ph])):.4f}")
print(f"  Max |U_scat|         = {np.max(np.abs(U_scat)):.4f}  (tip-induced only)")
print(f"  Scat/Total ratio     = {np.max(np.abs(U_scat))/np.max(np.abs(U_total)):.3f}")
print(f"  Total time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("sample/inputs", exist_ok=True)
np.savez("sample/inputs/afm_fdm_solution.npz",
         X=X_ph, Y=Y_ph,
         U_total_re=U_total.real, U_total_im=U_total.imag,
         U_scat_re=U_scat.real,   U_scat_im=U_scat.imag,
         U_bg_re=U_bg_ph.real,    U_bg_im=U_bg_ph.imag,
         U_inc_re=U_inc.real,     U_inc_im=U_inc.imag,
         in_cyl=in_cyl_ph, in_sub=in_sub_ph,
         params=np.array([N_RE, N_IM, k, a, cx, cy,
                          SUB_N_RE, SUB_N_IM, Y_SUB,
                          np.degrees(theta_inc)], dtype=float))
print("Saved: sample/inputs/afm_fdm_solution.npz")

# ─────────────────────────────────────────────────────────────────────────────
# Plot — 3×3 + PML inset
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
    im    = ax.pcolormesh(X_ph, Y_ph, data, shading='auto',
                          cmap=cmap, vmin=vmin, vmax=scale)
    ax.add_patch(plt.Circle((cx, cy), a, **circle_kw))
    ax.axhline(Y_SUB, **sub_line_kw)
    ax.axhspan(Y_LO, Y_SUB, alpha=0.07, color='white', zorder=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal')
    ax.set_xlim(X_LO, X_HI); ax.set_ylim(Y_LO, Y_HI)
    plt.colorbar(im, ax=ax, shrink=0.92)

panel(axes[0,0], U_bg_ph.real,    "Background  Re(u_bg)\n[Fresnel, no tip]",   "RdBu_r", True)
panel(axes[0,1], U_bg_ph.imag,    "Background  Im(u_bg)",                       "RdBu_r", True)
panel(axes[0,2], np.abs(U_bg_ph), "Background  |u_bg|",                         "viridis", False)

panel(axes[1,0], U_scat.real,     "Scattered   Re(u_scat)\n[tip-induced only]", "RdBu_r", True)
panel(axes[1,1], U_scat.imag,     "Scattered   Im(u_scat)",                     "RdBu_r", True)
panel(axes[1,2], np.abs(U_scat),  "Scattered   |u_scat|",                       "hot",    False)

panel(axes[2,0], U_total.real,    "Total       Re(u_total)",                    "RdBu_r", True)
panel(axes[2,1], U_total.imag,    "Total       Im(u_total)",                    "RdBu_r", True)
panel(axes[2,2], np.abs(U_total), "Total       |u_total|",                      "inferno", False)

# PML σ inset — show y profile over full domain including PML
ax_pml = fig.add_axes([0.91, 0.36, 0.065, 0.28])
yi_plot = np.linspace(y_full_lo, y_full_hi, 600)
si_plot = pml_sigma(yi_plot, Y_LO, Y_HI)
ax_pml.fill_betweenx(yi_plot, 0, si_plot, where=yi_plot >= Y_LO,
                     color='orange', alpha=0.6, label='air')
ax_pml.fill_betweenx(yi_plot, 0, si_plot, where=yi_plot < Y_LO,
                     color='steelblue', alpha=0.6, label='sub')
ax_pml.axhline(Y_LO,  color='white',  lw=0.8, ls='--')
ax_pml.axhline(Y_HI,  color='white',  lw=0.8, ls='--')
ax_pml.axhline(Y_SUB, color='yellow', lw=0.8, ls='--')
ax_pml.set_title("PML σ(y)", fontsize=8)
ax_pml.set_xlabel("σ", fontsize=7)
ax_pml.set_ylabel("y", fontsize=7)
ax_pml.tick_params(labelsize=6)
ax_pml.legend(fontsize=5, loc='lower right')

handles = [
    matplotlib.patches.Circle((0,0), 0.1, ec='cyan', fc='none', ls='--', lw=1.5,
                               label='Tip boundary'),
    matplotlib.lines.Line2D([0],[0], color='yellow', ls='--', lw=1.5,
                             label=f'Substrate interface (y={Y_SUB})'),
]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.45, 0.005))

fig.suptitle(
    f"AFM Tip + Substrate — Scattered-Field FDM+PML  |  "
    f"n_tip={N_RE}+{N_IM}i  n_sub={SUB_N_RE}+{SUB_N_IM}i  "
    f"theta={np.degrees(theta_inc):.0f}deg\n"
    f"Physical domain: x=[{X_LO},{X_HI}], y=[{Y_LO},{Y_HI}]  |  "
    f"Solved for v=u_scat directly; source = -(k²_tip - k²_air)*u_bg inside cylinder",
    fontsize=11, y=1.01)

plt.tight_layout(rect=[0, 0.03, 0.91, 1])
os.makedirs("sample/figures", exist_ok=True)
plt.savefig("sample/figures/afm_fdm_pml_fields.png", dpi=150, bbox_inches='tight')
print("Saved: sample/figures/afm_fdm_pml_fields.png")

# plt.show()
print("\nDone.")