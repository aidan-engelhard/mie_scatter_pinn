"""
FDM Reference — 2D Dielectric Cylinder Scattering (TM mode)
============================================================
Computes the exact (up to discretisation error) solution to the same
problem as the dielectric_pinn.py script, using a sparse direct solve.

FORMULATION
-----------
Solves for the TOTAL field u_total everywhere on a single unified grid.
Inside the cylinder k²(x,y) = k_int² (complex).
Outside the cylinder k²(x,y) = k_ext² = k² (real).

PDE:
    (∇² + k²(x,y)) u_total = 0    everywhere in [0,1]²

The interface conditions (Ez and ∂Ez/∂n continuity) are automatically
embedded in the 5-point stencil — no special treatment needed because
u_total is one continuous unknown.

Outer boundary — Sommerfeld ABC on the SCATTERED field u_scat = u_total - u_inc:
    ∂u_scat/∂n - ik u_scat = 0
    → ∂u_total/∂n - ik u_total = ∂u_inc/∂n - ik u_inc   (known RHS)

With u_inc = exp(ikx):
    ∂u_inc/∂x = ik u_inc,   ∂u_inc/∂y = 0

RHS per wall (outward normal n̂):
    RIGHT  (n̂=+x̂): RHS = ik u_inc - ik u_inc = 0       ← wave exits cleanly
    LEFT   (n̂=-x̂): RHS = -ik u_inc - ik u_inc = -2ik u_inc
    TOP    (n̂=+ŷ): RHS = 0 - ik u_inc = -ik u_inc
    BOTTOM (n̂=-ŷ): RHS = 0 - ik u_inc = -ik u_inc

Normal derivative discretised with 2nd-order one-sided stencil:
    ∂u/∂n ≈ (3u₀ - 4u₁ + u₂) / (2h)
where u₀ is the boundary node and u₁, u₂ are the two adjacent interior nodes.

PARAMETERS (must match dielectric_pinn.py)
------------------------------------------
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
N_RE   = 1.5
N_IM   = 0.5
lam    = 0.5
k      = 2.0 * np.pi / lam
a      = 0.05
cx, cy = 0.5, 0.5
# Direction of propagation: angle in radians from +x axis
theta_inc  = -np.pi / 4
kx     = k * np.cos(theta_inc)
ky     = k * np.sin(theta_inc)

N      = 401     # grid points per side  (increase for higher accuracy)

# ─────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────
k_int    = (N_RE + 1j * N_IM) * k
k_int_sq = k_int**2
k_ext_sq = k**2

print("=" * 58)
print("FDM — Dielectric Cylinder (TM mode)")
print("=" * 58)
print(f"  n         = {N_RE} + {N_IM}i")
print(f"  k_ext     = {k:.4f}")
print(f"  k_int     = {k_int:.4f}")
print(f"  k_int²    = {k_int_sq:.4f}")
print(f"  a         = {a},   ka = {k*a:.3f}")
print(f"  Grid      = {N}×{N} = {N**2:,} DOF")
print("=" * 58)

# ─────────────────────────────────────────────────────────────────────────────
# Grid — rows = y (first index), cols = x (second index)
# → pcolormesh(X, Y, data) puts y on vertical axis naturally
# ─────────────────────────────────────────────────────────────────────────────
h = 1.0 / (N - 1)
x = np.linspace(0.0, 1.0, N)
y = np.linspace(0.0, 1.0, N)

# Y[i,j]=y[i], X[i,j]=x[j]  → first index = y-row, second = x-col
Y, X = np.meshgrid(y, x, indexing='ij')

R2 = (X - cx)**2 + (Y - cy)**2
k2 = np.where(R2 <= a**2, k_int_sq, k_ext_sq)   # complex array
u_inc_grid = np.exp(1j * (kx * X + ky * Y))

# ─────────────────────────────────────────────────────────────────────────────
# Assemble sparse system
# DOF ordering: flat index I = i*N + j
# i = y-index (row, vertical), j = x-index (col, horizontal)
# ─────────────────────────────────────────────────────────────────────────────
n_dof  = N * N
inv_h2 = 1.0 / h**2
h2     = 2.0 * h   # 2h denominator for 2nd-order one-sided stencil

def flat(i, j):
    return i * N + j

print(f"\nAssembling {n_dof:,}-DOF sparse system ...")
t0 = time.time()

rows, cols, vals = [], [], []
rhs = np.zeros(n_dof, dtype=complex)

for i in range(N):        # i = y-index
    for j in range(N):    # j = x-index
        I   = flat(i, j)
        ui  = u_inc_grid[i, j]

        on_T = (i == N-1)   # top    y=1, n̂=+ŷ
        on_B = (i == 0)     # bottom y=0, n̂=-ŷ
        on_R = (j == N-1)   # right  x=1, n̂=+x̂
        on_L = (j == 0)     # left   x=0, n̂=-x̂
        is_corner = (on_T or on_B) and (on_R or on_L)

        if is_corner:
            rows.append(I); cols.append(I); vals.append(1.0)
            rhs[I] = ui

        elif on_R:
            rows += [I, I, I]
            cols += [I, flat(i,j-1), flat(i,j-2)]
            vals += [3/h2 - 1j*k, -4/h2, 1/h2]
            rhs[I] = (1j*kx - 1j*k) * ui          # (ik_n - ik) * u_inc

        elif on_L:
            rows += [I, I, I]
            cols += [I, flat(i,j+1), flat(i,j+2)]
            vals += [3/h2 - 1j*k, -4/h2, 1/h2]
            rhs[I] = (-1j*kx - 1j*k) * ui         # (-ik_n - ik) * u_inc

        elif on_T:
            rows += [I, I, I]
            cols += [I, flat(i-1,j), flat(i-2,j)]
            vals += [3/h2 - 1j*k, -4/h2, 1/h2]
            rhs[I] = (1j*ky - 1j*k) * ui          # (ik_n - ik) * u_inc

        elif on_B:
            rows += [I, I, I]
            cols += [I, flat(i+1,j), flat(i+2,j)]
            vals += [3/h2 - 1j*k, -4/h2, 1/h2]
            rhs[I] = (-1j*ky - 1j*k) * ui         # (-ik_n - ik) * u_inc

        else:
            # Interior: (∇² + k²(i,j)) u = 0
            rows += [I, I, I, I, I]
            cols += [flat(i+1,j), flat(i-1,j), flat(i,j+1), flat(i,j-1), I]
            vals += [inv_h2, inv_h2, inv_h2, inv_h2, -4*inv_h2 + k2[i,j]]
            rhs[I] = 0.0

A_mat = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsc()
print(f"  Built in {time.time()-t0:.2f}s  (nnz = {A_mat.nnz:,})")

# ─────────────────────────────────────────────────────────────────────────────
# Solve
# ─────────────────────────────────────────────────────────────────────────────
print("Solving ...")
t1 = time.time()
u_flat = spsolve(A_mat, rhs)
print(f"  Solved in {time.time()-t1:.2f}s")

U_total = u_flat.reshape(N, N)
U_inc   = u_inc_grid
U_scat  = U_total - U_inc
in_cyl  = R2 <= a**2

print(f"\nField check:")
print(f"  Max |U_total|          = {np.max(np.abs(U_total)):.4f}")
print(f"  Inside mean |U_total|  = {np.mean(np.abs(U_total[in_cyl])):.4f}")
print(f"  Outside mean |U_total| = {np.mean(np.abs(U_total[~in_cyl])):.4f}")
print(f"  Total time: {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save FDM solution for use as PINN reference data
# Saves: X, Y grids, U_total, U_scat, U_inc (all complex, shape [N,N])
# The PINN loads this and interpolates at its collocation points.
# ─────────────────────────────────────────────────────────────────────────────
np.savez("inputs/fdm_solution.npz",
         X=X, Y=Y,
         U_total_re=U_total.real, U_total_im=U_total.imag,
         U_scat_re=U_scat.real,   U_scat_im=U_scat.imag,
         U_inc_re=U_inc.real,     U_inc_im=U_inc.imag,
         in_cyl=in_cyl,
         params=np.array([N_RE, N_IM, k, a, cx, cy]))
print("Saved FDM solution: inputs/fdm_solution.npz")

# ─────────────────────────────────────────────────────────────────────────────
# Plot — same 2×3 layout as PINN
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

circle_kw = dict(color='cyan', lw=1.5, fill=False, zorder=5, ls='--')

def panel(ax, data, title, cmap, symmetric):
    valid = data[np.isfinite(data)]
    scale = float(np.percentile(np.abs(valid), 98)) if len(valid) else 1.0
    scale = max(scale, 1e-12)
    vmin  = -scale if symmetric else 0.0
    im    = ax.pcolormesh(X, Y, data, shading='auto',
                          cmap=cmap, vmin=vmin, vmax=scale)
    ax.add_patch(plt.Circle((cx,cy), a, **circle_kw))
    ax.set_title(title, fontsize=16); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, shrink=0.95)

panel(axes[0,0], U_inc.real,      "Incident    Re(u_inc)",   "RdBu_r", True)
panel(axes[0,1], U_scat.real,     "Scattered   Re(u_scat)",  "RdBu_r", True)
panel(axes[0,2], U_total.real,    "Total       Re(u_total)", "RdBu_r", True)
panel(axes[1,0], np.abs(U_inc),   "Incident    |u_inc|",     "viridis", False)
panel(axes[1,1], np.abs(U_scat),  "Scattered   |u_scat|",    "hot",    False)
panel(axes[1,2], np.abs(U_total), "Total       |u_total|",   "inferno", False)

plt.tight_layout()
plt.savefig("figures/mie_2d/dielectric_fdm_fields.png", dpi=150, bbox_inches='tight')
print("Saved: figures/mie_2d/dielectric_fdm_fields.png")

# plt.show()
print("\nDone.")