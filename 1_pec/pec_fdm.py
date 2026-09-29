"""
FDM — Rayleigh Scattering on a Unit Square
===========================================
Solves the 2D Helmholtz equation for the scattered field u_scat using a
Finite Difference Method on a uniform Cartesian grid.

Problem:
  (nabla^2 + k^2) u_scat = 0        in Omega \ D
  u_scat = -u_inc                    on dD  (sound-soft / PEC scatterer, Dirichlet)
  du_scat/dn - ik u_scat = 0        on dOmega  (first-order Sommerfeld absorbing BC)

  u_inc = exp(i k x),  D = disk(radius=a, centre=(0.5, 0.5))

Grid:
  N_grid x N_grid uniform nodes on [0,1]^2.
  Interior Helmholtz stencil: standard 5-point second-order FD.
  Scatterer nodes (inside D + boundary layer): Dirichlet u_scat = -u_inc.
  Outer boundary: Robin / first-order absorbing condition discretised
                  with one-sided differences.

The resulting sparse complex linear system A x = b is solved with
scipy.sparse.linalg.spsolve (direct LU).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os, time

# ─────────────────────────────────────────────────────────────────────────────
# Config  (identical physical parameters to the PINN script)
# ─────────────────────────────────────────────────────────────────────────────
k      = 2.0 * np.pi   # wave number  (wavelength = 1)
a      = 0.05           # scatterer radius  (ka ~ 0.314, Rayleigh regime)
cx, cy = 0.5, 0.5       # scatterer centre

N_grid = 200            # grid resolution (matches PINN evaluation grid)

# Output directories (mirror PINN script)
os.makedirs("figures/pec/fields",    exist_ok=True)
os.makedirs("figures/pec/training",  exist_ok=True)

ka = k * a
print(f"k={k:.4f},  a={a:.4f},  ka={ka:.4f}  (Rayleigh regime: ka < 1)")
print(f"Grid: {N_grid} x {N_grid}  =>  {N_grid**2:,} unknowns")

# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
xg = np.linspace(0, 1, N_grid)
yg = np.linspace(0, 1, N_grid)
h  = xg[1] - xg[0]                     # uniform spacing

Xg, Yg = np.meshgrid(xg, yg)           # shape (N_grid, N_grid), row = y-index
R2      = (Xg - cx)**2 + (Yg - cy)**2
in_sc   = R2 <= a**2                    # nodes inside / on the scatterer

# Node index helper: (i_row, j_col) -> flat index
def idx(i, j):
    return i * N_grid + j

N = N_grid * N_grid

# ─────────────────────────────────────────────────────────────────────────────
# Incident field
# ─────────────────────────────────────────────────────────────────────────────
U_inc_flat = np.exp(-1j * k * Xg).ravel()   # complex128  (propagates in x direction)

# ─────────────────────────────────────────────────────────────────────────────
# Build sparse system  A u_scat = b
# ─────────────────────────────────────────────────────────────────────────────
t0 = time.time()
print("Building FDM system ...", end=" ", flush=True)

rows, cols, vals = [], [], []
rhs = np.zeros(N, dtype=complex)

def add(r, c, v):
    rows.append(r); cols.append(c); vals.append(v)

for i in range(N_grid):         # i = row index (y direction)
    for j in range(N_grid):     # j = col index (x direction)
        p  = idx(i, j)
        x  = xg[j]
        y  = yg[i]
        r2 = (x - cx)**2 + (y - cy)**2

        # ── Scatterer / interior of disk: Dirichlet u_scat = -u_inc ──────────
        if r2 <= (a + 0.5*h)**2:
            add(p, p, 1.0)
            rhs[p] = -U_inc_flat[p]
            continue

        # ── Outer boundary: Robin absorbing BC  du/dn - ik u = 0 ────────────
        # One-sided first-order difference, outward normal direction per wall.

        on_boundary = (i == 0) or (i == N_grid-1) or (j == 0) or (j == N_grid-1)

        if on_boundary:
            # Right wall (x=1), outward normal +x:  du/dx - ik u = 0
            #   => (u_p - u_{j-1})/h - ik u_p = 0
            if j == N_grid - 1:
                add(p, p,            1.0/h - 1j*k)
                add(p, idx(i, j-1), -1.0/h)
                rhs[p] = 0.0
            # Left wall (x=0), outward normal -x:  -du/dx - ik u = 0
            #   => (u_p - u_{j+1})/h - ik u_p = 0
            elif j == 0:
                add(p, p,            1.0/h - 1j*k)
                add(p, idx(i, j+1), -1.0/h)
                rhs[p] = 0.0
            # Top wall (y=1), outward normal +y:  du/dy - ik u = 0
            #   => (u_p - u_{i-1,j})/h - ik u_p = 0
            elif i == N_grid - 1:
                add(p, p,            1.0/h - 1j*k)
                add(p, idx(i-1, j), -1.0/h)
                rhs[p] = 0.0
            # Bottom wall (y=0), outward normal -y:  -du/dy - ik u = 0
            #   => (u_p - u_{i+1,j})/h - ik u_p = 0
            else:  # i == 0
                add(p, p,            1.0/h - 1j*k)
                add(p, idx(i+1, j), -1.0/h)
                rhs[p] = 0.0
            continue

        # ── Interior Helmholtz: (nabla^2 + k^2) u_scat = 0 ──────────────────
        # Standard 5-point stencil: (u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}
        #                             - 4 u_{i,j})/h^2 + k^2 u_{i,j} = 0

        for (ni, nj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
            nr2 = (xg[nj]-cx)**2 + (yg[ni]-cy)**2
            if nr2 <= (a + 0.5*h)**2:
                # Neighbour is inside scatterer: u_nb = -u_inc(nb)
                rhs[p] -= (1.0/h**2) * (-U_inc_flat[idx(ni,nj)])
            else:
                add(p, idx(ni, nj), 1.0/h**2)

        add(p, p, -4.0/h**2 + k**2)

A   = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=complex)
print(f"done in {time.time()-t0:.2f}s  (nnz={A.nnz:,})")

# ─────────────────────────────────────────────────────────────────────────────
# Solve
# ─────────────────────────────────────────────────────────────────────────────
print("Solving sparse system (direct LU) ...", end=" ", flush=True)
t1 = time.time()
u_scat_flat = spla.spsolve(A, rhs)
print(f"done in {time.time()-t1:.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Reshape and mask scatterer interior
# ─────────────────────────────────────────────────────────────────────────────
U_scat        = u_scat_flat.reshape(N_grid, N_grid)
U_scat[in_sc] = 0.0

U_inc         = np.exp(1j * k * Xg)
U_inc[in_sc]  = 0.0
U_total       = U_inc + U_scat

sigma_rayleigh = (np.pi**2 / 2.0) * ka**4 * a
print(f"\nPhysical parameters:")
print(f"  Wave number k       = {k:.4f} rad/m")
print(f"  Wavelength lambda   = {2*np.pi/k:.4f} m")
print(f"  Scatterer radius a  = {a:.4f} m")
print(f"  ka (Rayleigh param) = {ka:.4f}  (Rayleigh regime: ka << 1)")
print(f"  Rayleigh sigma      = {sigma_rayleigh:.4e} m")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: Field panels  (exact same layout as PINN script)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

fields = [
    (U_inc.real,      "Incident field  Re(u_inc)",   "RdBu_r"),
    (U_scat.real,     "Scattered field Re(u_scat)",  "RdBu_r"),
    (U_total.real,    "Total field     Re(u_total)", "RdBu_r"),
    (np.abs(U_inc),   "Incident |u_inc|",            "viridis"),
    (np.abs(U_scat),  "Scattered |u_scat|",          "hot"),
    (np.abs(U_total), "Total field     |u_total|",   "inferno"),
]

for ax, (data, title, cmap) in zip(axes.flat, fields):
    lim  = np.percentile(np.abs(data[~in_sc]), 98)
    vmin = -lim if "Re" in title else 0.0
    im   = ax.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                     cmap=cmap, vmin=vmin, vmax=lim, interpolation='bilinear')
    ax.add_patch(plt.Circle((cx, cy), a, color='cyan', lw=1.5,
                             fill=False, zorder=5, ls='--'))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.85)

out_path = f"figures/pec/fields/pec_fdm.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
print(f"Total elapsed: {time.time()-t0:.2f}s")
print("\nDone.")