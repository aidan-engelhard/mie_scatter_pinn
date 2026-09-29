"""
FDM SWEEP — cylinder above a substrate, variable tip-sample gap
===============================================================================
Reference solutions for variable_sample_deeponet.py / variable_sample_mlp.py.

WHAT IS SWEPT
-------------
One parameter: the gap between the bottom of the cylinder and the substrate
surface,

    gap = cy - a - Y_SUB       swept over N_PARAM values in [GAP_LO, GAP_HI]

50 values over [0.01, 0.20], split 80/20 into 40 training and 10 test gaps.
The trainers use the FDM solutions at the TRAINING gaps as a data loss
alongside the physics, so the held-out ten are the only honest measure left.

Why so many: at 10 gaps the spacing was 0.021 and adjacent solutions differed
by relL2 0.18-0.29, which is coarse enough that fitting each one separately is
a viable strategy for the network. At 50 the spacing is 0.0039 and neighbours
differ by roughly 0.04, so memorising 40 separate solutions buys much less than
learning how the field moves with the gap.

so the cylinder centre sits at cy = Y_SUB + a + gap. Everything else — radius,
indices, wavelength, incidence angle, PML — is held at the values from
sample/mie_fdm_sample.py, the configuration whose PINN
(sample/mie_pinn_sample.py) converged cleanly in ~10000 L-BFGS steps.

WHY THIS PHYSICS RATHER THAN final/
-----------------------------------
final/final_fdm.py solves a cone-plus-apex AFM tip and discretises the UPML
with the PLAIN second-derivative stencil

    (s_y/s_x) v_xx + (s_x/s_y) v_yy + k^2 s_x s_y v

whose coefficients are evaluated AT THE NODE and which therefore drops the
stretch-gradient (first-derivative) terms. sample/mie_fdm_sample.py uses the
CONSERVATIVE flux form, with the coefficients averaged to cell FACES:

    d/dx[ (s_y/s_x) dv/dx ] + d/dy[ (s_x/s_y) dv/dy ] + k^2 s_x s_y v

which is the discrete equivalent of the divergence-form residual the PINN
minimises, and absorbs measurably better. Since the whole point of this
directory is to reuse the physics that already worked, this script uses the
conservative stencil. Do not "simplify" it back to the node form: the FDM and
the PINN would then be solving different operators and every relL2 in the sweep
would carry that difference as a floor.

SCATTERED-FIELD FORMULATION
---------------------------
Solved for v = u_scat directly. With u_total = v + u_bg and u_bg satisfying the
background (no cylinder) problem,

    (grad^2 + k^2) v = -(k^2 - k^2_bg) u_bg

whose right-hand side is nonzero ONLY inside the cylinder. u_bg is the Fresnel
field of the bare substrate (incident + reflected above, transmitted below),
and it appears only in the RHS, never in the stencil.

WHAT IS SAVED
-------------
variable_sample/inputs/fdm_variable_sample_sweep.npz

    X, Y                 : [N_ph, N_ph]        physical-domain grid
    gap                  : [N_PARAM]           swept gap per index
    cy                   : [N_PARAM]           cylinder centre height per index
    U_scat_re/_im        : [N_PARAM, N, N]     scattered field
    U_total_re/_im       : [N_PARAM, N, N]     total field
    U_bg_re/_im          : [N, N]              background (gap-independent)
    in_cyl               : [N_PARAM, N, N]     cylinder mask per index
    in_sub               : [N, N]              substrate mask (gap-independent)
    train_idx, test_idx  : parameter-index split (test = interpolation gaps)
    solved_idx           : indices actually solved (see SOLVE_SUBSET)
    params               : the scalar physics constants

RUNTIME
-------
N = 601 gives a 361,201-DOF complex system per gap. On CPU that is roughly a
minute each, so the full 50-point sweep is ~50-70 minutes; with CuPy installed
the solve moves to the GPU and it is far quicker. For a smoke test set
SOLVE_SUBSET to a short list of indices — the arrays for unsolved indices stay
zero and the trainer skips them.
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Optional GPU direct solve. Same arrangement as final/final_fdm.py: only the
# spsolve call moves, assembly stays vectorised on numpy.
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    import cupyx.scipy.sparse.linalg as cusparse_linalg
    HAVE_GPU = True
except Exception:
    HAVE_GPU = False

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS — every value below is from sample/mie_fdm_sample.py
# ─────────────────────────────────────────────────────────────────────────────
N_RE     = 3.0     # cylinder: near-PEC, strong absorber
N_IM     = 30.0
SUB_N_RE = 1.5     # substrate: SiO2-like dielectric half-space
SUB_N_IM = 0.1
Y_SUB    = 0.4     # flat interface height
lam      = 0.5
k        = 2.0 * np.pi / lam
a        = 0.05    # cylinder radius
cx       = 0.5     # cylinder centre x (fixed; only cy is swept)

theta_inc = -np.pi / 4
kx = k * np.cos(theta_inc)
ky = k * np.sin(theta_inc)          # negative: downward

X_LO, X_HI = 0.0, 1.0
Y_LO, Y_HI = 0.0, 1.0

d_pml     = 0.6                     # PML thickness, matched to the PINN
sigma_max = 6.0 * k                 # peak conductivity, matched to the PINN
pml_order = 3

# Overridable too, so a smoke test can run the whole path at a coarse grid.
N = globals().get("N_OVERRIDE", 601)   # grid points across physical + PML

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP — the gap is the only parameter
# ─────────────────────────────────────────────────────────────────────────────
# Overridable so evaluate_deeponet.py can build a reference on a DIFFERENT
# set of gaps (its extrapolation tail) by running this file through runpy with
# init_globals, rather than duplicating the solver. Same pattern the trainer
# uses for ARCH and the output paths.
N_PARAM = globals().get("N_PARAM_OVERRIDE", 50)
GAP_LO  = globals().get("GAP_LO_OVERRIDE", 0.01)
GAP_HI  = globals().get("GAP_HI_OVERRIDE", 0.20)
GAP_LIST = globals().get("GAP_LIST_OVERRIDE", None)   # explicit gaps win

GAP_VALS = (np.asarray(GAP_LIST, dtype=float) if GAP_LIST is not None
            else np.linspace(GAP_LO, GAP_HI, N_PARAM))
N_PARAM = len(GAP_VALS)
CY_VALS  = Y_SUB + a + GAP_VALS     # cylinder centre height per index

# The cylinder must stay clear of the substrate for the contrast source to be
# purely tip-induced: below Y_SUB the background k^2 is k_sub^2, not k_ext^2,
# and the single delta_k2 used in the RHS would be the wrong contrast there.
assert (CY_VALS - a > Y_SUB).all(), "cylinder dips into the substrate"
# ... and it must not reach the absorbing layer at the top either. gap = 0.5 is
# the exact limit: cy + a = Y_SUB + 2a + gap = 1.0, tangent to the PML.
assert (CY_VALS + a <= Y_HI + 1e-12).all(), (
    f"cylinder enters the top PML: max gap is {Y_HI - Y_SUB - 2 * a:.3f}")

# Held-out gaps for the generalisation test. Drawn from the INTERIOR of the
# sweep (indices 1..N_PARAM-2) so both are interpolation, not extrapolation —
# a network asked for gap 0.01 or 0.20 having seen neither would be measuring
# something else. Seeded, so the FDM sweep and both trainers agree without
# having to pass the split around.
# 80/20, held out from the INTERIOR of the sweep so every test gap lies
# inside the training range and is therefore an interpolation, not an
# extrapolation off the end. The draw is unconstrained beyond that, so test
# indices can land adjacent to each other -- with the default seed two do,
# at 0.0139 and 0.0178 -- which leaves those two with a training gap two
# steps away (0.0078) rather than one. Still interpolation, just a slightly
# wider bracket. Extrapolation is measured separately by
# evaluate_deeponet.py, against its own reference out to gap 0.5.
N_TEST   = globals().get("N_TEST_OVERRIDE", N_PARAM // 5)
_rng     = np.random.default_rng(0)
test_idx = (np.sort(_rng.choice(np.arange(1, N_PARAM - 1), size=N_TEST,
                                replace=False))
            if N_TEST > 0 else np.zeros(0, dtype=int))
train_idx = np.array([p for p in range(N_PARAM) if p not in set(test_idx.tolist())])

# Solve only these indices. None = the whole sweep. Set to e.g. [0] for a smoke
# test: the unsolved entries stay zero in the npz and the trainer skips them.
SOLVE_SUBSET = None

solve_idx = (list(range(N_PARAM)) if SOLVE_SUBSET is None
             else sorted(int(p) for p in SOLVE_SUBSET))

# ─────────────────────────────────────────────────────────────────────────────
# Derived wavenumbers and Fresnel coefficients
# ─────────────────────────────────────────────────────────────────────────────
k_int    = (N_RE + 1j * N_IM) * k
k_int_sq = k_int ** 2
k_ext_sq = k ** 2 + 0j
k_sub    = (SUB_N_RE + 1j * SUB_N_IM) * k
k_sub_sq = k_sub ** 2

# Transmitted vertical wavenumber. numpy's sqrt returns the Re > 0 root; the
# downward-propagating branch is the one matching the incident ky < 0.
ky_sub = np.sqrt(k_sub_sq - kx ** 2 + 0j)
if ky_sub.real > 0:
    ky_sub = -ky_sub

# s-polarisation (TE, E out of plane) Fresnel pair. These are the coefficients
# for which u AND du/dy are continuous at y = Y_SUB, which is exactly what the
# scalar operator here enforces. With them the background satisfies both
# matching conditions, so u_scat carries no interface line source and is purely
# cylinder-induced — the same object the PINN's merged exterior net represents.
q_air = abs(ky)
q_sub = -ky_sub                     # Re > 0
r_s   = (q_air - q_sub) / (q_air + q_sub)
t_s   = (2.0 * q_air) / (q_air + q_sub)

delta_k2 = k_int_sq - k_ext_sq      # contrast source strength inside the cylinder

print("=" * 74)
print("FDM SWEEP — cylinder above substrate, variable gap")
print("=" * 74)
print(f"  n_cyl        = {N_RE} + {N_IM}i      radius a = {a}")
print(f"  n_sub        = {SUB_N_RE} + {SUB_N_IM}i      Y_SUB = {Y_SUB}")
print(f"  lam          = {lam}   k = {k:.4f}   theta_inc = "
      f"{np.degrees(theta_inc):.0f} deg")
print(f"  r_s          = {r_s:.4f}")
print(f"  t_s          = {t_s:.4f}")
print(f"  ky_sub       = {ky_sub:.4f}")
print(f"  delta_k2     = {delta_k2:.4f}")
print(f"  gap          in [{GAP_LO}, {GAP_HI}], {N_PARAM} values  "
      f"-> cy in [{CY_VALS[0]:.3f}, {CY_VALS[-1]:.3f}]")
print(f"  gap / lam    in [{GAP_LO/lam:.3f}, {GAP_HI/lam:.3f}]")
print(f"  Train / Test = {len(train_idx)} / {len(test_idx)}   "
      f"test gaps = {[f'{GAP_VALS[p]:.3f}' for p in test_idx]}")
print(f"  PML          d={d_pml}, sigma_max={sigma_max/k:.1f}k, order={pml_order}")
print(f"  Grid         {N}x{N} over "
      f"[{X_LO-d_pml:.2f}, {X_HI+d_pml:.2f}]^2   (conservative flux stencil)")
print(f"  GPU solve    {'yes (CuPy)' if HAVE_GPU else 'no (CPU spsolve)'}")
print("=" * 74)

# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
x_coords = np.linspace(X_LO - d_pml, X_HI + d_pml, N)
y_coords = np.linspace(Y_LO - d_pml, Y_HI + d_pml, N)
hx = x_coords[1] - x_coords[0]
hy = y_coords[1] - y_coords[0]

Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing="ij")
in_sub_all = Y_grid < Y_SUB

# ─────────────────────────────────────────────────────────────────────────────
# Background field — Fresnel, no cylinder. Gap-independent, so computed once.
# Phase referenced to the interface, matching the PINN's u_bg_* functions.
# ─────────────────────────────────────────────────────────────────────────────
_bg_air = (np.exp(1j * (kx * X_grid + ky * (Y_grid - Y_SUB))) +
           r_s * np.exp(1j * (kx * X_grid - ky * (Y_grid - Y_SUB))))
_bg_sub = t_s * np.exp(1j * (kx * X_grid + ky_sub * (Y_grid - Y_SUB)))
U_bg_grid = np.where(in_sub_all, _bg_sub, _bg_air)

# ─────────────────────────────────────────────────────────────────────────────
# Medium-aware PML stretch. Gap-independent: k_ref depends only on position
# relative to Y_SUB, never on where the cylinder is.
# ─────────────────────────────────────────────────────────────────────────────
def pml_sigma(coord, lo, hi):
    s = np.zeros(len(coord))
    m = coord < lo
    s[m] = sigma_max * ((lo - coord[m]) / d_pml) ** pml_order
    m = coord > hi
    s[m] = sigma_max * ((coord[m] - hi) / d_pml) ** pml_order
    return s


sigma_x_arr = pml_sigma(x_coords, X_LO, X_HI)
sigma_y_arr = pml_sigma(y_coords, Y_LO, Y_HI)

SX = np.ones((N, N), dtype=complex)
SY = np.ones((N, N), dtype=complex)
for _i in range(N):
    _y = y_coords[_i]
    SX[_i, :] = 1.0 + 1j * sigma_x_arr / (abs(k_sub) if _y < Y_SUB else k)
    if _y < Y_LO:
        SY[_i, :] = 1.0 + 1j * sigma_y_arr[_i] / abs(k_sub)   # bottom: substrate
    elif _y > Y_HI:
        SY[_i, :] = 1.0 + 1j * sigma_y_arr[_i] / k            # top: air

# ─────────────────────────────────────────────────────────────────────────────
# Conservative flux stencil, geometry-independent part
#
#   d/dx[A dv/dx] -> [ A_{j+1/2}(v_{j+1}-v_j) - A_{j-1/2}(v_j-v_{j-1}) ] / hx^2
#
# with A = s_y/s_x, B = s_x/s_y and the face values arithmetic means of the two
# adjacent nodes. Only the reaction term k^2 s_x s_y and the RHS depend on the
# gap, so everything here is built once for the whole sweep.
# ─────────────────────────────────────────────────────────────────────────────
inv_hx2 = 1.0 / hx ** 2
inv_hy2 = 1.0 / hy ** 2
n_dof   = N * N

A_coef = SY / SX
B_coef = SX / SY
SXSY   = SX * SY

IDX = np.arange(n_dof).reshape(N, N)          # flat(i, j) = i*N + j
interior_mask = np.zeros((N, N), dtype=bool)
interior_mask[1:-1, 1:-1] = True
I_int = IDX[interior_mask]
I_bnd = IDX[~interior_mask]

aE = np.zeros((N, N), dtype=complex)
aW = np.zeros((N, N), dtype=complex)
bN = np.zeros((N, N), dtype=complex)
bS = np.zeros((N, N), dtype=complex)
aE[:, :-1] = 0.5 * (A_coef[:, :-1] + A_coef[:, 1:]) * inv_hx2
aW[:, 1:]  = 0.5 * (A_coef[:, 1:]  + A_coef[:, :-1]) * inv_hx2
bN[:-1, :] = 0.5 * (B_coef[:-1, :] + B_coef[1:, :]) * inv_hy2
bS[1:, :]  = 0.5 * (B_coef[1:, :]  + B_coef[:-1, :]) * inv_hy2

aE_int = aE[interior_mask]
aW_int = aW[interior_mask]
bN_int = bN[interior_mask]
bS_int = bS[interior_mask]
face_sum_int = -(aE_int + aW_int + bN_int + bS_int)   # diagonal, minus reaction
sxsy_int     = SXSY[interior_mask]

rows_static = np.concatenate([I_int] * 5 + [I_bnd])
cols_static = np.concatenate([I_int + 1, I_int - 1,   # j+-1 -> x neighbours
                              I_int + N, I_int - N,   # i+-1 -> y neighbours
                              I_int,                  # diagonal
                              I_bnd])                 # boundary identity

_gpu_failed = [False]


def _solve(A, rhs):
    """Direct sparse solve, on GPU when CuPy is importable and working."""
    if HAVE_GPU and not _gpu_failed[0]:
        try:
            return cp.asnumpy(cusparse_linalg.spsolve(
                cusparse.csr_matrix(A.tocsr()), cp.asarray(rhs)))
        except Exception as e:
            _gpu_failed[0] = True
            print(f"  GPU solve failed ({type(e).__name__}: {e}) -- "
                  f"falling back to CPU for the rest of the sweep")
    return spsolve(A, rhs)


def assemble_and_solve(in_cyl):
    """Assemble and solve the scattered-field UPML system for one gap."""
    k2 = np.where(in_cyl, k_int_sq,
                  np.where(in_sub_all, k_sub_sq, k_ext_sq))
    diag = face_sum_int + k2[interior_mask] * sxsy_int

    vals = np.concatenate([aE_int, aW_int, bN_int, bS_int, diag,
                           np.ones(len(I_bnd), dtype=complex)])
    A = coo_matrix((vals, (rows_static, cols_static)),
                   shape=(n_dof, n_dof)).tocsc()

    rhs = np.zeros(n_dof, dtype=complex)
    src = in_cyl & interior_mask
    # The cylinder sits wholly in air (asserted above), so the background k^2
    # under it is k_ext^2 and the contrast is the single delta_k2.
    rhs[IDX[src]] = -delta_k2 * SXSY[src] * U_bg_grid[src]

    return _solve(A, rhs).reshape(N, N)


# ─────────────────────────────────────────────────────────────────────────────
# Physical-subdomain extraction
# ─────────────────────────────────────────────────────────────────────────────
j_lo = np.searchsorted(x_coords, X_LO - hx / 2)
j_hi = np.searchsorted(x_coords, X_HI + hx / 2)
i_lo = np.searchsorted(y_coords, Y_LO - hy / 2)
i_hi = np.searchsorted(y_coords, Y_HI + hy / 2)

X_ph      = X_grid[i_lo:i_hi, j_lo:j_hi]
Y_ph      = Y_grid[i_lo:i_hi, j_lo:j_hi]
N_ph      = X_ph.shape[0]
U_bg_ph   = U_bg_grid[i_lo:i_hi, j_lo:j_hi]
in_sub_ph = Y_ph < Y_SUB

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
U_scat_re_all  = np.zeros((N_PARAM, N_ph, N_ph), dtype=np.float32)
U_scat_im_all  = np.zeros((N_PARAM, N_ph, N_ph), dtype=np.float32)
U_total_re_all = np.zeros((N_PARAM, N_ph, N_ph), dtype=np.float32)
U_total_im_all = np.zeros((N_PARAM, N_ph, N_ph), dtype=np.float32)
in_cyl_all     = np.zeros((N_PARAM, N_ph, N_ph), dtype=bool)

_solve_set = set(solve_idx)
print(f"\nSolving {len(_solve_set)} of {N_PARAM} gaps"
      + ("" if len(_solve_set) == N_PARAM else f"  (SOLVE_SUBSET = {solve_idx})"))
t0 = time.time()

for p in range(N_PARAM):
    cy_p  = float(CY_VALS[p])
    gap_p = float(GAP_VALS[p])

    # The mask is needed on the PHYSICAL grid for every index (the trainer uses
    # it to build the composite scattered field), even for gaps left unsolved.
    R2_ph = (X_ph - cx) ** 2 + (Y_ph - cy_p) ** 2
    in_cyl_all[p] = R2_ph <= a ** 2

    if p not in _solve_set:
        continue

    t1 = time.time()
    R2_full = (X_grid - cx) ** 2 + (Y_grid - cy_p) ** 2
    V_full  = assemble_and_solve(R2_full <= a ** 2)

    U_scat  = V_full[i_lo:i_hi, j_lo:j_hi]
    U_total = U_scat + U_bg_ph

    U_scat_re_all[p]  = U_scat.real.astype(np.float32)
    U_scat_im_all[p]  = U_scat.imag.astype(np.float32)
    U_total_re_all[p] = U_total.real.astype(np.float32)
    U_total_im_all[p] = U_total.imag.astype(np.float32)

    tag = "TEST " if p in set(test_idx.tolist()) else "train"
    print(f"  [{p:>2}/{N_PARAM}] {tag}  gap={gap_p:.4f}  cy={cy_p:.4f}  "
          f"max|u_scat|={np.abs(U_scat).max():.4f}  "
          f"max|u_tot|={np.abs(U_total).max():.4f}  "
          f"({time.time()-t1:.1f}s)")

print(f"\nSweep done in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("variable_sample/inputs", exist_ok=True)
OUT = globals().get("OUT_OVERRIDE",
                    "variable_sample/inputs/fdm_variable_sample_sweep.npz")
np.savez_compressed(
    OUT,
    X=X_ph.astype(np.float32), Y=Y_ph.astype(np.float32),
    gap=GAP_VALS.astype(np.float32),
    cy=CY_VALS.astype(np.float32),
    U_scat_re=U_scat_re_all,   U_scat_im=U_scat_im_all,
    U_total_re=U_total_re_all, U_total_im=U_total_im_all,
    U_bg_re=U_bg_ph.real.astype(np.float32),
    U_bg_im=U_bg_ph.imag.astype(np.float32),
    in_cyl=in_cyl_all, in_sub=in_sub_ph,
    train_idx=train_idx.astype(np.int32),
    test_idx=test_idx.astype(np.int32),
    solved_idx=np.asarray(solve_idx, dtype=np.int32),
    params=np.array([N_RE, N_IM, SUB_N_RE, SUB_N_IM, Y_SUB, lam, k, a, cx,
                     np.degrees(theta_inc), d_pml, sigma_max, pml_order],
                    dtype=float),
)
print(f"Saved: {OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# Sweep figure — |u_scat| for every solved gap
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("variable_sample/figures", exist_ok=True)
ncol = 5 if N_PARAM <= 25 else 10
nrow = int(np.ceil(N_PARAM / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.5 * nrow))
axes = np.atleast_2d(axes)
circle_kw = dict(color="cyan", lw=1.0, fill=False, zorder=5, ls="--")
sub_kw    = dict(color="yellow", lw=0.8, zorder=6, ls="--")

for p in range(N_PARAM):
    ax = axes.ravel()[p]
    if p in _solve_set:
        data = np.abs(U_scat_re_all[p] + 1j * U_scat_im_all[p])
        ax.imshow(data, origin="lower", extent=[0, 1, 0, 1], cmap="hot",
                  vmin=0.0, vmax=max(float(np.percentile(data, 99)), 1e-12),
                  interpolation="bilinear")
    else:
        ax.text(0.5, 0.5, "not solved", ha="center", va="center", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Circle((cx, CY_VALS[p]), a, **circle_kw))
    ax.axhline(Y_SUB, **sub_kw)
    tag = " [TEST]" if p in set(test_idx.tolist()) else ""
    ax.set_title(f"gap = {GAP_VALS[p]:.3f}{tag}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

for q in range(N_PARAM, nrow * ncol):
    axes.ravel()[q].axis("off")

fig.suptitle("FDM |u_scat| across the gap sweep", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.8)
FIG = globals().get("FIG_OVERRIDE",
                    "variable_sample/figures/variable_sample_fdm_sweep.png")
plt.savefig(FIG, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {FIG}")
