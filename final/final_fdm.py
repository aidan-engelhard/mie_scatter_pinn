"""
FDM Sweep — AFM Tip + Substrate, 2-D sweep over HALF-ANGLE and TIP HEIGHT
=========================================================================
Combines variable_tips/tip_sweep_fdm.py (parametric tip sweep, vectorised
UPML assembly) with afm_pinn/afm_fdm.py (dielectric substrate, Fresnel
background, medium-aware PML), and adds a second sweep axis.

    half_angle : N_ALPHA  values in [10 deg, 30 deg]
    apex_y     : N_HEIGHT values in [APEX_Y_LO, APEX_Y_HI]

apex_y is the y of the LOWEST point of the tip, so the tip-sample gap is
apex_y - Y_SUB, spanning 0.01 to 0.20 for the defaults below. The two axes
are combined into a full tensor grid and FLATTENED to a single parameter
index p in [0, N_PARAM), N_PARAM = N_ALPHA * N_HEIGHT. Downstream that keeps
every per-parameter mechanism (train/test split, caches, batching) working
unchanged — the DeepONet just reads two values per index instead of one.

RESOLUTION WARNING. The smallest gap is 0.01 = lam/50. At N = 601 over
[-0.5, 1.5] the spacing is h = 0.0033, so that gap spans only ~3 cells. The
near-field in the narrowest configurations is therefore under-resolved and
those samples carry noticeably more discretisation error than the wide ones.
Raise N if the tightest gaps matter quantitatively; the cost is roughly
O(N^3) for the sparse solve.
Shape-parametric extension of afm_tip_fdm.py, structured exactly like
fdm_ellipse_sweep.py.

The scatterer is the conospherical AFM tip of afm_tip_fdm.py — a cone of
half-angle alpha opening upward, apex pointing down, capped by a circle of
radius R_APEX, shank running straight out through the top boundary and the
top PML. The sweep iterates over

    N_ALPHA half-angles x N_HEIGHT apex heights

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
    half_angle           : [N_PARAM]            cone half-angle per index
    apex_y               : [N_PARAM]            apex height per index
    alpha_idx/height_idx : [N_PARAM]            grid indices for reshaping
    alpha_deg_vals       : [N_ALPHA]            the swept angle values
    apex_y_vals          : [N_HEIGHT]           the swept height values
    U_bg_re/im, in_sub   : background + substrate mask (geometry-independent)
    U_scat_re/im         : [N_ALPHA, Nph, Nph]  scattered field (physical dom.)
    U_total_re/im        : [N_ALPHA, Nph, Nph]  total field
    in_tip               : [N_ALPHA, Nph, Nph]  interior masks
    X, Y                 : [Nph, Nph]           physical-domain grid
    train_idx, test_idx  : parameter-index split (test = interpolation angles)
    meta                 : [N_RE, N_IM, lam, k, TIP_X, R_APEX,
                            SUB_N_RE, SUB_N_IM, Y_SUB, N_ALPHA, N_HEIGHT,
                            theta_inc_deg]

Run this before final/final_deeponet.py.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os
import time

# ─────────────────────────────────────────────────────────────────────────────
# Optional GPU acceleration for the sparse solve, as in afm_pinn/afm_fdm.py.
# It matters more here than anywhere else in the repo: this script runs
# N_PARAM = N_ALPHA * N_HEIGHT = 100 independent solves of the same-sized
# system, so the solve is essentially the entire runtime (~1 h on CPU at
# N=601). Only the spsolve call changes; assembly stays vectorised on numpy.
# Falls back to scipy/CPU automatically if CuPy or a CUDA device is missing,
# or if the GPU solve raises (direct sparse solvers have far less memory
# headroom on GPU). Install with `pip install cupy-cudaXXx`.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
    import cupyx.scipy.sparse.linalg as cusparse_linalg
    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
    _GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)['name'].decode() if HAVE_GPU else ""
except Exception:
    HAVE_GPU = False
    _GPU_NAME = ""

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — material/wave identical to afm_tip_fdm.py / afm_tip_pinn.py
# ─────────────────────────────────────────────────────────────────────────────
N_RE   = 3.0
N_IM   = 30.0
# ── Substrate (dielectric half-space) — as in afm_pinn/afm_fdm.py ───────────
SUB_N_RE = 1.5
SUB_N_IM = 0.1
Y_SUB    = 0.4                # flat substrate interface height
# Unit mapping from the physical AFM problem: R_APEX = 0.05 <-> 25 nm tip
# apex radius, so 1 script-unit = 500 nm. Target illumination 5-10 um would map
# to lam = 10-20 here, but at that scale the unit domain is deeply
# sub-wavelength and the PINN/DeepONet training has consistently failed to
# converge (see afm_pinn_2/). lam = 1.0 matches the afm_pinn run that reached
# relL2 = 0.08 -- the longest wavelength the operator learning is known to
# handle on this domain. Must match final_deeponet.py's lam (asserted there).
lam    = globals().get("LAM_OVERRIDE", 1.0)
k      = 2.0 * np.pi / lam
theta_inc = -np.pi / 4
kx = k * np.cos(theta_inc)
ky = k * np.sin(theta_inc)

# ── AFM conospherical tip geometry (half-angle and apex height both swept) ──
TIP_X   = 0.5                 # x-coordinate of the tip axis
R_APEX  = 0.05                # apex radius of curvature

N = 601   # grid points over the FULL (physical + PML) domain.
          # Raised from variable_tips' 401 because the tightest tip-sample gap
          # is 0.01: at N=401 that is 2 cells, at N=601 it is 3.

# ── 2-D parameter sweep: half-angle x apex height ───────────────────────────
# N_ALPHA is 10 rather than variable_tips' 20 because this is now a tensor
# grid: 10 x 10 = 100 solves and ~55 MB of saved fields, against 200 solves
# and ~110 MB at 20 angles (over GitHub's per-file limit). Per-axis resolution
# of 10 is the same either way.
# Saved fields are subsampled by SAVE_STRIDE. The SOLVE stays at N=601 (needed
# for the 0.01 gap), but storing all 301^2 physical points for 100 combos comes
# to 141 MB — past GitHub's 100 MB per-file limit. Stride 2 gives 151^2, i.e.
# h_saved = 0.0067 = lam/75 (~15 points per wavelength), which is ample for the
# relL2 metric the DeepONet monitor computes by interpolation.
SAVE_STRIDE  = 2

N_ALPHA      = globals().get("N_ALPHA_OVERRIDE", 10)
N_HEIGHT     = globals().get("N_HEIGHT_OVERRIDE", 10)
ALPHA_DEG    = np.linspace(10.0, 30.0, N_ALPHA)
APEX_Y_LO    = 0.41           # lowest apex -> gap 0.01 (near contact)
APEX_Y_HI    = 0.60           # highest apex -> gap 0.20
APEX_Y_VALS  = np.linspace(APEX_Y_LO, APEX_Y_HI, N_HEIGHT)

# Flatten the tensor grid to a single parameter index. Order is alpha-major:
# p = ia * N_HEIGHT + ih, so half_angle_flat[p] and apex_y_flat[p] give the
# pair for index p. Everything downstream treats p as an opaque index.
_IA, _IH        = np.meshgrid(np.arange(N_ALPHA), np.arange(N_HEIGHT),
                              indexing='ij')
alpha_idx       = _IA.ravel()
height_idx      = _IH.ravel()
half_angle      = np.deg2rad(ALPHA_DEG)[alpha_idx]      # [N_PARAM], radians
apex_y_flat     = APEX_Y_VALS[height_idx]               # [N_PARAM]
N_PARAM         = N_ALPHA * N_HEIGHT
alpha_deg_flat  = ALPHA_DEG[alpha_idx]

# ── Train/test split ─────────────────────────────────────────────────────────
# Test cases must INTERPOLATE in BOTH parameters, so they are drawn only from
# grid points interior in alpha AND in height. With a 1-D sweep it was enough
# to exclude the two end angles; here the whole boundary of the 10x10 grid is
# excluded, leaving the 8x8 interior to draw from.
rng       = np.random.default_rng(0)
_int_mask = ((alpha_idx > 0) & (alpha_idx < N_ALPHA - 1) &
             (height_idx > 0) & (height_idx < N_HEIGHT - 1))
interior  = np.flatnonzero(_int_mask)
# N_TEST = 0 is the NO-SPLIT mode the data-quantity experiment uses: every
# solved point is training data, and generalisation is measured instead by
# evaluate_final.py at off-grid parameters, which solves a fresh FDM
# reference at each one. test_idx is then set to train_idx rather than left
# empty, so every downstream consumer (the DeepONet's relL2 monitor, the
# figures) keeps working -- but what it reports is TRAINING accuracy, not a
# held-out number. The banner below says so.
N_TEST    = globals().get("N_TEST_OVERRIDE", 12)
N_TEST    = min(N_TEST, len(interior))
if N_TEST > 0:
    test_idx  = np.sort(rng.choice(interior, size=N_TEST, replace=False))
    train_idx = np.array([p for p in range(N_PARAM)
                          if p not in set(test_idx)])
else:
    train_idx = np.arange(N_PARAM)
    test_idx  = train_idx.copy()

# ── SOLVE_ONLY ───────────────────────────────────────────────────────────────
# Which grid points actually get a linear solve. The PARAMETER GRID above stays
# 10x10 regardless: final_deeponet.py derives its branch feature normalisation
# (FEAT_MIN/FEAT_MAX) from the full ranges, and the grid indices drive its plot
# layout, so shrinking the grid would silently change the conditioning of a
# later sweep run. Only the expensive part is skipped.
#
# Set to None to solve the whole sweep. Set to a list of indices to solve just
# those; every other entry stays zero in the saved arrays (savez_compressed, so
# they cost nothing on disk) and the sweep figures are skipped, since 99 blank
# panels are not worth 50 minutes of CPU.
#
# The default matches final_deeponet.py's SINGLE_GEOMETRY selection: the grid
# point nearest afm_pinn's own tip, 20 deg at apex_y = 0.45. Keep the two in
# step -- if you change one, change the other, or the DeepONet will train
# against an array of zeros.
SOLVE_ONE_NEAREST = globals().get("SOLVE_ONE_NEAREST_OVERRIDE", (20.0, 0.45))
#                   (half-angle deg, apex_y); None = solve the whole sweep

if SOLVE_ONE_NEAREST is None:
    solve_idx = np.arange(N_PARAM)
else:
    _a_t, _h_t = SOLVE_ONE_NEAREST
    _d = ((alpha_deg_flat - _a_t) / max(np.ptp(alpha_deg_flat), 1e-9))**2 + \
         ((apex_y_flat   - _h_t) / max(np.ptp(apex_y_flat),   1e-9))**2
    solve_idx = np.array([int(np.argmin(_d))])

# ── PML parameters ────────────────────────────────────────────────────────
# d_pml/PML_R0 widened from 0.5/1e-8 to match afm_pinn/afm_fdm.py exactly. That
# script moved to 1.0/1e-10 after a corner-value check found a residual PML leak
# at the narrower setting: |U_total| dropped by ~0.02-0.03 at every corner
# regardless of local field strength. Since final_deeponet.py is now being
# compared line-for-line against the afm_pinn run, the reference it trains
# against has to have the same absorbing boundary. The full domain grows from
# 2x2 to 3x3, so at N = 601 the spacing goes 0.00333 -> 0.005 = lam/200 at
# lam = 1.0 -- ample for the exterior, and the interior skin (lam/188 = 0.0053)
# is marginally resolved either way at n = 3+30i.
d_pml     = 1.0
pml_order = 3
# sigma_max is NOT tied to k (see afm_pinn/afm_fdm.py for the derivation):
# round-trip PML reflection depends only on
# integral(sigma dx) = sigma_max*d_pml/(pml_order+1), wavelength-independent
# once the stretch factor uses a matched k_ref. The previous fixed "20.0" (the
# afm_tip_fdm.py convention) left ~0.7% round-trip reflection uncharacterised;
# this ties it to an explicit target instead, same value at any lam.
PML_R0    = 1e-10   # target round-trip reflection coefficient
sigma_max = -(pml_order + 1) * np.log(PML_R0) / (2.0 * d_pml)

# ── Derived substrate quantities + Fresnel background ───────────────────────
# s-polarisation (Ez out of plane): u AND du/dy continuous at y = Y_SUB, which
# is what this stencil enforces, so the background satisfies both matching
# conditions exactly and u_scat is purely tip-induced. Identical construction
# to afm_pinn/afm_fdm.py.
k_sub    = (SUB_N_RE + 1j * SUB_N_IM) * k
k_sub_sq = k_sub**2

ky_sub = np.sqrt(k_sub_sq - kx**2 + 0j)
if ky_sub.real > 0:
    ky_sub = -ky_sub          # ensure Re < 0: downward-propagating
q_air = abs(ky)
q_sub = -ky_sub               # Re(q_sub) > 0
r_TM  = (q_air - q_sub) / (q_air + q_sub)
t_TM  = (2.0 * q_air)   / (q_air + q_sub)

# ─────────────────────────────────────────────────────────────────────────────
# Signed distance function for the conospherical AFM tip, parametrised by the
# cone half-angle alpha. Same construction as afm_tip_fdm.py:
#   apex_centre_y = apex_y + R_APEX
#   cone_vertex_y = apex_centre_y - R_APEX / sin(alpha)
#   tangent_y     = apex_centre_y - R_APEX * sin(alpha)
# ─────────────────────────────────────────────────────────────────────────────
def _alpha_consts(alpha, apex_y):
    """Geometry constants for (half-angle, apex height). apex_y is the LOWEST
    point of the tip, so the apex sphere centre sits at apex_y + R_APEX."""
    sin_a = float(np.sin(alpha))
    cos_a = float(np.cos(alpha))
    apex_centre_y = apex_y + R_APEX
    cone_vertex_y = apex_centre_y - R_APEX / sin_a
    tangent_y     = apex_centre_y - R_APEX * sin_a
    return sin_a, cos_a, cone_vertex_y, tangent_y, apex_centre_y


def phi_np(x, y, alpha, apex_y):
    """Signed distance to the conospherical tip with half-angle alpha whose
    lowest point is at apex_y. Negative inside, positive outside."""
    sin_a, cos_a, cone_vertex_y, tangent_y, apex_centre_y = _alpha_consts(alpha, apex_y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    q_x    = np.abs(x - TIP_X)
    d_cone = q_x * cos_a - (y - cone_vertex_y) * sin_a
    d_apex = np.sqrt((x - TIP_X)**2 + (y - apex_centre_y)**2) - R_APEX
    return np.where(y < tangent_y, d_apex, d_cone)


def tip_outline_xy(alpha, apex_y, y_top=1.0, n=200):
    """(x, y) polyline tracing the tip surface for plot overlays."""
    sin_a, cos_a, cone_vertex_y, tangent_y, apex_centre_y = _alpha_consts(alpha, apex_y)
    yc = apex_centre_y
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
print("FDM SWEEP — AFM Tip + Substrate (TM, scattered-field + UPML)")
print("=" * 66)
print(f"  n            = {N_RE} + {N_IM}i   (matches afm_tip_pinn.py)")
print(f"  lam / k      = {lam} / {k:.4f}")
print(f"  tip          : axis x={TIP_X}, R={R_APEX}, kR={k*R_APEX:.3f}  (apex height swept)")
print(f"  half-angles  = {ALPHA_DEG[0]:.1f}° .. {ALPHA_DEG[-1]:.1f}°  "
      f"({N_ALPHA} values)")
print(f"  test combos  = " + ", ".join(
    f"{alpha_deg_flat[p]:.1f}°/gap{apex_y_flat[p]-Y_SUB:.2f}" for p in test_idx))
print(f"  PML: d={d_pml}, sigma_max={sigma_max:.2f} (R0 target={PML_R0:.0e}), order={pml_order}")
print(f"  GPU solve: {'enabled (' + _GPU_NAME + ')' if HAVE_GPU else 'disabled -- no CuPy/GPU found, using CPU'}")

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
      f"radius")
print(f"  Sweep: {N_ALPHA} half-angles × {N_HEIGHT} heights = {N_PARAM} solves")
print(f"  apex_y ∈ [{APEX_Y_LO}, {APEX_Y_HI}]  →  gap ∈ "
      f"[{APEX_Y_LO - Y_SUB:.2f}, {APEX_Y_HI - Y_SUB:.2f}]  "
      f"({(APEX_Y_LO - Y_SUB)/hx:.1f} cells at the tightest gap)")
print(f"  n_sub = {SUB_N_RE}+{SUB_N_IM}i,  Y_SUB = {Y_SUB},  |r_TM| = {abs(r_TM):.4f}")
print("  NO-SPLIT mode: every solved point is training data; "
      "generalisation comes from evaluate_final.py off-grid"
      if N_TEST == 0 else "")
print(f"  Train / Test = {len(train_idx)} / {len(test_idx)}  "
      f"(test interpolates in BOTH parameters)")
print("=" * 66)

# Background field (no tip): incident + Fresnel-reflected above the
# substrate, transmitted below. Appears only in the RHS and in u_total, never
# in the stencil, so there is no discretisation-consistency issue.
in_sub_all = Y_grid < Y_SUB          # substrate region, full grid incl. PML
_bg_air = (np.exp(1j * (kx * X_grid + ky * (Y_grid - Y_SUB))) +
           r_TM * np.exp(1j * (kx * X_grid - ky * (Y_grid - Y_SUB))))
_bg_sub = t_TM * np.exp(1j * (kx * X_grid + ky_sub * (Y_grid - Y_SUB)))
U_bg_grid = np.where(in_sub_all, _bg_sub, _bg_air)

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

# Medium-aware stretch (afm_pinn/afm_fdm.py): sigma/k_ref must be referenced
# to the LOCAL wavenumber or the substrate PML under-absorbs. k_ref depends
# only on position relative to Y_SUB, never on the tip material, which is why
# these arrays stay geometry-independent and can be built once.
SX = np.ones((N, N), dtype=complex)
SY = np.ones((N, N), dtype=complex)
for _i in range(N):
    _y = y_coords[_i]
    _k_ref_x = abs(k_sub) if _y < Y_SUB else k
    SX[_i, :] = 1.0 + 1j * sigma_x_arr / _k_ref_x
    # Bottom PML: the wave actually crossing it travels VERTICALLY with
    # wavenumber ky_sub, not the medium's full |k_sub| -- at 45 deg incidence
    # those differ by ~11%, which left the stretch under-matched to the true
    # local wave and degraded the achieved reflection below the nominal
    # PML_R0 target. Same fix as afm_pinn/afm_fdm.py.
    if _y < Y_LO:
        SY[_i, :] = 1.0 + 1j * sigma_y_arr[_i] / abs(ky_sub)  # bottom PML: sub
    elif _y > Y_HI:
        SY[_i, :] = 1.0 + 1j * sigma_y_arr[_i] / k            # top PML: air

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


_gpu_failed = [False]   # latch: report a GPU fallback once, not 100 times


def _solve(A, rhs):
    """Direct sparse solve, on GPU when available. See the CuPy note at the
    top -- with 100 solves in the sweep this is nearly all of the runtime."""
    if HAVE_GPU and not _gpu_failed[0]:
        try:
            A_gpu   = cusparse.csr_matrix(A.tocsr())
            rhs_gpu = cp.asarray(rhs)
            return cp.asnumpy(cusparse_linalg.spsolve(A_gpu, rhs_gpu))
        except Exception as e:
            _gpu_failed[0] = True
            print(f"  GPU solve failed ({type(e).__name__}: {e}) -- "
                  f"falling back to CPU for the rest of the sweep")
    return spsolve(A, rhs)


def assemble_and_solve(in_tip):
    """Assemble the UPML scattered-field system for a given tip mask and
    solve. Identical stencil to afm_tip_fdm.py, vectorised."""
    # Three media. Only in_tip varies across the sweep; the substrate is fixed,
    # so the geometry-independent precomputation above still holds.
    k2_int_flat = np.where(in_tip, k_int_sq,
                  np.where(in_sub_all, k_sub_sq, k_ext_sq))[interior_mask]
    diag = -2.0 * (cx_int + cy_int) + k2_int_flat * sxsy_int

    vals = np.concatenate([cx_int, cx_int, cy_int, cy_int, diag,
                           np.ones(len(I_bnd), dtype=complex)])
    A = coo_matrix((vals, (rows_static, cols_static)),
                   shape=(n_dof, n_dof)).tocsc()

    rhs = np.zeros(n_dof, dtype=complex)
    src = in_tip & interior_mask
    # Contrast source. The tip sits wholly in air (apex_y >= 0.41 > Y_SUB), so
    # the background k^2 under the tip is k_ext^2 and the contrast is delta_k2.
    rhs[IDX[src]] = -delta_k2 * SXSY[src] * U_bg_grid[src]

    return _solve(A, rhs).reshape(N, N)


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
U_bg_ph  = U_bg_grid[i_lo:i_hi, j_lo:j_hi]
in_sub_ph = Y_ph < Y_SUB

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
U_scat_re_all  = np.zeros((N_PARAM, N_ph, N_ph), dtype=np.float32)
U_scat_im_all  = np.zeros((N_PARAM, N_ph, N_ph), dtype=np.float32)
U_total_re_all = np.zeros((N_PARAM, N_ph, N_ph), dtype=np.float32)
U_total_im_all = np.zeros((N_PARAM, N_ph, N_ph), dtype=np.float32)
in_tip_all     = np.zeros((N_PARAM, N_ph, N_ph), dtype=bool)

t0 = time.time()
_solve_set = set(int(p) for p in solve_idx)
print(f"\nSolving {len(_solve_set)} of {N_PARAM} grid points"
      + ("" if len(_solve_set) == N_PARAM else
         f"  (SOLVE_ONE_NEAREST = {SOLVE_ONE_NEAREST}; the rest stay zero)"))

for p in range(N_PARAM):
    alpha  = float(half_angle[p])
    apex_y = float(apex_y_flat[p])

    # Tip material mask on the FULL grid (physical + PML) — the shank
    # automatically extends into the top PML, as in afm_tip_fdm.py.
    # Computed for EVERY p, solved or not: it is a geometry mask, costs no
    # solve, and downstream code indexes in_tip freely.
    in_tip = phi_np(X_grid, Y_grid, alpha, apex_y) < 0.0
    in_tip_all[p] = in_tip[i_lo:i_hi, j_lo:j_hi]

    if p not in _solve_set:
        continue

    t1 = time.time()
    V_full  = assemble_and_solve(in_tip)
    U_scat  = V_full[i_lo:i_hi, j_lo:j_hi]
    U_total = U_scat + U_bg_ph

    U_scat_re_all[p]  = U_scat.real;   U_scat_im_all[p]  = U_scat.imag
    U_total_re_all[p] = U_total.real;  U_total_im_all[p] = U_total.imag

    tag = "TEST " if p in set(test_idx) else "train"
    print(f"  [{p+1:>3}/{N_PARAM}] {tag}  alpha={alpha_deg_flat[p]:5.2f}°  "
          f"apex_y={apex_y:.3f} (gap {apex_y - Y_SUB:.3f})  "
          f"max|u_scat|={np.max(np.abs(U_scat)):.3f}  ({time.time()-t1:.1f}s)")

print(f"\nSweep done in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR = globals().get("INPUT_DIR_OVERRIDE", "final/inputs")
OUT_NPZ   = globals().get("OUT_NPZ_OVERRIDE",
                          f"{INPUT_DIR}/fdm_final_sweep.npz")
os.makedirs(INPUT_DIR, exist_ok=True)
np.savez_compressed(
    OUT_NPZ,
    half_angle=half_angle.astype(np.float32),
    apex_y=apex_y_flat.astype(np.float32),
    alpha_idx=alpha_idx.astype(np.int32),
    height_idx=height_idx.astype(np.int32),
    # Which indices actually carry a solved field. Every other entry in
    # U_scat_*/U_total_* is zero, so a consumer that picks its own index by a
    # geometric rule can silently end up training against zeros -- which is
    # exactly what happened when final_deeponet.py duplicated the
    # SOLVE_ONE_NEAREST rule: 20 deg lies exactly halfway between the 18.89 and
    # 21.11 grid values and float rounding broke the tie the other way there.
    # Read this array instead of re-deriving the choice.
    solved_idx=np.asarray(solve_idx, dtype=np.int32),
    alpha_deg_vals=ALPHA_DEG.astype(np.float32),
    apex_y_vals=APEX_Y_VALS.astype(np.float32),
    U_bg_re=U_bg_ph.real[::SAVE_STRIDE, ::SAVE_STRIDE].astype(np.float32),
    U_bg_im=U_bg_ph.imag[::SAVE_STRIDE, ::SAVE_STRIDE].astype(np.float32),
    in_sub=in_sub_ph[::SAVE_STRIDE, ::SAVE_STRIDE],
    save_stride=np.int32(SAVE_STRIDE),
    U_scat_re=U_scat_re_all[:, ::SAVE_STRIDE, ::SAVE_STRIDE],
    U_scat_im=U_scat_im_all[:, ::SAVE_STRIDE, ::SAVE_STRIDE],
    U_total_re=U_total_re_all[:, ::SAVE_STRIDE, ::SAVE_STRIDE],
    U_total_im=U_total_im_all[:, ::SAVE_STRIDE, ::SAVE_STRIDE],
    in_tip=in_tip_all[:, ::SAVE_STRIDE, ::SAVE_STRIDE],
    X=X_ph[::SAVE_STRIDE, ::SAVE_STRIDE].astype(np.float32),
    Y=Y_ph[::SAVE_STRIDE, ::SAVE_STRIDE].astype(np.float32),
    train_idx=train_idx, test_idx=test_idx,
    meta=np.array([N_RE, N_IM, lam, k, TIP_X, R_APEX,
                   SUB_N_RE, SUB_N_IM, Y_SUB,
                   N_ALPHA, N_HEIGHT, np.degrees(theta_inc)]))
print(f"Saved: {OUT_NPZ}  "
      f"(fields {N_PARAM}x{X_ph[::SAVE_STRIDE].shape[0]}^2, stride {SAVE_STRIDE} from the {N_ph}^2 solve)")

# ─────────────────────────────────────────────────────────────────────────────
# Overview figures — four 4×5 grids over the half-angle sweep:
#   Re(u_scat), |u_scat|, Re(u_total), |u_total|
# Panels ordered by increasing half-angle (left→right, top→bottom).
# ─────────────────────────────────────────────────────────────────────────────
FIG_DIR = globals().get("FIG_DIR_OVERRIDE", "final/figures")
os.makedirs(FIG_DIR, exist_ok=True)
matplotlib.rcParams.update({'font.size': 7})
# One row per half-angle, one column per height: the grid layout mirrors the
# parameter grid, so trends along either axis are read off directly.
N_ROWS, N_COLS = N_ALPHA, N_HEIGHT
outline_kw = dict(color='cyan',   lw=1.0, zorder=5, ls='--')
sub_kw     = dict(color='yellow', lw=0.8, zorder=6, ls='--')


def plot_sweep_grid(get_data, title, cmap, symmetric, fname):
    """4×5 panel grid over all half-angles.
    get_data(p) returns the [N_ph, N_ph] field to show for combo p."""
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(2.4 * N_COLS, 2.4 * N_ROWS))
    for p in range(N_PARAM):
        ax_plot = axes[alpha_idx[p], height_idx[p]]
        data  = get_data(p)
        scale = max(float(np.percentile(np.abs(data), 98)), 1e-12)
        vmin  = -scale if symmetric else 0.0
        ax_plot.imshow(data, origin='lower', extent=[0, 1, 0, 1],
                       cmap=cmap, vmin=vmin, vmax=scale)
        ox, oy = tip_outline_xy(float(half_angle[p]), float(apex_y_flat[p]))
        ax_plot.plot(ox, oy, **outline_kw)
        ax_plot.axhline(Y_SUB, **sub_kw)
        tag = " [TEST]" if p in set(test_idx) else ""
        ax_plot.set_title(f"α={alpha_deg_flat[p]:.1f}°  "
                          f"gap={apex_y_flat[p] - Y_SUB:.2f}{tag}", fontsize=6)
        ax_plot.set_xticks([]); ax_plot.set_yticks([])
        ax_plot.set_xlim(0, 1); ax_plot.set_ylim(0, 1)
    fig.suptitle(f"FDM AFM-tip + substrate sweep — {title},  "
                 f"n_tip={N_RE}+{N_IM}i,  n_sub={SUB_N_RE}+{SUB_N_IM}i,  "
                 f"λ={lam}   (rows: half-angle, cols: tip-sample gap)",
                 fontsize=11)
    plt.tight_layout()
    path = f"{FIG_DIR}/{fname}.png"
    plt.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


if len(_solve_set) < N_PARAM:
    print(f"\nSkipped: the four sweep-grid figures "
          f"({N_PARAM - len(_solve_set)} of {N_PARAM} panels would be blank)")
    _p = int(solve_idx[0])
    print(f"  solved point: p={_p}  alpha={alpha_deg_flat[_p]:.2f}°  "
          f"apex_y={apex_y_flat[_p]:.4f}  (gap {apex_y_flat[_p] - Y_SUB:.4f})")
else:
    plot_sweep_grid(lambda p: U_scat_re_all[p],
                    "Re(u_scat)", "RdBu_r", True,
                    "fdm_final_sweep_scat_re")
    plot_sweep_grid(lambda p: np.hypot(U_scat_re_all[p], U_scat_im_all[p]),
                    "|u_scat|", "hot", False,
                    "fdm_final_sweep_scat_mag")
    plot_sweep_grid(lambda p: U_total_re_all[p],
                    "Re(u_total)", "RdBu_r", True,
                    "fdm_final_sweep_total_re")
    plot_sweep_grid(lambda p: np.hypot(U_total_re_all[p], U_total_im_all[p]),
                    "|u_total|", "inferno", False,
                    "fdm_final_sweep_total_mag")

print("\nDone.")