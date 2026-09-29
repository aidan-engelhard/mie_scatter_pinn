"""
Post-hoc accuracy evaluation of final/models/final_deeponet_weights.pt
=====================================================================
Rebuilds the trained DeepONet from its checkpoint (no retraining) and reports
accuracy in two regimes:

  1. ON-GRID   — the (half-angle, apex height) combinations the FDM sweep
                 solved. Reference comes straight from
                 final/inputs/fdm_final_sweep.npz. Reported as a
                 N_ALPHA x N_HEIGHT table so trends along either axis, and any
                 penalty at the tightest gaps, are visible at a glance.

  2. OFF-GRID  — parameter values the network never saw, DEFAULTING TO THE
                 CELL CENTRES of the training grid, i.e. maximally far from
                 every training point in both axes simultaneously.

WHY OFF-GRID NEEDS A FRESH SOLVE. The npz only holds the grid points, so the
only honest reference at an off-grid parameter is a new FDM solution AT that
parameter. This script therefore re-runs final_fdm.py's own solver with its
parameter arrays replaced by the requested pairs — same stencil, same PML, same
background, guaranteed by construction rather than by a reimplementation that
could drift. The cost is one sparse solve per point (~48 s at N = 601), which
is why the default off-grid set is small and configurable.

Interpolating the FDM across the parameter grid would NOT be a valid reference:
it would measure how close the DeepONet is to linear interpolation of the
training data, not how close it is to the truth.

Usage
-----
    python final/evaluate_final.py                  # on-grid + cell centres
    python final/evaluate_final.py --no-offgrid     # on-grid only (seconds)
    python final/evaluate_final.py --gap-scan 7     # + continuous gap line scan
    python final/evaluate_final.py --n-offgrid 4    # fewer off-grid points

Outputs
-------
    final/figures/eval_ongrid_relL2.png    relL2 heat map over the grid
    final/figures/eval_offgrid.png         off-grid fields vs FDM + errors
    final/figures/eval_gap_scan.png        relL2 and |u| vs gap (if requested)
"""

import argparse
import os
import re
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# Default; --ckpt points at another, e.g. final/models/final_mlp_weights.pt
CKPT_PATH = "final/models/final_deeponet_weights.pt"
FDM_PATH  = "final/inputs/fdm_final_sweep.npz"
FDM_SRC   = "final/final_fdm.py"
FIG_DIR   = "final/figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint -> constants + networks
#
# Every constant is read from the checkpoint rather than restated here. That is
# deliberate: the diagnostics written during development each re-derived things
# like B_int's scaling and the feature normalisation by hand, and any one of
# those copies could have silently measured the wrong thing. If the training
# script's architecture changes, this script follows automatically or fails
# loudly on a missing key.
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(path=None):
    path = path or CKPT_PATH
    if not os.path.exists(path):
        raise SystemExit(f"No checkpoint at {path} — train first.")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    hp = ck["hyperparams"]
    print("=" * 74)
    print("Loaded checkpoint")
    print("=" * 74)
    if hp.get("ARCH", "deeponet") == "mlp":
        print(f"  ARCH = mlp: {hp['MLP_HIDDEN']}x{hp['MLP_NEURONS']} "
              f"on [fourier(x,y), params]")
    else:
        print(f"  ARCH = deeponet: branch {hp['BRANCH_HIDDEN']}x"
              f"{hp['BRANCH_NEURONS']} (in {hp['PHYS_DIM']})  |  trunk "
              f"{hp['TRUNK_HIDDEN']}x{hp['TRUNK_NEURONS']}  |  "
              f"dot {hp['BRANCH_DIM']}")
    print(f"  SKIN_ANSATZ = {hp.get('SKIN_ANSATZ', False)}")
    print(f"  n_tip = {hp['N_RE']}+{hp['N_IM']}i   "
          f"n_sub = {hp['SUB_N_RE']}+{hp['SUB_N_IM']}i   Y_SUB = {hp['Y_SUB']}")
    print(f"  trained: {hp['ADAM_EPOCHS']} Adam epochs, "
          f"{hp['LBFGS_STEPS']} L-BFGS closure evals")
    print(f"  parameter combos in checkpoint: {len(ck['half_angle'])}")
    return ck, hp


class _Const:
    """Physics/geometry constants derived from the checkpoint hyperparameters."""
    def __init__(self, hp):
        self.N_RE, self.N_IM = hp["N_RE"], hp["N_IM"]
        self.SUB_N_RE, self.SUB_N_IM = hp["SUB_N_RE"], hp["SUB_N_IM"]
        self.Y_SUB   = hp["Y_SUB"]
        self.lam     = hp["lam"]
        self.k       = 2.0 * np.pi / self.lam
        self.TIP_X   = hp["TIP_X"]
        self.R_APEX  = hp["R_APEX"]
        # Whether net_int's output is u_int itself or the envelope A -- see
        # predict_scat. Absent on checkpoints written before the flag existed,
        # which is exactly the plain-net_int case.
        self.SKIN_ANSATZ = bool(hp.get("SKIN_ANSATZ", False))
        self.theta   = -np.pi / 4          # fixed across this whole repo
        self.kx      = self.k * np.cos(self.theta)
        self.ky      = self.k * np.sin(self.theta)
        # Fresnel, s-polarisation (matches final_fdm.py / final_deeponet.py)
        ky_sub = np.sqrt((self.SUB_N_RE + 1j * self.SUB_N_IM) ** 2 * self.k ** 2
                         - self.kx ** 2 + 0j)
        if ky_sub.real > 0:
            ky_sub = -ky_sub
        self.ky_sub = ky_sub
        q_air, q_sub = abs(self.ky), -ky_sub
        self.r_TM = (q_air - q_sub) / (q_air + q_sub)
        self.t_TM = (2.0 * q_air) / (q_air + q_sub)

    # ── tip geometry, parametric in (alpha, apex_y) ──────────────────────────
    def consts(self, alpha, apex_y):
        s, c = float(np.sin(alpha)), float(np.cos(alpha))
        centre = apex_y + self.R_APEX
        return s, c, centre - self.R_APEX / s, centre - self.R_APEX * s, centre

    def phi(self, x, y, alpha, apex_y):
        s, c, cone_v, tan_y, centre = self.consts(alpha, apex_y)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d_cone = np.abs(x - self.TIP_X) * c - (y - cone_v) * s
        d_apex = np.sqrt((x - self.TIP_X) ** 2 + (y - centre) ** 2) - self.R_APEX
        return np.where(y < tan_y, d_apex, d_cone)

    def outline(self, alpha, apex_y, y_top=1.0, n=200):
        s, c, cone_v, tan_y, centre = self.consts(alpha, apex_y)
        ys = np.linspace(tan_y, y_top, n)
        hw = (ys - cone_v) * np.tan(alpha)
        ang = np.linspace(-np.pi / 2 - (np.pi / 2 - alpha),
                          -np.pi / 2 + (np.pi / 2 - alpha), n)
        xs = np.concatenate([(self.TIP_X + hw)[::-1],
                            self.TIP_X + self.R_APEX * np.cos(ang),
                            self.TIP_X - hw])
        yy = np.concatenate([ys[::-1],
                            centre + self.R_APEX * np.sin(ang), ys])
        return xs, yy

    def u_bg(self, X, Y):
        """Fresnel background on a numpy meshgrid."""
        air = (np.exp(1j * (self.kx * X + self.ky * (Y - self.Y_SUB))) +
               self.r_TM * np.exp(1j * (self.kx * X
                                        - self.ky * (Y - self.Y_SUB))))
        sub = self.t_TM * np.exp(1j * (self.kx * X
                                       + self.ky_sub * (Y - self.Y_SUB)))
        return np.where(Y < self.Y_SUB, sub, air)


def build_nets(ck, hp):
    """Rebuild the trained networks exactly as final_deeponet.py defines them,
    for whichever architecture the checkpoint records."""
    PD   = hp["PHYS_DIM"]
    BD   = hp["BRANCH_DIM"]
    ARCH = hp.get("ARCH", "deeponet")   # absent on pre-ARCH checkpoints
    if hp["TRUNK_SEES_SDF"]:
        raise SystemExit("TRUNK_SEES_SDF checkpoints need phi in the trunk; "
                         "not supported by this evaluator.")

    # Trunk input width comes from the SAVED B matrix, not from hp["N_FOURIER"].
    # The embedding is cat([cos(xy @ B), sin(xy @ B)]), so it is 2 * B.shape[1]
    # wide -- and final_deeponet.py now concatenates a second, tip-scale band
    # into B (N_FOURIER @ |k| + N_FOURIER_TIP @ 1/R_APEX), which 2 * N_FOURIER
    # silently under-counts. Reading the matrix cannot drift from the checkpoint
    # however many bands are added later.
    def _trunk_in(B):
        return 2 * int(B.shape[1]) + (1 if hp["TRUNK_SEES_SDF"] else 0)

    def mlp(d_in, d_hid, n_hid, d_out):
        layers = [nn.Linear(d_in, d_hid), nn.Tanh()]
        for _ in range(n_hid - 1):
            layers += [nn.Linear(d_hid, d_hid), nn.Tanh()]
        return nn.Sequential(*(layers + [nn.Linear(d_hid, d_out)]))

    class DON(nn.Module):
        def __init__(self, B):
            super().__init__()
            self.B      = B
            self.branch = nn.Module()
            self.branch.net = mlp(PD, hp["BRANCH_NEURONS"],
                                  hp["BRANCH_HIDDEN"], BD)
            self.trunk  = nn.Module()
            self.trunk.net = mlp(_trunk_in(B), hp["TRUNK_NEURONS"],
                                 hp["TRUNK_HIDDEN"], BD)
            # TRUNK_NORM in final_deeponet.py: a parameter-free LayerNorm on the
            # trunk output. Absent on checkpoints written before it existed, and
            # False means the plain trunk -- either way this must match how the
            # model was TRAINED or the rebuilt field is not the trained one.
            # eps defaults to 1e-5 for checkpoints written before it was made
            # explicit, which is what those were trained with.
            self.norm = (nn.LayerNorm(BD, eps=hp.get("TRUNK_NORM_EPS", 1e-5),
                                      elementwise_affine=False)
                         if hp.get("TRUNK_NORM", False) else None)
            self.bias_re = nn.Parameter(torch.zeros(1))
            self.bias_im = nn.Parameter(torch.zeros(1))
            self.half    = BD // 2

        def forward(self, pf, xy):
            proj = xy @ self.B
            e    = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            b    = self.branch.net(pf)
            if hp.get("BRANCH_NORM", False):   # see BRANCH_NORM in the trainer
                b = b * (BD ** 0.5
                         / b.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            t    = self.trunk.net(e)
            if self.norm is not None:
                t = self.norm(t)
            bt   = b * t
            return (bt[:, :self.half].sum(-1, keepdim=True) + self.bias_re,
                    bt[:, self.half:].sum(-1, keepdim=True) + self.bias_im)

    class MLPN(nn.Module):
        """ARCH == 'mlp': u = MLP([fourier(x,y), params]). Same (pf, xy)
        signature as DON so predict_scat is untouched."""
        def __init__(self, B):
            super().__init__()
            self.B = B
            self.net = mlp(_trunk_in(B) + PD, hp["MLP_NEURONS"],
                           hp["MLP_HIDDEN"], 2)

        def forward(self, pf, xy):
            proj = xy @ self.B
            e = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            p = pf if pf.shape[0] == e.shape[0] else pf.expand(e.shape[0], -1)
            out = self.net(torch.cat([e, p], dim=-1))
            return out[:, 0:1], out[:, 1:2]

    _Net     = MLPN if ARCH == "mlp" else DON
    net_int  = _Net(ck["B_int"].to(DEVICE)).to(DEVICE)
    net_scat = _Net(ck["B_ext"].to(DEVICE)).to(DEVICE)
    net_int.load_state_dict(ck["net_int_state_dict"])
    net_scat.load_state_dict(ck["net_scat_state_dict"])
    net_int.eval(); net_scat.eval()
    return net_int, net_scat


def make_features(ck, C):
    """param_to_features, reproduced from the checkpoint's normalisation."""
    fmin = np.asarray(ck["FEAT_MIN"], dtype=np.float32)
    fmax = np.asarray(ck["FEAT_MAX"], dtype=np.float32)

    def raw(alpha, apex_y):
        _, _, cone_v, _, _ = C.consts(alpha, apex_y)
        gap = apex_y - C.Y_SUB
        return np.array([
            alpha, np.sin(alpha), np.cos(alpha), np.tan(alpha),
            (1.0 - cone_v) * np.tan(alpha),
            gap, np.log(max(gap, 1e-6)),
            np.cos(2.0 * abs(C.ky) * gap),
        ], dtype=np.float32)

    def feats(alpha, apex_y):
        nd = 2.0 * (raw(alpha, apex_y) - fmin) / (fmax - fmin + 1e-10) - 1.0
        return torch.tensor(nd, device=DEVICE).unsqueeze(0)

    return raw, feats


# ─────────────────────────────────────────────────────────────────────────────
# Field prediction and the metric
# ─────────────────────────────────────────────────────────────────────────────
def predict_scat(net_int, net_scat, feats, C, alpha, apex_y, X, Y):
    """Composite u_scat = u_total - u_bg on the given grid, matching exactly the
    definition final_deeponet.py and final_fdm.py both use."""
    pf = feats(alpha, apex_y)
    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1).astype(np.float32),
                      device=DEVICE)
    with torch.no_grad():
        iR, iI = net_int(pf, xy)
        sR, sI = net_scat(pf, xy)
    shp = X.shape
    to_c = lambda R, I: (R.cpu().numpy().reshape(shp)
                         + 1j * I.cpu().numpy().reshape(shp))
    U_int, U_sc = to_c(iR, iI), to_c(sR, sI)

    # SKIN_ANSATZ checkpoints store the ENVELOPE A, not u_int: the training
    # script applies u_int = A * exp(-i k_int phi) in a wrapper that holds no
    # parameters and so leaves no trace in the state_dict. Reading the flag and
    # applying the same factor here is what keeps the two consistent; without it
    # the interior field -- and hence the composite inside the tip -- is wrong by
    # the whole boundary layer.
    if C.SKIN_ANSATZ:
        k_int = (C.N_RE + 1j * C.N_IM) * C.k
        # clamped at 0 for the same reason the training wrapper clamps: outside
        # the tip the exponent is positive and would overflow, and U_int is not
        # used there anyway
        U_int = U_int * np.exp(-1j * k_int *
                               np.minimum(C.phi(X, Y, alpha, apex_y), 0.0))
    bg      = C.u_bg(X, Y)
    in_tip  = C.phi(X, Y, alpha, apex_y) < 0.0
    U_total = np.where(in_tip, U_int, bg + U_sc)
    return U_total - bg, U_total, in_tip


def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Off-grid reference: re-run final_fdm.py's solver at arbitrary pairs
# ─────────────────────────────────────────────────────────────────────────────
def fdm_at_pairs(pairs, quiet=True, lam=None):
    """Solve the FDM at explicit (alpha_rad, apex_y) pairs.

    Executes final_fdm.py with only its parameter-array construction replaced,
    and its save/plot sections disabled. Same stencil, PML, background and
    resolution as the training reference by construction. `lam` defaults to
    final_fdm.py's own; pass the checkpoint's to be certain they agree.
    """
    import contextlib
    import io

    src = pathlib.Path(FDM_SRC).read_text(encoding="utf-8")

    anchor = "_IA, _IH        = np.meshgrid(np.arange(N_ALPHA), np.arange(N_HEIGHT),"
    assert anchor in src, "final_fdm.py's parameter block moved; update anchor"
    head, tail = src.split(anchor, 1)
    tail = tail.split("alpha_deg_flat  = ALPHA_DEG[alpha_idx]", 1)[1]

    inject = f"""
# —— injected by evaluate_final.py: explicit off-grid parameter pairs ——
half_angle     = np.array({[float(a) for a, _ in pairs]!r})
apex_y_flat    = np.array({[float(h) for _, h in pairs]!r})
N_PARAM        = len(half_angle)
alpha_idx      = np.zeros(N_PARAM, dtype=int)
height_idx     = np.arange(N_PARAM)
alpha_deg_flat = np.degrees(half_angle)
"""
    src = head + inject + tail

    # final_fdm.py now exposes, as globals().get() hooks, everything this
    # function used to patch by text surgery, so those are set in the exec
    # namespace below instead of edited into the source:
    #
    #   N_TEST_OVERRIDE = 0              no train/test split. The regex that
    #                                    used to delete the split block matched
    #                                    only up to its first branch once that
    #                                    block became an if/else, leaving the
    #                                    rest dangling (a SyntaxError).
    #   SOLVE_ONE_NEAREST_OVERRIDE=None  solve EVERY requested pair. Its default,
    #                                    (20 deg, apex_y 0.45), solves only the
    #                                    single pair nearest that tip and leaves
    #                                    the rest as all-zero fields. Every
    #                                    multi-point call since 79dcde0 did
    #                                    exactly that: final/figures/
    #                                    eval_offgrid.png (2026-08-28) holds one
    #                                    real reference, relL2 0.374, and four
    #                                    blank ones scored at relL2 ~1.55e14.
    #   LAM_OVERRIDE                     the caller's wavelength, if given.
    #
    # Truncate at the first line of the save section -- the INPUT_DIR hook --
    # so neither the npz nor the sweep figures are written.
    cut = 'INPUT_DIR = globals().get("INPUT_DIR_OVERRIDE", "final/inputs")'
    assert cut in src, "final_fdm.py's save section moved; update the cut anchor"
    src = src.split(cut)[0]

    g = {"__name__": "__main__",
         "N_TEST_OVERRIDE": 0,
         "SOLVE_ONE_NEAREST_OVERRIDE": None}
    if lam is not None:
        g["LAM_OVERRIDE"] = float(lam)
    code = compile(src, "<fdm_offgrid>", "exec")
    # redirect_stdout, not the previous rebinding of builtins.print inside the
    # executed source: that restore line never ran if the solve raised or was
    # interrupted, which left print() disabled for the rest of the process.
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, g)
    else:
        exec(code, g)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────
def report_ongrid(ck, C, net_int, net_scat, feats):
    d   = np.load(FDM_PATH)
    X, Y = d["X"], d["Y"]
    ref  = d["U_scat_re"] + 1j * d["U_scat_im"]
    ha, ay = d["half_angle"].astype(float), d["apex_y"].astype(float)
    ai, hi = d["alpha_idx"], d["height_idx"]
    adeg, ayv = d["alpha_deg_vals"], d["apex_y_vals"]
    tst = set(int(t) for t in d["test_idx"])
    NA, NH = len(adeg), len(ayv)

    # Only score grid points the FDM actually solved. With final_fdm.py's
    # SOLVE_ONE_NEAREST the other entries are all zeros, and rel_l2 divides by
    # ||ref||, so they come out as ~1e13 and swamp the one real number -- as
    # well as poisoning the train/held-out means. Prefer the solved_idx the
    # sweep records; fall back to a zero-norm test for sweeps predating it.
    if "solved_idx" in d.files:
        solved = set(int(s) for s in np.asarray(d["solved_idx"]).ravel())
    else:
        solved = set(p for p in range(len(ha))
                     if np.linalg.norm(ref[p]) > 0.0)
    if len(solved) < len(ha):
        print(f"\n  NOTE: {len(ha) - len(solved)} of {len(ha)} grid points carry "
              f"no FDM solution and are skipped (shown as '.').")

    errs = np.full(len(ha), np.nan)
    for p in sorted(solved):
        pred, _, _ = predict_scat(net_int, net_scat, feats, C,
                                  ha[p], ay[p], X, Y)
        errs[p] = rel_l2(pred, ref[p])

    grid = np.full((NA, NH), np.nan)
    for p in range(len(ha)):
        grid[ai[p], hi[p]] = errs[p]

    print("\n" + "=" * 74)
    print("ON-GRID relL2  (rows: half-angle, cols: tip-sample gap)")
    print("=" * 74)
    print("   alpha \\ gap " + "".join(f"{v - C.Y_SUB:7.3f}" for v in ayv))
    for iA in range(NA):
        print(f"   {adeg[iA]:6.1f}°     " +
              "".join("      ." if np.isnan(grid[iA, iH])
                      else f"{grid[iA, iH]:7.3f}" for iH in range(NH)))
    tr = [errs[p] for p in sorted(solved) if p not in tst]
    te = [errs[p] for p in sorted(solved) if p in tst]
    print("-" * 74)
    if tr:
        print(f"  train mean {np.mean(tr):.4f}   (over {len(tr)} solved)")
    if te:
        print(f"  held-out mean {np.mean(te):.4f}   held-out max "
              f"{np.max(te):.4f}   (over {len(te)} solved)")
    if not te:
        print("  no solved point falls in the held-out split -- a "
              "single-geometry run makes no generalisation claim.")
    gaps = ayv - C.Y_SUB
    # Only meaningful when both end columns carry solved points; with
    # SOLVE_ONE_NEAREST they are empty and nanmean warns on an empty slice.
    if not np.all(np.isnan(grid[:, 0])) and not np.all(np.isnan(grid[:, -1])):
        print(f"  by gap: tightest ({gaps[0]:.3f}) mean "
              f"{np.nanmean(grid[:, 0]):.4f}"
              f"   widest ({gaps[-1]:.3f}) mean {np.nanmean(grid[:, -1]):.4f}")
        print("  (a tightest-gap penalty is expected: the FDM reference itself "
              "only resolves\n   that gap with ~3 cells, so part of any gap "
              "trend is the reference, not the net)")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[gaps[0], gaps[-1], adeg[0], adeg[-1]])
    for p in range(len(ha)):
        if p in tst:
            ax.plot(ay[p] - C.Y_SUB, np.degrees(ha[p]), "r*", ms=14,
                    mec="white", zorder=5)
    ax.set_xlabel("tip-sample gap"); ax.set_ylabel("half-angle [deg]")
    ax.set_title("On-grid relL2 vs FDM   (red stars = held out from training)")
    plt.colorbar(im, ax=ax, label="relL2")
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(f"{FIG_DIR}/eval_ongrid_relL2.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/eval_ongrid_relL2.png")
    return adeg, ayv


def report_offgrid(C, net_int, net_scat, feats, adeg, ayv, n_pts):
    """Cell centres of the training grid: maximally far from every training
    point in BOTH axes at once, which is the hardest interpolation ask."""
    ia = np.linspace(0, len(adeg) - 2, n_pts).round().astype(int)
    ih = np.linspace(0, len(ayv) - 2, n_pts).round().astype(int)
    pairs = [(float(np.deg2rad(0.5 * (adeg[a] + adeg[a + 1]))),
              float(0.5 * (ayv[h] + ayv[h + 1]))) for a, h in zip(ia, ih)]

    print("\n" + "=" * 74)
    print(f"OFF-GRID: {len(pairs)} cell centres, fresh FDM solve at each")
    print("=" * 74)
    for a, h in pairs:
        print(f"   alpha={np.degrees(a):6.2f}°  apex_y={h:.4f}  "
              f"gap={h - C.Y_SUB:.4f}")
    print("  solving ...", flush=True)
    t0 = time.time()
    g = fdm_at_pairs(pairs)
    print(f"  {len(pairs)} solves in {time.time() - t0:.0f}s")

    X, Y  = g["X_ph"], g["Y_ph"]
    ref_r, ref_i = g["U_scat_re_all"], g["U_scat_im_all"]

    print(f"\n   {'alpha':>8}{'gap':>9}{'relL2':>9}{'max|u_s| FDM':>14}"
          f"{'max|u_s| DON':>14}")
    print("   " + "-" * 54)
    errs, preds, refs = [], [], []
    for i, (a, h) in enumerate(pairs):
        ref = ref_r[i] + 1j * ref_i[i]
        pred, _, _ = predict_scat(net_int, net_scat, feats, C, a, h, X, Y)
        e = rel_l2(pred, ref)
        errs.append(e); preds.append(pred); refs.append(ref)
        print(f"   {np.degrees(a):8.2f}{h - C.Y_SUB:9.4f}{e:9.4f}"
              f"{np.abs(ref).max():14.4f}{np.abs(pred).max():14.4f}")
    print("   " + "-" * 54)
    print(f"   off-grid mean relL2 = {np.mean(errs):.4f}")

    ncol = len(pairs)
    fig, axes = plt.subplots(3, ncol, figsize=(4.2 * ncol, 11), squeeze=False)
    for i, (a, h) in enumerate(pairs):
        vmax = max(float(np.percentile(np.abs(refs[i].real), 98)), 1e-10)
        kw   = dict(origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax)
        ox, oy = C.outline(a, h)
        for row, data, ttl in ((0, preds[i].real, "DeepONet Re(u_scat)"),
                               (1, refs[i].real,  "FDM Re(u_scat)")):
            ax = axes[row][i]
            im = ax.imshow(data, **kw)
            ax.plot(ox, oy, "c--", lw=1.0)
            ax.axhline(C.Y_SUB, color="yellow", ls="--", lw=1.0)
            ax.set_title(f"{ttl}\nα={np.degrees(a):.2f}° gap={h-C.Y_SUB:.3f}"
                         f"  relL2={errs[i]:.3f}", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            plt.colorbar(im, ax=ax, shrink=0.8)
        diff = preds[i].real - refs[i].real
        dmax = max(float(np.percentile(np.abs(diff), 99)), 1e-10)
        ax = axes[2][i]
        im = ax.imshow(diff, origin="lower", extent=[0, 1, 0, 1], cmap="PiYG",
                       vmin=-dmax, vmax=dmax)
        ax.plot(ox, oy, "c--", lw=1.0)
        ax.axhline(C.Y_SUB, color="yellow", ls="--", lw=1.0)
        ax.set_title("DeepONet − FDM", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("OFF-GRID parameters (never trained on) — fresh FDM reference",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eval_offgrid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {FIG_DIR}/eval_offgrid.png")


def report_gap_scan(C, net_int, net_scat, feats, adeg, ayv, n_pts):
    """Continuous line scan in gap at a fixed mid-grid angle. The background is
    a standing wave of period pi/|ky|, so relL2 is worth inspecting against
    that scale rather than assuming it grows monotonically with distance."""
    alpha = float(np.deg2rad(0.5 * (adeg[len(adeg) // 2 - 1]
                                    + adeg[len(adeg) // 2])))
    heights = np.linspace(ayv[0], ayv[-1], n_pts)
    pairs = [(alpha, float(h)) for h in heights]
    period = np.pi / abs(C.ky)
    print("\n" + "=" * 74)
    print(f"GAP SCAN at alpha = {np.degrees(alpha):.2f}° (off-grid angle), "
          f"{n_pts} points")
    print(f"  background standing-wave period = {period:.4f}")
    print("=" * 74)
    print("  solving ...", flush=True)
    t0 = time.time()
    g = fdm_at_pairs(pairs)
    print(f"  {n_pts} solves in {time.time() - t0:.0f}s")

    X, Y = g["X_ph"], g["Y_ph"]
    rr, ri = g["U_scat_re_all"], g["U_scat_im_all"]
    errs, amp_f, amp_d = [], [], []
    for i, (a, h) in enumerate(pairs):
        ref = rr[i] + 1j * ri[i]
        pred, _, _ = predict_scat(net_int, net_scat, feats, C, a, h, X, Y)
        errs.append(rel_l2(pred, ref))
        amp_f.append(float(np.abs(ref).max()))
        amp_d.append(float(np.abs(pred).max()))
    gaps = heights - C.Y_SUB
    for gp, e, af, ad in zip(gaps, errs, amp_f, amp_d):
        print(f"   gap={gp:.4f}  relL2={e:.4f}   max|u_s| FDM={af:.4f} "
              f"DON={ad:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gaps, errs, "o-", color="crimson", label="relL2 vs FDM")
    ax.set_xlabel("tip-sample gap"); ax.set_ylabel("relL2", color="crimson")
    ax.tick_params(axis="y", labelcolor="crimson")
    ax.grid(True, alpha=0.3)
    for m in np.arange(0, gaps[-1] + period, period / 2):
        if gaps[0] <= m <= gaps[-1]:
            ax.axvline(m, color="gray", ls=":", lw=1.0)
    ax2 = ax.twinx()
    ax2.plot(gaps, amp_f, "s--", color="steelblue", label="max|u_scat| FDM")
    ax2.plot(gaps, amp_d, "^--", color="seagreen",  label="max|u_scat| DeepONet")
    ax2.set_ylabel("max |u_scat|")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    ax.set_title(f"Gap scan at α={np.degrees(alpha):.2f}° (off-grid); dotted "
                 f"lines = half-periods of the background standing wave")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/eval_gap_scan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {FIG_DIR}/eval_gap_scan.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, metavar="PATH",
                    help="checkpoint to evaluate (default: the DeepONet one; "
                         "pass final/models/final_mlp_weights.pt for the "
                         "MLP control)")
    ap.add_argument("--no-offgrid", action="store_true",
                    help="on-grid only (no FDM solves, runs in seconds)")
    ap.add_argument("--n-offgrid", type=int, default=5,
                    help="number of off-grid cell centres (default 5)")
    ap.add_argument("--gap-scan", type=int, default=0, metavar="N",
                    help="also scan N gaps at a fixed off-grid angle")
    args = ap.parse_args()

    print(f"  device: {DEVICE}")
    ck, hp = load_checkpoint(args.ckpt)
    C = _Const(hp)
    net_int, net_scat = build_nets(ck, hp)
    _, feats = make_features(ck, C)
    adeg, ayv = report_ongrid(ck, C, net_int, net_scat, feats)

    if not args.no_offgrid:
        n_off = max(2, min(args.n_offgrid, len(adeg) - 1))
        print(f"\n  (each off-grid point needs one FDM solve, ~48 s at N=601)")
        report_offgrid(C, net_int, net_scat, feats, adeg, ayv, n_off)
    if args.gap_scan > 0:
        report_gap_scan(C, net_int, net_scat, feats, adeg, ayv, args.gap_scan)
    print("\nDone.")


if __name__ == "__main__":
    main()
