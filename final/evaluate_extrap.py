"""
EXTRAPOLATION: score the arms OUTSIDE the parameter box they were trained on
===============================================================================
Every arm trains inside half-angle 10-30 deg and tip-sample gap 0.01-0.20
(apex_y 0.41-0.60), and evaluate_arms.py only ever scores inside that box. This
scores them on a wider one,

    half-angle 5-35 deg      gap 0.01-0.30  (apex_y 0.41-0.70)

against fresh FDM solves, and reports the error separately by REGION:

    inside   the training box -- interpolation, a check against evaluate_arms
    angle    half-angle outside 10-30, gap inside     (both sides of the box)
    gap      gap above 0.20, half-angle inside
    both     outside on both axes: the four corners of the wider box

The gap axis only extends UPWARD. Below 0.01 there is almost no room -- at 0
the tip touches the substrate -- and final_fdm.py resolves 0.01 with 3 cells,
so a smaller gap would be under-resolved as well.

    python final/evaluate_extrap.py --solve-only              build the references
    python final/evaluate_extrap.py --solve-only --part 0/4   ... split over processes
    python final/evaluate_extrap.py                           score arms 4 and 4x2
                                                              (never solves anything)
    python final/evaluate_extrap.py --arms 4x2 --variants mlp_d mlp_dpx_w0p003
    python final/evaluate_extrap.py --partial                 score on the solved subset

THE GRID. 25 half-angles in 1.25 deg steps x 30 gaps in 0.01 steps = 750
points. Both steps divide the training-box edges exactly, so 10, 30 deg and
0.20 are grid lines, every point falls unambiguously in one region, and the
four training-box corners are grid points (marked "seen", as in
evaluate_arms.py). The region sizes: 17x20 = 340 inside, 8x20 = 160 angle,
17x10 = 170 gap, 8x10 = 80 both.

THE REFERENCES are solved exactly as evaluate_arms.py solves its own: through
evaluate_final.fdm_at_pairs, which executes final_fdm.py with only the parameter
arrays replaced, so stencil, PML, background and resolution match the training
data by construction. One file per point in
final/inputs/extrap_refs_lam<lam>/, written atomically, so a run can be stopped
and resumed at any point and --part runs never collide. At the ~50 s/solve
measured on a CPU the full set is ~10 h; run it on the GPU box.

    Nothing in final_fdm.py needs to change for the wider box. The physical
    domain is the unit square with the tip axis at x = 0.5: a 0.70 apex leaves
    the tip well inside it, and at 35 deg the shank still meets the top
    boundary at x = 0.06-0.94. It then widens through the top PML into the side
    PMLs, but that already happens at 30 deg (half-width 0.95 at the top of the
    PML against 1.14 at 35 deg, both inside the grid's 1.5), so the extremes of
    the wider box are not a new regime for the solver.

WHAT THE MODELS SEE out there. The branch input is eight geometric features
normalised by the checkpoint's FEAT_MIN/FEAT_MAX (evaluate_final.make_features)
to [-1, 1] over the training box. Outside it they leave that interval, linearly
in the normalised features, and nothing clamps them: this measures what the
network does with inputs it has never seen, which is the question.

OUTPUTS
    final_<n>/figures/final_<n>_<arch>_extrap.csv   alpha, gap, region, relL2, seen
    final_<n>/figures/final_<n>_<arch>_extrap.png   log-scale heat map, box marked
    final/figures/extrap_summary.csv                per model: mean/median/max by region
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LogNorm               # noqa: E402
from matplotlib.patches import Rectangle            # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import evaluate_final as EF                         # noqa: E402
import evaluate_arms as EA                          # noqa: E402
import data_experiment as DX                        # noqa: E402

EF.FDM_SRC = os.path.join(_HERE, "final_fdm.py")

Y_SUB = 0.4                      # final_fdm.py's substrate height
TRAIN_ALPHA = EA.ALPHA_DEG_RANGE                    # (10, 30)
TRAIN_APEX = EA.APEX_Y_RANGE                        # (0.41, 0.60)
# 1.25 deg and 0.01 steps: see THE GRID above
ALPHA_DEG = np.linspace(5.0, 35.0, 25)
APEX_Y = np.linspace(0.41, 0.70, 30)
REGIONS = ("inside", "angle", "gap", "both")
DEFAULT_ARMS = (4, "4x2")
TOL = 1e-6


def grid():
    """Alpha-major like evaluate_arms.eval_grid: flat index i is
    (alpha row i // len(APEX_Y), gap column i % len(APEX_Y))."""
    return [(float(np.deg2rad(a)), float(h)) for a in ALPHA_DEG for h in APEX_Y]


def region(alpha_deg, apex_y):
    a_out = (alpha_deg < TRAIN_ALPHA[0] - TOL) or (alpha_deg > TRAIN_ALPHA[1] + TOL)
    g_out = apex_y > TRAIN_APEX[1] + TOL
    return {(False, False): "inside", (True, False): "angle",
            (False, True): "gap", (True, True): "both"}[(a_out, g_out)]


def refs_dir(lam):
    return os.path.join(_HERE, "inputs", f"extrap_refs_lam{lam:g}")


# ─────────────────────────────────────────────────────────────────────────────
# The reference set
# ─────────────────────────────────────────────────────────────────────────────
def get_refs(lam, solve=False, part=0, parts=1, partial=False):
    """With solve, solve the missing references that belong to this part, one
    file each; without it, only read what is on disk -- so scoring never
    launches a sweep. Returns (X, Y, U, have) with U[i] valid where have[i], or
    None while any are missing -- unless partial, which returns whatever is
    solved."""
    pairs = grid()
    n = len(pairs)
    d = refs_dir(lam)
    os.makedirs(d, exist_ok=True)
    grid_file = os.path.join(d, "grid.npz")

    def shard(i):
        return os.path.join(d, f"p{i:05d}.npz")

    have = np.array([os.path.exists(shard(i)) for i in range(n)])
    mine = [i for i in range(n)
            if solve and not have[i] and i % parts == part]
    print(f"\n=== extrapolation references: {len(ALPHA_DEG)}x{len(APEX_Y)} = "
          f"{n} points at lam = {lam:g}\n  {d}{os.sep}\n  {int(have.sum())} "
          f"cached, {n - int(have.sum())} to solve"
          + (f" -- {len(mine)} of them in --part {part}/{parts}"
             if parts > 1 else ""))

    t_all = time.time()
    for j, i in enumerate(mine):
        a, h = pairs[i]
        t = time.time()
        g = EF.fdm_at_pairs([pairs[i]], lam=lam)
        s = int(g["SAVE_STRIDE"])
        u = (g["U_scat_re_all"][0] + 1j * g["U_scat_im_all"][0])[::s, ::s]
        if not np.any(u):
            raise RuntimeError(f"FDM returned an all-zero field at alpha="
                               f"{np.degrees(a):.2f} deg, apex_y={h:.4f}")
        if not os.path.exists(grid_file):
            EA._atomic_savez(grid_file, lam=np.float64(lam),
                             alpha_deg_vals=ALPHA_DEG, apex_y_vals=APEX_Y,
                             X=g["X_ph"][::s, ::s].astype(np.float32),
                             Y=g["Y_ph"][::s, ::s].astype(np.float32))
        EA._atomic_savez(shard(i), alpha=np.float64(a), apex_y=np.float64(h),
                         u_re=u.real.astype(np.float32),
                         u_im=u.imag.astype(np.float32))
        g = None
        left = (time.time() - t_all) / (j + 1) * (len(mine) - j - 1)
        print(f"  [{j + 1:>4}/{len(mine)}]  point {i:>4}  alpha="
              f"{np.degrees(a):6.2f}  gap={h - Y_SUB:.3f}  "
              f"{region(np.degrees(a), h):<6}  max|u_scat|={np.abs(u).max():.3f}"
              f"  ({time.time() - t:.0f}s, ~{left / 3600:.1f} h left)",
              flush=True)

    have = np.array([os.path.exists(shard(i)) for i in range(n)])
    if not have.all() and not partial:
        return None, int((~have).sum())
    if not have.any():
        return None, n

    with np.load(grid_file) as gf:
        if abs(float(gf["lam"]) - lam) > 1e-12:
            raise SystemExit(f"{grid_file} does not match lam={lam}")
        X, Y = gf["X"].copy(), gf["Y"].copy()
    U = np.zeros((n,) + X.shape, dtype=np.complex64)
    for i in np.flatnonzero(have):
        with np.load(shard(i)) as f:
            if (abs(float(f["alpha"]) - pairs[i][0]) > 1e-12
                    or abs(float(f["apex_y"]) - pairs[i][1]) > 1e-12):
                raise SystemExit(f"{shard(i)} was solved for a different point, "
                                 f"so the grid definition has changed; remove "
                                 f"{d} and solve again")
            U[i] = f["u_re"] + 1j * f["u_im"]
    return (X, Y, U, have), int((~have).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
def _out(n, arch, suffix):
    return os.path.join(_REPO, f"final_{n}", "figures",
                        f"final_{n}_{arch}_extrap{suffix}")


def score_model(n, arch, lam, refs):
    X, Y, U, have = refs
    path = EA._ckpt(n, arch)
    with contextlib.redirect_stdout(io.StringIO()):
        ck, hp = EF.load_checkpoint(path)
    if abs(float(hp["lam"]) - lam) > 1e-12:
        raise SystemExit(f"{path} was trained at lam={hp['lam']}, but the "
                         f"references are at lam={lam}")
    C = EF._Const(hp)
    net_int, net_scat = EF.build_nets(ck, hp)
    _, feats = EF.make_features(ck, C)
    ta = np.degrees(np.asarray(ck["half_angle"], dtype=float))
    th = np.asarray(ck["apex_y"], dtype=float)

    pairs = grid()
    errs = np.full(len(pairs), np.nan)
    seen = np.zeros(len(pairs), dtype=bool)
    regs = []
    for i, (a, h) in enumerate(pairs):
        regs.append(region(np.degrees(a), h))
        seen[i] = np.any((np.abs(ta - np.degrees(a)) < 1e-3)
                         & (np.abs(th - h) < 1e-5))
        if have[i]:
            pred, _, _ = EF.predict_scat(net_int, net_scat, feats, C, a, h, X, Y)
            errs[i] = EF.rel_l2(pred, U[i])
    regs = np.array(regs)

    with open(_out(n, arch, ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha_deg", "apex_y", "gap", "region", "relL2", "seen"])
        for i, (a, h) in enumerate(pairs):
            if have[i]:
                w.writerow([f"{np.degrees(a):.4f}", f"{h:.5f}",
                            f"{h - Y_SUB:.5f}", regs[i], f"{errs[i]:.6e}",
                            int(seen[i])])

    tag = f"final_{n}_{arch}"
    row = {"arm": n, "arch": arch, "n_scored": int(have.sum())}
    print(f"\n=== {tag}: relL2 vs FDM by region (unseen points)")
    for r in REGIONS:
        m = (regs == r) & have & ~seen
        e = errs[m]
        row[f"{r}_n"] = int(m.sum())
        row[f"{r}_mean"] = float(e.mean()) if e.size else float("nan")
        row[f"{r}_median"] = float(np.median(e)) if e.size else float("nan")
        row[f"{r}_max"] = float(e.max()) if e.size else float("nan")
        print(f"   {r:<7} " + (f"mean {e.mean():.4f}   median {np.median(e):.4f}"
                               f"   max {e.max():.4f}   over {e.size} points"
                               if e.size else "no points scored yet"))
    _plot(tag, errs, ta, th, _out(n, arch, ".png"))
    print(f"   Saved: {_out(n, arch, '.csv')}\n   Saved: {_out(n, arch, '.png')}")
    return row


def _plot(tag, errs, ta, th, path):
    E = errs.reshape(len(ALPHA_DEG), len(APEX_Y))
    gaps = APEX_Y - Y_SUB
    dg, da = 0.5 * (gaps[1] - gaps[0]), 0.5 * (ALPHA_DEG[1] - ALPHA_DEG[0])
    ok = E[np.isfinite(E)]
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    im = ax.imshow(np.ma.masked_invalid(E), origin="lower", aspect="auto",
                   cmap="viridis", interpolation="nearest",
                   norm=LogNorm(vmin=max(ok.min(), 1e-4), vmax=ok.max()),
                   extent=[gaps[0] - dg, gaps[-1] + dg,
                           ALPHA_DEG[0] - da, ALPHA_DEG[-1] + da])
    g0, g1 = TRAIN_APEX[0] - Y_SUB, TRAIN_APEX[1] - Y_SUB
    ax.add_patch(Rectangle((g0, TRAIN_ALPHA[0]), g1 - g0,
                           TRAIN_ALPHA[1] - TRAIN_ALPHA[0], fill=False,
                           edgecolor="white", lw=1.8, ls="--",
                           label="training box"))
    ax.scatter(th - Y_SUB, ta, marker="x", s=45, c="red", linewidths=1.5,
               zorder=5, label=f"training samples ({len(ta)})")
    ax.set_xlabel("tip-sample gap")
    ax.set_ylabel("half-angle [deg]")
    ax.set_title(f"{tag} — relL2 vs fresh FDM, extrapolation grid "
                 f"{len(ALPHA_DEG)}x{len(APEX_Y)}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, ncol=2)
    fig.colorbar(im, ax=ax, label="relL2 (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(arms=DEFAULT_ARMS, variants=None, lam=None, solve_only=False,
             part=0, parts=1, partial=False):
    if not (parts >= 1 and 0 <= part < parts):
        raise SystemExit(f"--part {part}/{parts}: need 0 <= I < N")
    lam = float(DX.LAM if lam is None else lam)
    arms = [int(a) if isinstance(a, str) and a.isdigit() else a for a in arms]
    unknown = [n for n in arms if n not in DX.GRIDS]
    if unknown:
        raise SystemExit(f"no such arm(s) {unknown}; the arms are "
                         f"{DX.arms_sorted()}")
    # _best and _latest are the same run's checkpoints at other moments; by
    # default only the run's final weights are scored, --variants names any
    todo = [(n, a) for n in arms for a in EA.arch_variants(n)
            if (a in variants if variants
                else not a.endswith(("_best", "_latest")))]
    if variants:
        missing = sorted(set(variants) - {a for _, a in todo})
        if missing:
            raise SystemExit(f"no checkpoint for variant(s) {missing} in arms "
                             f"{arms}")
    if not todo and not solve_only:
        raise SystemExit(f"no trained checkpoints in arms {arms}")

    refs, n_missing = get_refs(lam, solve_only, part, parts, partial)
    if solve_only:
        if n_missing:
            print(f"\n  {n_missing} of {len(grid())} references still missing"
                  + (" -- they belong to the other --part runs" if parts > 1
                     else ""))
        return
    if refs is None:
        raise SystemExit(f"\n  {n_missing} of {len(grid())} references are "
                         f"still missing. Solve them with --solve-only, or "
                         f"pass --partial to score on the ones that exist.")
    if n_missing:
        print(f"\n  --partial: scoring on {len(grid()) - n_missing} of "
              f"{len(grid())} points")

    rows = [score_model(n, a, lam, refs) for n, a in todo]
    out = os.path.join(_HERE, "figures", "extrap_summary.csv")
    # merge with rows already on disk, so arms scored in separate runs
    # accumulate into one table; this run's rows replace their own
    old = []
    if os.path.exists(out):
        with open(out, newline="") as f:
            done = {(str(r["arm"]), r["arch"]) for r in rows}
            old = [r for r in csv.DictReader(f)
                   if (r["arm"], r["arch"]) not in done]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(old + rows)

    print(f"\n=== extrapolation summary: mean relL2 on unseen points")
    print(f"   {'model':<34}" + "".join(f"{r:>9}" for r in REGIONS))
    for r in rows:
        print(f"   final_{str(r['arm']) + '_' + r['arch']:<28}"
              + "".join(f"{r[f'{g}_mean']:9.4f}" for g in REGIONS))
    print(f"   Saved: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score the arms outside their training box: half-angle "
                    "5-35 deg, gap 0.01-0.30.")
    ap.add_argument("--arms", nargs="+", metavar="ARM",
                    default=[str(a) for a in DEFAULT_ARMS],
                    help="which arms (default: 4 4x2)")
    ap.add_argument("--variants", nargs="+", metavar="ARCH",
                    help="only these checkpoints, e.g. mlp_d mlp_dpx_w0p003 "
                         "(default: every run's final weights, not _best/_latest)")
    ap.add_argument("--solve-only", action="store_true",
                    help="build the reference set without scoring anything")
    ap.add_argument("--part", default="0/1", metavar="I/N",
                    help="solve only the points whose index %% N == I "
                         "(default 0/1)")
    ap.add_argument("--partial", action="store_true",
                    help="score on whichever references exist so far")
    args = ap.parse_args(argv)
    try:
        part, parts = (int(x) for x in args.part.split("/"))
    except ValueError:
        raise SystemExit(f"--part expects I/N, e.g. 0/4, not {args.part!r}")
    evaluate(arms=args.arms, variants=args.variants, solve_only=args.solve_only,
             part=part, parts=parts, partial=args.partial)
    print("\nDone.")


if __name__ == "__main__":
    main()
