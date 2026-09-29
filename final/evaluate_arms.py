"""
EVALUATE THE DATA-QUANTITY ARMS at parameters they did not train on
===============================================================================
Scores the trained checkpoints of final_4 ... final_100 against fresh FDM
solutions on a fixed k x k grid spanning the SAME ranges they were trained over:
half-angle 10-30 deg, apex_y 0.41-0.60 (tip-sample gap 0.01-0.20).

    python final/evaluate_arms.py                 every arm with a trained model
    python final/evaluate_arms.py --arms 4        one arm (or several: --arms 4 16)
    python final/evaluate_arms.py --k 9           a denser grid
    python final/evaluate_arms.py --solve-only    build the reference set only
    python final_4/run.py --eval                  the same, from an arm's launcher
    python final/evaluate_arms.py --k 50          a detailed heat map, 2500 points
    python final_4/run.py --eval-fine             the same, from an arm's launcher

ONE REFERENCE SET FOR EVERY ARM
-------------------------------
A reference depends only on (half-angle, apex_y), never on a model, so the k*k
FDM solutions are computed once, into final/inputs/eval_refs_k<k>_lam<lam>/,
and every arm and architecture is scored against the same fields. That is far
cheaper than solving per model -- ten models share one set -- and it is what
makes the arms comparable at all: identical points, identical references.

Each reference is its own file, written atomically the moment it is solved, so
a run can be stopped at any point and resumes where it left off. (The k=5 set
was built earlier as a single eval_refs_k5_lam1.npz and is still read as-is.)
Solves go through
evaluate_final.fdm_at_pairs, which executes final_fdm.py itself with only the
parameter arrays replaced, so stencil, PML, background and resolution match the
training references by construction.

WHY k = 5 OR 9, AND NOT 7
-------------------------
Every arm's training grid is a linspace over the same closed range, so every
one of them contains the four corners. What matters is whether the evaluation
grid contains anything ELSE an arm trained on. Checked, not assumed:

    5x5  shares only the 4 corners with every arm   -> 21 unseen points
    7x7  contains all 16 of final_16's training samples, and 16 of final_100's
    9x9  shares only the 4 corners with every arm   -> 77 unseen points

7x7 would therefore score final_16 and final_100 partly on their own training
data while the other arms got fresh points. The corners are kept rather than
dropped: they are marked "seen", reported separately, and double as a check
that the evaluation reproduces the training number -- final_4 at its own four
corners scores 0.1254 here against 0.1247 in its training log.

No grid asks the branch to extrapolate: at every point all eight features fall
inside the checkpoint's FEAT_MIN/FEAT_MAX.

THE FINE GRID: k = 50
---------------------
For a detailed picture of WHERE the error is, --k 50 (run.py --eval-fine)
scores 2500 points. At that size only the CSV and the heat map are written:
2500 cell labels, geometry panels or field plots would be unreadable. The heat
map drops its cell labels and per-cell ticks but keeps the red training-sample
markers.

The cost is almost all FDM: 2500 solves, ~35 h at the ~50 s/solve measured on
a CPU. Storing each reference as its own file is what makes that practical: a
stopped run loses at most the solve in progress, and progress costs one ~80 kB
write per solve, where rewriting a single file after every solve (as the k=5
set was built) would have written ~255 GB by the end. The solves can also be
split over processes or GPUs, then scored once:

    python final/evaluate_arms.py --k 50 --solve-only --part 0/4
    python final/evaluate_arms.py --k 50 --solve-only --part 1/4    ... to 3/4
    python final/evaluate_arms.py --k 50                            score

Part I takes every point whose index % N == I, so parts never overlap.

The 50x50 grid is NOT arm-neutral the way 5x5 is. Its step, 20/49 deg in
half-angle, is exactly a seventh of final_64's 20/7 deg (and likewise in gap),
so ALL 64 of final_64's training samples lie on it -- 2436 unseen points --
while every other arm shares only the four corners (2496 unseen). Seen points
are marked and kept out of the unseen statistics as always, but compare the
arms on the 5x5 numbers; the fine grid is for seeing the shape of the error.

OUTPUTS
-------
Per arm and architecture, in final_<n>/figures/:
    final_<n>_<arch>_eval_k<k>.csv          alpha, gap, relL2, seen, distance
    final_<n>_<arch>_eval_k<k>.png          relL2 heat map, training samples marked
    final_<n>_<arch>_eval_k<k>_geometry.png the tip at every grid point, laid out
                                            exactly like the heat map
    final_<n>_<arch>_eval_k<k>_side_by_side.png   geometry and heat map together
Above k = 12 -- so at the fine grid -- only the CSV and the heat map.

Across arms, in final/figures/:
    eval_arms_k<k>.csv / .png               unseen relL2 against training samples

The cross-arm summary is rebuilt from every per-arm CSV on disk for that k, so
arms evaluated at different times accumulate into one comparison. A CSV older
than its checkpoint is flagged STALE and left out of the plot rather than
silently compared against models it did not score.
"""

import argparse
import contextlib
import csv
import glob
import io
import os
import re
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Patch                # noqa: E402
from matplotlib.ticker import NullFormatter         # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import evaluate_final as EF                         # noqa: E402
import data_experiment as DX                        # noqa: E402

# fdm_at_pairs reads final_fdm.py through this module global, relative to the
# working directory by default. Absolute, the evaluation runs from anywhere.
EF.FDM_SRC = os.path.join(_HERE, "final_fdm.py")

ALPHA_DEG_RANGE = (10.0, 30.0)   # final_fdm.py's sweep -- every arm spans it
APEX_Y_RANGE = (0.41, 0.60)
# The two architectures every arm is expected to have. Everything else -- the
# --warm continuations, the data-only and norm64 variants, and the _best /
# _latest checkpoints a run writes as it goes -- is DISCOVERED from the files
# present (see arch_variants), so a new variant needs no edit here and its
# absence is never reported as missing.
ARCHS = ("deeponet", "mlp")
# Safety copies rather than models to score: ..._best_prev<N> is a superseded
# best, ..._pre_warm is what a continuation started from (and is identical to
# some earlier state of the run that continued it).
_SKIP_ARCH = re.compile(r"(_best_prev\d+|_pre_warm)$")
DEFAULT_K = 5
FINE_K = 50                      # --eval-fine: a detailed heat map, nothing else
MAX_ANNOTATED_K = 12             # above this: no cell labels, no per-point figures
SEEN_TOL = 1e-5                  # distance in range-normalised units

# the palette of the geometry_grid_*.png figures in the arm directories
_SKY, _TIP, _EDGE, _SUBSTRATE = "#f3f6fb", "#5b6573", "#2b2f36", "#c9b28e"


# ─────────────────────────────────────────────────────────────────────────────
# Paths and the evaluation grid
# ─────────────────────────────────────────────────────────────────────────────
def eval_grid(k):
    """k x k points over the full range, alpha-major like final_fdm.py, so
    flat index i is (alpha row i // k, height column i % k)."""
    A = np.linspace(ALPHA_DEG_RANGE[0], ALPHA_DEG_RANGE[1], k)
    H = np.linspace(APEX_Y_RANGE[0], APEX_Y_RANGE[1], k)
    pairs = [(float(np.deg2rad(a)), float(h)) for a in A for h in H]
    return A, H, pairs


def refs_path(k, lam):
    """The single-file cache the k=5 set was built as; read if present."""
    return os.path.join(_HERE, "inputs", f"eval_refs_k{k}_lam{lam:g}.npz")


def refs_dir(k, lam):
    """One file per reference, plus grid.npz -- how every new set is built."""
    return os.path.join(_HERE, "inputs", f"eval_refs_k{k}_lam{lam:g}")


def _ckpt(n, arch):
    return os.path.join(_REPO, f"final_{n}", "models",
                        f"final_{n}_{arch}_weights.pt")


def arch_variants(n):
    """Every <arch> this arm has a checkpoint for, from the file names
    final_<n>_<arch>_weights.pt. Discovery rather than a hard-coded list: runs
    now write <tag>_best_ and <tag>_latest_ checkpoints of their own, and the
    launchers add variants (_warm, _dataonly, _norm64, ...) as experiments need
    them."""
    pat = os.path.join(_REPO, f"final_{n}", "models", f"final_{n}_*_weights.pt")
    head, tail = f"final_{n}_", "_weights.pt"
    out = []
    for p in glob.glob(pat):
        a = os.path.basename(p)[len(head):-len(tail)]
        if a and not _SKIP_ARCH.search(a):
            out.append(a)
    # the base architectures first, then the variants alphabetically
    return ([a for a in ARCHS if a in out]
            + sorted(a for a in out if a not in ARCHS))


def _out(n, arch, k, suffix):
    return os.path.join(_REPO, f"final_{n}", "figures",
                        f"final_{n}_{arch}_eval_k{k}{suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# The shared reference set
# ─────────────────────────────────────────────────────────────────────────────
def _atomic_savez(path, **arrays):
    """np.savez_compressed through a temporary file and os.replace, so a run
    killed mid-write leaves the old file or none, never a truncated one. The
    pid in the temporary name keeps parallel --part runs from colliding."""
    tmp = f"{path[:-len('.npz')]}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def get_refs(k, lam, resolve=False, part=0, parts=1):
    """Returns ((A, H, pairs, X, Y, U), n_missing), with refs = None while any
    reference is still missing -- e.g. while other --part runs are unfinished.

    Reads the single-file cache if one exists; otherwise works in refs_dir,
    solving the missing points that belong to this part, one file each."""
    A, H, pairs = eval_grid(k)
    n = len(pairs)
    header = f"\n=== reference set: {k}x{k} = {n} points at lam = {lam:g}"

    legacy = refs_path(k, lam)
    if os.path.exists(legacy) and not resolve:
        # The context manager matters on Windows, where an open NpzFile keeps
        # its zip locked.
        with np.load(legacy) as d:
            if int(d["k"]) != k or abs(float(d["lam"]) - lam) > 1e-12:
                raise SystemExit(f"{legacy} does not match k={k}, lam={lam}")
            solved = d["solved"].astype(bool)
            if not solved.all():
                raise SystemExit(f"{legacy} holds only {int(solved.sum())} of "
                                 f"{n} references; delete it and run again to "
                                 f"rebuild the set one file per reference")
            X, Y = d["X"].copy(), d["Y"].copy()
            U = (d["U_scat_re"] + 1j * d["U_scat_im"]).astype(np.complex64)
        print(header)
        print(f"  {legacy}\n  all {n} cached")
        return (A, H, pairs, X, Y, U), 0

    d = refs_dir(k, lam)
    os.makedirs(d, exist_ok=True)
    grid_file = os.path.join(d, "grid.npz")

    def shard(i):
        return os.path.join(d, f"p{i:05d}.npz")

    have = np.array([(not resolve) and os.path.exists(shard(i))
                     for i in range(n)], dtype=bool)
    mine = [i for i in range(n) if not have[i] and i % parts == part]
    print(header)
    print(f"  {d}{os.sep}")
    print(f"  {int(have.sum())} cached, {n - int(have.sum())} to solve"
          + (f" -- {len(mine)} of them in --part {part}/{parts}"
             if parts > 1 else ""))
    if mine:
        print("  each solve is saved on its own, so this can be stopped and "
              "resumed at any point")

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
            _atomic_savez(grid_file, k=np.int32(k), lam=np.float64(lam),
                          alpha_deg_vals=A, apex_y_vals=H,
                          X=g["X_ph"][::s, ::s].astype(np.float32),
                          Y=g["Y_ph"][::s, ::s].astype(np.float32))
        _atomic_savez(shard(i), alpha=np.float64(a), apex_y=np.float64(h),
                      u_re=u.real.astype(np.float32),
                      u_im=u.imag.astype(np.float32))
        y_sub = float(g["Y_SUB"])
        g = None                       # release the solver's namespace now
        done = j + 1
        left = (time.time() - t_all) / done * (len(mine) - done)
        print(f"  [{done:>4}/{len(mine)}]  point {i:>4}  alpha="
              f"{np.degrees(a):6.2f}  gap={h - y_sub:.4f}  max|u_scat|="
              f"{np.abs(u).max():.3f}  ({time.time() - t:.0f}s, "
              f"~{left / 3600:.1f} h left)", flush=True)
    if mine:
        print(f"  {len(mine)} solves in {time.time() - t_all:.0f}s")

    n_missing = n - sum(os.path.exists(shard(i)) for i in range(n))
    if n_missing:
        return None, n_missing

    with np.load(grid_file) as gf:
        if int(gf["k"]) != k or abs(float(gf["lam"]) - lam) > 1e-12:
            raise SystemExit(f"{grid_file} does not match k={k}, lam={lam}")
        X, Y = gf["X"].copy(), gf["Y"].copy()
    # float32 components -> complex64: 2500 references at 101x101 is ~200 MB
    # here, against ~400 MB as complex128
    U = np.empty((n,) + X.shape, dtype=np.complex64)
    for i in range(n):
        with np.load(shard(i)) as f:
            if (abs(float(f["alpha"]) - pairs[i][0]) > 1e-12
                    or abs(float(f["apex_y"]) - pairs[i][1]) > 1e-12):
                raise SystemExit(f"{shard(i)} was solved for a different "
                                 f"point, so the grid definition has changed; "
                                 f"remove {d} and solve again")
            U[i] = f["u_re"] + 1j * f["u_im"]
    return (A, H, pairs, X, Y, U), 0


# ─────────────────────────────────────────────────────────────────────────────
# Scoring one model
# ─────────────────────────────────────────────────────────────────────────────
def score_model(n, arch, k, lam, refs):
    path = _ckpt(n, arch)
    if not os.path.exists(path):
        return None
    A, H, pairs, X, Y, U = refs
    tag = f"final_{n}_{arch}"

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
    a_span = ALPHA_DEG_RANGE[1] - ALPHA_DEG_RANGE[0]
    h_span = APEX_Y_RANGE[1] - APEX_Y_RANGE[0]
    if (abs(ta.min() - ALPHA_DEG_RANGE[0]) > 1e-3
            or abs(ta.max() - ALPHA_DEG_RANGE[1]) > 1e-3
            or abs(th.min() - APEX_Y_RANGE[0]) > 1e-5
            or abs(th.max() - APEX_Y_RANGE[1]) > 1e-5):
        print(f"  WARNING: {tag} trained over alpha [{ta.min():.2f}, "
              f"{ta.max():.2f}], apex_y [{th.min():.4f}, {th.max():.4f}] -- "
              f"not the full range, so part of this grid is extrapolation")

    m = len(pairs)
    errs, dist = np.zeros(m), np.zeros(m)
    for i, (a, h) in enumerate(pairs):
        pred, _, _ = EF.predict_scat(net_int, net_scat, feats, C, a, h, X, Y)
        errs[i] = EF.rel_l2(pred, U[i])
        dist[i] = np.sqrt(((ta - np.degrees(a)) / a_span) ** 2
                          + ((th - h) / h_span) ** 2).min()
    seen = dist < SEEN_TOL
    un = errs[~seen]

    os.makedirs(os.path.dirname(_out(n, arch, k, ".csv")), exist_ok=True)
    with open(_out(n, arch, k, ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha_deg", "apex_y", "gap", "relL2", "seen",
                    "dist_to_nearest_train"])
        for i, (a, h) in enumerate(pairs):
            w.writerow([f"{np.degrees(a):.4f}", f"{h:.5f}",
                        f"{h - C.Y_SUB:.5f}", f"{errs[i]:.6e}", int(seen[i]),
                        f"{dist[i]:.5f}"])

    fine = k > MAX_ANNOTATED_K
    print(f"\n=== {tag}: relL2 vs FDM on the {k}x{k} grid"
          + ("" if fine else f"   (* = one of its {len(ta)} training samples)"))
    if not fine:
        print("   alpha \\ gap" + "".join(f"{hh - C.Y_SUB:9.3f}" for hh in H))
        for ia in range(k):
            print(f"   {A[ia]:8.2f}   " + "".join(
                f"{errs[ia * k + ih]:8.4f}{'*' if seen[ia * k + ih] else ' '}"
                for ih in range(k)))
    if un.size:
        print(f"   unseen : mean {un.mean():.4f}   median {np.median(un):.4f}"
              f"   max {un.max():.4f}   over {un.size} points")
    if seen.any():
        print(f"   seen   : mean {errs[seen].mean():.4f}   over {seen.sum()} "
              f"training samples")

    _plot_heatmap(tag, k, A, H, errs, seen, ta, th, C.Y_SUB,
                  _out(n, arch, k, ".png"))
    saved = [".csv", ".png"]
    # On a fine grid the heat map IS the result; 2500 geometry panels or cell
    # labels would be unreadable, so the per-point figures are coarse-only.
    if not fine:
        _plot_geometry(tag, k, A, H, seen, C, _out(n, arch, k, "_geometry.png"))
        _plot_side_by_side(tag, k, A, H, errs, seen, ta, th, C,
                           _out(n, arch, k, "_side_by_side.png"))
        saved += ["_geometry.png", "_side_by_side.png"]
    for suffix in saved:
        print(f"   Saved: {_out(n, arch, k, suffix)}")
    return errs, seen


def _draw_heatmap(ax, k, A, H, errs, seen, ta, th, y_sub):
    """relL2 over the grid on `ax`: gap across, half-angle UP the page.

    Up to MAX_ANNOTATED_K per side every cell carries its value and a tick. On
    a fine grid neither is legible, so the cells are left bare and the axes are
    labelled in five even steps; the training-sample markers are drawn either
    way."""
    fine = k > MAX_ANNOTATED_K
    gaps = H - y_sub
    dg = 0.5 * (gaps[1] - gaps[0])
    da = 0.5 * (A[1] - A[0])
    vmax = float(errs.max()) if errs.max() > 0 else 1.0
    im = ax.imshow(errs.reshape(k, k), origin="lower", aspect="auto",
                   cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest",
                   extent=[gaps[0] - dg, gaps[-1] + dg, A[0] - da, A[-1] + da])
    if not fine:
        for i in range(k * k):
            ia, ih = divmod(i, k)
            ax.text(gaps[ih], A[ia],
                    f"{errs[i]:.3f}" + ("\n(train)" if seen[i] else ""),
                    ha="center", va="center", fontsize=8,
                    color="black" if errs[i] > 0.6 * vmax else "white")
    ax.scatter(th - y_sub, ta, marker="x", s=45 if len(ta) <= 36 else 30,
               c="red", linewidths=1.5, zorder=5,
               label=f"training samples ({len(ta)})")
    gt, at = ((np.linspace(gaps[0], gaps[-1], 5), np.linspace(A[0], A[-1], 5))
              if fine else (gaps, A))
    ax.set_xticks(gt)
    ax.set_xticklabels([f"{g:.3f}" for g in gt])
    ax.set_yticks(at)
    ax.set_yticklabels([f"{a:.1f}" for a in at])
    ax.set_xlabel("tip-sample gap")
    ax.set_ylabel("half-angle [deg]")
    return im


def _unseen_line(errs, seen):
    un = errs[~seen]
    return (f"unseen: mean {un.mean():.3f}, max {un.max():.3f} over "
            f"{un.size} points" if un.size else "every point is a training sample")


def _plot_heatmap(tag, k, A, H, errs, seen, ta, th, y_sub, path):
    fine = k > MAX_ANNOTATED_K
    fig, ax = plt.subplots(figsize=(9.5, 8) if fine else (8.5, 7))
    im = _draw_heatmap(ax, k, A, H, errs, seen, ta, th, y_sub)
    ax.set_title(f"{tag} \u2014 relL2 vs fresh FDM, {k}x{k} grid\n"
                 f"{_unseen_line(errs, seen)}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.colorbar(im, ax=ax, label="relL2")
    fig.tight_layout()
    fig.savefig(path, dpi=170 if fine else 140, bbox_inches="tight")
    plt.close(fig)


def tip_polygon(C, alpha, apex_y, y_top=1.0, n=120):
    """Closed outline of the rounded cone tip, for filling and stroking.

    _Const.outline lists the apex arc left-to-right between a right side drawn
    downwards and a left side drawn upwards, so its path crosses the apex and
    comes back along the same segment. It still FILLS the right region
    (checked: 0 of 160000 pixels differ from phi < 0), but that doubled segment
    is drawn whenever the outline is stroked -- it is the horizontal line across
    the apex in the field plots -- and would appear on every panel here. Running
    the arc right tangent -> under the apex -> left tangent gives the same region
    without it."""
    s, c, cone_v, tan_y, centre = C.consts(alpha, apex_y)
    ys = np.linspace(tan_y, y_top, n)
    hw = (ys - cone_v) * np.tan(alpha)
    ang = np.linspace(-alpha, -np.pi + alpha, n)
    xs = np.concatenate([(C.TIP_X + hw)[::-1],
                         C.TIP_X + C.R_APEX * np.cos(ang),
                         C.TIP_X - hw])
    yy = np.concatenate([ys[::-1], centre + C.R_APEX * np.sin(ang), ys])
    return xs, yy


def _draw_geometry(fig, spec, k, A, H, seen, C):
    """k x k tip schematics inside gridspec cell `spec`, oriented EXACTLY like
    the heat map -- half-angle increasing up the page, gap to the right -- so
    the two can be read row for row. Note this is the opposite row order to
    the geometry_grid_*.png figures, whose narrowest tip is at the top."""
    gs = spec.subgridspec(k, k, wspace=0.06, hspace=0.06)
    for ia in range(k):
        for ih in range(k):
            ax = fig.add_subplot(gs[k - 1 - ia, ih])   # grid row 0 = TOP
            i = ia * k + ih
            ax.add_patch(plt.Rectangle((0, C.Y_SUB), 1, 1 - C.Y_SUB,
                                       color=_SKY, zorder=0))
            ax.add_patch(plt.Rectangle((0, 0), 1, C.Y_SUB,
                                       color=_SUBSTRATE, zorder=1))
            xs, ys = tip_polygon(C, float(np.deg2rad(A[ia])), float(H[ih]))
            ax.fill(xs, ys, facecolor=_TIP, edgecolor=_EDGE, lw=0.8, zorder=2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("red" if seen[i] else "black")
                sp.set_linewidth(2.4 if seen[i] else 0.8)
            if ih == 0:
                ax.set_ylabel(f"{A[ia]:.1f}", rotation=0, ha="right",
                              va="center", fontsize=9)
            if ia == 0:
                ax.set_xlabel(f"{H[ih] - C.Y_SUB:.3f}", fontsize=9)


def _geometry_legend(fig, x, y, k, seen):
    """Key for the red panel borders, centred at figure coords (x, y).

    Counts the bordered PANELS rather than all training samples: for the
    denser arms most training samples fall between grid points and have no
    panel, so a bare total would not match the number of red borders."""
    key = Patch(facecolor="none", edgecolor="red", linewidth=2.4,
                label=f"training sample ({seen.sum()})")
    fig.legend(handles=[key], loc="upper center", bbox_to_anchor=(x, y),
               bbox_transform=fig.transFigure, fontsize=9, frameon=True)


def _plot_geometry(tag, k, A, H, seen, C, path):
    fig = plt.figure(figsize=(7.5, 8.3))
    gs = fig.add_gridspec(1, 1, left=0.13, right=0.97, bottom=0.15, top=0.89)
    _draw_geometry(fig, gs[0, 0], k, A, H, seen, C)
    b = gs[0, 0].get_position(fig)
    fig.text(0.025, 0.5 * (b.y0 + b.y1), "half-angle [deg]", rotation=90,
             ha="center", va="center")
    fig.text(0.5 * (b.x0 + b.x1), b.y0 - 0.06, "tip-sample gap", ha="center")
    _geometry_legend(fig, 0.5 * (b.x0 + b.x1), b.y0 - 0.085, k, seen)
    fig.suptitle(f"{tag} \u2014 tip geometry on the {k}x{k} evaluation grid\n"
                 f"laid out as in the heat map", fontsize=11)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_side_by_side(tag, k, A, H, errs, seen, ta, th, C, path):
    """Geometry grid and heat map at the same height, rows aligned."""
    # The left cell has to stay at least as wide as it is tall, or the square
    # geometry block shrinks below the heat map's height and the rows stop
    # lining up: 0.90*14.8/(2.2 + 0.12*1.1) = 5.71 in wide vs 0.74*7.6 = 5.62 tall.
    fig = plt.figure(figsize=(14.8, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], left=0.07,
                          right=0.97, bottom=0.12, top=0.86, wspace=0.12)
    _draw_geometry(fig, gs[0, 0], k, A, H, seen, C)
    b = gs[0, 0].get_position(fig)
    fig.text(b.x0 - 0.045, 0.5 * (b.y0 + b.y1), "half-angle [deg]",
             rotation=90, ha="center", va="center")
    fig.text(0.5 * (b.x0 + b.x1), b.y0 - 0.075, "tip-sample gap", ha="center")
    _geometry_legend(fig, 0.5 * (b.x0 + b.x1), b.y0 - 0.1, k, seen)

    ax = fig.add_subplot(gs[0, 1])
    im = _draw_heatmap(ax, k, A, H, errs, seen, ta, th, C.Y_SUB)
    # a square box, like the geometry block, so the rows line up across
    ax.set_box_aspect(1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.colorbar(im, ax=ax, label="relL2", fraction=0.046, pad=0.02)
    fig.suptitle(f"{tag} \u2014 tip geometry (left) and relL2 vs fresh FDM "
                 f"(right) on the {k}x{k} grid\n{_unseen_line(errs, seen)}",
                 fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Across arms
# ─────────────────────────────────────────────────────────────────────────────
def rebuild_summary(k):
    rows = []
    for n in DX.arms_sorted():
        for arch in arch_variants(n):
            p = _out(n, arch, k, ".csv")
            if not os.path.exists(p):
                continue
            ck = _ckpt(n, arch)
            stale = os.path.exists(ck) and os.path.getmtime(ck) > os.path.getmtime(p)
            with open(p, newline="") as f:
                recs = list(csv.DictReader(f))
            e = np.array([float(r["relL2"]) for r in recs])
            s = np.array([r["seen"] == "1" for r in recs])
            un = e[~s]
            rows.append(dict(
                n=n, samples=DX.n_samples(n), grid=DX.grid_str(n), arch=arch,
                n_unseen=int(un.size),
                unseen_mean=float(un.mean()) if un.size else float("nan"),
                unseen_median=float(np.median(un)) if un.size else float("nan"),
                unseen_max=float(un.max()) if un.size else float("nan"),
                seen_mean=float(e[s].mean()) if s.any() else float("nan"),
                stale=bool(stale)))
    if not rows:
        return

    out_csv = os.path.join(_HERE, "figures", f"eval_arms_k{k}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== across arms: {k}x{k} grid, relL2 on points each arm did not "
          f"train on")
    print(f"   {'arm':<10}{'grid':>7}{'arch':>22}{'unseen mean':>13}"
          f"{'median':>9}{'max':>9}{'train pts':>11}")
    for r in rows:
        print(f"   final_{r['n']:<4}{r['grid']:>7}{r['arch']:>22}"
              f"{r['unseen_mean']:13.4f}{r['unseen_median']:9.4f}"
              f"{r['unseen_max']:9.4f}{r['seen_mean']:11.4f}"
              f"{'   STALE' if r['stale'] else ''}")
    if any(r["stale"] for r in rows):
        print("   STALE = the checkpoint is newer than its evaluation; re-run "
              "--eval for that arm. Left out of the plot.")
    print(f"   Saved: {out_csv}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _archs = sorted({r["arch"] for r in rows},
                    key=lambda a: (a not in ARCHS, a))
    _cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for arch, colour in zip(_archs, _cycle * (len(_archs) // len(_cycle) + 1)):
        pts = [r for r in rows if r["arch"] == arch and not r["stale"]]
        if not pts:
            continue
        ns = [r["samples"] for r in pts]
        ax.plot(ns, [r["unseen_mean"] for r in pts], "o-", color=colour,
                lw=1.8, ms=7, label=f"{arch}: mean")
        ax.plot(ns, [r["unseen_max"] for r in pts], "s--", color=colour,
                lw=1.0, ms=5, alpha=0.6, label=f"{arch}: worst point")
    ax.set_xscale("log")
    ax.set_xticks([DX.n_samples(a) for a in DX.arms_sorted()])
    ax.set_xticklabels([f"{DX.n_samples(a)}\n({DX.grid_str(a)})"
                        for a in DX.arms_sorted()])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(3, 130)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("FDM samples used in training")
    ax.set_ylabel("relL2 on unseen parameters")
    nu = sorted({r["n_unseen"] for r in rows})
    nu_txt = f"{nu[0]}" if len(nu) == 1 else f"{nu[0]}-{nu[-1]}"
    ax.set_title(f"Data-quantity experiment \u2014 {k}x{k} evaluation grid, "
                 f"{nu_txt} unseen points per arm")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = out_csv[:-len(".csv")] + ".png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(arms=None, k=DEFAULT_K, lam=None, resolve=False, solve_only=False,
             part=0, parts=1):
    if k < 2:
        raise SystemExit("k must be at least 2")
    if not (parts >= 1 and 0 <= part < parts):
        raise SystemExit(f"--part {part}/{parts}: need 0 <= I < N")
    lam = float(DX.LAM if lam is None else lam)
    # "4" from the command line means the arm keyed 4; "4x2" stays a string.
    arms = DX.arms_sorted() if not arms else [
        int(a) if isinstance(a, str) and a.isdigit() else a for a in arms]
    unknown = [n for n in arms if n not in DX.GRIDS]
    if unknown:
        raise SystemExit(f"no such arm(s) {unknown}; the arms are "
                         f"{DX.arms_sorted()}")
    trained = [(n, a) for n in arms for a in arch_variants(n)]
    # only the two base architectures are "expected"; the rest are variants that
    # exist only where they were run, so their absence is not worth reporting
    missing = [(n, a) for n in arms for a in ARCHS
               if not os.path.exists(_ckpt(n, a))]
    # Checked BEFORE solving anything: k*k FDM solves for no model to score
    # would be a long way to find that out.
    if not trained and not solve_only:
        raise SystemExit(f"no trained checkpoints in arms {arms}. Use "
                         "--solve-only to build the reference set anyway.")

    refs, n_missing = get_refs(k, lam, resolve, part, parts)
    if solve_only:
        if n_missing:
            print(f"\n  {n_missing} of {k * k} references still missing -- "
                  f"they belong to the other --part runs")
        return
    if n_missing:
        raise SystemExit(f"\n  {n_missing} of {k * k} references are still "
                         f"missing, so nothing can be scored yet. Finish the "
                         f"other --part runs (or run without --part), then "
                         f"score.")
    for n, a in trained:
        score_model(n, a, k, lam, refs)
    if missing:
        print("\n  not trained yet, skipped: "
              + ", ".join(f"final_{n}_{a}" for n, a in missing))
    rebuild_summary(k)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score the data-quantity arms on a shared grid of "
                    "parameters they did not train on.")
    ap.add_argument("--arms", nargs="+", metavar="ARM",
                    help=f"which arms (default: all of {DX.arms_sorted()} "
                         "that have a trained model)")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help=f"evaluation grid is k x k (default {DEFAULT_K}; 9 is "
                         f"the next size sharing only the corners with every "
                         f"arm; {FINE_K} gives a detailed heat map only)")
    ap.add_argument("--resolve", action="store_true",
                    help="solve the reference set again, overwriting it")
    ap.add_argument("--solve-only", action="store_true",
                    help="build the reference set without scoring anything")
    ap.add_argument("--part", default="0/1", metavar="I/N",
                    help="solve only the points whose index %% N == I, to split "
                         "the reference solves over processes (default 0/1)")
    args = ap.parse_args(argv)
    try:
        part, parts = (int(x) for x in args.part.split("/"))
    except ValueError:
        raise SystemExit(f"--part expects I/N, e.g. 0/4, not {args.part!r}")
    evaluate(arms=args.arms, k=args.k, resolve=args.resolve,
             solve_only=args.solve_only, part=part, parts=parts)
    print("\nDone.")


if __name__ == "__main__":
    main()
