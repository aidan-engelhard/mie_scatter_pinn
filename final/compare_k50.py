"""
COMPARE two or more models' error maps on ONE colour scale
===============================================================================
evaluate_arms.py scales every heat map to its own error range, which is right
for reading a single model and wrong for comparing two: the same colour means a
different number in each figure. This redraws the k x k evaluations of several
models side by side with a shared scale and a single colour bar, from the CSVs
those evaluations already wrote.

    python final/compare_k50.py --arm 4x2 --archs mlp deeponet_dataonly
    python final/compare_k50.py --arm 4x2 --archs mlp deeponet_norm64_best --k 50

Writes final_<arm>/figures/compare_k<k>_<arch>_vs_<arch>.png. Nothing is
recomputed -- if a model has no CSV yet, score it first with evaluate_arms.
"""

import argparse
import csv
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _read(arm, arch, k):
    p = os.path.join(_REPO, f"final_{arm}", "figures",
                     f"final_{arm}_{arch}_eval_k{k}.csv")
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found -- score it first:\n"
                         f"  python final/evaluate_arms.py --k {k} --arms {arm}")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    A = np.array([float(r["alpha_deg"]) for r in rows])
    G = np.array([float(r["gap"]) for r in rows])
    E = np.array([float(r["relL2"]) for r in rows])
    # eval_grid is alpha-major: row i is alpha i, column j is gap j
    return A.reshape(k, k), G.reshape(k, k), E.reshape(k, k)


def _training_samples(arm, arch):
    """The arm's training geometries, from the checkpoint (empty if unreadable)."""
    p = os.path.join(_REPO, f"final_{arm}", "models",
                     f"final_{arm}_{arch}_weights.pt")
    if not os.path.exists(p):
        return None, None
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None, None
    return (np.degrees(np.asarray(ck["half_angle"], dtype=float)),
            np.asarray(ck["apex_y"], dtype=float) - ck["hyperparams"]["Y_SUB"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help='4, 16, "4x2", ...')
    ap.add_argument("--archs", nargs="+", required=True, metavar="ARCH")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", default=None, help="output path (default: alongside)")
    args = ap.parse_args(argv)
    arm = int(args.arm) if args.arm.isdigit() else args.arm

    data = [(a,) + _read(arm, a, args.k) for a in args.archs]
    vmax = max(float(d[3].max()) for d in data)      # ONE scale for all panels
    ta, tg = _training_samples(arm, args.archs[0])

    fig, axes = plt.subplots(1, len(data), figsize=(6.2 * len(data), 5.6),
                             squeeze=False)
    for ax, (arch, A, G, E) in zip(axes[0], data):
        im = ax.imshow(E, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=vmax, interpolation="nearest",
                       extent=[G.min(), G.max(), A.min(), A.max()])
        if ta is not None:
            ax.plot(tg, ta, "x", color="red", ms=9, mew=2, ls="none")
        ax.set_title(f"final_{arm}_{arch}\nunseen mean {E.mean():.4f}, "
                     f"max {E.max():.4f}", fontsize=11)
        ax.set_xlabel("tip-sample gap")
    axes[0][0].set_ylabel("half-angle [deg]")
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.92)
    cb.set_label("relL2 vs FDM  (shared scale)")
    if ta is not None:
        fig.legend(handles=[Line2D([0], [0], marker="x", color="red", ls="none",
                                   ms=9, mew=2,
                                   label=f"training samples ({len(ta)})")],
                   loc="lower center", ncol=1, frameon=True)
    out = args.out or os.path.join(
        _REPO, f"final_{arm}", "figures",
        f"compare_k{args.k}_" + "_vs_".join(args.archs) + ".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    for arch, _A, _G, E in data:
        print(f"  {arch:<28} mean {E.mean():.4f}  median {np.median(E):.4f}  "
              f"max {E.max():.4f}")
    print(f"  shared colour scale 0 .. {vmax:.4f}")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
