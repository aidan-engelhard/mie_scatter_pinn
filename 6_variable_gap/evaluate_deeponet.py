"""
EVALUATE — interpolation and extrapolation in the tip-sample gap
===============================================================================
Takes a trained checkpoint and asks it for gaps it was never shown, well past
the range it was trained on, and plots the relative L2 error against the gap.

    python variable_sample/evaluate_deeponet.py
    python variable_sample/evaluate_deeponet.py --arch mlp
    python variable_sample/evaluate_deeponet.py --gaps 0.01,0.50,25

WHAT IT MEASURES
----------------
Three regimes, distinguished on the plot rather than averaged together:

  TRAINED       gaps the checkpoint's train_idx names. With W_DATA > 0 these
                were supervised by the FDM, so the error here is a fit, not a
                prediction.
  INTERPOLATED  gaps inside the training range that the model never saw.
  EXTRAPOLATED  gaps beyond the training range, out to 0.50.

0.50 is the hard geometric limit, not a round number: the cylinder centre sits
at cy = Y_SUB + a + gap, so its top is at 0.5 + gap, and at gap = 0.50 that is
exactly y = 1.0 -- tangent to the absorbing layer. Past that the scatterer
would be inside the PML and the reference would describe a different problem.
The last stretch is flagged on the plot for the same reason: by gap 0.45 the
cylinder is within one radius of the PML and its near field is already
interacting with the absorber.

THE REFERENCE
-------------
Extrapolation needs FDM solutions at the new gaps, which do not exist in the
training sweep. This script produces them by running variable_sample_fdm.py
through runpy with the gap list overridden -- the same solver, not a second
copy of it -- and caches the result in

    variable_sample/inputs/fdm_eval_gaps.npz

so later runs are instant. Pass --no-solve to refuse to solve and use only what
is cached. The solve is the expensive part: roughly a minute per gap on CPU,
far less with CuPy.

WHY THE NETWORK IS REBUILT BY EXECUTING THE TRAINER
---------------------------------------------------
variable_sample_deeponet.py is executed up to (not including) its optimiser, so
the architecture, the Fourier embeddings and the feature construction come from
the file that trained the weights instead of from a second implementation that
can drift. final/evaluate_final.py rebuilt its networks by hand and silently
went stale when a Fourier band was added; this cannot.

Two things are then taken from the CHECKPOINT rather than recomputed:

  B_ext / B_int   drawn with torch.manual_seed(1) ON THE TRAINING DEVICE, and
                  the CUDA and CPU generators differ -- regenerating them would
                  hand the trained weights a different basis and produce a
                  meaningless field.
  FEAT_MIN/MAX    the branch's feature normalisation, fixed by the TRAINING
                  gaps. Recomputing it over the evaluation gaps would rescale
                  the branch input and quietly change what the model is being
                  asked.

OUTPUTS
-------
    variable_sample/figures/<tag>_gap_generalisation.png
    variable_sample/figures/<tag>_gap_fields.png
"""

import argparse
import io
import os
import re
import runpy
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAINER = os.path.join(HERE, "variable_sample_deeponet.py")
FDM_SRC = os.path.join(HERE, "variable_sample_fdm.py")
EVAL_NPZ = os.path.join(ROOT, "variable_sample", "inputs", "fdm_eval_gaps.npz")

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="deeponet",
                choices=("deeponet", "mlp", "single"))
ap.add_argument("--ckpt", default=None, help="checkpoint path")
ap.add_argument("--gaps", default="0.01,0.50,25",
                help="LO,HI,N for the evaluation gaps")
ap.add_argument("--no-solve", action="store_true",
                help="use only cached FDM references; do not solve")
ap.add_argument("--stride", type=int, default=3,
                help="subsample stride of the FDM grid when comparing")
args = ap.parse_args()

os.chdir(ROOT)
# single_deeponet.py sets RUN_TAG_OVERRIDE = "single_deeponet" rather than
# following the variable_sample_* pattern, so its checkpoint is named
# differently. Kept as a map rather than a rule so a new launcher with another
# tag fails loudly here instead of silently looking in the wrong place.
_DEFAULT_CKPT = {
    "deeponet": "variable_sample_deeponet_weights.pt",
    "mlp":      "variable_sample_mlp_weights.pt",
    "single":   "single_deeponet_weights.pt",
}
CKPT = args.ckpt or os.path.join("variable_sample", "models",
                                 _DEFAULT_CKPT[args.arch])
if not os.path.exists(CKPT):
    raise SystemExit(f"checkpoint not found: {CKPT}")

_lo, _hi, _n = args.gaps.split(",")
_grid = np.linspace(float(_lo), float(_hi), int(_n))

# ─────────────────────────────────────────────────────────────────────────────
# 1. What the checkpoint was trained on
# ─────────────────────────────────────────────────────────────────────────────
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
TRAIN_GAPS = np.asarray([float(ck["gap"][int(p)]) for p in ck["train_idx"]])
TEST_GAPS = np.asarray([float(ck["gap"][int(p)]) for p in ck["test_idx"]])
G_LO, G_HI = float(TRAIN_GAPS.min()), float(TRAIN_GAPS.max())
HP = ck.get("hyperparams", {})

# Rebuilding the wrong architecture loads the weights without complaint and
# produces a field that looks plausible and means nothing, so the checkpoint's
# own record wins over the flag.
_ck_arch = HP.get("ARCH")
if _ck_arch is not None and _ck_arch != args.arch:
    raise SystemExit(
        f"--arch {args.arch!r} but {CKPT} was trained with ARCH={_ck_arch!r}. "
        f"Pass --arch {_ck_arch} (and --ckpt if the file is not the default).")

# The checkpoint's OWN gaps are always evaluated on top of the requested grid.
# Without them the "trained" and "held out" regimes would appear only when a
# grid point happened to land on one, and the plot would have nothing to anchor
# the in-range behaviour against.
EVAL_GAPS = np.unique(np.round(
    np.concatenate([_grid, TRAIN_GAPS, TEST_GAPS]), 6))

print("=" * 78)
print(f"EVALUATE — {args.arch} — interpolation and extrapolation in the gap")
print("=" * 78)
print(f"  checkpoint     : {CKPT}")
print(f"  trained on     : {len(TRAIN_GAPS)} gaps in "
      f"[{G_LO:.4f}, {G_HI:.4f}]")
print(f"  held out       : {len(TEST_GAPS)} gaps "
      f"{[round(float(g), 4) for g in TEST_GAPS]}")
print(f"  data loss      : W_DATA = {HP.get('W_DATA', 'unknown')}"
      + ("   (so the trained gaps below are a FIT, not a prediction)"
         if HP.get("W_DATA", 0) else ""))
print(f"  evaluating     : {len(EVAL_GAPS)} gaps in "
      f"[{EVAL_GAPS[0]:.3f}, {EVAL_GAPS[-1]:.3f}]")
print("=" * 78)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FDM reference at the evaluation gaps, cached
# ─────────────────────────────────────────────────────────────────────────────
def load_cache():
    if not os.path.exists(EVAL_NPZ):
        return None
    d = np.load(EVAL_NPZ)
    have = set(np.round(d["gap"].astype(float), 6).tolist())
    if set(np.round(EVAL_GAPS, 6).tolist()) <= have:
        return d
    return None


_cached = load_cache()
if _cached is None:
    if args.no_solve:
        raise SystemExit(
            f"no cached reference covering these gaps in {EVAL_NPZ}, and "
            f"--no-solve was given")
    print(f"\nSolving {len(EVAL_GAPS)} FDM references (cached afterwards in "
          f"{EVAL_NPZ}).")
    print("This is the slow part: roughly a minute per gap on CPU, far less "
          "with CuPy.\n")
    runpy.run_path(FDM_SRC, init_globals={
        "GAP_LIST_OVERRIDE": EVAL_GAPS.tolist(),
        "N_TEST_OVERRIDE": 0,
        "OUT_OVERRIDE": EVAL_NPZ,
        "FIG_OVERRIDE": os.path.join(
            "variable_sample", "figures", "fdm_eval_gaps_sweep.png"),
    }, run_name="__main__")
    _cached = load_cache()
    if _cached is None:
        raise SystemExit("the FDM run did not produce the expected gaps")

d = _cached
_gap_all = np.round(d["gap"].astype(float), 6)
_row = {g: i for i, g in enumerate(_gap_all.tolist())}
ST = args.stride
REF_X = d["X"][::ST, ::ST]
REF_Y = d["Y"][::ST, ::ST]
REF_SUB = d["in_sub"][::ST, ::ST].astype(bool)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Rebuild the trained network from the trainer's own definitions
# ─────────────────────────────────────────────────────────────────────────────
print("Rebuilding the network from variable_sample_deeponet.py ...")
_src = io.open(TRAINER, encoding="utf-8").read()
_src, _n = re.subn(r"^USE_COMPILE = True$", lambda _m: "USE_COMPILE = False",
                   _src, flags=re.M)
assert _n == 1, "could not disable torch.compile in the trainer source"
_CUT = "optimizer = torch.optim.Adam(all_params, lr=LR_ADAM)"
assert _src.count(_CUT) == 1, "trainer cut marker not found"
NS = {"__name__": "__main__", "__file__": TRAINER, "ARCH_OVERRIDE": args.arch}
exec(compile(_src[:_src.index(_CUT)], TRAINER, "exec"), NS)

DEV = NS["DEVICE"]
NS["B_ext"] = ck["B_ext"].to(DEV)
NS["B_int"] = ck["B_int"].to(DEV)
NS["FEAT_MIN"] = ck["FEAT_MIN"]
NS["FEAT_MAX"] = ck["FEAT_MAX"]
NS["raw_module"](NS["net_int"]).load_state_dict(ck["net_int_state_dict"])
NS["raw_module"](NS["net_scat"]).load_state_dict(ck["net_scat_state_dict"])
NS["net_int"].eval()
NS["net_scat"].eval()

cx, a, Y_SUB = NS["cx"], NS["a"], NS["Y_SUB"]
XY = torch.tensor(np.stack([REF_X.ravel(), REF_Y.ravel()], 1).astype(np.float32),
                  device=DEV)
with torch.no_grad():
    _bgR, _bgI = NS["u_bg_air_re"](XY), NS["u_bg_air_im"](XY)
BG = (_bgR.cpu().numpy().reshape(REF_X.shape)
      + 1j * _bgI.cpu().numpy().reshape(REF_X.shape))


def predict(gap):
    """Composite u_scat on the reference grid for an arbitrary gap."""
    cy = Y_SUB + a + float(gap)
    feats = NS["param_to_features"](float(gap))
    with torch.no_grad():
        sR, sI = NS["net_scat"](feats, XY)
        iR, iI = NS["net_int"](feats, XY)
    to_c = lambda R, I: (R.cpu().numpy().reshape(REF_X.shape)
                         + 1j * I.cpu().numpy().reshape(REF_X.shape))
    in_cyl = (REF_X - cx) ** 2 + (REF_Y - cy) ** 2 <= a ** 2
    return np.where(in_cyl, to_c(iR, iI) - BG, to_c(sR, sI)), in_cyl


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
_tr = set(np.round(TRAIN_GAPS, 4).tolist())
_te = set(np.round(TEST_GAPS, 4).tolist())


def label_of(g):
    r = round(float(g), 4)
    if r in _tr:
        return "trained"
    if r in _te:
        return "held out"
    return "interp" if G_LO <= g <= G_HI else "EXTRAP"


rows = []
print(f"\n{'gap':>7}{'regime':>10}{'relL2':>9}{'|c|':>8}"
      f"{'||net||/||fdm||':>17}{'feat range':>13}")
print("-" * 66)
for g in EVAL_GAPS:
    i = _row[round(float(g), 6)]
    ref = (d["U_scat_re"][i] + 1j * d["U_scat_im"][i])[::ST, ::ST]
    if np.linalg.norm(ref) == 0.0:
        continue
    u, _ = predict(g)
    rel = float(np.linalg.norm(u - ref) / np.linalg.norm(ref))
    c = np.vdot(ref, u) / np.vdot(ref, ref)
    amp = float(np.linalg.norm(u) / np.linalg.norm(ref))
    # how far outside [-1, 1] the branch input has been pushed
    f = NS["param_to_features"](float(g)).cpu().numpy().ravel()
    fx = float(np.abs(f).max())
    rows.append((float(g), label_of(g), rel, float(abs(c)), amp, fx))
    print(f"{g:>7.3f}{label_of(g):>10}{rel:>9.3f}{abs(c):>8.3f}"
          f"{amp:>17.3f}{fx:>13.2f}")

G = np.array([r[0] for r in rows])
REL = np.array([r[2] for r in rows])
AMP = np.array([r[4] for r in rows])
LAB = [r[1] for r in rows]
FX = np.array([r[5] for r in rows])

for nm in ("trained", "held out", "interp", "EXTRAP"):
    m = [i for i, l in enumerate(LAB) if l == nm]
    if m:
        print(f"\n  {nm:<9}: {len(m):>2} gaps   mean relL2 {REL[m].mean():.3f}"
              f"   worst {REL[m].max():.3f}")

# The zero-field baseline: relL2 = 1 exactly. Worth drawing, since a model that
# has stopped predicting anything sits there.
print(f"\n  predicting zero would give relL2 = 1.000 at every gap")
_edge = 1.0 - (Y_SUB + 2 * a)          # gap at which the cylinder touches PML
print(f"  the cylinder touches the PML at gap {_edge:.3f}; from gap "
      f"{_edge - a:.3f} it is within one radius of it")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot
# ─────────────────────────────────────────────────────────────────────────────
TAG = os.path.splitext(os.path.basename(CKPT))[0].replace("_weights", "")
FIGD = os.path.join("variable_sample", "figures")
os.makedirs(FIGD, exist_ok=True)

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

for A in (ax, ax2):
    A.axvspan(G_LO, G_HI, color="tab:blue", alpha=0.08, zorder=0)
    A.axvspan(_edge - a, EVAL_GAPS[-1], color="tab:red", alpha=0.07, zorder=0)
    A.grid(alpha=0.3, which="both")

ax.axhline(1.0, color="k", ls=":", lw=1.2)
ax.text(EVAL_GAPS[-1], 1.0, "predicting zero ", va="bottom", ha="right",
        fontsize=8, color="0.3")
ax.semilogy(G, REL, "-", color="0.35", lw=1.4, zorder=2)
for nm, col, mk, sz in [("trained", "tab:blue", "o", 6),
                        ("held out", "tab:red", "s", 8),
                        ("interp", "tab:green", "^", 6),
                        ("EXTRAP", "tab:orange", "D", 5)]:
    m = [i for i, l in enumerate(LAB) if l == nm]
    if m:
        ax.semilogy(G[m], REL[m], mk, color=col, ms=sz, label=nm, zorder=3)
ax.set_ylabel("relative L2 error vs FDM")
ax.set_title(f"{TAG} — error against the tip-sample gap\n"
             f"blue band = training range   red band = cylinder within one "
             f"radius of the PML", fontsize=11)
ax.legend(loc="lower right", fontsize=9)

ax2.axhline(1.0, color="k", ls=":", lw=1.2)
ax2.plot(G, AMP, "-", color="0.35", lw=1.4)
ax2.plot(G, AMP, ".", color="tab:purple", ms=7)
ax2.set_ylabel("||u_net|| / ||u_fdm||")
ax2.set_xlabel("gap between cylinder and sample")
ax2.set_title("amplitude ratio — 1.0 is correct, 0 is the collapsed solution",
              fontsize=10)
ax2.set_ylim(bottom=0.0)

fig.tight_layout()
P1 = os.path.join(FIGD, f"{TAG}_gap_generalisation.png")
plt.savefig(P1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {P1}")

# ── fields at a few representative gaps ─────────────────────────────────────
# One trained gap, one held out, then the FIRST and LAST extrapolated ones --
# the near tail shows how fast it degrades, the far tail what it degrades to.
def _pick(nm, which):
    c = [i for i, l in enumerate(LAB) if l == nm]
    return c[which] if c else None


picks = [p for p in (_pick("trained", len([1 for l in LAB if l == "trained"])
                           // 2),
                     _pick("held out", 0),
                     _pick("EXTRAP", 0),
                     _pick("EXTRAP", -1)) if p is not None]
picks = list(dict.fromkeys(picks))

if picks:
    fig, axes = plt.subplots(3, len(picks), figsize=(3.4 * len(picks), 9.6),
                             squeeze=False)
    circ = dict(color="cyan", lw=1.0, fill=False, zorder=5, ls="--")
    subk = dict(color="yellow", lw=0.9, zorder=6, ls="--")
    for col, i in enumerate(picks):
        g = G[i]
        u, _ = predict(g)
        ref = (d["U_scat_re"][_row[round(float(g), 6)]]
               + 1j * d["U_scat_im"][_row[round(float(g), 6)]])[::ST, ::ST]
        sc = max(float(np.percentile(np.abs(ref), 99)), 1e-12)
        cy = Y_SUB + a + g
        for r, (dat, ttl) in enumerate(
                [(u.real, f"gap {g:.3f}  [{LAB[i]}]\nrelL2 {REL[i]:.3f}"
                          f"\nnetwork Re(u_scat)"),
                 (ref.real, "FDM Re(u_scat)"),
                 (u.real - ref.real, "network - FDM")]):
            A = axes[r, col]
            A.imshow(dat, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
                     vmin=-sc, vmax=sc, interpolation="bilinear")
            A.add_patch(plt.Circle((cx, cy), a, **circ))
            A.axhline(Y_SUB, **subk)
            A.set_title(ttl, fontsize=8)
            A.set_xticks([]); A.set_yticks([])
    fig.suptitle(f"{TAG} — fields across the gap range", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    P2 = os.path.join(FIGD, f"{TAG}_gap_fields.png")
    plt.savefig(P2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {P2}")
