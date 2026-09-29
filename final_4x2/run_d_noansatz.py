"""final_4x2 MLP, data only, WITHOUT the skin ansatz -- the ablation that
justifies it.

With SKIN_ANSATZ on, net_int learns the ENVELOPE A and the field is
u_int = A * exp(-i k_int phi), phi being the signed distance to the tip at the
actual (alpha, gap). The tip interior is then pinned by construction rather
than learned, and it shows: measured against the 50x50 references, the interior
carries ~22% of the field energy and the models reach relL2 0.0010 there
(final_4x2 cell D) and 0.0006 (final_64), contributing ~0% of the squared
error. A fifth of the domain is right for free.

That is worth reporting, and reporting it means showing what happens without
it. This run is cell D with SKIN_ANSATZ off and nothing else changed, so the
difference is attributable to the ansatz alone.

WHY IT SHOULD BE HARD. n = 3.0 + 30.0i puts the skin depth at
1/Im(k_int) = 0.0053. Without the ansatz net_int has to represent a field that
decays by e^-1 over that distance directly, from Fourier features banded at
|k_int| = 189, with collocation points kept 0.005 clear of the surface
(_MARGIN_IN reverts to afm_pinn's value) -- so the layer that sets the
impedance is never sampled. That is the configuration the ansatz replaced.

BUDGET. MATCHED_*, not DATAONLY_*, even though cell D itself converges in
minutes at DATAONLY_*. An ablation that hands the alternative LESS optimisation
than the method it is being compared with proves nothing: the first objection
would be that it simply needed longer. At the matched budget that objection is
gone, and if the relL2 curve is still falling at the end, say so rather than
claiming the ansatz is necessary when it may only be faster.

    python final_4x2/run_d_noansatz.py
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

ARM, ARCH = '4x2', "mlp"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
_, _, models, figs = DX._paths(ARM)
os.makedirs(models, exist_ok=True)
os.makedirs(figs, exist_ok=True)
npz = DX.npz_path(ARM)
if not os.path.exists(npz):
    raise SystemExit(f"{npz} not found -- run: python final_{ARM}/run.py --fdm")

cfg = {
    "SINGLE_GEOMETRY": False,
    "N_PARAM_BATCH": min(DX.PARAM_BATCH, DX.n_samples(ARM)),
    "ACCUM_CHUNK": DX.ACCUM_CHUNK,
    "SEED": SEED,
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    "ADAM_EPOCHS": DX.DATAONLY_ADAM, "LBFGS_STEPS": DX.DATAONLY_LBFGS,
    # the one thing under test
    "SKIN_ANSATZ": False,
    # cell D, disabled by exactly the mechanism run_d.py uses, so this differs
    # from it in the ansatz and nothing else
    "PHYS_AT_DATA": False,
    "N_PHYS_EXTRA": 0, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 1.0,
}

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": f"final_{ARM}_{ARCH}_d_noansatz"
                        + (f"_s{SEED}" if SEED else ""),
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": None,
    "CONFIG_OVERRIDES": cfg,
})
