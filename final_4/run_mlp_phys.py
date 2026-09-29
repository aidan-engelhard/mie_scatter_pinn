"""final_4 MLP, with the physics also imposed where there is no FDM data.

The 2x2 arm's standings make the case for trying it here first:

    mlp_dataonly   data only, no physics      unseen 0.0963, seen 0.0020
    mlp            physics + data             unseen 0.1189, seen 0.0350
    deeponet_norm64  physics + data + draws   unseen 0.1234, seen 0.0118
    deeponet       physics + data             unseen 0.1252, seen 0.0648

Every model interpolates about equally smoothly here (gap-axis spread
1.14-1.29x), so the ranking is set by how well each matches the FDM at the four
geometries it trained on -- and the physics terms, at W_FIELD/W_FLUX = 50
against the data's 1, cost the PINN MLP a 17x worse fit there than data alone.

What has NOT been tried is physics imposed where it can actually add
information: at parameters with no FDM solve. N_PHYS_EXTRA = 8 is twice the
four training geometries. Spans are left at 1.0 because final_4's error map is
nearly flat -- there is no ridge to aim at, unlike 4x2's.

Budget is data_experiment.MATCHED_*, the one budget every method on final_4 and
final_4x2 is compared at, so the comparison is of objectives and not of
training length.

RE-RUNNING THIS OVERWRITES the 0.0760 model in final_4/models, which was
trained at 20000 + 100000 and then continued for another 100000 (the
continuation made it worse: 0.0743 before it, 0.0760 after, with 62% of its
loss reduction going into the PML residual). The 0.0743 weights survive as
final_4_mlp_phys_pre_warm_weights.pt. Move both aside first if either is wanted
for the record.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))
import data_experiment as DX

# INT, not "4": GRIDS keys the square arms by integer.
ARM, ARCH = 4, "mlp"
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
    "W_DATA_SCAT": DX.W_DATA, "W_DATA_INT": DX.W_DATA, "W_DATA_SUB": DX.W_DATA,
    "ADAM_EPOCHS": DX.MATCHED_ADAM, "LBFGS_STEPS": DX.MATCHED_LBFGS,
    "N_PHYS_EXTRA": 8, "PHYS_EXTRA_WEIGHT": 1.0,
    "PHYS_SPAN_ALPHA": 1.0, "PHYS_SPAN_APEX": 1.0,
}

runpy.run_path(DX.NET_SCRIPT, run_name="__main__", init_globals={
    "LAM_OVERRIDE": DX.LAM,
    "FDM_PATH_OVERRIDE": npz,
    "ARCH_OVERRIDE": ARCH,
    "RUN_TAG_OVERRIDE": f"final_{ARM}_{ARCH}_phys",
    "FIG_DIR_OVERRIDE": figs,
    "MODEL_DIR_OVERRIDE": models,
    "INIT_WEIGHTS_OVERRIDE": None,
    "CONFIG_OVERRIDES": cfg,
})
