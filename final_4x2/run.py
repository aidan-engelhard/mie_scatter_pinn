"""
final_4x2 — 4 half-angles x 2 tip-sample gaps = 8 FDM samples.

One of the two arms that refine a SINGLE axis. final_4 (2x2) showed that two
gaps already interpolate well across the range, while the error depends mostly
on the half-angle, so these arms hold the gap axis at its two ends (0.01 and
0.20) and ask how many ANGLES the surrogate needs: 4 here, 6 in final_6x2.

Every setting lives in final/data_experiment.py, shared by all arms, and the
physics lives in final/final_fdm.py and final/final_deeponet.py. This file only
says WHICH arm it is, so the arms cannot drift apart.

    python final_4x2/run.py --fdm                solve the 4x2 sweep into inputs/
    python final_4x2/run.py --deeponet           train the DeepONet
    python final_4x2/run.py --mlp                train the plain-MLP control
    python final_4x2/run.py --both               both architectures, in turn
    python final_4x2/run.py --both --quick       short budgets, for timing
    python final_4x2/run.py --deeponet --warm    continue training from the existing
                                                 checkpoint, L-BFGS only. Extends this
                                                 arm's figures, relL2 log and checkpoint
                                                 rather than starting a second set
    python final_4x2/run.py --eval               score the trained models on unseen
                                                 parameters (final/evaluate_arms.py)
    python final_4x2/run.py --eval-fine          the same on a 50x50 grid: heat map only

Outputs
    final_4x2/inputs/fdm_final_4x2.npz
    final_4x2/models/final_4x2_<arch>_weights.pt
    final_4x2/models/final_4x2_<arch>_pre_warm_weights.pt  what --warm started from
    final_4x2/figures/final_4x2_<arch>_*.png, fdm_final_sweep_*.png
    final_4x2/figures/final_4x2_<arch>_eval_k5*    the --eval results
    final_4x2/figures/final_4x2_<arch>_eval_k50.*  the --eval-fine results
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))

from data_experiment import main   # noqa: E402

ARM = "4x2"

if __name__ == "__main__":
    main(ARM, sys.argv[1:])
