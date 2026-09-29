"""
final_6x2 — 6 half-angles x 2 tip-sample gaps = 12 FDM samples.

The denser of the two single-axis arms; see final_4x2/run.py for why the gap
axis stays at two points (0.01 and 0.20) while only the half-angle is refined.
Against final_4x2's 4 angles, this one has 6.

Every setting lives in final/data_experiment.py, shared by all arms, and the
physics lives in final/final_fdm.py and final/final_deeponet.py. This file only
says WHICH arm it is, so the arms cannot drift apart.

    python final_6x2/run.py --fdm                solve the 6x2 sweep into inputs/
    python final_6x2/run.py --deeponet           train the DeepONet
    python final_6x2/run.py --mlp                train the plain-MLP control
    python final_6x2/run.py --both               both architectures, in turn
    python final_6x2/run.py --both --quick       short budgets, for timing
    python final_6x2/run.py --deeponet --warm    continue training from the existing
                                                 checkpoint, L-BFGS only. Extends this
                                                 arm's figures, relL2 log and checkpoint
                                                 rather than starting a second set
    python final_6x2/run.py --eval               score the trained models on unseen
                                                 parameters (final/evaluate_arms.py)
    python final_6x2/run.py --eval-fine          the same on a 50x50 grid: heat map only

Outputs
    final_6x2/inputs/fdm_final_6x2.npz
    final_6x2/models/final_6x2_<arch>_weights.pt
    final_6x2/models/final_6x2_<arch>_pre_warm_weights.pt  what --warm started from
    final_6x2/figures/final_6x2_<arch>_*.png, fdm_final_sweep_*.png
    final_6x2/figures/final_6x2_<arch>_eval_k5*    the --eval results
    final_6x2/figures/final_6x2_<arch>_eval_k50.*  the --eval-fine results
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))

from data_experiment import main   # noqa: E402

ARM = "6x2"

if __name__ == "__main__":
    main(ARM, sys.argv[1:])
