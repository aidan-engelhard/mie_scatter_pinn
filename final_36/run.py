"""
final_36 — the 6x6 = 36-sample arm of the data-quantity experiment.

Every setting lives in final/data_experiment.py, shared by all five arms, and
the physics lives in final/final_fdm.py and final/final_deeponet.py. This file
only says WHICH arm it is, so the arms cannot drift apart.

    python final_36/run.py --fdm                solve the 6x6 sweep into inputs/
    python final_36/run.py --deeponet           train the DeepONet
    python final_36/run.py --mlp                train the plain-MLP control
    python final_36/run.py --both               both architectures, in turn
    python final_36/run.py --both --quick       short budgets, for timing
    python final_36/run.py --deeponet --warm    continue training from the existing
                                                checkpoint, L-BFGS only. Extends this arm's
                                                figures, relL2 log and checkpoint rather than
                                                starting a second set
    python final_36/run.py --eval               score the trained models on unseen
                                               parameters (final/evaluate_arms.py)
    python final_36/run.py --eval-fine          the same on a 50x50 grid: heat map only,
                                               2500 FDM solves the first time

Outputs
    final_36/inputs/fdm_final_36.npz
    final_36/models/final_36_deeponet_weights.pt
    final_36/models/final_36_mlp_weights.pt
    final_36/models/final_36_<arch>_pre_warm_weights.pt  what --warm started from
    final_36/figures/final_36_<arch>_*.png, fdm_final_sweep_*.png
    final_36/figures/final_36_<arch>_eval_k5*    the --eval results
    final_36/figures/final_36_<arch>_eval_k50.*  the --eval-fine results
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))

from data_experiment import main   # noqa: E402

N_SAMPLES = 36

if __name__ == "__main__":
    main(N_SAMPLES, sys.argv[1:])
