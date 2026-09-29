"""
final_100 — the 10x10 = 100-sample arm of the data-quantity experiment.

Every setting lives in final/data_experiment.py, shared by all five arms, and
the physics lives in final/final_fdm.py and final/final_deeponet.py. This file
only says WHICH arm it is, so the arms cannot drift apart.

    python final_100/run.py --fdm                solve the 10x10 sweep into inputs/
    python final_100/run.py --deeponet           train the DeepONet
    python final_100/run.py --mlp                train the plain-MLP control
    python final_100/run.py --both               both architectures, in turn
    python final_100/run.py --both --quick       short budgets, for timing
    python final_100/run.py --deeponet --warm    continue training from the existing
                                                 checkpoint, L-BFGS only. Extends this arm's
                                                 figures, relL2 log and checkpoint rather than
                                                 starting a second set
    python final_100/run.py --eval               score the trained models on unseen
                                               parameters (final/evaluate_arms.py)
    python final_100/run.py --eval-fine          the same on a 50x50 grid: heat map only,
                                               2500 FDM solves the first time

Outputs
    final_100/inputs/fdm_final_100.npz
    final_100/models/final_100_deeponet_weights.pt
    final_100/models/final_100_mlp_weights.pt
    final_100/models/final_100_<arch>_pre_warm_weights.pt  what --warm started from
    final_100/figures/final_100_<arch>_*.png, fdm_final_sweep_*.png
    final_100/figures/final_100_<arch>_eval_k5*    the --eval results
    final_100/figures/final_100_<arch>_eval_k50.*  the --eval-fine results
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "final"))

from data_experiment import main   # noqa: E402

N_SAMPLES = 100

if __name__ == "__main__":
    main(N_SAMPLES, sys.argv[1:])
