# legacy

Things kept because they are recoverable evidence, not because anything still
runs them.

## The scripts

`final_deeponet_with_field_figs.py` and `evaluate_arms_with_field_figs.py` are
copies taken immediately before the per-run field figures were removed. They
still work. Use them if the writeup needs field panels again:

* the trainer keeps `save_frame` and the training MP4, the four sweep-overview
  grids, the PINN-vs-FDM field comparison and the error-vs-half-angle plot;
* the evaluation script keeps `_plot_fields`, the model-vs-FDM panels at the
  best, median and worst unseen geometry.

`save_frame` was not only an end-of-run cost -- it rendered a field figure on
every evaluation during training, which is why removing it was worth doing
before running the comparison grid.

## The results

`final_4/{models,figures}/legacy/` and `final_4x2/{models,figures}/legacy/`
hold every run trained before the budgets were matched. Their L-BFGS budgets
range from 10000 to 250000 evaluations and each is a single seed, so they can
be quoted individually but not compared with each other. Two of them are worth
keeping in mind:

* `final_4_mlp_phys` -- the first model on the 2x2 arm to beat bilinear
  interpolation of the four EXACT FDM corner fields (0.0743 against 0.1212).
  Its `_pre_warm_weights.pt` is that 0.0743 model; the plain `_weights.pt` is
  after a 100000-evaluation continuation that made it worse (0.0760), with 62%
  of the continuation's loss reduction going into the PML residual.
* `final_4_mlp` at 269809 L-BFGS evaluations -- cell DP at far more than the
  matched budget, and still worse than the alternatives. It is the convergence
  control for the matched comparison: DP loses at 3.5x the optimisation too.

## The physics-only run, and why it is a result

`final_4_mlp_physonly_*` is a training run with `W_DATA_* = 0`: the seven
physics terms alone, imposed at 16 parameter points redrawn through training.
It was stopped at 41949 L-BFGS evaluations. It has no final `_weights.pt`
because it was killed, not because it crashed.

It did not fail by diverging. It failed by succeeding at the wrong thing:

    step        relL2   mag_ratio      total    field     flux  pde_sub
    adam 1     0.8673      0.8853   1.045e+02  4.97e+01 5.03e+01 9.79e-01
    adam 20000 0.7101      0.2663   3.872e-01  1.80e-02 1.72e-03 2.80e-03
    lbfgs 41949 0.6887     0.2545   8.740e-02  1.43e-02 3.14e-03 2.73e-03

The objective fell by three orders of magnitude and every physics residual came
down with it -- field continuity 49.7 -> 0.014, flux 50.3 -> 0.003 -- while
relL2 stalled at 0.69 and `mag_ratio` walked AWAY from the FDM target of
0.7193, down to 0.25 and still falling. The network found a field that
satisfies the operator, the interface conditions, the PML and the outer
Dirichlet condition to 1e-2, and whose amplitude is a quarter of the incident
wave's.

That is the expected shape of a homogeneous problem with no data. The exterior
PDE is homogeneous, `l_pde_sub` is homogeneous (u = 0 satisfies it exactly),
the outer condition is u_scat = 0, and the only inhomogeneous forcing is the
incident field entering through the tip-surface continuity conditions -- a
measure-zero set against the collocation volume. Driving those two surface
terms to 1e-2 while everything shrinks is cheaper than finding the true field,
so that is what the optimiser does.

The result to carry into the writeup: in this formulation the data is not
accelerating convergence, it is SELECTING the solution. The seven physics terms
do not determine the field on their own.

Anyone tempted to retry it should change the formulation rather than the
budget -- an amplitude constraint, a non-trivial source, or a single data point
to fix the scale. More epochs will not help; the loss was already at 8.7e-2 and
falling.
