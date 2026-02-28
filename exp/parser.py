from distutils.util import strtobool
import argparse


def str2bool(x):
    if type(x) == bool:
        return x
    elif type(x) == str:
        return bool(strtobool(x))
    else:
        raise ValueError(f'Unrecognised type {type(x)}')

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))


def get_parser():
    parser = argparse.ArgumentParser()
    # Optimisation params
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--sheaf_decay', type=float, default=None)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--min_acc', type=float, default=0.0,
                        help="Minimum test acc on the first fold to continue training.")
    parser.add_argument('--stop_strategy', type=str, choices=['loss', 'acc'], default='loss')

    # Model configuration
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--normalised', dest='normalised', type=str2bool, default=True,
                        help="Use a normalised Laplacian")
    parser.add_argument('--deg_normalised', dest='deg_normalised', type=str2bool, default=False,
                        help="Use a degree-normalised Laplacian")
    parser.add_argument('--linear', dest='linear', type=str2bool, default=False,
                        help="Whether to learn a new Laplacian at each step.")
    parser.add_argument('--hidden_channels', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--left_weights', dest='left_weights', type=str2bool, default=True,
                        help="Applies left linear layer")
    parser.add_argument('--right_weights', dest='right_weights', type=str2bool, default=True,
                        help="Applies right linear layer")
    parser.add_argument('--add_lp', dest='add_lp', type=str2bool, default=False,
                        help="Adds fixed high pass filter in the restriction maps")
    parser.add_argument('--add_hp', dest='add_hp', type=str2bool, default=False,
                        help="Adds fixed low pass filter in the restriction maps")
    parser.add_argument('--use_act', dest='use_act', type=str2bool, default=True)
    parser.add_argument('--second_linear', dest='second_linear', type=str2bool, default=False)
    parser.add_argument('--orth', type=str, choices=['matrix_exp', 'cayley', 'householder', 'euler'],
                        default='householder', help="Parametrisation to use for the orthogonal group.")
    parser.add_argument('--sheaf_act', type=str, default="tanh",
                        help="Activation to use in sheaf learner.")
    parser.add_argument('--edge_weights', dest='edge_weights', type=str2bool, default=True,
                        help="Learn edge weights for connection Laplacian")
    parser.add_argument('--sparse_learner', dest='sparse_learner', type=str2bool, default=False)

    # Experiment parameters
    parser.add_argument('--dataset', default='texas')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--model', type=str, choices=[
        # Discrete baseline
        'DiagSheaf', 'BundleSheaf', 'GeneralSheaf',
        # Discrete polynomial
        'DiagSheafPolynomial', 'BundleSheafPolynomial', 'GeneralSheafPolynomial',
        # ODE variants
        'DiagSheafODE', 'BundleSheafODE', 'GeneralSheafODE',
        # ODE polynomial
        'DiagSheafODEPolynomial', 'BundleSheafODEPolynomial', 'GeneralSheafODEPolynomial'
    ], default=None)
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None,
                        help="W&B project name. Defaults to an internal name if not set.")
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], default='classification',
                        help="Task type: 'classification' (nll_loss + accuracy) or 'regression' (mse_loss + MAE).")
    parser.add_argument('--learning_mode', type=str, choices=['transductive', 'inductive'], default='transductive',
                        help="Whether to use inductive splits (if supported by dataset, e.g. TokyoRailway).")
    parser.add_argument('--evectors', type=int, default=0,
                        help="Number of Laplacian PE eigenvectors to use.")
    parser.add_argument('--norm', type=str, default='global', choices=['global', 'row'],
                        help="Normalization mode for Tokyo Railway: 'global' (single min/max) or 'row' (per-station min/max)")

    # ---------- Polynomial filter args (generalized) ----------
    parser.add_argument("--lambda_max_choice", choices=["analytic", "iterative"], default="analytic",
                        help="How to estimate/upper-bound λ_max for polynomial spectral filters.")
    parser.add_argument("--polynomial_type", type=str, default="ChebyshevType1",
                        choices=["Chebyshev", "ChebyshevType1", "ChebyshevType2",
                                 "ChebyshevType3", "ChebyshevType4", "ChebyshevInterpolation",
                                 "Legendre", "Gegenbauer", "Jacobi"],
                        help="Polynomial family for spectral filtering.")
    # General order K (preferred)
    parser.add_argument("--poly_layers_K", type=int, default=3,
                        help="Order K of the polynomial filter.")
    # Backward compatibility (still parsed, classes fall back to it if poly_layers_K is unset)
    parser.add_argument("--chebyshev_layers_K", type=int, default=3,
                        help="(Deprecated) Chebyshev order K; kept for backward compatibility.")

    # Family-specific knobs
    parser.add_argument("--gegenbauer_lambda", type=float, default=1.0,
                        help="λ parameter for Gegenbauer (must be > 0).")
    parser.add_argument("--jacobi_alpha", type=float, default=0.0,
                        help="α parameter for Jacobi (must be > -1).")
    parser.add_argument("--jacobi_beta", type=float, default=0.0,
                        help="β parameter for Jacobi (must be > -1).")

    # ---------- ODE args ----------
    parser.add_argument('--max_t', type=float, default=1.0, help="Maximum integration time.")
    parser.add_argument('--int_method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')

    # ---------- Misc / legacy experiment controls (kept for compatibility) ----------
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "Used if getting OOM errors at test time")
    parser.add_argument("--maps_lr", type=float, default=None)
    parser.add_argument("--classes_corr", type=list_of_floats, default=None)
    parser.add_argument("--num_nodes", type=int, default=200)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--het_coef", type=float, default=0.9)
    parser.add_argument("--edge_noise", type=float, default=0.05)
    parser.add_argument("--node_degree", type=int, default=10)
    parser.add_argument("--num_feats", type=int, default=10)
    parser.add_argument("--new_synth_data", type=str2bool, default=False)
    parser.add_argument("--feat_noise", type=float, default=0.25)
    parser.add_argument("--new_synth_edges", type=str2bool, default=False)
    parser.add_argument("--use_epsilons", type=str2bool, default=True)
    parser.add_argument("--use_embedding", type=str2bool, default=True)
    parser.add_argument("--ellipsoid_radius", type=float, default=1)
    parser.add_argument("--just_add_noise", type=str2bool, default=False)
    parser.add_argument("--ellipsoids", type=str2bool, default=True)
    parser.add_argument("--rotation_invariant_sheaf_learner", type=str2bool, default=False, help="Use the RotationInvariantSheafLearner (O(d) maps).")
    parser.add_argument("--node_edge_sims_time_dependent", type=str2bool, default=False, help="Make node-edge similarity features time-dependent for the rotation-invariant learner.")

    # ---------- Resource Analysis ----------
    parser.add_argument("--resource_analysis", action="store_true",
                            help="Enable logging of system/GPU/time/FLOPs metrics to W&B.")
    parser.add_argument("--sys_log_every_s", type=float, default=1.0,
                            help="Resource polling period in seconds (only if resource_analysis).")
    parser.add_argument("--profile_flops", action="store_true",
                            help="Enable torch.profiler FLOPs (best-effort; only if resource_analysis).")
    parser.add_argument("--flops_profile_epochs", type=int, default=1,
                            help="Number of initial epochs per fold to profile for FLOPs (only if resource_analysis).")
    parser.add_argument("--deterministic", action="store_true",
                            help="Enable deterministic flags for better reproducibility.")
    parser.add_argument("--strict_determinism", action="store_true",
                            help="Try strict deterministic algorithms (may warn/error on sparse ops).")

    # ---------- Restriction map saving ----------
    parser.add_argument("--save_restriction_maps", action="store_true",
                        help="Save restriction maps from the best epoch to <save_dir>/restriction_maps.pt for visualization.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save restriction_maps.pt. Defaults to checkpoints/<dataset>/<model>_seed<seed>.")

    return parser