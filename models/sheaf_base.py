import numpy as np
import torch
from torch import nn


class SheafDiffusion(nn.Module):
    """Base class for sheaf diffusion models."""

    def __init__(self, edge_index, args):
        super(SheafDiffusion, self).__init__()

        # ---- Core graph / fiber config ---------------------------------------------------------
        assert args['d'] > 0, "Fiber dimension d must be > 0"
        self.d = args['d']
        self.edge_index = edge_index                           # [2, E].
        self.add_lp = args.get('add_lp', False)                # Add a global low-pass channel.
        self.add_hp = args.get('add_hp', False)                # Add a global high-pass channel.

        # Final fiber size = d (+ optional lp / hp channels).
        self.final_d = self.d + (1 if self.add_hp else 0) + (1 if self.add_lp else 0)

        # ---- Runtime / dataset info ------------------------------------------------------------
        self.device = args['device']
        self.graph_size = args['graph_size']                   # Number of nodes.
        self.layers = args['layers']                           # Diffusion layers.

        # Laplacian normalizations.
        self.normalised = args['normalised']                   # symmetric normalization (D^{-1/2} L D^{-1/2}).
        self.deg_normalised = args['deg_normalised']           # degree normalization (D^{-1} L).

        # Nonlinearity regime: if linear=False we rebuild sheaf at each layer.
        self.nonlinear = not args['linear']

        # Dropouts / activations.
        self.input_dropout = args.get('input_dropout', 0.0)    # On raw node features.
        self.dropout = args.get('dropout', 0.0)                # Between diffusion steps.
        self.use_act = args.get('use_act', True)

        # Linear “sandwich” flags around diffusion.
        self.left_weights = args.get('left_weights', True)
        self.right_weights = args.get('right_weights', True)

        # Sheaf learner specifics.
        self.sparse_learner = args.get('sparse_learner', False)
        self.sheaf_act = args.get('sheaf_act', "tanh")
        self.second_linear = args.get('second_linear', False)
        self.orth_trans = args.get('orth', 'householder')      # Orthogonal parameterization type (used by bundle/general).
        self.use_edge_weights = args.get('edge_weights', True) # Learn scalar edge weights (bundle/general).
        self.use_embedding = args.get('use_embedding', True)

        # Feature dims.
        self.input_dim = args['input_dim']
        self.hidden_channels = args['hidden_channels']
        self.output_dim = args['output_dim']

        # Task type: 'classification' or 'regression'.
        self.task = args.get('task', 'classification')

        # ODE options (for continuous variants).
        self.t = args.get('max_t', 1.0)
        self.time_range = torch.tensor([0.0, self.t], device=self.device)

        # Will be set by subclasses to a builder object that can generate a sparse Laplacian.
        self.laplacian_builder = None

        # ---- Polynomial spectral filter configuration -----------------------------------------
        # Generic configuration (Chebyshev as default).
        self.lambda_max_choice = args.get('lambda_max_choice', 'analytic')
        # Backward compat: keep chebyshev_layers_K, but prefer poly_layers_K when present.
        self.chebyshev_layers_K = args.get('chebyshev_layers_K', 3)
        self.poly_layers_K = args.get('poly_layers_K', self.chebyshev_layers_K)
        self.polynomial_type = args.get('polynomial_type', 'ChebyshevType1')

        # Family-specific knobs (no-ops unless your subclass uses them).
        self.gegenbauer_lambda = args.get('gegenbauer_lambda', 1.0)  # λ > 0
        self.jacobi_alpha = args.get('jacobi_alpha', 0.0)            # α > -1
        self.jacobi_beta = args.get('jacobi_beta', 0.0)              # β > -1

        # No rotation-invariant path: just use plain dropout for sheaf learners.
        self.sheaf_dropout = self.dropout

        # ---- Per-edge external weights (valued masking, e.g. distance/correlation) ----
        # Shape [E] matching edge_index.  None means "all ones" (no modulation).
        _ew = args.get('sheaf_edge_weights', None)
        if _ew is not None and isinstance(_ew, torch.Tensor):
            self.register_buffer('sheaf_edge_weights', _ew.float())
        elif _ew is not None and isinstance(_ew, (list, np.ndarray)):
            self.register_buffer('sheaf_edge_weights', torch.tensor(_ew, dtype=torch.float32))
        else:
            # None, or wandb serialized it to a string — treat as disabled
            self.sheaf_edge_weights = None

        # ---- Hidden “flattened fiber” dimension ------------------------------------------------
        # If no explicit dim_list is provided, default to [final_d, hidden_channels].
        dim_list = args.get('dim_list', [])
        if not dim_list:
            dim_list = [self.final_d, self.hidden_channels]
        self.dim_list = dim_list

        hidden_dim = 1
        for di in dim_list:
            hidden_dim *= di
        self.hidden_dim = hidden_dim

    # --------------------------------------------------------------------------------------------
    # Utility methods used by Discrete Models.
    # --------------------------------------------------------------------------------------------
    def update_edge_index(self, edge_index):
        """
        Update the edge_index (graph connectivity).
        If a Laplacian builder exists, refresh it too.
        """
        assert edge_index.max() <= self.graph_size, "edge_index contains a node id >= graph_size"
        self.edge_index = edge_index
        if self.laplacian_builder is not None:
            self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)

    def grouped_parameters(self):
        """
        Separate parameters that belong to the sheaf learners from the rest, so
        you can apply a different optimizer group (LR/WD) to the sheaf maps.

        We include both 'sheaf_learners' (correct ModuleList name) and the legacy
        'sheaf_learner' substring for backward compatibility with older code.
        """
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if ("sheaf_learners" in name) or ("sheaf_learner" in name):
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others