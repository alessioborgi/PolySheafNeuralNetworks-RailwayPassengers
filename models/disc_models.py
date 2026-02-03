# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from operator import pos
import torch
import torch.nn.functional as F
import torch_sparse
import warnings
from typing import Optional
from torch import nn
from models.sheaf_base import SheafDiffusion
from models.polynomial_sheaf_base import PolynomialSheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerVariant, RotationInvariantSheafLearner
from lib import laplace as lap
from models.orthogonal import Orthogonal

 


class DiscreteDiagSheafDiffusion(SheafDiffusion):
    '''Discrete Sheaf Diffusion with diagonal-maps.'''

    def __init__(self, edge_index, args):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        # 1) Learnable Linear Maps definition. 
        # Learnable Linear map of size [hidden_channels, hidden_channels], sitting AFTER each graph convolution.
        # From left to right. 
        self.lin_right_weights = nn.ModuleList()
        # Learnable Linear map of size [hidden_channels, hidden_channels], sitting BEFORE each graph convolution.
        # From right to left.
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        # Initialize the linear maps with orthogonal(right map) and identity(left map) weights.
        for i in range(self.layers):
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)
        
        # 2) Sheaf Learners Layers definition.
        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):

            if self.sparse_learner:
                # Sparse: Learns per-edge diagonal maps via LocalConcat variant.
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
                
            # elif self.rotation_learner:
            #     # Rotation-Invariant: Learns per-edge diagonal maps via RotationInvariantSheafLearner.
            #     self.sheaf_learners.append(RotationInvariantSheafLearner(self.d,
            #                                                               self.hidden_channels,
            #                                                               self.edge_index,
            #                                                               self.graph_size,
            #                                                               time_dep=self.time_dep,
            #                                                               out_shape=(self.d,),
            #                                                               transform = torch.diag,
            #                                                               sheaf_act=self.sheaf_act))
            else:
                # Default: Learns per-edge diagonal maps via LocalConcatSheafLearner (concatenation+linear).
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        
        # 3) Laplacian Builder definition.
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)
        
        # Given the per-edge diagonal maps of size (edges × d), we constructs a sparse graph Laplacian L where 
        # each edge weight is −map_i · map_j and the diagonal ensures row‐sums zero.
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons']))  # epsilon[layer] parametrizes the residual update: x^{t+1} = (1 + tanh(epsilon)) x^{t} - L*x^{t}  .

        #self.lin_alt1 = nn.Linear(self.input_dim,self.hidden_channels+self.d)
        #self.lin_alt2 = nn.Linear(self.d,self.d)
        #self.lin_alt3 = nn.Linear(self.hidden_channels,self.hidden_channels)
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)                  # First layer mapping input features to hidden features.
        if self.second_linear:                                                  # Second layer before diffusion. 
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)                 # Final laayer mapping per-node features back to class logits. 

    def forward(self, x):

        # 1) Initial embedding + dropout + activation
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            
            x = self.lin1(x)
            
            if self.use_act:
                x = F.elu(x)
        '''
        x1 = self.lin_alt1(x)
        xf = x1[:,0:self.hidden_channels]
        xd = x1[:,self.hidden_channels:]
        x = torch.bmm(xd.reshape(-1,self.d,1),xf.reshape(-1,1,self.hidden_channels))
        x = F.elu(x).view(self.graph_size,-1)
        '''
        # 2) (Optional) second linear + sheaf‐dropout → reshape to long vector.
        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        # 3) Loop over Diffusion Layers.
        x0, maps = x, None
        for layer in range(self.layers):
            # 3.1) If first layer or nonlinear, compute new sheaf-diagonal-maps on each edge. 
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1) # Reshape to (N, hidden_dim) for sheaf learner.
                learner = self.sheaf_learners[layer]

                # The rotation invariant learner expects also old maps too. 
                if isinstance(learner, RotationInvariantSheafLearner):
                    # Expects (x, edge_index, old_maps).
                    maps = learner(x_maps, self.edge_index, maps)
                else:
                    # All the others only want (x, edge_index).
                    maps = learner(x_maps, self.edge_index)
                
                # Build Laplacian and set the laplacian in the sheaf learner.
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            # 3.2) Do one graph convolution and optionally apply linear maps (sandwich with left/right weights).
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                # Apply left‐linear per‐channel.
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                # Apply right‐linear per‐channel.
                x = self.lin_right_weights[layer](x)

            # 3.3) Apply the Laplacian to the current features. Sparse matrix multiply: x ← L · x   .
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            # 3.4) Apply activation function + residual update. x₀ ← coeff * x₀ – x     .
            if self.use_act:
                x = F.elu(x)

            # The coeff is the “momentum” or “step‐size” multiplying the previous state before you subtract the graph‐Laplacian term.
            # coeff_{n,c} = (1 + tanh(epsilon_{c})) for node n and channel c. 
            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        # 4) Reshape back to Nxhidden_dim and apply final linear layer + log_softmax.
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)



class DiscreteBundleSheafDiffusion(SheafDiffusion):
    '''Discrete Sheaf Diffusion with bundle-sheaf-maps.'''

    def __init__(self, edge_index, args):
        super(DiscreteBundleSheafDiffusion, self).__init__(edge_index, args)

        assert args['d'] > 1
        assert not self.deg_normalised          # Bundle sheaf diffusion does not support degree-normalised Laplacian.

        # 1) Learnable Linear Maps definition.
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Learnable Linear map of size [hidden_channels, hidden_channels], sitting AFTER each graph convolution and initialize with orthogonal weights.
        for i in range(self.layers):
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        # Learnable Linear map of size [final_d, final_d], sitting BEFORE each graph convolution and initialize with identity weights.
        for i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)


        self.sheaf_learners = nn.ModuleList()       # This will produce the per-edge connection maps.
        self.weight_learners = nn.ModuleList()      # This will (optionally) produce scalar edge-weights to modulate those maps.
        # Decide how many times to recompute the sheaf connection maps. (If you're learning nonlinearly, get one learner per layer, 
        # otherwise, through linear diffusion, you learn at the very first layer).
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)

        # 2) Decide which sheaf learner to use.
        for i in range(num_sheaf_learners):

            if self.sparse_learner:
                # Sparse: Learns per-edge bundle-sheaf maps via LocalConcatSheafLearnerVariant.
                # It usses a variant of the local-concatenation trick to predict each edge’s diagonal map from the two endpoint features.
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            
            # elif self.rotation_learner:
            #     # Rotation-Invariant: Learns per-edge bundle-sheaf maps via RotationInvariantSheafLearner.
            #     # It predicts a parameter vector which is turned into an orthogonal matrix via the chosen orthogonal transform,
            #     # ensuring the learned connection lie in O(d).
            #     self.sheaf_learners.append(RotationInvariantSheafLearner(self.d,
            #                                                               self.hidden_channels,
            #                                                               self.edge_index,
            #                                                               self.graph_size,
            #                                                               time_dep=self.time_dep,
            #                                                               transform = Orthogonal(d=self.d, orthogonal_map=self.orth_trans),
            #                                                               out_shape=(self.get_param_size(),),
            #                                                               sheaf_act=self.sheaf_act))
            else:
                # Default: Learns per-edge bundle-sheaf maps via LocalConcatSheafLearner (concatenation+linear).
                # It is a simple linear concatenation of endpoint features, followed by activation, to predict each edge’s connection parameters.
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

            # If you have asked to learn per-edge weights (to re-weight the Laplacian beyond the connection maps), add the edge weight learner.    
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        
        # 3) Connection Laplacian Builder instantiation.        
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)
        # We always learn also the epsilons, which are the residual update coefficients.
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons']))
        
        # 4) Linear layers definition.
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        ''' Function to see how many free parameters each edge-map learner must predict, depending on the chosen orthogonal parameterisation.'''
        
        # For full skew-symmetric maps, we need d(d+1)/2 parameters.
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        # Otherwise it's the off-diagonals only, i.e.,  d(d-1)/2 parameters.
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        ''' Function to apply the left and right linear maps to the input features x.'''
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        ''' Function used if you change the graph structure at runtime, update both the base Laplacian builder and 
            any edge-weight learners to point at the new edges.'''
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x):
        ''' Forward pass of the Discrete Bundle Sheaf Diffusion model.'''
        # 1) Initial embedding + dropout + activation.
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)
        
        # 2) (Optional) second linear + sheaf‐dropout → reshape to long vector (size (num_nodes x fiber_dim) x hidden_channels).
        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)  # Reshape to (N, fiber_dim) for sheaf learner.

        x0, L, maps = x, None, None
        # 3) Loop over Diffusion Layers.
        for layer in range(self.layers):
            
            # 3.1) If first layer or nonlinear regime, compute the per-edge connection parameters. 
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                # The rotation invariant learner expects also old maps too.
                if isinstance(learner, RotationInvariantSheafLearner):
                    # Expects (x, edge_index, old_maps).
                    maps = learner(x_maps, self.edge_index, maps)
                else:
                    # All the others only want (x, edge_index).
                    maps = learner(x_maps, self.edge_index)
                
                # Optionally, produce scalar edge weights, then build the full sparse Laplacian L=(index, values).
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            # Graph convolution sandwich with left/right linear maps.
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal. A sparse matrix multiply: x ← L · x    .
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            # 3.4) Apply activation function + residual update. x₀ ← coeff * x₀ – x     .
            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)


class DiscreteGeneralSheafDiffusion(SheafDiffusion):
    '''Discrete Sheaf Diffusion with general sheaf-maps.'''

    def __init__(self, edge_index, args):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        # 1) Learnable Linear Maps definition to apply before/after each graph diffusion step.
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        # Initialize the linear maps with orthogonal(right map) and identity(left map) weights.
        for i in range(self.layers):
            self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()   # Output d × d sheaf-maps for each edge.
        self.weight_learners = nn.ModuleList()  # Optional: output edge weights to modulate (scalar weights) the sheaf-maps.
        # If you’re doing nonlinear diffusion, you re-learn maps at each layer; otherwise learn once.
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)

        # 2) Decide which sheaf learner to use.
        for i in range(num_sheaf_learners):
            # Sparse: Learns per-edge sheaf-maps via LocalConcatSheafLearnerVariant.
            # It learns each full d×d map via a two-stage concatenation variant.
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            # Rotation-Invariant: Learns per-edge sheaf-maps via RotationInvariantSheafLearner.
            # It predicts a parameter vector which is turned into an orthogonal d×d (but here only if you set it to produce full maps).      
            # elif self.rotation_learner:
            #     self.sheaf_learners.append(RotationInvariantSheafLearner(self.d,
            #                                                               self.hidden_channels,
            #                                                               self.edge_index,
            #                                                               self.graph_size,
            #                                                               time_dep=self.time_dep,
            #                                                               out_shape=(self.d,self.d),
            #                                                               sheaf_act=self.sheaf_act))
            # Default: Learns per-edge sheaf-maps via LocalConcatSheafLearner (concatenation+linear).
            # It does a simple concatenation+linear to directly output each d×d map.
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        
        # Build a General Laplacian Builder from d x d sheaf-maps.
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)
        # Per-layer "momentum" parameters controlling the residual update scale.
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons']))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right):
        ''' Function to apply the left and right linear maps to the input features x.'''
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        ''' Forward pass of the Discrete General Sheaf Diffusion model.'''
        # 1) Initial embedding + dropout + activation.
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        # 2) (Optional) second linear + sheaf‐dropout → reshape to long vector (size (num_nodes x fiber_dim) x hidden_channels).
        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        # 3) Loop over Diffusion Layers.
        x0, L, maps = x, None, None
        for layer in range(self.layers):
            # 3.1) If first layer or nonlinear regime, compute the per-edge sheaf-maps.
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                # The rotation invariant learner expects also old maps too.
                if isinstance(learner, RotationInvariantSheafLearner):
                    # This one expects (x, edge_index, old_maps)
                    maps = learner(x_maps, self.edge_index, maps)
                else:
                    # All the others only want (x, edge_index)
                    maps = learner(x_maps, self.edge_index)
                L, trans_maps = self.laplacian_builder(maps) 
                self.sheaf_learners[layer].set_L(trans_maps)

            # 3.2) Do one graph convolution and optionally apply linear maps (sandwich with left/right weights).
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # 3.3) Graph Diffusion step.
            # Use the adjacency matrix rather than the diagonal. 
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD (i.e, NaN/Infs).
        assert torch.all(torch.isfinite(x))
        # 4) Reshape back to (N, hidden_dim) and apply final linear layer + log_softmax.
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)
















#################################################################################################################################
######################### POLYNOMIAL MODELS ( Orthogonal bases on [−1,1] ) ######################################################
#################################################################################################################################

#################################################################################################################################
######################### ChebyshevType1, ChebyshevType2, ChebyshevType3, ChebyshevType4 ########################################
#########################             Legendre, Gegenbauer, Jacobi                       ########################################
#################################################################################################################################

class DiscreteDiagSheafDiffusionPolynomial(PolynomialSheafDiffusion):
    """Discrete Sheaf Diffusion with diagonal maps + configurable polynomial spectral filter.

    polynomial_type ∈ {
        'Chebyshev', 'ChebyshevType1', 'ChebyshevType2', 'ChebyshevType3', 'ChebyshevType4',
        'Legendre', 'Gegenbauer', 'Jacobi'
    }

    - Chebyshev / ChebyshevType1: T_k (1st kind)
    - ChebyshevType2: U_k  (2nd kind)
    - ChebyshevType3: V_k  (3rd kind)
    - ChebyshevType4: W_k  (4th kind)
    - Legendre:       P_k
    - Gegenbauer:     C_k^{(λ)} with λ>0 (args['gegenbauer_lambda'])
    - Jacobi:         P_k^{(α,β)} with α,β>-1 (args['jacobi_alpha'], args['jacobi_beta'])
    """

    def __init__(self, edge_index, args):
        super().__init__(edge_index, args)
        assert args['d'] > 0


        # If the Sheaf Laplacian is Normalised, since spectrum is bounded in [0,2], set it =2.
        if self.normalised:
            self.lambda_max = 2.0
        # If the Sheaf Laplacian is not Normalised. 
        else:
            # Build initial trivial maps to estimate degrees / power-iterate.
            trivial_maps = torch.ones((edge_index.shape[1], self.d), device=self.device)
            L, _ = lb.DiagLaplacianBuilder(
                self.graph_size, edge_index, d=self.d,
                normalised=self.normalised,
                deg_normalised=self.deg_normalised,
                add_hp=self.add_hp, add_lp=self.add_lp
            )(trivial_maps)

            (idx_i, idx_j), vals = L
            # If the Choice is analytic, set it using Gershgorin's Theorem.
            if self.lambda_max_choice == 'analytic':
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg  = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
                print("Analytic bound for λ_max:", self.lambda_max)
            # If the Choice is Iterative, set it using Rayleight Iteration.
            else:
                N = self.graph_size * self.final_d
                torch.manual_seed(0)
                self.lambda_max = self.estimate_largest_eig((idx_i, idx_j), vals, N)
                print(f"Estimated largest eigenvalue λ_max: {self.lambda_max}")

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights  = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight)
            self.lin_right_weights.append(r)
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight)
            self.lin_left_weights.append(l)

        # ---- Sheaf Learners (Diag) ----
        self.sheaf_learners = nn.ModuleList()
        num_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(self.final_d,
                                                   self.hidden_channels,
                                                   out_shape=(self.d,),
                                                   sheaf_act=self.sheaf_act)
                )
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(self.hidden_dim,
                                            out_shape=(self.d,),
                                            sheaf_act=self.sheaf_act)
                )

        # ---- Laplacian Builder (Diag) ----
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp
        )

        # ---- Residual Epsilons ----
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.final_d, 1)),
                         requires_grad=args['use_epsilons'])
            for _ in range(self.layers)
        ])

        # ---- Embedding/Projection ----
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        # ---- High-Pass Skip for contrasting Oversmoothing Bias ----
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # 1) Embedding + Dropout + Act.
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        # 2) Optional Second Linear + Sheaf-dropout -> Flatten.
        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        # 3) Diffusion Layers.
        x0, maps = x, None
        for layer in range(self.layers):

            # (Re)learn diag maps and build Laplacian.
            if layer == 0 or self.nonlinear:
                xm = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                xm = xm.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                maps = (learner(xm, self.edge_index, maps)
                        if isinstance(learner, RotationInvariantSheafLearner)
                        else learner(xm, self.edge_index))
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
                idx, vals = L

            # ---- Polynomial Spectral Filtering p(L) x ----
            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)  # for HP skip
            x_poly = self._poly_eval(idx, vals, x)

            # 4) High-pass Reinjection STep.
            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp

            # 5) Residual + Nonlinearity.
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_act:
                x = F.elu(x)
            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x = coeff * x0 - x
            x0 = x

        # 6) Projection to output.
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)














class DiscreteBundleSheafDiffusionPolynomial(PolynomialSheafDiffusion):
    """Discrete Sheaf Diffusion with bundle maps + configurable polynomial spectral filter + HP skip."""

    def __init__(self, edge_index, args, K=15):
        super().__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        
        if self.normalised:
            self.lambda_max = 2.0
        else:
            if self.lambda_max_choice == 'analytic':
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg  = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
            else:
                E = edge_index.shape[1]
                triv_maps = torch.eye(self.d, device=self.device).unsqueeze(0).expand(E, self.d, self.d)
                tmp_builder = lb.NormConnectionLaplacianBuilder(
                    self.graph_size, edge_index, d=self.d,
                    add_hp=self.add_hp, add_lp=self.add_lp, orth_map=self.orth_trans
                )
                (idx, vals), _ = tmp_builder(triv_maps)
                Nd = self.graph_size * self.final_d
                self.lambda_max = self.estimate_largest_eig(idx, vals, Nd)

        # ---- Linear Maps ----
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights  = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight.data); self.lin_right_weights.append(r)
        for _ in range(self.layers):
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight.data); self.lin_left_weights.append(l)

        # ---- Sheaf learners / Edge weights ----
        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(self.final_d, self.hidden_channels,
                                                   out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act)
                )
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(self.hidden_dim,
                                            out_shape=(self.get_param_size(),),
                                            sheaf_act=self.sheaf_act)
                )
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))

        # ---- Laplacian Builder ----
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans
        )

        # ---- Residual Epsilons, Embed/Proj, Polynomial Mix ----
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons'])
            for _ in range(self.layers)
        ])
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

    def get_param_size(self):
        # Match the Builder’s Expectation:
        # - matrix_exp / cayley: allow skew + diag  -> d(d+1)/2
        # - others (e.g., householder): skew only    -> d(d-1)/2
        if self.orth_trans in ('matrix_exp', 'cayley'):
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()
        if self.right_weights:
            x = right(x)
        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for w in self.weight_learners:
            w.update_edge_index(edge_index)

    def _prepare_maps_for_builder(self, maps: torch.Tensor) -> torch.Tensor:
        """Ensure correct per-edge parameter count for the chosen orth_map."""
        E = self.edge_index.size(1)
        expect_full = self.orth_trans in ('matrix_exp', 'cayley')
        P_full  = self.d * (self.d + 1) // 2
        P_skew  = self.d * (self.d - 1) // 2
        P_skew1 = P_skew + 1  # (Skew + Single diag scalar), optional convenience.

        if maps.dim() == 1:
            maps = maps.unsqueeze(1)
        maps = maps.contiguous().view(E, -1)

        if expect_full:
            if maps.size(1) == P_full:
                return maps
            if maps.size(1) == P_skew1:
                skew = maps[:, :P_skew]
                diag_scalar = maps[:, -1:].expand(-1, self.d)  # (E, d)
                return torch.cat([skew, diag_scalar], dim=1)
            raise RuntimeError(f"Expected {P_full} (or {P_skew1}) params/edge for '{self.orth_trans}', got {maps.size(1)}")
        else:
            if maps.size(1) == P_skew:
                return maps
            if maps.size(1) == P_full:
                return maps[:, :P_skew]
            raise RuntimeError(f"Expected {P_skew} params/edge for '{self.orth_trans}', got {maps.size(1)}")


    def forward(self, x):
        # Embedding.
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None

        for layer in range(self.layers):
            # (Re)learn maps.
            if layer == 0 or self.nonlinear:
                xm = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                xm = xm.reshape(self.graph_size, -1)

                learner = self.sheaf_learners[layer]
                maps = learner(xm, self.edge_index) if not isinstance(learner, RotationInvariantSheafLearner) \
                       else learner(xm, self.edge_index, None)
                maps = self._prepare_maps_for_builder(maps)

                # Edge weights -> (E,1).
                E = self.edge_index.size(1)
                if self.use_edge_weights:
                    ew = self.weight_learners[layer](xm, self.edge_index)
                    if ew.dim() == 1: ew = ew.unsqueeze(1)
                else:
                    ew = xm.new_ones(E, 1)

                L, trans_maps = self.laplacian_builder(maps, ew)
                self.sheaf_learners[layer].set_L(trans_maps)
                idx, vals = L

            # Linear sandwich.
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Polynomial Filter p(L) x .
            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)  # For HP skip.
            x_poly = self._poly_eval(idx, vals, x)

            # High-pass skip + Activation + Residual.
            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp
            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)















class DiscreteGeneralSheafDiffusionPolynomial(PolynomialSheafDiffusion):
    """Discrete Sheaf Diffusion with general maps + configurable polynomial spectral filter + HP skip."""
    
    def __init__(self, edge_index, args, K=15):
        super().__init__(edge_index, args)
        assert args['d'] > 1

        # ---- Polynomial Config / λ_max ----
        self.polynomial_type = str(args.get('polynomial_type', 'ChebyshevType1'))
        if self.polynomial_type.lower() == 'chebyshev':
            self.polynomial_type = 'ChebyshevType1'

        self.K = int(args.get('poly_layers_K', args.get('chebyshev_layers_K', K)))
        self.gc_lambda = float(args.get('gegenbauer_lambda', 1.0))  # > 0
        self.jac_alpha = float(args.get('jacobi_alpha', 0.0))       # > -1
        self.jac_beta  = float(args.get('jacobi_beta', 0.0))        # > -1
        self._eps = 1e-8

        self.lambda_max_choice = args.get('lambda_max_choice', 'analytic')
        assert self.lambda_max_choice in ('analytic', 'iterative', None)

        if self.normalised:
            self.lambda_max = 2.0
        else:
            if self.lambda_max_choice == 'analytic':
                ones = torch.ones(edge_index.size(1), device=self.device)
                deg  = torch.zeros(self.graph_size, device=self.device)
                deg.scatter_add_(0, edge_index[0], ones)
                self.lambda_max = 2.0 * deg.max().item()
            else:
                E = edge_index.shape[1]
                triv_maps = torch.eye(self.d, device=self.device).unsqueeze(0).expand(E, self.d, self.d)
                tmp_builder = lb.GeneralLaplacianBuilder(
                    self.graph_size, edge_index, d=self.d,
                    add_lp=self.add_lp, add_hp=self.add_hp,
                    normalised=self.normalised, deg_normalised=self.deg_normalised
                )
                (idx, vals), _ = tmp_builder(triv_maps)
                Nd = self.graph_size * self.final_d
                self.lambda_max = self.estimate_largest_eig(idx, vals, Nd)

        # ---- Linear Maps ----
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights  = nn.ModuleList()
        for _ in range(self.layers):
            r = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(r.weight.data)
            self.lin_right_weights.append(r)
        for _ in range(self.layers):
            l = nn.Linear(self.final_d, self.final_d, bias=False)
            nn.init.eye_(l.weight.data)
            self.lin_left_weights.append(l)

        # ---- Sheaf Learners (full d×d maps) ----
        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()  
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for _ in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(
                    LocalConcatSheafLearnerVariant(self.final_d, self.hidden_channels,
                                                   out_shape=(self.d, self.d), sheaf_act=self.sheaf_act)
                )
            else:
                self.sheaf_learners.append(
                    LocalConcatSheafLearner(self.hidden_dim,
                                            out_shape=(self.d, self.d),
                                            sheaf_act=self.sheaf_act)
                )

        # ---- Laplacian Builder ----
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised
        )

        # ---- Residual, Embed/Proj, Polynomial Mix ----
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.zeros((self.final_d, 1)), requires_grad=args['use_epsilons'])
            for _ in range(self.layers)
        ])
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.poly_logits = nn.Parameter(torch.zeros(self.K + 1))  # Softmax Mixture
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

        if self.polynomial_type == 'Gegenbauer' and not (self.gc_lambda > 0.0):
            warnings.warn("gegenbauer_lambda must be > 0; clamping to 0.1")
            self.gc_lambda = max(0.1, self.gc_lambda)
        if self.polynomial_type == 'Jacobi' and not (self.jac_alpha > -1.0 and self.jac_beta > -1.0):
            warnings.warn("Jacobi requires alpha,beta > -1; clamping to -0.9")
            self.jac_alpha = max(self.jac_alpha, -0.9)
            self.jac_beta  = max(self.jac_beta,  -0.9)


    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()
        if self.right_weights:
            x = right(x)
        return x

    def forward(self, x):
        # Embedding
        if self.use_embedding:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act:
                x = F.elu(x)

        x = F.dropout(x, p=self.dropout if self.second_linear else self.sheaf_dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None

        for layer in range(self.layers):
            # (Re)learn Maps
            if layer == 0 or self.nonlinear:
                xm = F.dropout(x, p=self.sheaf_dropout if layer > 0 else 0., training=self.training)
                xm = xm.reshape(self.graph_size, -1)
                learner = self.sheaf_learners[layer]
                maps = learner(xm, self.edge_index) if not isinstance(learner, RotationInvariantSheafLearner) \
                       else learner(xm, self.edge_index, None)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
                idx, vals = L

            # Linear Sandwich
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Polynomial Filter p(L) x
            T0 = x
            Lx0 = self._apply_L(idx, vals, T0)  # for HP skip
            x_poly = self._poly_eval(idx, vals, x)

            # High-pass Skip + Activation + Residual
            hp = x0 - (1.0 / self.lambda_max) * Lx0
            x = x_poly + self.hp_alpha * hp
            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return x if getattr(self, "task", None) == "regression" else F.log_softmax(x, dim=1)
