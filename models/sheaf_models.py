# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np
import torch_sparse

from typing import Tuple
from abc import abstractmethod
from torch import nn
from lib import laplace as lap
from models import laplacian_builders as lb


class SheafLearner(nn.Module):
    """Base model that learns a sheaf from the features and the graph structure."""

    def __init__(self):
        super(SheafLearner, self).__init__()
        self.L = None

    @abstractmethod         # Must be overridden by subclasses.
    def forward(self, x, edge_index, **kwargs):
        """kwargs may contain:
           - pos: (N, n) node coordinates (used only through E(n)-invariants)
           - node_scalars: (N, S) gauge-invariant scalars
           - maps, diff_strength, ... (kept for opinion dynamics variants)
        """
        raise NotImplementedError()

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """ 
        Learns per-edge sheaf maps by concatenating the features of both nodes in an edge, then passing 
        through a linear layer (optionally with activation).
    """

    def __init__(self, in_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearner, self).__init__()

        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.linear1 = torch.nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False)

        # Set the activation function. 
        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index, **kwargs):

        # For each edge, gathers the features of the two nodes, concatenates them, passes through the 
        # linear layer and activation and reshapes the output into the required shape for each edge.
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        x_cat = torch.cat([x_row,x_col],dim=1)
        maps = self.linear1(x_cat)
        maps = self.act(maps)

        # sign = maps.sign()
        # maps = maps.abs().clamp(0.05, 1.0) * sign

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class LocalConcatSheafLearnerVariant(SheafLearner):
    """ Variant that can handle an additional dimension and internal reshaping. """

    def __init__(self, d: int, hidden_channels: int, out_shape: Tuple[int, ...], sheaf_act="tanh"):
        super(LocalConcatSheafLearnerVariant, self).__init__()

        # Like the previous class but expects node features of shape [num_nodes, d, hidden_channels].
        # Linear layer will output the desired sheaf map shape per edge.
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels*2, int(np.prod(out_shape)), bias=False)
        # self.linear2 = torch.nn.Linear(self.d, 1, bias=False)

        # std1 = 1.414 * math.sqrt(2. / (hidden_channels * 2 + 1))
        # std2 = 1.414 * math.sqrt(2. / (d + 1))
        #
        # nn.init.normal_(self.linear1.weight, 0.0, std1)
        # nn.init.normal_(self.linear2.weight, 0.0, std2)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index, **kwargs):

        # For each edge, gathers and concatenates the features of its endpoints, reshapes and sums 
        # over dimension d, applies the linear and activation layers, then returns the result in the 
        # required shape.
        row, col = edge_index

        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.reshape(-1, self.d, self.hidden_channels*2).sum(dim=1)

        x_cat = self.linear1(x_cat)

        # x_cat = x_cat.t().reshape(-1, self.d)
        # x_cat = self.linear2(x_cat)
        # x_cat = x_cat.reshape(-1, edge_index.size(1)).t()

        maps = self.act(x_cat)

        if len(self.out_shape) == 2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])




class AttentionSheafLearner(SheafLearner):
    '''
        Sheaf Learner that uses attention to compute the sheaf maps. 
        For each edge, it will learn a d x d matrix using concatenated node features and a linear layer. 

        - Why attention?
            Here, for each edge, you are producing a learned weight matrix via softmax—each row acts like an 
            attention distribution over the output dimension.
            So, this is a generalization of attention: Instead of scalar attention weights (like in GAT), 
            you have a whole d x d matrix where each row is a softmax over d possible directions/values. 
            The matrix is "row-stochastic," i.e., each row sums to 1—just like attention distributions.
    '''

    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels*2, d**2, bias=False)

    def forward(self, x, edge_index, **kwargs):

        # For each edge, concatenates the features of both nodes, passed though a linear layer, reshapes to [num_edges, d, d],
        # applies softmax row-wise to produce stochastic matrices, and returns the difference between identity and the softmax 
        # results: I - softmax(M), where M is the learned matrix for each edge. 
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        return id - torch.softmax(maps, dim=-1)



class EdgeWeightLearner(SheafLearner):
    """
        For each edge, learns a scalar edge weight from concatenated node features via a linear layer and sigmoid.
        It stores the "full" left-right edge indices for edge-to-edge computations.
    """

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()

        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels*2, 1, bias=False)
        # Compute the full left-right index mapping for edge-to-edge computations.
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

    def forward(self, x, edge_index, **kwargs):

        # For each edge, learns a weight from concatenated node features (applies sigmoid to bound between 0 and 1).
        # For each edge, multiplies the weight by the weight of the "partner" edge (possibly reverse direction or 
        # corresponding undirected edge). 
        # Return then the combined edge weights.
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        ''' Updates the full left-right index mapping based on a new edge_index.'''
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)


class QuadraticFormSheafLearner(SheafLearner):
    """
        Learns m = prod(out_shape) quadratic forms on concatenated endpoint features.
        z_e = [x_u || x_v] in R^{2*in_channels}, output q_m(e) = z_e^T M_m z_e.
    """
    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super().__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        D2 = 2 * in_channels
        M0 = torch.eye(D2).unsqueeze(0)                           # (1, 2D, 2D)
        self.tensor = nn.Parameter(M0.repeat(int(np.prod(out_shape)), 1, 1))  # (m, 2D, 2D)

    def forward(self, x, edge_index, **_):
        row, col = edge_index
        x_row = torch.index_select(x, 0, row)
        x_col = torch.index_select(x, 0, col)
        z = torch.cat([x_row, x_col], dim=1)                      # (E, 2D)
        q = torch.einsum('ei,mij,ej->em', z, self.tensor, z)      # (E, m)
        q = torch.tanh(q)
        if len(self.out_shape) == 2:
            return q.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return q.view(-1, self.out_shape[0])


class RotationInvariantSheafLearner(SheafLearner):
    def __init__(self, d: int, hidden_channels: int, edge_index, graph_size, out_shape: Tuple[int, ...], time_dep: bool, transform = None, 
                 sheaf_act="tanh"):
        super(RotationInvariantSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape
        self.d = d
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(d*d, int(np.prod(out_shape)), bias=True)

        self.time_dep = time_dep
        self.left_right_idx, _ = lap.compute_left_right_map_index(edge_index)
        right_left_idx = torch.cat((self.left_right_idx[1].reshape(1,-1),self.left_right_idx[0].reshape(1,-1)),0)
        sheaf_edge_index_unsorted = torch.cat((self.left_right_idx,right_left_idx),1)
        self.graph_size = graph_size
        self.dual_laplacian_builder = lb.GeneralLaplacianBuilder(
            edge_index.shape[1],sheaf_edge_index_unsorted, d = self.d,
            normalised=False,deg_normalised = True, augmented=False)
        
        self.transform = transform

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu

    def forward(self, x, edge_index, Maps, **kwargs):

        if Maps == None or not self.time_dep:
            Maps2 = torch.eye(self.d).reshape(1,self.d,self.d).repeat(edge_index.shape[1],1,1).to(x.device)
        elif self.transform is not None:
            if self.transform == torch.diag:
                Maps2 = torch.stack([self.transform(map1) for map1 in Maps])
            else:
                Maps2 = self.transform(Maps)
        else:
            Maps2 = Maps
        xT = torch.index_select(torch.transpose(x.reshape(self.graph_size,-1,self.hidden_channels)[:,0:self.d,:], -2, -1), 
                                dim = 0, index = edge_index[0])
        OldMaps = torch.transpose(Maps2,-1,-2).reshape(-1,self.d)
        
        xTmaps = torch.cat((torch.index_select(xT,0,self.left_right_idx[0]),
                            torch.index_select(xT,0,self.left_right_idx[1])), 0)
        Lsheaf, _ = self.dual_laplacian_builder(xTmaps)
        node_edge_sims = 2*torch.transpose(torch_sparse.spmm(Lsheaf[0], Lsheaf[1], OldMaps.size(0), OldMaps.size(0), OldMaps).reshape((-1,self.d,self.d)),-2,-1)

        node_edge_sims = self.linear1(node_edge_sims.reshape(-1,self.d*self.d))
        maps = self.act(node_edge_sims)
        if len(self.out_shape) == 2:
            maps = maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            maps = maps.view(-1, self.out_shape[0])
        return maps    