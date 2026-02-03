# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import lib.laplace as lap

from torch import nn
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from models.orthogonal import Orthogonal

# This code implements modules for building (Sheaf) Laplacians on graphs in PyTorch. 

class LaplacianBuilder(nn.Module):
    '''    Base class for building Sheaf Laplacians on graphs.'''

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False, add_hp=False, add_lp=False,
                 augmented=True):
        ''' 
            Args: - size: number of nodes in the graph.
                  - edge_index: edge list as [2 x num_edges] tensor (standard in PyTorch Geometric).
                  - d: feature dimension per node/edge (number of dimensions of the Sheaf Laplacian).
                  - normalised: flag to use scalar normalization Laplacian.
                  - deg_normalised: flag to use the degree normalised Laplacian.
                  - add_hp: whether to add a horizontal part (high-pass) fixed components to the Laplacian.
                  - add_lp: whether to add a vertical part (low-pass) fixed components to the Laplacian.
                  - augmented: whether to use the augmented Laplacian (L + I).
        '''
        super(LaplacianBuilder, self).__init__()
        assert not (normalised and deg_normalised)

        self.d = d
        self.final_d = d
        # If we add high-pass or low-pass components, we increase the dimension of the Laplacian.
        if add_hp:
            self.final_d += 1
        if add_lp:
            self.final_d += 1
        self.size = size
        # The number of edges is half of total edge count (assuming undirected graphs are stored with both directions).
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.device = edge_index.device
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.augmented = augmented

        # Precomputes the indices for Sparse Laplacian construction.
        # full_left_right_idx contains the indices for the full matrix.
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)
        self.left_right_idx, self.vertex_tril_idx = lap.compute_left_right_map_index(edge_index)
        # If LP/HP are used, also computes the indices for the left and right maps.
        if self.add_lp or self.add_hp:
            self.fixed_diag_indices, self.fixed_tril_indices = lap.compute_fixed_diag_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.final_d)
        # Computes the node degrees from the source node indices.
        self.deg = degree(self.edge_index[0], num_nodes=self.size)

    def get_fixed_maps(self, size, dtype):
        ''' Returns the fixed diagonal and non-diagonal maps for the low-pass and high-pass components. '''
        
        assert self.add_lp or self.add_hp
        
        # Builds fixed diagonal/non-diagonal vectors for inclusion in the Laplacian.
        fixed_diag, fixed_non_diag = [], []
        # Adds to both maps the Low-Pass(adds ones) and High-Pass(adds negative ones) components, if present.
        if self.add_lp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(torch.ones(size=(size, 1), device=self.device, dtype=dtype))
        if self.add_hp:
            fixed_diag.append(self.deg.view(-1, 1))
            fixed_non_diag.append(-torch.ones(size=(size, 1), device=self.device, dtype=dtype))

        # Concatenates the fixed diagonal and non-diagonal maps over the batch dimension.
        fixed_diag = torch.cat(fixed_diag, dim=1)
        fixed_non_diag = torch.cat(fixed_non_diag, dim=1)

        # Checks that indices and valus are consistent in number.
        assert self.fixed_tril_indices.size(1) == fixed_non_diag.numel()
        assert self.fixed_diag_indices.size(1) == fixed_diag.numel()

        # Returns fixed maps for the diagonal and off-diagonal parts.
        return fixed_diag, fixed_non_diag

    def scalar_normalise(self, diag, tril, row, col):
        ''' Implements the scalar normalisation of the Laplacian. (D^{-1/2} * L * D^{-1/2}) '''

        # If the Laplacian is a matrix, we need to reshape the diagonal and tril (this is the lower triangular part) matrices. 
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        # # If using "augmented" normalization, add 1 to diag before taking the inverse sqrt;
        # this is often used to avoid division by zero when the degree is zero.
        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            # Otherwise, use the true degree values, but zero out any infinities (division by zero).
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float('inf'), 0)
        
        # Reshape the inverse square root so it can be broadcast properly.
        # For matrix-valued maps, use 3D (batch_size, 1, 1), else (batch_size, d).
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if tril.dim() > 2 else diag_sqrt_inv.view(-1, d)
        # Gather the normalization coefficients for the source (row) and target (col) of each entry.
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        # Normalize the off-diagonal (lower triangular) entries using D^{-1/2} on both sides.
        non_diag_maps = left_norm * tril * right_norm
        # Reshape again if diag is matrix-valued.
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if diag.dim() > 2 else diag_sqrt_inv.view(-1, d)
        # Normalize the diagonal entries: D^{-1/2} * diag * D^{-1/2} = D^{-1} * diag for diagonal matrices.
        diag_maps = diag_sqrt_inv**2 * diag

        # Return the normalized diagonal and off-diagonal entries.
        return diag_maps, non_diag_maps

    def append_fixed_maps(self, size, diag_indices, diag_maps, tril_indices, tril_maps):
        ''' Appends fixed diagonal and non-diagonal maps to the learnable Laplacian.'''

        # If neither low-pass nor high-pass components are required, return the learnable (input) maps as-is.
        if not self.add_lp and not self.add_hp:
            return (diag_indices, diag_maps), (tril_indices, tril_maps)

        # Compute the fixed (high-pass and/or low-pass) diagonal and off-diagonal maps.
        fixed_diag, fixed_non_diag = self.get_fixed_maps(size, tril_maps.dtype)
        tril_row, tril_col = self.vertex_tril_idx

        # Normalise the fixed parts.
        if self.normalised:
            fixed_diag, fixed_non_diag = self.scalar_normalise(fixed_diag, fixed_non_diag, tril_row, tril_col)
        
        # Flatten the fixed diagonal and off-diagonal maps for concatenation.
        fixed_diag, fixed_non_diag = fixed_diag.view(-1), fixed_non_diag.view(-1)

        # Merge (concatenate) the fixed maps and their indices with the learnable maps and indices for the off-diagonal entries.
        # lap.mergesp that stacks or merges sparse indices and their values.
        tril_indices, tril_maps = lap.mergesp(self.fixed_tril_indices, fixed_non_diag, tril_indices, tril_maps)
        diag_indices, diag_maps = lap.mergesp(self.fixed_diag_indices, fixed_diag, diag_indices, diag_maps)

        # Returns all diagonal and off-diagonal indices and values.
        return (diag_indices, diag_maps), (tril_indices, tril_maps)

    # def create_with_new_edge_index(self, edge_index):
    #     ''' Creates a new LaplacianBuilder with a new edge index. '''

    #     # Check that all edge indices are valid and do not exceed the number of nodes.
    #     assert edge_index.max().item() < self.size
    #     # Create a new instance of the same class (LaplacianBuilder or any subclass)
    #     # with the new edge_index but keeping all other parameters the same.
    #     # This makes it easy to generate a Laplacian for a different graph structure 
    #     # (e.g., a masked graph or one with dropped edges during training) without 
    #     # losing configuration.
    #     new_builder = self.__class__(
    #         self.size, edge_index, self.d,
    #         normalised=self.normalised, deg_normalised=self.deg_normalised, add_hp=self.add_hp, add_lp=self.add_lp,
    #         augmented=self.augmented)
    #     new_builder.train(self.training)
    #     return new_builder

    def create_with_new_edge_index(self, edge_index: torch.Tensor):
        """
        Recreate *this* builder with a new edge_index, forwarding only the
        kwargs that the specific builder's __init__ supports.
        """
        ei = edge_index.to(getattr(self, 'edge_index', edge_index).device)
        cur_size = int(getattr(self, 'size', 0))
        max_id = (ei.max().item() + 1) if ei.numel() else 0
        new_size = max(cur_size, max_id)

        Cls = self.__class__

        # core args every builder takes
        kwargs = dict(size=new_size, edge_index=ei, d=self.d)

        # optional common flags (only if the instance actually has them)
        for k in ('add_hp', 'add_lp'):
            if hasattr(self, k):
                kwargs[k] = getattr(self, k)

        # builder-specific flags
        # Diag / General builders: normalization flags
        try:
            from .laplacian_builders import DiagLaplacianBuilder, GeneralLaplacianBuilder, \
                                            NormConnectionLaplacianBuilder, EquivariantNormConnectionLaplacianBuilder
        except Exception:
            # same module; classes should already be defined below in this file
            DiagLaplacianBuilder = GeneralLaplacianBuilder = NormConnectionLaplacianBuilder = EquivariantNormConnectionLaplacianBuilder = tuple()

        if isinstance(self, (DiagLaplacianBuilder, GeneralLaplacianBuilder)):
            if hasattr(self, 'normalised'):
                kwargs['normalised'] = self.normalised
            if hasattr(self, 'deg_normalised'):
                kwargs['deg_normalised'] = self.deg_normalised

        # Connection builders: orth_map (+ optionally augmented if ctor supports it)
        if isinstance(self, (NormConnectionLaplacianBuilder, EquivariantNormConnectionLaplacianBuilder)):
            if hasattr(self, 'orth_map'):
                kwargs['orth_map'] = self.orth_map
            # many connection builders also take `augmented` (default True)
            if 'augmented' in Cls.__init__.__code__.co_varnames and hasattr(self, 'augmented'):
                kwargs['augmented'] = self.augmented

        return Cls(**kwargs)



class DiagLaplacianBuilder(LaplacianBuilder):
    """
        Learns a a Sheaf Laplacian with diagonal restriction maps.
        This class specializes the LaplacianBuilder for the case where only diagonal (scalar) restriction 
        maps are learned for each node/edge (i.e., the Laplacian is diagonal in the feature space).
    
    """

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False, add_hp=False, add_lp=False,
                 augmented=True):
        super(DiagLaplacianBuilder, self).__init__(
            size, edge_index, d, normalised, deg_normalised, add_hp, add_lp, augmented)

        # Initializes the diagonal and lower-triangular indices needed for building the sparse Laplacian.
        # These indices determine where the learnable diagonal and off-diagonal values will be inserted 
        # in the Laplacian matrix.
        self.diag_indices, self.tril_indices = lap.compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)

    def normalise(self, diag, tril, row, col):
        ''' 
            Function to normalise the diagonal and lower-triangular parts of the Laplacian.
            Here:
            - If self.normalised: The normalization is based on the actual learned map (the per-node values in diag).
                                  so the normalization factor depends on the current value of the learned Laplacian 
                                  diagonal.
            - If self.deg_normalised: The normalization is based on the graph degree (the number of edges per node) 
                                  as a fixed structural property of the graph.
            
            For both, the formula is: D^{-1/2} L D^{-1/2}. The only difference is whether D is the learned
            diagonal or the degree diagonal.
        '''

        # If self.normalised is set, performs standard Laplacian normalization. 
        # possibly augmented to avoid division by zero. 
        if self.normalised:
            d_sqrt_inv = (diag + 1).pow(-0.5) if self.augmented else diag.pow(-0.5)
            left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = d_sqrt_inv * diag * d_sqrt_inv

        # Otherwise, if self.deg_normalised is set, performs degree normalization.
        elif self.deg_normalised:
            deg_sqrt_inv = (self.deg + 1).pow(-0.5) if self.augmented else self.deg.pow(-0.5)
            deg_sqrt_inv = deg_sqrt_inv.unsqueeze(-1)

            # Mask infinities to avoid division by zero.
            deg_sqrt_inv.masked_fill_(deg_sqrt_inv == float('inf'), 0)
            left_norm, right_norm = deg_sqrt_inv[row], deg_sqrt_inv[col]
            # Apply the degree normalization to the lower-triangular and diagonal parts.
            tril = left_norm * tril * right_norm
            diag = deg_sqrt_inv * diag * deg_sqrt_inv

        # Return the normalized diagonal and lower-triangular parts.
        return diag, tril

    def forward(self, maps):
        
        # For E(n) equivariance, allow scalar α per edge and broadcast to α·I
        if maps.dim() == 1 or maps.size(1) == 1:
            maps = maps.view(-1, 1).expand(-1, self.d)
        # We expect the input maps to be of shape [num_edges, d].
        assert len(maps.size()) == 2
        assert maps.size(1) == self.d
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        row, _ = self.edge_index

        # 1) Compute the un-normalised (raw) Laplacian entries.
        # For each edge (off-diagonal), computes -f_i * f_j . 
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -left_maps * right_maps
        saved_tril_maps = tril_maps.detach().clone()
        # For the diagonal entries, computes f_i^2, for each node across its incident edges. 
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.size)

        # 2) Normalise the entries if the normalised Laplacian is used.
        # Apply normalization to the diagonal and off-diagonal entries.
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        # Flatten the values to 1D for easier sparse reresentation and retrieve the associated 
        # indices for where to insert the values. 
        tril_maps, diag_maps = tril_maps.view(-1), diag_maps.view(-1)

        # 3) Append fixed HP/LP components (diagonal values) in the non-learnable dimensions, if requested.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps)

        # 4) Add the upper triangular part. 
        # For undirected graphs, ads the upper traingular entries (mirror of the lower triangle).
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = lap.mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        # 5) Merge diagonal and non-diagonal.
        # Combines all non-diagonal (off-diagonal) and diagonal indices and values into the final list.
        # These can be used as the sparse representation of the Laplacian (edge list + weights)
        edge_index, weights = lap.mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        # 6) Return the edge index and weights, along with the saved tril maps (possibly normalized off-diagonal 
        #    maps before symmetrization, which can be useful for analysis).
        return (edge_index, weights), saved_tril_maps




class NormConnectionLaplacianBuilder(LaplacianBuilder):
    """Learns a a Sheaf Laplacian with learnable orthogonal restriction maps"""

    def __init__(self, size, edge_index, d, add_hp=False, add_lp=False, orth_map=None, augmented=True):
        super(NormConnectionLaplacianBuilder, self).__init__(
            size, edge_index, d, add_hp=add_hp, add_lp=add_lp, normalised=True, augmented=augmented)
        
        # Storing the orthogonal transformation objec, which produces orthogonal matrices from learnable parameters, 
        # as well as the map type. 
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=orth_map)
        self.orth_map = orth_map

        # Pre-compute the sparse indices (row, col) for where to insert the lower triangular (off-diagonal)
        # entries (tril_indices) and the diagonal entries (diag_indices) in the Laplacian matrix.
        _, self.tril_indices = lap.compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)
        self.diag_indices, _ = lap.compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)

    def create_with_new_edge_index(self, edge_index):
        ''' 
            Function to create a new builder instance with a different edge list but all the same parameters.
            Ensures the mode (train/eval) is the same as the current instance.
        '''

        assert edge_index.max().item() < self.size
        new_builder = self.__class__(
            self.size, edge_index, self.d, add_hp=self.add_hp, add_lp=self.add_lp, augmented=self.augmented,
            orth_map=self.orth_map)
        new_builder.train(self.training)
        return new_builder

    def normalise(self, diag, tril, row, col):
        ''' Function to normalise the diagonal and lower-triangular parts of the Laplacian.
            This is similar to the scalar normalisation but uses the orthogonal transformation.
            The formula is: D^{-1/2} L D^{-1/2}, as before, but now D is the orthogonal matrix.
        '''
        # If the Laplacian is a matrix, we need to reshape the diagonal (should be 2D) 
        # and tril (last two dimensions should be square). 
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)

        # If using "augmented" normalization, add 1 to diag before taking the inverse sqrt;
        # This is the standard symmetric normalization for Laplacians. 
        if self.augmented:
            diag_sqrt_inv = (diag + 1).pow(-0.5)
        else:
            diag_sqrt_inv = diag.pow(-0.5)
            diag_sqrt_inv.masked_fill_(diag_sqrt_inv == float('inf'), 0)
        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if tril.dim() > 2 else diag_sqrt_inv.view(-1, d)
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = diag_sqrt_inv.view(-1, 1, 1) if diag.dim() > 2 else diag_sqrt_inv.view(-1, d)
        diag_maps = diag_sqrt_inv**2 * diag

        # It returns the normalized diagonal and off-diagonal entries.
        return diag_maps, non_diag_maps

    def forward(self, map_params, edge_weights=None):
        
        # Check that edge_weights, if provided, are of the correct shape: [num_edges, 1] and that 
        # the parameter tensor has the correct shape for the orthogonal transformation.
        # Also check that the #params matches the expected for the orthogonal map.
        if edge_weights is not None:
            # accept (E,) or (E,1) and coerce to (E,1)
            if edge_weights.dim() == 1:
                edge_weights = edge_weights.view(-1, 1)
            else:
                assert edge_weights.size(1) == 1

        # Accept both proper 2-D (E, P) or a flattened 1-D vector of length E*P and reshape.
        if map_params.dim() == 1:
            E = self.edge_index.size(1)
            # Try both plausible P values and reshape if divisible.
            full_P = self.d * (self.d + 1) // 2
            skew_P = self.d * (self.d - 1) // 2
            if map_params.numel() % E == 0:
                P_guess = map_params.numel() // E
                # minimal guard to avoid silent shape bugs
                assert P_guess in (full_P, skew_P), (
                    f"Flattened map_params has per-edge size {P_guess}, "
                    f"expected {full_P} (skew+diag) or {skew_P} (skew)."
                )
                map_params = map_params.view(E, P_guess)
            else:
                raise AssertionError("Flattened map_params length must be divisible by #edges.")
        else:
            assert len(map_params.size()) == 2

        if self.orth_map in ["matrix_exp", "cayley"]:
            # For these maps we expect skew (d(d-1)/2) possibly with an additional diagonal
            # parameterization already handled inside Orthogonal. Keep the original strictness:
            assert map_params.size(1) in (
                self.d * (self.d + 1) // 2,  # skew + full diagonal
                self.d * (self.d - 1) // 2   # pure skew
            ), "Unexpected #params per edge for orthogonal map."
        else:
            assert map_params.size(1) == self.d * (self.d - 1) // 2

        # 1) Retrieve all pre-computed index arrays for mapping between edge/node indices and 
        #    Laplacian entries. 
        _, full_right_idx = self.full_left_right_idx
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # 2) Convert the parameters to orthogonal matrices.
        # Mapping the learnable parameters to actual orthogonal matrices for each edge, using 
        # the selected orthogonal map (matrix exponential, Cayley, etc.).
        maps = self.orth_transform(map_params)

        # 3) Settig up the diagonal Laplacian values. 
        # If no edge weights are provided, the diagonal maps are simply the degrees of the nodes.
        # If instead edge weights are provided, we compute the diagonal entries are the sum of 
        # squares of incident edge weights.  
        # Also, multiplies the orthogonal maps by the edge weights (applies edge scaling to transport maps).
        if edge_weights is None:
            diag_maps = self.deg.unsqueeze(-1)
        else:
            diag_maps = scatter_add(edge_weights ** 2, row, dim=0, dim_size=self.size)
            maps = maps * edge_weights.unsqueeze(-1)

        # 4) Compute the transport maps.
        # Compute the off-diagonal Laplacian values. for each edge, select the left and right restriction maps.
        # Use batched matrix multiplication to compute -R_i^T * R_j for each edge, , where R_i and R_j are the left and right restriction maps.
        # Keep a clone of these off-diagonal values for analysis. 
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(torch.transpose(left_maps, -1, -2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()

        # 5) Normalise the entries if the normalised Laplacian is used.
        # Normalizes the diagonal and off-diagonal maps using scalar normalization.
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        tril_maps = tril_maps.contiguous().view(-1)
        # BUGFIX: do NOT squeeze before expand; keep (N,1) -> (N,d) then flatten.
        diag_maps = diag_maps.view(-1, 1).expand(-1, self.d).reshape(-1)

        # 6) Append HP/LP fixed diagonal values in the non-learnable dimensions.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps)

        # 7) Add the upper triangular part. 
        # Symmetrizes the Laplacian by adding the upper triangular part.
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = lap.mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        # 8) Merge diagonal and off-diagonal entries and their indices/values into the final sparse representation.
        edge_index, weights = lap.mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        # 9) Return the full Laplacian in sparse format (the edge indices and weights), along with the saved (raw) 
        #    tril maps ( normalized off-diagonal matrix blocks ) for analysis .
        return (edge_index, weights), saved_tril_maps



class GeneralLaplacianBuilder(LaplacianBuilder):
    """ Learns a Sheaf Laplacian with general restriction maps.
        This class extends the LaplacianBuilder to handle general restriction maps, constructing a general sheaf 
        laplacian where the restriction maps are FULL (not necessarily diagonal or orthogonal), being d x d matrices 
        learned from data"""

    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False,
                 add_hp=False, add_lp=False, augmented=True):
        super(GeneralLaplacianBuilder, self).__init__(size, edge_index, d,
                                                      normalised=normalised, deg_normalised=deg_normalised,
                                                      add_hp=add_hp, add_lp=add_lp, augmented=augmented)

        # Precomputes the sparse indices required to compute the Sheaf Laplacian.
        self.diag_indices, self.tril_indices = lap.compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.final_d)

    def normalise(self, diag_maps, non_diag_maps, tril_row, tril_col):
        ''' Normalises the diagonal and non-diagonal maps of the Sheaf Laplacian. '''

        # If using the self.normalise, we do a generalization of D^{-1/2} L D^{-1/2} normalization but for matrix-valued maps.
        if self.normalised:

            # Normalise the entries if the normalised Laplacian is used.
            if self.training:
                # During training, we perturb the matrices to ensure they have different singular values.
                # Without this, the gradients of batched_sym_matrix_pow, which uses SVD are non-finite.
                eps = torch.empty(self.d, device=self.device).uniform_(-1e-3, 1e-3)
            else:
                # At test time, of course, no perturbation is needed.
                eps = torch.zeros(self.d, device=self.device)

            # If augmented, we add I (plus noise 1) to the diagonal (to ensure invertibility) before taking the inverse square root.
            to_be_inv_diag_maps = diag_maps + torch.diag(1. + eps).unsqueeze(0) if self.augmented else diag_maps
            # Compute the symmetric matrix power for each diagonal block.
            d_sqrt_inv = lap.batched_sym_matrix_pow(to_be_inv_diag_maps, -0.5).detach()
            assert torch.all(torch.isfinite(d_sqrt_inv))
            left_norm = d_sqrt_inv[tril_row]
            right_norm = d_sqrt_inv[tril_col]
            # Applies normalization to off-diagonal: D^{-1/2} * M * D^{-1/2}. Clamp in [-1, 1] to ensure numerical stability.
            non_diag_maps = (left_norm @ non_diag_maps @ right_norm).clamp(min=-1, max=1)
            # Applies normalization to diagonal: D^{-1/2} * diag * D^{-1/2}. Clamp in [-1, 1] to ensure numerical stability.
            diag_maps = (d_sqrt_inv @ diag_maps @ d_sqrt_inv).clamp(min=-1, max=1)
            assert torch.all(torch.isfinite(non_diag_maps))
            assert torch.all(torch.isfinite(diag_maps))

        # If using degree normalization, we compute normalization based on the product of the node degree and the map dimension, for each node
        elif self.deg_normalised:
            deg = (self.deg * self.d + (1 if self.augmented else 0)).pow(-0.5).view(-1, 1, 1)
            left_norm, right_norm = deg[tril_row], deg[tril_col]
            non_diag_maps = left_norm * non_diag_maps * right_norm
            diag_maps = deg * diag_maps * deg
        
        # Return diagonal and non-diagonal maps.
        return diag_maps, non_diag_maps


    def forward(self, maps):

        # 0) Fetching all relevant indices for assembling the Laplacian.
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # 1) Compute transport maps. 
        # For each edge, constructs the off-diagonal block as -R_i^T * R_j, where R_i and R_j are the left and right restriction maps
        # (i.e., the learned d x d maps at nodes i and j).
        # The diagonal instead is the sum of R_i^T * R_i over all incident edges for node i.
        assert torch.all(torch.isfinite(maps))
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(torch.transpose(left_maps, dim0=-1, dim1=-2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = torch.bmm(torch.transpose(maps, dim0=-1, dim1=-2), maps)
        diag_maps = scatter_add(diag_maps, row, dim=0, dim_size=self.size)

        # 2) Normalise the transport maps and flattens the matrices for sparse storage.
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        diag_maps, tril_maps = diag_maps.view(-1), tril_maps.view(-1)

        # 3) Appends any fixed HP/LP components (if present) to the diagonal and off-diagonal entries.
        (diag_indices, diag_maps), (tril_indices, tril_maps) = self.append_fixed_maps(
            len(left_maps), diag_indices, diag_maps, tril_indices, tril_maps)

        # 4) Add the upper triangular part. Ensure symmetry by mirroring lower triangular off-diagonal entries to upper triangle for undirected graphs. 
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = lap.mergesp(tril_indices, tril_maps, triu_indices, tril_maps)

        # 5) Combine all entries and indices into final sparse Lapplacian format. 
        edge_index, weights = lap.mergesp(non_diag_indices, non_diag_values, diag_indices, diag_maps)

        # 6) Return the edge index and weights, along with the saved tril maps (possibly normalized off-diagonal maps).
        return (edge_index, weights), saved_tril_maps




