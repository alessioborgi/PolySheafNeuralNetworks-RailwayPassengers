# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from torch import nn
from torch_householder import torch_householder_orgqr


class Orthogonal(nn.Module):
    """
        Based on https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#orthogonal
        This class implements various orthogonal transformations (rotation or reflection) for a given dimension `d`.
    """
    def __init__(self, d, orthogonal_map):
        super().__init__()
        # Checking whether the orthogonal map is one of the valid ones.
        assert orthogonal_map in ["matrix_exp", "cayley", "householder", "euler"]
        self.d = d
        self.orthogonal_map = orthogonal_map

    def get_2d_rotation(self, params):
        ''' 
            Get a 2D rotation matrix from the parameters batch of size [N, 1]), 
            create a 2D rotation matrix for each.
        '''
        # assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 1
        
        # The parameters are in [-1, 1], we map them to [0, 2pi]. 
        sin = torch.sin(params * 2 * math.pi)
        cos = torch.cos(params * 2 * math.pi)
        
        # Returning the 2D(2x2) rotation matrix as a batch.
        return torch.cat([cos, -sin,
                          sin, cos], dim=1).view(-1, 2, 2)

    def get_3d_rotation(self, params):
        ''' 
            Get a 3D rotation matrix from the parameters batch of size [N, 1]), 
            create a 3D rotation matrix for each.
        '''
        assert params.min() >= -1.0 and params.max() <= 1.0
        assert params.size(-1) == 3

        # The parameters are in [-1, 1], we map them to [0, 2pi].
        alpha = params[:, 0].view(-1, 1) * 2 * math.pi
        beta = params[:, 1].view(-1, 1) * 2 * math.pi
        gamma = params[:, 2].view(-1, 1) * 2 * math.pi

        # Calculating the sine and cosine of the angles.
        sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
        sin_b, cos_b = torch.sin(beta),  torch.cos(beta)
        sin_g, cos_g = torch.sin(gamma), torch.cos(gamma)

        # Returning the 3D rotation matrix (ZYZ Euler Angles).
        return torch.cat(
            [cos_a*cos_b, cos_a*sin_b*sin_g - sin_a*cos_g, cos_a*sin_b*cos_g + sin_a*sin_g,
             sin_a*cos_b, sin_a*sin_b*sin_g + cos_a*cos_g, sin_a*sin_b*cos_g - cos_a*sin_g,
             -sin_b, cos_b*sin_g, cos_b*cos_g], dim=1).view(-1, 3, 3)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        
        # Checking as first whether the orthogonal map is different from "Euler". 
        if self.orthogonal_map != "euler":
            # For all maps except "Eurler", fill the lower triangular part of a dxd zero matrix 
            # (for each batch element) with parameters.
            
            # Offset controls which part of the lower triangle is filled. 
            offset = -1 if self.orthogonal_map == 'householder' else 0  
            # Create a mask for the lower triangular part of the matrix.
            tril_indices = torch.tril_indices(row=self.d, col=self.d, offset=offset, device=params.device)
            # If the parameters are not in the lower triangular part, we need to fill it.
            new_params = torch.zeros(
                (params.size(0), self.d, self.d), dtype=params.dtype, device=params.device)
            new_params[:, tril_indices[0], tril_indices[1]] = params
            params = new_params

        # 1) If the orthogonal map is "matrix_exp" or "cayley", we need to ensure that only 
        #    lower-triangular entries are considered. 
        if self.orthogonal_map == "matrix_exp" or self.orthogonal_map == "cayley":
            
            # We just need n x k - k(k-1)/2 parameters.
            params = params.tril()
            # A is skew-symmetric (or skew-hermitian). 
            A = params - params.transpose(-2, -1)
            
            # 1.2) If the orthogonal map is "matrix_exp" or "cayley", we need to compute the orthogonal matrix in a different way. 
            if self.orthogonal_map == "matrix_exp":
                # Q = exp(A)
                Q = torch.matrix_exp(A)
                
            elif self.orthogonal_map == "cayley":
                # Cayley retraction Q = (I+A/2)(I-A/2)^{-1}
                Id = torch.eye(self.d, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
        
        # 2) If the orthogonal map is "householder", we need to compute the orthogonal matrix differently:
        #    Makes a lower-triangular matrix (excluding diagonal) and add identity (s.t. diagonal is 1).        
        elif self.orthogonal_map == 'householder':
            
            # Create a lower triangular matrix(excluding diagonal) with the parameters.
            eye = torch.eye(self.d, device=params.device).unsqueeze(0).repeat(params.size(0), 1, 1)
            # Add the identity matrix to the lower triangular matrix.
            A = params.tril(diagonal=-1) + eye
            # Compute the orthogonal matrix using Householder transformation.
            Q = torch_householder_orgqr(A)
        
        # 3) If the orthogonal map is "euler", we need to compute the orthogonal matrix differently:
        #    It's only defined for 2D and 3D rotations.
        elif self.orthogonal_map == 'euler':
            
            assert 2 <= self.d <= 3
            # For 2D and 3D rotations, we need to compute the orthogonal matrix differently.
            if self.d == 2:
                Q = self.get_2d_rotation(params)
            else:
                Q = self.get_3d_rotation(params)
        else:
            raise ValueError(f"Unsupported transformations {self.orthogonal_map}")
        return Q