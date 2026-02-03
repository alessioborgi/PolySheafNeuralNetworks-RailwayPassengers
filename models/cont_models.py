# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Optional

import torch
import torch.nn.functional as F
import torch_sparse

from torch import nn
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint


class LaplacianODEFunc(nn.Module):
    """Implements Laplacian-based diffusion."""

    def __init__(self,
                 d, sheaf_learner, laplacian_builder, edge_index, graph_size, hidden_channels,
                 left_weights=False,
                 right_weights=False,
                 use_act=False,
                 nonlinear=False,
                 weight_learner=None):
        """
        Args:
            L: A sparse Laplacian matrix.
        """
        super(LaplacianODEFunc, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner
        self.laplacian_builder = laplacian_builder
        self.edge_index = edge_index
        self.nonlinear = nonlinear
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        self.L = None

        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

    def update_laplacian_builder(self, laplacian_builder):
        self.edge_index = laplacian_builder.edge_index
        self.laplacian_builder = laplacian_builder

    def forward(self, t, x):
        if self.nonlinear or self.L is None:
            # Update the laplacian at each step.
            x_maps = x.view(self.graph_size, -1)
            maps = self.sheaf_learner(x_maps, self.edge_index)
            if self.weight_learner is not None:
                edge_weights = self.weight_learner(x_maps, self.edge_index)
                L, _ = self.laplacian_builder(maps, edge_weights)
            else:
                L, _ = self.laplacian_builder(maps)
            self.L = L
        else:
            # Cache the Laplacian obtained at the first layer for the rest of the integration.
            L = self.L

        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights(x)
            x = x.reshape(-1, self.graph_size * self.d).t()

        if self.right_weights:
            x = self.lin_right_weights(x)

        x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), -x)

        if self.use_act:
            x = F.elu(x)

        return x


class ODEBlock(nn.Module):
    """Module performing the ODE Integration."""

    def __init__(self, odefunc, t, opt):
        super(ODEBlock, self).__init__()
        self.t = t
        self.opt = opt
        self.odefunc = odefunc
        self.set_tol()

    def set_tol(self):
        self.atol = self.opt['tol_scale'] * 1e-7
        self.rtol = self.opt['tol_scale'] * 1e-9
        if self.opt['adjoint']:
            self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    def forward(self, x):
        if self.opt["adjoint"] and self.training:
            z = odeint_adjoint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                adjoint_method=self.opt['adjoint_method'],
                adjoint_options=dict(step_size=self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint)
        else:
            z = odeint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol)
        self.odefunc.L = None
        z = z[1]
        return z


class GraphLaplacianDiffusion(SheafDiffusion):
    """This is a diffusion model based on the weighted graph Laplacian."""

    def __init__(self, edge_index, args):
        super(GraphLaplacianDiffusion, self).__init__(edge_index, args)
        assert args['d'] == 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = EdgeWeightLearner(self.hidden_dim, edge_index)
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)

        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)
        self.sheaf_learner.update_edge_index(edge_index)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiagSheafDiffusion(SheafDiffusion):
    """Performs diffusion using a sheaf Laplacian with diagonal restriction maps."""

    def __init__(self, edge_index, args):
        super(DiagSheafDiffusion, self).__init__(edge_index, args)

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class BundleSheafDiffusion(SheafDiffusion):
    """Performs diffusion using a sheaf Laplacian with diagonal restriction maps."""

    def __init__(self, edge_index, args):
        super(BundleSheafDiffusion, self).__init__(edge_index, args)
        # Should use diagonal sheaf diffusion instead if d=1.
        assert args['d'] > 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.weight_learner = EdgeWeightLearner(self.hidden_dim, edge_index) if self.use_edge_weights else None
        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.get_param_size(),),
                                                     sheaf_act=self.sheaf_act)
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act, weight_learner=self.weight_learner)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)
        self.weight_learner.update_edge_index(edge_index)

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class GeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(GeneralSheafDiffusion, self).__init__(edge_index, args)
        # Should use diagoal diffusion if d == 1
        assert args['d'] > 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act)
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.odefunc = LaplacianODEFunc(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index, self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear, left_weights=self.left_weights, right_weights=self.right_weights,
            use_act=self.use_act)
        self.odeblock = ODEBlock(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)











############################################################################################################################################################
############################################################# POLYNOMIAL SHEAF DIFFUSION ##################################################################
############################################################################################################################################################

# ------------------------------ helpers ------------------------------

def _spmm(idx, vals, n, x):
    """Sparse matmul: (n×n) · (n×h) -> (n×h)."""
    return torch_sparse.spmm(idx, vals, n, n, x)

@torch.no_grad()
def _estimate_largest_eig(idx, vals, n, iters: int = 30, eps: float = 1e-9):
    """Power iteration for λ_max(L)."""
    device = vals.device
    v = torch.randn(n, 1, device=device)
    v = v / (v.norm() + eps)
    for _ in range(iters):
        v = _spmm(idx, vals, n, v)
        v = v / (v.norm() + eps)
    w = _spmm(idx, vals, n, v)
    lam = (v.t() @ w) / (v.t() @ v)
    return lam.abs().item()


# --------------------------- ODE RHS (polynomial) ---------------------------

class LaplacianODEFunc_Polynomial(nn.Module):
    """
    Polynomial-filter sheaf diffusion:

        dx/dt = - p(L̂) x + α_hp * ( x - (1/λ_max) * L x )

    where L̂ = (2/λ_max) L - I, and p(·) is a softmax mixture of an orthogonal
    basis up to degree K (Chebyshev {T,U,V,W}, Legendre, Gegenbauer, Jacobi).
    """

    def __init__(
        self,
        d: int,
        sheaf_learner: nn.Module,
        laplacian_builder: nn.Module,
        edge_index: torch.Tensor,
        graph_size: int,
        hidden_channels: int,
        *,
        nonlinear: bool = False,
        left_weights: bool = False,
        right_weights: bool = False,
        use_act: bool = False,
        weight_learner: Optional[nn.Module] = None,
        args: Optional[dict] = None,
        builder_kind: str = "diag",       # 'graph'|'diag'|'bundle'|'general'
        normalised: bool = False,
        deg_normalised: bool = False,
    ):
        super().__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.weight_learner = weight_learner
        self.sheaf_learner = sheaf_learner
        self.laplacian_builder = laplacian_builder
        self.edge_index = edge_index
        self.nonlinear = nonlinear
        self.graph_size = graph_size
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act

        self.builder_kind = builder_kind
        self.normalised = normalised
        self.deg_normalised = deg_normalised

        # Caches.
        self.L = None
        self._lambda_max = None

        # Optional left/right linear "sandwich".
        if self.left_weights:
            self.lin_left_weights = nn.Linear(self.d, self.d, bias=False)
            nn.init.eye_(self.lin_left_weights.weight.data)
        if self.right_weights:
            self.lin_right_weights = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
            nn.init.orthogonal_(self.lin_right_weights.weight.data)

        # -------- Polynomial Configuration --------
        args = {} if args is None else args
        self.polynomial_type = str(args.get("polynomial_type", "ChebyshevType1"))
        if self.polynomial_type.lower() == "chebyshev":
            self.polynomial_type = "ChebyshevType1"
        self.K = int(args.get("poly_layers_K", args.get("chebyshev_layers_K", 15)))

        self.gc_lambda = float(args.get("gegenbauer_lambda", 1.0))  # > 0
        self.jac_alpha = float(args.get("jacobi_alpha", 0.0))       # > -1
        self.jac_beta  = float(args.get("jacobi_beta", 0.0))        # > -1
        self._eps = 1e-8

        self.lambda_max_choice = args.get("lambda_max_choice", "analytic")
        if self.lambda_max_choice is not None:
            assert self.lambda_max_choice in ("analytic", "iterative")

        # Learnable mixture and high-pass reinjection.
        self.poly_logits = nn.Parameter(torch.zeros(self.K + 1))
        self.hp_alpha = nn.Parameter(torch.tensor(0.0))

        # Parameter sanity (match discrete).
        if self.polynomial_type == "Gegenbauer" and not (self.gc_lambda > 0.0):
            warnings.warn("gegenbauer_lambda must be > 0; clamping to 0.1")
            self.gc_lambda = max(0.1, self.gc_lambda)
        if self.polynomial_type == "Jacobi" and not (self.jac_alpha > -1.0 and self.jac_beta > -1.0):
            warnings.warn("Jacobi requires alpha,beta > -1; clamping to -0.9")
            self.jac_alpha = max(self.jac_alpha, -0.9)
            self.jac_beta  = max(self.jac_beta,  -0.9)

    # --- Wiring / Caches ---

    def update_laplacian_builder(self, laplacian_builder):
        self.edge_index = laplacian_builder.edge_index
        self.laplacian_builder = laplacian_builder
        self.L = None
        self._lambda_max = None

    def _ensure_lambda_max(self, idx, vals, n_rows: int):
        if self._lambda_max is not None:
            return
        if self.normalised:
            self._lambda_max = 2.0
            return
        if self.lambda_max_choice == "analytic" and self.builder_kind in ("diag", "graph"):
            # fast unnormalized bound (2 * max degree)
            ones = torch.ones(self.edge_index.size(1), device=vals.device)
            deg = torch.zeros(self.graph_size, device=vals.device)
            deg.scatter_add_(0, self.edge_index[0], ones)
            self._lambda_max = 2.0 * deg.max().item()
        else:
            self._lambda_max = _estimate_largest_eig(idx, vals, n_rows)

    # --- Operators ---

    def _apply_L(self, idx, vals, x):
        return _spmm(idx, vals, x.size(0), x)

    def _apply_Lhat(self, idx, vals, x):
        Lx = self._apply_L(idx, vals, x)
        return (2.0 / self._lambda_max) * Lx - x

    # --- Polynomial Evaluation p(L̂) x ---

    def _poly_eval(self, idx, vals, x):
        K = self.K
        w = F.softmax(self.poly_logits, dim=0)
        Lhat = lambda v: self._apply_Lhat(idx, vals, v)

        def add(acc, k, vec): return acc + (w[k] * vec)
        poly = self.polynomial_type

        # Chebyshev Family.
        if poly in ("ChebyshevType1", "Chebyshev"):
            T0 = x
            out = w[0] * T0
            if K >= 1:
                T1 = Lhat(x); out = add(out, 1, T1)
                for k in range(1, K):
                    LT1 = Lhat(T1); Tk1 = 2.0 * LT1 - T0
                    out = add(out, k + 1, Tk1); T0, T1 = T1, Tk1
            return out

        if poly == "ChebyshevType2":
            U0 = x; out = w[0] * U0
            if K >= 1:
                U1 = 2.0 * Lhat(x); out = add(out, 1, U1)
                for k in range(1, K):
                    LU1 = Lhat(U1); Uk1 = 2.0 * LU1 - U0
                    out = add(out, k + 1, Uk1); U0, U1 = U1, Uk1
            return out

        if poly == "ChebyshevType3":
            V0 = x; out = w[0] * V0
            if K >= 1:
                V1 = 2.0 * Lhat(x) - x; out = add(out, 1, V1)
                for k in range(1, K):
                    LV1 = Lhat(V1); Vk1 = 2.0 * LV1 - V0
                    out = add(out, k + 1, Vk1); V0, V1 = V1, Vk1
            return out

        if poly == "ChebyshevType4":
            W0 = x; out = w[0] * W0
            if K >= 1:
                W1 = 2.0 * Lhat(x) + x; out = add(out, 1, W1)
                for k in range(1, K):
                    LW1 = Lhat(W1); Wk1 = 2.0 * LW1 - W0
                    out = add(out, k + 1, Wk1); W0, W1 = W1, Wk1
            return out

        # Legendre
        if poly == "Legendre":
            P0 = x; out = w[0] * P0
            if K >= 1:
                P1 = Lhat(x); out = add(out, 1, P1)
                for k in range(1, K):
                    ak = (2.0 * k + 1.0) / (k + 1.0)
                    ck = k / (k + 1.0)
                    LP1 = Lhat(P1); Pk1 = ak * LP1 - ck * P0
                    out = add(out, k + 1, Pk1); P0, P1 = P1, Pk1
            return out

        # Gegenbauer (λ>0)
        if poly == "Gegenbauer":
            lam = max(self.gc_lambda, 1e-3)
            C0 = x; out = w[0] * C0
            if K >= 1:
                C1 = (2.0 * lam) * Lhat(x); out = add(out, 1, C1)
                for k in range(1, K):
                    ak = 2.0 * (k + lam) / (k + 1.0)
                    ck = (k + 2.0 * lam - 1.0) / (k + 1.0)
                    LC1 = Lhat(C1); Ck1 = ak * LC1 - ck * C0
                    out = add(out, k + 1, Ck1); C0, C1 = C1, Ck1
            return out

        # Jacobi (α,β>-1)
        if poly == "Jacobi":
            a, b = self.jac_alpha, self.jac_beta
            P0 = x; out = w[0] * P0
            if K >= 1:
                den = (a + b + 2.0)
                c1 = den / 2.0
                c0 = (a - b) / (den + 0.0)
                P1 = c1 * Lhat(P0) + c0 * P0
                out = add(out, 1, P1)
                for k in range(1, K):
                    den1 = 2.0 * k + a + b
                    den2 = den1 + 2.0
                    Ak = 2.0 * (k + 1.0) * (k + a + b + 1.0) / ((den1 + 1.0) * den2 + self._eps)
                    Bk = (b * b - a * a) / (den1 * den2 + self._eps)
                    Ck = 2.0 * (k + a) * (k + b) / (den1 * (den1 + 1.0) + self._eps)
                    LP1 = Lhat(P1); Pk1 = Ak * LP1 + Bk * P1 - Ck * P0
                    out = add(out, k + 1, Pk1); P0, P1 = P1, Pk1
            return out

        raise ValueError(f"Unknown polynomial_type: {self.polynomial_type}")

    # --- ODE RHS ---

    def forward(self, t, x):
        # Build or refresh L (nonlinear dynamics or first call).
        if self.nonlinear or self.L is None:
            x_maps = x.view(self.graph_size, -1)
            maps = self.sheaf_learner(x_maps, self.edge_index)
            if self.weight_learner is not None:
                edge_weights = self.weight_learner(x_maps, self.edge_index)
                L, _ = self.laplacian_builder(maps, edge_weights)
            else:
                L, _ = self.laplacian_builder(maps)
            self.L = L
            if self._lambda_max is None:
                self._ensure_lambda_max(L[0], L[1], x.size(0))
        else:
            L = self.L

        # Optional linear sandwich.
        if self.left_weights:
            x = x.t().reshape(-1, self.d)
            x = self.lin_left_weights(x)
            x = x.reshape(-1, self.graph_size * self.d).t()
        if self.right_weights:
            x = self.lin_right_weights(x)

        # Polynomial field + HP reinjection.
        idx, vals = L
        px = self._poly_eval(idx, vals, x)
        out = -px
        Lx = self._apply_L(idx, vals, x)
        hp = x - (1.0 / self._lambda_max) * Lx
        out = out + self.hp_alpha * hp

        if self.use_act:
            out = F.elu(out)
        return out


# ------------------------------ ODE Wrapper ------------------------------

class ODEBlock_Polynomial(nn.Module):
    """ODE integrator wrapper (polynomial RHS)."""

    def __init__(self, odefunc, t, opt):
        super().__init__()
        self.t = t
        self.opt = opt
        self.odefunc = odefunc
        self.set_tol()

    def set_tol(self):
        self.atol = self.opt['tol_scale'] * 1e-7
        self.rtol = self.opt['tol_scale'] * 1e-9
        if self.opt['adjoint']:
            self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
            self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    def forward(self, x):
        if self.opt["adjoint"] and self.training:
            z = odeint_adjoint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                adjoint_method=self.opt['adjoint_method'],
                adjoint_options=dict(step_size=self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol, rtol=self.rtol,
                adjoint_atol=self.atol_adjoint, adjoint_rtol=self.rtol_adjoint)
        else:
            z = odeint(
                self.odefunc, x, self.t,
                method=self.opt['int_method'],
                options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol, rtol=self.rtol)
        self.odefunc.L = None
        z = z[1]
        return z


# =============================== Models (Continuous, Polynomial) ===============================

class GraphLaplacianDiffusion_Polynomial(SheafDiffusion):
    """Weighted graph Laplacian diffusion (d=1) with polynomial spectral filter."""
    def __init__(self, edge_index, args):
        super().__init__(edge_index, args)
        assert args['d'] == 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = EdgeWeightLearner(self.hidden_dim, edge_index)
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp, add_lp=self.add_lp)

        self.odefunc = LaplacianODEFunc_Polynomial(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index,
            self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear,
            left_weights=self.left_weights, right_weights=self.right_weights, use_act=self.use_act,
            weight_learner=None,  # Edge weights come from sheaf_learner here
            args=args, builder_kind="graph", normalised=False, deg_normalised=False
        )
        self.odeblock = ODEBlock_Polynomial(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)
        self.sheaf_learner.update_edge_index(edge_index)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act: x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear: x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class DiagSheafDiffusion_Polynomial(SheafDiffusion):
    """Sheaf Laplacian diffusion with diagonal maps + polynomial spectral filter."""
    def __init__(self, edge_index, args):
        super().__init__(edge_index, args)

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            normalised=self.normalised, deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp)

        self.odefunc = LaplacianODEFunc_Polynomial(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index,
            self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear,
            left_weights=self.left_weights, right_weights=self.right_weights, use_act=self.use_act,
            weight_learner=None,
            args=args, builder_kind="diag",
            normalised=self.normalised, deg_normalised=self.deg_normalised
        )
        self.odeblock = ODEBlock_Polynomial(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act: x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear: x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class BundleSheafDiffusion_Polynomial(SheafDiffusion):
    """Sheaf diffusion with orthogonal bundle maps + polynomial spectral filter."""
    def __init__(self, edge_index, args):
        super().__init__(edge_index, args)
        assert args['d'] > 1  # prefer Diag when d == 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.weight_learner = EdgeWeightLearner(self.hidden_dim, edge_index) if self.use_edge_weights else None
        self.sheaf_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act)

        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            add_hp=self.add_hp, add_lp=self.add_lp, orth_map=self.orth_trans)

        self.odefunc = LaplacianODEFunc_Polynomial(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index,
            self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear,
            left_weights=self.left_weights, right_weights=self.right_weights, use_act=self.use_act,
            weight_learner=self.weight_learner,
            args=args, builder_kind="bundle",
            normalised=False, deg_normalised=False  # handled in builder; λ_max via iteration by default
        )
        self.odeblock = ODEBlock_Polynomial(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)
        if self.weight_learner is not None:
            self.weight_learner.update_edge_index(edge_index)

    def get_param_size(self):
        # match builder expectations (skew vs skew+diag)
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act: x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear: x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)
        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class GeneralSheafDiffusion_Polynomial(SheafDiffusion):
    """Sheaf diffusion with general d×d maps + polynomial spectral filter."""
    def __init__(self, edge_index, args):
        super().__init__(edge_index, args)
        assert args['d'] > 1

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.sheaf_learner = LocalConcatSheafLearner(
            self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act)

        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.odefunc = LaplacianODEFunc_Polynomial(
            self.final_d, self.sheaf_learner, self.laplacian_builder, edge_index,
            self.graph_size, self.hidden_channels,
            nonlinear=self.nonlinear,
            left_weights=self.left_weights, right_weights=self.right_weights, use_act=self.use_act,
            weight_learner=None,
            args=args, builder_kind="general",
            normalised=self.normalised, deg_normalised=self.deg_normalised
        )
        self.odeblock = ODEBlock_Polynomial(self.odefunc, self.time_range, args)

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        self.odefunc.update_laplacian_builder(self.laplacian_builder)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act: x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear: x = self.lin12(x)

        if self.t > 0:
            x = x.view(self.graph_size * self.final_d, -1)
            x = self.odeblock(x)

        assert torch.all(torch.isfinite(x))

        x = x.view(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
