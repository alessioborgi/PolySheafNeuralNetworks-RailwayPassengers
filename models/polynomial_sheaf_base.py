import torch
import torch.nn as nn
import torch_sparse
import torch.nn.functional as F
import warnings
from models.sheaf_base import  SheafDiffusion

class PolynomialSheafDiffusion(SheafDiffusion):
    """
    Inherits from SheafDiffusion and implements polynomial filters.
    Provides poly_eval, _apply_L and _apply_Lhat. That implement various 
    polynomial bases:
    - Chebyshev Type 1,2,3,4
    - Chebyshev Interpolation
    - Legendre
    - Gegenbauer
    - Jacobi

    Further implements estimation of largest eigenvalue via power iteration.

    What is remained to implement in child classes, depending on sheaf type::
    self.lambda_max computation depending on sheaf laplacian type.
    linear maps, sheaf learners, laplacian builders, residual, epsilons, forward,

    """
    def __init__(self, edge_index, args, K=15):
        super().__init__(edge_index, args)

        # ---- Polynomial Configuration ----
        self.polynomial_type = str(args.get('polynomial_type', 'ChebyshevType1'))
        # Treating 'Chebyshev'(with no indication w.r.t. the type) as alias for first kind (T_k).
        if self.polynomial_type.lower() == 'chebyshev':
            self.polynomial_type = 'ChebyshevType1'
            
        # Order K.
        self.K = int(args.get('poly_layers_K', args.get('chebyshev_layers_K', K)))

        # Parameters for each of the families.
        self.gc_lambda = float(args.get('gegenbauer_lambda', 1.0))   # > 0
        self.jac_alpha = float(args.get('jacobi_alpha', 0.0))        # > -1
        self.jac_beta  = float(args.get('jacobi_beta', 0.0))         # > -1
        self._eps = 1e-8  # Small numeric guard

        # ---- λ_max Handling: Set its value depending on the type of sheaf laplacian we use. ----
        self.lambda_max_choice = args.get('lambda_max_choice', 'analytic')
        assert self.lambda_max_choice in ('analytic', 'iterative', None)

        # ---- Polynomial Coefficients (Convex combo, like  Chebyshev) ----
        self.poly_logits = nn.Parameter(torch.zeros(self.K + 1))
        assert self.poly_logits.numel() == self.K + 1

        # Sanity checks for parameters.
        if self.polynomial_type == 'Gegenbauer':
            if not (self.gc_lambda > 0.0):
                warnings.warn("gegenbauer_lambda must be > 0; clamping to 0.1")
                self.gc_lambda = max(0.1, self.gc_lambda)
        if self.polynomial_type == 'Jacobi':
            if not (self.jac_alpha > -1.0 and self.jac_beta > -1.0):
                warnings.warn("Jacobi requires alpha,beta > -1; clamping to -0.9")
                self.jac_alpha = max(self.jac_alpha, -0.9)
                self.jac_beta  = max(self.jac_beta,  -0.9)

        # ---------- Utilities: L@v and Lhat@v ----------
    def _apply_L(self, idx, vals, v):
        return torch_sparse.spmm(idx, vals, v.size(0), v.size(0), v)

    def _apply_Lhat(self, idx, vals, v):
        # scale L to [-1,1]: Lhat = (2/λ_max) * L - I
        Lv = self._apply_L(idx, vals, v)
        return (2.0 / self.lambda_max) * Lv - v

    def estimate_largest_eig(index_pair, vals, N, num_iter=10):
        """
        Approximate the largest eigenvalue λ_max of the sparse Laplacian defined by (index_pair, vals)
        via power iteration.
        - index_pair: a tuple (idx_i, idx_j) of 1-D long tensors of length nnz
        - vals:        a 1-D tensor of length nnz
        - N:           the dimension of the square matrix (i.e. number of rows/cols)
        """
        if isinstance(index_pair, tuple):  # (row_idx, col_idx)
            index_pair = torch.stack(index_pair, dim=0)
        # initialize with a random vector of shape (N,1)
        x = torch.randn((N, 1), device=vals.device)
        for _ in range(num_iter):
            x = torch_sparse.spmm(index_pair, vals, N, N, x)
            x = x / (x.norm() + 1e-6)
        y = torch_sparse.spmm(index_pair, vals, N, N, x)
        # Rayleigh quotient
        return ((x * y).sum() / (x * x).sum()).item()
    
    # ---------- Polynomial Evaluation on a vector x ----------
    def _poly_eval(self, idx, vals, x):
            """Return p(L) x using the chosen polynomial basis and learned coefficients."""
            poly = self.polynomial_type
            K = self.K
            w = F.softmax(self.poly_logits, dim=0) if poly != "ChebyshevInterpolation" else self.poly_logits # Convex mixture.
            
            # Helpers for consistent accumulation.
            def add(acc, k, vec):
                return acc + (w[k] * vec)

            # We need Lhat for all bases used here.
            Lhat = lambda v: self._apply_Lhat(idx, vals, v)

            

            # --- Chebyshev Family ---

            if poly in ('ChebyshevType1', 'Chebyshev'):
            # T0 = x; T1 = Lhat x; T_{k+1} = 2 Lhat T_k - T_{k-1}
                T0 = x
                out = w[0] * T0
                if K >= 1:
                    T1 = Lhat(x)
                    out = add(out, 1, T1)
                    for k in range(1, K):
                        LT1 = Lhat(T1)
                        Tk1 = 2.0 * LT1 - T0
                        out = add(out, k+1, Tk1)
                        T0, T1 = T1, Tk1
                return out

            if poly == 'ChebyshevType2':
            # U0 = x; U1 = 2 Lhat x; U_{k+1} = 2 Lhat U_k - U_{k-1}
                U0 = x
                out = w[0] * U0
                if K >= 1:
                    U1 = 2.0 * Lhat(x)
                    out = add(out, 1, U1)
                    for k in range(1, K):
                        LU1 = Lhat(U1)
                        Uk1 = 2.0 * LU1 - U0
                        out = add(out, k+1, Uk1)
                        U0, U1 = U1, Uk1
                return out

            if poly == 'ChebyshevType3':
                # V0 = x; V1 = 2 Lhat x - x; V_{k+1} = 2 Lhat V_k - V_{k-1}
                V0 = x
                out = w[0] * V0
                if K >= 1:
                    V1 = 2.0 * Lhat(x) - x
                    out = add(out, 1, V1)
                    for k in range(1, K):
                        LV1 = Lhat(V1)
                        Vk1 = 2.0 * LV1 - V0
                        out = add(out, k+1, Vk1)
                        V0, V1 = V1, Vk1
                return out

            if poly == 'ChebyshevType4':
                # W0 = x; W1 = 2 Lhat x + x; W_{k+1} = 2 Lhat W_k - W_{k-1}
                W0 = x
                W1 = (2.0 * Lhat(x) + x) if K >= 1 else None
                out = w[0] * W0
                if K >= 1: out = add(out, 1, W1)
                for k in range(1, K):
                    LW1 = Lhat(W1)
                    Wk1 = 2.0 * LW1 - W0
                    out = add(out, k+1, Wk1)
                    W0, W1 = W1, Wk1
                return out
            
            # ---- Interpolation --- #
            if poly == 'ChebyshevInterpolation':                 
                # T0 = x; T1 = Lhat x; T_{k+1} = 2 Lhat T_k - T_{k-1}
                j_values = torch.arange(K + 1, device=self.device, dtype=torch.float32)
                x_j = torch.cos((j_values + 0.5) * torch.pi / (K + 1))

                
                # Precompute all T_k(x_j) values efficiently using recurrence
                # T_k_xj[k, j] = T_k(x_j)
                T_k_xj = torch.zeros((K + 1, K + 1), device=self.device, dtype=torch.float32)
                T_k_xj[0, :] = 1.0
                if K >= 1:
                    T_k_xj[1, :] = x_j  # T_1(x_j) = x_j
                
                for k in range(2, K + 1):
                    T_k_xj[k, :] = 2 * x_j * T_k_xj[k - 1, :] - T_k_xj[k - 2, :]
                
                # Generate T_k(Lhat) using recurrence relation
                T = [None] * (K + 1)
                T[0] = x
                if K >= 1:
                    T[1] = Lhat(x)
                
                for k in range(2, K + 1):
                    T[k] = 2.0 * Lhat(T[k - 1]) - T[k - 2]
                
                # Compute sum_{k=0}^K T_k(Lhat) * sum_{j=0}^K gamma_j * T_k(x_j)
                out = 0.0
                for k in range(K + 1):
                    # Inner sum: sum_{j=0}^K g_j * T_k(x_j)
                    coeff = torch.dot(w, T_k_xj[k, :])
                    out = out + coeff * T[k]
                
                # Apply f0(X) and scaling factor 2/(K+1)
                return (2.0 / (K + 1)) * out

            # --- Legendre ---
            if poly == 'Legendre':
                # P0 = x; P1 = Lhat x
                P0 = x
                P1 = Lhat(x) if K >= 1 else None
                out = w[0] * P0
                if K >= 1: out = add(out, 1, P1)
                for k in range(1, K):
                    ak = (2.0 * k + 1.0) / (k + 1.0)
                    ck = k / (k + 1.0)
                    LP1 = Lhat(P1)
                    Pk1 = ak * LP1 - ck * P0
                    out = add(out, k+1, Pk1)
                    P0, P1 = P1, Pk1
                return out

            # --- Gegenbauer (λ>0) ---
            if poly == 'Gegenbauer':
                lam = max(self.gc_lambda, 1e-3)
                C0 = x
                C1 = (2.0 * lam) * Lhat(x) if K >= 1 else None
                out = w[0] * C0
                if K >= 1: out = add(out, 1, C1)
                for k in range(1, K):
                    ak = 2.0 * (k + lam) / (k + 1.0)
                    ck = (k + 2.0 * lam - 1.0) / (k + 1.0)
                    LC1 = Lhat(C1)
                    Ck1 = ak * LC1 - ck * C0
                    out = add(out, k+1, Ck1)
                    C0, C1 = C1, Ck1
                return out

            # --- Jacobi (α,β>-1) ---
            if poly == 'Jacobi':
                a = self.jac_alpha
                b = self.jac_beta
                # P0 = x
                P0 = x
                out = w[0] * P0
                if K >= 1:
                    # P1 = c1 * Lhat(P0) + c0 * P0
                    den = (a + b + 2.0)
                    c1 = den / 2.0
                    c0 = (a - b) / (den + 0.0)
                    P1 = c1 * Lhat(P0) + c0 * P0
                    out = add(out, 1, P1)
                    for k in range(1, K):
                        # P_{k+1} = (A_k * Lhat + B_k) P_k - C_k P_{k-1}
                        den1 = 2.0 * k + a + b
                        den2 = den1 + 2.0
                        # A_k
                        Ak = 2.0 * (k + 1.0) * (k + a + b + 1.0) / ((den1 + 1.0) * den2 + self._eps)
                        # B_k
                        Bk = ((b * b - a * a) /
                            (den1 * den2 + self._eps))
                        # C_k
                        Ck = 2.0 * (k + a) * (k + b) / (den1 * (den1 + 1.0) + self._eps)

                        LP1 = Lhat(P1)
                        Pk1 = Ak * LP1 + Bk * P1 - Ck * P0
                        out = add(out, k+1, Pk1)
                        P0, P1 = P1, Pk1
                return out

            raise ValueError(f"Unknown polynomial_type: {self.polynomial_type}")
