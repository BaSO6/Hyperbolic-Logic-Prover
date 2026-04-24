# ==========================================
# Filename: src/system1/operators.py
# Version: v4.0 (Matrix Factorization Ready)
# Functionality: Logical Lie Algebra Operators + Ideal Matrix Computation (Supports Inverse Retrieval)
# ==========================================

import torch
import torch.nn as nn

# Importing our zero-dependency mathematical core
# Assuming manifold_math.py is already in the src/system1/ directory
from src.system1.manifold_math import PoincareBall

class LogicalLieAlgebra(nn.Module):
    """
    Logical Lie Algebra Module (v4.0).
    
    Core Functionality:
    1. Tactic -> Matrix: Maps discrete symbols to continuous Lie group matrices SO(n,1).
    2. Gap -> Matrix: [New] Calculates the ideal transformation matrix between two points 
       for "matrix factorization retrieval".
    
    Correspondence: 
    - Each Tactic ID -> A Basis for the Lie Algebra.
    - Search Process -> Finding a matrix M such that M * z_curr ≈ z_next.
    """
    def __init__(self, feat_dim, num_basis, c=5.0):
        super().__init__()
        self.feat_dim = feat_dim       # n (Poincare Ball dimension)
        self.hyp_dim = feat_dim + 1    # n+1 (Hyperboloid embedding dimension)
        self.num_basis = num_basis     # Must equal num_tactics here
        self.c = c                     # Curvature
        
        # Define Lie Algebra Basis
        # 1. Rotations: skew-symmetric matrices part
        # Initialized with very small values so the initial operator is close to Identity 
        # (no operation), ensuring training stability.
        self.rot_basis = nn.Parameter(torch.randn(num_basis, feat_dim, feat_dim) * 1e-4)
        
        # 2. Boosts: translation vectors part
        self.boost_basis = nn.Parameter(torch.randn(num_basis, feat_dim) * 1e-4)
        
        # Register locally implemented hyperbolic manifold
        self.ball = PoincareBall(c=c)
        
        # Register Minkowski Metric: G = diag(-1, 1, ..., 1)
        # Used for hyperbolic inner products when calculating ideal matrices
        self.register_buffer("minkowski_metric", self._create_minkowski_metric())
        
        print(f"🔧 LogicalLieAlgebra Ready: {num_basis} tactic matrices loaded (Matrix Factorization Mode).")

    def _create_minkowski_metric(self):
        G = torch.eye(self.hyp_dim)
        G[0, 0] = -1.0
        return G

    def get_lie_algebra_element(self, coefficients):
        """
        Synthesizes a Lie algebra element X (in matrix form) based on coefficients.
        coefficients: [Batch, num_basis]
        """
        batch_size = coefficients.shape[0]
        
        # 1. Weighted Sum of Rotations
        # weighted_rot: [Batch, n, n]
        weighted_rot = torch.einsum('bi,ijk->bjk', coefficients, self.rot_basis)
        # Force Skew-symmetry
        weighted_rot = (weighted_rot - weighted_rot.transpose(-1, -2)) * 0.5
        
        # 2. Weighted Sum of Boosts
        # weighted_boost: [Batch, n]
        weighted_boost = torch.einsum('bi,ij->bj', coefficients, self.boost_basis)
        
        # 3. Assemble Lorentz Lie Algebra matrix
        # so(n, 1) matrix structure:
        # | 0    v^T |
        # | v    S   |
        lie_algebra_elem = torch.zeros(batch_size, self.hyp_dim, self.hyp_dim, device=coefficients.device)
        
        # Fill S (Rotation part) -> bottom-right n*n
        lie_algebra_elem[:, 1:, 1:] = weighted_rot
        
        # Fill v (Boost part) -> first column and first row
        lie_algebra_elem[:, 1:, 0] = weighted_boost
        lie_algebra_elem[:, 0, 1:] = weighted_boost 
        
        return lie_algebra_elem

    def get_operator(self, coefficients):
        """
        [Mixed Mode] Generates a hybrid operator M = exp(X) based on weight vectors.
        """
        X = self.get_lie_algebra_element(coefficients)
        return torch.matrix_exp(X)

    def get_single_operator(self, tactic_idx):
        """
        [Exact Mode] Retrieves the geometric operator matrix M_tactic corresponding to a specific Tactic.
        Used by System 2 to calculate || M_ideal - M_tactic ||.
        """
        # Construct One-hot vector
        batch_size = 1
        coeffs = torch.zeros(batch_size, self.num_basis, device=self.rot_basis.device)
        coeffs[0, tactic_idx] = 1.0 # Activate the corresponding basis
        
        # Call general logic
        return self.get_operator(coeffs)

    # ------------------------------------------------------------------------
    # [CORE ADDITION] Ideal Matrix Calculation (Inverse Lie Group Factorization)
    # ------------------------------------------------------------------------
    def compute_ideal_matrix(self, z_start, z_end):
        """
        Calculates the ideal geometric transformation matrix M_ideal from z_start to z_end.
        i.e., solves for M such that M * z_start ≈ z_end.
        
        Mathematical Principle:
        In the Lorentz model, the pure Boost matrix between two points u and v has a closed-form solution.
        M = I + ((u+v)(u+v)^T G) / (1 + <u, v>_L) - 2 v u^T G
        (Note: The inner product <,>_L here is the Minkowski inner product)
        """
        # 1. Convert to Hyperboloid model (u, v)
        # Shape: [Batch, D+1]
        u = self.to_hyperboloid(z_start)
        v = self.to_hyperboloid(z_end)
        
        # Ensure Batch mode
        if u.dim() == 1: u = u.unsqueeze(0)
        if v.dim() == 1: v = v.unsqueeze(0)
        
        batch_size = u.size(0)
        G = self.minkowski_metric.to(u.device) # [D+1, D+1]
        
        # 2. Calculate Minkowski Inner Product <u, v>_L = u^T G v
        # u @ G: [B, D+1]
        u_G = torch.matmul(u, G) 
        # <u, v>_L: [B, 1]
        # Note: Under normal conditions <u, v>_L <= -1 (-1 when u=v)
        uv_dot = torch.sum(u_G * v, dim=-1, keepdim=True)
        
        # 3. Numerical Stability Handling
        # denominator = 1 + <u, v>_L
        # When u approaches v, <u,v> -> -1, and the denominator -> 0.
        # We need to handle this singularity.
        denom = 1.0 + uv_dot
        
        # Create Identity matrix
        I = torch.eye(self.hyp_dim, device=u.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # If the two points are extremely close (diff < epsilon), return the Identity matrix I directly
        dist_mask = torch.abs(denom) < 1e-5
        if dist_mask.all():
            return I
            
        # 4. Construct Lorentz Translation Matrix
        # Term 1: (u+v)(u+v)^T G / denom
        sum_uv = u + v # [B, D+1]
        # (u+v)^T G -> [B, D+1]
        sum_uv_G = torch.matmul(sum_uv, G)
        # Outer product: [B, D+1, 1] @ [B, 1, D+1] -> [B, D+1, D+1]
        term1 = torch.matmul(sum_uv.unsqueeze(-1), sum_uv_G.unsqueeze(1))
        term1 = term1 / (denom.unsqueeze(-1) + 1e-8) # Add epsilon to prevent division by zero
        
        # Term 2: 2 * v * u^T * G
        # v: [B, D+1, 1], u_G: [B, 1, D+1]
        term2 = 2 * torch.matmul(v.unsqueeze(-1), u_G.unsqueeze(1))
        
        # M_ideal = I + Term1 - Term2
        # Note the sign differences in formula derivations; standard Transvection formula used here.
        M_ideal = I + term1 - term2
        
        # For points that are extremely close, force Identity to prevent numerical explosion
        M_ideal = torch.where(dist_mask.unsqueeze(-1), I, M_ideal)
        
        return M_ideal

    def apply_tactic(self, state_emb, operator_matrix):
        """
        Applies the operator: z_new = M * z_old
        """
        # Ensure input is in Batch form
        if state_emb.dim() == 1:
            state_emb = state_emb.unsqueeze(0)
            
        # 1. Poincare Ball -> Hyperboloid
        x_h = self.to_hyperboloid(state_emb)
        
        # 2. Matrix Multiplication (Linear Action)
        # x_new = M @ x
        # [B, D+1, D+1] @ [B, D+1, 1] -> [B, D+1, 1]
        x_h_new = torch.matmul(operator_matrix, x_h.unsqueeze(-1)).squeeze(-1)
        
        # 3. Hyperboloid -> Poincare Ball
        x_p_new = self.to_poincare(x_h_new)
        
        return x_p_new

    # ==============================
    # Auxiliary Geometric Conversions (Local implementation retained, ensures no geoopt calls)
    # ==============================
    def to_hyperboloid(self, x_poincare):
        x_norm_sq = x_poincare.norm(dim=-1, keepdim=True).pow(2)
        lambda_val = 2 / (1 - x_norm_sq + 1e-6)
        x0 = (1 + x_norm_sq) / (1 - x_norm_sq + 1e-6)
        xi = lambda_val * x_poincare
        return torch.cat([x0, xi], dim=-1)

    def to_poincare(self, x_hyperboloid):
        x0 = x_hyperboloid[..., 0:1]
        xi = x_hyperboloid[..., 1:]
        return xi / (1 + x0 + 1e-6)