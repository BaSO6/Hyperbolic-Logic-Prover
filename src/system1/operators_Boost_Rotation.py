# ==============================================================================
# File: src/system1/operators_Boost_Rotation.py
# Type: Core Mechanism (Ablation Ready)
# Paper Reference: Section 4.2 "Reasoning as Isometric Lie Group Action"
# Formula: Eq (8) A = [[0, v^T], [v, S]]
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class LogicalLieAlgebra(nn.Module):
    """
    Implements the so(n, 1) Lie Algebra operator with explicit support for
    structural ablation studies (Decoupling Boosts vs. Rotations).
    
    This class maps a latent vector 'coeff' into a transformation matrix A
    in the Lie Algebra so(n, 1), and then applies the exponential map
    to update the state z on the Poincaré ball.
    """

    def __init__(self, input_dim: int, manifold_dim: int, c: float = 1.0):
        """
        Args:
            input_dim (int): Dimension of the input coefficient vector (from Policy Head).
            manifold_dim (int): Dimension of the hyperbolic space (n).
            c (float): Curvature of the manifold (usually 1.0).
        """
        super().__init__()
        self.n = manifold_dim
        self.c = c
        self.input_dim = input_dim

        # ----------------------------------------------------------------------
        # Parameter Projection
        # We need to map the flat input 'coeff' to two distinct geometric components:
        # 1. Boost Vector v (size n): Controls radial movement (Abstraction).
        # 2. Rotation Matrix S (skew-symmetric): Controls angular movement (Reasoning).
        #
        # To maintain O(n) complexity instead of O(n^2), we use a block-diagonal 
        # approximation for rotations if n is large, or a full mapping if n is small.
        # Here we map to (n) boost params + (n) rotation params for efficiency.
        # ----------------------------------------------------------------------
        self.boost_proj = nn.Linear(input_dim, manifold_dim)
        self.rot_proj = nn.Linear(input_dim, manifold_dim) 

    def _construct_algebra_matrix(self, v: torch.Tensor, rot_params: torch.Tensor) -> torch.Tensor:
        """
        Constructs the Lie Algebra matrix A in so(n, 1).
        Structure:
            A = |  0   v^T |
                |  v    S  |
        where S is a skew-symmetric matrix derived from rot_params.
        
        Args:
            v: [batch, n] - Boost vector
            rot_params: [batch, n] - Parameters to build S
        """
        batch_size = v.shape[0]
        n = self.n
        device = v.device

        # 1. Construct Boost Parts (Row 0 and Col 0)
        # ------------------------------------------
        # The algebra element in Minkowski representation is (n+1) x (n+1)
        A = torch.zeros(batch_size, n + 1, n + 1, device=device)
        
        # Set A[0, 1:] = v^T
        A[:, 0, 1:] = v
        # Set A[1:, 0] = v
        A[:, 1:, 0] = v

        # 2. Construct Rotation Part (S)
        # ------------------------------------------
        # We construct a skew-symmetric matrix S from rot_params.
        # Efficient approximation: construct block-diagonal 2x2 rotations
        # to avoid O(n^2) parameter explosion.
        #
        # For full expressivity in smaller dims, one could map to n(n-1)/2.
        # Here we use a tridiagonal-like expansion for stability and efficiency.
        
        # Create indices for the skew-symmetric positions
        # Simple strategy: S_ij = -S_ji. 
        # We use rot_params to fill the first off-diagonal band.
        indices = torch.arange(n - 1, device=device)
        
        # S[i, i+1] = rot_param[i]
        # S[i+1, i] = -rot_param[i]
        # We offset by +1 in A because index 0 is the "time" dimension in Minkowski
        row_idx = indices + 1
        col_idx = indices + 2
        
        # Safe slicing in case n is huge
        vals = rot_params[:, :n-1]
        
        A[:, row_idx, col_idx] = vals
        A[:, col_idx, row_idx] = -vals

        return A

    def get_operator(self, coeff: torch.Tensor, ablation_mode: str = None) -> torch.Tensor:
        """
        Generates the Lie Algebra operator A from input coefficients,
        applying specific masks for ablation studies.

        Args:
            coeff: [batch, input_dim] - Raw output from the Policy Network.
            ablation_mode: 
                - "no_boost": Forces boost vector v=0 (Frozen Abstraction).
                - "no_rotation": Forces rotation matrix S=0 (Frozen Semantics).
                - None: Standard full dynamics.

        Returns:
            A: [batch, n+1, n+1] - The element in so(n, 1).
        """
        # 1. Project to Geometric Components
        v = self.boost_proj(coeff)       # Boosts (Abstraction/Instantiation)
        r = self.rot_proj(coeff)         # Rotations (Semantic Paraphrasing)

        # 2. Apply Ablation Logic (The Physics of Logic)
        # -----------------------------------------------------------
        if ablation_mode == "no_boost":
            # Hypothesis: Model loses ability to traverse hierarchy.
            # It will be stuck at the current level of abstraction.
            v = torch.zeros_like(v)
            
        elif ablation_mode == "no_rotation":
            # Hypothesis: Model loses ability to navigate lateral dependencies.
            # It can only move radially (dive to axioms or instatiate), 
            # but cannot adjust semantic angle.
            r = torch.zeros_like(r)
        # -----------------------------------------------------------

        # 3. Construct the Matrix
        A = self._construct_algebra_matrix(v, r)
        return A

    def exp_map_action(self, A: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the continuous group action via Matrix Exponential in Minkowski space.
        
        Pipeline: 
        Poincaré (z) -> Minkowski (x) -> exp(A)*x -> Poincaré (z_new)
        
        Args:
            A: [batch, n+1, n+1] - The Lie Algebra operator.
            z: [batch, n] - Current state in Poincaré ball.
        """
        # 1. Lift to Minkowski Space (Hyperboloid Model)
        # x = (x0, x1...xn) where x0^2 - ||x_rest||^2 = 1
        # Formula: x_rest = 2z / (1 - ||z||^2)
        #          x_0    = (1 + ||z||^2) / (1 - ||z||^2)
        
        z_norm_sq = torch.sum(z ** 2, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        denom = 1.0 - z_norm_sq
        
        x_rest = (2.0 * z) / denom
        x_0 = (1.0 + z_norm_sq) / denom
        
        # x shape: [batch, n+1]
        x = torch.cat([x_0, x_rest], dim=-1).unsqueeze(-1) # [batch, n+1, 1]

        # 2. Compute Group Action: M = exp(A)
        # M is an element of SO(n, 1)
        M = torch.matrix_exp(A) 

        # 3. Apply Action: x_new = M * x
        x_new = torch.bmm(M, x).squeeze(-1) # [batch, n+1]

        # 4. Project back to Poincaré Ball
        # z = x_rest / (1 + x_0)
        x_new_0 = x_new[:, 0:1]
        x_new_rest = x_new[:, 1:]
        
        # Numerical stability check for division
        denom_back = 1.0 + x_new_0 + 1e-8
        z_new = x_new_rest / denom_back
        
        # Clamp to ensure we stay within the ball (numerical drift safety)
        z_new_norm = torch.norm(z_new, p=2, dim=-1, keepdim=True)
        mask = z_new_norm >= 1.0
        if mask.any():
            z_new = torch.where(mask, (z_new / z_new_norm) * (1.0 - 1e-5), z_new)

        return z_new

    def forward(self, z: torch.Tensor, coeff: torch.Tensor, ablation_mode: str = None) -> torch.Tensor:
        """
        Main forward pass for the Navigator.
        
        Args:
            z: Current state [batch, n]
            coeff: Policy output [batch, input_dim]
            ablation_mode: "no_boost" | "no_rotation" | None
        """
        A = self.get_operator(coeff, ablation_mode=ablation_mode)
        z_next = self.exp_map_action(A, z)
        return z_next