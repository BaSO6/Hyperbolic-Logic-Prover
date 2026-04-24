# ==========================================
# Filename: src/system1/manifold_math.py
# Version: v4.0 (Fully Geometric - Navigation Ready)
# Functionality: Pure PyTorch implementation of the Poincaré ball mathematical core (supports arbitrary point navigation)
# ==========================================
import torch

class PoincareBall:
    """
    Mathematical core library for the Poincaré ball model.
    
    Supports:
    1. Mobius Addition: handles displacement/translation.
    2. Universal Exponential/Logarithmic Mapping (ExpMap/LogMap): handles tangent vector conversion between arbitrary points.
    3. Conformal Factor (Lambda): handles local scale factors at different locations.
    """
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c
        self.sqrt_c = c ** 0.5
        # Minimal value to prevent division by zero
        self.min_norm = 1e-15
        self.max_norm = 1.0 - 1e-5

    def _lambda_x(self, x):
        """
        Calculate the Conformal Factor $\lambda_x$:
        $$\lambda_x = \frac{2}{1 - c \|x\|^2}$$
        This is the scaling factor that makes hyperbolic space "appear larger" than Euclidean space.
        """
        x_sqnorm = torch.sum(x.pow(2), dim=-1, keepdim=True)
        return 2.0 / (1.0 - self.c * x_sqnorm).clamp(min=self.min_norm)

    def mobius_add(self, x, y):
        """
        Mobius Addition: $x \oplus y$
        This is "vector addition" in hyperbolic space, used to move points.
        """
        
        x2 = torch.sum(x.pow(2), dim=-1, keepdim=True)
        y2 = torch.sum(y.pow(2), dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        
        return num / denom.clamp(min=self.min_norm)

    def dist(self, x, y):
        """
        Hyperbolic Distance:
        $$d(x, y) = \frac{2}{\sqrt{c}} \text{artanh}(\sqrt{c} \|-x \oplus y\|)$$
        """
        diff = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1, keepdim=False)
        res = 2.0 / self.sqrt_c * torch.atanh((self.sqrt_c * diff_norm).clamp(max=self.max_norm))
        return res

    def dist0(self, x):
        """
        Distance to the origin (norm): $d(0, x)$
        """
        x_norm = x.norm(dim=-1, keepdim=False)
        res = 2.0 / self.sqrt_c * torch.atanh((self.sqrt_c * x_norm).clamp(max=self.max_norm))
        return res

    def logmap0(self, y):
        """
        Logarithmic map at the origin: $\log_0(y)$
        Maps a point $y$ on the manifold to the tangent space at the origin (used for System 1 Embedding).
        """
        y_norm = y.norm(dim=-1, keepdim=True)
        scale = torch.atanh((self.sqrt_c * y_norm).clamp(max=self.max_norm)) / (self.sqrt_c * y_norm).clamp(min=self.min_norm)
        return y * scale

    def expmap0(self, u):
        """
        Exponential map at the origin: $\exp_0(u)$
        Maps a tangent vector $u$ at the origin onto the manifold (used for System 1 Embedding).
        """
        
        u_norm = u.norm(dim=-1, keepdim=True)
        scale = torch.tanh((self.sqrt_c * u_norm)) / (self.sqrt_c * u_norm).clamp(min=self.min_norm)
        return u * scale

    def logmap(self, x, y):
        """
        [Key Addition] Universal Logarithmic Map: $\log_x(y)$
        Calculates the tangent vector at point $x$ pointing toward point $y$.
        
        Purpose: 
        In "matrix decomposition," this is the "Geometric Gap" we need to fill.
        $v = \log_x(y)$ is the ideal advancement direction we want the Tactic matrix to produce.
        """
        sub = self.mobius_add(-x, y)
        sub_norm = sub.norm(dim=-1, keepdim=True)
        lam = self._lambda_x(x)
        
        scale = 2.0 / lam * torch.atanh((self.sqrt_c * sub_norm).clamp(max=self.max_norm)) / (self.sqrt_c * sub_norm).clamp(min=self.min_norm)
        return scale * sub

    def expmap(self, x, u):
        """
        [Key Addition] Universal Exponential Map: $\exp_x(u)$
        Moves along the tangent vector $u$ at point $x$ to find a new point on the manifold.
        
        Purpose:
        Used for validation steps. If I know I am currently at $x$ and want to apply an advancement vector $u$, where will I end up?
        """
        u_norm = u.norm(dim=-1, keepdim=True)
        lam = self._lambda_x(x)
        
        # The formula derivation here is relatively complex, based on Mobius addition:
        # res = x \oplus (tanh(sqrt(c) * lambda * ||u|| / 2) * u / (sqrt(c) * ||u||))
        
        factor = torch.tanh(self.sqrt_c * lam * u_norm / 2.0) / (self.sqrt_c * u_norm).clamp(min=self.min_norm)
        scaled_u = u * factor
        
        return self.mobius_add(x, scaled_u)