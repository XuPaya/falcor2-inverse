"""Mesh optimization: the large-steps reparameterization and its uniform Adam (Nicolet et al. 2021, "Large Steps in
Inverse Rendering of Geometry", as in their largesteps package and PSDR-Enzyme).

    M = LargeSteps(faces, len(vertices), lmbda=20.0)
    u = M.to_differential(vertices).requires_grad_()
    optimizer = AdamUniform([u], lr=0.01)
    for _ in range(iterations):
        vertices = M.from_differential(u)       # smooth: gradients are filtered by (I + lmbda L)^-1
        loss = ...render with vertices...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""

import torch

from falcor2_inverse.device import torch_device
from falcor2_inverse.geometry import laplacian


class LargeSteps:
    """The matrix M = I + lmbda L (L: combinatorial Laplacian of the mesh) and solves with it: inverted directly
    (on the GPU) for meshes up to dense_limit vertices, by conjugate gradients otherwise."""

    def __init__(self, faces, vertex_count, lmbda=20.0, device=None, dense_limit=12000):
        device = torch.device(device) if device is not None else torch_device()
        eye = torch.sparse_coo_tensor(torch.arange(vertex_count, device=device).repeat(2, 1),
                                      torch.ones(vertex_count, device=device), (vertex_count, vertex_count))
        self.matrix = (eye + lmbda * laplacian(faces, vertex_count, device)).coalesce()
        self.inverse = self.diagonal = self._last = None
        if vertex_count <= dense_limit:
            self.inverse = torch.linalg.inv(self.matrix.to_dense().double()).float()
        else:
            i = self.matrix.indices()
            diagonal = i[0] == i[1]
            self.diagonal = torch.zeros(vertex_count, device=device).index_add_(
                0, i[0][diagonal], self.matrix.values()[diagonal])[:, None]

    def to_differential(self, vertices: torch.Tensor) -> torch.Tensor:
        """u = M v."""
        return (self.matrix @ vertices.detach().to(self.matrix.device)).detach()

    def from_differential(self, u: torch.Tensor) -> torch.Tensor:
        """v = M^-1 u, differentiable."""
        return _Solve.apply(u, self)

    def solve(self, b: torch.Tensor) -> torch.Tensor:
        if self.inverse is not None:
            return self.inverse @ b
        # Jacobi-preconditioned conjugate gradients, from the previous solution.
        x = self._last if self._last is not None and self._last.shape == b.shape else torch.zeros_like(b)
        r = b - self.matrix @ x
        z = r / self.diagonal
        p, rz = z, (r * z).sum(0)
        for _ in range(500):
            q = self.matrix @ p
            alpha = rz / (p * q).sum(0).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * q
            if r.abs().max() <= 1e-7 * b.abs().max().clamp_min(1e-30):
                break
            z = r / self.diagonal
            rz_new = (r * z).sum(0)
            p = z + rz_new / rz.clamp_min(1e-30) * p
            rz = rz_new
        self._last = x
        return x


class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, steps):
        ctx.steps = steps
        return steps.solve(u.detach().to(steps.matrix.device)).to(u)

    @staticmethod
    def backward(ctx, grad):
        return ctx.steps.solve(grad.to(ctx.steps.matrix.device)).to(grad), None  # M is symmetric


class AdamUniform(torch.optim.Optimizer):
    """Adam whose steps for each parameter tensor are divided by its largest second-moment estimate (instead of
    per element), so that large-steps updates stay smooth."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"], state["m"], state["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                state["step"] += 1
                m, v, t = state["m"], state["v"], state["step"]
                m.mul_(b1).add_(p.grad, alpha=1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                denominator = (v / (1 - b2 ** t)).sqrt().max() + group["eps"]
                p.sub_(group["lr"] * (m / (1 - b1 ** t)) / denominator)
        return loss
