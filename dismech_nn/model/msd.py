import torch

class MSD(torch.nn.Module):
    """
    Single mass spring damper system as a torch module for odeint_adjoint
    """

    def __init__(self, m: float, k: float, b: float):
        super().__init__()
        self.matrix = torch.tensor([[0.0, 1.0], [-k/m, -b/m]])

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return x @ self.matrix.T

