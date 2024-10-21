import typing
import torch


class MLP(torch.nn.Module):
    """
    Simple MLP implementation of a differential Equation RHS for positon and velocity.

    Respects v = dx/dt
    """

    # TODO: Make this scalable (beyond 2 layers)
    def __init__(self,
                 in_dim: int = 2,
                 out_dim: int = 2,
                 hidden: int = 50,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim, dtype=dtype)
        )

    def forward(self, _, x):
        return self.net(x)


class PhysicsMLP(torch.nn.Module):
    """
    Simple MLP implementation of a differential Equation RHS for positon and velocity.

    Respects v = dx/dt
    """

    # TODO: Make this scalable (beyond 2 layers)
    def __init__(self,
                 in_out: int = 2,
                 hidden: int = 50,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        assert (in_out % 2 == 0)
        self.out = int(in_out / 2)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_out, hidden, dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, self.out, dtype=dtype)
        )

    def forward(self, _, x):
        return torch.cat([x[..., self.out:], self.net(x),], dim=-1)
