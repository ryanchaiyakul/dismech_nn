import torch


class MultiMSD(torch.nn.Module):
    """
    Multiple mass spring damper system as a torch module for odeint_adjoint
    """

    def __init__(self, ms, ks, bs):
        super().__init__()

        assert (len(ms) == len(ks) == len(bs) and len(ms) > 1)

        n = len(ms)
        rows = []

        for i in range(n):
            x_1 = torch.zeros(n * 2)
            x_2 = torch.zeros(n * 2)

            x_1[i * 2 + 1] = 1.0

            # first mass does not have a preceding mass
            if i > 0:
                x_2[i * 2 - 2] = ks[i] / ms[i]
                x_2[i * 2 - 1] = bs[i] / ms[i]

            x_2[i * 2] = -(ks[i] + (ks[i + 1] if i + 1 < n else 0)) / ms[i]
            x_2[i * 2 + 1] = -(bs[i] + (bs[i + 1] if i + 1 < n else 0)) / ms[i]

            # last mass does not have a following mass
            if i + 1 < n:
                x_2[i * 2 + 2] = ks[i + 1] / ms[i]
                x_2[i * 2 + 3] = bs[i + 1] / ms[i]

            rows += [x_1, x_2]

        self.matrix = torch.stack(rows)

    def forward(self, _, x):
        return x @ self.matrix.T
