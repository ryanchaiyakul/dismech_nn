from .dr2d import DiscreteRod2D

import torch

# TODO: Create Neural network within __init__


class NeuralDiscereteElasticRod2D(DiscreteRod2D):

    def __init__(self, l, ms, nn):
        super().__init__(l, ms)

        self.nn = nn

    def forward(self, t, x):
        _, v, F = super().forward(t, x)
        return torch.cat([v, (F + self.nn(t, x)) / self.M], dim=-1)
